#!/bin/bash
# ============================================================================
# NousResearch/Meta-Llama-3.1-8B — 2-bit & 1-bit PTQ benchmark suite (Nibi)
# ============================================================================
#
# NOTE: meta-llama/Llama-3.1-8B is gated. The current HF token (./.hf_token)
# was confirmed via HfApi to be 403 Forbidden against meta-llama/Llama-3.1-8B
# on 2026-04-27. NousResearch/Meta-Llama-3.1-8B is an ungated identical
# weight mirror (verified config: hidden=4096, layers=32, heads=32,
# kv_heads=8, intermediate=14336 — matches gated upstream exactly). Phase
# 14B used the same NousResearch mirror pattern for Llama-2-7b-hf.
#
# To switch back to the gated meta-llama version, request access at
# https://huggingface.co/meta-llama/Llama-3.1-8B (usually approved within
# minutes), then change MODEL/MODEL_CACHE_DIR below to the meta-llama path.
# ============================================================================
#
# Sister script to run_llama3_1b_benchmark.sh, run_llama3_3b_benchmark.sh,
# and run_qwen_*_benchmark.sh. Dispatches one sbatch job per quantization
# method on NousResearch/Meta-Llama-3.1-8B. All jobs append results to a shared
# CSV via fcntl-locked src/csv_utils.append_result.
#
# Model facts (Llama-3.1-8B, standard config):
#   - 8.03 B params; ~16.06 GB fp16 on disk + runtime overhead
#   - hidden=4096, layers=32, heads=32, num_kv_heads=8 (GQA),
#     intermediate=14336, vocab=128256, dtype=bfloat16
#   - Same hidden as Qwen3-8B (4096) but 32 layers vs 36 → 0.89× layer ratio
#   - Wider intermediate (14336 vs 12288) → 1.17× per-layer FLOPs
#   - Net: 1.04× of Qwen3-8B wall — within Phase-16 margin; reuse budgets
#
# Why 8B needs special handling (mirrors Qwen3-8B):
#   1. Model alone is 16 GB — 2g.20gb (20 GB MIG) is barely enough for the
#      model and leaves nothing for activations. Base slice is 3g.40gb (40 GB).
#   2. LNQ Fisher peaks ≥40 GB on 8B (Phase 8 measured 42 GB on Qwen3-8B
#      A40); 3g.40gb is insufficient → full h100 (80 GB).
#   3. LNQ 1024 samples × 4096 seqlen exceeded 125 GB system RAM on Qwen3-8B
#      (Phase 8 OOM at layer 32/35). Use 512 samples for 8B.
#   4. TesseraQ paper-exact (250 iters × 32 blocks) on 8B is the highest
#      blast-radius job in the suite — budget 96 h on full h100.
#
# Methods benchmarked: same as 1B/3B + Qwen3 suite.
#
# Usage:
#   ./sbatch/run_llama3_8b_benchmark.sh                      # submit all
#   ./sbatch/run_llama3_8b_benchmark.sh --account def-foo
#   ./sbatch/run_llama3_8b_benchmark.sh --local              # serial, no SLURM
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT=""
LOCAL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--local]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL="NousResearch/Meta-Llama-3.1-8B"
MODEL_SHORT="llama3_8b"
DATASET="wikitext2"
SEED=0

MODEL_CACHE_DIR="models--NousResearch--Meta-Llama-3.1-8B"

CSV_NAME="${MODEL_SHORT}_ptq_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES (full long-form names required; see Gotcha #11). 4g.40gb is NOT
# provisioned (Gotcha #33).
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"                              # full 80 GB H100

# Shared quantization hyperparameters.
BLOCKSIZE=128
SALIENT_METRIC="magnitude"
LNQ_CALIB="redpajama"
# 8B-specific: MUST use 512 samples (not 1024). Phase 8 measured 125 GB
# RAM OOM on Qwen3-8B at layer 32/35 with 1024 samples; same hidden=4096
# applies here.
LNQ_NSAMPLES=512
LNQ_SEQLEN=4096
LNQ_NUM_GROUPS=4
LNQ_NBITS=2
LEANQUANT_CALIB="redpajama"
LEANQUANT_NSAMPLES=128
LEANQUANT_SEQLEN=2048
LEANQUANT_NBITS=2
LEANQUANT_EXPONENT=4.0
LEANQUANT_PERCDAMP=0.1
SINQ_NBITS=2
SINQ_GROUPSIZE=64
TESSERAQ_BIT=2
TESSERAQ_GROUPSIZE=128
TESSERAQ_ITERATIONS=250
TESSERAQ_BATCH_SIZE=4
TESSERAQ_NSAMPLES=512
PBLLM_METHOD="xnor"
PBLLM_LOW_FRAC=0.9
PBLLM_HIGH_BIT=8

techniques=(
    "fp16"
    "rtn-2bit"
    "gptq-2bit"
    "sinq"
    "lnq"
    "leanquant-nu"
    "tesseraq"
    "pb-llm"
    "doml"
    "doml-binary"
    "braq"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_resources() {
    # Llama-3.1-8B: model alone is 16 GB. Base slice is 3g.40gb (40 GB) for
    # every method; LNQ + tesseraq escalate to full h100 (80 GB).
    #
    # Phase 16 (2026-04-26) budgets carried over from Qwen3-8B. Llama-3.1-8B
    # has identical hidden (4096), slightly wider intermediate (14336 vs
    # 12288), 32 layers vs 36 — net 1.04× of Qwen3-8B work, within margin.
    case $1 in
        lnq)
            # Fisher peaks ~42 GB on Qwen3-8B (A40 measured Phase 8). 3g.40gb
            # would be insufficient — must use full h100 (80 GB MIG-free).
            # 768 GB/node RAM allows generous --mem.
            gpu_resource="$GPU_FULL"
            cpus=16
            mem="240G"
            ;;
        tesseraq)
            # Paper-exact (iter=250, bsz=4, nsamples=512). 32 blocks × 250
            # iters × per-sample backward pass. Highest blast-radius job in
            # the suite. Phase 16 sized this at 400 GB RAM / 96 h walltime
            # for Qwen3-8B; same here.
            gpu_resource="$GPU_FULL"
            cpus=16
            mem="400G"
            ;;
        leanquant-nu)
            # 128×2048 + GPTQ + weighted k-means: model 16 + activations ~2
            # + blockwise work ~4 = ~22 GB peak. 3g.40gb fits with margin.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="64G"
            ;;
        pb-llm)
            # PB-LLM keeps fp16 outliers + GPTQ Hessians. Peak ~25 GB on 8B.
            # Phase 16 doubled host RAM budget for PB-LLM (Phase-13 datautils
            # crash history); same here.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="96G"
            ;;
        fp16)
            # Eval only — ~19 GB peak. 3g.40gb safest.
            gpu_resource="$GPU_LARGE"
            cpus=6
            mem="48G"
            ;;
        *)
            # rtn-2bit, gptq-2bit, sinq, doml, doml-binary, braq.
            # Peak 22-25 GB ⇒ 3g.40gb fits with headroom.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="64G"
            ;;
    esac
}

get_time_limit() {
    # Llama-3.1-8B is ~1.04× the wall of Qwen3-8B. Use Qwen3-8B Phase-16
    # budgets directly. Alliance Nibi tiers: b1≤3h, b2≤12h, b3≤24h, b4≤72h,
    # b5≤168h.
    case $1 in
        fp16)         echo "10:00:00" ;;  # b2 — eval-bound; FP16 model load + ~6 h eval suite
        rtn-2bit)     echo "12:30:00" ;;  # off b2/b3 boundary → b3
        gptq-2bit)    echo "13:00:00" ;;  # b3
        sinq)         echo "11:00:00" ;;  # b2
        pb-llm)       echo "14:00:00" ;;  # b3
        doml)         echo "16:00:00" ;;  # b3 — quant + ~5 h evals + margin
        doml-binary)  echo "16:00:00" ;;  # b3
        braq)         echo "16:00:00" ;;  # b3
        tesseraq)     echo "96:00:00" ;;  # b5 — 92 h quant + ~4 h evals
        lnq)          echo "20:00:00" ;;  # b3 — 16 h Fisher/saliency/refine + 4 h margin
        leanquant-nu) echo "11:00:00" ;;  # b2 — ~1 h quant + ~6 h evals + margin
        *)            echo "11:00:00" ;;
    esac
}

build_python_cmd() {
    local technique=$1

    local common_ds_seed="$MODEL $DATASET --seed $SEED --device cuda:0"
    # --full_eval = PPL on (wikitext2,c4,ptb) + MMLU + HellaSwag + ARC-Easy + ARC-Challenge.
    local common_evals="--full_eval"

    case $technique in
        fp16)
            echo "python3 -u run.py $MODEL $DATASET fp16 --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        rtn-2bit)
            echo "python3 -u run.py $MODEL $DATASET 2bit --disable_gptq --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        gptq-2bit)
            echo "python3 -u run.py $MODEL $DATASET 2bit --partition 1 --global_scale --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        doml)
            echo "python3 -u run.py $MODEL $DATASET doml --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        doml-binary)
            echo "python3 -u run.py $MODEL $DATASET doml_binary --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        braq)
            echo "python3 -u run.py $MODEL $DATASET braq --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        sinq)
            echo "python3 -u src/run_sinq.py $MODEL $DATASET --nbits $SINQ_NBITS --group_size $SINQ_GROUPSIZE --seed $SEED --device cuda:0 $common_evals"
            ;;
        tesseraq)
            echo "python3 -u src/run_tesseraq.py $MODEL $DATASET --bit $TESSERAQ_BIT --group_size $TESSERAQ_GROUPSIZE --iterations $TESSERAQ_ITERATIONS --batch_size $TESSERAQ_BATCH_SIZE --nsamples $TESSERAQ_NSAMPLES --seed $SEED --device cuda:0 $common_evals"
            ;;
        lnq)
            echo "python3 -u src/run_lnq.py $MODEL $DATASET --full_pipeline --no_propagate --calib_dataset $LNQ_CALIB --nsamples $LNQ_NSAMPLES --seqlen $LNQ_SEQLEN --num_groups $LNQ_NUM_GROUPS --nbits $LNQ_NBITS --seed $SEED --device cuda:0 $common_evals"
            ;;
        leanquant-nu)
            echo "python3 -u src/run_leanquant.py $MODEL $DATASET --nbits $LEANQUANT_NBITS --exponent $LEANQUANT_EXPONENT --percdamp $LEANQUANT_PERCDAMP --true_sequential --act_order --calib_dataset $LEANQUANT_CALIB --nsamples $LEANQUANT_NSAMPLES --seqlen $LEANQUANT_SEQLEN --seed $SEED --device cuda:0 $common_evals"
            ;;
        pb-llm)
            echo "python3 -u PB-LLM/gptq_pb/run.py $MODEL $DATASET $PBLLM_METHOD --low_frac $PBLLM_LOW_FRAC --high_bit $PBLLM_HIGH_BIT --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED $common_evals"
            ;;
        *)
            echo "echo 'Unknown technique: $technique'; exit 1"
            ;;
    esac
}

get_technique_desc() {
    case $1 in
        fp16)         echo "FP16 baseline (no quantization)"                                ;;
        rtn-2bit)     echo "RTN @ 2-bit (GPTQ disabled, blocksize=$BLOCKSIZE)"              ;;
        gptq-2bit)    echo "GPTQ @ 2-bit paper-faithful (--partition 1 --global_scale, per-row, blocksize=$BLOCKSIZE)" ;;
        sinq)         echo "SINQ (nbits=$SINQ_NBITS, group_size=$SINQ_GROUPSIZE)"           ;;
        lnq)          echo "GuidedQuant/LNQ (nbits=$LNQ_NBITS, $LNQ_CALIB/$LNQ_NSAMPLES/$LNQ_SEQLEN, groups=$LNQ_NUM_GROUPS, no_propagate)" ;;
        leanquant-nu) echo "LeanQuant_nu (nbits=$LEANQUANT_NBITS, p=$LEANQUANT_EXPONENT, percdamp=$LEANQUANT_PERCDAMP, $LEANQUANT_CALIB/$LEANQUANT_NSAMPLES/$LEANQUANT_SEQLEN, act_order+true_sequential)" ;;
        tesseraq)     echo "TesseraQ paper-exact (bit=$TESSERAQ_BIT, gs=$TESSERAQ_GROUPSIZE, iters=$TESSERAQ_ITERATIONS, bsz=$TESSERAQ_BATCH_SIZE, nsamples=$TESSERAQ_NSAMPLES, AWQ init on)" ;;
        pb-llm)       echo "PB-LLM ($PBLLM_METHOD, low_frac=$PBLLM_LOW_FRAC, high_bit=$PBLLM_HIGH_BIT, blocksize=$BLOCKSIZE)" ;;
        doml)         echo "DOML (K=4, Lloyd-Max + GPTQ + structural partition, salient=$SALIENT_METRIC)" ;;
        doml-binary)  echo "DOML-binary (K=2, Lloyd-Max + GPTQ + structural partition)"     ;;
        braq)         echo "BRAQ 1-bit baseline (blocksize=$BLOCKSIZE)"                     ;;
    esac
}

# ============================================================================
# MAIN LOOP
# ============================================================================

job_count=0
mkdir -p ./logs ./results

echo "============================================"
echo "Llama-3.1-8B PTQ Benchmark Suite (Nibi)"
echo "============================================"
echo "Model:       $MODEL"
echo "Dataset:     $DATASET"
echo "Seed:        $SEED"
echo "Techniques:  ${techniques[*]}"
echo "Shared CSV:  $CSV_ABS"
echo "Logs:        ./logs/"
echo "============================================"
echo ""

for technique in "${techniques[@]}"; do
    technique_desc=$(get_technique_desc "$technique")
    time_limit=$(get_time_limit "$technique")
    get_job_resources "$technique"

    job_name="${MODEL_SHORT}_${technique}"
    python_cmd=$(build_python_cmd "$technique")

    if [[ "$LOCAL_MODE" == true ]]; then
        echo "========================================"
        echo "Running locally: $job_name"
        echo "Config: $technique_desc"
        echo "Command: $python_cmd"
        echo "========================================"
        if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]]; then
            CACHE_ROOT_LOCAL="$SCRATCH/billm2_cache"
            DOWNLOADS_LOCAL="$CACHE_ROOT_LOCAL/downloads"
        else
            CACHE_ROOT_LOCAL="$(pwd)"
            DOWNLOADS_LOCAL="$(pwd)/downloads"
        fi
        if [[ "$DOWNLOADS_LOCAL" != "$(pwd)/downloads" && ! -L ./downloads && ! -e ./downloads ]]; then
            ln -sfn "$DOWNLOADS_LOCAL" ./downloads
        fi
        export BILLM_BENCH_CSV="$CSV_ABS"
        export BILLM_DOWNLOADS_DIR="$DOWNLOADS_LOCAL"
        export HF_HOME="$CACHE_ROOT_LOCAL/hf"
        export HF_HUB_CACHE="$BILLM_DOWNLOADS_DIR"
        export HF_DATASETS_CACHE="$CACHE_ROOT_LOCAL/hf/datasets"
        export TORCH_HOME="$CACHE_ROOT_LOCAL/torch"
        export HF_HUB_DISABLE_XET=1
        export NUMBA_CACHE_DIR="$CACHE_ROOT_LOCAL/.cache/numba"
        export PIP_CACHE_DIR="$CACHE_ROOT_LOCAL/.cache/pip"
        export XDG_CACHE_HOME="$CACHE_ROOT_LOCAL/.cache"
        export TMPDIR="${SLURM_TMPDIR:-$CACHE_ROOT_LOCAL/tmp}"
        mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TORCH_HOME" \
                 "$NUMBA_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" \
                 "$BILLM_DOWNLOADS_DIR"
        export HF_DATASETS_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export HF_HUB_OFFLINE=1
        export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
        eval "$python_cmd"
        ((job_count++))
        continue
    fi

    account_line=""
    if [[ -n "$ACCOUNT" ]]; then
        account_line="#SBATCH --account=$ACCOUNT"
    fi

    sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=$time_limit
#SBATCH $gpu_resource
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$mem
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

if [[ -n "\${SCRATCH:-}" && -d "\${SCRATCH}" ]]; then
    CACHE_ROOT="\$SCRATCH/billm2_cache"
else
    CACHE_ROOT="\$(pwd)/downloads"
fi

if [[ "\$CACHE_ROOT" != "\$(pwd)/downloads" && ! -L ./downloads && ! -e ./downloads ]]; then
    ln -sfn "\$CACHE_ROOT" ./downloads
fi

export BILLM_DOWNLOADS_DIR="\$CACHE_ROOT/downloads"
export HF_HOME="\$CACHE_ROOT/hf"
export HF_HUB_CACHE="\$BILLM_DOWNLOADS_DIR"
export HF_DATASETS_CACHE="\$CACHE_ROOT/hf/datasets"
export TORCH_HOME="\$CACHE_ROOT/torch"
export HF_HUB_DISABLE_XET=1
export NUMBA_CACHE_DIR="\$CACHE_ROOT/.cache/numba"
export PIP_CACHE_DIR="\$CACHE_ROOT/.cache/pip"
export XDG_CACHE_HOME="\$CACHE_ROOT/.cache"
if [[ -n "\${SLURM_TMPDIR:-}" && -d "\${SLURM_TMPDIR}" ]]; then
    export TMPDIR="\$SLURM_TMPDIR"
else
    export TMPDIR="\$CACHE_ROOT/tmp"
fi
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$HF_HUB_CACHE" "\$TORCH_HOME" \\
         "\$NUMBA_CACHE_DIR" "\$PIP_CACHE_DIR" "\$XDG_CACHE_HOME" "\$TMPDIR" \\
         "\$BILLM_DOWNLOADS_DIR"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export BILLM_BENCH_CSV="$CSV_ABS"

export PYTHONPATH=\$PYTHONPATH:\$(pwd):\$(pwd)/src

# LNQ Fisher computation on MIG slices benefits from expandable segments —
# Phase 8 found it prevents fragmentation-OOM at the 8B Fisher peak.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo '========================================'
echo "Job:        $job_name"
echo "Model:      $MODEL"
echo "Technique:  $technique"
echo "Config:     $technique_desc"
echo "Dataset:    $DATASET"
echo "Seed:       $SEED"
echo "Time limit: $time_limit"
echo "GPU:        $gpu_resource"
echo "CPUs / Mem: $cpus / $mem"
echo "CSV:        \$BILLM_BENCH_CSV"
echo "Cache root: \$CACHE_ROOT (\$HF_HOME)"
echo "TMPDIR:     \$TMPDIR"
echo "Started:    \$(date)"
echo "SLURM job:  \$SLURM_JOB_ID"
echo '========================================'
echo "Python:             \$(which python)"
python --version
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
python -c "from transformers import LlamaForCausalLM; import transformers; print('transformers', transformers.__version__)" || {
    echo "FATAL: LlamaForCausalLM not importable — check transformers install."
    exit 1
}
ls -d "\$BILLM_DOWNLOADS_DIR"/$MODEL_CACHE_DIR 2>/dev/null || {
    echo "FATAL: no $MODEL snapshot under \$BILLM_DOWNLOADS_DIR — run ./sbatch/download_cache.sh first."
    exit 1
}
for _refs in "\$BILLM_DOWNLOADS_DIR"/$MODEL_CACHE_DIR/refs/main; do
    [ -s "\$_refs" ] || {
        echo "FATAL: \$_refs missing or empty."
        echo "       Re-run ./sbatch/download_cache.sh — needs"
        echo "       snapshot_download(..., local_files_only=True) second pass."
        exit 1
    }
done
nvidia-smi || true
$python_cmd
echo '========================================'
echo "Finished:   \$(date)"
echo '========================================'
EOF
)
    echo "  [$sbatch_id] $job_name  ($technique_desc, t=${time_limit}, gpu=${gpu_resource#--gres=gpu:})"
    ((job_count++))
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results CSV:          $CSV_ABS"
echo "Logs directory:       ./logs/"
echo "============================================"
