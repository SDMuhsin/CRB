#!/bin/bash
# ============================================================================
# Llama-3.2-1B — 2-bit & 1-bit PTQ benchmark suite (Nibi / Alliance Canada)
# ============================================================================
#
# Sister script to run_qwen_*_benchmark.sh. Dispatches one sbatch job per
# quantization method on meta-llama/Llama-3.2-1B. All jobs append results
# to a shared CSV via fcntl-locked src/csv_utils.append_result.
#
# Model facts (Llama-3.2-1B, verified 2026-04-27 from local config.json):
#   - 1.24 B params (1,235,814,400 exact); ~2.47 GB fp16 on disk + runtime
#   - hidden=2048, layers=16, heads=32, num_kv_heads=8 (GQA),
#     intermediate=8192, vocab=128256, dtype=bfloat16
#   - tied_word_embeddings=True (so lm_head is small)
#
# Sizing vs Qwen3-1.7B (the most directly comparable Qwen3 size):
#   - Same hidden (2048) but 16 layers vs 28 → ~0.57× the per-pass walltime
#   - Slightly wider intermediate (8192 vs 6144) → 1.33× per-layer FLOPs
#   - Net: ~0.76× of Qwen3-1.7B wall, comfortably within 1.7B's budget
#
# Why 1B is not the smallest tier (1g.10gb / 16 GB cgroup):
#   - Phase 16 (Qwen3-0.6B): even 0.6B was promoted off 1g.10gb because
#     the eval suite's `model.to(dev)` for MMLU/HellaSwag/ARC pushes peak
#     past 10 GB once the full model is on GPU. Llama-3.2-1B is 2× the
#     params of Qwen3-0.6B → 2g.20gb base everywhere.
#
# Methods benchmarked (mirrors Qwen3 suite — see run_qwen_benchmark.sh):
#   ~2 bpw : fp16, rtn-2bit, gptq-2bit, sinq, lnq, leanquant-nu, tesseraq,
#            pb-llm, doml (flagship)
#   ~1 bpw : doml-binary, braq
#
# Usage:
#   ./sbatch/run_llama3_1b_benchmark.sh                      # submit all
#   ./sbatch/run_llama3_1b_benchmark.sh --account def-foo
#   ./sbatch/run_llama3_1b_benchmark.sh --local              # serial, no SLURM
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

MODEL="meta-llama/Llama-3.2-1B"
MODEL_SHORT="llama3_1b"
DATASET="wikitext2"
SEED=0

# HF cache directory uses the canonical models--<org>--<repo> layout.
MODEL_CACHE_DIR="models--meta-llama--Llama-3.2-1B"

CSV_NAME="${MODEL_SHORT}_ptq_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES strings. Full long-form MIG names are REQUIRED — short forms
# like `1g.10gb` are rejected (Gotcha #11). 4g.40gb is NOT provisioned
# (Gotcha #33; sinfo confirms).
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"

# Shared quantization hyperparameters (match Qwen3 suite — same paper-faithful
# settings keep the cross-method comparison fair across architectures).
BLOCKSIZE=128
SALIENT_METRIC="magnitude"
LNQ_CALIB="redpajama"
LNQ_NSAMPLES=1024
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

# Methods to benchmark — one sbatch job per entry.
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
    # Llama-3.2-1B GPU footprint:
    #   - Model fp16 on GPU: ~2.5 GB
    #   - DOML/GPTQ peak (128 × 2048 calib): ~5-7 GB
    #   - LeanQuant (128 × 2048 activations): ~6-8 GB
    #   - LNQ (Fisher + 4-group saliency + 1024 × 4096 calib):
    #       - hidden² = 2048² = 4.2 M, similar per-layer footprint to
    #         Qwen3-1.7B at 1024 samples; expected peak ~25-30 GB → 3g.40gb
    #   - TesseraQ (250 iter × 16 blocks): GPU peak ~10-12 GB on 1B; full
    #     h100 used for compute, not memory.
    #
    # Phase 16 budgets carried over (proven safe on Qwen3 1.7B which has
    # similar hidden + 1.75× layers).
    case $1 in
        lnq)
            # 1.7B Qwen3 LNQ ran on 3g.40gb (40 GB) and stayed comfortable.
            # Llama-3.2-1B has same hidden=2048 and fewer layers, so peak is
            # smaller. Use the same slot for headroom.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="96G"
            ;;
        tesseraq)
            # Paper-exact (iter=250, bsz=4, nsamples=512). The 0.6B Qwen3
            # tesseraq OOM'd at 48 GB cgroup (Phase 16 root cause). 1B has
            # 4× per-sample AWQ memory of 0.6B (hidden 1024 → 2048), so use
            # the 1.7B Qwen3 budget (160 GB) directly.
            gpu_resource="$GPU_FULL"
            cpus=10
            mem="160G"
            ;;
        leanquant-nu)
            # 128 × 2048 activations + GPTQ + weighted k-means. Peak < 10 GB.
            gpu_resource="$GPU_MEDIUM"
            cpus=4
            mem="48G"
            ;;
        pb-llm)
            # PB-LLM xnor + 8-bit outliers + GPTQ Hessian. Peak ~6-8 GB.
            # Phase 16 promoted to 2g.20gb on Qwen3-1.7B; same here.
            gpu_resource="$GPU_MEDIUM"
            cpus=4
            mem="32G"
            ;;
        *)
            # fp16, rtn-2bit, gptq-2bit, sinq, doml, doml-binary, braq.
            # 2g.20gb (20 GB) covers per-method peak (5-8 GB) + full-model
            # GPU load during MMLU/HellaSwag/ARC.
            gpu_resource="$GPU_MEDIUM"
            cpus=4
            mem="32G"
            ;;
    esac
}

get_time_limit() {
    # Llama-3.2-1B is ~0.76× the wall of Qwen3-1.7B (16 layers × 1.33×
    # per-layer FLOPs vs 28 × 1.0×). Budgets below mirror Qwen3-1.7B with
    # the same generous Phase-16 margin — undershooting the model size only.
    # Alliance Nibi tiers: b1≤3h, b2≤12h, b3≤24h, b4≤72h, b5≤168h.
    case $1 in
        fp16)         echo "03:30:00" ;;  # b2 — eval-bound (~30-60 min)
        rtn-2bit)     echo "04:00:00" ;;  # b2
        gptq-2bit)    echo "04:30:00" ;;  # b2
        sinq)         echo "04:00:00" ;;  # b2
        pb-llm)       echo "05:00:00" ;;  # b2
        doml)         echo "05:00:00" ;;  # b2
        doml-binary)  echo "05:00:00" ;;  # b2
        braq)         echo "05:00:00" ;;  # b2
        tesseraq)     echo "28:00:00" ;;  # b4 — paper-exact (250 × 16 blocks)
        lnq)          echo "14:00:00" ;;  # b3 — Fisher (1024×4096) + LNQ refine
        leanquant-nu) echo "05:00:00" ;;  # b2
        *)            echo "05:00:00" ;;
    esac
}

build_python_cmd() {
    local technique=$1

    local common_ds_seed="$MODEL $DATASET --seed $SEED --device cuda:0"
    # --full_eval = PPL on (wikitext2,c4,ptb) + MMLU + HellaSwag + ARC-Easy + ARC-Challenge.
    # Each task writes its own row to BILLM_BENCH_CSV with the (dataset, metric) tag.
    local common_evals="--full_eval"

    case $technique in
        fp16)
            echo "python3 -u run.py $MODEL $DATASET fp16 --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        rtn-2bit)
            echo "python3 -u run.py $MODEL $DATASET 2bit --disable_gptq --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        gptq-2bit)
            # Paper-faithful GPTQ-2bit (--partition 1 --global_scale, per-row,
            # no groupsize). Validated on OPT-1.3B W4 within 0.5% of paper.
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
            # Phase 14B paper-exact: AWQ init + PAR + grad-clip + bfloat16
            # autocast + batched forward.
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
echo "Llama-3.2-1B PTQ Benchmark Suite (Nibi)"
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
# Note: the venv legitimately borrows idna / certifi / safetensors / yaml /
# tqdm / accelerate / typing_extensions from \$HOME/.local/. Do NOT set
# PYTHONNOUSERSITE=1 here.

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

# LNQ Fisher computation on MIG slices benefits from expandable segments.
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
# Sanity check: LlamaForCausalLM must be importable. Aborts before GPU time
# is spent if the venv is broken (e.g., torchvision ABI clash from ~/.local).
python -c "from transformers import LlamaForCausalLM; import transformers; print('transformers', transformers.__version__)" || {
    echo "FATAL: LlamaForCausalLM not importable — check transformers install."
    exit 1
}
ls -d "\$BILLM_DOWNLOADS_DIR"/$MODEL_CACHE_DIR 2>/dev/null || {
    echo "FATAL: no $MODEL snapshot under \$BILLM_DOWNLOADS_DIR — run ./sbatch/download_cache.sh first."
    exit 1
}
# Phase 11: refs/main is a prerequisite for offline from_pretrained under
# HF_HUB_OFFLINE=1 (Gotcha #22).
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
