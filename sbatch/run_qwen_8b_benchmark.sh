#!/bin/bash
# ============================================================================
# Qwen3-8B — 2-bit & 1-bit PTQ benchmark suite (Nibi / Alliance Canada)
# ============================================================================
#
# Sister script to run_qwen_benchmark.sh (0.6B), run_qwen_1.7b_benchmark.sh,
# and run_qwen_4b_benchmark.sh. Dispatches one sbatch job per quantization
# method on Qwen3-8B. All jobs append WikiText-2 perplexity to a single
# shared CSV.
#
# Model facts (hidden=4096, layers=36, heads=32, kv=8, inter=12288):
#   - ~8.19B params, 16.38 GB fp16 on disk + runtime overhead
#   - attn_weights fp32 peak per sample at seqlen=4096: 2 GiB (32 heads)
#   - 36 layers × 7 sublayers = 252 sublayers
#
# Why 8B needs special handling:
#   1. Model alone is 16.38 GB — 2g.20gb (20 GB MIG) is barely enough for
#      the model and leaves nothing for activations. Bumped base slice to
#      3g.40gb (40 GB) for every method.
#   2. LNQ Fisher computation peaks at 42 GB on A40 (measured Phase 8).
#      Even 4g.40gb (40 GB) is insufficient ⇒ must use full h100 (80 GB).
#   3. LNQ 1024 samples × 4096 seqlen exceeds 125 GB system RAM on 8B
#      (Phase 8 OOM'd at layer 32/35). Use 512 samples for 8B.
#      (CONTEXT.md "Setup": must use 512 samples for 8B GuidedQuant.)
#   4. TesseraQ block backward on 8B scales ~12× the 0.6B wall. Even on
#      full h100 this is ~15-20 h. Budget 20 h; accept that it may still
#      TIME LIMIT.
#
# Resource scaling — 8B vs 0.6B on Nibi compute:
#   - LeanQuant A40 times: 0.6B 520s → 8B 2447s (4.71×)
#   - Applied to Nibi 0.6B wall times:
#       fp16 0s | sinq ~100s | rtn/gptq/braq/doml ~35 min |
#       leanquant ~55 min
#   - LNQ with 512 samples: Fisher ~1 h + saliency ~1.5 h + LNQ opt ~1 h
#     = ~3-4 h on A40-equivalent compute; ~2-3 h on full h100.
#   - TesseraQ: 0.6B Nibi was 192 min on 2g.20gb; 8B ~= 12× (params) × 1.29
#     (layers) = 15.5× wall. On full h100 (~3.5× compute of 2g.20gb):
#     192 × 15.5 / 3.5 = 850 min ≈ 14 h. Budget 20 h.
#
# Memory sizing (measured on dev-box A40 Phase 8 + computed from fp16 size):
#   - fp16 eval peak: ~19 GB
#   - GPTQ-style quantize peak: ~22-25 GB
#   - LeanQuant_nu (128×2048): ~22 GB
#   - PB-LLM: ~25 GB
#   - TesseraQ (block backward): ~28-32 GB
#   - LNQ (Fisher grads + saliency + 512×4096): 42+ GB → full h100
#
# Methods benchmarked: fp16, rtn-2bit, gptq-2bit, sinq, lnq, leanquant-nu,
# tesseraq, pb-llm, doml (flagship), doml-binary, braq.
#
# Usage:
#   ./sbatch/run_qwen_8b_benchmark.sh                    # submit all
#   ./sbatch/run_qwen_8b_benchmark.sh --account def-foo
#   ./sbatch/run_qwen_8b_benchmark.sh --local
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

MODEL="Qwen/Qwen3-8B"
MODEL_SHORT="qwen3_8b"
DATASET="wikitext2"
SEED=0

CSV_NAME="${MODEL_SHORT}_ptq_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES (full long-form names required; see Gotcha #11).
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_XLARGE="--gres=gpu:nvidia_h100_80gb_hbm3_4g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"                              # full 80 GB H100

# Shared quantization hyperparameters.
BLOCKSIZE=128
SALIENT_METRIC="magnitude"
LNQ_CALIB="redpajama"
# 8B-specific: MUST use 512 samples (not 1024) — CONTEXT.md: "8B GuidedQuant
# used 512 samples due to 125 GB RAM constraint" (Phase 8 at layer 32/35 OOM).
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
PBLLM_METHOD="xnor"
PBLLM_LOW_FRAC=0.9
PBLLM_HIGH_BIT=8

# Methods to benchmark.
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
    # 8B model alone is 16.38 GB. Base slice is 3g.40gb (40 GB) for every
    # method; LNQ escalates to full h100 (80 GB) for Fisher headroom.
    case $1 in
        lnq)
            # Fisher peaks ~42 GB (A40 measured, Phase 8). 4g.40gb would
            # be insufficient even at 512 samples — must use full h100
            # (80 GB MIG-free). 768 GB/node RAM allows generous --mem.
            gpu_resource="$GPU_FULL"
            cpus=16
            mem="180G"          # 512 samples × 4096 × 4096 × 2 bytes = 16 GB
                                # CPU-offloaded activations per layer; plus
                                # Fisher (32 GB fp16) + saliency buffers.
            ;;
        tesseraq)
            # 8B TesseraQ on 4g.40gb would be ~24 h. Full h100 (~3.5×
            # compute of 2g.20gb, 2× 4g.40gb) brings it down to ~14 h.
            # Peak GPU ~28-32 GB.
            gpu_resource="$GPU_FULL"
            cpus=12
            mem="64G"
            ;;
        leanquant-nu)
            # 128×2048 activations + GPTQ + weighted k-means on 8B:
            # model 16 + activations ~2 + blockwise work ~4 = ~22 GB peak.
            # 3g.40gb (40 GB) has comfortable margin.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="48G"
            ;;
        pb-llm)
            # PB-LLM keeps fp16 outliers + GPTQ Hessians. Peak ~25 GB on 8B.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="48G"
            ;;
        fp16)
            # Eval only — ~19 GB peak. 3g.40gb is safest.
            gpu_resource="$GPU_LARGE"
            cpus=6
            mem="32G"
            ;;
        *)
            # rtn-2bit, gptq-2bit, sinq, doml, doml-binary, braq
            # Peak 22-25 GB ⇒ 3g.40gb (40 GB) fits with headroom.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="48G"
            ;;
    esac
}

get_time_limit() {
    # Scale: 8B / 0.6B = 4.71× from LeanQuant A40 times. TesseraQ scales
    # worse (params + hidden²). Apply ~2× safety buffer.
    case $1 in
        fp16)         echo "00:45:00" ;;  # pure eval ~3 min + 16 GB model load
        # NOTE (2026-04-21): dev-box A40 measurements (confirmed, not
        # projected): DOML 1.7B 1870s = 31 min; DOML 4B 6523s = 109 min;
        # ratio 4B/1.7B = 3.49× (theoretical hidden²×layers predicted only
        # 2.01×). For 8B/4B the hidden² ratio is 2.56× and MLP is 2.02×;
        # weighted ~2.2× → 8B DOML ≈ 240 min = 4 h on A40. Nibi 3g.40gb
        # best case (3× A40) ~80 min; worst case (A40-equivalent) 240 min.
        # Budgets below give ~1.75× margin over worst case.
        rtn-2bit)     echo "05:00:00" ;;
        gptq-2bit)    echo "06:00:00" ;;
        sinq)         echo "01:00:00" ;;  # no GPTQ pass; dominated by load
        pb-llm)       echo "07:00:00" ;;  # PB-LLM backward pass on top of GPTQ
        doml)         echo "07:00:00" ;;  # 240 min A40 worst case → 420 min = 1.75× margin
        doml-binary)  echo "07:00:00" ;;  # K=2 same envelope as DOML
        braq)         echo "07:00:00" ;;  # same class as DOML
        tesseraq)     echo "20:00:00" ;;  # 0.6B Nibi 192 min × 15.5× scaling ÷
                                          # 3.5× compute on full h100 = ~14 h;
                                          # budget 20 h to absorb per-batch var
        lnq)          echo "12:00:00" ;;  # Fisher (~68 min at 512 samples on A40)
                                          # + saliency (~90 min) + LNQ opt (~60 min)
                                          # = 3-4 h on A40; ~2-3 h on h100;
                                          # 12 h budget is 3-4× expected
        leanquant-nu) echo "02:30:00" ;;  # A40 40.8 min × ~1.5× Nibi factor = 61 min
        *)            echo "03:00:00" ;;
    esac
}

build_python_cmd() {
    local technique=$1

    local common_ds_seed="$MODEL $DATASET --seed $SEED --device cuda:0"
    local common_evals=""

    case $technique in
        fp16)
            echo "python3 -u run.py $MODEL $DATASET fp16 --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        rtn-2bit)
            echo "python3 -u run.py $MODEL $DATASET 2bit --disable_gptq --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        gptq-2bit)
            echo "python3 -u run.py $MODEL $DATASET 2bit --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
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
            echo "python3 -u src/run_tesseraq.py $MODEL $DATASET --bit $TESSERAQ_BIT --group_size $TESSERAQ_GROUPSIZE --iterations $TESSERAQ_ITERATIONS --seed $SEED --device cuda:0 $common_evals"
            ;;
        lnq)
            # 8B uses 512 samples (LNQ_NSAMPLES=512) per RAM constraint.
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
        gptq-2bit)    echo "GPTQ @ 2-bit (blocksize=$BLOCKSIZE, salient=$SALIENT_METRIC)"   ;;
        sinq)         echo "SINQ (nbits=$SINQ_NBITS, group_size=$SINQ_GROUPSIZE)"           ;;
        lnq)          echo "GuidedQuant/LNQ (nbits=$LNQ_NBITS, $LNQ_CALIB/$LNQ_NSAMPLES/$LNQ_SEQLEN, groups=$LNQ_NUM_GROUPS, no_propagate)" ;;
        leanquant-nu) echo "LeanQuant_nu (nbits=$LEANQUANT_NBITS, p=$LEANQUANT_EXPONENT, percdamp=$LEANQUANT_PERCDAMP, $LEANQUANT_CALIB/$LEANQUANT_NSAMPLES/$LEANQUANT_SEQLEN, act_order+true_sequential)" ;;
        tesseraq)     echo "TesseraQ (bit=$TESSERAQ_BIT, gs=$TESSERAQ_GROUPSIZE, iters=$TESSERAQ_ITERATIONS)" ;;
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
echo "Qwen3-8B PTQ Benchmark Suite (Nibi)"
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
            # Dev-box fallback: use the repo-root ./downloads directory that
            # already holds the models. Don't append /downloads again — that
            # double-nests and breaks the offline refs/main lookup.
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
# Do NOT set PYTHONNOUSERSITE=1 — the venv legitimately borrows packages
# from \$HOME/.local/. The torchvision/torchaudio ABI crash was fixed in
# ./sbatch/fix_venv_torchvision.sh.

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

# 8B LNQ Fisher peaks at 42 GB; expandable_segments prevents
# fragmentation-OOM (Phase 8 solution on A40).
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
python -c "from transformers.models.qwen3 import Qwen3ForCausalLM; import transformers; print('transformers', transformers.__version__)" || {
    echo "FATAL: Qwen3ForCausalLM not importable — transformers is too old (<4.51)."
    exit 1
}
ls -d "\$BILLM_DOWNLOADS_DIR"/models--Qwen--Qwen3-8B 2>/dev/null || {
    echo "FATAL: no Qwen3-8B snapshot under \$BILLM_DOWNLOADS_DIR — run ./sbatch/download_cache.sh first."
    exit 1
}
for _refs in "\$BILLM_DOWNLOADS_DIR"/models--Qwen--Qwen3-8B/refs/main; do
    [ -s "\$_refs" ] || {
        echo "FATAL: \$_refs missing or empty."
        echo "       Re-run ./sbatch/download_cache.sh with Qwen3-8B added to the list;"
        echo "       needs snapshot_download(..., local_files_only=True) second pass."
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
