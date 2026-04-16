#!/bin/bash
# ============================================================================
# Qwen2.5-0.5B — 2-bit & 1-bit PTQ benchmark suite (Nibi / Alliance Canada)
# ============================================================================
#
# Dispatches one sbatch job per quantization method on Qwen2.5-0.5B.
# All jobs append WikiText-2 perplexity (+ any enabled downstream evals) to
# a single shared CSV, guarded by fcntl LOCK_EX inside src/csv_utils.py.
#
# Methods benchmarked:
#
#   ~2 bits per weight ---------------------------------------------------
#     fp16          FP16 baseline (no quantization)
#     rtn-2bit      Round-to-nearest, 2-bit, no GPTQ error feedback
#     gptq-2bit     Vanilla GPTQ at 2 bits
#     sinq          SINQ, nbits=2, group_size=64   (~2.51 bpw)
#     lnq           GuidedQuant/LNQ faithful: redpajama/1024/4096,
#                   no_propagate, num_groups=4     (~2.03 bpw)
#     tesseraq      TesseraQ, bit=2, 250 optim iters (~2.25 bpw)
#     pb-llm        PB-LLM partial binarization: xnor, low_frac=0.9,
#                   high_bit=8 (~1.7 bpw — closest PB-LLM config to 2 bit)
#
#   ~1 bit per weight ----------------------------------------------------
#     doml-binary   DOML K=2 (Lloyd-Max 2-level) + GPTQ + structural
#                   partition, magnitude salience (~1.07 bpw)
#     braq          BRAQ baseline, magnitude salience (~1.07 bpw)
#
# Hyperparameters below are chosen to match each method's paper / known-
# best setting at the target bit width, so cross-method numbers are fair.
# Identifying info (model, method, bpw, seed, blocksize, salient_metric,
# extra_params JSON, quantization time, timestamp, SLURM job id) is written
# per row by src/csv_utils.py.
#
# Nibi hardware notes (from Alliance Canada Migration 2025 docs):
#   - 288 H100 80GB GPUs, 8 per node, 192 cores/node, 768 GB RAM/node
#   - MIG slices: 1g.10gb (1/8 compute, 10 GB), 2g.20gb (2/8, 20 GB),
#     3g.40gb (3/8, 40 GB), and full H100 (80 GB).
#   - Memory:compute ratio on Nibi is ~4 GB per CPU core.
#   - Cluster-specific MIG gres name may be e.g. `1g.10gb` or the Rorqual-
#     style `nvidia_h100_80gb_hbm3_1g.10gb`; set GPU_SMALL / GPU_MEDIUM
#     below once you've confirmed the correct name with `sinfo -o "%G"`.
#
# Usage:
#   ./sbatch/run_qwen_benchmark.sh                    # submit all to SLURM
#   ./sbatch/run_qwen_benchmark.sh --account def-foo  # with account
#   ./sbatch/run_qwen_benchmark.sh --local            # run serially, no SLURM
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

MODEL="Qwen/Qwen2.5-0.5B"
MODEL_SHORT="qwen25_05b"
DATASET="wikitext2"             # PPL eval dataset
SEED=0

# Shared CSV — all jobs append here with file-level locking.
CSV_NAME="${MODEL_SHORT}_ptq_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi MIG GRES strings. If `sinfo -o "%G"` shows Rorqual-style long names,
# switch to the commented alternatives below.
GPU_SMALL="--gres=gpu:1g.10gb:1"
GPU_MEDIUM="--gres=gpu:2g.20gb:1"
# GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
# GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"

# Shared quantization hyperparameters (match existing BiLLM2 defaults /
# published best settings for fair cross-method comparison at 2 bits).
BLOCKSIZE=128
SALIENT_METRIC="magnitude"      # per README / CONTEXT.md — DOML uses this
LNQ_CALIB="redpajama"           # faithful GuidedQuant calibration
LNQ_NSAMPLES=1024
LNQ_SEQLEN=4096
LNQ_NUM_GROUPS=4
LNQ_NBITS=2
SINQ_NBITS=2
SINQ_GROUPSIZE=64
TESSERAQ_BIT=2
TESSERAQ_GROUPSIZE=128
TESSERAQ_ITERATIONS=250
PBLLM_METHOD="xnor"             # PB-LLM partial-binarization variant
PBLLM_LOW_FRAC=0.9              # fraction of weights kept binary (10% high-prec)
PBLLM_HIGH_BIT=8                # high-precision tier bit width

# Methods to benchmark — one sbatch job per entry.
techniques=(
    "fp16"
    #"rtn-2bit"
    #"gptq-2bit"
    #"sinq"
    #"lnq"
    #"tesseraq"
    #"pb-llm"
    #"doml-binary"
    #"braq"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_resources() {
    # Sets: gpu_resource, cpus, mem
    # Qwen2.5-0.5B is small (~1 GB fp16). 1g.10gb covers every method
    # except LNQ (1024×4096 activations + Fisher grads) and TesseraQ
    # (250 iters with backward pass), which want 2g.20gb.
    case $1 in
        lnq|tesseraq)
            gpu_resource="$GPU_MEDIUM"
            cpus=8
            mem="32G"
            ;;
        *)
            gpu_resource="$GPU_SMALL"
            cpus=4
            mem="16G"
            ;;
    esac
}

get_time_limit() {
    # Comfortable buffer (~1.5–2x expected) per method on a 1g.10gb/2g.20gb
    # slice. Figures calibrated from known Qwen3-0.6B wall times in
    # llmdocs/CONTEXT.md scaled down for the smaller 0.5B model.
    case $1 in
        fp16)         echo "00:20:00" ;;  # eval only
        rtn-2bit)     echo "00:30:00" ;;  # no GPTQ pass
        gptq-2bit)    echo "00:45:00" ;;
        sinq)         echo "00:40:00" ;;
        pb-llm)       echo "00:50:00" ;;
        doml-binary)  echo "00:50:00" ;;
        braq)         echo "00:50:00" ;;
        tesseraq)     echo "02:30:00" ;;  # 250 optim iters
        lnq)          echo "04:00:00" ;;  # Fisher + saliency + LNQ refine
        *)            echo "01:00:00" ;;
    esac
}

build_python_cmd() {
    # Prints the python command line for a given method.
    local technique=$1

    local common_ds_seed="$MODEL $DATASET --seed $SEED --device cuda:0"
    local common_evals=""   # add "--eval_arc --eval_mmlu --eval_hellaswag" here to enable

    case $technique in
        fp16)
            echo "python3 -u run.py $MODEL $DATASET fp16 --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        rtn-2bit)
            # Pure RTN at 2 bits = the '2bit' method with GPTQ disabled.
            echo "python3 -u run.py $MODEL $DATASET 2bit --disable_gptq --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        gptq-2bit)
            echo "python3 -u run.py $MODEL $DATASET 2bit --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
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
            echo "python3 -u src/run_lnq.py $MODEL $DATASET --full_pipeline --no_propagate --calib_dataset $LNQ_CALIB --nsamples $LNQ_NSAMPLES --seqlen $LNQ_SEQLEN --num_groups $LNQ_NUM_GROUPS --nbits $LNQ_NBITS --seed $SEED --device cuda:0 $common_evals"
            ;;
        pb-llm)
            # PB-LLM has its own runner under PB-LLM/gptq_pb/run.py; it sets
            # PYTHONPATH internally to find ../../src/csv_utils.
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
        tesseraq)     echo "TesseraQ (bit=$TESSERAQ_BIT, gs=$TESSERAQ_GROUPSIZE, iters=$TESSERAQ_ITERATIONS)" ;;
        pb-llm)       echo "PB-LLM ($PBLLM_METHOD, low_frac=$PBLLM_LOW_FRAC, high_bit=$PBLLM_HIGH_BIT, blocksize=$BLOCKSIZE)" ;;
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
echo "Qwen2.5-0.5B PTQ Benchmark Suite (Nibi)"
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
        else
            CACHE_ROOT_LOCAL="$(pwd)/downloads"
        fi
        if [[ "$CACHE_ROOT_LOCAL" != "$(pwd)/downloads" && ! -L ./downloads && ! -e ./downloads ]]; then
            ln -sfn "$CACHE_ROOT_LOCAL" ./downloads
        fi
        export BILLM_BENCH_CSV="$CSV_ABS"
        export HF_HOME="$CACHE_ROOT_LOCAL/hf"
        export HF_DATASETS_CACHE="$CACHE_ROOT_LOCAL/hf/datasets"
        export HF_HUB_CACHE="$CACHE_ROOT_LOCAL/hf/hub"
        export TRANSFORMERS_CACHE="$CACHE_ROOT_LOCAL/hf"
        export TORCH_HOME="$CACHE_ROOT_LOCAL/torch"
        export NUMBA_CACHE_DIR="$CACHE_ROOT_LOCAL/.cache/numba"
        export PIP_CACHE_DIR="$CACHE_ROOT_LOCAL/.cache/pip"
        export XDG_CACHE_HOME="$CACHE_ROOT_LOCAL/.cache"
        export TMPDIR="${SLURM_TMPDIR:-$CACHE_ROOT_LOCAL/tmp}"
        export BILLM_DOWNLOADS_DIR="$CACHE_ROOT_LOCAL/downloads"
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
$gpu_resource
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$mem
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

# Route every cache to \$SCRATCH (1 TB soft / 20 TB hard on Nibi).
# /project is code-only; \$HOME has only ~50 GiB. Falling back to defaults
# corrupts arrow caches and OOMs the disk quota.
if [[ -n "\${SCRATCH:-}" && -d "\${SCRATCH}" ]]; then
    CACHE_ROOT="\$SCRATCH/billm2_cache"
else
    CACHE_ROOT="\$(pwd)/downloads"
fi

# Re-create ./downloads -> \$CACHE_ROOT symlink defensively (datautils.py
# and PB-LLM both hardcode the relative './downloads' path).
if [[ "\$CACHE_ROOT" != "\$(pwd)/downloads" && ! -L ./downloads && ! -e ./downloads ]]; then
    ln -sfn "\$CACHE_ROOT" ./downloads
fi

export HF_HOME="\$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="\$CACHE_ROOT/hf/datasets"
export HF_HUB_CACHE="\$CACHE_ROOT/hf/hub"
export TRANSFORMERS_CACHE="\$CACHE_ROOT/hf"
export TORCH_HOME="\$CACHE_ROOT/torch"
# Project-level override read by datautils.py / run.py / PB-LLM / src/run_*.py
# — replaces every hardcoded ./downloads/... path with an absolute scratch path.
export BILLM_DOWNLOADS_DIR="\$CACHE_ROOT/downloads"
# \$HOME-default caches that must not leak back: numba JIT (LNQ uses it),
# pip wheels, generic XDG cache, and unpacking TMPDIR.
export NUMBA_CACHE_DIR="\$CACHE_ROOT/.cache/numba"
export PIP_CACHE_DIR="\$CACHE_ROOT/.cache/pip"
export XDG_CACHE_HOME="\$CACHE_ROOT/.cache"
# Use the per-job NVMe local disk for transient unpacking when available
# (gets wiped at job end; perfect for arrow shard extraction).
if [[ -n "\${SLURM_TMPDIR:-}" && -d "\${SLURM_TMPDIR}" ]]; then
    export TMPDIR="\$SLURM_TMPDIR"
else
    export TMPDIR="\$CACHE_ROOT/tmp"
fi
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$HF_HUB_CACHE" "\$TORCH_HOME" \\
         "\$NUMBA_CACHE_DIR" "\$PIP_CACHE_DIR" "\$XDG_CACHE_HOME" "\$TMPDIR" \\
         "\$BILLM_DOWNLOADS_DIR"

# Compute nodes have no internet — caches must be pre-warmed via download_cache.sh.
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Shared thread-safe CSV target (fcntl LOCK_EX inside src/csv_utils.py).
export BILLM_BENCH_CSV="$CSV_ABS"

# src/ must be on PYTHONPATH so PB-LLM / standalone runners find csv_utils.
export PYTHONPATH=\$PYTHONPATH:\$(pwd):\$(pwd)/src

# LNQ Fisher computation on small MIG slices benefits from expandable segments.
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
nvidia-smi
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
