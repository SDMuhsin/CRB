#!/bin/bash
# ============================================================================
# Llama-3.2-1B — higher-bit (3/4/5-bit) PTQ fan-out benchmark suite (Nibi)
# ============================================================================
#
# Sister script to run_llama3_1b_benchmark.sh. Dispatches one sbatch job per
# (method, bit-width) combination on Llama-3.2-1B, writing to a SEPARATE CSV
# from the 2-bit/1-bit benchmark matrix so the legacy paper rows stay clean.
#
# Methods × bit-widths benchmarked:
#
#   DOML            (run.py doml --codebook_bits {3,4,5})       → tags
#                   doml-3bit / doml-4bit / doml-5bit
#   TesseraQ        (src/run_tesseraq.py --bit {3,4,5})         → tags
#                   tesseraq-3bit / tesseraq-4bit / tesseraq-5bit
#   LeanQuant_nu    (src/run_leanquant.py --nbits {3,4,5})      → tags
#                   leanquant_nu-3bit / leanquant_nu-4bit / leanquant_nu-5bit
#   GuidedQuant+LNQ (src/run_lnq.py --full_pipeline --nbits {3,4,5}) → tags
#                   guidedquant-3bit / guidedquant-4bit / guidedquant-5bit
#
# 4 methods × 3 bit-widths = 12 jobs. Each job runs --full_eval and writes
# 7 CSV rows to llama3_1b_higher_bits_benchmark.csv.
#
# Walltimes are padded ~50–60% over the 2-bit reference budgets in the legacy
# Llama-3-1B script. Llama-3.2-1B is ~0.76× the wall of Qwen3-1.7B (16 layers
# vs 28 + 1.33× per-layer FLOPs), so 1B-class budgets stay comfortably under
# 1.7B-class envelopes even at K=32.
#
# Usage:
#   ./sbatch/run_llama3_1b_higher_bits_benchmark.sh                    # submit all
#   ./sbatch/run_llama3_1b_higher_bits_benchmark.sh --account def-foo
#   ./sbatch/run_llama3_1b_higher_bits_benchmark.sh --local            # serial, no SLURM
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

# Separate CSV — keeps the higher-bit fan-out rows out of the legacy 2-bit
# benchmark matrix so paper-table dedupe is not affected.
CSV_NAME="${MODEL_SHORT}_higher_bits_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES strings (full long-form MIG names — short forms rejected).
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"

# Shared quantization hyperparameters (match 2-bit suite for fair comparison).
BLOCKSIZE=128
SALIENT_METRIC="magnitude"
LNQ_CALIB="redpajama"
LNQ_NSAMPLES=1024
LNQ_SEQLEN=4096
LNQ_NUM_GROUPS=4
LEANQUANT_CALIB="redpajama"
LEANQUANT_NSAMPLES=128
LEANQUANT_SEQLEN=2048
LEANQUANT_EXPONENT=4.0
LEANQUANT_PERCDAMP=0.1
TESSERAQ_GROUPSIZE=128
TESSERAQ_ITERATIONS=250
TESSERAQ_BATCH_SIZE=4
TESSERAQ_NSAMPLES=512

# Methods × bit-widths to benchmark (4 × 3 = 12 jobs).
techniques=(
    "doml-3bit"
    "doml-4bit"
    "doml-5bit"
    "tesseraq-3bit"
    "tesseraq-4bit"
    "tesseraq-5bit"
    "leanquant-nu-3bit"
    "leanquant-nu-4bit"
    "leanquant-nu-5bit"
    "lnq-3bit"
    "lnq-4bit"
    "lnq-5bit"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_resources() {
    # Resource buckets mirror the 2-bit Llama-3-1B reference suite. The
    # higher-K codebook adds at most ~256 MB to the Lloyd-Max distance
    # tensor at K=32 on Llama-3-1B (rows × cols × K × 4 bytes), well within
    # MIG headroom.
    case $1 in
        lnq-*bit)
            # Same as 2-bit lnq: Llama-3-1B has hidden=2048 + 16 layers,
            # smaller per-layer footprint than Qwen3-1.7B. 3g.40gb / 96 GB
            # host RAM matches the legacy budget.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="96G"
            ;;
        tesseraq-*bit)
            # Same as 2-bit tesseraq paper-exact (iter=250, bsz=4, nsamples=512).
            # AWQ grid + PAR optimization are bit-independent in dominant
            # cost. Full h100 + 160 GB RAM matches the Phase-16 1.7B-equivalent
            # budget that the legacy 2-bit Llama-3-1B script uses.
            gpu_resource="$GPU_FULL"
            cpus=10
            mem="160G"
            ;;
        leanquant-nu-*bit)
            # 128 × 2048 activations + GPTQ + weighted k-means; K=32 case
            # is ~8× slower in k-means but still seconds.
            gpu_resource="$GPU_MEDIUM"
            cpus=4
            mem="48G"
            ;;
        doml-*bit)
            # DOML K=32 Lloyd-Max iter ~8× slower than K=4 in the inner
            # Voronoi loop, but Lloyd-Max is a small fraction of total wall.
            # 2g.20gb covers 2-bit base methods on this model size.
            gpu_resource="$GPU_MEDIUM"
            cpus=4
            mem="32G"
            ;;
        *)
            gpu_resource="$GPU_MEDIUM"
            cpus=4
            mem="32G"
            ;;
    esac
}

get_time_limit() {
    # 50–60% margin over the legacy 2-bit Llama-3-1B budgets to absorb
    # O(K) scaling in codebook-fitting steps. Alliance Nibi tiers:
    # b1≤3h, b2≤12h, b3≤24h, b4≤72h, b5≤168h. Walltimes nudged off
    # boundary values.
    case $1 in
        doml-*bit)         echo "08:00:00" ;;  # b2 — 2-bit was 5h, +60% for K-scaling
        tesseraq-*bit)     echo "44:00:00" ;;  # b4 — 2-bit was 28h; bit-independent dominant cost, +57% pad
        lnq-*bit)          echo "22:00:00" ;;  # b3 — 2-bit was 14h; Fisher dominates, +57% pad
        leanquant-nu-*bit) echo "08:00:00" ;;  # b2 — 2-bit was 5h, +60%
        *)                 echo "08:00:00" ;;
    esac
}

build_python_cmd() {
    local technique=$1

    local common_evals="--full_eval"

    local bits=${technique##*-}
    bits=${bits%bit}

    case $technique in
        doml-*bit)
            echo "python3 -u run.py $MODEL $DATASET doml --codebook_bits $bits --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        tesseraq-*bit)
            echo "python3 -u src/run_tesseraq.py $MODEL $DATASET --bit $bits --group_size $TESSERAQ_GROUPSIZE --iterations $TESSERAQ_ITERATIONS --batch_size $TESSERAQ_BATCH_SIZE --nsamples $TESSERAQ_NSAMPLES --seed $SEED --device cuda:0 $common_evals"
            ;;
        leanquant-nu-*bit)
            echo "python3 -u src/run_leanquant.py $MODEL $DATASET --nbits $bits --exponent $LEANQUANT_EXPONENT --percdamp $LEANQUANT_PERCDAMP --true_sequential --act_order --calib_dataset $LEANQUANT_CALIB --nsamples $LEANQUANT_NSAMPLES --seqlen $LEANQUANT_SEQLEN --seed $SEED --device cuda:0 $common_evals"
            ;;
        lnq-*bit)
            echo "python3 -u src/run_lnq.py $MODEL $DATASET --full_pipeline --no_propagate --calib_dataset $LNQ_CALIB --nsamples $LNQ_NSAMPLES --seqlen $LNQ_SEQLEN --num_groups $LNQ_NUM_GROUPS --nbits $bits --seed $SEED --device cuda:0 $common_evals"
            ;;
        *)
            echo "echo 'Unknown technique: $technique'; exit 1"
            ;;
    esac
}

get_technique_desc() {
    local bits=${1##*-}; bits=${bits%bit}
    case $1 in
        doml-*bit)         echo "DOML (K=$((2**bits)) Lloyd-Max + GPTQ + structural partition, ${bits}-bit codebook)" ;;
        tesseraq-*bit)     echo "TesseraQ paper-exact (bit=$bits, gs=$TESSERAQ_GROUPSIZE, iters=$TESSERAQ_ITERATIONS, AWQ init on)" ;;
        leanquant-nu-*bit) echo "LeanQuant_nu (nbits=$bits, p=$LEANQUANT_EXPONENT, $LEANQUANT_CALIB/$LEANQUANT_NSAMPLES/$LEANQUANT_SEQLEN)" ;;
        lnq-*bit)          echo "GuidedQuant/LNQ (nbits=$bits, $LNQ_CALIB/$LNQ_NSAMPLES/$LNQ_SEQLEN, groups=$LNQ_NUM_GROUPS, no_propagate)" ;;
    esac
}

# ============================================================================
# MAIN LOOP
# ============================================================================

job_count=0
mkdir -p ./logs ./results

echo "============================================"
echo "Llama-3.2-1B PTQ Higher-Bits Fan-out (Nibi)"
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
