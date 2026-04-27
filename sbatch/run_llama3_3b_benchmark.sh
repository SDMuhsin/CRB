#!/bin/bash
# ============================================================================
# Llama-3.2-3B — 2-bit & 1-bit PTQ benchmark suite (Nibi / Alliance Canada)
# ============================================================================
#
# Sister script to run_llama3_1b_benchmark.sh and run_qwen_*_benchmark.sh.
# Dispatches one sbatch job per quantization method on
# meta-llama/Llama-3.2-3B. All jobs append results to a shared CSV via
# fcntl-locked src/csv_utils.append_result.
#
# Model facts (Llama-3.2-3B, standard config):
#   - 3.21 B params; ~6.4 GB fp16 on disk + runtime overhead
#   - hidden=3072, layers=28, heads=24, num_kv_heads=8 (GQA),
#     intermediate=8192, vocab=128256, dtype=bfloat16
#   - Closely scaled from Llama-3.2-1B: same intermediate (8192), wider
#     hidden (2048 → 3072), 1.75× layers (16 → 28)
#
# Sizing vs Qwen3-4B (the most directly comparable Qwen3 size):
#   - Llama-3.2-3B per-layer FLOPs: 3072² × 8192 = 77.3 G
#   - Qwen3-4B   per-layer FLOPs: 2560² × 9728 = 63.8 G
#   - Per-layer ratio: 1.21× ; layer ratio: 28/36 = 0.78× → net 0.94× wall
#   - Use Qwen3-4B budgets directly (slight under-utilisation, plenty of margin).
#
# Methods benchmarked: same as 1B + Qwen3 suite — fp16, rtn-2bit, gptq-2bit,
# sinq, lnq, leanquant-nu, tesseraq, pb-llm, doml (flagship), doml-binary, braq.
#
# Usage:
#   ./sbatch/run_llama3_3b_benchmark.sh                      # submit all
#   ./sbatch/run_llama3_3b_benchmark.sh --account def-foo
#   ./sbatch/run_llama3_3b_benchmark.sh --local              # serial, no SLURM
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

MODEL="meta-llama/Llama-3.2-3B"
MODEL_SHORT="llama3_3b"
DATASET="wikitext2"
SEED=0

MODEL_CACHE_DIR="models--meta-llama--Llama-3.2-3B"

CSV_NAME="${MODEL_SHORT}_ptq_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES (full long-form names required; see Gotcha #11). 4g.40gb is NOT
# provisioned (Gotcha #33).
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"

# Shared quantization hyperparameters (same paper-faithful settings as Qwen3
# suite — keeps cross-method, cross-architecture comparisons fair).
BLOCKSIZE=128
SALIENT_METRIC="magnitude"
LNQ_CALIB="redpajama"
LNQ_NSAMPLES=1024              # 3B Fisher with 1024 fits per Phase 8 scaling (4B does fine; 3B easier)
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
    # Llama-3.2-3B GPU footprint (extrapolated from Qwen3-4B Phase 16
    # measurements, scaled by parameter ratio 0.78×):
    #   - Model fp16 on GPU: ~6.4 GB
    #   - DOML/GPTQ peak (128 × 2048 calib): ~10-12 GB
    #   - LeanQuant peak: ~10-13 GB
    #   - LNQ (Fisher + 4-group saliency + 1024 × 4096): ~22-28 GB → 3g.40gb
    #   - TesseraQ paper-exact: GPU peak ~14 GB; full h100 for compute walltime
    case $1 in
        lnq)
            # Fisher grads (~6 GB fp16) + saliency Hessian + 1024×4096 attn
            # peak ~22-28 GB. 3g.40gb (40 GB) fits with margin.
            gpu_resource="$GPU_LARGE"
            cpus=10
            mem="128G"
            ;;
        tesseraq)
            # Paper-exact (iter=250, bsz=4, nsamples=512). 28 blocks × 250 iters
            # × per-sample backward pass. Memory scales with hidden² + FP32
            # block promotion + per-sample input_feat caches. Use Qwen3-4B
            # budget (256 GB / full h100) directly.
            gpu_resource="$GPU_FULL"
            cpus=12
            mem="256G"
            ;;
        leanquant-nu)
            # 128 × 2048 activations on 3B peaks ~10-13 GB GPU. 2g.20gb fits.
            gpu_resource="$GPU_MEDIUM"
            cpus=6
            mem="48G"
            ;;
        pb-llm)
            # PB-LLM keeps fp16 outliers + GPTQ Hessians. Peak ~12 GB on 3B.
            # Phase 16 promoted to 3g.40gb on Qwen3-4B for safety; same here.
            gpu_resource="$GPU_LARGE"
            cpus=6
            mem="64G"
            ;;
        *)
            # fp16, rtn-2bit, gptq-2bit, sinq, doml, doml-binary, braq
            # Peaks 9-12 GB on 3B ⇒ 2g.20gb (20 GB) fits with eval margin.
            gpu_resource="$GPU_MEDIUM"
            cpus=6
            mem="48G"
            ;;
    esac
}

get_time_limit() {
    # Llama-3.2-3B is ~0.94× the wall of Qwen3-4B (per-layer 1.21× × layer
    # ratio 0.78×). Use Qwen3-4B Phase-16 budgets directly — under-utilised
    # by ~6%, well within margin. Alliance Nibi tiers: b1≤3h, b2≤12h,
    # b3≤24h, b4≤72h, b5≤168h.
    case $1 in
        fp16)         echo "05:30:00" ;;  # b2 — eval-suite-dominated
        rtn-2bit)     echo "05:30:00" ;;  # b2
        gptq-2bit)    echo "06:30:00" ;;  # b2
        sinq)         echo "05:30:00" ;;  # b2
        pb-llm)       echo "07:30:00" ;;  # b2
        doml)         echo "08:00:00" ;;  # b2
        doml-binary)  echo "08:00:00" ;;  # b2
        braq)         echo "08:00:00" ;;  # b2
        tesseraq)     echo "48:00:00" ;;  # b4 — 250 iters × 28 blocks
        lnq)          echo "26:00:00" ;;  # b4 (off b3/b4 boundary) — Fisher 1024×4096 + LNQ refine
        leanquant-nu) echo "06:00:00" ;;  # b2
        *)            echo "06:00:00" ;;
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
echo "Llama-3.2-3B PTQ Benchmark Suite (Nibi)"
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
