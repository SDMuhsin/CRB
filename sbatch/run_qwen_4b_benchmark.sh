#!/bin/bash
# ============================================================================
# Qwen3-4B — 2-bit & 1-bit PTQ benchmark suite (Nibi / Alliance Canada)
# ============================================================================
#
# Sister script to run_qwen_benchmark.sh (0.6B) and run_qwen_1.7b_benchmark.sh
# (1.7B). Dispatches one sbatch job per quantization method on Qwen3-4B.
# All jobs append WikiText-2 perplexity to a single shared CSV.
#
# Model facts (hidden=2560, layers=36, heads=32, kv=8, inter=9728):
#   - ~4.13B params, 8.26 GB fp16 on disk + runtime overhead
#   - attn_weights fp32 peak per sample at seqlen=4096: 2 GiB
#     (bsz=1 × 32 heads × 4096 × 4096 × 4 bytes) — DOUBLE 0.6B/1.7B (16 heads)
#   - 36 layers instead of 28 for 0.6B/1.7B → ~29% more sublayers to quantize
#
# Why 4B is materially harder than 1.7B:
#   1. Model load alone is 8.3 GB — 1g.10gb (10 GB MIG) no longer enough for
#      most methods; bumped base slice to 2g.20gb.
#   2. attn_weights peak per sample is 2 GiB (double the 1.7B value) due to
#      32 heads. This bites all methods that do full-seqlen forward passes.
#   3. 36 layers × 7 sublayers = 252 sublayers vs 196 for 0.6B/1.7B.
#
# Resource scaling — 4B vs 0.6B on Nibi compute:
#   - LeanQuant A40 times: 0.6B 520s → 4B 2193s (4.22×)
#   - Applied to measured Nibi 0.6B wall times:
#       fp16 0s | sinq ~90s | rtn/gptq/braq/doml ~30 min |
#       leanquant ~45 min on 2g.20gb
#   - LNQ scales with Fisher (params × nsamples × seqlen): Phase 8 measured
#     ~7s/sample on 4B with gradient checkpointing ⇒ Fisher alone = 2 h on
#     1024 samples. Add saliency Hessian + LNQ optimize ⇒ ~6 h total.
#   - TesseraQ block opt scales with hidden² (wider MLP) + more blocks:
#     ~5× 0.6B wall on same slice. Must use 4g.40gb (2× compute of 2g.20gb)
#     to keep under 12 h walltime.
#
# Memory sizing (measured on dev-box A40 + computed from first principles):
#   - fp16 eval peak: ~11 GB  (model 8 + activation buffer + headroom)
#   - GPTQ-style quantize peak: ~12-14 GB
#   - LeanQuant_nu (128×2048 calib): ~13 GB
#   - PB-LLM (GPTQ + fp16 outlier cache): ~14 GB
#   - TesseraQ (per-block backward): ~16-20 GB
#   - LNQ (Fisher grads + 4-group saliency at 1024×4096): ~25-30 GB
#
# Methods benchmarked: fp16, rtn-2bit, gptq-2bit, sinq, lnq, leanquant-nu,
# tesseraq, pb-llm, doml (flagship), doml-binary, braq.
#
# Usage:
#   ./sbatch/run_qwen_4b_benchmark.sh                    # submit all
#   ./sbatch/run_qwen_4b_benchmark.sh --account def-foo
#   ./sbatch/run_qwen_4b_benchmark.sh --local
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

MODEL="Qwen/Qwen3-4B"
MODEL_SHORT="qwen3_4b"
DATASET="wikitext2"
SEED=0

CSV_NAME="${MODEL_SHORT}_ptq_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES (full long-form names required; see Gotcha #11).
# NOTE (2026-04-21): 4g.40gb is NOT provisioned on Nibi — sinfo shows only
# 1g.10gb / 2g.20gb / 3g.40gb / full h100. Jobs requesting 4g.40gb fail
# instantly with ExitCode=1:0 and no .out/.err (silent unsatisfiable-gres).
# Killed jobs 12539886 (4b lnq) and 12539888 (4b tesseraq).
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"                              # full 80 GB H100

# Shared quantization hyperparameters.
BLOCKSIZE=128
SALIENT_METRIC="magnitude"
LNQ_CALIB="redpajama"
LNQ_NSAMPLES=1024               # 4B Fisher with 1024 fits per Phase 8
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
# Pared down 2026-04-21 to ONLY the methods that still need a result for
# Qwen3-4B. See ./results/qwen3_4b_ptq_benchmark.csv for what already
# landed from the 12539xxx batch. Uncomment a line to re-queue that method.
techniques=(
    # "fp16"          # job 12539882 COMPLETED, PPL 13.67
    # "rtn-2bit"      # job 12539883 COMPLETED, PPL 256,762 (quality collapse)
    # "gptq-2bit"     # job 12539884 COMPLETED, PPL 692.9    (quality collapse)
    # "sinq"          # job 12539885 COMPLETED, PPL 131,113  (quality collapse)
    "lnq"           # job 12539886 FAILED (unsatisfiable GPU_XLARGE=4g.40gb); MIG fix: now 3g.40gb
    # "leanquant-nu"  # job 12539887 COMPLETED, PPL 45.83
    "tesseraq"      # job 12539888 FAILED (unsatisfiable GPU_XLARGE=4g.40gb); MIG fix: now full h100
    "pb-llm"        # job 12539889 FAILED in Python (LocalEntryNotFoundError); root cause TBD —
                    # don't actually re-queue until diagnose_pb_llm_compute.sh returns a fix.
    # "doml"          # job 12539890 COMPLETED, PPL 15.84
    # "doml-binary"   # job 12539891 COMPLETED, PPL 1,634
    # "braq"          # job 12539892 COMPLETED, PPL 357.3
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_resources() {
    # Qwen3-4B baseline slice is 2g.20gb because the 8.3 GB model + any
    # non-trivial activations will not fit in 1g.10gb (10 GB).
    case $1 in
        lnq)
            # Fisher grads (8 GB fp16) + saliency Hessian + 1024×4096 peak
            # attn = ~25-30 GB expected. 3g.40gb (40 GB MIG) fits — same
            # slice the currently-running 1.7B lnq (12539874) uses. Memory-
            # bound so 3/8 vs 4/8 compute doesn't change the schedule much.
            gpu_resource="$GPU_LARGE"
            cpus=10
            mem="96G"
            ;;
        tesseraq)
            # 4B TesseraQ on 2g.20gb projected ~20 h (5× 0.6B wall). Use
            # full h100 (8× 2g.20gb compute) — finishes in ~4-6 h.
            gpu_resource="$GPU_FULL"
            cpus=10
            mem="48G"
            ;;
        leanquant-nu)
            # 128×2048 at 4B peaks ~13 GB GPU; 2g.20gb fits with margin.
            gpu_resource="$GPU_MEDIUM"
            cpus=6
            mem="32G"
            ;;
        pb-llm)
            # PB-LLM keeps fp16 outliers + GPTQ Hessians. Peak ~14 GB on 4B.
            gpu_resource="$GPU_MEDIUM"
            cpus=6
            mem="32G"
            ;;
        *)
            # fp16, rtn-2bit, gptq-2bit, sinq, doml, doml-binary, braq
            # Peaks 11-14 GB on 4B ⇒ 2g.20gb fits.
            gpu_resource="$GPU_MEDIUM"
            cpus=6
            mem="32G"
            ;;
    esac
}

get_time_limit() {
    # Scale: 4B / 0.6B = 4.22× from LeanQuant A40 times; verified by the
    # 36/28 layer ratio × 2560²/1024² hidden ratio ~= 4× scaling. Apply
    # ~2× safety buffer on top of baseline estimate.
    case $1 in
        fp16)         echo "00:30:00" ;;  # pure eval ~2 min even with loading
        # NOTE (2026-04-21): dev-box A40 DOML measurements — 1.7B 1870s =
        # 31 min; 4B projecting ~92 min at 65% progress (ratio 2.97×).
        # That's ~50% above the hidden²×layers theory (2.01×), so MLP and
        # num_heads=32 widen the sublayer workload more than expected.
        # On Nibi 2g.20gb (2× compute of 1g.10gb ≈ 2× A40): ~46 min best
        # case, ~92 min worst case (if slice underdelivers). Budgets give
        # 1.6–2× margin over worst case.
        rtn-2bit)     echo "02:00:00" ;;
        gptq-2bit)    echo "02:30:00" ;;
        sinq)         echo "00:45:00" ;;  # 0.6B Nibi 22s × 4.22 = ~93s
        pb-llm)       echo "03:30:00" ;;  # PB-LLM backward pass on top of GPTQ
        doml)         echo "03:00:00" ;;  # measured 92 min A40 → budget 180 min
        doml-binary)  echo "03:00:00" ;;  # K=2 same envelope as DOML
        braq)         echo "03:00:00" ;;
        tesseraq)     echo "10:00:00" ;;  # 0.6B Nibi 192 min × 5× scaling ÷ 8× compute (full h100)
                                          # = 120 min = 2 h; 10 h budget for safety.
        lnq)          echo "18:00:00" ;;  # Fisher 2 h + saliency 2 h + LNQ optimize ~1.5 h
                                          # = 5-6 h on A40-equivalent. On 3g.40gb (~0.75× 4g.40gb
                                          # compute) expect 6-9 h; budget 18 h for safety. If it
                                          # runs long, retry on full h100 (compute 2.5× 3g.40gb).
        leanquant-nu) echo "02:00:00" ;;  # A40 36.6 min × ~1.5× Nibi factor = 55 min
        *)            echo "02:30:00" ;;
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
echo "Qwen3-4B PTQ Benchmark Suite (Nibi)"
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
ls -d "\$BILLM_DOWNLOADS_DIR"/models--Qwen--Qwen3-4B 2>/dev/null || {
    echo "FATAL: no Qwen3-4B snapshot under \$BILLM_DOWNLOADS_DIR — run ./sbatch/download_cache.sh first."
    exit 1
}
for _refs in "\$BILLM_DOWNLOADS_DIR"/models--Qwen--Qwen3-4B/refs/main; do
    [ -s "\$_refs" ] || {
        echo "FATAL: \$_refs missing or empty."
        echo "       Re-run ./sbatch/download_cache.sh with Qwen3-4B added to the list;"
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
