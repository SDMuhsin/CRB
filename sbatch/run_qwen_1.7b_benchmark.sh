#!/bin/bash
# ============================================================================
# Qwen3-1.7B — 2-bit & 1-bit PTQ benchmark suite (Nibi / Alliance Canada)
# ============================================================================
#
# Sister script to run_qwen_benchmark.sh (which handles the 0.6B variant).
# Dispatches one sbatch job per quantization method on Qwen3-1.7B.
# All jobs append WikiText-2 perplexity to a single shared CSV, guarded
# by fcntl LOCK_EX inside src/csv_utils.py.
#
# Model facts (hidden=2048, layers=28, heads=16, kv=8, inter=6144):
#   - ~2.03B params, 4.06 GB fp16 on disk + runtime overhead
#   - attn_weights fp32 peak per sample at seqlen=4096: 1 GiB
#     (bsz=1 × 16 heads × 4096 × 4096 × 4 bytes)
#   - Measured on dev-box A40 (see /tmp/billm2_smoke/): fp16 eval
#     peaked at 5.1 GiB GPU; DOML quantization peaked at ~4.5 GiB.
#
# Resource scaling — 1.7B vs 0.6B on Nibi compute:
#   - LeanQuant A40 times: 0.6B 520s → 1.7B 865s (1.66×)
#   - Measured Nibi 0.6B wall times (from results/qwen3_06b_ptq_benchmark.csv):
#       fp16 0s | sinq 22s | 2bit(rtn) 407s | 2bit(gptq) 431s |
#       braq 432s | doml 450s
#   - ⇒ Predicted Nibi 1.7B: sinq ~37s | doml ~12 min | etc.
#
# Phase 11 lessons carried forward:
#   1. LNQ faithful (redpajama / 1024 × 4096 + Fisher + 4-group saliency)
#      did NOT fit on 2g.20gb even at 0.6B (job 12504042 OOM at 19.62 GiB).
#      1.7B is 3× the model footprint ⇒ use 3g.40gb (40 GiB).
#   2. TesseraQ at 0.6B needed 5 h on 2g.20gb; at 1.7B the backward pass
#      scales ~3× (params + 4× attn width). Promote to 4g.40gb and budget
#      10 h. If this still TIME LIMITs we move to h100 for 1.7B retry.
#   3. `refs/main` pre-flight guard matches `models--*Qwen3*` glob so it
#      catches Qwen3-1.7B too — no per-model change needed.
#
# Methods benchmarked: fp16, rtn-2bit, gptq-2bit, sinq, lnq, leanquant-nu,
# tesseraq, pb-llm, doml (flagship), doml-binary, braq.
#
# Usage:
#   ./sbatch/run_qwen_1.7b_benchmark.sh                    # submit all
#   ./sbatch/run_qwen_1.7b_benchmark.sh --account def-foo  # with account
#   ./sbatch/run_qwen_1.7b_benchmark.sh --local            # run serially
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

MODEL="Qwen/Qwen3-1.7B"
MODEL_SHORT="qwen3_1.7b"
DATASET="wikitext2"             # PPL eval dataset
SEED=0

# Shared CSV — all jobs append here with file-level locking.
CSV_NAME="${MODEL_SHORT}_ptq_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES strings. Full long-form MIG names are REQUIRED — short forms
# like `1g.10gb` are rejected (see Gotcha #11 in CONTEXT.md).
#   h100                            (full 80 GB H100)
#   nvidia_h100_80gb_hbm3_3g.40gb   (3/8 compute, 40 GB)
#   nvidia_h100_80gb_hbm3_2g.20gb   (2/8 compute, 20 GB)
#   nvidia_h100_80gb_hbm3_1g.10gb   (1/8 compute, 10 GB)
# NOTE (2026-04-21): 4g.40gb was expected per the CC MIG catalogue but is
# NOT provisioned on Nibi. sinfo confirms only 1g.10gb/2g.20gb/3g.40gb and
# full h100 are advertised. Jobs that request 4g.40gb FAIL instantly with
# ExitCode=1:0, Elapsed=00:00:00, and no .out/.err written (silent
# unsatisfiable-gres rejection). Killed jobs 12539876/12539886/12539888.
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"                             # full 80 GB H100 — tesseraq compute driver

# Shared quantization hyperparameters (match existing BiLLM2 defaults /
# published best settings for fair cross-method comparison at 2 bits).
BLOCKSIZE=128
SALIENT_METRIC="magnitude"      # per README / CONTEXT.md — DOML uses this
LNQ_CALIB="redpajama"           # faithful GuidedQuant calibration
LNQ_NSAMPLES=1024               # 1.7B Fisher + saliency fits on ~64 GB RAM
LNQ_SEQLEN=4096
LNQ_NUM_GROUPS=4
LNQ_NBITS=2
# LeanQuant_nu — paper-faithful settings (arXiv:2407.10032, README 2-bit recipe)
LEANQUANT_CALIB="redpajama"
LEANQUANT_NSAMPLES=128
LEANQUANT_SEQLEN=2048
LEANQUANT_NBITS=2
LEANQUANT_EXPONENT=4.0          # p in sample_weight = diag(Hinv)^(-p)
LEANQUANT_PERCDAMP=0.1          # README's 2-bit recipe
SINQ_NBITS=2
SINQ_GROUPSIZE=64
TESSERAQ_BIT=2
TESSERAQ_GROUPSIZE=128
TESSERAQ_ITERATIONS=250
PBLLM_METHOD="xnor"             # PB-LLM partial-binarization variant
PBLLM_LOW_FRAC=0.9
PBLLM_HIGH_BIT=8

# Methods to benchmark — one sbatch job per entry.
# Pared down 2026-04-21 to ONLY the methods that still need a result for
# Qwen3-1.7B. See ./results/qwen3_1.7b_ptq_benchmark.csv for what already
# landed from the 12539xxx batch. Uncomment a line to re-queue that method.
techniques=(
    # "fp16"          # job 12539870 COMPLETED, PPL 16.72
    # "rtn-2bit"      # job 12539871 COMPLETED, PPL 4.64M (quality collapse, not a crash)
    # "gptq-2bit"     # job 12539872 COMPLETED, PPL 156k  (quality collapse, not a crash)
    # "sinq"          # job 12539873 COMPLETED, PPL 30,930
    # "lnq"           # job 12539874 currently RUNNING on 3g.40gb — do NOT re-queue
    # "leanquant-nu"  # job 12539875 COMPLETED, PPL 128.81
    "tesseraq"      # job 12539876 FAILED (unsatisfiable GPU_XLARGE=4g.40gb); MIG fix applied, re-run
    "pb-llm"        # job 12539877 FAILED in Python (LocalEntryNotFoundError); root cause TBD —
                    # don't actually re-queue until diagnose_pb_llm_compute.sh returns a fix.
    # "doml"          # job 12539878 COMPLETED, PPL 35.01
    # "doml-binary"   # job 12539879 COMPLETED, PPL 42,639
    # "braq"          # job 12539880 COMPLETED, PPL 124k
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_resources() {
    # Sets: gpu_resource, cpus, mem
    #
    # Qwen3-1.7B footprint at runtime:
    #   - Model fp16 on GPU: ~4 GB (measured: fp16 eval peaked at 5.1 GB)
    #   - DOML quantization peak (128 × 2048 calib): ~5-7 GB (measured)
    #   - LeanQuant (128 × 2048 activations): ~7-9 GB
    #   - LNQ (Fisher grads + 4-group saliency Hessian + 1024 × 4096 calib):
    #     ≥ 3× the 0.6B peak of 15 GB ⇒ ≥ 30 GB ⇒ 3g.40gb (40 GB).
    #   - TesseraQ (20 steps × 250 iter backward pass on each of 28 blocks):
    #     peak GPU ~10-14 GB, but compute-bound; larger slice for walltime.
    case $1 in
        lnq)
            # 0.6B LNQ peaked at 15.19 GB on 2g.20gb before OOM (job 12504042).
            # 1.7B Fisher grads ≈ 3× bigger, saliency Hessian 4× (hidden²).
            # 3g.40gb has 40 GB MIG — expected peak ~30 GB.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="64G"
            ;;
        tesseraq)
            # 0.6B TesseraQ needed 5h on 2g.20gb (Phase 11). 1.7B with 3×
            # the block-size and wider MLP scales ~3-5× compute. The
            # previous 4g.40gb plan does not schedule on Nibi (see GPU
            # table note above), so go straight to full h100 — 8× compute
            # of 2g.20gb finishes the same work in ~2-3 h.
            gpu_resource="$GPU_FULL"
            cpus=8
            mem="32G"
            ;;
        leanquant-nu)
            # 128 × 2048 activations on 1.7B ≈ 1 GB CPU-side (below 8 GB
            # threshold, so no CPU offload). Peak GPU ~7-9 GB → fits 1g.10gb
            # but GPTQ propagation + weighted k-means wants headroom. Use
            # 2g.20gb for safety.
            gpu_resource="$GPU_MEDIUM"
            cpus=4
            mem="24G"
            ;;
        pb-llm)
            # PB-LLM xnor with high_frac=0.1 at 8-bit keeps ~200 MB in fp16
            # outliers per layer + GPTQ Hessians. Peak ~7-8 GB on 1.7B.
            gpu_resource="$GPU_SMALL"
            cpus=4
            mem="16G"
            ;;
        *)
            # fp16, rtn-2bit, gptq-2bit, sinq, doml, doml-binary, braq
            # Peaks measured / projected at 5-8 GB on 1.7B — 1g.10gb fits.
            gpu_resource="$GPU_SMALL"
            cpus=4
            mem="16G"
            ;;
    esac
}

get_time_limit() {
    # Scale baseline from measured Nibi 0.6B wall times × LeanQuant scaling
    # factor (1.7B / 0.6B = 1.66). Apply ~2× safety buffer on top.
    case $1 in
        fp16)         echo "00:30:00" ;;  # pure eval ~1 min; plenty of buffer for model load
        rtn-2bit)     echo "00:45:00" ;;  # 0.6B Nibi 407s × 1.66 = 676s = 11 min
        gptq-2bit)    echo "01:00:00" ;;  # 0.6B Nibi 431s × 1.66 = 715s = 12 min
        sinq)         echo "00:45:00" ;;  # 0.6B Nibi 22s × 1.66 = 37s
        pb-llm)       echo "01:30:00" ;;  # not on Nibi CSV; est 20 min from A40 scaling
        doml)         echo "01:30:00" ;;  # 0.6B Nibi 450s × 1.66 = 747s = 12.5 min
        doml-binary)  echo "01:30:00" ;;  # K=2 variant — same cost envelope as DOML
        braq)         echo "01:30:00" ;;  # 0.6B Nibi 432s × 1.66 = 717s = 12 min
        tesseraq)     echo "05:00:00" ;;  # 0.6B Nibi >150 min on 2g.20gb before TIME LIMIT;
                                          # 1.7B on full h100 (8× 2g.20gb compute) ⇒ ~2-3 h; 5 h buffer
        lnq)          echo "08:00:00" ;;  # Fisher (~1024 × ~3s/sample on 1.7B = 50 min)
                                          # + saliency Hessian (~50 min) + LNQ optimize
                                          # (33 min on dev-box A40 Phase-6 core only) × 1.5
                                          # for full pipeline ⇒ ~3-4 h, budget 8 h.
        leanquant-nu) echo "01:15:00" ;;  # A40 14.4 min × ~2× Nibi 1g/2g.20gb factor = 30 min
        *)            echo "02:00:00" ;;
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
echo "Qwen3-1.7B PTQ Benchmark Suite (Nibi)"
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
# Note: the venv legitimately borrows idna / certifi / safetensors / yaml /
# tqdm / accelerate / typing_extensions from \$HOME/.local/. Do NOT set
# PYTHONNOUSERSITE=1 here — it would break the import chain. The
# torchvision/torchaudio ABI crash was fixed by uninstalling those two
# specifically (./sbatch/fix_venv_torchvision.sh).

# Route every cache to \$SCRATCH (1 TB soft / 20 TB hard on Nibi).
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
python -c "from transformers.models.qwen3 import Qwen3ForCausalLM; import transformers; print('transformers', transformers.__version__)" || {
    echo "FATAL: Qwen3ForCausalLM not importable — transformers is too old (<4.51)."
    exit 1
}
ls -d "\$BILLM_DOWNLOADS_DIR"/models--Qwen--Qwen3-1.7B 2>/dev/null || {
    echo "FATAL: no Qwen3-1.7B snapshot under \$BILLM_DOWNLOADS_DIR — run ./sbatch/download_cache.sh first."
    exit 1
}
# Phase 11: refs/main is a prerequisite for offline from_pretrained under
# HF_HUB_OFFLINE=1. Verify it exists and is non-empty (gotchas 22, 26).
for _refs in "\$BILLM_DOWNLOADS_DIR"/models--Qwen--Qwen3-1.7B/refs/main; do
    [ -s "\$_refs" ] || {
        echo "FATAL: \$_refs missing or empty."
        echo "       Re-run ./sbatch/download_cache.sh with Qwen3-1.7B added to the list;"
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
