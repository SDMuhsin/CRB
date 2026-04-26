#!/bin/bash
# ============================================================================
# Qwen3-0.6B — 2-bit & 1-bit PTQ benchmark suite (Nibi / Alliance Canada)
# ============================================================================
#
# Dispatches one sbatch job per quantization method on Qwen3-0.6B.
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
#     leanquant-nu  LeanQuant_nu faithful: per-row weighted k-means over
#                   diag(Hinv)^(-p=4) + GPTQ, redpajama/128/2048,
#                   act_order + true_sequential + propagate
#                                                   (~2.05 bpw)
#     tesseraq      TesseraQ, bit=2, 250 optim iters (~2.25 bpw)
#     pb-llm        PB-LLM partial binarization: xnor, low_frac=0.9,
#                   high_bit=8 (~1.7 bpw — closest PB-LLM config to 2 bit)
#     doml          DOML K=4 (Lloyd-Max 4-level) + GPTQ + structural
#                   partition, magnitude salience (~2.06–2.15 bpw — flagship)
#
#   ~1 bit per weight ----------------------------------------------------
#     doml-binary   DOML K=2 variant (~1.07 bpw)
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

MODEL="Qwen/Qwen3-0.6B"
MODEL_SHORT="qwen3_06b"
DATASET="wikitext2"             # PPL eval dataset
SEED=0

# Shared CSV — all jobs append here with file-level locking.
CSV_NAME="${MODEL_SHORT}_ptq_benchmark.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES strings. Verified against SLURM error message — short forms like
# `1g.10gb` are NOT registered; you must use the full nvidia_h100_80gb_hbm3_*
# prefix. Authoritative list of accepted GPU types on Nibi:
#   h100                            (full 80 GB H100)
#   nvidia_h100_80gb_hbm3_4g.40gb   (4/8 compute, 40 GB)
#   nvidia_h100_80gb_hbm3_3g.40gb   (3/8 compute, 40 GB)
#   nvidia_h100_80gb_hbm3_2g.20gb   (2/8 compute, 20 GB)
#   nvidia_h100_80gb_hbm3_1g.10gb   (1/8 compute, 10 GB)
#   mi300a, a100, a5000, t4
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"    # 40 GB MIG — for lnq (Fisher + 1024×4096 saliency Hessians)

# Shared quantization hyperparameters (match existing BiLLM2 defaults /
# published best settings for fair cross-method comparison at 2 bits).
BLOCKSIZE=128
SALIENT_METRIC="magnitude"      # per README / CONTEXT.md — DOML uses this
LNQ_CALIB="redpajama"           # faithful GuidedQuant calibration
LNQ_NSAMPLES=1024
LNQ_SEQLEN=4096
LNQ_NUM_GROUPS=4
LNQ_NBITS=2
# LeanQuant_nu — paper-faithful settings (arXiv:2407.10032, README 2-bit recipe)
LEANQUANT_CALIB="redpajama"     # same C4 shard the paper uses
LEANQUANT_NSAMPLES=128
LEANQUANT_SEQLEN=2048
LEANQUANT_NBITS=2
LEANQUANT_EXPONENT=4.0          # p in sample_weight = diag(Hinv)^(-p)
LEANQUANT_PERCDAMP=0.1          # README's 2-bit recipe
SINQ_NBITS=2
SINQ_GROUPSIZE=64
TESSERAQ_BIT=2
TESSERAQ_GROUPSIZE=128
TESSERAQ_ITERATIONS=250        # paper default (20 thresholds × 250 iters = 5000 iters/block)
TESSERAQ_BATCH_SIZE=4          # paper default
TESSERAQ_NSAMPLES=512          # paper default (was 128); 4× more calibration data + per-sample AWQ forwards
PBLLM_METHOD="xnor"             # PB-LLM partial-binarization variant
PBLLM_LOW_FRAC=0.9              # fraction of weights kept binary (10% high-prec)
PBLLM_HIGH_BIT=8                # high-precision tier bit width

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
    #"doml-binary"
    "braq"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_resources() {
    # Sets: gpu_resource, cpus, mem
    # Qwen3-0.6B is small (~1.2 GB fp16). 1g.10gb covers every method
    # except LNQ (1024×4096 activations + Fisher grads) and TesseraQ
    # (250 iters with backward pass), which want 2g.20gb.
    case $1 in
        lnq)
            # 2g.20gb was insufficient: job 12504042 died with CUDA OOM at
            # Qwen3-0.6B Layer-1 attention-softmax (19.62 GB cap, 15.19 GB in
            # use, tried to alloc 1024 MiB for attn_weights fp32 softmax).
            # The faithful Phase-8 config (redpajama / 1024 × 4096 / Fisher
            # gradients + 4-group saliency Hessian + full-seqlen activations)
            # needs ≥ ~25 GB peak. 3g.40gb gives 40 GB headroom.
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="48G"
            ;;
        tesseraq)
            # Paper-exact TesseraQ on 0.6B (iter=250, bsz=4, nsamples=512).
            # CPU peak: 4 input_feat subsets × 4 GB + 4 GB FP32 targets +
            # 4 GB inps/outs + 2 GB model = ~26 GB. Bump --mem 32 → 48 GB for
            # safety.
            gpu_resource="$GPU_MEDIUM"
            cpus=8
            mem="48G"
            ;;
        leanquant-nu)
            # 128×2048 activations (~512 MB on 0.6B) + per-row weighted k-means
            # fit comfortably on 1g.10gb. Keeps CPU count higher than DOML —
            # k-means is GPU-vectorised but CPU is used for calibration
            # pre-tokenisation (flock-serialised against the shared cache).
            gpu_resource="$GPU_SMALL"
            cpus=4
            mem="16G"
            ;;
        *)
            gpu_resource="$GPU_SMALL"
            cpus=4
            mem="16G"
            ;;
    esac
}

get_time_limit() {
    # Phase 15 update: budgets now include the full eval suite (PPL on
    # wikitext2/c4/ptb + MMLU + HellaSwag + ARC-Easy + ARC-Challenge),
    # which adds ~30-60 min on Qwen3-0.6B beyond quantization. All
    # ~50-min budgets bumped to 3 h to leave 2-3x headroom for the
    # downstream eval pass.
    case $1 in
        fp16)         echo "02:00:00" ;;  # eval only — no quantization
        rtn-2bit)     echo "02:30:00" ;;  # no GPTQ pass
        gptq-2bit)    echo "02:45:00" ;;
        sinq)         echo "02:30:00" ;;
        pb-llm)       echo "03:00:00" ;;
        doml)         echo "03:00:00" ;;  # 2-bit DOML — flagship method
        doml-binary)  echo "03:00:00" ;;  # 1-bit variant
        braq)         echo "03:00:00" ;;
        tesseraq)     echo "12:00:00" ;;  # 10 h quantization + 2 h evals
        lnq)          echo "06:00:00" ;;  # 4 h Fisher/refine + 2 h evals
        leanquant-nu) echo "02:45:00" ;;  # measured 8.7 min on A40, 3-4x slower on 1g.10gb H100 MIG; +2 h evals
        *)            echo "03:00:00" ;;
    esac
}

build_python_cmd() {
    # Prints the python command line for a given method.
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
            # Pure RTN at 2 bits = the '2bit' method with GPTQ disabled.
            echo "python3 -u run.py $MODEL $DATASET 2bit --disable_gptq --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        gptq-2bit)
            # Paper-faithful GPTQ-2bit (Frantar et al. 2023, Table 3 no-groupsize):
            # per-row global scale + single mask (partition=1). Historically this
            # column defaulted to partition=3 (DOML's structural split), which
            # Phase 14 identified as a DOML-uniform ablation, not paper GPTQ.
            # OPT-1.3B W4 validation: --partition 1 --global_scale matched paper
            # Table 3 within 0.5% (15.54 vs 15.47 PPL, 2026-04-22).
            echo "python3 -u run.py $MODEL $DATASET 2bit --partition 1 --global_scale --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        doml)
            # Flagship 2-bit method: K=4 Lloyd-Max + GPTQ + structural partition (~2.06–2.15 bpw).
            echo "python3 -u run.py $MODEL $DATASET doml --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        doml-binary)
            # 1-bit DOML variant (K=2, ~1.07 bpw).
            echo "python3 -u run.py $MODEL $DATASET doml_binary --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        braq)
            echo "python3 -u run.py $MODEL $DATASET braq --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        sinq)
            echo "python3 -u src/run_sinq.py $MODEL $DATASET --nbits $SINQ_NBITS --group_size $SINQ_GROUPSIZE --seed $SEED --device cuda:0 $common_evals"
            ;;
        tesseraq)
            # Paper-faithful TesseraQ (ICLR 2025): AWQ init + PAR + grad-clip
            # + bfloat16 autocast + batched forward. All paper-reported PPLs
            # use AWQ init (`load_transform: True`). `--use_awq_init` is the
            # default in `src/run_tesseraq.py` as of 2026-04-22; pass
            # `--no_awq_init` to reproduce the previous PAR-only ablation.
            echo "python3 -u src/run_tesseraq.py $MODEL $DATASET --bit $TESSERAQ_BIT --group_size $TESSERAQ_GROUPSIZE --iterations $TESSERAQ_ITERATIONS --batch_size $TESSERAQ_BATCH_SIZE --nsamples $TESSERAQ_NSAMPLES --seed $SEED --device cuda:0 $common_evals"
            ;;
        lnq)
            echo "python3 -u src/run_lnq.py $MODEL $DATASET --full_pipeline --no_propagate --calib_dataset $LNQ_CALIB --nsamples $LNQ_NSAMPLES --seqlen $LNQ_SEQLEN --num_groups $LNQ_NUM_GROUPS --nbits $LNQ_NBITS --seed $SEED --device cuda:0 $common_evals"
            ;;
        leanquant-nu)
            # Paper Algorithm 1: per-row weighted k-means over diag(Hinv)^(-p)
            # + block-wise GPTQ. act_order + true_sequential + propagate match
            # the README's 2-bit recipe; csv_utils writes method="leanquant_nu".
            echo "python3 -u src/run_leanquant.py $MODEL $DATASET --nbits $LEANQUANT_NBITS --exponent $LEANQUANT_EXPONENT --percdamp $LEANQUANT_PERCDAMP --true_sequential --act_order --calib_dataset $LEANQUANT_CALIB --nsamples $LEANQUANT_NSAMPLES --seqlen $LEANQUANT_SEQLEN --seed $SEED --device cuda:0 $common_evals"
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
        gptq-2bit)    echo "GPTQ @ 2-bit paper-faithful (--partition 1 --global_scale, per-row no-groupsize, blocksize=$BLOCKSIZE)" ;;
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
echo "Qwen3-0.6B PTQ Benchmark Suite (Nibi)"
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
        export BILLM_DOWNLOADS_DIR="$CACHE_ROOT_LOCAL/downloads"
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

# Project-level override read by datautils.py / run.py / PB-LLM / src/run_*.py
# — replaces every hardcoded ./downloads/... path with an absolute scratch path.
export BILLM_DOWNLOADS_DIR="\$CACHE_ROOT/downloads"

# HF_HUB_CACHE must equal BILLM_DOWNLOADS_DIR — see download_cache.sh comment
# for the rationale (AutoTokenizer.from_pretrained has no cache_dir override).
export HF_HOME="\$CACHE_ROOT/hf"
export HF_HUB_CACHE="\$BILLM_DOWNLOADS_DIR"
export HF_DATASETS_CACHE="\$CACHE_ROOT/hf/datasets"
# TRANSFORMERS_CACHE is deprecated since transformers 4.36; HF_HOME covers it.
export TORCH_HOME="\$CACHE_ROOT/torch"
# Offline XET resolution fails on compute nodes; disable so the snapshot
# under HF_HUB_CACHE is read directly.
export HF_HUB_DISABLE_XET=1
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
echo "Python:             \$(which python)"
python --version
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
# Sanity check: Qwen3 must be importable. If this fails the job aborts early
# instead of 40 min into a partial run.
python -c "from transformers.models.qwen3 import Qwen3ForCausalLM; import transformers; print('transformers', transformers.__version__)" || {
    echo "FATAL: Qwen3ForCausalLM not importable — transformers is too old (<4.51)."
    exit 1
}
# Verify model snapshot actually exists before going offline.
ls -d "\$BILLM_DOWNLOADS_DIR"/models--*Qwen3* 2>/dev/null || {
    echo "FATAL: no Qwen3 snapshot under \$BILLM_DOWNLOADS_DIR — run ./sbatch/download_cache.sh first."
    exit 1
}
# Verify refs/main was written. Without this file, \`AutoModelForCausalLM.from_pretrained\`
# under HF_HUB_OFFLINE=1 can raise LocalEntryNotFoundError for config.json
# (observed: job 12501760 pb-llm, 2026-04-20 13:08 — cache downloaded Apr 16,
# refs/main not written until Apr 20 14:05 when the snapshot_download
# local_files_only=True fix landed in download_cache.sh). Fail loud early
# rather than burn GPU time just to die at model load.
for _refs in "\$BILLM_DOWNLOADS_DIR"/models--*Qwen3*/refs/main; do
    [ -s "\$_refs" ] || {
        echo "FATAL: \$_refs missing or empty."
        echo "       Re-run ./sbatch/download_cache.sh (commit 6d7eb56 or later — must do"
        echo "       snapshot_download(..., local_files_only=True) as second pass)."
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
