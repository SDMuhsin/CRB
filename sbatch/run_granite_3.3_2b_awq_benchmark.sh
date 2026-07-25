#!/bin/bash
# ============================================================================
# ibm-granite/granite-3.3-2b-base — 2-bit PTQ benchmark: AWQ ARM: AWQ v1 scaling + DOML K31 chain (Nibi)
# ============================================================================
#
# ######################### AWQ ARM (2026-07-24 evening) #########################
# Same honest DOML K31 chain as the sister non-AWQ script, with ONE change: the
# build step applies the AWQ v1 activation-scaling transform BEFORE quantization
# (CRB_AWQ_ALPHA=0.5; kernels/pack/awq_transform.py: norm-fed q/k/v + gate/up
# columns scaled by s_j = a_j^0.5, geometric mean 1, inverse fold into the
# feeding RMSNorm weight -> exactly output-preserving on the FP model,
# bpw-neutral). Scales are saved to <dump>/awq_scales.safetensors and re-applied
# automatically by btune/atune/restore (they key on that file) — tune/eval
# stages need NO flag. Motivated by the Falcon3 root-cause finding (2026-07-24):
# the calib-measurable activation-outlier ratio predicts AWQ applicability, and
# this model never got scaling.
#
# COLLISION SAFETY vs the non-AWQ arm (both may run concurrently on Nibi):
#   - dump dir   awqfix05-…             (non-AWQ uses k31-…)   -> disjoint files
#   - CSV        results/<model>_awq_ptq_benchmark.csv          -> disjoint CSV
#   - job name   <model>_doml_awq                               -> disjoint logs
#   - separate SLURM allocation (same 3g.40gb class); the HF snapshot cache is
#     shared READ-ONLY by design (same as every existing job pair).
#   - NO tesseraq job here — TQ already runs in the non-AWQ sister script;
#     duplicating it would double-book the h100 and its CSV rows.
# OLMo2 deliberately has NO AWQ arm: Olmo2 is a norm-AFTER architecture (its
# decoder layer has NO input_layernorm feeding q/k/v — verified against the
# installed transformers; same reason src/run_tesseraq.py restricts its AWQ
# subsets there), so the v1 fold is mathematically inapplicable.
# Requires: checkout >= 7634dea ("Mo models", carries the committed AWQ
# machinery) + nibi_delta.patch (ships this script), or the 6675f49 +
# nibi_sync.patch route.
# ##############################################################################
#
# Sister script to run_llama3_1b_benchmark.sh / run_qwen_1.7b_benchmark.sh,
# extended to the new-model generalization arm (2026-07-24). Dispatches one
# sbatch job per method:
#
#   doml     — the full honest DOML K31 chain (the recipe of every rerun
#              DOML paper row; scripts/research/doml_pipeline.sh precedent):
#                doml_group_refit --run  (build: λ rd-split, g256 codebook
#                    groups, fp8-e4m3 codebooks, hdiag cb weighting,
#                    intra-block GPTQ, refit-iters 2, bulk-K 2)
#              → k31_block_tune          (stage-1 output-aware level tune)
#              → k31_assign_tune --mode pair  (stage-2 assignment tune;
#                    on failure the btuned container ships — tracker
#                    precedent, still a complete honest DOML row)
#              → k29_honest_bpw          (honest bpw of the shipped container)
#              → doml_group_refit --restore-dpk ... --eval-extra-ppl
#                    --full-eval         (wt2/c4/ptb + MMLU/HellaSwag/ARC)
#
# Model facts: Granite-3.3 2B base (GraniteForCausalLM, 40 layers, h=2048, ~2.5B params). Repo has NATIVE granite plumbing (run.py/eval_utils 'granite' branches predate this campaign). Substituted 2026-07-24 for kyutai/helium-1-2b, whose released checkpoint is unusable (FP16 baseline wt2 418 on dev box — model broken as shipped, not a harness/DOML issue). 40 layers => longer chain than helium; walltime sized accordingly.
#
# Dev-box measurements (2026-07-24, Blackwell MIG slices — used for sizing,
# never undercut):
#   - DOML build+tunes for the 1B/2B class fit a 24 GB slice at the batch/
#     stream-chunk knobs pinned below. 2g.20gb (20 GB) is SMALLER than that
#     proven envelope ⇒ DOML chain jobs use 3g.40gb.
#   - TesseraQ paper-exact peaked >23.6 GB GPU (OOMed a 24 GB slice; needs
#     ~27 GB) and >100 GB host RSS ⇒ full h100 + 160 G, the house tesseraq
#     precedent for this size class (4g.40gb is NOT provisioned on Nibi,
#     Gotcha #33 — the house 1.7B tesseraq profile is GPU_FULL + 160G).
#
# PREREQ: nibi_sync.patch applied (the campaign fixes in run.py /
# src/eval_utils.py / src/run_tesseraq.py and the kernels/pack fixes are
# uncommitted on the dev box — without them every job here crashes with
# "Unsupported model"), and ./sbatch/download_cache.sh re-run (fetches
# ibm-granite/granite-3.3-2b-base + the namespaced wikitext repo).
#
# Usage:
#   ./sbatch/run_granite_3.3_2b_awq_benchmark.sh                      # submit the job
#   ./sbatch/run_granite_3.3_2b_awq_benchmark.sh --account def-foo
#   ./sbatch/run_granite_3.3_2b_awq_benchmark.sh --local              # run serially, no SLURM
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

MODEL="ibm-granite/granite-3.3-2b-base"
MODEL_SHORT="granite33_2b"
MODEL_SUB="granite33-2b"                # downloads/doml_dumps/<sub>/ dump subdir
DATASET="wikitext2"
SEED=0

# HF cache directory uses the canonical models--<org>--<repo> layout.
MODEL_CACHE_DIR="models--ibm-granite--granite-3.3-2b-base"

CSV_NAME="${MODEL_SHORT}_awq_ptq_benchmark.csv"   # AWQ arm: NEVER the non-AWQ CSV
CSV_ABS="$(pwd)/results/$CSV_NAME"

# ----------------------------------------------------------------------------
# DOML λ (rd-split rate weight) — PROVISIONAL best-known rate-matched pick.
# 16e-5 is the dev-box rate-matched pick (honest bpw 2.1160; matches the paper Qwen3-1.7B/Llama-3.2-1B λ; 2026-07-24).
# Re-pick only if the honest-bpw log lands far off the ≤2.25-class target
# (tracker λ-audit rule). The dump dir name tracks this value.
# ----------------------------------------------------------------------------
DOML_LAMBDA="16e-5"
# Tune-stage memory knobs, measured on the dev box for THIS model (24 GB
# slice envelope). Do not raise on MIG slices without re-measuring.
DOML_BATCH=2
DOML_SCHUNK=2
# AWQ v1 exponent (uniform recipe; matches every dev-box awqfix05-* run).
AWQ_ALPHA="0.5"
DOML_DUMP_DIR="downloads/doml_dumps/${MODEL_SUB}/awqfix05-granite-lam${DOML_LAMBDA}-g256"

# Nibi GRES strings. Full long-form MIG names are REQUIRED — short forms
# like `1g.10gb` are rejected (Gotcha #11). 4g.40gb is NOT provisioned
# (Gotcha #33; sinfo confirms).
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"

# Methods to benchmark — one sbatch job per entry.
techniques=(
    "doml_awq"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_resources() {
    # Sets: gpu_resource, cpus, mem
    case $1 in
        doml_awq)
            # Full K31 chain. Dev-box proven envelope is a 24 GB slice at
            # batch $DOML_BATCH; 2g.20gb (20 GB) undercuts it ⇒ 3g.40gb.
            # Host RAM: 64G = 2× the house 32G DOML budget (container +
            # calib + restore copies).
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="64G"
            ;;
        *)
            gpu_resource="$GPU_MEDIUM"
            cpus=4
            mem="32G"
            ;;
    esac
}

get_time_limit() {
    # Alliance Nibi tiers: b1≤3h, b2≤12h, b3≤24h, b4≤72h, b5≤168h.
    # tesseraq: house precedent ×1 (28 h, b4) — do NOT shrink.
    # doml: dev-box chain wall + eval suite, ×2+ margin for the slower
    #       per-SM Nibi MIG slice; never below the 05:00:00 house doml floor.
    case $1 in
        doml_awq)  echo "14:00:00" ;;
        *)         echo "05:00:00" ;;
    esac
}

build_python_cmd() {
    # Prints the job payload. doml emits the full multi-line K31 chain
    # (idempotent stages, per-stage FATAL guards — requeued jobs resume);
    # tesseraq emits the house one-liner. Everything except EVAL_DIR is
    # resolved at submit time; \$EVAL_DIR survives to run at job time.
    local technique=$1
    local dd="$DOML_DUMP_DIR"

    # --full_eval = PPL on (wikitext2,c4,ptb) + MMLU + HellaSwag + ARC-Easy +
    # ARC-Challenge. Each task writes its own row to BILLM_BENCH_CSV.
    local common_evals="--full_eval"

    case $technique in
        doml_awq)
            cat <<CHAIN
mkdir -p downloads/doml_dumps/${MODEL_SUB}
echo "=== [0/5] preflight: safetensors torch.uint32 round-trip (dpk container dtype) ==="
python3 -c "import os, tempfile, torch, safetensors; from safetensors.torch import save_file, load_file; p = os.path.join(tempfile.gettempdir(), 'st_u32_probe.safetensors'); save_file({'t': torch.zeros(8, dtype=torch.int32).view(torch.uint32)}, p); load_file(p); os.remove(p); print('preflight OK: safetensors', safetensors.__version__, 'at', safetensors.__file__)" || { echo "FATAL: safetensors cannot serialize torch.uint32 — dpk containers need safetensors>=0.6.1 inside ./env (a stale ~/.local copy may be shadowing it). Fix on the login node: ./env/bin/pip install -U 'safetensors>=0.6.1'"; exit 1; }
echo "=== [1/5] AWQ v1 (alpha=${AWQ_ALPHA}) + DOML K31 build (lam=${DOML_LAMBDA}, g256, fp8 cb, hdiag, intra-block GPTQ, refit2, bulk-K2, rd-split) ==="
if [ ! -f "${dd}/manifest.json" ]; then
    CRB_AWQ_ALPHA=${AWQ_ALPHA} python3 -u kernels/pack/doml_group_refit.py --run --model $MODEL --g 256 --dump-dir "${dd}" --codebook-dtype float8_e4m3fn --cb-weight hdiag --intra-block-gptq --refit-iters 2 --bulk-k 2 --rd-split $DOML_LAMBDA || { echo "FATAL: doml awq build failed"; exit 1; }
else
    echo "build manifest exists — skipping (idempotent resume)"
fi
[ -f "${dd}/awq_scales.safetensors" ] || { echo "FATAL: ${dd}/awq_scales.safetensors missing — this dump is NOT an AWQ build (CRB_AWQ_ALPHA lost, or a stale plain dump squatting on the awqfix tag). Refusing to continue: tuning it would produce a mislabeled row."; exit 1; }
echo "=== [2/5] stage-1 block tune (batch $DOML_BATCH / stream-chunk $DOML_SCHUNK) ==="
if [ ! -f "${dd}-btuned/manifest.json" ]; then
    python3 -u kernels/pack/k31_block_tune.py --src "${dd}" --batch $DOML_BATCH --stream-chunk $DOML_SCHUNK || { echo "FATAL: k31_block_tune failed"; exit 1; }
else
    echo "btuned manifest exists — skipping (idempotent resume)"
fi
echo "=== [3/5] stage-2 assignment tune (pair mode) ==="
EVAL_DIR="${dd}-btuned"
if [ ! -f "${dd}-atuned/manifest.json" ]; then
    python3 -u kernels/pack/k31_assign_tune.py --src "${dd}-btuned" --mode pair --batch $DOML_BATCH --stream-chunk $DOML_SCHUNK || echo "WARN: k31_assign_tune failed — shipping the btuned container (tracker precedent: btuned is a complete honest DOML row)"
else
    echo "atuned manifest exists — skipping (idempotent resume)"
fi
if [ -f "${dd}-atuned/manifest.json" ] && ls "${dd}-atuned"/*.dpk.safetensors >/dev/null 2>&1; then
    EVAL_DIR="${dd}-atuned"
fi
echo "=== [4/5] honest bpw (k29, on \$EVAL_DIR) ==="
python3 -u kernels/pack/k29_honest_bpw.py --dir "\$EVAL_DIR" || { echo "FATAL: k29_honest_bpw failed"; exit 1; }
echo "=== [5/5] restore + full eval (\$EVAL_DIR) ==="
python3 -u kernels/pack/doml_group_refit.py --run --restore-dpk "\$EVAL_DIR" --eval-extra-ppl --full-eval || { echo "FATAL: restore eval failed"; exit 1; }
CHAIN
            ;;
        *)
            echo "echo 'Unknown technique: $technique'; exit 1"
            ;;
    esac
}

get_technique_desc() {
    case $1 in
        doml_awq)  echo "AWQ v1 (alpha=$AWQ_ALPHA, norm-fed qkv+gate/up, bpw-neutral) + DOML K31 chain (lam=$DOML_LAMBDA g256 fp8-cb hdiag intra-gptq refit2 bulkK2 rd-split -> btune -> atune pair -> honest bpw -> full eval)" ;;
    esac
}

# ============================================================================
# MAIN LOOP
# ============================================================================

job_count=0
mkdir -p ./logs ./results

echo "============================================"
echo "$MODEL PTQ Benchmark Suite (Nibi)"
echo "============================================"
echo "Model:       $MODEL"
echo "Dataset:     $DATASET"
echo "Seed:        $SEED"
echo "DOML lam:    $DOML_LAMBDA (provisional rate-matched pick)"
echo "AWQ alpha:   $AWQ_ALPHA (v1 scope: norm-fed qkv + gate/up)"
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
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
# Note: the venv legitimately borrows idna / certifi / yaml / tqdm /
# accelerate / typing_extensions from \$HOME/.local/. Do NOT set
# PYTHONNOUSERSITE=1 here. Exception: safetensors must live IN ./env at
# >=0.6.1 — the ~/.local copy predates torch.uint32 and KeyError'd every
# dpk save (jobs 18481807/09/11); the [0/5] preflight guards this.

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

# kernels/pack tools resolve the repo root from CRB_REPO (rerun-campaign fix).
export CRB_REPO="\$(pwd)"

export PYTHONPATH=\$PYTHONPATH:\$(pwd):\$(pwd)/src

# Tune stages / TesseraQ on MIG slices need expandable segments (tracker
# lesson 2026-07-22; same fix as LNQ Fisher).
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
# Sanity check: aborts before GPU time is spent if the venv is broken or
# too old for this model family.
python -c "from transformers import GraniteForCausalLM; import transformers; print('transformers', transformers.__version__)" || {
    echo "FATAL: GraniteForCausalLM not importable — transformers too old for Granite-3.3."
    exit 1
}
python -c "from kernels.pack.awq_transform import collect_awq_scales, apply_awq_" || {
    echo "FATAL: kernels/pack/awq_transform.py not importable — checkout must carry the committed AWQ machinery (>= 7634dea) with the patch applied."
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
