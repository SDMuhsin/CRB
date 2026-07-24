#!/bin/bash
# ============================================================================
# SDOML sparsity sweep — Qwen3-1.7B + Llama-3.2-1B (Nibi / Alliance Canada)
# ============================================================================
#
# Fills every missing SDOML cell of the nips26 tab:sparse / tab:bpw_sparse
# tables (rerun mandate, llmdocs/tracker/RERUN_2026-07-22_paper_doml_sdoml.md
# §D): s ∈ {5,20,25,40,50,75} on BOTH models, plus SDOML+partition asymmetric
# s ∈ {20,50} on both models. One sbatch job per (model, config); 16 jobs.
#
# Two payload kinds:
#
#   sdoml-sNN  — the FULL honest-container chain (the recipe that produced
#     the verified 2.25-bpw frontier, llmdocs/cuda_kernel/SDOML_honest_frontier.md
#     §5/§8, smoke-tested on the dev box 2026-07-23, sdoml_17b_s{0,5} logs):
#         sdoml_dump --run             (BASE container, g128, bf16 cb)
#       → sdoml_honest_bpw            (base reference bpw)
#       → sdoml_block_tune            (stage-1 levels; casts cb to fp8-e4m3,
#                                      -0.25 bpw; --batch 4 --stream-chunk 4)
#       → sdoml_assign_tune --mode pair  (stage-2 assignments, zero bit cost;
#                                      --batch 2 --stream-chunk 2 — the knobs
#                                      that survived the Llama-3.2-1B width;
#                                      on failure the btuned container ships,
#                                      tracker precedent = still a complete
#                                      honest SDOML row)
#       → sdoml_honest_bpw            (FINAL honest bpw → tab:bpw_sparse)
#       → sdoml_restore_eval --eval-extra-ppl --full-eval
#                                      (wt2/c4/ptb PPL + MMLU/HellaSwag/
#                                       ARC-E/ARC-C → tab:sparse)
#     NOTE — no λ / no --rd-split / no g256 in this chain: SDOML has no
#     rate-distortion split stage (λ is a DOML-build knob), and the g256
#     codebook refit was measured a DEAD END for SDOML (frontier doc §9:
#     +4.2 wt2 / +18 c4 / +47 ptb for -0.125 bpw). The paper recipe is
#     g128 + fp8-cb (via block-tune) + assignment-tune, exactly as below.
#
#   sdoml_part_asym-sNN — SDOML+partition asymmetric (S9), via the SAME
#     run.py command the May cells were produced with (house precedent in
#     run_qwen_1.7b_benchmark.sh / run_llama3_1b_benchmark.sh):
#         run.py <model> wikitext2 sdoml_partition --sparsity s
#                --sdoml_asymmetric ... --full_eval
#     There is NO container/honest-bpw tooling for the asymmetric arm (the
#     frontier doc dead-ended it as an iterable vehicle); its bpw stays the
#     analytic eq:bpw_asym value. Flagged in llmdocs/NIBI_SUBMISSION.md.
#
# Each job writes to its OWN CSV (results/sdoml_sweep_<model>_<config>.csv)
# so the rerun campaign's cells never mix with the April *_ptq_benchmark.csv
# rows. Dump dirs are per-(model, s) — collision-free by construction.
#
# Chain stages are idempotent (skip on existing manifest.json), so a
# requeued job resumes past finished stages.
#
# PREREQ: nibi_sync.patch applied (kernels/pack tools + run.py fixes are
# uncommitted on the dev box; without them sdoml_restore_eval has no
# --full-eval and the tools carry a hardcoded /workspace/BiLLM2 path).
#
# Usage:
#   ./sbatch/run_sdoml_sweep.sh                    # submit all 16
#   ./sbatch/run_sdoml_sweep.sh --account def-foo  # with account
#   ./sbatch/run_sdoml_sweep.sh --local            # run serially (smoke)
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

DATASET="wikitext2"
SEED=0
BLOCKSIZE=128
SALIENT_METRIC="magnitude"

# Models: HF name | short (CSV/job names) | dump-dir sub | HF cache dir |
#         asym walltime (house precedent from the per-model benchmark script)
MODELS=(
    "Qwen/Qwen3-1.7B|qwen3_1.7b|qwen3-1.7b|models--Qwen--Qwen3-1.7B|36:00:00"
    "meta-llama/Llama-3.2-1B|llama3_1b|llama3.2-1b|models--meta-llama--Llama-3.2-1B|26:00:00"
)

# Nibi GRES strings. Full long-form MIG names are REQUIRED — short forms
# like `1g.10gb` are rejected (Gotcha #11). 4g.40gb is NOT provisioned
# (Gotcha #33; sinfo confirms — jobs requesting it die silently).
GPU_SMALL="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_MEDIUM="--gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_LARGE="--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
GPU_FULL="--gres=gpu:h100:1"

# One job per (model, config). tab:sparse missing cells (tracker §D).
techniques=(
    # SDOML — base K=4 single-codebook (partition=1), full honest chain.
    "sdoml-s5"
    "sdoml-s20"
    "sdoml-s25"
    "sdoml-s40"
    "sdoml-s50"
    "sdoml-s75"
    # SDOML — asymmetric partition (S9 Pareto extension), bulk-only mask.
    "sdoml_part_asym-s20"
    "sdoml_part_asym-s50"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_resources() {
    # Sets: gpu_resource, cpus, mem
    #
    # sdoml-s* (chain): the tune stages are the driver, not the base dump.
    #   Dev-box measurements (tracker, 2026-07-22/23): assignment-tune used
    #   29–35 GB on the old A40s at batch 8; on 24 GB slices 1.7B needed
    #   batch4 and Llama-3.2-1B OOM'd even at batch4 (wider MLP, 8192) and
    #   needed batch2. 2g.20gb (20 GB) is BELOW the only proven envelope, so
    #   the smallest known-good Nibi profile is 3g.40gb — no smaller slice
    #   has ever run these tune stages. Host RAM: model + containers +
    #   activation streams are GPU-side; 64G doubles the 32G house DOML
    #   budget for the extra per-stage CPU copies.
    #
    # sdoml_part_asym-s*: house precedent VERBATIM (run_qwen_1.7b_benchmark.sh
    #   / run_llama3_1b_benchmark.sh sdoml_part_asym-s* case — 2g.20gb, 4 cpus,
    #   32G; these exact profiles produced the May asym rows).
    case $1 in
        sdoml-s*)
            gpu_resource="$GPU_LARGE"
            cpus=8
            mem="64G"
            ;;
        sdoml_part_asym-s*)
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
    # Alliance Nibi partition tiers: b1≤3h, b2≤12h, b3≤24h, b4≤72h, b5≤168h.
    # Walltimes nudged off boundaries so SLURM routes unambiguously.
    #
    # sdoml-s* chain: dev-box wall (Blackwell 24 GB slice, 1.7B smoke) was
    #   dump ~25 min + btune 26 min + atune ~40 min + bpw ~min + eval suite.
    #   Nibi 3g.40gb is slower per-SM; with the house 60–90 min eval-suite
    #   budget and 2× compute margin the chain lands ≈5–6 h ⇒ 10:00:00 (b2).
    #   Do NOT shrink below the 05:00:00 single-stage sdoml-s* precedent.
    # sdoml_part_asym-s*: house precedent ×1 (per-row loop fallback dominates;
    #   36 h on Qwen3-1.7B, 26 h on Llama-3.2-1B — passed per model below).
    case $1 in
        sdoml-s*)            echo "10:00:00" ;;  # b2 — full chain + eval suite
        sdoml_part_asym-s*)  echo "$MODEL_ASYM_TIME" ;;  # house precedent ×1
        *)                   echo "10:00:00" ;;
    esac
}

build_python_cmd() {
    # Prints the job payload for a given config. sdoml-s* emits the full
    # multi-line chain (idempotent stages, per-stage FATAL guards, mirroring
    # the dev-box sdoml_one.sh driver); sdoml_part_asym-s* emits the house
    # one-liner. Everything except EVAL_DIR is resolved at submit time;
    # \$EVAL_DIR survives to run at job time.
    local technique=$1
    local s_pct s_frac dd

    # --full_eval = PPL on (wikitext2,c4,ptb) + MMLU + HellaSwag + ARC-Easy +
    # ARC-Challenge. Each task writes its own row to BILLM_BENCH_CSV.
    local common_evals="--full_eval"

    case $technique in
        sdoml-s*)
            s_pct=${technique#sdoml-s}
            if [[ "$s_pct" -lt 10 ]]; then s_frac="0.0${s_pct}"; else s_frac="0.${s_pct}"; fi
            dd="downloads/doml_dumps/${MODEL_SUB}/sdoml-s${s_pct}"
            cat <<CHAIN
mkdir -p downloads/doml_dumps/${MODEL_SUB}
echo "=== [1/6] SDOML base dump -> ${dd} ==="
if [ ! -f "${dd}/manifest.json" ]; then
    python3 -u kernels/pack/sdoml_dump.py --run --model $MODEL --sparsity $s_frac --dump-dir "${dd}" || { echo "FATAL: sdoml_dump failed"; exit 1; }
else
    echo "dump manifest exists — skipping (idempotent resume)"
fi
echo "=== [2/6] honest bpw (base, bf16 cb) ==="
python3 -u kernels/pack/sdoml_honest_bpw.py --dir "${dd}" || { echo "FATAL: base honest_bpw failed"; exit 1; }
echo "=== [3/6] stage-1 block tune (levels -> fp8-e4m3 cb) ==="
if [ ! -f "${dd}-btuned/manifest.json" ]; then
    python3 -u kernels/pack/sdoml_block_tune.py --src "${dd}" --batch 4 --stream-chunk 4 || { echo "FATAL: sdoml_block_tune failed"; exit 1; }
else
    echo "btuned manifest exists — skipping (idempotent resume)"
fi
echo "=== [4/6] stage-2 assignment tune (pair mode, zero bit cost) ==="
EVAL_DIR="${dd}-btuned"
if [ ! -f "${dd}-atuned/manifest.json" ]; then
    python3 -u kernels/pack/sdoml_assign_tune.py --src "${dd}-btuned" --mode pair --batch 2 --stream-chunk 2 || echo "WARN: sdoml_assign_tune failed — shipping the btuned container (tracker precedent: btuned is a complete honest SDOML row)"
else
    echo "atuned manifest exists — skipping (idempotent resume)"
fi
if [ -f "${dd}-atuned/manifest.json" ] && ls "${dd}-atuned"/*.sdpk.safetensors >/dev/null 2>&1; then
    EVAL_DIR="${dd}-atuned"
fi
echo "=== [5/6] honest bpw (FINAL: \$EVAL_DIR) — the tab:bpw_sparse number ==="
python3 -u kernels/pack/sdoml_honest_bpw.py --dir "\$EVAL_DIR" || { echo "FATAL: final honest_bpw failed"; exit 1; }
echo "=== [6/6] restore + full eval (\$EVAL_DIR) ==="
python3 -u kernels/pack/sdoml_restore_eval.py --dir "\$EVAL_DIR" --eval-extra-ppl --full-eval || { echo "FATAL: sdoml_restore_eval failed"; exit 1; }
CHAIN
            ;;
        sdoml_part_asym-s*)
            # House command verbatim (run_qwen_1.7b_benchmark.sh /
            # run_llama3_1b_benchmark.sh sdoml_part_asym-s* case).
            s_pct=${technique#sdoml_part_asym-s}
            if [[ "$s_pct" -lt 10 ]]; then s_frac="0.0${s_pct}"; else s_frac="0.${s_pct}"; fi
            echo "python3 -u run.py $MODEL $DATASET sdoml_partition --sparsity $s_frac --sdoml_asymmetric --blocksize $BLOCKSIZE --salient_metric $SALIENT_METRIC --seed $SEED --device cuda:0 $common_evals"
            ;;
        *)
            echo "echo 'Unknown technique: $technique'; exit 1"
            ;;
    esac
}

get_technique_desc() {
    case $1 in
        sdoml-s*)            echo "SDOML honest chain (dump -> btune fp8 -> atune pair -> honest bpw -> full eval), s pattern" ;;
        sdoml_part_asym-s*)  echo "SDOML+partition asymmetric (s_bulk pattern, mid+salient dense, S9) via run.py, full eval" ;;
    esac
}

# ============================================================================
# MAIN LOOP
# ============================================================================

job_count=0
mkdir -p ./logs ./results

echo "============================================"
echo "SDOML Sparsity Sweep (Nibi) — tab:sparse / tab:bpw_sparse"
echo "============================================"
echo "Models:      Qwen/Qwen3-1.7B + meta-llama/Llama-3.2-1B"
echo "Dataset:     $DATASET (calibration + wt2 PPL; c4/ptb + downstream via full eval)"
echo "Seed:        $SEED"
echo "Techniques:  ${techniques[*]}"
echo "CSVs:        ./results/sdoml_sweep_<model>_<config>.csv (one per job)"
echo "Logs:        ./logs/"
echo "============================================"
echo ""

for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_SHORT MODEL_SUB MODEL_CACHE_DIR MODEL_ASYM_TIME <<< "$model_spec"

    # Import sanity check per family (house pattern): aborts before GPU time
    # is spent if the venv is broken.
    if [[ "$MODEL" == Qwen/* ]]; then
        import_check="from transformers.models.qwen3 import Qwen3ForCausalLM; import transformers; print('transformers', transformers.__version__)"
        import_fatal="Qwen3ForCausalLM not importable — transformers is too old (<4.51)."
    else
        import_check="from transformers import LlamaForCausalLM; import transformers; print('transformers', transformers.__version__)"
        import_fatal="LlamaForCausalLM not importable — check transformers install."
    fi

    for technique in "${techniques[@]}"; do
        technique_desc=$(get_technique_desc "$technique")
        time_limit=$(get_time_limit "$technique")
        get_job_resources "$technique"

        job_name="${MODEL_SHORT}_${technique}"
        # Per-job CSV — never collides with the April *_ptq_benchmark.csv.
        CSV_ABS="$(pwd)/results/sdoml_sweep_${MODEL_SHORT}_${technique}.csv"
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
                # already holds the models. Don't append /downloads again.
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
# Note: the venv legitimately borrows idna / certifi / safetensors / yaml /
# tqdm / accelerate / typing_extensions from \$HOME/.local/. Do NOT set
# PYTHONNOUSERSITE=1 here.

# Route every cache to \$SCRATCH (1 TB soft / 20 TB hard on Nibi).
if [[ -n "\${SCRATCH:-}" && -d "\${SCRATCH}" ]]; then
    CACHE_ROOT="\$SCRATCH/billm2_cache"
else
    CACHE_ROOT="\$(pwd)/downloads"
fi

# Re-create ./downloads -> \$CACHE_ROOT symlink defensively (datautils.py and
# the kernels/pack tools use the relative './downloads' path for dump dirs).
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

# kernels/pack tools resolve the repo root from CRB_REPO (rerun-campaign fix;
# falls back to 3× dirname of the tool file — set it explicitly anyway).
export CRB_REPO="\$(pwd)"

export PYTHONPATH=\$PYTHONPATH:\$(pwd):\$(pwd)/src

# Tune stages on MIG slices need expandable segments (tracker lesson 2026-07-22:
# prevents fragmentation-OOM in assignment tuning; same fix as LNQ Fisher).
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
python -c "$import_check" || {
    echo "FATAL: $import_fatal"
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
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results CSVs:         ./results/sdoml_sweep_*.csv (one per job)"
echo "Logs directory:       ./logs/"
echo "============================================"
