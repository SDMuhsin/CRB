#!/bin/bash
# ============================================================================
# Pre-download all models and datasets to local cache (Nibi / Alliance Canada)
# ============================================================================
#
# Run this on a LOGIN NODE (with internet) BEFORE submitting sbatch jobs.
# Compute nodes have no internet access.
#
# QUOTA NOTE — Alliance Canada (Nibi)
# -----------------------------------
#   /project/<def-xxx>  : tiny quota, code only (NOT for caches/datasets).
#   $SCRATCH            : 1 TB soft / 20 TB hard, 60-day grace — caches go here.
#   $SLURM_TMPDIR       : per-job local NVMe, wiped at job end — no good for
#                         cross-job cache, but fine for transient unpacking.
#
# This script (a) auto-detects $SCRATCH and routes ALL HF / Torch / project
# caches to $SCRATCH/billm2_cache, and (b) replaces ./downloads with a
# symlink to that scratch dir so the project's hardcoded `./downloads/...`
# paths (datautils.py, PB-LLM/gptq_pb/run.py) write to scratch automatically.
#
# Usage:
#   ./sbatch/download_cache.sh
#
# ============================================================================

set -euo pipefail

source ./env/bin/activate
# Note: the venv legitimately borrows idna / certifi / safetensors / yaml /
# tqdm / accelerate / typing_extensions from $HOME/.local/. Do NOT set
# PYTHONNOUSERSITE=1 here. The torchvision ABI crash is fixed separately
# by uninstalling torchvision/torchaudio from ~/.local (see
# ./sbatch/fix_venv_torchvision.sh).

# ---------------------------------------------------------------------------
# 1. Pick CACHE_ROOT — $SCRATCH on Alliance, ./downloads elsewhere.
# ---------------------------------------------------------------------------
if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]]; then
    CACHE_ROOT="$SCRATCH/billm2_cache"
    echo "Detected Alliance Canada — caching under \$SCRATCH"
else
    CACHE_ROOT="$(pwd)/downloads"
    echo "No \$SCRATCH found — caching under project ./downloads"
fi
mkdir -p "$CACHE_ROOT"

# ---------------------------------------------------------------------------
# 2. Ensure ./downloads → CACHE_ROOT symlink (so hardcoded relative paths
#    in datautils.py and PB-LLM/gptq_pb/run.py land on scratch).
# ---------------------------------------------------------------------------
if [[ "$CACHE_ROOT" != "$(pwd)/downloads" ]]; then
    if [[ -L ./downloads ]]; then
        # Replace stale symlink (idempotent re-runs).
        rm ./downloads
    elif [[ -e ./downloads ]]; then
        echo ""
        echo "ERROR: ./downloads exists as a real directory, not a symlink."
        echo "       Move it to scratch and re-run, e.g.:"
        echo "         mv ./downloads $CACHE_ROOT/downloads_legacy"
        echo "         ./sbatch/download_cache.sh"
        exit 1
    fi
    ln -sfn "$CACHE_ROOT" ./downloads
    echo "Symlinked ./downloads -> $CACHE_ROOT"
fi

# ---------------------------------------------------------------------------
# 3. Point every HuggingFace / Torch cache env var at CACHE_ROOT.
#    HF_DATASETS_CACHE is the key one — that's where arrow shards land.
# ---------------------------------------------------------------------------
# Project-level override read by datautils.py / run.py / PB-LLM / src/run_*.py.
# This is the explicit, non-symlink-based way to redirect every hardcoded
# ./downloads/... path the project uses.
export BILLM_DOWNLOADS_DIR="$CACHE_ROOT/downloads"

# CRITICAL — HF_HUB_CACHE must equal BILLM_DOWNLOADS_DIR. The runners call
# AutoModelForCausalLM.from_pretrained(model, cache_dir=downloads_dir, ...)
# but datautils.get_tokenizer calls AutoTokenizer.from_pretrained(model) with
# NO cache_dir, which falls back to HF_HUB_CACHE. If those two are different
# directories, the offline compute job finds the model but not the tokenizer
# (or vice-versa). Aligning them here means snapshot_download, AutoModel,
# and AutoTokenizer all read/write the same models--*/ tree.
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$BILLM_DOWNLOADS_DIR"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
# TRANSFORMERS_CACHE is deprecated since transformers 4.36; HF_HOME covers it.
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_DISABLE_XET=1
# Anything that defaults to $HOME/.cache/* must be redirected too — $HOME on
# Nibi has only ~50 GiB and goes over-quota fast (numba JIT, pip wheels, etc.).
export NUMBA_CACHE_DIR="$CACHE_ROOT/.cache/numba"
export PIP_CACHE_DIR="$CACHE_ROOT/.cache/pip"
export XDG_CACHE_HOME="$CACHE_ROOT/.cache"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TORCH_HOME" \
         "$NUMBA_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" \
         "$BILLM_DOWNLOADS_DIR"

echo ""
echo "Cache layout (resolved real paths — these are what writes actually hit):"
echo "  CACHE_ROOT             = $CACHE_ROOT"
echo "  BILLM_DOWNLOADS_DIR    = $BILLM_DOWNLOADS_DIR"
echo "  ./downloads            -> $(readlink -f ./downloads 2>/dev/null || echo MISSING)"
echo "  HF_HOME                -> $(readlink -f "$HF_HOME" 2>/dev/null || echo "$HF_HOME (not yet created)")"
echo "  HF_DATASETS_CACHE      -> $(readlink -f "$HF_DATASETS_CACHE" 2>/dev/null || echo "$HF_DATASETS_CACHE (not yet created)")"
echo "  HF_HUB_CACHE           -> $(readlink -f "$HF_HUB_CACHE" 2>/dev/null || echo "$HF_HUB_CACHE (not yet created)")"
echo "  TORCH_HOME             -> $(readlink -f "$TORCH_HOME" 2>/dev/null || echo "$TORCH_HOME (not yet created)")"
echo "  NUMBA_CACHE_DIR        -> $(readlink -f "$NUMBA_CACHE_DIR" 2>/dev/null || echo "$NUMBA_CACHE_DIR (not yet created)")"
echo ""

# Hard guard: if ./downloads ends up resolving anywhere under /project or
# /home, abort — that would silently fill the wrong quota.
DL_REAL=$(readlink -f ./downloads 2>/dev/null || echo "")
case "$DL_REAL" in
    /project/*|/home/*|"$HOME"/*)
        echo "ABORT: ./downloads resolves to '$DL_REAL', which is on /project or \$HOME."
        echo "       Those filesystems have small quotas. Set \$SCRATCH or fix the symlink."
        exit 1
        ;;
esac
unset DL_REAL

# ============================================================================
# Models — Qwen3-0.6B is the target for this benchmark suite
# ============================================================================
#
# IMPORTANT: write the model under \$BILLM_DOWNLOADS_DIR (cache_dir argument),
# NOT the default HF_HUB_CACHE. The project's runners call
# from_pretrained(..., cache_dir=downloads_dir) so they read from there at
# job time. snapshot_download must use the SAME directory or the offline
# compute job will fail to find the model under HF_HUB_OFFLINE=1.
# ============================================================================

echo "=== Downloading models (snapshot to BILLM_DOWNLOADS_DIR) ==="
echo "    target: $BILLM_DOWNLOADS_DIR"

python -c "
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

cache_dir = os.environ['BILLM_DOWNLOADS_DIR']
models = [
    'Qwen/Qwen3-0.6B',
]

for model_name in models:
    print(f'Downloading {model_name} -> {cache_dir} ...')
    path = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    print(f'  Snapshot: {path}')
    # Touch the tokenizer + config so the transformers cache index is warmed
    # under the SAME cache_dir the runners will read from offline.
    AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f'  Tokenizer + config ready.')

# Abort here if this transformers is too old for Qwen3 — catches wheelhouse
# wheels that lack Qwen3ForCausalLM before any sbatch job is submitted.
try:
    from transformers.models.qwen3 import Qwen3ForCausalLM  # noqa: F401
    import transformers
    print(f'Qwen3 import OK (transformers {transformers.__version__}).')
except Exception as exc:
    raise SystemExit(
        f'FATAL: Qwen3ForCausalLM not importable ({exc!r}). '
        'Install transformers>=4.51 from PyPI (see requirements.txt header).'
    )
"

echo ""

# ============================================================================
# Calibration / evaluation datasets
# ============================================================================
#
# - wikitext-2-raw-v1: PPL eval + default calibration for run.py.
# - C4 ONE SHARD: matches datautils.get_redpajama(), which loads only
#   en/c4-train.00000-of-01024.json.gz (~150 MB). DO NOT call
#   load_dataset('allenai/c4', 'en') without `data_files=` — it triggers
#   the full 365M-row train split build (>700 GB on disk) and will blow
#   through scratch quota.
# ============================================================================

echo "=== Downloading datasets ==="

python -c "
from datasets import load_dataset

print('Downloading wikitext (wikitext-2-raw-v1)...')
load_dataset('wikitext', 'wikitext-2-raw-v1')
print('  Done: wikitext-2')

print('Downloading allenai/c4 — single shard only (matches get_redpajama)...')
ds = load_dataset(
    'allenai/c4',
    data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
    split='train',
)
print(f'  Done: allenai/c4 (1 shard, {len(ds):,} rows)')
"

echo ""

# ============================================================================
# NOTE: pre-tokenization is intentionally NOT done here.
# ============================================================================
#
# Earlier versions tokenized the full redpajama (1024 × 4096) calibration set
# on the login node so the first benchmark job wouldn't pay the cost. That
# step joins ~83K C4 documents into a single string and tokenizes to a
# 2.5M-token tensor — it OOM-killed (login-node cgroup ~16 GB).
#
# Race-safety on the cache file is now handled by an fcntl flock added to
# datautils.get_loaders (DOWNLOAD_*.pt.lock). Concurrent jobs serialize:
# the first one tokenizes, the rest block on the lock then load the cache.
# So the only cost we pay is one extra tokenization on the FIRST sbatch
# job — which has the right memory budget for it (16 GB, see GPU_SMALL job
# spec in run_qwen_benchmark.sh).
#
# If you really want to pre-tokenize (e.g., to keep first-job wall time
# down), submit it as its own tiny sbatch job — don't run it on the login
# node:
#
#   sbatch --time=00:30:00 --mem=16G --cpus-per-task=4 --wrap='
#     source ./env/bin/activate
#     export BILLM_DOWNLOADS_DIR=$SCRATCH/billm2_cache/downloads
#     export HF_HOME=$SCRATCH/billm2_cache/hf
#     python -c "
#     import sys; sys.path.insert(0, \".\")
#     from datautils import get_loaders
#     get_loaders(\"wikitext2\", nsamples=128, seed=0, seqlen=2048,
#                 model=\"Qwen/Qwen3-0.6B\")
#     get_loaders(\"redpajama\", nsamples=1024, seed=0, seqlen=4096,
#                 model=\"Qwen/Qwen3-0.6B\")"
#   '
# ============================================================================

echo "=== Skipping pre-tokenization on login node (would OOM-kill). ==="
echo "    The first sbatch job will tokenize and write the cache;"
echo "    subsequent jobs block on the .lock then load it."
echo ""
echo "============================================"
echo "All downloads complete."
echo "Cache root:                       $CACHE_ROOT"
echo "Total cache size on \$SCRATCH:     $(du -sh "$CACHE_ROOT" 2>/dev/null | cut -f1)"
# -x stops at filesystem boundary, so this only counts bytes that LIVE on
# /project (i.e., excludes anything reached via the symlink). It must be ~0.
echo "Bytes that leaked to /project:    $(du -shx "$(pwd)/downloads" 2>/dev/null | cut -f1) (should be tiny — just the symlink)"
echo "Next step:                        sbatch ./sbatch/run_qwen_benchmark.sh"
echo "============================================"
