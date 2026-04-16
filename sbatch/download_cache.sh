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
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_DISABLE_XET=1
# Anything that defaults to $HOME/.cache/* must be redirected too — $HOME on
# Nibi has only ~50 GiB and goes over-quota fast (numba JIT, pip wheels, etc.).
export NUMBA_CACHE_DIR="$CACHE_ROOT/.cache/numba"
export PIP_CACHE_DIR="$CACHE_ROOT/.cache/pip"
export XDG_CACHE_HOME="$CACHE_ROOT/.cache"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TORCH_HOME" \
         "$NUMBA_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

echo ""
echo "Cache layout:"
echo "  CACHE_ROOT          = $CACHE_ROOT"
echo "  HF_HOME             = $HF_HOME"
echo "  HF_DATASETS_CACHE   = $HF_DATASETS_CACHE"
echo "  HF_HUB_CACHE        = $HF_HUB_CACHE"
echo "  ./downloads         -> $(readlink -f ./downloads)"
echo ""

# ============================================================================
# Models — Qwen2.5-0.5B is the target for this benchmark suite
# ============================================================================

echo "=== Downloading models (snapshot only, no loading) ==="

python -c "
from huggingface_hub import snapshot_download

models = [
    'Qwen/Qwen2.5-0.5B',
]

for model_name in models:
    print(f'Downloading {model_name}...')
    snapshot_download(repo_id=model_name)
    print(f'  Done: {model_name}')
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
# Pre-tokenize calibration sets to avoid first-job cache miss races
# ============================================================================
#
# datautils.get_loaders() caches tokenized samples at
# ./downloads/DOWNLOAD_<name>_<nsamples>_<seed>_<seqlen>_<model>.pt
# (now resolves through the symlink to $SCRATCH/billm2_cache/).
# Pre-warming on the login node prevents two compute jobs racing on the
# same write target (and the resulting file corruption).
# ============================================================================

echo "=== Pre-tokenizing calibration caches ==="

MODEL="Qwen/Qwen2.5-0.5B"

python -c "
import sys, os
sys.path.insert(0, os.getcwd())
from datautils import get_loaders

model = '$MODEL'

# wikitext2: 128 samples × 2048 — used by run.py default + SINQ + TesseraQ + PB-LLM
print(f'Tokenizing wikitext2/128/2048 for {model}...')
get_loaders('wikitext2', nsamples=128, seed=0, model=model, seqlen=2048)

# redpajama (one C4 shard): 1024 samples × 4096 — used by run_lnq.py faithful preset
print(f'Tokenizing redpajama/1024/4096 for {model}...')
get_loaders('redpajama', nsamples=1024, seed=0, model=model, seqlen=4096)

print('Calibration caches written under ./downloads/ (-> \$SCRATCH).')
"

echo ""
echo "============================================"
echo "All downloads complete."
echo "Cache root:   $CACHE_ROOT"
echo "Disk usage:   $(du -sh "$CACHE_ROOT" 2>/dev/null | cut -f1)"
echo "Next step:    sbatch ./sbatch/run_qwen_benchmark.sh"
echo "============================================"
