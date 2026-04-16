#!/bin/bash
# ============================================================================
# Pre-download all models and datasets to local cache (Nibi / Alliance Canada)
# ============================================================================
#
# Run this on a LOGIN NODE (with internet) BEFORE submitting sbatch jobs.
# Compute nodes have no internet access.
#
# Project convention: cache lives at ./downloads/ (matches existing CONTEXT.md
# and src/run_*.py). The benchmark sbatch script exports HF_HOME=./downloads.
#
# Usage:
#   ./sbatch/download_cache.sh
#
# ============================================================================

set -e

source ./env/bin/activate

export HF_HOME=$(pwd)/downloads
export TRANSFORMERS_CACHE=$(pwd)/downloads
export TORCH_HOME=$(pwd)/downloads
export HF_HUB_DISABLE_XET=1
mkdir -p $HF_HOME

echo "Cache directory: $HF_HOME"
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
# - wikitext-2-raw-v1: PPL evaluation + default calibration for run.py
# - allenai/c4 (en, validation): used by datautils.get_redpajama() as the
#   RedPajama substitute for GuidedQuant/LNQ faithful calibration
#   (RedPajama-Data-1T-Sample was removed from HuggingFace).
# ============================================================================

echo "=== Downloading datasets ==="

python -c "
from datasets import load_dataset

print('Downloading wikitext (wikitext-2-raw-v1)...')
load_dataset('wikitext', 'wikitext-2-raw-v1')
print('  Done: wikitext-2')

# C4 'en' validation subset is what get_redpajama() streams.
# The first ~4096 documents are enough for 1024 samples × seqlen 4096.
print('Downloading allenai/c4 (en, validation, streaming pull)...')
ds = load_dataset('allenai/c4', 'en', split='validation', streaming=False)
print(f'  Done: allenai/c4 validation ({len(ds)} rows)')
"

echo ""

# ============================================================================
# Pre-tokenize calibration sets to avoid first-job cache miss races
# ============================================================================
#
# datautils.get_loaders() caches tokenized samples at
# ./downloads/DOWNLOAD_<name>_<nsamples>_<seed>_<seqlen>_<model>.pt
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

# wikitext2: 128 samples × 2048 — used by run.py default + SINQ + TesseraQ
print(f'Tokenizing wikitext2/128/2048 for {model}...')
get_loaders('wikitext2', nsamples=128, seed=0, model=model, seqlen=2048)

# redpajama (C4): 1024 samples × 4096 — used by run_lnq.py faithful preset
print(f'Tokenizing redpajama/1024/4096 for {model}...')
get_loaders('redpajama', nsamples=1024, seed=0, model=model, seqlen=4096)

print('Calibration caches written under ./downloads/')
"

echo ""
echo "============================================"
echo "All downloads complete."
echo "Cache directory: $HF_HOME"
echo "Next step: sbatch ./sbatch/run_qwen_benchmark.sh   (on a login node)"
echo "============================================"
