#!/bin/bash
# ============================================================================
# Pre-download all models and datasets to local cache
# ============================================================================
#
# Run this on a LOGIN NODE (with internet) before submitting sbatch jobs.
# Compute nodes have no internet access.
#
# Usage:
#   ./sbatch/download_cache.sh
#
# ============================================================================

set -e

source ./env/bin/activate

export HF_HOME=$(pwd)/data
export TORCH_HOME=$(pwd)/data
export HF_HUB_DISABLE_XET=1
mkdir -p $HF_HOME

echo "Cache directory: $HF_HOME"
echo ""

# ============================================================================
# Models
# ============================================================================

echo "=== Downloading models (files only, no loading) ==="

python -c "
from huggingface_hub import snapshot_download

models = [
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'huggyllama/llama-7b',
]

for model_name in models:
    print(f'Downloading {model_name}...')
    snapshot_download(repo_id=model_name)
    print(f'  Done: {model_name}')
"

echo ""

# ============================================================================
# Datasets
# ============================================================================

echo "=== Downloading datasets ==="

python -c "
from datasets import load_dataset

# GLUE tasks
glue_tasks = ['cola', 'mrpc', 'sst2', 'rte', 'qnli', 'stsb']
for task in glue_tasks:
    print(f'Downloading glue/{task}...')
    load_dataset('glue', task)
    print(f'  Done: glue/{task}')

# SuperGLUE tasks
superglue_tasks = ['boolq', 'cb']
for task in superglue_tasks:
    print(f'Downloading super_glue/{task}...')
    load_dataset('super_glue', task)
    print(f'  Done: super_glue/{task}')

# WikiText-2
print('Downloading wikitext (wikitext-2-raw-v1)...')
load_dataset('wikitext', 'wikitext-2-raw-v1')
print('  Done: wikitext-2')
"

echo ""

# ============================================================================
# Evaluation metrics
# ============================================================================

echo "=== Downloading evaluation metrics ==="

python -c "
import evaluate

metrics = ['glue', 'super_glue', 'accuracy', 'f1', 'matthews_correlation', 'pearsonr', 'spearmanr', 'perplexity']
for m in metrics:
    try:
        print(f'Downloading metric: {m}...')
        evaluate.load(m)
        print(f'  Done: {m}')
    except Exception as e:
        print(f'  Skipped {m}: {e}')
"

echo ""
echo "============================================"
echo "All downloads complete."
echo "Cache directory: $HF_HOME"
echo "============================================"
