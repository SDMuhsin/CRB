#!/bin/bash
# Qwen3-1.7B Selective Binarization: keep sensitive layers in FP16
# Tests CRB-Native (coupling=0.25) with different layer skip configs
#
# Qwen3-1.7B has 28 layers (indices 0-27)
# --minlayer X --maxlayer Y binarizes layers [X, Y), leaves rest in FP16

set -e
source env/bin/activate
export TRANSFORMERS_CACHE="./downloads" HF_HOME="./downloads" PYTHONPATH="$PYTHONPATH:$(pwd)"

MODEL="Qwen/Qwen3-1.7B"
DEVICE="cuda:0"
EVALS="--eval_mmlu --eval_hellaswag --eval_arc"
COMMON="--blocksize 128 --salient_metric magnitude --device=${DEVICE} --coupling 0.25"

echo "=========================================="
echo "Qwen3-1.7B Selective Binarization"
echo "Started at $(date)"
echo "=========================================="

# --- Config 1: Skip first + last layer (eff. ~2.17 bits) ---
echo ""
echo "=== skip-ends (layers 1-26 binarized, 0+27 FP16, ~2.17 bits): Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 crb_native ${COMMON} --minlayer 1 --maxlayer 27 ${EVALS} 2>&1 | tee results/qwen3_1.7b_crb_native_skip_ends.log
echo "=== skip-ends: Finished at $(date) ==="

# --- Config 2: Skip first 2 + last 2 layers (eff. ~3.24 bits) ---
echo ""
echo "=== skip-2 (layers 2-25 binarized, 0-1+26-27 FP16, ~3.24 bits): Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 crb_native ${COMMON} --minlayer 2 --maxlayer 26 ${EVALS} 2>&1 | tee results/qwen3_1.7b_crb_native_skip_2.log
echo "=== skip-2: Finished at $(date) ==="

# --- Config 3: Skip first 4 + last 4 layers (eff. ~5.37 bits) ---
echo ""
echo "=== skip-4 (layers 4-23 binarized, 0-3+24-27 FP16, ~5.37 bits): Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 crb_native ${COMMON} --minlayer 4 --maxlayer 24 ${EVALS} 2>&1 | tee results/qwen3_1.7b_crb_native_skip_4.log
echo "=== skip-4: Finished at $(date) ==="

# --- Baseline: BRAQ skip-ends for comparison (same layer config, different method) ---
echo ""
echo "=== BRAQ skip-ends (layers 1-26, ~2.17 bits): Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 braq ${COMMON%--coupling*} --minlayer 1 --maxlayer 27 ${EVALS} 2>&1 | tee results/qwen3_1.7b_braq_skip_ends.log
echo "=== BRAQ skip-ends: Finished at $(date) ==="

echo ""
echo "=========================================="
echo "ALL SELECTIVE BINARIZATION RUNS COMPLETE at $(date)"
echo "=========================================="
echo ""
echo "Results saved to results/qwen3_1.7b_*_skip_*.log"
echo ""
echo "Expected comparison:"
echo "  Full CRB-Native (1.11 bits): PPL 8,169 / MMLU 24.8% / HellaSwag 24.4%"
echo "  skip-ends (~2.17 bits):      ?"
echo "  skip-2    (~3.24 bits):      ?"
echo "  skip-4    (~5.37 bits):      ?"
echo "  Full FP16 (16 bits):         PPL 16.72 / MMLU 59.9% / HellaSwag 58.5%"
