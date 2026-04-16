#!/bin/bash
# Qwen3-1.7B Benchmark Collection for Paper
# Compares: FP16 baseline, CRB-Native (winning config), CRB (original), BRAQ, GPTQ 2-bit, GPTQ 4-bit
#
# CRB-Native uses --coupling 0.25 (optimal power-of-2 value for float16 stability)
# All binary methods use --blocksize 128 --salient_metric magnitude

set -e
source env/bin/activate
export TRANSFORMERS_CACHE="./downloads" HF_HOME="./downloads" PYTHONPATH="$PYTHONPATH:$(pwd)"

MODEL="Qwen/Qwen3-1.7B"
DEVICE="cuda:0"
EVALS="--eval_mmlu --eval_hellaswag --eval_arc"
COMMON="--blocksize 128 --salient_metric magnitude --device=${DEVICE}"

echo "=========================================="
echo "Qwen3-1.7B Benchmark Suite"
echo "Started at $(date)"
echo "=========================================="

# --- FP16 Baseline (no quantization) ---
echo ""
echo "=== FP16 Baseline: Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 fp16 ${COMMON} ${EVALS} 2>&1 | tee results/qwen3_1.7b_fp16_benchmarks.log
echo "=== FP16 Baseline: Finished at $(date) ==="

# --- CRB-Native (winning method, coupling=0.25) ---
echo ""
echo "=== CRB-Native (c=0.25): Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 crb_native ${COMMON} --coupling 0.25 ${EVALS} 2>&1 | tee results/qwen3_1.7b_crb_native_benchmarks.log
echo "=== CRB-Native (c=0.25): Finished at $(date) ==="

# --- CRB (original, with sign refinement) ---
echo ""
echo "=== CRB: Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 crb ${COMMON} ${EVALS} 2>&1 | tee results/qwen3_1.7b_crb_benchmarks.log
echo "=== CRB: Finished at $(date) ==="

# --- BRAQ ---
echo ""
echo "=== BRAQ: Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 braq ${COMMON} ${EVALS} 2>&1 | tee results/qwen3_1.7b_braq_benchmarks.log
echo "=== BRAQ: Finished at $(date) ==="

# --- GPTQ 2-bit ---
echo ""
echo "=== GPTQ 2-bit: Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 2bit ${COMMON} ${EVALS} 2>&1 | tee results/qwen3_1.7b_gptq_2bit_benchmarks.log
echo "=== GPTQ 2-bit: Finished at $(date) ==="

# --- GPTQ 4-bit ---
echo ""
echo "=== GPTQ 4-bit: Starting at $(date) ==="
python3 -u run.py ${MODEL} wikitext2 4bit ${COMMON} ${EVALS} 2>&1 | tee results/qwen3_1.7b_gptq_4bit_benchmarks.log
echo "=== GPTQ 4-bit: Finished at $(date) ==="

echo ""
echo "=========================================="
echo "ALL BENCHMARKS COMPLETE at $(date)"
echo "=========================================="
echo ""
echo "Results saved to results/qwen3_1.7b_*_benchmarks.log"
echo "Expected columns: PPL (wikitext2), MMLU, HellaSwag, ARC-Easy, ARC-Challenge"
