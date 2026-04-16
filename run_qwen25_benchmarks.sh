#!/bin/bash
# Qwen2.5 Benchmark Collection for Paper
# Models: Qwen2.5-0.5B, then Qwen2.5-1.5B
# Compares: FP16 baseline, CRB (best on Qwen2.5), BRAQ, RTN 1-bit, GPTQ 2-bit, GPTQ 4-bit
#
# CRB with magnitude metric was the best-performing method on Qwen2.5-0.5B (PPL 2160 vs BRAQ 2896)
# All binary methods use --blocksize 128 --salient_metric magnitude

set -e
source env/bin/activate
export TRANSFORMERS_CACHE="./downloads" HF_HOME="./downloads" PYTHONPATH="$PYTHONPATH:$(pwd)"

DEVICE="cuda:0"
EVALS="--eval_mmlu --eval_hellaswag --eval_arc"
COMMON="--blocksize 128 --salient_metric magnitude --device=${DEVICE}"

for MODEL_TAG in "Qwen/Qwen2.5-0.5B:qwen25_0.5b" "Qwen/Qwen2.5-1.5B:qwen25_1.5b"; do
    MODEL="${MODEL_TAG%%:*}"
    PREFIX="${MODEL_TAG##*:}"

    echo "=========================================="
    echo "${MODEL} Benchmark Suite"
    echo "Started at $(date)"
    echo "=========================================="

    # --- FP16 Baseline (no quantization) ---
    echo ""
    echo "=== FP16 Baseline: Starting at $(date) ==="
    python3 -u run.py ${MODEL} wikitext2 fp16 ${COMMON} ${EVALS} 2>&1 | tee results/${PREFIX}_fp16_benchmarks.log
    echo "=== FP16 Baseline: Finished at $(date) ==="

    # --- CRB (best method on Qwen2.5, with sign refinement) ---
    echo ""
    echo "=== CRB: Starting at $(date) ==="
    python3 -u run.py ${MODEL} wikitext2 crb ${COMMON} ${EVALS} 2>&1 | tee results/${PREFIX}_crb_benchmarks.log
    echo "=== CRB: Finished at $(date) ==="

    # --- BRAQ ---
    echo ""
    echo "=== BRAQ: Starting at $(date) ==="
    python3 -u run.py ${MODEL} wikitext2 braq ${COMMON} ${EVALS} 2>&1 | tee results/${PREFIX}_braq_benchmarks.log
    echo "=== BRAQ: Finished at $(date) ==="

    # --- RTN 1-bit ---
    echo ""
    echo "=== RTN 1-bit: Starting at $(date) ==="
    python3 -u run.py ${MODEL} wikitext2 rtn ${COMMON} ${EVALS} 2>&1 | tee results/${PREFIX}_rtn_1bit_benchmarks.log
    echo "=== RTN 1-bit: Finished at $(date) ==="

    # --- GPTQ 2-bit ---
    echo ""
    echo "=== GPTQ 2-bit: Starting at $(date) ==="
    python3 -u run.py ${MODEL} wikitext2 2bit ${COMMON} ${EVALS} 2>&1 | tee results/${PREFIX}_gptq_2bit_benchmarks.log
    echo "=== GPTQ 2-bit: Finished at $(date) ==="

    # --- GPTQ 4-bit ---
    echo ""
    echo "=== GPTQ 4-bit: Starting at $(date) ==="
    python3 -u run.py ${MODEL} wikitext2 4bit ${COMMON} ${EVALS} 2>&1 | tee results/${PREFIX}_gptq_4bit_benchmarks.log
    echo "=== GPTQ 4-bit: Finished at $(date) ==="

    echo ""
    echo "=========================================="
    echo "${MODEL} BENCHMARKS COMPLETE at $(date)"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "ALL BENCHMARKS COMPLETE at $(date)"
echo "=========================================="
echo ""
echo "Results saved to results/qwen25_*_benchmarks.log"
echo "Expected columns: PPL (wikitext2), MMLU, HellaSwag, ARC-Easy, ARC-Challenge"
