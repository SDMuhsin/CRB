#!/bin/bash
# BLOOM-1.7B Camera-Ready Benchmark Collection
# Run all remaining methods sequentially after RTN

set -e
source env/bin/activate
export TRANSFORMERS_CACHE="./downloads" HF_HOME="./downloads" PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "=== CRB: Starting at $(date) ==="
python3 -u run.py bigscience/bloom-1b7 wikitext2 crb --blocksize 128 --salient_metric magnitude --device="cuda:0" --eval_mmlu --eval_hellaswag --eval_arc 2>&1 | tee results/bloom_crb_benchmarks.log
echo "=== CRB: Finished at $(date) ==="

echo "=== BRAQ: Starting at $(date) ==="
python3 -u run.py bigscience/bloom-1b7 wikitext2 braq --blocksize 128 --salient_metric magnitude --device="cuda:0" --eval_mmlu --eval_hellaswag --eval_arc 2>&1 | tee results/bloom_braq_benchmarks.log
echo "=== BRAQ: Finished at $(date) ==="

echo "=== GPTQ 2-bit: Starting at $(date) ==="
cd /workspace/BiLLM2
python3 -u ./gptq/bloom.py bigscience/bloom-1b7 wikitext2 --wbits 2 --groupsize 128 --device cuda:0 --eval_mmlu --eval_hellaswag --eval_arc 2>&1 | tee results/bloom_gptq_2bit.log
echo "=== GPTQ 2-bit: Finished at $(date) ==="

echo "=== GPTQ 4-bit: Starting at $(date) ==="
python3 -u ./gptq/bloom.py bigscience/bloom-1b7 wikitext2 --wbits 4 --groupsize 128 --device cuda:0 --eval_mmlu --eval_hellaswag --eval_arc 2>&1 | tee results/bloom_gptq_4bit.log
echo "=== GPTQ 4-bit: Finished at $(date) ==="

echo "=== PB-LLM: Starting at $(date) ==="
python3 -u ./PB-LLM/gptq_pb/run.py bigscience/bloom-1b7 wikitext2 xnor --low_frac 0.9 --high_bit 8 --blocksize 128 --salient_metric magnitude --eval_mmlu --eval_hellaswag --eval_arc 2>&1 | tee results/bloom_pbllm.log
echo "=== PB-LLM: Finished at $(date) ==="

echo "=== ALL BLOOM BENCHMARKS COMPLETE at $(date) ==="
