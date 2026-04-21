#!/bin/bash
# Sequentially run LeanQuant_nu on Qwen3 {1.7B, 4B, 8B} on GPU 0.
# Waits for Qwen3-0.6B (launched separately) to free GPU 0 first.
set -u
cd /workspace/BiLLM2
source env/bin/activate

# Wait for 0.6B to finish before starting 1.7B (polls its log marker)
echo "[chain] waiting for Qwen3-0.6B RESULT marker..."
until grep -q '^RESULT: leanquant_nu' logs/leanquant_qwen3_0.6b.log 2>/dev/null; do
    sleep 10
done
echo "[chain] Qwen3-0.6B done; starting 1.7B"

python3 -u src/run_leanquant.py Qwen/Qwen3-1.7B wikitext2 \
  --nbits 2 --exponent 4.0 --percdamp 0.1 \
  --true_sequential --act_order \
  --calib_dataset redpajama --nsamples 128 --seqlen 2048 \
  --device cuda:0 --seed 0 > logs/leanquant_qwen3_1.7b.log 2>&1

echo "[chain] 1.7B done; starting 4B"
python3 -u src/run_leanquant.py Qwen/Qwen3-4B wikitext2 \
  --nbits 2 --exponent 4.0 --percdamp 0.1 \
  --true_sequential --act_order \
  --calib_dataset redpajama --nsamples 128 --seqlen 2048 \
  --device cuda:0 --seed 0 > logs/leanquant_qwen3_4b.log 2>&1

echo "[chain] 4B done; 8B was launched separately on GPU 1 (parallel) once Mistral finished"
echo "[chain] all Qwen3 sizes (0.6B, 1.7B, 4B) done on GPU 0"
