#!/usr/bin/env bash
# Wait for tq_chain to finish, then run the (fixed) SmolLM2 TesseraQ job on the 48G slice.
set +e; cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/scratch/ckp908/crb/logs/rerun; D="$LOG/pipe_tq-chain.drv"
for i in $(seq 1 180); do
  grep -q "### TQ CHAIN DONE" "$D" 2>/dev/null && break
  sleep 300
done
grep -q "### TQ CHAIN DONE" "$D" || { echo "### TQ smollm2-retry ABORT chain never finished $(date '+%H:%M:%S')" >> "$D"; exit 1; }
echo "### TQ smollm2-retry START $(date '+%H:%M:%S')" >> "$D"
SECONDS=0
BILLM_BENCH_CSV="$LOG/tq-smollm2_eval.csv" python3 -u src/run_tesseraq.py HuggingFaceTB/SmolLM2-1.7B wikitext2 \
  --bit 2 --group_size 128 --iterations 250 --batch_size 4 \
  --nsamples 512 --seed 0 --device cuda:0 --full_eval > "$LOG/tq-smollm2.log" 2>&1
echo "eval  rc=$? tq-smollm2-retry elapsed=${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
