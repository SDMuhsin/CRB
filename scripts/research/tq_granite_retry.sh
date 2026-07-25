#!/usr/bin/env bash
# Granite TQ retry: first attempt died in 5s (get_model whitelist lacked granite,
# fixed in src/run_tesseraq.py). Wait for the SmolLM2 RETRY eval marker (the
# "-retry" tag specifically -- the plain tq-smollm2 rc=1 line from 14:08 is stale
# and already tripped the supervisor's wait loop once), then run granite TQ on
# the 48G slice.
set +e; cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/scratch/ckp908/crb/logs/rerun; D="$LOG/pipe_tq-chain.drv"
for i in $(seq 1 144); do
  grep -q "eval  rc=.*tq-smollm2-retry" "$D" 2>/dev/null && break
  sleep 300
done
grep -q "eval  rc=.*tq-smollm2-retry" "$D" || { echo "### TQ granite-retry ABORT smollm2-retry never landed $(date '+%H:%M:%S')" >> "$D"; exit 1; }
echo "### TQ granite-retry START $(date '+%H:%M:%S')" >> "$D"
SECONDS=0
BILLM_BENCH_CSV="$LOG/tq-granite_eval.csv" python3 -u src/run_tesseraq.py ibm-granite/granite-3.3-2b-base wikitext2 \
  --bit 2 --group_size 128 --iterations 250 --batch_size 4 \
  --nsamples 512 --seed 0 --device cuda:0 --full_eval > "$LOG/tq-granite.log" 2>&1
echo "eval  rc=$? tq-granite-retry elapsed=${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
