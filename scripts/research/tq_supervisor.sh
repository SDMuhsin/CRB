#!/usr/bin/env bash
# Replaces the killed tq_chain tail: olmo2 (pid arg) -> markers -> [smollm2 retry fires externally] -> granite TQ.
set +e; cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh >/dev/null 2>&1
LOG=/scratch/ckp908/crb/logs/rerun; D="$LOG/pipe_tq-chain.drv"; OLMO_PID="$1"
# wait for olmo2 python to exit (bounded 8h)
for i in $(seq 1 480); do kill -0 "$OLMO_PID" 2>/dev/null || break; sleep 60; done
rows=$(grep -c "," "$LOG/tq-olmo2_eval.csv" 2>/dev/null); rows=${rows:-0}
if [ "$rows" -ge 7 ]; then rc=0; else rc="NA-chain-killed-rows-$rows"; fi
echo "eval  rc=$rc tq-olmo2 (supervisor marker) $(date '+%H:%M:%S')" >> "$D"
echo "### TQ CHAIN DONE $(date '+%H:%M:%S') ### (helium leg cancelled: checkpoint unusable; supervisor)" >> "$D"
# smollm2 retry watcher fires on the DONE marker and runs TQ-smollm2; wait for its eval line (bounded 10h)
base=$(grep -c "tq-smollm2" "$D"); base=${base:-0}
for i in $(seq 1 600); do
  cur=$(grep -c "eval  rc=.*tq-smollm2" "$D"); cur=${cur:-0}
  [ "$cur" -gt 0 ] && break; sleep 60
done
# granite TQ (serial, 48G slice)
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "### TQ granite START $(date '+%H:%M:%S')" >> "$D"
SECONDS=0
BILLM_BENCH_CSV="$LOG/tq-granite_eval.csv" python3 -u src/run_tesseraq.py ibm-granite/granite-3.3-2b-base wikitext2 \
  --bit 2 --group_size 128 --iterations 250 --batch_size 4 \
  --nsamples 512 --seed 0 --device cuda:0 --full_eval > "$LOG/tq-granite.log" 2>&1
echo "eval  rc=$? tq-granite elapsed=${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
