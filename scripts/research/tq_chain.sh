#!/usr/bin/env bash
# Serial TesseraQ queue on the 48G slice (paper-exact recipe needs ~25 GiB GPU +
# >100 GB RSS => strictly one at a time). drv-style markers for mon_wave.
set +e; cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/scratch/ckp908/crb/logs/rerun
D="$LOG/pipe_tq-chain.drv"

run_tq () {  # $1=tag $2=HF model
  echo "### TQ $1 START $(date '+%H:%M:%S')" >> "$D"
  SECONDS=0
  BILLM_BENCH_CSV="$LOG/tq-$1_eval.csv" python3 -u src/run_tesseraq.py "$2" wikitext2 \
    --bit 2 --group_size 128 --iterations 250 --batch_size 4 \
    --nsamples 512 --seed 0 --device cuda:0 --full_eval \
    > "$LOG/tq-$1.log" 2>&1
  echo "eval  rc=$? tq-$1 elapsed=${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
}

run_tq falcon3  tiiuae/Falcon3-1B-Base
run_tq smollm2  HuggingFaceTB/SmolLM2-1.7B
run_tq olmo2    allenai/OLMo-2-0425-1B
run_tq helium   kyutai/helium-1-2b
echo "### TQ CHAIN DONE $(date '+%H:%M:%S') ###" >> "$D"
