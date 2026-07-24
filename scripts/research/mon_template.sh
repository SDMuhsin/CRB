#!/usr/bin/env bash
# Fire when the 48GB slice frees (re-atune fully done) or after cap. Also report which pipelines reached eval.
LOG=/scratch/ckp908/crb/logs/rerun
for i in $(seq 1 25); do   # 25*60s = 25min cap
  if grep -q "### DONE 17b_b8" "$LOG/rdrv_17b_b8.log" 2>/dev/null; then echo "REATUNE_FULLY_DONE_48GB_FREE at iter $i"; exit 0; fi
  sleep 60 || exit 2
done
echo "CAP_REACHED (re-atune eval still running)"
