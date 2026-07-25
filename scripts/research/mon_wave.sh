#!/usr/bin/env bash
# Event monitor: exits the instant a NEW pipeline reaches eval (slice freed + result ready)
# or the b8 job finishes. Bounded (75*60s = 75min cap). NFS-wobble tolerant. No spin-loop.
LOG=/scratch/ckp908/crb/logs/rerun
DRVS="pipe_k31-granite-lam16e-5-g256 pipe_tq-chain"

# baseline COUNTS per line type per drv: fires whenever a drv gains a NEW
# eval/rawRT line (handles multi-eval drvs like the TQ chain).
declare -A cntE cntR
for d in $DRVS; do
  e=$(grep -cE "^eval +rc=" "$LOG/$d.drv" 2>/dev/null); cntE[$d]=${e:-0}
  r=$(grep -cE "^rawRT rc=" "$LOG/$d.drv" 2>/dev/null); cntR[$d]=${r:-0}
done

for i in $(seq 1 75); do
  for d in $DRVS; do
    e=$(grep -cE "^eval +rc=" "$LOG/$d.drv" 2>/dev/null); e=${e:-0}
    r=$(grep -cE "^rawRT rc=" "$LOG/$d.drv" 2>/dev/null); r=${r:-0}
    if [ "$e" -gt "${cntE[$d]:-0}" ]; then
      echo "TUNED_EVAL_REACHED: $d  (iter $i)"; grep -E "^eval +rc=" "$LOG/$d.drv" | tail -1; exit 0
    fi
    if [ "$r" -gt "${cntR[$d]:-0}" ]; then
      echo "RAWRT_REACHED: $d  (iter $i)"; grep -E "^rawRT rc=" "$LOG/$d.drv" | tail -1; exit 0
    fi
  done
  sleep 60 || { echo "SLEEP_INTERRUPTED iter $i"; exit 2; }
done
echo "CAP_REACHED_75min — nothing newly at eval"
