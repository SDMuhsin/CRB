#!/bin/bash
# P2d citable pin-mode bake-off under the shared-box protocol: per round
# {hog purge -> evidence -> ik 24t -> v2 24t x {cpu,node,none} -> evidence},
# alternating v2-first/ik-first. Logs: verify/p2c/p2d_pins_<ts>.log
cd /workspace/BiLLM2 || exit 1
TS=$(date +%Y%m%d_%H%M%S)
LOG=llmdocs/cpu_kernel_rnd/verify/p2c/p2d_pins_${TS}.log
ROUNDS=${ROUNDS:-3}
busy_pct() { read -r _ u n s i _ < /proc/stat; a=$((u+n+s)); b=$i; sleep 10; read -r _ u n s i _ < /proc/stat; echo $(( ((u+n+s)-a)*100 / ( ((u+n+s)-a)+(i-b)+1 ) )); }
evidence() { echo "--- evidence $1 $(date) ---"; uptime; echo "busy_pct(10s)=$(busy_pct)%"; pgrep -af 'bench_i[k]|gemv2_benc[h]|ho[g] ' || echo "(none)"; }
run_ik() { ./kernels/cpu/bench_ik/bench_ik --type iq2_kl --dout 2048 --din 1024 --ny 1 --threads 24 --bench --reps 9 < /dev/null; }
run_v2() { for P in cpu node none; do echo "### DOML2_PIN=$P"; DOML2_PIN=$P ./kernels/cpu/gemv2/gemv2_bench --variant i8 --dout 2048 --din 1024 --threads 24 --reps 9 < /dev/null; done; }
{
  echo "=== P2d pin bake-off start $(date) ROUNDS=$ROUNDS ==="
  for r in $(seq 1 "$ROUNDS"); do
    echo "=== ROUND $r hog purge $(date) ==="
    ./kernels/cpu/roofline/hog 102400
    evidence "round$r pre"
    if [ $((r % 2)) -eq 1 ]; then run_ik; run_v2; else run_v2; run_ik; fi
    evidence "round$r post"
  done
  echo "=== ALL DONE $(date) ==="
} > "$LOG" 2>&1
echo "$LOG"
