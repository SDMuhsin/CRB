#!/bin/bash
# P2b shared-box paired bench protocol (user-approved 2026-07-16, replaces idle-wait):
# ROUNDS rounds of {hog page-cache purge -> sweep A -> sweep B}, where A/B =
# {bench_ik --sweep --reps 9, gemv_bench --sweep} in alternating order per round
# (cancels slow co-tenant drift). Evidence (uptime, 10-s busy%, pgrep) logged
# before/between/after the sweeps of every round. Analysis: per-config
# median-of-round-medians with min/max error bars; ik-vs-gemv ratios computed
# WITHIN each round (paired ratios cancel common-mode contention).
cd /workspace/BiLLM2 || exit 1
TS=$(date +%Y%m%d_%H%M%S)
DIR=llmdocs/cpu_kernel_rnd/verify/p2b
LOG=$DIR/combined_paired_${TS}.log
ROUNDS=${ROUNDS:-6}
mkdir -p "$DIR"

busy_pct() {
    read -r _ u n s i _ < /proc/stat
    a=$((u + n + s)); b=$i
    sleep 10
    read -r _ u n s i _ < /proc/stat
    c=$((u + n + s)); d=$i
    echo $(( (c - a) * 100 / ( (c - a) + (d - b) + 1) ))
}

evidence() {
    echo "--- evidence $1 $(date) ---"
    uptime
    echo "busy_pct(10s)=$(busy_pct)%"
    echo "pgrep bench_ik|python|hog|gemv:"
    pgrep -af 'bench_ik|python3?|hog|gemv_bench' | grep -v $$ || echo "(none)"
}

run_ik() {
    echo "=== ROUND $1 IK start $(date) ==="
    ./kernels/cpu/bench_ik/bench_ik --sweep --reps 9 < /dev/null
    echo "=== ROUND $1 IK end $(date) ==="
}

run_gemv() {
    echo "=== ROUND $1 GEMV start $(date) ==="
    ./kernels/cpu/gemv/gemv_bench --sweep < /dev/null
    echo "=== ROUND $1 GEMV end $(date) ==="
}

{
    echo "=== P2b paired sweep runner start $(date)  ROUNDS=$ROUNDS ==="
    echo "protocol: shared box (user-approved 2026-07-16); paired A/B per round, alternating order"
    for r in $(seq 1 "$ROUNDS"); do
        echo "=== ROUND $r hog purge $(date) ==="
        ./kernels/cpu/roofline/hog 102400
        evidence "round$r pre"
        if [ $((r % 2)) -eq 1 ]; then
            run_ik "$r";   evidence "round$r mid";  run_gemv "$r"
        else
            run_gemv "$r"; evidence "round$r mid";  run_ik "$r"
        fi
        evidence "round$r post"
    done
    echo "=== ALL DONE $(date) ==="
} > "$LOG" 2>&1
echo "$LOG"
