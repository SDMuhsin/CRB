#!/bin/bash
# P2c MID-BUILD CHECKPOINT (P2C_DESIGN_BRIEF.md): paired same-window rounds of
# {hog page-cache purge -> A -> B} on 2048x1024 ny=1 only, threads {1,24},
# A/B = {bench_ik iq2_kl, gemv2 i8 flagship} in alternating order per round.
# Evidence (uptime, 10-s busy%, pgrep) before/between/after per round.
# Pass bar: v2 1t <= 110 us AND 24t <= 9.5 us (medians of round-medians).
cd /workspace/BiLLM2 || exit 1
TS=$(date +%Y%m%d_%H%M%S)
DIR=llmdocs/cpu_kernel_rnd/verify/p2c
LOG=$DIR/checkpoint_paired_${TS}.log
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
    pgrep -af 'bench_ik|python3?|hog|gemv' | grep -v $$ || echo "(none)"
}

run_ik() {
    echo "=== ROUND $1 IK start $(date) ==="
    for T in 1 24; do
        ./kernels/cpu/bench_ik/bench_ik --type iq2_kl --dout 2048 --din 1024 \
            --ny 1 --threads $T --bench --reps 9 < /dev/null
    done
    echo "=== ROUND $1 IK end $(date) ==="
}

run_v2() {
    echo "=== ROUND $1 V2 start $(date) ==="
    for T in 1 24; do
        ./kernels/cpu/gemv2/gemv2_bench --variant i8 --dout 2048 --din 1024 \
            --threads $T --reps 9 < /dev/null
    done
    echo "=== ROUND $1 V2 end $(date) ==="
}

{
    echo "=== P2c checkpoint runner start $(date)  ROUNDS=$ROUNDS ==="
    echo "protocol: shared box paired A/B per round, alternating order (P2b protocol)"
    for r in $(seq 1 "$ROUNDS"); do
        echo "=== ROUND $r hog purge $(date) ==="
        ./kernels/cpu/roofline/hog 102400
        evidence "round$r pre"
        if [ $((r % 2)) -eq 1 ]; then
            run_ik "$r";  evidence "round$r mid";  run_v2 "$r"
        else
            run_v2 "$r";  evidence "round$r mid";  run_ik "$r"
        fi
        evidence "round$r post"
    done
    echo "=== ALL DONE $(date) ==="
} > "$LOG" 2>&1
echo "$LOG"
