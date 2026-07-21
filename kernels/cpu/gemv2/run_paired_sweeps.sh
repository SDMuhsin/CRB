#!/bin/bash
# P2c full paired sweep (P2b protocol, user-approved 2026-07-16): ROUNDS rounds
# of {hog page-cache purge -> A -> B}, A/B alternating per round:
#   IK  = bench_ik, {iq2_kl, q2_k_r4} x 5 shapes x ny=1 x threads {1,24}
#   V2  = gemv2_bench --sweep ({i8,fp} x 5 shapes x {1,24}t, static slices)
# plus, in every round, the flagship thread curve (i8 2048x1024 x
# {1,6,12,24,48}t) paired with the same curve for bench_ik iq2_kl.
# Evidence (uptime, 10-s busy%, pgrep) before/between/after per round.
cd /workspace/BiLLM2 || exit 1
TS=$(date +%Y%m%d_%H%M%S)
DIR=llmdocs/cpu_kernel_rnd/verify/p2c
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
    pgrep -af 'bench_ik|python3?|hog|gemv' | grep -v $$ || echo "(none)"
}

run_ik() {
    R=$1  # 'set --' below clobbers $1
    echo "=== ROUND $R IK start $(date) ==="
    for TY in iq2_kl q2_k_r4; do
        for SH in "2048 1024" "1024 1024" "1024 2048" "3072 1024" "1024 3072"; do
            set -- $SH
            for T in 1 24; do
                ./kernels/cpu/bench_ik/bench_ik --type $TY --dout $1 --din $2 \
                    --ny 1 --threads $T --bench --reps 9 < /dev/null
            done
        done
    done
    for T in 1 6 12 24 48; do
        ./kernels/cpu/bench_ik/bench_ik --type iq2_kl --dout 2048 --din 1024 \
            --ny 1 --threads $T --bench --reps 9 < /dev/null
    done
    echo "=== ROUND $R IK end $(date) ==="
}

run_v2() {
    echo "=== ROUND $1 V2 start $(date) ==="
    ./kernels/cpu/gemv2/gemv2_bench --sweep < /dev/null
    ./kernels/cpu/gemv2/gemv2_bench --curve < /dev/null
    echo "=== ROUND $1 V2 end $(date) ==="
}

{
    echo "=== P2c paired sweep runner start $(date)  ROUNDS=$ROUNDS ==="
    echo "protocol: shared box paired A/B per round, alternating order"
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
