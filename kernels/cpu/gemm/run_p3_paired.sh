#!/bin/bash
# P3 full paired sweep (G-SPEED-P3, shared-box protocol): ROUNDS rounds of
# {hog page-cache purge -> A -> B}, A/B alternating per round:
#   IK = bench_ik {iq2_kl, q8_k_r16} x 5 shapes x ny=512 x threads {24,48}
#   P3 = gemm_bench --sweep (5 shapes x ny=512 x threads {24,48}, mk 2x8)
# Evidence (uptime, 10-s busy%, pgrep) before/between/after per round.
cd /workspace/BiLLM2 || exit 1
TS=$(date +%Y%m%d_%H%M%S)
DIR=llmdocs/cpu_kernel_rnd/verify/p3
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
    echo "pgrep bench_ik|python|hog|gemv|gemm:"
    pgrep -af 'bench_ik|python3?|hog|gem[vm]' | grep -v $$ || echo "(none)"
}

run_ik() {
    R=$1  # 'set --' below clobbers $1
    echo "=== ROUND $R IK start $(date) ==="
    for TY in iq2_kl q8_k_r16; do
        for SH in "2048 1024" "1024 1024" "1024 2048" "3072 1024" "1024 3072"; do
            set -- $SH
            for T in 24 48; do
                ./kernels/cpu/bench_ik/bench_ik --type $TY --dout $1 --din $2 \
                    --ny 512 --threads $T --bench --reps 9 < /dev/null
            done
        done
    done
    echo "=== ROUND $R IK end $(date) ==="
}

run_p3() {
    echo "=== ROUND $1 P3 start $(date) ==="
    ./kernels/cpu/gemm/gemm_bench --sweep --reps 9 < /dev/null
    echo "=== ROUND $1 P3 end $(date) ==="
}

{
    echo "=== P3 paired sweep runner start $(date)  ROUNDS=$ROUNDS ==="
    echo "protocol: shared box paired A/B per round, alternating order"
    for r in $(seq 1 "$ROUNDS"); do
        echo "=== ROUND $r hog purge $(date) ==="
        ./kernels/cpu/roofline/hog 102400
        evidence "round$r pre"
        if [ $((r % 2)) -eq 1 ]; then
            run_ik "$r";  evidence "round$r mid";  run_p3 "$r"
        else
            run_p3 "$r";  evidence "round$r mid";  run_ik "$r"
        fi
        evidence "round$r post"
    done
    echo "=== ALL DONE $(date) ==="
} > "$LOG" 2>&1
echo "$LOG"
