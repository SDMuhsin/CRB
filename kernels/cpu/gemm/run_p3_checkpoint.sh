#!/bin/bash
# P3 MID-BUILD CHECKPOINT (P3_DESIGN_BRIEF.md): paired same-window rounds of
# {hog page-cache purge -> A -> B} on 2048x1024 ny=512 threads=24 only,
# A/B = {bench_ik iq2_kl, doml3 gemm_bench} in alternating order per round.
# Each round also runs a gemm_bench --split rep set (diagnostic: quant /
# convert / GEMM phase split — required alongside the checkpoint verdict).
# Evidence (uptime, 10-s busy%, pgrep) before/between/after per round.
# Pass bar: within-round ratio ours/iq2_kl <= 1.00 (median of rounds).
cd /workspace/BiLLM2 || exit 1
TS=$(date +%Y%m%d_%H%M%S)
DIR=llmdocs/cpu_kernel_rnd/verify/p3
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
    echo "pgrep bench_ik|python|hog|gemv|gemm:"
    pgrep -af 'bench_ik|python3?|hog|gem[vm]' | grep -v $$ || echo "(none)"
}

run_ik() {
    echo "=== ROUND $1 IK start $(date) ==="
    ./kernels/cpu/bench_ik/bench_ik --type iq2_kl --dout 2048 --din 1024 \
        --ny 512 --threads 24 --bench --reps 9 < /dev/null
    echo "=== ROUND $1 IK end $(date) ==="
}

run_p3() {
    echo "=== ROUND $1 P3 start $(date) ==="
    ./kernels/cpu/gemm/gemm_bench --dout 2048 --din 1024 --ny 512 \
        --threads 24 --reps 9 < /dev/null
    ./kernels/cpu/gemm/gemm_bench --dout 2048 --din 1024 --ny 512 \
        --threads 24 --reps 5 --split < /dev/null
    echo "=== ROUND $1 P3 end $(date) ==="
}

{
    echo "=== P3 checkpoint runner start $(date)  ROUNDS=$ROUNDS ==="
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
