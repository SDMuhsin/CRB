#!/bin/bash
# P2b citable same-window sweep runner (PI): waits for an idle box
# (1-min load <= 1 AND measured busy fraction < 4%), then runs BOTH
# bench_ik --sweep and gemv_bench --sweep back-to-back in the same idle
# window, each preceded by a hog page-cache purge, with idle/placement
# evidence recorded in-log. Numbers only from logs (G-IDLE/G-BASELINE).
cd /workspace/BiLLM2 || exit 1
TS=$(date +%Y%m%d_%H%M%S)
DIR=llmdocs/cpu_kernel_rnd/verify/p2b
LOG=$DIR/combined_${TS}.log
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

{
    echo "=== P2b combined sweep runner start $(date) ==="
    OK=0
    for try in $(seq 1 1440); do
        L1=$(cut -d' ' -f1 /proc/loadavg)
        BP=$(busy_pct)
        OK=$(awk -v l="$L1" -v b="$BP" 'BEGIN{print (l<=1.0 && b<4) ? 1 : 0}')
        echo "poll $try: load1=$L1 busy=${BP}% ok=$OK  $(date +%H:%M:%S)"
        if [ "$OK" = "1" ]; then break; fi
        sleep 50
    done
    if [ "$OK" != "1" ]; then
        echo "GAVE UP waiting for idle box"; exit 1
    fi
    echo "=== box idle ==="

    echo "=== hog page-cache purge #1 ==="
    ./kernels/cpu/roofline/hog 102400
    evidence "pre-bench_ik"
    echo "=== bench_ik --sweep --reps 9 ==="
    ./kernels/cpu/bench_ik/bench_ik --sweep --reps 9 < /dev/null
    evidence "post-bench_ik"

    echo "=== hog page-cache purge #2 ==="
    ./kernels/cpu/roofline/hog 102400
    evidence "pre-gemv"
    echo "=== gemv_bench --sweep ==="
    ./kernels/cpu/gemv/gemv_bench --sweep < /dev/null
    evidence "post-gemv"

    echo "=== ALL DONE $(date) ==="
} > "$LOG" 2>&1
echo "$LOG"
