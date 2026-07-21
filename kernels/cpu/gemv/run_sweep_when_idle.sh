#!/bin/bash
# P2b citable sweep runner: waits for an idle box (1-min load <= 1 AND
# measured busy fraction < 4%), purges page cache with hog, records evidence,
# then runs the full gemv_bench sweep. Numbers only from logs (PROMPT.md).
cd /workspace/BiLLM2 || exit 1
TS=$(date +%Y%m%d_%H%M%S)
LOG=llmdocs/cpu_kernel_rnd/verify/p2b/sweep_${TS}.log
mkdir -p llmdocs/cpu_kernel_rnd/verify/p2b

busy_pct() {
    read -r _ u n s i _ < /proc/stat
    a=$((u + n + s)); b=$i
    sleep 10
    read -r _ u n s i _ < /proc/stat
    c=$((u + n + s)); d=$i
    echo $(( (c - a) * 100 / ( (c - a) + (d - b) + 1) ))
}

{
    echo "=== P2b sweep runner start $(date) ==="
    for try in $(seq 1 480); do
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
    echo "=== box idle; running hog page-cache purge ==="
    ./kernels/cpu/roofline/hog 102400
    echo "=== evidence before timed run ==="
    uptime
    echo "pgrep bench_ik|python|hog:"
    pgrep -af 'bench_ik|python|hog' | grep -v $$ || echo "(none)"
    echo "=== gemv_bench --sweep ==="
    ./kernels/cpu/gemv/gemv_bench --sweep
    echo "=== post-run uptime ==="
    uptime
    echo "=== done $(date) ==="
} > "$LOG" 2>&1
echo "$LOG"
