#!/bin/bash
# run_all.sh - roofline measurement campaign runner
# usage: ./run_all.sh <phase>   phase in: stream triad cache instr vnni mmap
# Logs -> /workspace/BiLLM2/llmdocs/cpu_kernel_rnd/verify/roofline/<phase>_<ts>.log
# CSV  -> /workspace/BiLLM2/kernels/cpu/roofline/results/<phase>.csv
set -u
DIR=$(dirname "$(readlink -f "$0")")
VERIFY=/workspace/BiLLM2/llmdocs/cpu_kernel_rnd/verify/roofline
RESULTS=$DIR/results
mkdir -p "$VERIFY" "$RESULTS"
TS=$(date +%Y%m%d_%H%M%S)
PHASE=${1:?phase required}
LOG=$VERIFY/${PHASE}_${TS}.log

check_idle() {
    for i in $(seq 1 20); do
        load=$(awk '{print $1}' /proc/loadavg)
        ok=$(awk -v l="$load" 'BEGIN{print (l<=1.0)?1:0}')
        [ "$ok" = "1" ] && { echo "load OK: $load"; return 0; }
        echo "load too high ($load), waiting 30s (attempt $i)"
        sleep 30
    done
    echo "ABORT: load never dropped below 1.0"
    exit 1
}

{
echo "=== roofline phase=$PHASE ts=$TS host=$(hostname) ==="
echo "uptime: $(uptime)"
check_idle
grep -E 'MemFree|FilePages' /sys/devices/system/node/node0/meminfo /sys/devices/system/node/node1/meminfo | tr -s ' '

case $PHASE in
stream)
    for nt in 1 2 4 6 12 24; do
        echo "--- stream mode=a nt=$nt $(date +%T) ---"
        "$DIR/stream_read" 4096 $nt a 7 2>&1 | tee -a "$RESULTS/stream.csv.raw"
    done
    for mode in b c; do
        for nt in 1 2 4 6 12 24 48; do
            echo "--- stream mode=$mode nt=$nt $(date +%T) ---"
            "$DIR/stream_read" 4096 $nt $mode 7 2>&1 | tee -a "$RESULTS/stream.csv.raw"
        done
    done
    grep '^stream_read,' "$RESULTS/stream.csv.raw" > "$RESULTS/stream.csv"
    ;;
triad)
    echo "--- triad 24t node-local $(date +%T) ---"
    "$DIR/triad" 1536 24 7 2>&1 | tee "$RESULTS/triad.csv.raw"
    grep '^triad,' "$RESULTS/triad.csv.raw" > "$RESULTS/triad.csv"
    ;;
cache)
    echo "--- cache_bw cpu2 $(date +%T) ---"
    "$DIR/cache_bw" 2 7 2>&1 | tee "$RESULTS/cache.csv.raw"
    grep '^cache_bw,' "$RESULTS/cache.csv.raw" > "$RESULTS/cache.csv"
    ;;
instr)
    echo "--- instr_tp cpu2 5 reps $(date +%T) ---"
    "$DIR/instr_tp" 2 5 2>&1 | tee "$RESULTS/instr.csv.raw"
    grep '^instr_tp,' "$RESULTS/instr.csv.raw" > "$RESULTS/instr.csv"
    ;;
vnni)
    for nt in 1 12 24 48; do
        g=$(( nt >= 24 ? 100 : (nt == 12 ? 50 : 2) ))
        echo "--- vnni_gemm nt=$nt M=512 K=1024 N=2048 gemms=$g $(date +%T) ---"
        "$DIR/vnni_gemm" $nt 512 1024 2048 $g 7 2>&1 | tee -a "$RESULTS/vnni.csv.raw"
    done
    grep -E '^vnni_(gemm|freq),' "$RESULTS/vnni.csv.raw" > "$RESULTS/vnni.csv"
    ;;
mmap)
    F=$DIR/mmap_testfile.bin
    if [ ! -f "$F" ]; then
        echo "creating 2 GiB random file $F"
        dd if=/dev/urandom of="$F" bs=64M count=32 status=none
        sync
    fi
    echo "--- mmap_read 24 unpinned threads $(date +%T) ---"
    "$DIR/mmap_read" "$F" 24 7 2>&1 | tee "$RESULTS/mmap.csv.raw"
    grep '^mmap_read,' "$RESULTS/mmap.csv.raw" > "$RESULTS/mmap.csv"
    ;;
*) echo "unknown phase $PHASE"; exit 1;;
esac

echo "=== done phase=$PHASE $(date +%T) ==="
echo "uptime after: $(uptime)"
} 2>&1 | tee "$LOG"
