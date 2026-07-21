#!/usr/bin/env bash
# Phase 6: Qwen3-8B speed benchmarks, BOTH frameworks, one sequential driver
# (machine must be otherwise idle; run only after prep_8b.sh finished and
# BEFORE any PPL runs — benches are the priority at 8B).
#   Per quant: llama-bench -p 512 -n 128 -r 3 -t 1,6,12,24,48 (-o json/-oe md)
#     + numactl --cpunodebind=0 point at -t 12 (membind EPERM in container)
#     + for ik quants only: -rtr 1 run-time-repack point at -t 24
#   BF16 reference: ONE point per framework at -t 24 only (slow anchor).
# Resumable: a phase is skipped when its .json exists non-empty.
# Outputs: llamacpp8b_bench_<TAG>*.json/.md, ikllamacpp8b_bench_<TAG>*.json/.md
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ML_BENCH="$ROOT/temp/llama.cpp/build/bin/llama-bench"
IK_BENCH="$ROOT/temp/ik_llama.cpp/build/bin/llama-bench"
ML_GGUF="$ROOT/downloads/cpu_baselines/llama.cpp"
IK_GGUF="$ROOT/downloads/cpu_baselines/ik_llama.cpp"
BF16="$ML_GGUF/qwen3-8b-bf16.gguf"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
PROGRESS="$LOG_DIR/phase6_bench_8b_progress.log"

ML_QUANTS=(Q4_K_M Q2_K IQ2_XS)
IK_QUANTS=(Q4_K_M IQ2_KL IQ2_KT)

run_bench() {  # run_bench <stem> <bin> <gguf> "<extra bench args>" [numa0]
    local STEM="$1" BIN="$2" GGUF="$3" EXTRA="$4" PIN="${5:-}"
    if [ -s "$LOG_DIR/${STEM}.json" ]; then
        echo "[$(date -Is)] SKIP  $STEM (json exists)" >> "$PROGRESS"
        return 0
    fi
    echo "[$(date -Is)] START $STEM" >> "$PROGRESS"
    if [ "$PIN" = "numa0" ]; then
        # shellcheck disable=SC2086
        numactl --cpunodebind=0 \
            "$BIN" -m "$GGUF" -p 512 -n 128 -r 3 $EXTRA -o json -oe md \
            > "$LOG_DIR/${STEM}.json" 2> "$LOG_DIR/${STEM}.md" \
            || echo "[$(date -Is)] FAIL  $STEM" >> "$PROGRESS"
    else
        # shellcheck disable=SC2086
        "$BIN" -m "$GGUF" -p 512 -n 128 -r 3 $EXTRA -o json -oe md \
            > "$LOG_DIR/${STEM}.json" 2> "$LOG_DIR/${STEM}.md" \
            || echo "[$(date -Is)] FAIL  $STEM" >> "$PROGRESS"
    fi
}

echo "=== phase6 8B bench sweep started $(date -Is) ===" >> "$PROGRESS"

# --- BF16 anchors, t=24 only ---
run_bench llamacpp8b_bench_BF16   "$ML_BENCH" "$BF16" "-t 24"
run_bench ikllamacpp8b_bench_BF16 "$IK_BENCH" "$BF16" "-t 24"

# --- mainline quants ---
for TAG in "${ML_QUANTS[@]}"; do
    GGUF="$ML_GGUF/qwen3-8b-${TAG}.gguf"
    if [ ! -s "$GGUF" ]; then
        echo "[$(date -Is)] SKIP  ml $TAG (gguf missing)" >> "$PROGRESS"; continue
    fi
    run_bench "llamacpp8b_bench_${TAG}"       "$ML_BENCH" "$GGUF" "-t 1,6,12,24,48"
    run_bench "llamacpp8b_bench_${TAG}_numa0" "$ML_BENCH" "$GGUF" "-t 12" numa0
    echo "[$(date -Is)] DONE  ml $TAG" >> "$PROGRESS"
done

# --- ik quants ---
for TAG in "${IK_QUANTS[@]}"; do
    GGUF="$IK_GGUF/qwen3-8b-${TAG}.gguf"
    if [ ! -s "$GGUF" ]; then
        echo "[$(date -Is)] SKIP  ik $TAG (gguf missing)" >> "$PROGRESS"; continue
    fi
    run_bench "ikllamacpp8b_bench_${TAG}"       "$IK_BENCH" "$GGUF" "-t 1,6,12,24,48"
    run_bench "ikllamacpp8b_bench_${TAG}_numa0" "$IK_BENCH" "$GGUF" "-t 12" numa0
    run_bench "ikllamacpp8b_bench_${TAG}_rtr"   "$IK_BENCH" "$GGUF" "-t 24 -rtr 1"
    echo "[$(date -Is)] DONE  ik $TAG" >> "$PROGRESS"
done

echo "=== phase6 8B bench sweep finished $(date -Is) ===" >> "$PROGRESS"
