#!/usr/bin/env bash
# Phase 6: ik_llama.cpp quantization ladder for Qwen3-8B.
# Types: Q4_K_M (no imatrix, matches Phase-4 practice), IQ2_KL + IQ2_KT (with
# ik's OWN imatrix — everything below 4-bit gets one).
# Source = the mainline 8B bf16 GGUF (ik loads/quantizes it directly).
# NOTE: ik's quantize prints NO whole-file BPW; bpw parsed later from the
# model-load line "model size = ... (X.XXX BPW)" in PPL/bench-stderr logs.
# Skip-if-done per type; a failing type is logged and skipped, never fatal.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IK_DIR="$ROOT/temp/ik_llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/ik_llama.cpp"
IMATRIX="$GGUF_DIR/qwen3-8b-wt2train.ik.imatrix"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
SRC="$ROOT/downloads/cpu_baselines/llama.cpp/qwen3-8b-bf16.gguf"
QBIN="$IK_DIR/build/bin/llama-quantize"

mkdir -p "$LOG_DIR" "$GGUF_DIR"

NO_IM_TYPES=(Q4_K_M)
IM_TYPES=(IQ2_KL IQ2_KT)

for T in "${NO_IM_TYPES[@]}"; do
    OUT="$GGUF_DIR/qwen3-8b-${T}.gguf"
    if [ -s "$OUT" ]; then echo "SKIP $T (exists)"; continue; fi
    echo "=== quantize $T (no imatrix) ==="
    "$QBIN" "$SRC" "$OUT" "$T" 24 \
        > "$LOG_DIR/ikllamacpp8b_quantize_${T}.log" 2>&1 \
        && echo "OK $T" || echo "FAILED $T (see log)"
done

for T in "${IM_TYPES[@]}"; do
    OUT="$GGUF_DIR/qwen3-8b-${T}.gguf"
    if [ -s "$OUT" ]; then echo "SKIP $T (exists)"; continue; fi
    echo "=== quantize $T (with imatrix) ==="
    "$QBIN" --imatrix "$IMATRIX" \
        "$SRC" "$OUT" "$T" 24 \
        > "$LOG_DIR/ikllamacpp8b_quantize_${T}.log" 2>&1 \
        && echo "OK $T" || echo "FAILED $T (see log)"
done

ls -l "$GGUF_DIR"/qwen3-8b-*.gguf
