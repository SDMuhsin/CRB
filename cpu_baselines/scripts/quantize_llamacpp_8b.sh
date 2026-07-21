#!/usr/bin/env bash
# Phase 6: mainline llama.cpp quantization ladder for Qwen3-8B.
# Types: Q4_K_M (no imatrix, matches Phase-2 practice), Q2_K + IQ2_XS (with
# imatrix — everything below 4-bit gets one). Source = the 8B bf16 GGUF.
# Logs (with printed whole-file BPW): llamacpp8b_quantize_<TAG>.log.
# Skip-if-done per type; a failing type is logged and skipped, never fatal.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
IMATRIX="$GGUF_DIR/qwen3-8b-wt2train.imatrix"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
SRC="$GGUF_DIR/qwen3-8b-bf16.gguf"
QBIN="$LLAMA_DIR/build/bin/llama-quantize"

mkdir -p "$LOG_DIR"

NO_IM_TYPES=(Q4_K_M)
IM_TYPES=(Q2_K IQ2_XS)

for T in "${NO_IM_TYPES[@]}"; do
    OUT="$GGUF_DIR/qwen3-8b-${T}.gguf"
    if [ -s "$OUT" ]; then echo "SKIP $T (exists)"; continue; fi
    echo "=== quantize $T (no imatrix) ==="
    "$QBIN" "$SRC" "$OUT" "$T" 24 \
        > "$LOG_DIR/llamacpp8b_quantize_${T}.log" 2>&1 \
        && echo "OK $T" || echo "FAILED $T (see log)"
done

for T in "${IM_TYPES[@]}"; do
    OUT="$GGUF_DIR/qwen3-8b-${T}.gguf"
    if [ -s "$OUT" ]; then echo "SKIP $T (exists)"; continue; fi
    echo "=== quantize $T (with imatrix) ==="
    "$QBIN" --imatrix "$IMATRIX" \
        "$SRC" "$OUT" "$T" 24 \
        > "$LOG_DIR/llamacpp8b_quantize_${T}.log" 2>&1 \
        && echo "OK $T" || echo "FAILED $T (see log)"
done

ls -l "$GGUF_DIR"/qwen3-8b-*.gguf
