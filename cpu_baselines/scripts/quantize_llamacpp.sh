#!/usr/bin/env bash
# Quantization ladder for Qwen3-0.6B (llama.cpp).
# imatrix used for every type at or below 3-bit (Q3_K_M, Q2_K, IQ2_M, IQ2_XS,
# IQ2_XXS, IQ1_S); Q4_K_M and Q4_0 quantized without imatrix.
# llama-quantize prints the overall bpw ("X.XX bpw") per model -> captured in
# per-type logs llamacpp_quantize_<TYPE>.log for the CSV.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
IMATRIX="$GGUF_DIR/qwen3-0.6b-wt2train.imatrix"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
SRC="$GGUF_DIR/qwen3-0.6b-bf16.gguf"
QBIN="$LLAMA_DIR/build/bin/llama-quantize"

mkdir -p "$LOG_DIR"

# no-imatrix types (4-bit)
for T in Q4_K_M Q4_0; do
    echo "=== quantize $T (no imatrix) ==="
    "$QBIN" "$SRC" "$GGUF_DIR/qwen3-0.6b-${T}.gguf" "$T" 24 \
        2>&1 | tee "$LOG_DIR/llamacpp_quantize_${T}.log" | tail -3
done

# imatrix types (<= 3-bit)
for T in Q3_K_M Q2_K IQ2_M IQ2_XS IQ2_XXS IQ1_S; do
    echo "=== quantize $T (with imatrix) ==="
    "$QBIN" --imatrix "$IMATRIX" \
        "$SRC" "$GGUF_DIR/qwen3-0.6b-${T}.gguf" "$T" 24 \
        2>&1 | tee "$LOG_DIR/llamacpp_quantize_${T}.log" | tail -3
done

ls -l "$GGUF_DIR"/qwen3-0.6b-*.gguf
