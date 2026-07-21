#!/usr/bin/env bash
# Quantization ladder for Qwen3-0.6B with ik_llama.cpp's llama-quantize.
# Source = Phase-2 mainline bf16 GGUF (verified loadable by ik binaries).
# imatrix (ik's own, wt2-train ~100k tokens) for every type <= ~3.5 bpw:
#   Q2_K, IQ2_XS, IQ2_XXS, IQ3_K, IQ2_K, IQ2_KS, IQ2_KL, IQ2_KT
# 4-bit types quantized WITHOUT imatrix (matches Phase-2 practice):
#   Q4_K_M, Q4_0, IQ4_KS
# A failing type is logged and skipped (set +e per type), never fatal.
# NOTE: ik's quantize does NOT print a whole-file BPW; bpw is parsed later
# from the model-load line "model size = ... (X.XXX BPW)" in the PPL logs.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IK_DIR="$ROOT/temp/ik_llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/ik_llama.cpp"
IMATRIX="$GGUF_DIR/qwen3-0.6b-wt2train.ik.imatrix"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
SRC="$ROOT/downloads/cpu_baselines/llama.cpp/qwen3-0.6b-bf16.gguf"
QBIN="$IK_DIR/build/bin/llama-quantize"

mkdir -p "$LOG_DIR" "$GGUF_DIR"

NO_IM_TYPES=(Q4_K_M Q4_0 IQ4_KS)
IM_TYPES=(Q2_K IQ2_XS IQ2_XXS IQ3_K IQ2_K IQ2_KS IQ2_KL IQ2_KT)

for T in "${NO_IM_TYPES[@]}"; do
    echo "=== quantize $T (no imatrix) ==="
    "$QBIN" "$SRC" "$GGUF_DIR/qwen3-0.6b-${T}.gguf" "$T" 24 \
        > "$LOG_DIR/ikllamacpp_quantize_${T}.log" 2>&1 \
        && echo "OK $T" || echo "FAILED $T (see log)"
done

for T in "${IM_TYPES[@]}"; do
    echo "=== quantize $T (with imatrix) ==="
    "$QBIN" --imatrix "$IMATRIX" \
        "$SRC" "$GGUF_DIR/qwen3-0.6b-${T}.gguf" "$T" 24 \
        > "$LOG_DIR/ikllamacpp_quantize_${T}.log" 2>&1 \
        && echo "OK $T" || echo "FAILED $T (see log)"
done

ls -l "$GGUF_DIR"/qwen3-0.6b-*.gguf
