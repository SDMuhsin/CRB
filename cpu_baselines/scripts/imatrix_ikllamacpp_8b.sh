#!/usr/bin/env bash
# Phase 6: ik_llama.cpp's OWN importance matrix for Qwen3-8B (legacy binary
# format — mainline's 2026 GGUF-format imatrix is NOT readable by this fork).
# Source model = mainline bf16 GGUF (ik loads it directly, Phase-4 verified).
# Same calib text as Phases 2/4, -c 2048 -t 24.
# Skip-if-done: exits early when the imatrix file already exists non-empty.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IK_DIR="$ROOT/temp/ik_llama.cpp"
SRC_GGUF="$ROOT/downloads/cpu_baselines/llama.cpp/qwen3-8b-bf16.gguf"
OUT_DIR="$ROOT/downloads/cpu_baselines/ik_llama.cpp"
CALIB="$ROOT/downloads/cpu_baselines/calib/wt2_train_calib.txt"
OUT="$OUT_DIR/qwen3-8b-wt2train.ik.imatrix"

mkdir -p "$OUT_DIR"

if [ -s "$OUT" ]; then
    echo "SKIP imatrix: $OUT already exists"
else
    "$IK_DIR/build/bin/llama-imatrix" \
        -m "$SRC_GGUF" \
        -f "$CALIB" \
        -c 2048 \
        -t 24 \
        -o "$OUT"
fi

ls -l "$OUT"
