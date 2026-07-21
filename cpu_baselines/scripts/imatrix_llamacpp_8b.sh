#!/usr/bin/env bash
# Phase 6: mainline llama.cpp importance matrix for Qwen3-8B on the bf16 GGUF.
# Same calib text as Phases 2/4 (~100k tokens wikitext-2-raw TRAIN prefix),
# -c 2048 -t 24. Needed for all sub-4-bit quants (Q2_K, IQ2_XS here).
# Skip-if-done: exits early when the imatrix file already exists non-empty.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
CALIB="$ROOT/downloads/cpu_baselines/calib/wt2_train_calib.txt"
OUT="$GGUF_DIR/qwen3-8b-wt2train.imatrix"

if [ -s "$OUT" ]; then
    echo "SKIP imatrix: $OUT already exists"
else
    "$LLAMA_DIR/build/bin/llama-imatrix" \
        -m "$GGUF_DIR/qwen3-8b-bf16.gguf" \
        -f "$CALIB" \
        -c 2048 \
        -t 24 \
        -o "$OUT"
fi

ls -l "$OUT"
