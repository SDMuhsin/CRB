#!/usr/bin/env bash
# Compute an importance matrix (imatrix) for Qwen3-0.6B on the bf16 GGUF
# using ~100k tokens of wikitext-2-raw-v1 TRAIN (see make_eval_text.py).
# Defaults, -c 2048, CPU 24 threads. Needed by IQ2_*/IQ1_* (and used for
# Q2_K/Q3_K_M too, per campaign protocol: imatrix for everything <= 3-bit).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
CALIB="$ROOT/downloads/cpu_baselines/calib/wt2_train_calib.txt"

"$LLAMA_DIR/build/bin/llama-imatrix" \
    -m "$GGUF_DIR/qwen3-0.6b-bf16.gguf" \
    -f "$CALIB" \
    -c 2048 \
    -t 24 \
    -o "$GGUF_DIR/qwen3-0.6b-wt2train.imatrix"

ls -l "$GGUF_DIR/qwen3-0.6b-wt2train.imatrix"
