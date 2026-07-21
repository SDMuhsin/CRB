#!/usr/bin/env bash
# Compute ik_llama.cpp's OWN importance matrix for Qwen3-0.6B.
# Reuses the Phase-2 mainline bf16 GGUF (verified to load in ik) and the
# Phase-2 calibration text (~100k tokens wikitext-2-raw TRAIN prefix).
# Do NOT reuse mainline's .imatrix file: the 2026 mainline writes GGUF-format
# imatrix files which this fork may not read -> generate ik's own binary one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IK_DIR="$ROOT/temp/ik_llama.cpp"
SRC_GGUF="$ROOT/downloads/cpu_baselines/llama.cpp/qwen3-0.6b-bf16.gguf"
OUT_DIR="$ROOT/downloads/cpu_baselines/ik_llama.cpp"
CALIB="$ROOT/downloads/cpu_baselines/calib/wt2_train_calib.txt"

mkdir -p "$OUT_DIR"

"$IK_DIR/build/bin/llama-imatrix" \
    -m "$SRC_GGUF" \
    -f "$CALIB" \
    -c 2048 \
    -t 24 \
    -o "$OUT_DIR/qwen3-0.6b-wt2train.ik.imatrix"

ls -l "$OUT_DIR/qwen3-0.6b-wt2train.ik.imatrix"
