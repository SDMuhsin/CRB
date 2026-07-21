#!/usr/bin/env bash
# Convert Qwen3-0.6B HF snapshot -> GGUF for llama.cpp CPU baselines.
#   1. bf16 GGUF (lossless: HF weights are bf16) via convert_hf_to_gguf.py
#   2. f16 GGUF via llama-quantize bf16->F16 — this is the PPL/speed ANCHOR
#      (Ice Lake has no AVX512-BF16; F16 is the conventional llama.cpp anchor).
# Conversion runs in the MAIN venv (transformers 5.3.0). llama.cpp pins 4.57.6;
# if this ever breaks on transformers 5.x, build a throwaway venv at
# temp/convert_venv with llama.cpp's requirements — do NOT downgrade env/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"
SNAP="$ROOT/downloads/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
OUT_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"

source "$ROOT/env/bin/activate"
mkdir -p "$OUT_DIR"

python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$SNAP" \
    --outfile "$OUT_DIR/qwen3-0.6b-bf16.gguf" \
    --outtype bf16

"$LLAMA_DIR/build/bin/llama-quantize" \
    "$OUT_DIR/qwen3-0.6b-bf16.gguf" \
    "$OUT_DIR/qwen3-0.6b-f16.gguf" F16

ls -l "$OUT_DIR"
