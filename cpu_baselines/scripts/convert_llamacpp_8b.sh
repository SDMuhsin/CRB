#!/usr/bin/env bash
# Phase 6: Convert Qwen3-8B HF snapshot -> bf16 GGUF (~16 GB) for the 8B
# scale-up verification. Mainline convert_hf_to_gguf.py in the MAIN venv
# (transformers 5.3.0 — worked clean for 0.6B in Phase 2).
# NO f16 copy at 8B (saves 16 GB + an hour): the BF16 GGUF is the anchor for
# both frameworks (ik loads mainline GGUFs directly — verified in Phase 4).
# Skip-if-done: exits early when the output GGUF already exists non-empty.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"
SNAP="$ROOT/downloads/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
OUT_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
OUT="$OUT_DIR/qwen3-8b-bf16.gguf"

source "$ROOT/env/bin/activate"
mkdir -p "$OUT_DIR"

if [ -s "$OUT" ]; then
    echo "SKIP convert: $OUT already exists ($(stat -c%s "$OUT") bytes)"
else
    python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$SNAP" \
        --outfile "$OUT" \
        --outtype bf16
fi

ls -l "$OUT"
