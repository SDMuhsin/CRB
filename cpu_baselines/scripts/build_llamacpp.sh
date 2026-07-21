#!/usr/bin/env bash
# Build llama.cpp (CPU-only, native optimizations) for the CPU-baselines campaign.
# Clone expected at $ROOT/temp/llama.cpp (Phase 1 pinned commit 2969d6d15d67a08e7b83f26164b15350c79c5248).
# cmake 4.4.0 + ninja live in the repo venv -> must activate it first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"

source "$ROOT/env/bin/activate"

cd "$LLAMA_DIR"
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF

cmake --build build -j 24 --target \
    llama-bench llama-perplexity llama-quantize llama-imatrix llama-cli

# Sanity: binaries exist and report AVX512 in system info
"$LLAMA_DIR/build/bin/llama-bench" --help >/dev/null
echo "== system-info line =="
"$LLAMA_DIR/build/bin/llama-cli" --version 2>&1 || true
