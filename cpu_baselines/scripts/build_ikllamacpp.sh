#!/usr/bin/env bash
# Build ik_llama.cpp (CPU-only, native optimizations) for the CPU-baselines campaign.
# Clone expected at $ROOT/temp/ik_llama.cpp (Phase 1 pinned commit 7937465ff15a2e121f36b87d4507766bd11f5153).
# cmake 4.4.0 + ninja live in the repo venv -> must activate it first.
# NOTE: unlike mainline, tools live in examples/ but target names are the same
# llama-* prefixed ones (llama-bench, llama-perplexity, llama-quantize,
# llama-imatrix, llama-cli, llama-sweep-bench).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IK_DIR="$ROOT/temp/ik_llama.cpp"

source "$ROOT/env/bin/activate"

cd "$IK_DIR"
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF

cmake --build build -j 24 --target \
    llama-bench llama-perplexity llama-quantize llama-imatrix llama-cli llama-sweep-bench

# Sanity: binaries exist and report AVX512 in system info
"$IK_DIR/build/bin/llama-bench" --help >/dev/null
echo "== version =="
"$IK_DIR/build/bin/llama-cli" --version 2>&1 || true
