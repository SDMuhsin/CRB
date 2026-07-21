#!/usr/bin/env bash
# Sequential wikitext-2-raw TEST perplexity for the full llama.cpp ladder.
# llama-perplexity -f wt2_test.raw -c 2048 -t 24 per model; FULL test set
# (no truncation). Per-model logs: llamacpp_ppl_<TAG>.log; one-line summary
# appended to llamacpp_ppl_progress.log after each model finishes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
TEST="$ROOT/downloads/cpu_baselines/calib/wt2_test.raw"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
PROGRESS="$LOG_DIR/llamacpp_ppl_progress.log"

MODELS=(F16 Q4_K_M Q4_0 Q3_K_M Q2_K IQ2_M IQ2_XS IQ2_XXS IQ1_S)

echo "=== PPL sweep started $(date -Is) ===" >> "$PROGRESS"
for TAG in "${MODELS[@]}"; do
    LTAG=$(echo "$TAG" | tr '[:upper:]' '[:lower:]')
    if [ "$TAG" = "F16" ]; then
        GGUF="$GGUF_DIR/qwen3-0.6b-f16.gguf"
    else
        GGUF="$GGUF_DIR/qwen3-0.6b-${TAG}.gguf"
    fi
    LOG="$LOG_DIR/llamacpp_ppl_${TAG}.log"
    echo "[$(date -Is)] START $TAG" >> "$PROGRESS"
    "$LLAMA_DIR/build/bin/llama-perplexity" \
        -m "$GGUF" -f "$TEST" -c 2048 -t 24 > "$LOG" 2>&1
    PPL=$(grep -oE "Final estimate: PPL = [0-9.]+( \+/- [0-9.]+)?" "$LOG" | tail -1 || echo "PARSE_FAIL")
    echo "[$(date -Is)] DONE  $TAG  $PPL" >> "$PROGRESS"
done
echo "=== PPL sweep finished $(date -Is) ===" >> "$PROGRESS"
