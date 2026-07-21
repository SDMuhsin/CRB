#!/usr/bin/env bash
# Sequential wikitext-2-raw TEST perplexity for the ik_llama.cpp ladder.
# ik llama-perplexity -f wt2_test.raw -c 2048 -t 24 per model; FULL test set
# (same file as Phase 2 -> ratios comparable via each harness's own anchor).
# Anchor = mainline F16 GGUF measured with IK'S OWN perplexity tool (do not
# reuse mainline's 18.4881 number). Missing GGUFs (failed quants) are skipped.
# Per-model logs: ikllamacpp_ppl_<TAG>.log; progress appended to
# ikllamacpp_ppl_progress.log.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IK_DIR="$ROOT/temp/ik_llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/ik_llama.cpp"
ML_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
TEST="$ROOT/downloads/cpu_baselines/calib/wt2_test.raw"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
PROGRESS="$LOG_DIR/ikllamacpp_ppl_progress.log"

MODELS=(F16 Q4_K_M Q4_0 IQ4_KS IQ3_K Q2_K IQ2_KL IQ2_K IQ2_XS IQ2_KS IQ2_KT IQ2_XXS)

echo "=== ik PPL sweep started $(date -Is) ===" >> "$PROGRESS"
for TAG in "${MODELS[@]}"; do
    if [ "$TAG" = "F16" ]; then
        GGUF="$ML_DIR/qwen3-0.6b-f16.gguf"
    else
        GGUF="$GGUF_DIR/qwen3-0.6b-${TAG}.gguf"
    fi
    if [ ! -s "$GGUF" ]; then
        echo "[$(date -Is)] SKIP  $TAG (gguf missing)" >> "$PROGRESS"
        continue
    fi
    LOG="$LOG_DIR/ikllamacpp_ppl_${TAG}.log"
    if [ -s "$LOG" ] && grep -q "Final estimate" "$LOG"; then
        echo "[$(date -Is)] SKIP  $TAG (already done)" >> "$PROGRESS"
        continue
    fi
    echo "[$(date -Is)] START $TAG" >> "$PROGRESS"
    "$IK_DIR/build/bin/llama-perplexity" \
        -m "$GGUF" -f "$TEST" -c 2048 -t 24 > "$LOG" 2>&1
    PPL=$(grep -oE "Final estimate: PPL over [0-9]+ chunks for n_ctx=[0-9]+ = [0-9.]+( \+/- [0-9.]+)?" "$LOG" | tail -1 || echo "PARSE_FAIL")
    echo "[$(date -Is)] DONE  $TAG  $PPL" >> "$PROGRESS"
done
echo "=== ik PPL sweep finished $(date -Is) ===" >> "$PROGRESS"
