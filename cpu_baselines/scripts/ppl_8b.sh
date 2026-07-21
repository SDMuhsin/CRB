#!/usr/bin/env bash
# Phase 6: Qwen3-8B wikitext-2-raw FULL-test perplexity, both frameworks,
# one sequential driver in PRIORITY order (run only AFTER bench_8b.sh —
# PPL is timing-insensitive; benches need the idle machine first):
#   1. mainline BF16 ANCHOR (needed for every ratio; ~1.5 h at 8B)
#   2. ik IQ2_KL   3. mainline Q2_K   4. ik IQ2_KT
#   5. mainline IQ2_XS   6. mainline Q4_K_M   7. ik Q4_K_M
# ik's own BF16 anchor is SKIPPED: the two harnesses' 0.6B anchors agreed to
# 4 sig figs (18.4881 vs 18.4883); ik ratios reuse the mainline anchor (noted
# in CSV). Each run: -c 2048 -t 24 on the full wt2 test file.
# Resumable: skipped when the log already contains "Final estimate".
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ML_PPL="$ROOT/temp/llama.cpp/build/bin/llama-perplexity"
IK_PPL="$ROOT/temp/ik_llama.cpp/build/bin/llama-perplexity"
ML_GGUF="$ROOT/downloads/cpu_baselines/llama.cpp"
IK_GGUF="$ROOT/downloads/cpu_baselines/ik_llama.cpp"
TEST="$ROOT/downloads/cpu_baselines/calib/wt2_test.raw"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
PROGRESS="$LOG_DIR/phase6_ppl_8b_progress.log"

# "fw:TAG" in priority order
QUEUE=(ml:BF16 ik:IQ2_KL ml:Q2_K ik:IQ2_KT ml:IQ2_XS ml:Q4_K_M ik:Q4_K_M)

echo "=== phase6 8B PPL sweep started $(date -Is) ===" >> "$PROGRESS"
for ITEM in "${QUEUE[@]}"; do
    FW="${ITEM%%:*}"; TAG="${ITEM##*:}"
    if [ "$FW" = "ml" ]; then
        BIN="$ML_PPL"; STEM="llamacpp8b_ppl_${TAG}"
        if [ "$TAG" = "BF16" ]; then GGUF="$ML_GGUF/qwen3-8b-bf16.gguf"
        else GGUF="$ML_GGUF/qwen3-8b-${TAG}.gguf"; fi
    else
        BIN="$IK_PPL"; STEM="ikllamacpp8b_ppl_${TAG}"
        GGUF="$IK_GGUF/qwen3-8b-${TAG}.gguf"
    fi
    LOG="$LOG_DIR/${STEM}.log"
    if [ ! -s "$GGUF" ]; then
        echo "[$(date -Is)] SKIP  $ITEM (gguf missing)" >> "$PROGRESS"; continue
    fi
    if [ -s "$LOG" ] && grep -q "Final estimate" "$LOG"; then
        echo "[$(date -Is)] SKIP  $ITEM (already done)" >> "$PROGRESS"; continue
    fi
    echo "[$(date -Is)] START $ITEM" >> "$PROGRESS"
    "$BIN" -m "$GGUF" -f "$TEST" -c 2048 -t 24 > "$LOG" 2>&1
    PPL=$(grep -oE "Final estimate: PPL[^=]*= [0-9.]+( \+/- [0-9.]+)?" "$LOG" | tail -1 || echo "PARSE_FAIL")
    echo "[$(date -Is)] DONE  $ITEM  $PPL" >> "$PROGRESS"
done
echo "=== phase6 8B PPL sweep finished $(date -Is) ===" >> "$PROGRESS"
