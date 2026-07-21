#!/usr/bin/env bash
# Phase 6 prep driver: convert -> imatrix (mainline, ik) -> quantize ladders
# (mainline, ik) -> sanity generations, strictly sequential (each step is
# skip-if-done, so re-running is cheap). Run under nohup; progress appended
# to llmdocs/cpu_kernel/verify/cpu_baseline_logs/phase6_prep_8b_progress.log.
# Speed benches must only start AFTER this driver has finished (idle machine).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$ROOT/cpu_baselines/scripts"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
PROGRESS="$LOG_DIR/phase6_prep_8b_progress.log"
ML_GGUF="$ROOT/downloads/cpu_baselines/llama.cpp"
IK_GGUF="$ROOT/downloads/cpu_baselines/ik_llama.cpp"

mkdir -p "$LOG_DIR"
step() {  # step <name> <cmd...>
    local NAME="$1"; shift
    echo "[$(date -Is)] START $NAME" >> "$PROGRESS"
    if "$@" >> "$LOG_DIR/phase6_prep_8b_${NAME}.log" 2>&1; then
        echo "[$(date -Is)] DONE  $NAME" >> "$PROGRESS"
    else
        echo "[$(date -Is)] FAIL  $NAME (rc=$?) — aborting chain" >> "$PROGRESS"
        exit 1
    fi
}

echo "=== phase6 prep driver started $(date -Is) ===" >> "$PROGRESS"

step convert     bash "$SCRIPTS/convert_llamacpp_8b.sh"
step imatrix_ml  bash "$SCRIPTS/imatrix_llamacpp_8b.sh"
step imatrix_ik  bash "$SCRIPTS/imatrix_ikllamacpp_8b.sh"
step quant_ml    bash "$SCRIPTS/quantize_llamacpp_8b.sh"
step quant_ik    bash "$SCRIPTS/quantize_ikllamacpp_8b.sh"

# Sanity gate: 2-chunk perplexity per framework (proves the 8B quants load
# and produce sane outputs). Do NOT use mainline llama-cli here: on this
# build (b50-2969d6d) its chat UI ignores -no-cnv AND stdin-EOF (</dev/null)
# and spins forever on its interactive '>' prompt — hung Phase 6 for 26 h.
echo "[$(date -Is)] START sanity_ml (2-chunk ppl Q4_K_M)" >> "$PROGRESS"
"$ROOT/temp/llama.cpp/build/bin/llama-perplexity" \
    -m "$ML_GGUF/qwen3-8b-Q4_K_M.gguf" \
    -f "$ROOT/downloads/cpu_baselines/calib/wt2_test.raw" \
    -c 2048 -t 24 --chunks 2 \
    < /dev/null > "$LOG_DIR/llamacpp8b_sanity_ppl2_Q4_K_M.log" 2>&1 \
    && echo "[$(date -Is)] DONE  sanity_ml" >> "$PROGRESS" \
    || echo "[$(date -Is)] FAIL  sanity_ml" >> "$PROGRESS"

echo "[$(date -Is)] START sanity_ik (2-chunk ppl IQ2_KL)" >> "$PROGRESS"
"$ROOT/temp/ik_llama.cpp/build/bin/llama-perplexity" \
    -m "$IK_GGUF/qwen3-8b-IQ2_KL.gguf" \
    -f "$ROOT/downloads/cpu_baselines/calib/wt2_test.raw" \
    -c 2048 -t 24 --chunks 2 \
    < /dev/null > "$LOG_DIR/ikllamacpp8b_sanity_ppl2_IQ2_KL.log" 2>&1 \
    && echo "[$(date -Is)] DONE  sanity_ik" >> "$PROGRESS" \
    || echo "[$(date -Is)] FAIL  sanity_ik" >> "$PROGRESS"

echo "=== phase6 prep driver finished $(date -Is) ===" >> "$PROGRESS"
