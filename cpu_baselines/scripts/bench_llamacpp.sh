#!/usr/bin/env bash
# Speed benchmarks for the llama.cpp ladder (machine must be otherwise idle;
# run only after all PPL runs finished).
#   llama-bench -p 512 -n 128 -r 3 -t 1,6,12,24,48  (single pass per model:
#     -o json to stdout -> .json;  -oe md to stderr -> .md, so both formats
#     come from the SAME run)
#   plus one NUMA-pinned point at -t 12: numactl --cpunodebind=0 ONLY.
#     (--membind=0 is NOT permitted in this container: set_mempolicy ->
#      EPERM. CPU binding to node0 + default first-touch policy is the best
#      available approximation; recorded as numa=node0-cpubind in the CSV.)
# Resumable: a model/phase is skipped when its .json already exists non-empty.
# Outputs: llamacpp_bench_<TAG>.json/.md and llamacpp_bench_<TAG>_numa0.json/.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT/temp/llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
PROGRESS="$LOG_DIR/llamacpp_bench_progress.log"
BENCH="$LLAMA_DIR/build/bin/llama-bench"

MODELS=(F16 Q4_K_M Q4_0 Q3_K_M Q2_K IQ2_M IQ2_XS IQ2_XXS IQ1_S)

echo "=== bench sweep started $(date -Is) ===" >> "$PROGRESS"
for TAG in "${MODELS[@]}"; do
    if [ "$TAG" = "F16" ]; then
        GGUF="$GGUF_DIR/qwen3-0.6b-f16.gguf"
    else
        GGUF="$GGUF_DIR/qwen3-0.6b-${TAG}.gguf"
    fi
    if [ ! -s "$LOG_DIR/llamacpp_bench_${TAG}.json" ]; then
        echo "[$(date -Is)] START $TAG threads-sweep" >> "$PROGRESS"
        "$BENCH" -m "$GGUF" -p 512 -n 128 -r 3 -t 1,6,12,24,48 -o json -oe md \
            > "$LOG_DIR/llamacpp_bench_${TAG}.json" 2> "$LOG_DIR/llamacpp_bench_${TAG}.md"
    fi
    if [ ! -s "$LOG_DIR/llamacpp_bench_${TAG}_numa0.json" ]; then
        echo "[$(date -Is)] START $TAG numa0-pinned t=12" >> "$PROGRESS"
        numactl --cpunodebind=0 \
            "$BENCH" -m "$GGUF" -p 512 -n 128 -r 3 -t 12 -o json -oe md \
            > "$LOG_DIR/llamacpp_bench_${TAG}_numa0.json" 2> "$LOG_DIR/llamacpp_bench_${TAG}_numa0.md"
    fi
    echo "[$(date -Is)] DONE  $TAG" >> "$PROGRESS"
done
echo "=== bench sweep finished $(date -Is) ===" >> "$PROGRESS"
