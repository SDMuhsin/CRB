#!/usr/bin/env bash
# Speed benchmarks for the ik_llama.cpp ladder (machine must be otherwise
# idle; run only after all PPL runs finished).
#   llama-bench -p 512 -n 128 -r 3 -t 1,6,12,24,48  (single pass per model:
#     -o json to stdout -> .json;  -oe md to stderr -> .md)
#   plus one NUMA-pinned point at -t 12: numactl --cpunodebind=0 ONLY
#     (--membind EPERM in this container; recorded numa=node0-cpubind)
#   plus one RUN-TIME-REPACK point at -t 24: -rtr 1 (ik-specific: repacks
#     tensors to interleaved _R4 layouts at load; recorded notes=rtr)
# Resumable: a phase is skipped when its .json exists non-empty. Missing
# GGUFs (failed quants) are skipped.
# Outputs: ikllamacpp_bench_<TAG>{,_numa0,_rtr}.json/.md
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IK_DIR="$ROOT/temp/ik_llama.cpp"
GGUF_DIR="$ROOT/downloads/cpu_baselines/ik_llama.cpp"
ML_DIR="$ROOT/downloads/cpu_baselines/llama.cpp"
LOG_DIR="$ROOT/llmdocs/cpu_kernel/verify/cpu_baseline_logs"
PROGRESS="$LOG_DIR/ikllamacpp_bench_progress.log"
BENCH="$IK_DIR/build/bin/llama-bench"

MODELS=(F16 Q4_K_M Q4_0 IQ4_KS IQ3_K Q2_K IQ2_KL IQ2_K IQ2_XS IQ2_KS IQ2_KT IQ2_XXS)

echo "=== ik bench sweep started $(date -Is) ===" >> "$PROGRESS"
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
    if [ ! -s "$LOG_DIR/ikllamacpp_bench_${TAG}.json" ]; then
        echo "[$(date -Is)] START $TAG threads-sweep" >> "$PROGRESS"
        "$BENCH" -m "$GGUF" -p 512 -n 128 -r 3 -t 1,6,12,24,48 -o json -oe md \
            > "$LOG_DIR/ikllamacpp_bench_${TAG}.json" 2> "$LOG_DIR/ikllamacpp_bench_${TAG}.md"
    fi
    if [ ! -s "$LOG_DIR/ikllamacpp_bench_${TAG}_numa0.json" ]; then
        echo "[$(date -Is)] START $TAG numa0-pinned t=12" >> "$PROGRESS"
        numactl --cpunodebind=0 \
            "$BENCH" -m "$GGUF" -p 512 -n 128 -r 3 -t 12 -o json -oe md \
            > "$LOG_DIR/ikllamacpp_bench_${TAG}_numa0.json" 2> "$LOG_DIR/ikllamacpp_bench_${TAG}_numa0.md"
    fi
    if [ ! -s "$LOG_DIR/ikllamacpp_bench_${TAG}_rtr.json" ]; then
        echo "[$(date -Is)] START $TAG rtr t=24" >> "$PROGRESS"
        "$BENCH" -m "$GGUF" -p 512 -n 128 -r 3 -t 24 -rtr 1 -o json -oe md \
            > "$LOG_DIR/ikllamacpp_bench_${TAG}_rtr.json" 2> "$LOG_DIR/ikllamacpp_bench_${TAG}_rtr.md" \
            || echo "[$(date -Is)] FAIL  $TAG rtr pass" >> "$PROGRESS"
    fi
    echo "[$(date -Is)] DONE  $TAG" >> "$PROGRESS"
done
echo "=== ik bench sweep finished $(date -Is) ===" >> "$PROGRESS"
