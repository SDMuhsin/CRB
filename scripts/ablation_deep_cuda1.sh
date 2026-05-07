#!/usr/bin/env bash
# Partition × Codebook-bits ablation — DEEP (full grid) — GPU 1.
#
# Runs 6 cells sequentially on cuda:1, all on meta-llama/Llama-3.2-1B:
#   partition ∈ {1, 3} × codebook_bits ∈ {2, 3, 4}    (K = 4, 8, 16)
#
# Each cell: full quantization + 7-task eval (3 PPL + 4 downstream).
# Expected wall time: ~50 min/cell × 6 = ~5 h total. (Llama-3.2-1B is ~1.5×
# the params of Qwen3-0.6B; quant + eval scales roughly linearly.)
#
# Output:
#   results/partition_K_ablation_llama3_1b.csv          (CSV — separate from main paper matrix)
#   logs/partition_K_ablation/cuda1_*.log               (per-cell logs)
#
# Usage:
#   nohup bash scripts/ablation_deep_cuda1.sh > logs/partition_K_ablation/cuda1_master.log 2>&1 &
#
# Pre-flight: confirm Llama-3.2-1B is in the cache. If not, run
# `bash sbatch/download_cache.sh` first (or hf-download manually) since
# the dev box is online and can fetch the model.

set -euo pipefail

cd "$(dirname "$0")/.."
source env/bin/activate

GPU_ID=1
DEVICE="cuda:0"  # CUDA_VISIBLE_DEVICES restricts visibility; from Python it's cuda:0
LOGDIR=./logs/partition_K_ablation
CSV=./results/partition_K_ablation_llama3_1b.csv
SEED=0
mkdir -p "$LOGDIR"
mkdir -p ./results

# Cells for this GPU (model, partition, codebook_bits). Sequential.
CELLS=(
    "meta-llama/Llama-3.2-1B 1 2"
    "meta-llama/Llama-3.2-1B 3 2"
    "meta-llama/Llama-3.2-1B 1 3"
    "meta-llama/Llama-3.2-1B 3 3"
    "meta-llama/Llama-3.2-1B 1 4"
    "meta-llama/Llama-3.2-1B 3 4"
)

GLOBAL_START=$(date +%s)
echo "[cuda${GPU_ID}] $(date +%Y-%m-%dT%H:%M:%S) starting DEEP ablation on cuda:${GPU_ID}"
echo "[cuda${GPU_ID}] cells: ${#CELLS[@]}"
echo "[cuda${GPU_ID}] csv:   $CSV"
echo "[cuda${GPU_ID}] logs:  $LOGDIR"

for cell in "${CELLS[@]}"; do
    read -r MODEL PARTITION CB <<< "$cell"
    K=$((1 << CB))
    if [[ "$PARTITION" == "1" ]]; then
        if [[ "$CB" == "2" ]]; then TAG="doml-p1"; else TAG="doml-p1-${CB}bit"; fi
    else
        if [[ "$CB" == "2" ]]; then TAG="doml"; else TAG="doml-${CB}bit"; fi
    fi
    SHORTNAME="$(echo "$MODEL" | tr '/' '_')"
    LOG="${LOGDIR}/cuda${GPU_ID}_${SHORTNAME}_${TAG}.log"

    # Skip if this cell already has >=7 rows in the target CSV.
    if [[ -f "$CSV" ]]; then
        n=$(awk -F, -v m="$MODEL" -v t="$TAG" -v s="$SEED" '$2==m && $3==t && $8==s {c++} END{print c+0}' "$CSV" 2>/dev/null || echo 0)
        if [[ "$n" -ge 7 ]]; then
            echo "[cuda${GPU_ID}] $(date +%H:%M:%S) SKIP $TAG on $MODEL — already has $n rows"
            continue
        fi
    fi

    echo "[cuda${GPU_ID}] $(date +%H:%M:%S) RUN  $TAG on $MODEL (partition=$PARTITION codebook_bits=$CB K=$K) -> $LOG"
    CELL_START=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    BILLM_BENCH_CSV="$CSV" \
    python3 -u run.py "$MODEL" wikitext2 doml \
        --partition "$PARTITION" \
        --codebook_bits "$CB" \
        --blocksize 128 \
        --salient_metric magnitude \
        --device="$DEVICE" \
        --seed "$SEED" \
        --full_eval > "$LOG" 2>&1
    CELL_RC=$?
    CELL_WALL=$(( $(date +%s) - CELL_START ))
    if [[ "$CELL_RC" == "0" ]]; then
        echo "[cuda${GPU_ID}] $(date +%H:%M:%S) DONE $TAG on $MODEL  rc=$CELL_RC  wall=${CELL_WALL}s"
    else
        echo "[cuda${GPU_ID}] $(date +%H:%M:%S) FAIL $TAG on $MODEL  rc=$CELL_RC  wall=${CELL_WALL}s  see $LOG"
    fi
done

GLOBAL_WALL=$(( $(date +%s) - GLOBAL_START ))
echo
echo "[cuda${GPU_ID}] $(date +%Y-%m-%dT%H:%M:%S) ablation complete (cuda:${GPU_ID})"
echo "[cuda${GPU_ID}] total wall: ${GLOBAL_WALL}s = $(( GLOBAL_WALL / 60 ))m $(( GLOBAL_WALL % 60 ))s"
if [[ -f "$CSV" ]]; then
    rows=$(( $(wc -l < "$CSV") - 1 ))
    echo "[cuda${GPU_ID}] csv rows: $rows  ($CSV)"
fi
