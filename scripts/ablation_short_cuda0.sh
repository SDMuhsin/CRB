#!/usr/bin/env bash
# Partition × Codebook-bits ablation — SHORT (1h smoke test) — GPU 0.
#
# Runs 2 cells sequentially on cuda:0:
#   1. Qwen3-0.6B  partition=1  codebook_bits=2  (K=4, single-codebook DOML)
#   2. Qwen3-0.6B  partition=3  codebook_bits=2  (K=4, structural-partition DOML — paper baseline)
#
# Each cell: full quantization + 7-task eval (3 PPL + 4 downstream).
# Expected wall time: ~32 min/cell × 2 = ~65 min total.
#
# Output:
#   results/partition_K_ablation_qwen3_06b_short.csv    (CSV — separate from main paper matrix)
#   logs/partition_K_ablation_short/cuda0_*.log         (per-cell logs + master log)
#
# Usage:
#   nohup bash scripts/ablation_short_cuda0.sh > logs/partition_K_ablation_short/cuda0_master.log 2>&1 &
#
# Existing main paper CSVs are NOT touched. Patch in run.py adds the new
# 'doml-p1' tag for partition=1 only — default partition=3 still emits the
# legacy 'doml' tag with bpw=2.09.

set -euo pipefail

cd "$(dirname "$0")/.."
source env/bin/activate

GPU_ID=0
DEVICE="cuda:0"
LOGDIR=./logs/partition_K_ablation_short
CSV=./results/partition_K_ablation_qwen3_06b_short.csv
SEED=0
mkdir -p "$LOGDIR"
mkdir -p ./results

# Cells for this GPU (model, partition, codebook_bits). Sequential.
CELLS=(
    "Qwen/Qwen3-0.6B 1 2"
    "Qwen/Qwen3-0.6B 3 2"
)

GLOBAL_START=$(date +%s)
echo "[cuda${GPU_ID}] $(date +%Y-%m-%dT%H:%M:%S) starting SHORT ablation on cuda:${GPU_ID}"
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
