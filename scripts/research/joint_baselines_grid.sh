#!/bin/bash
# Joint pruning+PTQ baseline grid (JSQ-wo / SLiM) — local MIG fan-out driver.
#
# Usage: joint_baselines_grid.sh <slice-uuid> <jobfile>
#   jobfile lines: <HF-model> <method> <sparsity> <nbits>
# Runs each line sequentially on the slice with the paper-parity protocol
# (calib c4 x 128 @ 2048, seed 0, full eval). Per-run log + per-run CSV under
# $LOGDIR; appends "eval rc=<rc> <tag>" lines to $LOGDIR/pipe_joint-baselines.drv
# (markers that don't pre-exist — watcher-safe).
set -u
cd "$(dirname "$0")/../.."
source scripts/env/env_common.sh

SLICE="$1"
JOBFILE="$2"
LOGDIR=${LOGDIR:-/scratch/ckp908/crb/logs/rerun}
DRV="$LOGDIR/pipe_joint-baselines.drv"
mkdir -p "$LOGDIR"

while read -r MODEL METHOD SPARSITY NBITS; do
    [[ -z "${MODEL:-}" || "${MODEL:0:1}" == "#" ]] && continue
    MS=$(basename "$MODEL"); SP=$(awk "BEGIN{printf \"%d\", $SPARSITY*100}")
    TAG="${METHOD}-s${SP}-w${NBITS}-${MS}"
    if grep -q "eval rc=0 ${TAG} " "$DRV" 2>/dev/null; then
        echo "skip ${TAG} (already done)"; continue
    fi
    echo "### ${TAG} START $(date +%T)" >> "$DRV"
    T0=$SECONDS
    BILLM_BENCH_CSV="$LOGDIR/${TAG}_eval.csv" \
    CUDA_VISIBLE_DEVICES="MIG-${SLICE}" \
    python3 -u src/run_joint_baselines.py "$MODEL" wikitext2 \
        --method "$METHOD" --sparsity "$SPARSITY" --nbits "$NBITS" \
        --calib_dataset c4 --nsamples 128 --seqlen 2048 --seed 0 \
        --device cuda:0 --full_eval \
        > "$LOGDIR/${TAG}.log" 2>&1
    RC=$?
    echo "eval rc=${RC} ${TAG} elapsed=$((SECONDS-T0))s $(date +%T)" >> "$DRV"
done < "$JOBFILE"
echo "### worker ${SLICE:0:8} jobfile ${JOBFILE##*/} DONE $(date +%T)" >> "$DRV"
