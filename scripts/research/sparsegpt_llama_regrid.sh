#!/bin/bash
# Same-harness rerun of the draft's SparseGPT Llama-3.2-1B rows.
# WHY: fp16-Llama-3.2-1B_eval.csv (2026-07-25) proves the draft's llama wt2/c4
# cells are ~2x off (draft 20.22/32.00 vs harness 9.75/14.02) while ptb+downstream
# match exactly -> April SparseGPT-llama PPL cells inherit the broken path.
# Usage: sparsegpt_llama_regrid.sh <slice-uuid> <jobfile>
#   jobfile lines: <sparsity> <nbits> <prunen> <prunem>   (nbits 16 = no quant)
set -u
cd "$(dirname "$0")/../.."
source scripts/env/env_common.sh

SLICE="$1"; JOBFILE="$2"
MODEL="meta-llama/Llama-3.2-1B"
LOGDIR=/scratch/ckp908/crb/logs/rerun
DRV="$LOGDIR/pipe_sparsegpt-llama.drv"
mkdir -p "$LOGDIR"

while read -r SPARSITY NBITS PRUNEN PRUNEM; do
    [[ -z "${SPARSITY:-}" || "${SPARSITY:0:1}" == "#" ]] && continue
    SP=$(awk "BEGIN{printf \"%d\", $SPARSITY*100}")
    if [[ "$PRUNEN" != "0" ]]; then TAG="sparsegpt-${PRUNEN}to${PRUNEM}-w${NBITS}-Llama-3.2-1B"
    elif [[ "$NBITS" == "16" ]]; then TAG="sparsegpt-s${SP}-Llama-3.2-1B"
    else TAG="sparsegpt-s${SP}-w${NBITS}-Llama-3.2-1B"; fi
    if grep -q "eval rc=0 ${TAG} " "$DRV" 2>/dev/null; then
        echo "skip ${TAG}"; continue
    fi
    EXTRA=""
    [[ "$NBITS" != "16" ]] && EXTRA="--nbits $NBITS"
    [[ "$PRUNEN" != "0" ]] && EXTRA="$EXTRA --prunen $PRUNEN --prunem $PRUNEM"
    echo "### ${TAG} START $(date +%T)" >> "$DRV"
    T0=$SECONDS
    BILLM_BENCH_CSV="$LOGDIR/${TAG}_eval.csv" \
    CUDA_VISIBLE_DEVICES="MIG-${SLICE}" \
    python3 -u src/run_sparsegpt.py "$MODEL" wikitext2 \
        --sparsity "$SPARSITY" $EXTRA \
        --percdamp 0.01 --true_sequential --calib_dataset c4 \
        --nsamples 128 --seqlen 2048 --seed 0 --device cuda:0 --full_eval \
        > "$LOGDIR/${TAG}.log" 2>&1
    echo "eval rc=$? ${TAG} elapsed=$((SECONDS-T0))s $(date +%T)" >> "$DRV"
done < "$JOBFILE"
echo "### worker ${SLICE:0:8} ${JOBFILE##*/} DONE $(date +%T)" >> "$DRV"
