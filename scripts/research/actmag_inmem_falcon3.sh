#!/bin/bash
# actmagIM-falcon3: CRB_SALIENT_METRIC=actmag in-memory build+eval on Falcon3-1B.
# Falcon3 root-cause follow-up (2026-07-24 ~21:20): scaling at matched rate is
# REFUTED (awqfix05 lam48 @2.2492 loses 7/7 to TQ; frontier-neutral). Last
# cheap hypothesis: activation-aware SELECTION (salient columns ranked by
# s_j * sum_i|W_ij|, s from calib alpha=0.5, weights UNTOUCHED). Gate: raw wt2
# vs plain raw 12.22 and awq raw 10.99, both lam36e-5 builds. Full honest
# chain only if this shows a real shift.
set -u
cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=MIG-6ec7b494-8fdc-5531-8226-d8b3ea71838a
export CRB_SALIENT_METRIC=actmag
export CRB_ACTMAG_ALPHA=0.5

LOG=/scratch/ckp908/crb/logs/rerun
D="$LOG/pipe_actmagIM-falcon3.drv"

_drv() {  # append one line to the drv; retry once on NFS wobble (no spin)
    echo "$1" >> "$D" || { sleep 10; echo "$1" >> "$D"; }
}

_drv "### INMEM actmagIM-falcon3 START $(date '+%H:%M:%S') salient=actmag alpha=0.5 model=tiiuae/Falcon3-1B-Base lam=36e-5 ###"
t0=$SECONDS
BILLM_BENCH_CSV=$LOG/actmagIM-falcon3_eval.csv python -u kernels/pack/doml_group_refit.py \
    --run --model tiiuae/Falcon3-1B-Base --g 256 --codebook-dtype float8_e4m3fn \
    --cb-weight hdiag --intra-block-gptq --refit-iters 2 --bulk-k 2 \
    --rd-split 36e-5 --eval-extra-ppl --full-eval \
    > "$LOG/actmagIM-falcon3_eval.log" 2>&1
rc=$?
el=$((SECONDS - t0))
_drv "eval    rc=$rc ${el}s $(date '+%H:%M:%S')"
_drv "### INMEM actmagIM-falcon3 DONE $(date '+%H:%M:%S') ###"
exit $rc
