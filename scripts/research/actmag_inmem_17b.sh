#!/bin/bash
# actmagIM-17b: CRB_SALIENT_METRIC=actmag in-memory build+eval on Qwen3-1.7B.
# Isolation experiment: AWQ *partition realignment* WITHOUT value reshaping —
# salient columns ranked by s_j * sum_i|W_ij| where s = AWQ alpha=0.5 scales
# computed from calib but NOT folded into weights/norms (weights stay raw).
# Counterpart of awq05IM-17b (which applies the full AWQ transform).
# Launch: nohup bash scripts/research/actmag_inmem_17b.sh & (survives exit).
set -u
cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# EXACTLY this slice — all other slices carry live campaign pipelines.
export CUDA_VISIBLE_DEVICES=MIG-475dbed1-782f-5c45-ae0c-1d2507268638
export CRB_SALIENT_METRIC=actmag
export CRB_ACTMAG_ALPHA=0.5

LOG=/scratch/ckp908/crb/logs/rerun
D="$LOG/pipe_actmagIM-17b.drv"

_drv() {  # append one line to the drv; retry once on NFS wobble (no spin)
    echo "$1" >> "$D" || { sleep 10; echo "$1" >> "$D"; }
}

_drv "### INMEM actmagIM-17b START $(date '+%H:%M:%S') salient=actmag alpha=0.5 model=Qwen/Qwen3-1.7B lam=16e-5 ###"
t0=$SECONDS
BILLM_BENCH_CSV=$LOG/actmagIM-17b_eval.csv python -u kernels/pack/doml_group_refit.py \
    --run --model Qwen/Qwen3-1.7B --g 256 --codebook-dtype float8_e4m3fn \
    --cb-weight hdiag --intra-block-gptq --refit-iters 2 --bulk-k 2 \
    --rd-split 16e-5 --eval-extra-ppl --full-eval \
    > "$LOG/actmagIM-17b_eval.log" 2>&1
rc=$?
el=$((SECONDS - t0))
_drv "eval    rc=$rc ${el}s $(date '+%H:%M:%S')"
_drv "### INMEM actmagIM-17b DONE $(date '+%H:%M:%S') ###"
exit $rc
