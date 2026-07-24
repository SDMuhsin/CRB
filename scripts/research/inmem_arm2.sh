#!/usr/bin/env bash
# in-memory K31 build+eval. $1=slice $2=tag $3=alpha(empty=plain) $4=HFmodel $5=lambda
set +e; cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$1"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "$3" ] && export CRB_AWQ_ALPHA="$3" || unset CRB_AWQ_ALPHA
LOG=/scratch/ckp908/crb/logs/rerun; TAG="$2"; D="$LOG/pipe_${TAG}.drv"
echo "### INMEM $TAG START $(date '+%H:%M:%S') alpha=${3:-none} model=$4 lam=$5 ###" >> "$D"
BILLM_BENCH_CSV="$LOG/${TAG}_eval.csv" python -u kernels/pack/doml_group_refit.py --run \
  --model "$4" --g 256 --codebook-dtype float8_e4m3fn --cb-weight hdiag \
  --intra-block-gptq --refit-iters 2 --bulk-k 2 --rd-split "$5" \
  --eval-extra-ppl --full-eval > "$LOG/${TAG}_eval.log" 2>&1
echo "eval    rc=$? $(date '+%H:%M:%S')" >> "$D"; echo "### INMEM $TAG DONE $(date '+%H:%M:%S') ###" >> "$D"
