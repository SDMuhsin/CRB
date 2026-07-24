#!/usr/bin/env bash
# 1.7B batch8 re-atune (recover the batch4 memory compromise) + restore-eval, on 48GB slice.
set +e
cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/scratch/ckp908/crb/logs/rerun
BT=downloads/doml_dumps/qwen3-1.7b/k31-rdsplit-lam16e-5-g256-btuned
OUT=downloads/doml_dumps/qwen3-1.7b/k31-rdsplit-lam16e-5-g256-atuned-b8
D="$LOG/rdrv_17b_b8.log"
echo ">>> atune-b8 START $(date '+%H:%M:%S')" >> "$D"
SECONDS=0
python -u kernels/pack/k31_assign_tune.py --src "$BT" --out "$OUT" \
    --mode pair --batch 8 --stream-chunk 8 --device cuda:0 > "$LOG/atune_17b_b8.log" 2>&1
rc=$?; echo "<<< atune-b8 rc=$rc elapsed=${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
if [ "$rc" -ne 0 ]; then echo "!!! atune failed; stop" >> "$D"; exit 1; fi
echo ">>> restore-eval START $(date '+%H:%M:%S')" >> "$D"
SECONDS=0
BILLM_BENCH_CSV="$LOG/eval_17b_b8.csv" python -u kernels/pack/doml_group_refit.py --run \
    --restore-dpk "$OUT" --eval-extra-ppl --full-eval > "$LOG/eval_17b_b8.log" 2>&1
echo "<<< restore-eval rc=$? elapsed=${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
echo "### DONE 17b_b8 $(date '+%H:%M:%S') ###" >> "$D"
