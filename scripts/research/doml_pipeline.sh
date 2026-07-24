#!/usr/bin/env bash
# Full DOML K31 pipeline on one MIG slice: build -> btune -> atune(pair) -> bpw -> eval.
# Usage: doml_pipeline.sh <MIG_UUID> <HF_MODEL> <lambda> <DUMP_DIR_rel>
# Caller may export CRB_SALIENT_METRIC=hessian (or other CRB_* knobs) to vary the build.
# Sequential, no polling. Stops if the build fails (nothing to tune).
set +e
cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF="$2"; LAM="$3"; DD="$4"; BATCH="${5:-8}"; SCHUNK="${6:-8}"
LOG=/scratch/ckp908/crb/logs/rerun
tag=$(basename "$DD")
D="$LOG/pipe_${tag}.drv"
echo "### PIPE $tag START $(date '+%H:%M:%S') slice=${1:4:8} salient=${CRB_SALIENT_METRIC:-magnitude} batch=$BATCH schunk=$SCHUNK ###" >> "$D"

SECONDS=0
python -u kernels/pack/doml_group_refit.py --run --model "$HF" --g 256 --dump-dir "$DD" \
    --codebook-dtype float8_e4m3fn --cb-weight hdiag --intra-block-gptq --refit-iters 2 \
    --bulk-k 2 --rd-split "$LAM" > "$LOG/${tag}_build.log" 2>&1
rc=$?; echo "build   rc=$rc ${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
if [ "$rc" -ne 0 ]; then echo "!!! build failed; stop" >> "$D"; exit 1; fi

SECONDS=0
python -u kernels/pack/k31_block_tune.py --src "$DD" --batch "$BATCH" --stream-chunk "$SCHUNK" \
    > "$LOG/${tag}_btune.log" 2>&1
rc=$?; echo "btune   rc=$rc ${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
if [ "$rc" -ne 0 ]; then echo "!!! btune failed; stop" >> "$D"; exit 1; fi

SECONDS=0
python -u kernels/pack/k31_assign_tune.py --src "${DD}-btuned" --mode pair --batch "$BATCH" --stream-chunk "$SCHUNK" \
    > "$LOG/${tag}_atune.log" 2>&1
rc=$?; echo "atune   rc=$rc ${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
if [ "$rc" -ne 0 ]; then echo "!!! atune failed; stop" >> "$D"; exit 1; fi

python kernels/pack/k29_honest_bpw.py --dir "${DD}-atuned" > "$LOG/${tag}_bpw.log" 2>&1
echo "bpw     rc=$? $(grep -i 'HONEST bpw' "$LOG/${tag}_bpw.log" | tail -1)" >> "$D"

SECONDS=0
BILLM_BENCH_CSV="$LOG/${tag}_eval.csv" python -u kernels/pack/doml_group_refit.py --run \
    --restore-dpk "${DD}-atuned" --eval-extra-ppl --full-eval > "$LOG/${tag}_eval.log" 2>&1
rc=$?; echo "eval    rc=$rc ${SECONDS}s $(date '+%H:%M:%S')" >> "$D"
echo "### PIPE $tag DONE $(date '+%H:%M:%S') ###" >> "$D"
