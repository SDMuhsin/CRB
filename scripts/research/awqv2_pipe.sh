#!/usr/bin/env bash
# AWQ v2 pipeline: awqfix_pipe.sh sequence + CRB_AWQ_V2=1 (o_proj/down_proj
# scaling folded into v_proj/up_proj rows pre-quantization; restore needs no
# extra fold). args: slice HF lambda DD batch schunk alpha
set +e; cd /workspace/CRB || exit 1
source ./scripts/env/env_common.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$1"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CRB_AWQ_V2=1
HF="$2"; LAM="$3"; DD="$4"; BATCH="${5:-8}"; SCHUNK="${6:-8}"; ALPHA="$7"
LOG=/scratch/ckp908/crb/logs/rerun; tag=$(basename "$DD"); D="$LOG/pipe_${tag}.drv"
echo "### AWQV2 $tag START $(date '+%H:%M:%S') alpha=$ALPHA batch=$BATCH ###" >> "$D"
# 1. build (saves awq_scales.safetensors + awq_v2_scales.safetensors)
CRB_AWQ_ALPHA="$ALPHA" python -u kernels/pack/doml_group_refit.py --run --model "$HF" --g 256 \
  --dump-dir "$DD" --codebook-dtype float8_e4m3fn --cb-weight hdiag --intra-block-gptq \
  --refit-iters 2 --bulk-k 2 --rd-split "$LAM" > "$LOG/${tag}_build.log" 2>&1
rc=$?; echo "build rc=$rc $(ls -1 $DD/awq_scales.safetensors 2>/dev/null && echo SCALES_SAVED) $(ls -1 $DD/awq_v2_scales.safetensors 2>/dev/null && echo V2_SCALES_SAVED) $(date '+%H:%M:%S')" >> "$D"
[ "$rc" -ne 0 ] && { echo "!!! build failed" >> "$D"; exit 1; }
# 2. EARLY GATE: raw restore-eval (restore pops CRB_AWQ_ALPHA/CRB_AWQ_V2; v1 norm fold from saved scales only)
CRB_AWQ_ALPHA="$ALPHA" BILLM_BENCH_CSV="$LOG/${tag}-rawrt_eval.csv" python -u kernels/pack/doml_group_refit.py \
  --run --restore-dpk "$DD" --eval-extra-ppl > "$LOG/${tag}-rawrt_eval.log" 2>&1
echo "rawRT rc=$? wt2=$(grep -oE 'wikitext2[^0-9]*[0-9.]+' $LOG/${tag}-rawrt_eval.csv 2>/dev/null|tail -1) $(date '+%H:%M:%S')" >> "$D"
# 3. btune 4. atune 5. bpw 6. tuned eval
python -u kernels/pack/k31_block_tune.py --src "$DD" --batch "$BATCH" --stream-chunk "$SCHUNK" > "$LOG/${tag}_btune.log" 2>&1
echo "btune rc=$? $(date '+%H:%M:%S')" >> "$D"
python -u kernels/pack/k31_assign_tune.py --src "${DD}-btuned" --mode pair --batch "$BATCH" --stream-chunk "$SCHUNK" > "$LOG/${tag}_atune.log" 2>&1
echo "atune rc=$? $(date '+%H:%M:%S')" >> "$D"
python kernels/pack/k29_honest_bpw.py --dir "${DD}-atuned" > "$LOG/${tag}_bpw.log" 2>&1
echo "bpw $(grep -i 'HONEST bpw' $LOG/${tag}_bpw.log|tail -1)" >> "$D"
BILLM_BENCH_CSV="$LOG/${tag}_eval.csv" python -u kernels/pack/doml_group_refit.py --run \
  --restore-dpk "${DD}-atuned" --eval-extra-ppl --full-eval > "$LOG/${tag}_eval.log" 2>&1
echo "eval  rc=$? $(date '+%H:%M:%S')" >> "$D"; echo "### AWQV2 $tag DONE $(date '+%H:%M:%S') ###" >> "$D"
