#!/bin/bash
# ============================================================================
# ARC-4B improvement campaign — G4 graft LEVER GRID on Qwen3-4B (Nibi)
# ============================================================================
#
# Sister script to run_qwen_4b_benchmark.sh — SAME module order, cache-root
# logic, offline env, GRES names, and sanity-gate structure. Dispatches one
# sbatch job per graft variant; each job chains
#     tune (k31_assign_tune.py, PRA graft)
#  -> honest-bpw gate (k29_honest_bpw.py; MUST print the expected bpw)
#  -> restore-eval wt2 + ARC (doml_group_refit.py --restore-dpk --eval-arc)
# and appends eval rows to results/arc4b_hpc.csv.
#
# PREREQUISITES ON NIBI (sync from the dev box FIRST — jobs hard-fail
# without them, mirroring the model-snapshot gate):
#   1. Branch code (arc-downstream-rca working tree): kernels/pack/*.py
#      (k31_assign_tune.py with --pra-stages/--nsamples, k31_block_tune.py
#      with nsamples param, doml_group_refit.py with --eval-arc).
#   2. Source dumps (8.3 GB each) under the job-visible ./downloads:
#        downloads/doml_dumps/qwen3-4b/k31-rdsplit-lam3e-4-g256/
#        downloads/doml_dumps/qwen3-4b/k31-rdsplit-lam3e-4-g256-g3downk4/
#      NOTE: on Nibi ./downloads symlinks to $SCRATCH/billm2_cache/downloads
#      (see cache block below) — rsync the dumps THERE, i.e.
#        $SCRATCH/billm2_cache/downloads/doml_dumps/qwen3-4b/
#
# Jobs (all FULL H100 — measured A40 peak 39.4 GB rules out 3g.40gb):
#   g4x-scalepar   ship container, steps 2400, pra 20, lr_lev 1e-2
#                  (TesseraQ scale-optimization parity — its scale_lr is
#                  co-equal with rounding; prior G4 ran the analog at 1e-3)
#   g3g4-scalepar  G3 down-K4 container, steps 2400, pra 20, lr_lev 1e-2
#                  (composition x scale-parity)
#   g3g4-4800      G3 down-K4 container, steps 4800, pra 20, lr_lev 1e-3
#                  (composition x full dose)
#   g4-h03         ship container, steps 2400, pra 20, h0 0.3
#                  (less stay-biased flip init)
#
# Walltime: measured A40 graft = 7.3 h @ 2400 steps; full H100 = ~7x A40
# (April TQ 4B: 5.1 h Nibi vs 39 h A40-projected). 2400-step chain ≈ 1.8 h
# ⇒ x1.75 margin ⇒ request 4:30 (b2 tier; deliberately off the 3 h b1
# boundary — truncation loses everything, no mid-run checkpoints).
# 4800-step chain ≈ 2.9 h ⇒ request 6:30 (b2).
#
# Usage:
#   ./sbatch/run_arc4b_graft_grid.sh                    # submit all 4
#   ./sbatch/run_arc4b_graft_grid.sh --account def-foo
#
# ============================================================================

ACCOUNT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL="Qwen/Qwen3-4B"
MODEL_SHORT="qwen3_4b"

CSV_NAME="arc4b_hpc.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES (full long-form names required; 4g.40gb is NOT provisioned).
GPU_FULL="--gres=gpu:h100:1"                              # full 80 GB H100

SHIP_DUMP="downloads/doml_dumps/qwen3-4b/k31-rdsplit-lam3e-4-g256"
G3_DUMP="downloads/doml_dumps/qwen3-4b/k31-rdsplit-lam3e-4-g256-g3downk4"

# variant  src_dump  out_suffix       steps  lr_lev  h0    time    expected_bpw
variants=(
    "g4x-scalepar|$SHIP_DUMP|g4x-scalepar|2400|1e-2|0.1|4:30:00|2.1143"
    "g3g4-scalepar|$G3_DUMP|g3g4-scalepar|2400|1e-2|0.1|4:30:00|2.2952"
    "g3g4-4800|$G3_DUMP|g3g4-4800|4800|1e-3|0.1|6:30:00|2.2952"
    "g4-h03|$SHIP_DUMP|g4-h03|2400|1e-3|0.3|4:30:00|2.1143"
)

mkdir -p ./logs ./results

account_line=""
if [[ -n "$ACCOUNT" ]]; then
    account_line="#SBATCH --account=$ACCOUNT"
fi

job_count=0
for spec in "${variants[@]}"; do
    IFS='|' read -r variant src_dump out_suffix steps lr_lev h0 time_limit expected_bpw <<< "$spec"

    job_name="arc4b_${variant}_${MODEL_SHORT}"
    out_dump="downloads/doml_dumps/qwen3-4b/k31-rdsplit-lam3e-4-g256-${out_suffix}"
    gpu_resource="$GPU_FULL"
    cpus=10
    mem="96G"

    # The three-stage chain. && so any failure stops the job with rc!=0.
    python_cmd="python -u kernels/pack/k31_assign_tune.py \\
        --src $src_dump --orig $src_dump --out $out_dump \\
        --model $MODEL --mode pair --nsamples 512 --pra-stages 20 \\
        --steps $steps --lr-lev $lr_lev --h0 $h0 --batch 4 \\
        --stream-chunk 2 --log-every 800 \\
    && python kernels/pack/k29_honest_bpw.py --dir $out_dump \\
       | tee /dev/stderr | grep 'HONEST bpw' \\
    && echo \"GATE: expected bpw $expected_bpw — verify the line above matches EXACTLY (zero-bit for ship, mixed-K for g3)\" \\
    && python -u kernels/pack/doml_group_refit.py --run \\
        --restore-dpk $out_dump --model $MODEL --eval-arc"

    sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=$time_limit
#SBATCH $gpu_resource
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$mem
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate
# Do NOT set PYTHONNOUSERSITE=1 — the venv legitimately borrows packages
# from \$HOME/.local/. The torchvision/torchaudio ABI crash was fixed in
# ./sbatch/fix_venv_torchvision.sh.

if [[ -n "\${SCRATCH:-}" && -d "\${SCRATCH}" ]]; then
    CACHE_ROOT="\$SCRATCH/billm2_cache"
else
    CACHE_ROOT="\$(pwd)/downloads"
fi

if [[ "\$CACHE_ROOT" != "\$(pwd)/downloads" && ! -L ./downloads && ! -e ./downloads ]]; then
    ln -sfn "\$CACHE_ROOT" ./downloads
fi

export BILLM_DOWNLOADS_DIR="\$CACHE_ROOT/downloads"
export HF_HOME="\$CACHE_ROOT/hf"
export HF_HUB_CACHE="\$BILLM_DOWNLOADS_DIR"
export HF_DATASETS_CACHE="\$CACHE_ROOT/hf/datasets"
export TORCH_HOME="\$CACHE_ROOT/torch"
export HF_HUB_DISABLE_XET=1
export NUMBA_CACHE_DIR="\$CACHE_ROOT/.cache/numba"
export PIP_CACHE_DIR="\$CACHE_ROOT/.cache/pip"
export XDG_CACHE_HOME="\$CACHE_ROOT/.cache"
if [[ -n "\${SLURM_TMPDIR:-}" && -d "\${SLURM_TMPDIR}" ]]; then
    export TMPDIR="\$SLURM_TMPDIR"
else
    export TMPDIR="\$CACHE_ROOT/tmp"
fi
mkdir -p "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$HF_HUB_CACHE" "\$TORCH_HOME" \\
         "\$NUMBA_CACHE_DIR" "\$PIP_CACHE_DIR" "\$XDG_CACHE_HOME" "\$TMPDIR" \\
         "\$BILLM_DOWNLOADS_DIR"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export BILLM_BENCH_CSV="$CSV_ABS"

export PYTHONPATH=\$PYTHONPATH:\$(pwd):\$(pwd)/src

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo '========================================'
echo "Job:        $job_name"
echo "Model:      $MODEL"
echo "Variant:    $variant (steps=$steps lr_lev=$lr_lev h0=$h0)"
echo "Src dump:   $src_dump"
echo "Out dump:   $out_dump"
echo "Expected bpw: $expected_bpw"
echo "Time limit: $time_limit"
echo "GPU:        $gpu_resource"
echo "CPUs / Mem: $cpus / $mem"
echo "CSV:        \$BILLM_BENCH_CSV"
echo "Cache root: \$CACHE_ROOT (\$HF_HOME)"
echo "TMPDIR:     \$TMPDIR"
echo "Started:    \$(date)"
echo "SLURM job:  \$SLURM_JOB_ID"
echo '========================================'
echo "Python:             \$(which python)"
python --version
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
python -c "from transformers.models.qwen3 import Qwen3ForCausalLM; import transformers; print('transformers', transformers.__version__)" || {
    echo "FATAL: Qwen3ForCausalLM not importable — transformers is too old (<4.51)."
    exit 1
}
ls -d "\$BILLM_DOWNLOADS_DIR"/models--Qwen--Qwen3-4B 2>/dev/null || {
    echo "FATAL: no Qwen3-4B snapshot under \$BILLM_DOWNLOADS_DIR — run ./sbatch/download_cache.sh first."
    exit 1
}
for _refs in "\$BILLM_DOWNLOADS_DIR"/models--Qwen--Qwen3-4B/refs/main; do
    [ -s "\$_refs" ] || {
        echo "FATAL: \$_refs missing or empty."
        exit 1
    }
done
# Campaign-specific gate: source dump must be synced (252 wq + 252 dpk files).
_n_wq=\$(ls $src_dump/*.wq.safetensors 2>/dev/null | wc -l)
if [[ "\$_n_wq" != "252" ]]; then
    echo "FATAL: $src_dump has \$_n_wq wq files (need 252) — rsync the dump from the dev box (see script header)."
    exit 1
fi
# PRA graft flags must exist in the synced code (branch arc-downstream-rca).
python kernels/pack/k31_assign_tune.py --help 2>/dev/null | grep -q "pra-stages" || {
    echo "FATAL: k31_assign_tune.py has no --pra-stages — sync the arc-downstream-rca kernels/pack/ code."
    exit 1
}
nvidia-smi || true
$python_cmd
echo '========================================'
echo "Finished:   \$(date)"
echo '========================================'
EOF
)
    echo "  [$sbatch_id] $job_name  (steps=$steps lr_lev=$lr_lev h0=$h0, t=${time_limit}, gpu=${gpu_resource#--gres=gpu:})"
    ((job_count++))
done

echo ""
echo "Submitted $job_count jobs. Results: logs/arc4b_*_%j.{out,err} + $CSV_ABS"
echo "Bring back: the four out-dump dirs + logs + $CSV_NAME."
