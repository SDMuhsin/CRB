#!/bin/bash
# ============================================================================
# ARC-4B campaign — honest-DOML DOWNSTREAM at small scales (Nibi)
#   Qwen3-0.6B / Llama-3.2-1B / Qwen3-1.7B
# ============================================================================
#
# Sister script to run_qwen_4b_benchmark.sh / run_arc4b_graft_grid.sh — SAME
# module order, cache-root logic, offline env, GRES names, sanity gates.
#
# Purpose: the April per-model CSVs already hold TesseraQ (paper-exact
# 250-iter) downstream rows at these scales; the HONEST K31 DOML containers
# have PPL-only coverage. Each job here regenerates the ship container
# CLUSTER-SIDE (no dump rsync needed) and runs the full downstream suite:
#     quantize (K31 ship recipe: g256 fp8 cb + hdiag + intra-block GPTQ +
#               refit-iters 2 + bulk-K2 + rd-split λ)
#  -> stage-1 btune (levels-only, 300 steps lr 1e-2 batch 8)
#  -> stage-2 atune (pair, reg-frac 0.05, drift-max 0, retune 150 @ 1e-2)
#     [ship recipe: output-aware tuning HELPS at <=1.7B; the 4B sign-flip
#      does not apply at these scales]
#  -> honest-bpw gate (k29_honest_bpw; expected value printed per job)
#  -> restore-eval: wt2/c4/ptb PPL + ARC-E/C + MMLU + HellaSwag
# appending eval rows to results/arc4b_hpc.csv.
#
# λ per model: 0.6B 7e-5 (ship 2.2299) · 1.7B 1.6e-4 (ship 2.1997) ·
# Llama-3.2-1B 1e-4 (INTERPOLATED — first K31 run on Llama; honest bpw is
# measured in-job and gated by log inspection, expected ~2.1-2.3).
#
# PREREQUISITES ON NIBI: branch code synced via git (kernels/pack/ with the
# model-parameterized K31 tools + --eval-arc/--eval-mmlu/--eval-hellaswag
# restore flags). Model snapshots must already be in $SCRATCH/billm2_cache
# (they are — the April jobs used them; gates below re-verify).
#
# Usage:
#   ./sbatch/run_arc4b_scaleout_downstream.sh                  # submit all 3
#   ./sbatch/run_arc4b_scaleout_downstream.sh --account def-foo
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

CSV_NAME="arc4b_hpc.csv"
CSV_ABS="$(pwd)/results/$CSV_NAME"

# Nibi GRES (full long-form names required; 4g.40gb is NOT provisioned).
# Full H100 for all three: the 1.7B tune stages measured 29-35 GB on A40
# (too tight for 3g.40gb); 0.6B/1B could fit smaller slices but uniform
# full-GPU keeps the chain simple and the queue behavior predictable.
GPU_FULL="--gres=gpu:h100:1"                              # full 80 GB H100

# model | short | snapshot_dirname | lambda | dump_name | time | expected_bpw
variants=(
    "Qwen/Qwen3-0.6B|qwen3_06b|models--Qwen--Qwen3-0.6B|7e-5|qwen3-0.6b/k31-rdsplit-lam7e-5-g256|4:00:00|2.2299"
    "meta-llama/Llama-3.2-1B|llama3_1b|models--meta-llama--Llama-3.2-1B|1e-4|llama3-1b/k31-rdsplit-lam1e-4-g256|4:30:00|~2.1-2.3 (first Llama K31 run — measured in-job)"
    "Qwen/Qwen3-1.7B|qwen3_17b|models--Qwen--Qwen3-1.7B|1.6e-4|qwen3-1.7b/k31-rdsplit-lam16e-5-g256|6:30:00|2.1997"
)

mkdir -p ./logs ./results

account_line=""
if [[ -n "$ACCOUNT" ]]; then
    account_line="#SBATCH --account=$ACCOUNT"
fi

job_count=0
for spec in "${variants[@]}"; do
    IFS='|' read -r model model_short snap_dir lam dump_rel time_limit expected_bpw <<< "$spec"

    job_name="arc4b_ds_${model_short}"
    base_dump="downloads/doml_dumps/${dump_rel}"
    atuned_dump="${base_dump}-atuned"
    gpu_resource="$GPU_FULL"
    cpus=10
    mem="96G"

    # Four-stage chain; && so any failure stops the job with rc!=0.
    # NOTE: if the base dump already exists with 252/196 wq files (resubmit
    # after a partial run), the quantize stage is skipped.
    python_cmd="if [ -f $atuned_dump/manifest.json ]; then \\
        echo 'atuned dump exists — skipping quantize+tunes'; \\
    else \\
        python -u kernels/pack/doml_group_refit.py --run --g 256 \\
            --codebook-dtype float8_e4m3fn --cb-weight hdiag \\
            --intra-block-gptq --refit-iters 2 --bulk-k 2 \\
            --rd-split $lam --model $model --dump-dir $base_dump \\
        && python -u kernels/pack/k31_block_tune.py --src $base_dump \\
            --model $model --steps 300 --lr 1e-2 --batch 8 \\
        && python -u kernels/pack/k31_assign_tune.py --src ${base_dump}-btuned \\
            --model $model --mode pair --reg-frac 0.05 --drift-max 0 \\
            --retune-steps 150 --retune-lr 1e-2; \\
    fi \\
    && python kernels/pack/k29_honest_bpw.py --dir $atuned_dump \\
       | tee /dev/stderr | grep 'HONEST bpw' \\
    && echo \"GATE: expected bpw $expected_bpw — verify the line above\" \\
    && python -u kernels/pack/doml_group_refit.py --run \\
        --restore-dpk $atuned_dump --model $model \\
        --eval-extra-ppl --eval-arc --eval-mmlu --eval-hellaswag"

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
echo "Model:      $model"
echo "Lambda:     $lam"
echo "Base dump:  $base_dump"
echo "Eval dump:  $atuned_dump"
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
ls -d "\$BILLM_DOWNLOADS_DIR"/$snap_dir 2>/dev/null || {
    echo "FATAL: no $snap_dir snapshot under \$BILLM_DOWNLOADS_DIR — run ./sbatch/download_cache.sh first."
    exit 1
}
for _refs in "\$BILLM_DOWNLOADS_DIR"/$snap_dir/refs/main; do
    [ -s "\$_refs" ] || {
        echo "FATAL: \$_refs missing or empty."
        exit 1
    }
done
# Campaign-specific gate: branch code must be synced (git).
python kernels/pack/doml_group_refit.py --help 2>/dev/null | grep -q "rd-split" || {
    echo "FATAL: doml_group_refit.py has no --rd-split — pull the arc-downstream-rca branch code."
    exit 1
}
python kernels/pack/doml_group_refit.py --help 2>/dev/null | grep -q "eval-mmlu" || {
    echo "FATAL: doml_group_refit.py has no --eval-mmlu — pull the arc-downstream-rca branch code."
    exit 1
}
mkdir -p "\$(dirname $base_dump)"
nvidia-smi || true
$python_cmd
echo '========================================'
echo "Finished:   \$(date)"
echo '========================================'
EOF
)
    echo "  [$sbatch_id] $job_name  (λ=$lam, t=${time_limit}, gpu=${gpu_resource#--gres=gpu:})"
    ((job_count++))
done

echo ""
echo "Submitted $job_count jobs. Results: logs/arc4b_ds_*_%j.{out,err} + $CSV_ABS"
echo "TesseraQ downstream bars for comparison already sit in the April CSVs:"
echo "  results/qwen3_06b_ptq_benchmark.csv / qwen3_1.7b_ / llama3_1b_"
