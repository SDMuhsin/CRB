#!/bin/bash
# ============================================================================
# diagnose_pb_llm_compute.sh — run on Nibi LOGIN NODE. Submits ONE small
# sbatch job (10 min, 1g.10gb) that reproduces PB-LLM's crashing call path
# on a compute node, in isolation, with no other cache-access contention.
#
# Why compute-side: round 1 and round 2 showed login cannot reproduce the
# LocalEntryNotFoundError that hit pb-llm jobs 12539877 and 12539889. Login
# AutoConfig.from_pretrained succeeded; on compute it failed. The login
# AutoModelForCausalLM call died on an unrelated mpmath ImportError before
# reaching the `cached_file` path, so we couldn't tell if the compute-side
# LocalEntryNotFoundError is deterministic or a transient concurrent-cache
# artifact. This job isolates the call.
#
# What the job does (inside the sbatch HEREDOC):
#   1. Print compute-node identity + module environment + env-var snapshot.
#   2. Walk the Qwen3-1.7B cache from the compute node's filesystem view:
#      refs/main, snapshot dir, every tracked file, blob link resolution.
#   3. Reproduce PB-LLM's `get_model("Qwen/Qwen3-1.7B")` — AutoConfig first,
#      then the full AutoModelForCausalLM.from_pretrained with PB-LLM's
#      exact kwargs. Prints success or full traceback.
#   4. If the AutoModel call fails, re-attempt with local_files_only=True
#      and with cache_dir pointed directly at the snapshot hash — narrows
#      whether the issue is HF hub resolution or the cache itself.
#
# Read-only. No cache mutation. No overlap with the actual benchmark queue.
#
# Usage (login node):
#   cd <repo root on Nibi>
#   ./sbatch/diagnose_pb_llm_compute.sh
#   # Submits one sbatch job; watch ./logs/diagnose_pb_llm_compute_<jid>.out
# ============================================================================

set -eu

REPO="$(pwd)"
ACCOUNT="${BILLM_ACCOUNT:-rrg-seokbum}"    # adjust if your default differs

if [[ ! -f "${REPO}/sbatch/run_qwen_1.7b_benchmark.sh" ]]; then
    echo "FATAL: run this from the BiLLM2 repo root."
    exit 1
fi

mkdir -p "${REPO}/logs"

# ----------------------------------------------------------------------------
# Submit the diagnostic job.
# ----------------------------------------------------------------------------

sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=diagnose_pb_llm_compute
#SBATCH --output=${REPO}/logs/diagnose_pb_llm_compute_%j.out
#SBATCH --error=${REPO}/logs/diagnose_pb_llm_compute_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --account=${ACCOUNT}

set -u

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

# Cache routing — mirrors the benchmark scripts exactly.
if [[ -n "\${SCRATCH:-}" && -d "\${SCRATCH}" ]]; then
    CACHE_ROOT="\$SCRATCH/billm2_cache"
else
    CACHE_ROOT="\$(pwd)/downloads"
fi
export BILLM_DOWNLOADS_DIR="\$CACHE_ROOT/downloads"
export HF_HOME="\$CACHE_ROOT/hf"
export HF_HUB_CACHE="\$BILLM_DOWNLOADS_DIR"
export HF_DATASETS_CACHE="\$CACHE_ROOT/hf/datasets"
export TORCH_HOME="\$CACHE_ROOT/torch"
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "========================================"
echo "Compute-side pb-llm reproduction"
echo "Host:     \$(hostname)"
echo "Started:  \$(date)"
echo "SLURM:    \$SLURM_JOB_ID"
echo "========================================"
echo ""
echo "--- env vars (what pb-llm sees) ---"
for v in BILLM_DOWNLOADS_DIR HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_DISABLE_XET; do
    echo "  \$v=\${!v:-<unset>}"
done

echo ""
echo "--- Qwen3-1.7B cache view from compute ---"
MDIR="\$BILLM_DOWNLOADS_DIR/models--Qwen--Qwen3-1.7B"
echo "  path: \$MDIR"
if [[ -d "\$MDIR" ]]; then
    ls -la "\$MDIR" | sed 's/^/    /'
    if [[ -f "\$MDIR/refs/main" ]]; then
        echo "    refs/main content: \$(cat \$MDIR/refs/main)"
    else
        echo "    refs/main: MISSING"
    fi
    for snap in "\$MDIR/snapshots/"*/; do
        echo "    snapshot \$(basename \$snap):"
        find "\$snap" -maxdepth 1 -mindepth 1 -printf '      %f (%y) -> ' \
             -exec readlink -f {} \; 2>&1 | head -20
    done
else
    echo "    NOT PRESENT on this compute node"
fi

echo ""
echo "--- Reproduction A: AutoConfig.from_pretrained (pb-llm inner call) ---"
python3 - <<'PYEOF'
import os, sys, traceback
cache_dir = os.environ["BILLM_DOWNLOADS_DIR"]
import transformers
print(f"  transformers {transformers.__version__}")
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained(
        "Qwen/Qwen3-1.7B", cache_dir=cache_dir,
        use_safetensors=True, attn_implementation="eager",
    )
    print(f"  OK: model_type={cfg.model_type}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc(limit=10)
PYEOF

echo ""
echo "--- Reproduction B: AutoModelForCausalLM.from_pretrained (full pb-llm call) ---"
python3 - <<'PYEOF'
import os, sys, traceback
cache_dir = os.environ["BILLM_DOWNLOADS_DIR"]
from transformers import AutoModelForCausalLM
try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype="auto",
        cache_dir=cache_dir,
        use_safetensors=True,
        attn_implementation="eager",
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"  OK: {n:,} params, dtype={model.dtype}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc(limit=12)
PYEOF

echo ""
echo "--- Reproduction C: with local_files_only=True (forces cache-only) ---"
python3 - <<'PYEOF'
import os, traceback
cache_dir = os.environ["BILLM_DOWNLOADS_DIR"]
from transformers import AutoModelForCausalLM
try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype="auto",
        cache_dir=cache_dir,
        use_safetensors=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"  OK: {n:,} params")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc(limit=10)
PYEOF

echo ""
echo "--- Reproduction D: load by direct snapshot path (bypasses HF hub resolver) ---"
python3 - <<'PYEOF'
import os, glob, traceback
cache_dir = os.environ["BILLM_DOWNLOADS_DIR"]
snap_glob = os.path.join(cache_dir, "models--Qwen--Qwen3-1.7B/snapshots/*/")
snaps = sorted(glob.glob(snap_glob))
print(f"  found {len(snaps)} snapshot(s): {snaps}")
if not snaps:
    print("  (no snapshot to test)")
else:
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            snaps[0],
            torch_dtype="auto",
            use_safetensors=True,
            attn_implementation="eager",
            local_files_only=True,
        )
        n = sum(p.numel() for p in model.parameters())
        print(f"  OK loaded from snapshot path: {n:,} params")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=10)
PYEOF

echo ""
echo "========================================"
echo "Finished: \$(date)"
echo "========================================"
EOF
)

echo "Submitted diagnostic job: ${sbatch_id}"
echo "Expected output: ./logs/diagnose_pb_llm_compute_${sbatch_id}.out"
echo ""
echo "When the job finishes (should be <10 min), rsync that .out and .err"
echo "back to the dev-box."
