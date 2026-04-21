#!/bin/bash
# ============================================================================
# diagnose_pb_llm_round2.sh — second round, Nibi LOGIN NODE.
#
# First round (diagnose_pb_llm_cache.sh) showed:
#   - cache structurally OK for all 4 Qwen3 sizes
#   - AutoConfig / AutoTokenizer offline: OK on login for all 4
#   - pb-llm still crashed on compute at AutoModelForCausalLM.from_pretrained
#   - 3 "instant FAILED" jobs (12539876, 12539886, 12539888) all ask for
#     nvidia_h100_80gb_hbm3_4g.40gb (GPU_XLARGE) with 00:00:00 elapsed
#   - 12539874 (GPU_LARGE = 3g.40gb) is RUNNING fine
#
# This round targets the two remaining unknowns:
#   A. Does AutoModelForCausalLM.from_pretrained work offline on login for
#      Qwen3-1.7B (sharded, needs model.safetensors.index.json), using the
#      exact call that pb-llm makes? If it also crashes here, cache is
#      missing a file the first scan didn't check for. If it works, the
#      compute-vs-login difference is environmental.
#   B. Does the nvidia_h100_80gb_hbm3_4g.40gb MIG slice actually exist on
#      the partition used for the failed jobs? Or is the GRES name wrong?
#      Also fetch the SLURM-side reason code for one of the FAILED jobs.
#
# Read-only. No job submission. No cache mutation.
#
# Usage:
#   cd <repo root on Nibi>
#   ./sbatch/diagnose_pb_llm_round2.sh 2>&1 | tee ./logs/diagnose_pb_llm_round2.log
# ============================================================================

set -u

echo "============================================================================"
echo "Section 0: env"
echo "============================================================================"
echo "Host: $(hostname)  Date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]]; then
    CACHE_ROOT="${SCRATCH}/billm2_cache"
else
    CACHE_ROOT="$(pwd)/downloads"
fi
DOWNLOADS_DIR="${CACHE_ROOT}/downloads"
echo "DOWNLOADS_DIR: ${DOWNLOADS_DIR}"

# ----------------------------------------------------------------------------
# Section 1: enumerate every file in each Qwen3 snapshot (find hidden misses)
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "Section 1: full file listing per Qwen3 snapshot (looking for index.json)"
echo "============================================================================"

for sz in 0.6B 1.7B 4B 8B; do
    dir="${DOWNLOADS_DIR}/models--Qwen--Qwen3-${sz}"
    [[ -d "${dir}/snapshots" ]] || { echo "--- ${sz}: no snapshots dir ---"; continue; }
    echo ""
    echo "--- Qwen3-${sz} ---"
    for snap in "${dir}/snapshots/"*/; do
        echo "  snapshot: $(basename "${snap}")"
        # List every file, whether it's a symlink, and whether the symlink
        # target resolves to something readable.
        find "${snap}" -maxdepth 1 -mindepth 1 -printf '    %f (%y)\n' 2>/dev/null | sort
        # Explicit checks for the sharded-model index file.
        for candidate in model.safetensors.index.json pytorch_model.bin.index.json; do
            p="${snap}${candidate}"
            if [[ -e "${p}" ]]; then
                real=$(readlink -f "${p}" 2>/dev/null || echo "?")
                size=$(stat -c '%s' "${real}" 2>/dev/null || echo "?")
                echo "    CHECK ${candidate}: present (${size} bytes)"
            else
                # Count .safetensors files to know if index is needed.
                nshards=$(ls "${snap}"*.safetensors 2>/dev/null | wc -l)
                if (( nshards > 1 )) && [[ "${candidate}" == model.safetensors.index.json ]]; then
                    echo "    CHECK ${candidate}: MISSING (but ${nshards} safetensors shards → index REQUIRED)"
                fi
            fi
        done
    done
done

# ----------------------------------------------------------------------------
# Section 2: exact PB-LLM get_model call, offline, on login, for Qwen3-1.7B
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "Section 2: PB-LLM get_model() reproduction, offline, login"
echo "============================================================================"
echo "This invokes AutoModelForCausalLM.from_pretrained with the EXACT kwargs"
echo "that PB-LLM/gptq_pb/run.py:43 uses. If it fails here, a file is missing"
echo "from the cache that the Section 1 scan didn't find. If it works here,"
echo "the compute-vs-login difference is environmental."
echo ""
echo "Only Qwen3-0.6B and Qwen3-1.7B attempted — 4B/8B would exceed the 16 GB"
echo "login cgroup. 1.7B in fp16 is ~4 GB, fits."

if command -v module >/dev/null 2>&1; then
    module load gcc arrow scipy-stack cuda cudnn python/3.11 2>&1 | sed 's/^/  module: /' || true
fi
if [[ -f ./env/bin/activate ]]; then
    # shellcheck disable=SC1091
    source ./env/bin/activate
fi

export BILLM_DOWNLOADS_DIR="${DOWNLOADS_DIR}"
export HF_HOME="${CACHE_ROOT}/hf"
export HF_HUB_CACHE="${DOWNLOADS_DIR}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf/datasets"
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python3 - <<'PYEOF'
import os, sys, traceback
cache_dir = os.environ["BILLM_DOWNLOADS_DIR"]

print(f"python {sys.version.split()[0]}")
import transformers
print(f"transformers {transformers.__version__}")
from transformers import AutoModelForCausalLM

for m in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]:
    print(f"\n=== {m} (AutoModelForCausalLM.from_pretrained, PB-LLM kwargs) ===")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            m,
            torch_dtype="auto",
            cache_dir=cache_dir,
            use_safetensors=True,
            attn_implementation="eager",
        )
        n = sum(p.numel() for p in model.parameters())
        print(f"  OK: loaded {n:,} params, dtype={model.dtype}, device={next(model.parameters()).device}")
        del model
        import gc; gc.collect()
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=8)
PYEOF

# ----------------------------------------------------------------------------
# Section 3: SLURM GRES availability for the 4g.40gb slice
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "Section 3: SLURM GRES availability"
echo "============================================================================"
echo "The 3 instant-FAILED jobs all requested nvidia_h100_80gb_hbm3_4g.40gb."
echo "The currently-running 1.7b lnq (12539874) asks for 3g.40gb. Below we"
echo "confirm what MIG slice names SLURM actually knows about right now."

if command -v sinfo >/dev/null 2>&1; then
    # Dump GRES per partition. %R=partition, %G=Gres spec.
    echo ""
    echo "sinfo per-partition GRES:"
    sinfo -h -o "  %20R  %100G" 2>&1 | sort -u

    echo ""
    echo "sinfo per-node GRES (first 20 nodes):"
    sinfo -h -N -o "  %20N  %20P  %80G" 2>&1 | sort -u | head -20

    echo ""
    echo "All distinct GRES tokens advertised cluster-wide:"
    sinfo -h -o "%G" 2>&1 | tr ',' '\n' | sed 's/[[:space:]]\+//g' | sort -u | sed 's/^/  /'
fi

echo ""
echo "Does the 4g.40gb GRES appear anywhere?"
if command -v sinfo >/dev/null 2>&1; then
    if sinfo -h -o "%G" 2>&1 | grep -q "4g.40gb"; then
        echo "  YES — 4g.40gb token seen in sinfo output."
    else
        echo "  NO — 4g.40gb is NOT in any sinfo GRES spec."
    fi
fi

# ----------------------------------------------------------------------------
# Section 4: SLURM-side reason for the FAILED jobs
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "Section 4: per-job scontrol for the 3 FAILED jobs"
echo "============================================================================"

for jid in 12539876 12539886 12539888; do
    echo ""
    echo "--- scontrol show job=${jid} ---"
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show job="${jid}" 2>&1 || true
    fi
    echo ""
    echo "--- sacct -j ${jid} --format=JobID,State,ExitCode,Reason%50,ReqTRES%60 ---"
    if command -v sacct >/dev/null 2>&1; then
        sacct -j "${jid}" --format=JobID,State,ExitCode,Reason%50,ReqTRES%60 2>&1 || true
    fi
done

echo ""
echo "============================================================================"
echo "Done. Rsync back:"
echo "  rsync nibi:\$(pwd)/logs/diagnose_pb_llm_round2.log ./logs/"
echo "============================================================================"
