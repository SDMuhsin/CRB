#!/bin/bash
# ============================================================================
# diagnose_pb_llm_cache.sh — run on Nibi LOGIN NODE.
#
# Goal: understand why pb-llm (jobs 12539877, 12539889) crashed with
# `LocalEntryNotFoundError` on Qwen3-1.7B and Qwen3-4B while every other
# job that loaded the same model at the same time succeeded.
#
# This script is read-only. It does NOT mutate the cache, submit jobs,
# or install anything. Output is plain text — rsync the script's stdout
# back to the dev-box for analysis.
#
# Usage:
#   cd /home/sdmuhsin/projects/def-seokbum/sdmuhsin/CRB   (or repo root on Nibi)
#   ./sbatch/diagnose_pb_llm_cache.sh 2>&1 | tee ./logs/diagnose_pb_llm_cache.log
#
# Safe to run repeatedly. Fits in the 16 GB login-node cgroup — only
# loads AutoConfig (small JSON), not model weights.
# ============================================================================

set -u

# ----------------------------------------------------------------------------
# Section 0: environment discovery
# ----------------------------------------------------------------------------
echo "============================================================================"
echo "Section 0: environment"
echo "============================================================================"
echo "Host:          $(hostname)"
echo "Date:          $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "PWD:           $(pwd)"
echo "USER:          ${USER:-unset}"
echo "SCRATCH:       ${SCRATCH:-unset}"
echo "HOME:          ${HOME:-unset}"
echo ""

# Resolve cache root the same way the sbatch scripts do.
if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]]; then
    CACHE_ROOT="${SCRATCH}/billm2_cache"
else
    CACHE_ROOT="$(pwd)/downloads"
fi
DOWNLOADS_DIR="${CACHE_ROOT}/downloads"
echo "CACHE_ROOT:    ${CACHE_ROOT}"
echo "DOWNLOADS_DIR: ${DOWNLOADS_DIR}"
echo ""

if [[ ! -d "${DOWNLOADS_DIR}" ]]; then
    echo "FATAL: ${DOWNLOADS_DIR} does not exist. download_cache.sh has not been run on this host."
    exit 2
fi

# ----------------------------------------------------------------------------
# Section 1: per-model cache layout for all 4 Qwen3 sizes
# ----------------------------------------------------------------------------
echo "============================================================================"
echo "Section 1: cache layout per Qwen3 size"
echo "============================================================================"

for model_short in 0.6B 1.7B 4B 8B; do
    hf_dir="${DOWNLOADS_DIR}/models--Qwen--Qwen3-${model_short}"
    echo ""
    echo "--- Qwen3-${model_short} ------------------------------------------------"
    echo "Path: ${hf_dir}"

    if [[ ! -d "${hf_dir}" ]]; then
        echo "  (not present — model not warmed on this host)"
        continue
    fi

    # Top-level layout.
    echo "Top-level contents:"
    ls -la "${hf_dir}" 2>&1 | sed 's/^/  /'

    # refs/main (the Phase 9 gotcha).
    refs_main="${hf_dir}/refs/main"
    if [[ -f "${refs_main}" ]]; then
        size=$(stat -c '%s' "${refs_main}" 2>/dev/null || echo "?")
        content=$(cat "${refs_main}" 2>/dev/null || echo "(unreadable)")
        echo "refs/main: size=${size} content='${content}'"
    else
        echo "refs/main: MISSING"
    fi

    # Snapshot dirs.
    if [[ -d "${hf_dir}/snapshots" ]]; then
        echo "snapshots/:"
        ls -la "${hf_dir}/snapshots" 2>&1 | sed 's/^/  /'
        for snap in "${hf_dir}/snapshots/"*/; do
            [[ -d "${snap}" ]] || continue
            snap_name=$(basename "${snap}")
            echo "  snapshot ${snap_name}:"
            # Check for key files.
            for fname in config.json tokenizer.json tokenizer_config.json generation_config.json; do
                fpath="${snap}${fname}"
                if [[ -e "${fpath}" ]]; then
                    # Follow symlink if any, show target validity.
                    real=$(readlink -f "${fpath}" 2>/dev/null || echo "?")
                    if [[ -f "${real}" ]]; then
                        bytes=$(stat -c '%s' "${real}" 2>/dev/null || echo "?")
                        echo "    ${fname}: OK (${bytes} bytes, -> ${real})"
                    else
                        echo "    ${fname}: BROKEN symlink -> ${real}"
                    fi
                else
                    echo "    ${fname}: absent"
                fi
            done
            # Count safetensors entries (just existence + validity, no read).
            sts=$(ls "${snap}"*.safetensors 2>/dev/null | wc -l)
            echo "    safetensors files: ${sts}"
            if (( sts > 0 )); then
                broken=0
                for st in "${snap}"*.safetensors; do
                    real=$(readlink -f "${st}" 2>/dev/null || echo "")
                    if [[ ! -f "${real}" ]]; then
                        broken=$((broken+1))
                    fi
                done
                echo "    safetensors broken symlinks: ${broken}"
            fi
        done
    else
        echo "snapshots/: MISSING"
    fi

    # Blobs.
    if [[ -d "${hf_dir}/blobs" ]]; then
        blob_count=$(ls "${hf_dir}/blobs" 2>/dev/null | wc -l)
        blob_bytes=$(du -sb "${hf_dir}/blobs" 2>/dev/null | awk '{print $1}')
        echo "blobs/: ${blob_count} files, ${blob_bytes} bytes total"
    else
        echo "blobs/: MISSING"
    fi
done

# ----------------------------------------------------------------------------
# Section 2: minimal offline AutoConfig reproduction
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "Section 2: offline AutoConfig.from_pretrained reproduction"
echo "============================================================================"
echo "Each invocation uses the same env vars the sbatch HEREDOC exports."
echo "HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1. This is the call that pb-llm"
echo "jobs 12539877/12539889 made and that crashed."
echo ""

# Activate venv and module stack the same way the sbatch scripts do.
if ! command -v module >/dev/null 2>&1; then
    echo "NOTE: Lmod 'module' not available in this shell. Skipping module load."
else
    module load gcc arrow scipy-stack cuda cudnn python/3.11 2>&1 | sed 's/^/  module: /' || true
fi
if [[ -f ./env/bin/activate ]]; then
    # shellcheck disable=SC1091
    source ./env/bin/activate
    echo "Activated venv: $(which python)"
else
    echo "WARNING: ./env/bin/activate not found — using system python: $(which python)"
fi

export BILLM_DOWNLOADS_DIR="${DOWNLOADS_DIR}"
export HF_HOME="${CACHE_ROOT}/hf"
export HF_HUB_CACHE="${DOWNLOADS_DIR}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf/datasets"
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo ""
echo "Env snapshot (what pb-llm sees):"
for v in BILLM_DOWNLOADS_DIR HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_DISABLE_XET; do
    echo "  ${v}=${!v:-<unset>}"
done

echo ""
echo "Reproduction test: AutoConfig + AutoTokenizer + AutoModelForCausalLM"
echo "using cache_dir=\$BILLM_DOWNLOADS_DIR, the same call pb-llm makes."
echo ""

python3 - <<'PYEOF'
import os, sys, traceback

print(f"python {sys.version.split()[0]}")
try:
    import transformers
    print(f"transformers {transformers.__version__}")
except Exception as e:
    print(f"transformers import failed: {e}")
    sys.exit(1)

cache_dir = os.environ["BILLM_DOWNLOADS_DIR"]

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

from transformers import AutoConfig, AutoTokenizer

for m in MODELS:
    print(f"\n=== {m} ===")
    # AutoConfig — this is the exact call that fails in pb-llm's traceback.
    try:
        cfg = AutoConfig.from_pretrained(
            m, cache_dir=cache_dir, use_safetensors=True,
            attn_implementation="eager",
        )
        print(f"  AutoConfig.from_pretrained: OK ({cfg.model_type})")
    except Exception as e:
        print(f"  AutoConfig.from_pretrained: FAIL {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        continue

    # AutoTokenizer — also runs inside pb-llm's load path (datautils.get_tokenizer).
    # Tokenizer.from_pretrained does NOT accept cache_dir in some runners;
    # falls back to HF_HUB_CACHE.
    try:
        tok = AutoTokenizer.from_pretrained(m, use_fast=True, cache_dir=cache_dir)
        print(f"  AutoTokenizer.from_pretrained: OK ({type(tok).__name__})")
    except Exception as e:
        print(f"  AutoTokenizer.from_pretrained: FAIL {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)

    # AutoTokenizer WITHOUT cache_dir — this is what datautils.get_tokenizer
    # does in the project. It falls through to HF_HUB_CACHE.
    try:
        tok = AutoTokenizer.from_pretrained(m, use_fast=True)
        print(f"  AutoTokenizer.from_pretrained (no cache_dir): OK")
    except Exception as e:
        print(f"  AutoTokenizer.from_pretrained (no cache_dir): FAIL {type(e).__name__}: {e}")
PYEOF

# ----------------------------------------------------------------------------
# Section 3: SLURM job history for the Apr 21 batch
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "Section 3: SLURM history for the 12539xxx batch (submitted 2026-04-21)"
echo "============================================================================"
if command -v sacct >/dev/null 2>&1; then
    # Show full status incl. pending lnq/tesseraq. --starttime bounds to Apr 21.
    sacct -u "$USER" --starttime=2026-04-21 --endtime=now \
          --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,ReqMem,AllocTRES%40 \
          2>&1
else
    echo "sacct not available on this host."
fi

echo ""
echo "Current queue:"
if command -v squeue >/dev/null 2>&1; then
    squeue -u "$USER" -o "%.18i %.20j %.8T %.10M %.9l %.6D %R" 2>&1
fi

# ----------------------------------------------------------------------------
# Section 4: pb-llm-specific — ./downloads symlink + DOWNLOAD_{model} path
# ----------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "Section 4: pb-llm-specific artifacts"
echo "============================================================================"

echo "./downloads symlink:"
if [[ -L ./downloads ]]; then
    target=$(readlink -f ./downloads)
    echo "  -> ${target}"
else
    echo "  not a symlink."
    ls -la ./downloads 2>/dev/null | head -3 | sed 's/^/  /'
fi

echo ""
echo "PB-LLM-style DOWNLOAD_{model_name} path probe:"
for m in "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B"; do
    p="${DOWNLOADS_DIR}/DOWNLOAD_${m}"
    if [[ -e "${p}" ]]; then
        echo "  ${p}: EXISTS ($(stat -c '%s bytes, type=%F' "${p}" 2>/dev/null))"
    else
        echo "  ${p}: absent (expected — torch.save path is commented out in run.py)"
    fi
done

echo ""
echo "============================================================================"
echo "Done. Rsync this log back to the dev-box:"
echo "  rsync nibi:\$(pwd)/logs/diagnose_pb_llm_cache.log ./logs/"
echo "============================================================================"
