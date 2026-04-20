#!/bin/bash
# ============================================================================
# Simulate the compute-node offline environment on the login node and try to
# load Qwen3-0.6B + C4 shard. Isolates whether the sbatch failures are caused
# by a broken cache layout, a missing refs/main, or env-var propagation bugs.
#
# Read-only. No installs, no deletes.
#
# Run on the login node, from the repo root:
#     ./sbatch/diagnose_offline.sh
# ============================================================================

set -u

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

CACHE_ROOT="${SCRATCH:-$(pwd)}/billm2_cache"
export BILLM_DOWNLOADS_DIR="$CACHE_ROOT/downloads"
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$BILLM_DOWNLOADS_DIR"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "============================================"
echo "Env — what the sbatch job sees"
echo "============================================"
for v in BILLM_DOWNLOADS_DIR HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE \
         HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE \
         HF_HUB_DISABLE_XET; do
    printf "  %-25s = %s\n" "$v" "${!v:-(unset)}"
done

QWEN_DIR="$BILLM_DOWNLOADS_DIR/models--Qwen--Qwen3-0.6B"

echo
echo "============================================"
echo "Qwen3 snapshot layout on disk"
echo "============================================"
if [[ -d "$QWEN_DIR" ]]; then
    echo "Top-level:"
    ls -la "$QWEN_DIR"
    echo
    echo "refs/:"
    ls -la "$QWEN_DIR/refs" 2>/dev/null || echo "  MISSING — no refs directory"
    if [[ -f "$QWEN_DIR/refs/main" ]]; then
        echo "refs/main contents:"
        cat "$QWEN_DIR/refs/main"
        echo
    fi
    echo
    echo "snapshots/*/:"
    ls -la "$QWEN_DIR"/snapshots/*/ 2>/dev/null | head -30
    echo
    echo "Is config.json present (resolved through symlink)?"
    ls -la "$QWEN_DIR"/snapshots/*/config.json 2>/dev/null || echo "  MISSING"
else
    echo "MISSING: $QWEN_DIR"
fi

echo
echo "============================================"
echo "Test 1 — snapshot_download(local_files_only=True) resolves the dir"
echo "============================================"
python - <<'PY'
import os
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="Qwen/Qwen3-0.6B",
        cache_dir=os.environ["BILLM_DOWNLOADS_DIR"],
        local_files_only=True,
    )
    print("OK snapshot_download ->", path)
    for n in ("config.json", "tokenizer.json", "model.safetensors"):
        full = os.path.join(path, n)
        print(f"   {n:25s} {'present' if os.path.exists(full) else 'MISSING'}")
except Exception as exc:
    print(f"FAIL snapshot_download: {type(exc).__name__}: {exc}")
PY

echo
echo "============================================"
echo "Test 2 — AutoConfig.from_pretrained offline"
echo "============================================"
python - <<'PY'
import os
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        "Qwen/Qwen3-0.6B",
        cache_dir=os.environ["BILLM_DOWNLOADS_DIR"],
    )
    print("OK AutoConfig.from_pretrained:", type(cfg).__name__)
except Exception as exc:
    print(f"FAIL AutoConfig: {type(exc).__name__}: {exc}")
PY

echo
echo "============================================"
echo "Test 3 — AutoModelForCausalLM.from_pretrained offline"
echo "============================================"
python - <<'PY'
import os
try:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype="auto",
        cache_dir=os.environ["BILLM_DOWNLOADS_DIR"],
        use_safetensors=True,
        attn_implementation="eager",
    )
    print("OK model loaded:", type(m).__name__)
except Exception as exc:
    print(f"FAIL model load: {type(exc).__name__}: {exc}")
PY

echo
echo "============================================"
echo "Test 4 — C4 shard direct hf_hub_download (the approach get_redpajama"
echo "         will use after the rewrite)"
echo "============================================"
python - <<'PY'
import os
try:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="allenai/c4",
        filename="en/c4-train.00000-of-01024.json.gz",
        repo_type="dataset",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        local_files_only=True,
    )
    print(f"OK hf_hub_download -> {path}")
    print(f"   exists: {os.path.exists(path)}  size: {os.path.getsize(path):,} bytes")
except Exception as exc:
    print(f"FAIL hf_hub_download: {type(exc).__name__}: {exc}")
PY

echo
echo "============================================"
echo "DONE — paste full output back"
echo "============================================"
