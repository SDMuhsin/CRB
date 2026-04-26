#!/bin/bash
# ============================================================================
# Phase 15 eval-pipeline cache diagnostic (Nibi login node, read-only).
#
# Verifies that the Phase-15 eval suite (3 PPL + 4 downstream benchmarks)
# can run OFFLINE on a Nibi compute node by simulating the offline env on
# the LOGIN node. Mirrors the diagnose_pb_llm_*.sh pattern.
#
# Specifically checks the things download_cache.sh now pre-warms:
#   - C4 train AND validation shards via hf_hub_download
#   - PTB Mikolov raw text files (train/test/valid.txt) — Phase 15 fix:
#     HF `ptb_text_only` is script-only; raw Mikolov is the canonical source.
#   - MMLU / HellaSwag / ARC-Easy / ARC-Challenge load_dataset caches
#
# Then attempts the actual loaders that the runners call:
#   - datautils.get_loaders('wikitext2', ...)
#   - datautils.get_loaders('c4', ...)
#   - datautils.get_loaders('ptb', ...)
#   - eval_arc / eval_mmlu / eval_hellaswag dataset loads
#
# Read-only: no installs, no deletes, no model downloads, no quantization.
# Exit code 0 ⇒ all caches present and offline-loadable. Non-zero ⇒ failure
# at the listed step; re-run ./sbatch/download_cache.sh to top up.
#
# Usage (login node, repo root):
#     ./sbatch/diagnose_phase15_eval_cache.sh
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

DATASETS_CACHE="$BILLM_DOWNLOADS_DIR/datasets"

echo "============================================"
echo "Env — what the sbatch job sees"
echo "============================================"
for v in BILLM_DOWNLOADS_DIR HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE \
         HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE \
         HF_HUB_DISABLE_XET; do
    printf "  %-25s = %s\n" "$v" "${!v:-(unset)}"
done
echo "  DATASETS_CACHE (eval *.py)  = $DATASETS_CACHE"

# ----------------------------------------------------------------------------
# Section 1 — disk layout
# ----------------------------------------------------------------------------
echo
echo "============================================"
echo "Section 1 — Per-dataset on-disk layout"
echo "============================================"

check_blob() {
    local label="$1"; local path="$2"
    if [[ -e "$path" ]]; then
        local size; size=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
        echo "  [OK]   $label  ($size)  $path"
    else
        echo "  [MISS] $label  $path"
    fi
}

# C4 — both train and validation shards (Phase 15 added validation)
check_blob "C4 train shard"      "$BILLM_DOWNLOADS_DIR/datasets--allenai--c4/snapshots"
ls "$BILLM_DOWNLOADS_DIR"/datasets--allenai--c4/snapshots/*/en/c4-train.00000-of-01024.json.gz 2>/dev/null \
    | sed 's/^/    train: /' || echo "    train: NOT FOUND"
ls "$BILLM_DOWNLOADS_DIR"/datasets--allenai--c4/snapshots/*/en/c4-validation.00000-of-00008.json.gz 2>/dev/null \
    | sed 's/^/    val:   /' || echo "    val:   NOT FOUND"

# PTB Mikolov raw text (Phase 15 — primary path)
check_blob "PTB Mikolov train.txt" "$DATASETS_CACHE/ptb_mikolov/train.txt"
check_blob "PTB Mikolov test.txt"  "$DATASETS_CACHE/ptb_mikolov/test.txt"
check_blob "PTB Mikolov valid.txt" "$DATASETS_CACHE/ptb_mikolov/valid.txt"

# Downstream eval datasets — load_dataset caches
check_blob "MMLU cache"            "$DATASETS_CACHE/cais___mmlu"
check_blob "HellaSwag cache"       "$DATASETS_CACHE/Rowan___hellaswag"
check_blob "ARC cache (combined)"  "$DATASETS_CACHE/allenai___ai2_arc"

# WikiText-2 — already used pre-Phase-15
check_blob "WikiText-2 cache"      "$DATASETS_CACHE/wikitext"

echo

# ----------------------------------------------------------------------------
# Section 2 — exercise the loaders the runners actually call
# ----------------------------------------------------------------------------
echo "============================================"
echo "Section 2 — Loader probes (offline)"
echo "============================================"

python3 - <<'PY'
import os, sys, traceback
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

failures = []

def probe(label, fn):
    print(f"\n--- {label} ---")
    try:
        fn()
        print(f"  PASS: {label}")
    except Exception as e:
        traceback.print_exc()
        failures.append((label, type(e).__name__, str(e)[:200]))
        print(f"  FAIL: {label} -- {type(e).__name__}: {str(e)[:200]}")

# Probe A — datautils.get_loaders for the three PPL datasets
def probe_loader(name):
    from datautils import get_loaders
    # Tiny calibration (4 samples) so the .pt cache is cheap if it
    # ends up being built during this diagnostic.
    _, testenc = get_loaders(name, nsamples=4, seed=0, seqlen=2048,
                             model='Qwen/Qwen3-0.6B')
    n = testenc.input_ids.numel()
    print(f"  {name}: testenc tokens = {n:,}")

probe("get_loaders('wikitext2')", lambda: probe_loader('wikitext2'))
probe("get_loaders('c4')",        lambda: probe_loader('c4'))
probe("get_loaders('ptb')",       lambda: probe_loader('ptb'))

# Probe B — downstream eval datasets via load_dataset (the same call
# the eval_*.py modules make under cache_dir = $BILLM_DOWNLOADS_DIR/datasets).
def probe_dataset(repo_id, name=None, split=None):
    from datasets import load_dataset
    cd = os.path.join(os.environ['BILLM_DOWNLOADS_DIR'], 'datasets')
    kwargs = {'cache_dir': cd}
    if name is not None:
        kwargs['name'] = name
    if split is not None:
        kwargs['split'] = split
    ds = load_dataset(repo_id, **kwargs)
    if hasattr(ds, '__len__'):
        print(f"  rows = {len(ds):,}")
    else:
        print(f"  splits = {list(ds.keys())}")

probe("load_dataset('cais/mmlu', name='all', split='test')",
      lambda: probe_dataset('cais/mmlu', name='all', split='test'))
probe("load_dataset('Rowan/hellaswag', split='validation')",
      lambda: probe_dataset('Rowan/hellaswag', split='validation'))
probe("load_dataset('allenai/ai2_arc', name='ARC-Easy', split='test')",
      lambda: probe_dataset('allenai/ai2_arc', name='ARC-Easy', split='test'))
probe("load_dataset('allenai/ai2_arc', name='ARC-Challenge', split='test')",
      lambda: probe_dataset('allenai/ai2_arc', name='ARC-Challenge', split='test'))

print()
print("=" * 44)
if failures:
    print(f"FAIL: {len(failures)} probe(s) failed")
    for label, exc, msg in failures:
        print(f"  - {label}: {exc}: {msg}")
    sys.exit(1)
print("PASS: all eval caches load offline")
sys.exit(0)
PY

rc=$?
echo
echo "============================================"
echo "Diagnostic exit: $rc  ($([ $rc -eq 0 ] && echo PASS || echo FAIL))"
echo "============================================"
exit $rc
