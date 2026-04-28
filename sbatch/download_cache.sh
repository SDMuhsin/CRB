#!/bin/bash
# ============================================================================
# Pre-download all models and datasets to local cache (Nibi / Alliance Canada)
# ============================================================================
#
# Run this on a LOGIN NODE (with internet) BEFORE submitting sbatch jobs.
# Compute nodes have no internet access.
#
# QUOTA NOTE — Alliance Canada (Nibi)
# -----------------------------------
#   /project/<def-xxx>  : tiny quota, code only (NOT for caches/datasets).
#   $SCRATCH            : 1 TB soft / 20 TB hard, 60-day grace — caches go here.
#   $SLURM_TMPDIR       : per-job local NVMe, wiped at job end — no good for
#                         cross-job cache, but fine for transient unpacking.
#
# This script (a) auto-detects $SCRATCH and routes ALL HF / Torch / project
# caches to $SCRATCH/billm2_cache, and (b) replaces ./downloads with a
# symlink to that scratch dir so the project's hardcoded `./downloads/...`
# paths (datautils.py, PB-LLM/gptq_pb/run.py) write to scratch automatically.
#
# Usage:
#   ./sbatch/download_cache.sh
#
# ============================================================================

set -euo pipefail

# Module loads must precede venv activation — the venv's python symlink
# resolves to the module-provided python/3.11 (via scipy-stack).
module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate
# Note: the venv legitimately borrows idna / certifi / safetensors / yaml /
# tqdm / accelerate / typing_extensions from $HOME/.local/. Do NOT set
# PYTHONNOUSERSITE=1 here. The torchvision ABI crash is fixed separately
# by uninstalling torchvision/torchaudio from ~/.local (see
# ./sbatch/fix_venv_torchvision.sh).

# ---------------------------------------------------------------------------
# 1. Pick CACHE_ROOT — $SCRATCH on Alliance, ./downloads elsewhere.
# ---------------------------------------------------------------------------
if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]]; then
    CACHE_ROOT="$SCRATCH/billm2_cache"
    echo "Detected Alliance Canada — caching under \$SCRATCH"
else
    CACHE_ROOT="$(pwd)/downloads"
    echo "No \$SCRATCH found — caching under project ./downloads"
fi
mkdir -p "$CACHE_ROOT"

# ---------------------------------------------------------------------------
# 2. Ensure ./downloads → CACHE_ROOT symlink (so hardcoded relative paths
#    in datautils.py and PB-LLM/gptq_pb/run.py land on scratch).
# ---------------------------------------------------------------------------
if [[ "$CACHE_ROOT" != "$(pwd)/downloads" ]]; then
    if [[ -L ./downloads ]]; then
        # Replace stale symlink (idempotent re-runs).
        rm ./downloads
    elif [[ -e ./downloads ]]; then
        echo ""
        echo "ERROR: ./downloads exists as a real directory, not a symlink."
        echo "       Move it to scratch and re-run, e.g.:"
        echo "         mv ./downloads $CACHE_ROOT/downloads_legacy"
        echo "         ./sbatch/download_cache.sh"
        exit 1
    fi
    ln -sfn "$CACHE_ROOT" ./downloads
    echo "Symlinked ./downloads -> $CACHE_ROOT"
fi

# ---------------------------------------------------------------------------
# 3. Point every HuggingFace / Torch cache env var at CACHE_ROOT.
#    HF_DATASETS_CACHE is the key one — that's where arrow shards land.
# ---------------------------------------------------------------------------
# Project-level override read by datautils.py / run.py / PB-LLM / src/run_*.py.
# This is the explicit, non-symlink-based way to redirect every hardcoded
# ./downloads/... path the project uses.
export BILLM_DOWNLOADS_DIR="$CACHE_ROOT/downloads"

# CRITICAL — HF_HUB_CACHE must equal BILLM_DOWNLOADS_DIR. The runners call
# AutoModelForCausalLM.from_pretrained(model, cache_dir=downloads_dir, ...)
# but datautils.get_tokenizer calls AutoTokenizer.from_pretrained(model) with
# NO cache_dir, which falls back to HF_HUB_CACHE. If those two are different
# directories, the offline compute job finds the model but not the tokenizer
# (or vice-versa). Aligning them here means snapshot_download, AutoModel,
# and AutoTokenizer all read/write the same models--*/ tree.
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$BILLM_DOWNLOADS_DIR"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
# TRANSFORMERS_CACHE is deprecated since transformers 4.36; HF_HOME covers it.
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_DISABLE_XET=1
# Anything that defaults to $HOME/.cache/* must be redirected too — $HOME on
# Nibi has only ~50 GiB and goes over-quota fast (numba JIT, pip wheels, etc.).
export NUMBA_CACHE_DIR="$CACHE_ROOT/.cache/numba"
export PIP_CACHE_DIR="$CACHE_ROOT/.cache/pip"
export XDG_CACHE_HOME="$CACHE_ROOT/.cache"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TORCH_HOME" \
         "$NUMBA_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" \
         "$BILLM_DOWNLOADS_DIR"

echo ""
echo "Cache layout (resolved real paths — these are what writes actually hit):"
echo "  CACHE_ROOT             = $CACHE_ROOT"
echo "  BILLM_DOWNLOADS_DIR    = $BILLM_DOWNLOADS_DIR"
echo "  ./downloads            -> $(readlink -f ./downloads 2>/dev/null || echo MISSING)"
echo "  HF_HOME                -> $(readlink -f "$HF_HOME" 2>/dev/null || echo "$HF_HOME (not yet created)")"
echo "  HF_DATASETS_CACHE      -> $(readlink -f "$HF_DATASETS_CACHE" 2>/dev/null || echo "$HF_DATASETS_CACHE (not yet created)")"
echo "  HF_HUB_CACHE           -> $(readlink -f "$HF_HUB_CACHE" 2>/dev/null || echo "$HF_HUB_CACHE (not yet created)")"
echo "  TORCH_HOME             -> $(readlink -f "$TORCH_HOME" 2>/dev/null || echo "$TORCH_HOME (not yet created)")"
echo "  NUMBA_CACHE_DIR        -> $(readlink -f "$NUMBA_CACHE_DIR" 2>/dev/null || echo "$NUMBA_CACHE_DIR (not yet created)")"
echo ""

# Hard guard: if ./downloads ends up resolving anywhere under /project or
# /home, abort — that would silently fill the wrong quota.
DL_REAL=$(readlink -f ./downloads 2>/dev/null || echo "")
case "$DL_REAL" in
    /project/*|/home/*|"$HOME"/*)
        echo "ABORT: ./downloads resolves to '$DL_REAL', which is on /project or \$HOME."
        echo "       Those filesystems have small quotas. Set \$SCRATCH or fix the symlink."
        exit 1
        ;;
esac
unset DL_REAL

# ============================================================================
# Models — all four Qwen3 sizes in the benchmark suite (0.6B/1.7B/4B/8B)
# ============================================================================
#
# IMPORTANT: write the model under \$BILLM_DOWNLOADS_DIR (cache_dir argument),
# NOT the default HF_HUB_CACHE. The project's runners call
# from_pretrained(..., cache_dir=downloads_dir) so they read from there at
# job time. snapshot_download must use the SAME directory or the offline
# compute job will fail to find the model under HF_HUB_OFFLINE=1.
# ============================================================================

echo "=== Downloading models (snapshot to BILLM_DOWNLOADS_DIR) ==="
echo "    target: $BILLM_DOWNLOADS_DIR"

# Optional HF token for gated repos (meta-llama/Llama-3.x-* requires auth).
# Read from ./.hf_token (gitignored) if present; export so the python block
# below can pick it up via huggingface_hub's standard env-var lookup.
if [[ -f ./.hf_token ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < ./.hf_token)"
    echo "    HF token loaded from ./.hf_token (length ${#HF_TOKEN})."
else
    echo "    NOTE: no ./.hf_token found — gated repos (e.g. meta-llama/*) will fail."
fi

python -c "
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

cache_dir = os.environ['BILLM_DOWNLOADS_DIR']
# huggingface_hub picks up HF_TOKEN automatically; pass explicitly anyway so
# AutoConfig/AutoTokenizer also authenticate against gated repos like
# meta-llama/Llama-3.x-*.
hf_token = os.environ.get('HF_TOKEN') or None

# All Qwen3 + Llama-3 sizes covered by the Nibi benchmark suites. Adding a
# new model here requires a companion re-run of this script to populate
# refs/main + transformers cache.
models = [
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-1.7B',
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-8B',
    # Llama-3 family. Llama-3.2-{1B,3B} use Llama-3.2 community gate; the
    # supplied .hf_token grants access. Llama-3.1-8B is gated separately and
    # the .hf_token was confirmed 403 Forbidden against it (2026-04-27);
    # NousResearch/Meta-Llama-3.1-8B is an ungated identical-weight mirror
    # (Phase-14B precedent — same pattern used for Llama-2-7b-hf).
    'meta-llama/Llama-3.2-1B',
    'meta-llama/Llama-3.2-3B',
    'NousResearch/Meta-Llama-3.1-8B',
]

for model_name in models:
    print(f'Downloading {model_name} -> {cache_dir} ...')
    path = snapshot_download(repo_id=model_name, cache_dir=cache_dir, token=hf_token)
    print(f'  Snapshot: {path}')
    # Touch the tokenizer + config so the transformers cache index is warmed
    # under the SAME cache_dir the runners will read from offline.
    AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    print(f'  Tokenizer + config ready.')
    # CRITICAL: a subsequent snapshot_download(local_files_only=True) is the
    # only reliable way to guarantee refs/main has been written. Without this
    # hf_hub_download(revision='main', local_files_only=True) can later fail
    # with LocalEntryNotFoundError even though every file is on disk.
    # local_files_only=True does NOT need the token (no network call).
    resolved = snapshot_download(repo_id=model_name, cache_dir=cache_dir, local_files_only=True)
    print(f'  refs/main verified offline -> {resolved}')

# Abort here if this transformers is too old for Qwen3 — catches wheelhouse
# wheels that lack Qwen3ForCausalLM before any sbatch job is submitted.
try:
    from transformers.models.qwen3 import Qwen3ForCausalLM  # noqa: F401
    import transformers
    print(f'Qwen3 import OK (transformers {transformers.__version__}).')
except Exception as exc:
    raise SystemExit(
        f'FATAL: Qwen3ForCausalLM not importable ({exc!r}). '
        'Install transformers>=4.51 from PyPI (see requirements.txt header).'
    )
# LlamaForCausalLM has been in transformers for ages but verify import too —
# catches venv breakage before any sbatch job goes out.
try:
    from transformers import LlamaForCausalLM  # noqa: F401
    print(f'LlamaForCausalLM import OK.')
except Exception as exc:
    raise SystemExit(
        f'FATAL: LlamaForCausalLM not importable ({exc!r}). '
        'Check transformers install (see fix_venv_torchvision.sh for ABI fixes).'
    )
"

echo ""

# ============================================================================
# Calibration / evaluation datasets
# ============================================================================
#
# - wikitext-2-raw-v1: PPL eval + default calibration for run.py.
# - C4 ONE SHARD: matches datautils.get_redpajama(), which loads only
#   en/c4-train.00000-of-01024.json.gz (~150 MB). DO NOT call
#   load_dataset('allenai/c4', 'en') without `data_files=` — it triggers
#   the full 365M-row train split build (>700 GB on disk) and will blow
#   through scratch quota.
# ============================================================================

echo "=== Downloading datasets ==="

python -c "
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download

print('Downloading wikitext (wikitext-2-raw-v1)...')
load_dataset('wikitext', 'wikitext-2-raw-v1')
print('  Done: wikitext-2')

# C4 shards — use hf_hub_download NOT load_dataset. The latter's config-hash
# changes between online pre-warm and offline runtime (resolved URLs differ),
# which made calibration jobs fail with 'Couldn't find cache for allenai/c4
# for config default-...'. datautils.get_redpajama reads the .json.gz file
# directly with gzip + json.loads — it just needs the raw blob present.
#
# Phase 15: also pre-warm the C4 VALIDATION shard for the C4 PPL eval
# (datautils.get_c4 -> _c4_read_shard).
for c4_filename in (
    'en/c4-train.00000-of-01024.json.gz',
    'en/c4-validation.00000-of-00008.json.gz',
):
    print(f'Downloading allenai/c4 {c4_filename}...')
    shard = hf_hub_download(
        repo_id='allenai/c4', filename=c4_filename, repo_type='dataset',
        cache_dir=os.environ['BILLM_DOWNLOADS_DIR'],
    )
    print(f'  shard: {shard}  ({os.path.getsize(shard):,} bytes)')
    # local_files_only=True follow-up to write refs/main (Phase 9 gotcha 22).
    shard_offline = hf_hub_download(
        repo_id='allenai/c4', filename=c4_filename, repo_type='dataset',
        cache_dir=os.environ['BILLM_DOWNLOADS_DIR'], local_files_only=True,
    )
    print(f'  offline-verified: {shard_offline}')

# Phase 15: downstream eval datasets (MMLU 5-shot, HellaSwag, ARC).
# These are datasets-script-format and load via load_dataset(). They live
# under \$BILLM_DOWNLOADS_DIR/datasets/ to match the cache_dir argument used
# in eval_*.py (which reads from \$BILLM_DOWNLOADS_DIR + '/datasets').
DATASETS_CACHE = os.path.join(os.environ['BILLM_DOWNLOADS_DIR'], 'datasets')
print(f'\\n=== Downstream eval datasets (cache_dir={DATASETS_CACHE}) ===')
for repo_id, name, kwargs in [
    ('cais/mmlu', 'MMLU', {'name': 'all'}),
    ('Rowan/hellaswag', 'HellaSwag', {}),
    ('allenai/ai2_arc', 'ARC-Easy', {'name': 'ARC-Easy'}),
    ('allenai/ai2_arc', 'ARC-Challenge', {'name': 'ARC-Challenge'}),
]:
    print(f'Downloading {name} ({repo_id})...')
    load_dataset(repo_id, cache_dir=DATASETS_CACHE, **kwargs)
    print(f'  OK')

# Phase 15: PTB.
# The HF \`ptb_text_only\` repo is script-only (no parquet/arrow shards).
# \`datasets >= 4.x\` no longer supports script loaders, so we cannot rely
# on \`load_dataset('ptb_text_only', ...)\` on Nibi (or anywhere).
# We fetch the canonical Mikolov 2010 LM-benchmark splits directly from a
# stable raw-text URL — same content as the HF Arrow rows after .strip().
import urllib.request
PTB_DIR = os.path.join(DATASETS_CACHE, 'ptb_mikolov')
os.makedirs(PTB_DIR, exist_ok=True)
PTB_BASE = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data'
print(f'\\n=== PTB (Mikolov LM splits, raw text -> {PTB_DIR}) ===')
for fname in ('ptb.train.txt', 'ptb.test.txt', 'ptb.valid.txt'):
    out = os.path.join(PTB_DIR, fname.replace('ptb.', '').replace('.txt', '.txt'))
    # Final filename: train.txt / test.txt / valid.txt (matches _ptb_load_split lookup).
    out_path = os.path.join(PTB_DIR, fname.split('.')[1] + '.txt')
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f'  {os.path.basename(out_path)} cached ({os.path.getsize(out_path):,} bytes)')
        continue
    url = f'{PTB_BASE}/{fname}'
    print(f'  fetching {url} -> {out_path}')
    urllib.request.urlretrieve(url, out_path)
    print(f'    {os.path.getsize(out_path):,} bytes')
"

echo ""

# ============================================================================
# NOTE: pre-tokenization is intentionally NOT done here.
# ============================================================================
#
# Earlier versions tokenized the full redpajama (1024 × 4096) calibration set
# on the login node so the first benchmark job wouldn't pay the cost. That
# step joins ~83K C4 documents into a single string and tokenizes to a
# 2.5M-token tensor — it OOM-killed (login-node cgroup ~16 GB).
#
# Race-safety on the cache file is now handled by an fcntl flock added to
# datautils.get_loaders (DOWNLOAD_*.pt.lock). Concurrent jobs serialize:
# the first one tokenizes, the rest block on the lock then load the cache.
# So the only cost we pay is one extra tokenization on the FIRST sbatch
# job — which has the right memory budget for it (16 GB, see GPU_SMALL job
# spec in run_qwen_benchmark.sh).
#
# If you really want to pre-tokenize (e.g., to keep first-job wall time
# down), submit it as its own tiny sbatch job — don't run it on the login
# node:
#
#   sbatch --time=00:30:00 --mem=16G --cpus-per-task=4 --wrap='
#     source ./env/bin/activate
#     export BILLM_DOWNLOADS_DIR=$SCRATCH/billm2_cache/downloads
#     export HF_HOME=$SCRATCH/billm2_cache/hf
#     python -c "
#     import sys; sys.path.insert(0, \".\")
#     from datautils import get_loaders
#     get_loaders(\"wikitext2\", nsamples=128, seed=0, seqlen=2048,
#                 model=\"Qwen/Qwen3-0.6B\")
#     get_loaders(\"redpajama\", nsamples=1024, seed=0, seqlen=4096,
#                 model=\"Qwen/Qwen3-0.6B\")"
#   '
# ============================================================================

echo "=== Skipping pre-tokenization on login node (would OOM-kill). ==="
echo "    The first sbatch job will tokenize and write the cache;"
echo "    subsequent jobs block on the .lock then load it."
echo ""

# ============================================================================
# Phase 18b: invalidate stale Llama-3 PTB calibration caches.
# ============================================================================
#
# Pre-Phase-18b runs used the slow LlamaTokenizer for Llama-3 names, which
# appends `<unk>` to added_tokens_encoder at id 128256 — out of range of
# Llama-3's 128256-row embed_tokens (valid IDs 0..128255). PTB Mikolov text
# contains 4794 literal `<unk>` substrings, so any DOWNLOAD_ptb_*.pt cache
# built by those runs holds id 128256 and triggers
# `Indexing.cu:1308: indexSelectLargeIndex` on every subsequent PTB eval —
# even after the `is_llama3` fix in datautils.get_tokenizer, because that
# fix only fires on cache MISS. Forcing a miss here lets the next sbatch
# job rebuild with the correct fast BPE tokenizer.
#
# Scoped to Llama-3 names only — Llama-2 (BOS=1, no `<unk>` appended) and
# Qwen3 (different tokenizer family) caches are unaffected. Wikitext2 and
# C4 caches are safe regardless (no literal `<unk>` in those corpora).
# See memory: feedback_get_loaders_cache_poisoning.md.
# ============================================================================

echo "=== Invalidating stale PTB calibration caches for Llama-3 models ==="
removed_any=0
for m in 'meta-llama/Llama-3.2-1B' \
         'meta-llama/Llama-3.2-3B' \
         'NousResearch/Meta-Llama-3.1-8B'; do
    for f in "$BILLM_DOWNLOADS_DIR"/DOWNLOAD_ptb_*_${m}.pt \
             "$BILLM_DOWNLOADS_DIR"/DOWNLOAD_ptb_*_${m}.pt.lock; do
        [[ -e "$f" ]] || continue
        echo "  removing $f"
        rm -f "$f"
        removed_any=1
    done
done
if [[ $removed_any -eq 0 ]]; then
    echo "  (no stale PTB calibration files found — clean)"
fi
echo ""

echo "============================================"
echo "All downloads complete."
echo "Cache root:                       $CACHE_ROOT"
echo "Total cache size on \$SCRATCH:     $(du -sh "$CACHE_ROOT" 2>/dev/null | cut -f1)"
# -x stops at filesystem boundary, so this only counts bytes that LIVE on
# /project (i.e., excludes anything reached via the symlink). It must be ~0.
echo "Bytes that leaked to /project:    $(du -shx "$(pwd)/downloads" 2>/dev/null | cut -f1) (should be tiny — just the symlink)"
echo "Next step:                        sbatch ./sbatch/run_qwen_benchmark.sh"
echo "============================================"
