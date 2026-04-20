#!/bin/bash
# ============================================================================
# Fix the torchvision/torch ABI mismatch that crashes transformers 5.x import.
#
# Context (from ./sbatch/diagnose_env.sh output):
#   venv     torch          2.5.1+computecanada   (/project/.../env/)
#   ~/.local torchvision    0.18.0+cu118          (built for torch 2.3.0 cu118)
#   ~/.local torchaudio     2.3.0+cu118           (same mismatch)
#
# transformers 5.5.4's image_utils.py imports torchvision under an
# `is_vision_available()` guard. That guard uses importlib.util.find_spec,
# which succeeds for a broken-but-present torchvision — so the import goes
# ahead and then dies at `torchvision::nms` registration, bubbling up as:
#     ModuleNotFoundError: Could not import module 'Qwen3ForCausalLM'
#
# Fix: remove just torchvision + torchaudio (and torio, a torchaudio dep)
# from ~/.local. Neither is used by BiLLM2. Everything else in ~/.local
# (idna, certifi, safetensors, yaml, tqdm, accelerate, typing_extensions)
# stays intact — the venv depends on those at runtime.
#
# Why rm -rf instead of `pip uninstall`:
#   - pip uninstall --user is not a valid flag.
#   - With venv active, `pip uninstall torchvision` resolves against the
#     venv's sys.path (which includes ~/.local) but pip can refuse or warn
#     when the target isn't under the active site-packages. rm is unambiguous.
#
# Run on a login node:
#     ./sbatch/fix_venv_torchvision.sh
# ============================================================================

set -u

# Module loads MUST precede venv activation on Alliance Canada. The venv's
# ./env/bin/python symlink resolves to the python/3.11 binary exposed by
# scipy-stack; without the modules, activation gives you a broken python
# and a broken pip.
module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

USER_SITE="$HOME/.local/lib/python3.11/site-packages"

echo "============================================"
echo "BEFORE — what's in $USER_SITE"
echo "============================================"
for pkg in torchvision torchaudio torio; do
    for d in "$USER_SITE"/${pkg} "$USER_SITE"/${pkg}.libs "$USER_SITE"/${pkg}-*.dist-info; do
        [[ -e "$d" ]] && echo "  FOUND  $d"
    done
done

# Confirm current state via Python as well.
python - <<'PY'
import importlib.util
for name in ("torch", "torchvision", "torchaudio", "torio"):
    spec = importlib.util.find_spec(name)
    print(f"  {name:12s} {spec.origin if spec else 'not installed'}")
PY

echo
echo "============================================"
echo "Removing torchvision + torchaudio + torio"
echo "============================================"
removed_any=0
for target in \
    "$USER_SITE/torchvision" \
    "$USER_SITE/torchvision.libs" \
    "$USER_SITE"/torchvision-*.dist-info \
    "$USER_SITE/torchaudio" \
    "$USER_SITE"/torchaudio-*.dist-info \
    "$USER_SITE/torio" \
    "$USER_SITE"/torio-*.dist-info \
; do
    # Globs that don't match leave the literal pattern — skip those.
    if [[ -e "$target" ]]; then
        echo "  rm -rf $target"
        rm -rf "$target"
        removed_any=1
    fi
done
if [[ "$removed_any" -eq 0 ]]; then
    echo "  (nothing to remove — may have already been cleaned)"
fi

echo
echo "============================================"
echo "AFTER — confirm removal"
echo "============================================"
python - <<'PY'
import importlib.util
for name in ("torchvision", "torchaudio", "torio"):
    spec = importlib.util.find_spec(name)
    if spec:
        print(f"  {name:12s} STILL PRESENT at {spec.origin}")
    else:
        print(f"  {name:12s} removed")
PY

echo
echo "============================================"
echo "Verify Qwen3 is importable"
echo "============================================"
python -c "
from transformers.models.qwen3 import Qwen3ForCausalLM
import transformers, torch
print('OK  transformers', transformers.__version__)
print('OK  torch       ', torch.__version__)
print('OK  Qwen3ForCausalLM imported cleanly')
" || {
    echo
    echo "Qwen3 import still failing — paste this output, further steps needed."
    exit 1
}

echo
echo "============================================"
echo "SUCCESS. Next steps:"
echo "  1. ./sbatch/download_cache.sh"
echo "  2. sbatch ./sbatch/run_qwen_benchmark.sh"
echo "============================================"
