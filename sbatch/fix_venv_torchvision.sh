#!/bin/bash
# ============================================================================
# Fix the torchvision/torch ABI mismatch that crashes transformers 5.x import.
#
# Context (from ./sbatch/diagnose_env.sh output):
#   venv   torch            2.5.1+computecanada   (/project/.../env/)
#   ~/.local torchvision    0.18.0+cu118          (built for torch 2.3.0 cu118)
#   ~/.local torchaudio     2.3.0+cu118           (same mismatch)
#
# transformers 5.5.4's image_utils.py imports torchvision under an
# `is_vision_available()` guard. That guard uses importlib.util.find_spec,
# which succeeds for a broken-but-present torchvision — so the import goes
# ahead and then dies at `torchvision::nms` registration, bubbling up as:
#     ModuleNotFoundError: Could not import module 'Qwen3ForCausalLM'
#
# Fix: uninstall just torchvision + torchaudio from ~/.local. Neither is used
# by BiLLM2. Everything else in ~/.local (idna, certifi, safetensors, yaml,
# tqdm, accelerate, typing_extensions) stays intact because the venv depends
# on those at runtime.
#
# Run on a login node (needs pip, no internet required since this only
# removes files):
#     ./sbatch/fix_venv_torchvision.sh
# ============================================================================

set -u
source ./env/bin/activate

echo "============================================"
echo "BEFORE — torchvision / torchaudio locations"
echo "============================================"
for pkg in torchvision torchaudio; do
    echo "--- $pkg ---"
    pip show "$pkg" 2>&1 | grep -E "^(Name|Version|Location)" || echo "(not installed)"
done

echo
echo "============================================"
echo "Uninstalling torchvision + torchaudio from ~/.local/"
echo "============================================"
pip uninstall --user -y torchvision torchaudio

echo
echo "============================================"
echo "AFTER — confirm both are gone"
echo "============================================"
python - <<'PY'
import importlib.util
for name in ("torchvision", "torchaudio"):
    spec = importlib.util.find_spec(name)
    print(f"{name:15s} {'STILL PRESENT at ' + spec.origin if spec else 'REMOVED (good)'}")
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
    echo "Qwen3 import STILL failing. Paste this output; further steps needed."
    exit 1
}

echo
echo "============================================"
echo "SUCCESS. Next steps:"
echo "  1. ./sbatch/download_cache.sh"
echo "  2. sbatch ./sbatch/run_qwen_benchmark.sh   # fp16 smoke test"
echo "============================================"
