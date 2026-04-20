#!/bin/bash
# ============================================================================
# Diagnose the venv vs ~/.local package split on Nibi.
#
# Run on the login node, inside the repo:
#     ./sbatch/diagnose_env.sh
#
# Paste the full output back. No packages are installed or removed; read-only.
# ============================================================================

set -u
source ./env/bin/activate

echo "============================================"
echo "1. Which file provides each core package?"
echo "============================================"
python - <<'PY'
import importlib.util
names = [
    "torch", "torchvision", "transformers", "huggingface_hub", "tokenizers",
    "safetensors", "idna", "certifi", "httpx", "yaml", "tqdm", "datasets",
    "numpy", "pyarrow", "accelerate", "flash1dkmeans", "numba",
    "packaging", "filelock", "regex", "fsspec", "typing_extensions",
]
for name in names:
    spec = importlib.util.find_spec(name)
    print(f"{name:22s} {spec.origin if spec else 'NOT FOUND'}")
PY

echo
echo "============================================"
echo "2. torch vs torchvision versions"
echo "============================================"
python -c "
import torch
print('torch       ', torch.__version__, torch.__file__)
try:
    import torchvision
    print('torchvision ', torchvision.__version__, torchvision.__file__)
except Exception as exc:
    print('torchvision import FAILED:', type(exc).__name__, exc)
" 2>&1 | head -20

echo
echo "============================================"
echo "3. Does transformers 5.5.4 guard the vision import?"
echo "    (look for is_vision_available, try/except, env-var opt-outs)"
echo "============================================"
IMAGE_UTILS=./env/lib/python3.11/site-packages/transformers/image_utils.py
if [[ -f "$IMAGE_UTILS" ]]; then
    grep -n "is_vision_available\|TRANSFORMERS_NO_VISION\|USE_TORCH\|torchvision\|try:\|except" \
        "$IMAGE_UTILS" | head -40
else
    echo "NOT FOUND: $IMAGE_UTILS"
fi

echo
echo "============================================"
echo "4a. Packages in ~/.local/lib/python3.11/site-packages/"
echo "============================================"
if [[ -d "$HOME/.local/lib/python3.11/site-packages" ]]; then
    ls "$HOME/.local/lib/python3.11/site-packages/" | sort
else
    echo "(no user-site dir)"
fi

echo
echo "============================================"
echo "4b. Top-level entries in venv site-packages"
echo "============================================"
ls ./env/lib/python3.11/site-packages/ | sort

echo
echo "============================================"
echo "5. pip show for the ABI-sensitive trio"
echo "============================================"
pip show torch torchvision transformers 2>&1 | head -40

echo
echo "============================================"
echo "DIAGNOSIS COMPLETE"
echo "============================================"
