"""Test BRAQ equivalence with float16 inputs (matching actual model weights)."""
import torch
import sys
sys.path.insert(0, '.')
import importlib
import binary
importlib.reload(binary)
from binary import high_order_residual, coupled_residual_binarization_seqalpha

torch.manual_seed(42)

# Create a realistic weight matrix and mask in FLOAT16 (matching actual Qwen3 weights)
oc, ic = 256, 128
x_fp32 = torch.randn(oc, ic, dtype=torch.float32).cuda() * 0.01
x_fp16 = x_fp32.half()

mask = torch.zeros(oc, ic, dtype=torch.bool).cuda()
mask[:, :23] = True

# Test 1: Float32 inputs (our verification was here)
braq_f32 = high_order_residual(x_fp32, mask, order=2)
seqalpha_f32 = coupled_residual_binarization_seqalpha(x_fp32, mask, order=2, skip_refinement=True)
diff_f32 = (braq_f32 - seqalpha_f32).abs().max().item()
print(f"Float32 max diff: {diff_f32:.2e}")

# Test 2: Float16 inputs (matching actual model weights)
braq_f16 = high_order_residual(x_fp16, mask, order=2)
seqalpha_f16 = coupled_residual_binarization_seqalpha(x_fp16, mask, order=2, skip_refinement=True)
diff_f16 = (braq_f16 - seqalpha_f16).abs().max().item()
print(f"Float16 max diff: {diff_f16:.2e}")

# Test 3: Check what dtypes are used internally
print(f"\nBRAQ output dtype: {braq_f16.dtype}")
print(f"SeqAlpha output dtype: {seqalpha_f16.dtype}")

# Check intermediate dtype of mask_f
mask_f = mask.float()
print(f"mask.float() dtype: {mask_f.dtype}")
print(f"x_fp16 * mask_f dtype: {(x_fp16 * mask_f).dtype}")

# The key question: does mask_f being float32 cause type promotion?
# In BRAQ, torch.where(mask, x_fp16, nan) preserves float16
where_result = torch.where(mask, x_fp16, torch.tensor(float('nan'), device=x_fp16.device))
print(f"torch.where(mask, x_fp16, nan) dtype: {where_result.dtype}")
print(f"nanmean result dtype: {torch.nanmean(where_result, dim=1).dtype}")
