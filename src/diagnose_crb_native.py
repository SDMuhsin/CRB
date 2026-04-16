"""Diagnostic: compare BRAQ vs joint alphas for Qwen3-1.7B first layer."""
import torch
import sys
sys.path.insert(0, '.')

from binary import coupled_residual_binarization_native, high_order_residual
import binary

# Simulate a typical weight block
torch.manual_seed(42)
# Mimic float16 weights (oc=256, ic=128)
W = torch.randn(256, 128, dtype=torch.float16, device='cuda:0') * 0.05
mask = torch.ones(256, 128, dtype=torch.bool, device='cuda:0')
# Make ~30% of columns unmasked (salient partition)
mask[:, 80:] = False

# Run BRAQ
binary.index = 0
Q_braq = high_order_residual(W.clone(), mask.clone(), order=2)
braq_err = ((W - Q_braq) * mask).pow(2).sum().item()

print(f"BRAQ MSE: {braq_err:.4f}")

# Run crb_native at different couplings
for c in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    binary.index = 0
    Q = coupled_residual_binarization_native(W.clone(), mask.clone(), order=2, coupling=c)
    err = ((W - Q) * mask).pow(2).sum().item()
    diff = ((Q - Q_braq) * mask).pow(2).sum().item()
    print(f"coupling={c:.1f}: MSE={err:.4f} (delta from BRAQ: {diff:.6f}, vs BRAQ MSE: {err/braq_err:.4f}x)")

# Also check: what are the joint alphas vs BRAQ alphas?
print("\n--- Alpha analysis for first 5 rows ---")
nan_val = torch.tensor(float('nan'), device=W.device, dtype=W.dtype)
new_matrix = W.clone() * mask

# BRAQ first expansion
masked_x = torch.where(mask, new_matrix, nan_val)
mean1 = torch.nanmean(masked_x, dim=1)
mean1 = torch.where(torch.isnan(mean1), torch.zeros_like(mean1), mean1)
masked_x -= mean1[:, None]
alpha1_braq = torch.nanmean(torch.abs(masked_x), dim=1)
alpha1_braq = torch.where(torch.isnan(alpha1_braq), torch.zeros_like(alpha1_braq), alpha1_braq)
B1 = torch.sign(masked_x)

exp1 = (B1 * alpha1_braq[:, None] + mean1[:, None]) * mask
residual = new_matrix - exp1
masked_r = torch.where(mask, residual, nan_val)
mean2 = torch.nanmean(masked_r, dim=1)
mean2 = torch.where(torch.isnan(mean2), torch.zeros_like(mean2), mean2)
masked_r -= mean2[:, None]
alpha2_braq = torch.nanmean(torch.abs(masked_r), dim=1)
alpha2_braq = torch.where(torch.isnan(alpha2_braq), torch.zeros_like(alpha2_braq), alpha2_braq)
B2 = torch.sign(masked_r)

# Joint solve (float32)
mask_f = mask.float()
d = mask_f.sum(dim=1)
d_safe = torch.clamp(d, min=1.0)
centered = (new_matrix - mean1[:, None]) * mask_f
c12 = (B1 * B2 * mask_f).sum(dim=1)
c1w = (centered * B1).sum(dim=1)
c2w = (centered * B2).sum(dim=1)

A = d_safe
denom = A * A - c12 * c12

alpha1_j_raw = (A * c1w - c12 * c2w) / denom
alpha2_j_raw = (A * c2w - c12 * c1w) / denom

for i in range(5):
    inf1 = (alpha1_j_raw[i] / alpha1_braq[i].float()).item() if alpha1_braq[i] > 0 else float('inf')
    inf2 = (alpha2_j_raw[i] / alpha2_braq[i].float()).item() if alpha2_braq[i] > 0 else float('inf')
    print(f"Row {i}: a1_braq={alpha1_braq[i].item():.6f}, a1_joint={alpha1_j_raw[i].item():.6f} ({inf1:.3f}x), "
          f"a2_braq={alpha2_braq[i].item():.6f}, a2_joint={alpha2_j_raw[i].item():.6f} ({inf2:.3f}x), "
          f"c12/d={c12[i].item()/d[i].item():.3f}")

# Check for extreme rows
ratio1 = alpha1_j_raw / alpha1_braq.float().clamp(min=1e-10)
ratio2 = alpha2_j_raw / alpha2_braq.float().clamp(min=1e-10)
print(f"\nalpha1 inflation: min={ratio1.min():.3f}, max={ratio1.max():.3f}, mean={ratio1.mean():.3f}, median={ratio1.median():.3f}")
print(f"alpha2 inflation: min={ratio2.min():.3f}, max={ratio2.max():.3f}, mean={ratio2.mean():.3f}, median={ratio2.median():.3f}")
print(f"Rows with a1_inflation > 3x: {(ratio1 > 3).sum().item()}/{len(ratio1)}")
print(f"Rows with a2_inflation > 3x: {(ratio2 > 3).sum().item()}/{len(ratio2)}")
