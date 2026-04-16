#!/usr/bin/env python3
"""
Quick test: Does CRB miss the mu2 offset? Compare on a single weight block.
"""
import torch
import sys, os

os.environ["TRANSFORMERS_CACHE"] = "./downloads"
os.environ["HF_HOME"] = "./downloads"
sys.path.insert(0, ".")

import binary
binary.index = 0

from binary import high_order_residual, coupled_residual_binarization_stable_v7
from utils.structure import structural_guassian_distribution

device = torch.device("cuda:0")

# Load model
from run import get_model
model = get_model("facebook/opt-1.3b")
model.eval()

layers = model.model.decoder.layers

# Test on a few representative blocks from different sublayers
from modelutils import find_layers

test_cases = [
    (0, "fc2", 0),    # Layer 0, fc2, block 0
    (0, "fc1", 0),    # Layer 0, fc1, block 0
    (12, "fc2", 0),   # Layer 12 (middle), fc2
    (23, "fc2", 0),   # Layer 23 (last), fc2
]

blocksize = 128

for layer_idx, sublayer_name, block_offset in test_cases:
    layer = layers[layer_idx]
    subset = find_layers(layer)

    if sublayer_name not in subset:
        # Try partial match
        for k in subset:
            if sublayer_name in k:
                sublayer_name = k
                break

    W = subset[sublayer_name].weight.data.clone().float().to(device)
    col_st = block_offset * blocksize
    col_ed = min(col_st + blocksize, W.shape[1])
    W_block = W[:, col_st:col_ed]
    n_cols = col_ed - col_st

    # Get masks (magnitude metric)
    H_block = torch.eye(n_cols, device=device)
    binary.index = 0
    mask1, mask2, mask3 = structural_guassian_distribution(W_block, H_block, "magnitude", 50)

    mask_f = mask3.float()
    n_salient = mask3.sum().item()
    d = mask_f.sum(dim=1)
    d_safe = torch.clamp(d, min=1.0)

    # === BRAQ ===
    binary.index = 0
    braq_q = high_order_residual(W_block, mask3, order=2)
    braq_err = ((W_block - braq_q) ** 2 * mask_f).sum().item() / n_salient

    # === CRB v7 ===
    binary.index = 0
    crb_q = coupled_residual_binarization_stable_v7(W_block, mask3, order=2, lam=1e-5, corr_damp=0.1)
    crb_err = ((W_block - crb_q) ** 2 * mask_f).sum().item() / n_salient

    # === Compute mu2 (BRAQ's second-pass offset) ===
    W_masked = W_block * mask_f
    mu1 = (W_masked).sum(dim=1) / d_safe
    centered = (W_masked - mu1[:, None]) * mask_f
    B1 = torch.sign(centered) * mask_f
    alpha1 = (centered.abs() * mask_f).sum(dim=1) / d_safe
    pass1 = (mu1[:, None] + alpha1[:, None] * B1) * mask_f
    residual = (W_masked - pass1) * mask_f
    mu2 = (residual * mask_f).sum(dim=1) / d_safe

    valid_rows = d > 0
    mu2_abs_mean = mu2[valid_rows].abs().mean().item()
    mu1_abs_mean = mu1[valid_rows].abs().mean().item()
    alpha1_mean = alpha1[valid_rows].mean().item()
    mu2_ratio = mu2_abs_mean / (alpha1_mean + 1e-12)

    # === CRB with mu_correction (proposed fix) ===
    # After CRB's optimization, add the mean of the residual as an offset correction
    crb_residual = (W_block - crb_q) * mask_f
    mu_correction = (crb_residual * mask_f).sum(dim=1) / d_safe
    crb_fixed_q = crb_q + mu_correction[:, None] * mask_f
    crb_fixed_err = ((W_block - crb_fixed_q) ** 2 * mask_f).sum().item() / n_salient

    # === CRB with full centering fix ===
    # Re-implement CRB with centering in refinement steps
    binary.index = 0
    def crb_with_centering(x, mask, lam=1e-5, corr_damp=0.1):
        sum_order = torch.zeros_like(x)
        new_matrix = x.clone() * mask
        mask_f = mask.float()
        d = mask_f.sum(dim=1)
        d_safe = torch.clamp(d, min=1.0)

        row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
        centered = (new_matrix - row_mean[:, None]) * mask_f

        # Step 2: B1
        B1 = torch.sign(centered) * mask_f
        alpha1_init = (centered.abs() * mask_f).sum(dim=1) / d_safe

        # Step 3: B2 with centering
        r = (centered - alpha1_init[:, None] * B1) * mask_f
        r_mean = (r * mask_f).sum(dim=1) / d_safe
        r_centered = (r - r_mean[:, None]) * mask_f
        B2 = torch.sign(r_centered) * mask_f

        def solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp):
            c12 = (B1 * B2 * mask_f).sum(dim=1)
            c12 = torch.where(c12 > 0, c12 * (1.0 - corr_damp), c12)
            A = d + lam
            denom = A * A - c12 * c12
            safe = denom.abs() > 1e-12
            safe_denom = torch.where(safe, denom, torch.ones_like(denom))
            a1 = torch.clamp((A * c1w - c12 * c2w) / safe_denom, min=0.0)
            a2 = torch.clamp((A * c2w - c12 * c1w) / safe_denom, min=0.0)
            a1 = torch.where(safe, a1, torch.zeros_like(a1))
            a2 = torch.where(safe, a2, torch.zeros_like(a2))
            return a1, a2

        # Step 4: Joint solve
        c1w = (centered * B1).sum(dim=1)
        c2w = (centered * B2).sum(dim=1)
        alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

        # Step 5: Refine B2 WITH centering
        temp = (centered - alpha1[:, None] * B1) * mask_f
        temp_mean = (temp * mask_f).sum(dim=1) / d_safe
        B2 = torch.sign((temp - temp_mean[:, None]) * mask_f) * mask_f
        c2w = (centered * B2).sum(dim=1)
        alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

        # Step 6: Refine B1 WITH centering
        temp = (centered - alpha2[:, None] * B2) * mask_f
        temp_mean = (temp * mask_f).sum(dim=1) / d_safe
        B1 = torch.sign((temp - temp_mean[:, None]) * mask_f) * mask_f
        c1w = (centered * B1).sum(dim=1)
        alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

        # Reconstruct WITH residual mean correction
        approx = (alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f
        residual_after = (centered - approx) * mask_f
        mu_corr = (residual_after * mask_f).sum(dim=1) / d_safe

        sum_order = (row_mean[:, None] + mu_corr[:, None] + alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f
        return sum_order

    crb_centered_q = crb_with_centering(W_block, mask3)
    crb_centered_err = ((W_block - crb_centered_q) ** 2 * mask_f).sum().item() / n_salient

    print(f"\n{'='*70}")
    print(f"Layer {layer_idx}, {sublayer_name}, block {col_st}")
    print(f"  Salient fraction: {n_salient/mask3.numel()*100:.1f}%")
    print(f"  |mu2|/alpha1 ratio: {mu2_ratio:.4f}")
    print(f"  BRAQ MSE:           {braq_err:.6e}")
    print(f"  CRB  MSE:           {crb_err:.6e}")
    print(f"  CRB+offset MSE:     {crb_fixed_err:.6e}")
    print(f"  CRB+centering MSE:  {crb_centered_err:.6e}")
    print(f"  Delta (CRB-BRAQ):   {crb_err - braq_err:.6e} {'CRB WORSE' if crb_err > braq_err else 'CRB better'}")
    print(f"  Delta (fixed-BRAQ): {crb_fixed_err - braq_err:.6e} {'STILL WORSE' if crb_fixed_err > braq_err else 'FIXED'}")
    print(f"  Delta (cent-BRAQ):  {crb_centered_err - braq_err:.6e} {'STILL WORSE' if crb_centered_err > braq_err else 'FIXED'}")

    del W
