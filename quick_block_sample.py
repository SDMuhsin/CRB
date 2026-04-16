#!/usr/bin/env python3
"""
Quick block sampling: Check more blocks to understand the CRB vs BRAQ pattern.
Sample blocks from layers 0, 6, 12, 18, 23 across all sublayers.
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
from modelutils import find_layers

device = torch.device("cuda:0")
blocksize = 128

from run import get_model
model = get_model("facebook/opt-1.3b")
model.eval()
layers = model.model.decoder.layers

results = []
sample_layers = [0, 6, 12, 18, 23]

for layer_idx in sample_layers:
    layer = layers[layer_idx]
    subset = find_layers(layer)

    for sublayer_name, sublayer in subset.items():
        W = sublayer.weight.data.clone().float().to(device)
        oc, ic = W.shape

        # Sample blocks: first, middle, last
        block_starts = [0, (ic // blocksize // 2) * blocksize, ((ic // blocksize) - 1) * blocksize]

        for col_st in block_starts:
            if col_st >= ic:
                continue
            col_ed = min(col_st + blocksize, ic)
            W_block = W[:, col_st:col_ed]
            n_cols = col_ed - col_st

            H_block = torch.eye(n_cols, device=device)
            binary.index = 0
            mask1, mask2, mask3 = structural_guassian_distribution(W_block, H_block, "magnitude", 50)

            mask_f = mask3.float()
            n_salient = mask3.sum().item()
            d = mask_f.sum(dim=1)
            d_safe = torch.clamp(d, min=1.0)

            if n_salient == 0:
                continue

            # BRAQ
            binary.index = 0
            braq_q = high_order_residual(W_block, mask3, order=2)
            braq_err = ((W_block - braq_q) ** 2 * mask_f).sum().item() / n_salient

            # CRB v7
            binary.index = 0
            crb_q = coupled_residual_binarization_stable_v7(W_block, mask3, order=2, lam=1e-5, corr_damp=0.1)
            crb_err = ((W_block - crb_q) ** 2 * mask_f).sum().item() / n_salient

            # Compute mu2
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
            alpha1_mean = alpha1[valid_rows].mean().item()

            # Compute mean(sign(centered)) — the skewness indicator
            B1_mean = (B1 * mask_f).sum(dim=1) / d_safe
            B1_mean_abs = B1_mean[valid_rows].abs().mean().item()

            winner = "CRB" if crb_err < braq_err else "BRAQ"
            delta = crb_err - braq_err
            mu2_ratio = mu2_abs_mean / (alpha1_mean + 1e-12)

            results.append({
                "layer": layer_idx,
                "sublayer": sublayer_name,
                "block": col_st,
                "braq_err": braq_err,
                "crb_err": crb_err,
                "delta": delta,
                "winner": winner,
                "mu2_ratio": mu2_ratio,
                "B1_mean_abs": B1_mean_abs,
                "salient_frac": n_salient / mask3.numel(),
            })

        del W

# Print summary
print("\n" + "=" * 100)
print(f"{'Layer':>5} {'Sublayer':>12} {'Block':>5} {'BRAQ MSE':>12} {'CRB MSE':>12} {'Delta':>12} {'Winner':>6} {'mu2/a1':>8} {'|B1_mean|':>10} {'Sal%':>6}")
print("=" * 100)

crb_wins = 0
braq_wins = 0
crb_delta_sum = 0
braq_delta_sum = 0

for r in results:
    print(f"{r['layer']:>5} {r['sublayer']:>12} {r['block']:>5} {r['braq_err']:>12.6e} {r['crb_err']:>12.6e} {r['delta']:>12.6e} {r['winner']:>6} {r['mu2_ratio']:>8.4f} {r['B1_mean_abs']:>10.4f} {r['salient_frac']*100:>5.1f}%")
    if r["winner"] == "CRB":
        crb_wins += 1
        crb_delta_sum += abs(r["delta"])
    else:
        braq_wins += 1
        braq_delta_sum += abs(r["delta"])

print(f"\nTotal: {len(results)} blocks")
print(f"CRB wins: {crb_wins} ({100*crb_wins/len(results):.1f}%), avg magnitude of advantage: {crb_delta_sum/max(crb_wins,1):.6e}")
print(f"BRAQ wins: {braq_wins} ({100*braq_wins/len(results):.1f}%), avg magnitude of advantage: {braq_delta_sum/max(braq_wins,1):.6e}")

# Check correlation
import numpy as np
deltas = [r["delta"] for r in results]
mu2s = [r["mu2_ratio"] for r in results]
b1_means = [r["B1_mean_abs"] for r in results]

print(f"\nCorrelation(delta_mse, mu2/alpha1): {np.corrcoef(deltas, mu2s)[0,1]:.4f}")
print(f"Correlation(delta_mse, |B1_mean|): {np.corrcoef(deltas, b1_means)[0,1]:.4f}")

# Separate by winner
braq_wins_mu2 = [r["mu2_ratio"] for r in results if r["winner"] == "BRAQ"]
crb_wins_mu2 = [r["mu2_ratio"] for r in results if r["winner"] == "CRB"]
print(f"\nAvg mu2/alpha1 where BRAQ wins: {np.mean(braq_wins_mu2):.4f}")
print(f"Avg mu2/alpha1 where CRB wins: {np.mean(crb_wins_mu2):.4f}")

braq_wins_b1 = [r["B1_mean_abs"] for r in results if r["winner"] == "BRAQ"]
crb_wins_b1 = [r["B1_mean_abs"] for r in results if r["winner"] == "CRB"]
print(f"Avg |B1_mean| where BRAQ wins: {np.mean(braq_wins_b1):.4f}")
print(f"Avg |B1_mean| where CRB wins: {np.mean(crb_wins_b1):.4f}")
