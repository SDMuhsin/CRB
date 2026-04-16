#!/usr/bin/env python3
"""
Diagnostic: compare CRB v7 vs BRAQ per-block reconstruction error.

For each layer/sublayer/column-block:
  1. Compute structural partition masks (same for both methods)
  2. Call BRAQ (high_order_residual) and CRB (coupled_residual_binarization_stable_v7)
     on the salient partition (mask3, order=2)
  3. Compare MSE: ||W - approx||^2

This isolates the binarization quality from GPTQ error propagation.

Usage:
  python3 -u diagnostic_crb_vs_braq.py Qwen/Qwen3-0.6B
  python3 -u diagnostic_crb_vs_braq.py facebook/opt-1.3b
"""
import sys
import os
import json
import time
import torch
import numpy as np
import torch.nn as nn

os.environ["TRANSFORMERS_CACHE"] = "./downloads"
os.environ["HF_HOME"] = "./downloads"
sys.path.insert(0, ".")

import binary
from binary import high_order_residual, coupled_residual_binarization_stable_v7
from utils.structure import structural_guassian_distribution
from modelutils import find_layers


def crb_diagnostic(W_block, mask3, lam=1e-5, corr_damp=0.1):
    """
    Run CRB v7 with diagnostic output — returns reconstruction + intermediate values.
    This is a copy of CRB v7 logic with added instrumentation.
    """
    mask_f = mask3.float()
    d = mask_f.sum(dim=1)
    d_safe = torch.clamp(d, min=1.0)
    active_rows = d > 0
    n_active = active_rows.sum().item()

    # Step 1: Row mean and centering
    row_mean = (W_block * mask_f).sum(dim=1) / d_safe
    centered = (W_block - row_mean[:, None]) * mask_f

    # Step 2: B1
    B1_init = torch.sign(centered) * mask_f
    alpha1_init = (centered.abs() * mask_f).sum(dim=1) / d_safe

    # Step 3: Residual -> B2
    r = (centered - alpha1_init[:, None] * B1_init) * mask_f
    r_mean = (r * mask_f).sum(dim=1) / d_safe
    r_centered = (r - r_mean[:, None]) * mask_f
    B2_init = torch.sign(r_centered) * mask_f

    B1, B2 = B1_init.clone(), B2_init.clone()

    def solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp):
        c12 = (B1 * B2 * mask_f).sum(dim=1)
        c12_raw = c12.clone()
        c12 = torch.where(c12 > 0, c12 * (1.0 - corr_damp), c12)
        A = d + lam
        denom = A * A - c12 * c12
        safe = denom.abs() > 1e-12
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        a1_raw = (A * c1w - c12 * c2w) / safe_denom
        a2_raw = (A * c2w - c12 * c1w) / safe_denom
        a1 = torch.clamp(a1_raw, min=0.0)
        a2 = torch.clamp(a2_raw, min=0.0)
        a1 = torch.where(safe, a1, torch.zeros_like(a1))
        a2 = torch.where(safe, a2, torch.zeros_like(a2))
        return a1, a2, c12_raw, a1_raw, a2_raw, denom

    # Step 4: Initial joint alpha solve
    c1w = (centered * B1).sum(dim=1)
    c2w = (centered * B2).sum(dim=1)
    alpha1_s4, alpha2_s4, c12_s4, a1r_s4, a2r_s4, denom_s4 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

    # Step 5: Refine B2
    temp5 = (centered - alpha1_s4[:, None] * B1) * mask_f
    temp5_mean = (temp5 * mask_f).sum(dim=1) / d_safe
    B2 = torch.sign((temp5 - temp5_mean[:, None]) * mask_f) * mask_f
    c2w = (centered * B2).sum(dim=1)
    alpha1_s5, alpha2_s5, c12_s5, a1r_s5, a2r_s5, denom_s5 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

    # How many B2 signs changed in step 5?
    b2_changed = ((B2_init != B2) & (mask_f > 0)).float().sum() / mask_f.sum()

    # Step 6: Refine B1
    temp6 = (centered - alpha2_s5[:, None] * B2) * mask_f
    temp6_mean = (temp6 * mask_f).sum(dim=1) / d_safe
    B1_new = torch.sign((temp6 - temp6_mean[:, None]) * mask_f) * mask_f
    c1w_new = (centered * B1_new).sum(dim=1)
    alpha1_s6, alpha2_s6, c12_s6, a1r_s6, a2r_s6, denom_s6 = solve_alphas_vec(B1_new, B2, c1w_new, c2w, d, lam, corr_damp)

    # How many B1 signs changed in step 6?
    b1_changed = ((B1_init != B1_new) & (mask_f > 0)).float().sum() / mask_f.sum()

    # Step 7: Reconstruction with mu_correction
    approx = (alpha1_s6[:, None] * B1_new + alpha2_s6[:, None] * B2) * mask_f
    residual_final = (centered - approx) * mask_f
    mu_correction = (residual_final * mask_f).sum(dim=1) / d_safe

    reconstruction = (row_mean[:, None] + mu_correction[:, None] + alpha1_s6[:, None] * B1_new + alpha2_s6[:, None] * B2) * mask_f

    # Now compute BRAQ's mu2 for comparison
    # BRAQ pass 1
    braq_pass1 = (row_mean[:, None] + alpha1_init[:, None] * B1_init) * mask_f
    braq_resid = (W_block - braq_pass1) * mask_f
    braq_mu2 = (braq_resid * mask_f).sum(dim=1) / d_safe

    diag = {}
    if n_active > 0:
        ar = active_rows
        diag = {
            "n_active_rows": int(n_active),
            # Step 4
            "s4_a1_neg_frac": float((a1r_s4[ar] < 0).float().mean()),
            "s4_a2_neg_frac": float((a2r_s4[ar] < 0).float().mean()),
            "s4_c12_mean": float(c12_s4[ar].mean()),
            "s4_c12_over_d": float((c12_s4[ar] / d_safe[ar]).mean()),
            "s4_alpha1_mean": float(alpha1_s4[ar].mean()),
            "s4_alpha2_mean": float(alpha2_s4[ar].mean()),
            "s4_cond": float((d_safe[ar] ** 2 / denom_s4[ar].abs().clamp(min=1e-30)).mean()),
            # Step 5 (after B2 refinement)
            "s5_a1_neg_frac": float((a1r_s5[ar] < 0).float().mean()),
            "s5_a2_neg_frac": float((a2r_s5[ar] < 0).float().mean()),
            "s5_c12_mean": float(c12_s5[ar].mean()),
            "s5_alpha1_mean": float(alpha1_s5[ar].mean()),
            "s5_alpha2_mean": float(alpha2_s5[ar].mean()),
            "b2_sign_changed_frac": float(b2_changed),
            # Step 6 (after B1 refinement)
            "s6_a1_neg_frac": float((a1r_s6[ar] < 0).float().mean()),
            "s6_a2_neg_frac": float((a2r_s6[ar] < 0).float().mean()),
            "s6_c12_mean": float(c12_s6[ar].mean()),
            "s6_alpha1_mean": float(alpha1_s6[ar].mean()),
            "s6_alpha2_mean": float(alpha2_s6[ar].mean()),
            "s6_alpha1_zero_frac": float((alpha1_s6[ar] == 0).float().mean()),
            "s6_alpha2_zero_frac": float((alpha2_s6[ar] == 0).float().mean()),
            "b1_sign_changed_frac": float(b1_changed),
            # mu_correction
            "mu_correction_mean": float(mu_correction[ar].mean()),
            "mu_correction_abs_mean": float(mu_correction[ar].abs().mean()),
            # BRAQ mu2 for comparison
            "braq_mu2_abs_mean": float(braq_mu2[ar].abs().mean()),
            "braq_mu2_over_alpha1": float((braq_mu2[ar].abs() / alpha1_init[ar].clamp(min=1e-12)).mean()),
        }

    return reconstruction, diag


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    device = "cuda:0"
    blocksize = 128

    print(f"{'='*80}")
    print(f"DIAGNOSTIC: CRB vs BRAQ Reconstruction Error")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    # Load model
    print(f"\nLoading model...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        cache_dir="./downloads",
    )
    model.eval()

    # Get layers based on model type
    if "opt" in model_name.lower():
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    n_layers = len(layers)
    print(f"Model has {n_layers} layers")

    results = []
    n_total = 0
    n_crb_better = 0
    n_crb_worse = 0
    total_braq_sse = 0.0
    total_crb_sse = 0.0
    t_start = time.time()

    for layer_idx in range(n_layers):
        layer = layers[layer_idx]
        sublayers = find_layers(layer)

        for sub_name in sorted(sublayers.keys()):
            sub_module = sublayers[sub_name]
            if not isinstance(sub_module, nn.Linear):
                continue

            W = sub_module.weight.data.clone().float().to(device)
            oc, ic = W.shape

            for col_st in range(0, ic, blocksize):
                col_ed = min(col_st + blocksize, ic)
                W_block = W[:, col_st:col_ed]
                n_cols = col_ed - col_st

                # Get structural partition masks
                H_block = torch.eye(n_cols, device=device)
                binary.index = 0
                mask1, mask2, mask3 = structural_guassian_distribution(
                    W_block, H_block, "magnitude", 50
                )

                n_salient_cols = int(mask3.any(dim=0).sum())
                n_salient_elems = int(mask3.sum())

                if n_salient_elems == 0:
                    continue

                # BRAQ on salient partition
                binary.index = 0
                braq_out = high_order_residual(W_block, mask3, order=2)
                braq_sse = ((W_block - braq_out) ** 2 * mask3.float()).sum().item()

                # CRB on salient partition (with diagnostics)
                binary.index = 0
                crb_out, crb_diag = crb_diagnostic(W_block, mask3)
                crb_sse = ((W_block - crb_out) ** 2 * mask3.float()).sum().item()

                # Verify diagnostic CRB matches library CRB
                binary.index = 0
                crb_lib_out = coupled_residual_binarization_stable_v7(W_block, mask3, order=2)
                crb_lib_sse = ((W_block - crb_lib_out) ** 2 * mask3.float()).sum().item()
                lib_match = abs(crb_sse - crb_lib_sse) / (crb_lib_sse + 1e-30) < 1e-4

                ratio = crb_sse / braq_sse if braq_sse > 1e-30 else float('nan')
                crb_wins = crb_sse < braq_sse

                # Weight distribution stats for salient partition
                vals = W_block[mask3].float()
                w_stats = {
                    "abs_mean": float(vals.abs().mean()),
                    "std": float(vals.std()),
                    "kurtosis": float((((vals - vals.mean()) / (vals.std() + 1e-12)) ** 4).mean()) if vals.numel() > 1 else 0,
                }

                entry = {
                    "layer": layer_idx,
                    "sublayer": sub_name,
                    "col_block": col_st,
                    "shape": [oc, n_cols],
                    "n_salient_cols": n_salient_cols,
                    "n_salient_elems": n_salient_elems,
                    "braq_sse": braq_sse,
                    "crb_sse": crb_sse,
                    "crb_lib_match": lib_match,
                    "ratio": ratio,
                    "crb_wins": bool(crb_wins),
                    "w_stats": w_stats,
                    "crb_diag": crb_diag,
                }
                results.append(entry)

                total_braq_sse += braq_sse
                total_crb_sse += crb_sse
                if crb_wins:
                    n_crb_better += 1
                else:
                    n_crb_worse += 1
                n_total += 1

                status = "WORSE" if not crb_wins else "better"
                if not lib_match:
                    status += " !MISMATCH!"
                print(f"L{layer_idx:02d} {sub_name:12s} col{col_st:4d} | "
                      f"sal={n_salient_cols:2d}col | "
                      f"BRAQ={braq_sse:.6f} CRB={crb_sse:.6f} ratio={ratio:.4f} [{status}]")

            del W

        # Free layer
        layer.cpu()
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        print(f"  --- Layer {layer_idx} done ({elapsed:.0f}s elapsed, "
              f"CRB better: {n_crb_better}, worse: {n_crb_worse}) ---")

    # ===== SUMMARY =====
    print(f"\n{'='*80}")
    print(f"SUMMARY for {model_name}")
    print(f"{'='*80}")
    print(f"  Total blocks: {n_total}")
    print(f"  CRB better: {n_crb_better} ({100*n_crb_better/n_total:.1f}%)")
    print(f"  CRB worse:  {n_crb_worse} ({100*n_crb_worse/n_total:.1f}%)")
    print(f"  Total BRAQ SSE: {total_braq_sse:.6f}")
    print(f"  Total CRB SSE:  {total_crb_sse:.6f}")
    print(f"  Overall ratio:  {total_crb_sse/total_braq_sse:.4f}")

    # Per-sublayer breakdown
    print(f"\nPer-sublayer summary:")
    sublayer_types = sorted(set(r["sublayer"] for r in results))
    for st in sublayer_types:
        sr = [r for r in results if r["sublayer"] == st]
        braq_t = sum(r["braq_sse"] for r in sr)
        crb_t = sum(r["crb_sse"] for r in sr)
        n_w = sum(1 for r in sr if not r["crb_wins"])
        print(f"  {st:20s}: {len(sr):3d} blocks | "
              f"BRAQ={braq_t:10.4f} CRB={crb_t:10.4f} ratio={crb_t/braq_t:.4f} | "
              f"CRB worse: {n_w}/{len(sr)}")

    # Worst CRB blocks
    worse = sorted([r for r in results if not r["crb_wins"]],
                   key=lambda r: r["ratio"], reverse=True)
    print(f"\nTop 20 blocks where CRB is WORST:")
    print(f"  {'L':>2} {'sublayer':>20} {'col':>4} {'ratio':>7} {'BRAQ':>10} {'CRB':>10} "
          f"{'sal':>3} {'a1neg%':>6} {'a2neg%':>6} {'c12/d':>6} {'B2chg%':>6} {'B1chg%':>6}")
    for r in worse[:20]:
        d = r["crb_diag"]
        print(f"  {r['layer']:>2} {r['sublayer']:>20} {r['col_block']:>4} {r['ratio']:>7.4f} "
              f"{r['braq_sse']:>10.6f} {r['crb_sse']:>10.6f} "
              f"{r['n_salient_cols']:>3} "
              f"{d.get('s6_a1_neg_frac',0)*100:>5.1f}% "
              f"{d.get('s6_a2_neg_frac',0)*100:>5.1f}% "
              f"{d.get('s4_c12_over_d',0):>6.3f} "
              f"{d.get('b2_sign_changed_frac',0)*100:>5.1f}% "
              f"{d.get('b1_sign_changed_frac',0)*100:>5.1f}%")

    # Best CRB blocks
    better = sorted([r for r in results if r["crb_wins"]],
                    key=lambda r: r["ratio"])
    print(f"\nTop 10 blocks where CRB is BEST:")
    for r in better[:10]:
        d = r["crb_diag"]
        print(f"  L{r['layer']:>2} {r['sublayer']:>20} col{r['col_block']:>4} | "
              f"ratio={r['ratio']:.4f} | BRAQ={r['braq_sse']:.6f} CRB={r['crb_sse']:.6f}")

    # Diagnostic patterns: compare CRB-worse vs CRB-better blocks
    for label, block_list in [("CRB-WORSE", worse), ("CRB-BETTER", better)]:
        if not block_list:
            continue
        print(f"\n  Diagnostic patterns for {label} blocks ({len(block_list)}):")
        keys = ["s4_a1_neg_frac", "s4_a2_neg_frac", "s4_c12_over_d",
                "s6_a1_neg_frac", "s6_a2_neg_frac",
                "s6_alpha1_zero_frac", "s6_alpha2_zero_frac",
                "b2_sign_changed_frac", "b1_sign_changed_frac",
                "mu_correction_abs_mean", "braq_mu2_abs_mean", "braq_mu2_over_alpha1",
                "s6_alpha1_mean", "s6_alpha2_mean"]
        for k in keys:
            vals = [r["crb_diag"].get(k, 0) for r in block_list]
            print(f"    {k:30s}: mean={np.mean(vals):.6f}  median={np.median(vals):.6f}")

    # Verify CRB library match
    mismatches = [r for r in results if not r["crb_lib_match"]]
    if mismatches:
        print(f"\n  WARNING: {len(mismatches)} blocks where diagnostic CRB != library CRB!")
    else:
        print(f"\n  All {n_total} blocks: diagnostic CRB matches library CRB")

    # Save results
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    output_path = f"{output_dir}/diagnostic_crb_vs_braq_{safe_name}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model_name,
            "summary": {
                "n_total": n_total,
                "n_crb_better": n_crb_better,
                "n_crb_worse": n_crb_worse,
                "total_braq_sse": total_braq_sse,
                "total_crb_sse": total_crb_sse,
                "overall_ratio": total_crb_sse / total_braq_sse if total_braq_sse > 0 else None,
            },
            "blocks": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
