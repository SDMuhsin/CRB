#!/usr/bin/env python3
"""
Sign Refinement Diagnostic: What exactly does sign refinement do to B1/B2?

For each model & sublayer weight matrix:
  1. Run CRB Steps 1-4 (no refinement) → B1_init, B2_init, alpha_init
  2. Run CRB Step 5 (refine B2 only) → B2_step5
  3. Run CRB Step 6 (refine B1 only) → B1_step6
  4. Record: sign flip counts, alpha ratios, Frobenius errors, correlation changes

Compares across models to find what distinguishes winning (OPT) from losing (Pythia, Qwen3).

Usage:
  source env/bin/activate
  export TRANSFORMERS_CACHE="./downloads" HF_HOME="./downloads"
  python3 -u src/sign_refinement_diagnostic.py
"""
import sys, os, gc, json
import torch
import numpy as np

os.environ["TRANSFORMERS_CACHE"] = "./downloads"
os.environ["HF_HOME"] = "./downloads"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda:0"
SALIENT_FRAC = 0.18  # Top 18% columns are salient (same as production)


def make_salient_mask(W, frac=0.18):
    """Create salient column mask (top frac by column magnitude sum)."""
    col_scores = W.abs().sum(dim=0)
    k = max(1, int(round(frac * W.shape[1])))
    threshold = torch.topk(col_scores, k).values[-1]
    col_mask = col_scores >= threshold
    return col_mask.unsqueeze(0).expand_as(W)


@torch.no_grad()
def crb_with_intermediates(x, mask, lam=1e-5, corr_damp=0.1):
    """
    Run CRB v7 step by step and return all intermediates.
    Returns dict with B1/B2/alphas at each stage and error metrics.
    """
    new_matrix = x.clone() * mask
    mask_f = mask.float()
    d = mask_f.sum(dim=1)
    d_safe = torch.clamp(d, min=1.0)

    def solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp):
        c12 = (B1 * B2 * mask_f).sum(dim=1)
        c12_raw = c12.clone()
        c12 = torch.where(c12 > 0, c12 * (1.0 - corr_damp), c12)
        A = d + lam
        denom = A * A - c12 * c12
        safe = denom.abs() > 1e-12
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        a1 = torch.clamp((A * c1w - c12 * c2w) / safe_denom, min=0.0)
        a2 = torch.clamp((A * c2w - c12 * c1w) / safe_denom, min=0.0)
        a1 = torch.where(safe, a1, torch.zeros_like(a1))
        a2 = torch.where(safe, a2, torch.zeros_like(a2))
        return a1, a2, c12_raw

    def frobenius_error(centered, alpha1, B1, alpha2, B2, mu_corr, mask_f, d_safe):
        approx = (alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f
        residual = (centered - approx) * mask_f
        if mu_corr is None:
            mu_corr = (residual * mask_f).sum(dim=1) / d_safe
        full_approx = mu_corr[:, None] + approx
        err = ((centered - full_approx) * mask_f)
        frob = (err ** 2).sum().item()
        norm = (centered ** 2).sum().item()
        return frob / max(norm, 1e-30), frob

    # Step 1: Center
    row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
    centered = (new_matrix - row_mean[:, None]) * mask_f

    # Step 2: B1 = sign(centered)
    B1_init = torch.sign(centered) * mask_f
    alpha1_raw = (centered.abs() * mask_f).sum(dim=1) / d_safe

    # Step 3: Residual -> B2
    r = (centered - alpha1_raw[:, None] * B1_init) * mask_f
    r_mean = (r * mask_f).sum(dim=1) / d_safe
    r_centered = (r - r_mean[:, None]) * mask_f
    B2_init = torch.sign(r_centered) * mask_f

    # Step 4: Joint solve
    c1w_init = (centered * B1_init).sum(dim=1)
    c2w_init = (centered * B2_init).sum(dim=1)
    alpha1_step4, alpha2_step4, c12_step4 = solve_alphas_vec(
        B1_init, B2_init, c1w_init, c2w_init, d, lam, corr_damp
    )

    # Error after Step 4 (no refinement)
    rel_err_step4, abs_err_step4 = frobenius_error(
        centered, alpha1_step4, B1_init, alpha2_step4, B2_init, None, mask_f, d_safe
    )

    # ---- Step 5: Refine B2 ----
    B1_for_step5 = B1_init.clone()
    temp5 = (centered - alpha1_step4[:, None] * B1_for_step5) * mask_f
    temp5_mean = (temp5 * mask_f).sum(dim=1) / d_safe
    B2_step5 = torch.sign((temp5 - temp5_mean[:, None]) * mask_f) * mask_f
    c2w_step5 = (centered * B2_step5).sum(dim=1)
    alpha1_step5, alpha2_step5, c12_step5 = solve_alphas_vec(
        B1_for_step5, B2_step5, c1w_init, c2w_step5, d, lam, corr_damp
    )

    rel_err_step5, abs_err_step5 = frobenius_error(
        centered, alpha1_step5, B1_for_step5, alpha2_step5, B2_step5, None, mask_f, d_safe
    )

    # ---- Step 6: Refine B1 (using Step 5's B2 and alphas) ----
    temp6 = (centered - alpha2_step5[:, None] * B2_step5) * mask_f
    temp6_mean = (temp6 * mask_f).sum(dim=1) / d_safe
    B1_step6 = torch.sign((temp6 - temp6_mean[:, None]) * mask_f) * mask_f
    c1w_step6 = (centered * B1_step6).sum(dim=1)
    alpha1_step6, alpha2_step6, c12_step6 = solve_alphas_vec(
        B1_step6, B2_step5, c1w_step6, c2w_step5, d, lam, corr_damp
    )

    rel_err_step6, abs_err_step6 = frobenius_error(
        centered, alpha1_step6, B1_step6, alpha2_step6, B2_step5, None, mask_f, d_safe
    )

    # ---- Compute sign flip statistics ----
    n_salient = mask_f.sum().item()

    # B2 flips from Step 5
    b2_flips = ((B2_step5 != B2_init) & mask).sum().item()

    # B1 flips from Step 6
    b1_flips = ((B1_step6 != B1_init) & mask).sum().item()

    # B1 sign agreement with centered weights after Step 6
    sign_centered = torch.sign(centered) * mask_f
    b1_init_agrees = ((B1_init == sign_centered) & mask).sum().item()
    b1_step6_agrees = ((B1_step6 == sign_centered) & mask).sum().item()

    # Alpha ratios
    alpha1_mean = alpha1_step4.mean().item()
    alpha2_mean = alpha2_step4.mean().item()
    alpha_ratio = alpha2_mean / max(alpha1_mean, 1e-30)

    alpha1_final_mean = alpha1_step6.mean().item()
    alpha2_final_mean = alpha2_step6.mean().item()

    # Correlation between B1 and B2 at each stage
    corr_init = c12_step4.mean().item() / max(d.mean().item(), 1)
    corr_step5 = c12_step5.mean().item() / max(d.mean().item(), 1)
    corr_step6 = c12_step6.mean().item() / max(d.mean().item(), 1)

    # Weight magnitude at flipped B1 positions vs non-flipped
    b1_flip_mask = (B1_step6 != B1_init) & mask
    b1_noflip_mask = (B1_step6 == B1_init) & mask
    mag_at_flips = centered.abs()[b1_flip_mask].mean().item() if b1_flip_mask.any() else 0.0
    mag_at_noflips = centered.abs()[b1_noflip_mask].mean().item() if b1_noflip_mask.any() else 0.0

    return {
        "n_salient": n_salient,
        # Error metrics
        "rel_err_step4": rel_err_step4,
        "rel_err_step5": rel_err_step5,
        "rel_err_step6": rel_err_step6,
        "err_reduction_step5": (rel_err_step4 - rel_err_step5) / max(rel_err_step4, 1e-30),
        "err_reduction_step6": (rel_err_step5 - rel_err_step6) / max(rel_err_step5, 1e-30),
        # Sign flips
        "b2_flip_frac": b2_flips / max(n_salient, 1),
        "b1_flip_frac": b1_flips / max(n_salient, 1),
        "b2_flips": b2_flips,
        "b1_flips": b1_flips,
        # B1 alignment with weight signs
        "b1_init_sign_agreement": b1_init_agrees / max(n_salient, 1),
        "b1_step6_sign_agreement": b1_step6_agrees / max(n_salient, 1),
        "b1_sign_agreement_drop": (b1_init_agrees - b1_step6_agrees) / max(n_salient, 1),
        # Alpha statistics
        "alpha1_init": alpha1_mean,
        "alpha2_init": alpha2_mean,
        "alpha_ratio": alpha_ratio,
        "alpha1_final": alpha1_final_mean,
        "alpha2_final": alpha2_final_mean,
        # Correlations
        "b1b2_corr_init": corr_init,
        "b1b2_corr_step5": corr_step5,
        "b1b2_corr_step6": corr_step6,
        # Weight magnitude at flip locations
        "mag_at_b1_flips": mag_at_flips,
        "mag_at_b1_noflips": mag_at_noflips,
        "flip_mag_ratio": mag_at_flips / max(mag_at_noflips, 1e-30),
    }


def get_sublayers(model, model_name):
    """Get all weight sublayers for analysis."""
    sublayers = []

    if "opt" in model_name.lower():
        layers = model.model.decoder.layers
        for li, layer in enumerate(layers):
            for name in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]:
                if name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    mod = getattr(layer.self_attn, name)
                else:
                    mod = getattr(layer, name)
                sublayers.append((f"L{li:02d}.{name}", mod.weight.data))

    elif "pythia" in model_name.lower():
        layers = model.gpt_neox.layers
        for li, layer in enumerate(layers):
            for name in ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]:
                if name in ("query_key_value", "dense"):
                    mod = getattr(layer.attention, name)
                else:
                    mod = getattr(layer.mlp, name)
                sublayers.append((f"L{li:02d}.{name}", mod.weight.data))

    elif "qwen" in model_name.lower():
        layers = model.model.layers
        for li, layer in enumerate(layers):
            for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    mod = getattr(layer.self_attn, name)
                else:
                    mod = getattr(layer.mlp, name)
                sublayers.append((f"L{li:02d}.{name}", mod.weight.data))

    elif "bloom" in model_name.lower():
        layers = model.transformer.h
        for li, layer in enumerate(layers):
            for name in ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]:
                if name in ("query_key_value", "dense"):
                    mod = getattr(layer.self_attention, name)
                else:
                    mod = getattr(layer.mlp, name)
                sublayers.append((f"L{li:02d}.{name}", mod.weight.data))

    return sublayers


def main():
    models_to_test = [
        ("facebook/opt-1.3b",      "OPT-1.3B",    "WINS",  dict(torch_dtype="auto", use_safetensors=True)),
        ("EleutherAI/pythia-1.4b", "Pythia-1.4B",  "LOSES", dict(torch_dtype=torch.bfloat16)),
        ("Qwen/Qwen3-0.6B",       "Qwen3-0.6B",   "LOSES", dict(torch_dtype="auto")),
        ("bigscience/bloom-1b7",   "BLOOM-1.7B",    "WINS",  dict(torch_dtype="auto")),
    ]

    from transformers import AutoModelForCausalLM

    all_model_stats = {}

    for model_path, model_label, crb_status, load_kwargs in models_to_test:
        print("=" * 80)
        print(f"MODEL: {model_label} (CRB {crb_status})")
        print("=" * 80)

        model = AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir="./downloads", attn_implementation="eager",
            **load_kwargs
        )
        model.eval()

        sublayers = get_sublayers(model, model_path)
        print(f"  Found {len(sublayers)} sublayers")

        all_stats = []
        for name, W in sublayers:
            W_gpu = W.clone().to(DEVICE).float()
            mask = make_salient_mask(W_gpu, SALIENT_FRAC)

            stats = crb_with_intermediates(W_gpu, mask)
            stats["sublayer"] = name
            all_stats.append(stats)

            del W_gpu, mask
            torch.cuda.empty_cache()

        # Aggregate statistics
        b1_flip_fracs = [s["b1_flip_frac"] for s in all_stats]
        b2_flip_fracs = [s["b2_flip_frac"] for s in all_stats]
        sign_drops = [s["b1_sign_agreement_drop"] for s in all_stats]
        alpha_ratios = [s["alpha_ratio"] for s in all_stats]
        err_red_s5 = [s["err_reduction_step5"] for s in all_stats]
        err_red_s6 = [s["err_reduction_step6"] for s in all_stats]
        flip_mag_ratios = [s["flip_mag_ratio"] for s in all_stats]
        corr_inits = [s["b1b2_corr_init"] for s in all_stats]
        corr_step6s = [s["b1b2_corr_step6"] for s in all_stats]

        model_summary = {
            "model": model_label,
            "status": crb_status,
            "n_sublayers": len(all_stats),
            "b1_flip_frac_mean": np.mean(b1_flip_fracs),
            "b1_flip_frac_max": np.max(b1_flip_fracs),
            "b2_flip_frac_mean": np.mean(b2_flip_fracs),
            "b1_sign_agreement_drop_mean": np.mean(sign_drops),
            "b1_sign_agreement_drop_max": np.max(sign_drops),
            "alpha_ratio_mean": np.mean(alpha_ratios),
            "alpha_ratio_max": np.max(alpha_ratios),
            "err_reduction_step5_mean": np.mean(err_red_s5),
            "err_reduction_step6_mean": np.mean(err_red_s6),
            "flip_mag_ratio_mean": np.mean(flip_mag_ratios),
            "b1b2_corr_init_mean": np.mean(corr_inits),
            "b1b2_corr_step6_mean": np.mean(corr_step6s),
        }
        all_model_stats[model_label] = model_summary

        print(f"\n  --- SUMMARY for {model_label} (CRB {crb_status}) ---")
        print(f"  B1 sign flip fraction: mean={model_summary['b1_flip_frac_mean']:.4f}, max={model_summary['b1_flip_frac_max']:.4f}")
        print(f"  B2 sign flip fraction: mean={model_summary['b2_flip_frac_mean']:.4f}")
        print(f"  B1 sign agreement drop: mean={model_summary['b1_sign_agreement_drop_mean']:.4f}, max={model_summary['b1_sign_agreement_drop_max']:.4f}")
        print(f"  Alpha2/Alpha1 ratio:   mean={model_summary['alpha_ratio_mean']:.4f}, max={model_summary['alpha_ratio_max']:.4f}")
        print(f"  Error reduction Step5: mean={model_summary['err_reduction_step5_mean']:.4f}")
        print(f"  Error reduction Step6: mean={model_summary['err_reduction_step6_mean']:.6f}")
        print(f"  Flip magnitude ratio:  mean={model_summary['flip_mag_ratio_mean']:.4f}")
        print(f"  B1-B2 corr init:       mean={model_summary['b1b2_corr_init_mean']:.4f}")
        print(f"  B1-B2 corr after S6:   mean={model_summary['b1b2_corr_step6_mean']:.4f}")

        # Print per-sublayer detail for first and last layers
        first_last = [s for s in all_stats if "L00." in s["sublayer"] or f"L{len(sublayers)//len([s for s in all_stats if 'L00.' in s['sublayer']])-1:02d}." in s["sublayer"]]
        if not first_last:
            first_last = all_stats[:6] + all_stats[-6:]

        print(f"\n  Per-sublayer detail (selected):")
        print(f"  {'Name':>25s}  B1flip%  B2flip%  SignDrop  α2/α1  ErrRedS5  ErrRedS6  FlipMagR")
        for s in all_stats[:8]:  # First layer
            print(f"  {s['sublayer']:>25s}  {s['b1_flip_frac']*100:6.2f}%  {s['b2_flip_frac']*100:6.2f}%  "
                  f"{s['b1_sign_agreement_drop']*100:7.3f}%  {s['alpha_ratio']:.4f}  "
                  f"{s['err_reduction_step5']*100:7.3f}%  {s['err_reduction_step6']*100:7.4f}%  "
                  f"{s['flip_mag_ratio']:.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print()

    # =========================================================================
    # Cross-model comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON")
    print("=" * 80)

    header = f"{'Model':>14s} {'Status':>6s}  B1flip%  B2flip%  SignDrop%  α2/α1  ErrRedS5  ErrRedS6  FlipMagR  Corr_init  Corr_S6"
    print(header)
    print("-" * len(header))

    for label, s in all_model_stats.items():
        print(f"{s['model']:>14s} {s['status']:>6s}  "
              f"{s['b1_flip_frac_mean']*100:6.2f}%  {s['b2_flip_frac_mean']*100:6.2f}%  "
              f"{s['b1_sign_agreement_drop_mean']*100:7.3f}%  "
              f"{s['alpha_ratio_mean']:.4f}  "
              f"{s['err_reduction_step5_mean']*100:7.3f}%  "
              f"{s['err_reduction_step6_mean']*100:7.4f}%  "
              f"{s['flip_mag_ratio_mean']:.4f}  "
              f"{s['b1b2_corr_init_mean']:.5f}  "
              f"{s['b1b2_corr_step6_mean']:.5f}")

    # Save
    output_path = "./results/sign_refinement_diagnostic.json"
    os.makedirs("./results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_model_stats, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
