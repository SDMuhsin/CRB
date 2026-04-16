#!/usr/bin/env python3
"""
Diagnostic experiment: CRB vs BRAQ (BiLLM) relative MSE on OPT-1.3B
weight matrices, per layer and sublayer.

Generates:
  - results/intermediate_quantities_opt1.3b.json  (raw data)
  - llmdocs/paper/v3_revision_1/intermediate_analysis.pdf  (camera-ready plot)
  - llmdocs/paper/v3_revision_1/intermediate_analysis.png  (visual verification)
"""

import json
import os
import sys
import gc

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from binary import coupled_residual_binarization_stable_v7, high_order_residual

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/opt-1.3b"
CACHE_DIR = os.path.join(PROJECT_ROOT, "downloads")
DEVICE = "cuda:0"
SALIENT_FRAC = 0.18  # top 18% columns are salient
NUM_LAYERS = 24
SUBLAYER_NAMES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def get_sublayer_module(layer, name):
    """Return the nn.Linear sublayer from an OPT decoder layer."""
    if name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        return getattr(layer.self_attn, name)
    return getattr(layer, name)


def make_salient_mask(W: torch.Tensor, frac: float) -> torch.Tensor:
    """
    Create a boolean mask where True = salient column.
    Salience = sum of absolute values per column; top `frac` fraction kept.
    W shape: (out_channels, in_channels)
    Returns mask of same shape (broadcast-safe).
    """
    col_scores = W.abs().sum(dim=0)  # (in_channels,)
    k = max(1, int(round(frac * W.shape[1])))
    threshold = torch.topk(col_scores, k).values[-1]
    col_mask = col_scores >= threshold  # (in_channels,)
    return col_mask.unsqueeze(0).expand_as(W)  # (oc, ic)


def relative_mse(W_sal: torch.Tensor, Q: torch.Tensor) -> float:
    """Relative MSE: ||W_sal - Q||^2 / ||W_sal||^2."""
    num = ((W_sal - Q) ** 2).sum().item()
    den = (W_sal ** 2).sum().item()
    if den == 0:
        return 0.0
    return num / den


def main():
    print("=" * 70)
    print("Diagnostic: CRB vs BRAQ relative MSE on OPT-1.3B")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM
    print(f"\nLoading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        use_safetensors=True,
        attn_implementation="eager",
    )
    model.eval()
    layers = model.model.decoder.layers
    print(f"Loaded {NUM_LAYERS} layers.\n")

    # ------------------------------------------------------------------
    # Run experiment
    # ------------------------------------------------------------------
    results = []

    with torch.no_grad():
        for li in range(NUM_LAYERS):
            layer = layers[li]
            for sname in SUBLAYER_NAMES:
                mod = get_sublayer_module(layer, sname)
                W = mod.weight.data.clone().to(DEVICE).float()

                mask = make_salient_mask(W, SALIENT_FRAC)
                W_sal = W * mask.float()

                # --- CRB ---
                Q_crb = coupled_residual_binarization_stable_v7(
                    W, mask, order=2, lam=1e-5, corr_damp=0.1
                )
                crb_mse = relative_mse(W_sal, Q_crb)

                # --- BRAQ ---
                Q_braq = high_order_residual(W, mask, order=2)
                braq_mse = relative_mse(W_sal, Q_braq)

                results.append({
                    "layer": li,
                    "sublayer": sname,
                    "crb_mse": crb_mse,
                    "braq_mse": braq_mse,
                })

                print(
                    f"  Layer {li:2d} / {sname:8s} :  "
                    f"CRB={crb_mse:.6e}  BRAQ={braq_mse:.6e}  "
                    f"{'CRB<' if crb_mse < braq_mse else 'BRAQ<'}"
                )

                # Free GPU
                del W, mask, W_sal, Q_crb, Q_braq
                torch.cuda.empty_cache()

            gc.collect()

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------
    json_path = os.path.join(PROJECT_ROOT, "results", "intermediate_quantities_opt1.3b.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {json_path}")

    # ------------------------------------------------------------------
    # Aggregate: average over sublayers per layer
    # ------------------------------------------------------------------
    crb_per_layer = np.zeros(NUM_LAYERS)
    braq_per_layer = np.zeros(NUM_LAYERS)
    counts = np.zeros(NUM_LAYERS)

    crb_wins = 0
    total_comparisons = 0

    for r in results:
        li = r["layer"]
        crb_per_layer[li] += r["crb_mse"]
        braq_per_layer[li] += r["braq_mse"]
        counts[li] += 1
        total_comparisons += 1
        if r["crb_mse"] < r["braq_mse"]:
            crb_wins += 1

    crb_per_layer /= counts
    braq_per_layer /= counts

    # Per-layer wins (averaged)
    layer_crb_wins = sum(1 for i in range(NUM_LAYERS) if crb_per_layer[i] < braq_per_layer[i])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    avg_crb = np.mean(crb_per_layer)
    avg_braq = np.mean(braq_per_layer)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"CRB  wins on {crb_wins}/{total_comparisons} sublayer comparisons "
          f"({100*crb_wins/total_comparisons:.1f}%)")
    print(f"CRB  wins on {layer_crb_wins}/{NUM_LAYERS} layers (averaged across sublayers)")
    print(f"Overall average relative MSE:  CRB={avg_crb:.6e}  BRAQ={avg_braq:.6e}")
    print(f"CRB advantage: {100*(1 - avg_crb/avg_braq):.2f}% lower relative MSE")

    if crb_wins < total_comparisons / 2:
        print("\n*** WARNING: CRB does NOT win on the majority of sublayers! ***")
    if layer_crb_wins < NUM_LAYERS / 2:
        print("*** WARNING: CRB does NOT win on the majority of layers! ***")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(5, 3))

    x_layers = np.arange(NUM_LAYERS)

    ax.plot(
        x_layers, crb_per_layer,
        color="tab:blue", linestyle="-", marker="o", markersize=4,
        linewidth=1.4, label="CRB (ours)", zorder=3,
    )
    ax.plot(
        x_layers, braq_per_layer,
        color="tab:red", linestyle="--", marker="^", markersize=4,
        linewidth=1.4, label="BiLLM", zorder=3,
    )

    # Shaded region between the two curves
    ax.fill_between(
        x_layers, crb_per_layer, braq_per_layer,
        color="tab:blue", alpha=0.15, zorder=1,
    )

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Relative MSE", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, color="lightgray", alpha=0.3)

    # Use scientific notation on Y axis if values are small
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    fig.tight_layout(pad=0.5)

    pdf_path = os.path.join(
        PROJECT_ROOT, "llmdocs", "paper", "v3_revision_1", "intermediate_analysis.pdf"
    )
    png_path = pdf_path.replace(".pdf", ".png")

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot saved to:\n  {pdf_path}\n  {png_path}")
    print("Done.")


if __name__ == "__main__":
    main()
