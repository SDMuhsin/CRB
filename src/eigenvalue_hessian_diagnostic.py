#!/usr/bin/env python3
"""
Eigenvalue Diagnostic: Hessian diagonal CV vs eigenvalue CV

Tests the core hypothesis: do eigenvalues of H = X^T X have much higher
coefficient of variation than the diagonal of H?

If eigenvalue CV >> diagonal CV, then rotating to the eigenbasis concentrates
column importance, enabling effective structural partitioning for binary
quantization. If eigenvalue CV <= diagonal CV, the rotation approach is dead.

Also computes effective rank, condition number, and the theoretical
improvement factor.

Usage:
  source env/bin/activate
  python3 -u src/eigenvalue_hessian_diagnostic.py
"""

import sys, os, json, gc, time
import torch
import torch.nn as nn
import numpy as np

os.environ["TRANSFORMERS_CACHE"] = "./downloads"
os.environ["HF_HOME"] = "./downloads"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modelutils import find_layers
from datautils import get_loaders

DEVICE = "cuda:0"
NSAMPLES = 128
SEED = 0
MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_qwen3():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype="auto", cache_dir="./downloads",
        attn_implementation="eager", use_safetensors=True,
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval()
    return model


def capture_inputs(model, dataloader, dev, nsamples):
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    return inps, cache["layer_kwargs"]


def compute_hessian_and_eigenvalues(sublayer, inps_for_sublayer):
    """
    Compute H = X^T X from stacked input activations.
    Return: H diagonal stats, eigenvalue stats, eigenvalues, eigenvectors.

    inps_for_sublayer: tensor of shape (total_tokens, input_dim) in float32
    """
    X = inps_for_sublayer.float()  # (N, k)
    k = X.shape[1]

    # Compute Hessian H = X^T X / N (normalized)
    H = (X.t() @ X) / X.shape[0]  # (k, k)

    # Diagonal stats
    diag = torch.diag(H).cpu().numpy()
    diag_mean = np.mean(diag)
    diag_std = np.std(diag)
    diag_cv = diag_std / (diag_mean + 1e-30)
    diag_max_over_mean = np.max(diag) / (diag_mean + 1e-30)

    # Eigendecomposition (on GPU for speed)
    eigenvalues, eigenvectors = torch.linalg.eigh(H)  # ascending order
    eigenvalues = eigenvalues.cpu().numpy()
    eigenvalues = eigenvalues[::-1].copy()  # descending order

    # Eigenvalue stats
    eig_pos = eigenvalues[eigenvalues > 0]
    eig_mean = np.mean(eig_pos)
    eig_std = np.std(eig_pos)
    eig_cv = eig_std / (eig_mean + 1e-30)
    eig_max_over_mean = np.max(eig_pos) / (eig_mean + 1e-30)

    # Effective rank = (sum lambda_i)^2 / sum(lambda_i^2)
    eig_sum = np.sum(eig_pos)
    eig_sum_sq = np.sum(eig_pos ** 2)
    effective_rank = (eig_sum ** 2) / (eig_sum_sq + 1e-30)

    # Condition number
    eig_min_pos = np.min(eig_pos) if len(eig_pos) > 0 else 1e-30
    condition_number = np.max(eig_pos) / (eig_min_pos + 1e-30)

    # Top-1% concentration
    top_1pct_idx = max(1, len(eig_pos) // 100)
    top_1pct_share = np.sum(eig_pos[:top_1pct_idx]) / (eig_sum + 1e-30)

    # Top-10% concentration
    top_10pct_idx = max(1, len(eig_pos) // 10)
    top_10pct_share = np.sum(eig_pos[:top_10pct_idx]) / (eig_sum + 1e-30)

    # Bottom-50% share (how much importance is in the least important half)
    bottom_50_share = np.sum(eig_pos[len(eig_pos)//2:]) / (eig_sum + 1e-30)

    return {
        'input_dim': k,
        'diag_cv': float(diag_cv),
        'diag_max_over_mean': float(diag_max_over_mean),
        'diag_mean': float(diag_mean),
        'eig_cv': float(eig_cv),
        'eig_max_over_mean': float(eig_max_over_mean),
        'eig_mean': float(eig_mean),
        'effective_rank': float(effective_rank),
        'effective_rank_frac': float(effective_rank / k),
        'condition_number': float(condition_number),
        'top_1pct_energy_share': float(top_1pct_share),
        'top_10pct_energy_share': float(top_10pct_share),
        'bottom_50pct_energy_share': float(bottom_50_share),
        'cv_ratio': float(eig_cv / (diag_cv + 1e-30)),  # key metric: how much does rotation help?
        'eigenvalues_top20': eigenvalues[:20].tolist(),
        'eigenvalues_bottom10': eigenvalues[-10:].tolist(),
    }


def main():
    print("=" * 70)
    print("EIGENVALUE DIAGNOSTIC: Hessian diagonal CV vs eigenvalue CV")
    print("Model: Qwen3-0.6B | Samples: 128 | Dataset: WikiText-2")
    print("=" * 70)

    # Load model and data
    print("\n[1/4] Loading model...")
    model = load_qwen3()

    print("[2/4] Loading calibration data...")
    dataloader, _ = get_loaders(
        "wikitext2", nsamples=NSAMPLES, seed=SEED,
        model=MODEL_NAME, seqlen=model.seqlen
    )

    print("[3/4] Capturing first-layer inputs...")
    inps, layer_kwargs = capture_inputs(model, dataloader, DEVICE, NSAMPLES)

    print("[4/4] Computing Hessian eigenvalues per sublayer per layer...")
    layers = model.model.layers
    n_layers = len(layers)

    all_results = {}

    # Track aggregates
    attn_diag_cvs = []
    attn_eig_cvs = []
    mlp_diag_cvs = []
    mlp_eig_cvs = []

    attn_names = {'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'}
    mlp_names = {'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'}

    outs = torch.zeros_like(inps)

    for layer_idx in range(n_layers):
        print(f"\n--- Layer {layer_idx}/{n_layers-1} ---")
        layer = layers[layer_idx].to(DEVICE)
        subset = find_layers(layer)

        layer_results = {}

        # For each sublayer, we need its input activations
        # Hook to capture per-sublayer inputs
        sublayer_inputs = {}

        def make_hook(name):
            def hook_fn(module, inp, out):
                # inp is a tuple; inp[0] is the activation
                if name not in sublayer_inputs:
                    sublayer_inputs[name] = []
                sublayer_inputs[name].append(inp[0].data.detach().reshape(-1, inp[0].shape[-1]))
            return hook_fn

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook(name)))

        # Forward all calibration samples through this layer
        for j in range(NSAMPLES):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        for h in handles:
            h.remove()

        # Compute Hessian eigenvalues for each sublayer
        for name in sorted(subset.keys()):
            if name not in sublayer_inputs:
                continue
            # Stack all captured inputs: (total_tokens, input_dim)
            all_inp = torch.cat(sublayer_inputs[name], dim=0)
            print(f"  {name}: input shape = {all_inp.shape}", end="")

            stats = compute_hessian_and_eigenvalues(subset[name], all_inp)
            layer_results[name] = stats

            is_attn = name in attn_names
            tag = "ATTN" if is_attn else "MLP"
            if is_attn:
                attn_diag_cvs.append(stats['diag_cv'])
                attn_eig_cvs.append(stats['eig_cv'])
            else:
                mlp_diag_cvs.append(stats['diag_cv'])
                mlp_eig_cvs.append(stats['eig_cv'])

            print(f"  [{tag}] diag_CV={stats['diag_cv']:.2f} -> eig_CV={stats['eig_cv']:.2f}"
                  f" (ratio={stats['cv_ratio']:.2f}x)"
                  f" | eff_rank={stats['effective_rank']:.0f}/{stats['input_dim']}"
                  f" ({stats['effective_rank_frac']:.1%})"
                  f" | top10%={stats['top_10pct_energy_share']:.1%}")

        all_results[f"layer_{layer_idx}"] = layer_results

        # Propagate outputs to next layer
        inps, outs = outs, inps
        layer = layer.cpu()
        del sublayer_inputs
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nAttention sublayers (n={len(attn_diag_cvs)}):")
    print(f"  Diagonal CV:    mean={np.mean(attn_diag_cvs):.2f} +/- {np.std(attn_diag_cvs):.2f}")
    print(f"  Eigenvalue CV:  mean={np.mean(attn_eig_cvs):.2f} +/- {np.std(attn_eig_cvs):.2f}")
    print(f"  CV ratio (eig/diag): {np.mean(attn_eig_cvs)/np.mean(attn_diag_cvs):.2f}x")

    print(f"\nMLP sublayers (n={len(mlp_diag_cvs)}):")
    print(f"  Diagonal CV:    mean={np.mean(mlp_diag_cvs):.2f} +/- {np.std(mlp_diag_cvs):.2f}")
    print(f"  Eigenvalue CV:  mean={np.mean(mlp_eig_cvs):.2f} +/- {np.std(mlp_eig_cvs):.2f}")
    print(f"  CV ratio (eig/diag): {np.mean(mlp_eig_cvs)/np.mean(mlp_diag_cvs):.2f}x")

    # Compute aggregate effective rank
    all_eff_ranks = []
    all_eff_rank_fracs = []
    for layer_key, layer_data in all_results.items():
        for name, stats in layer_data.items():
            all_eff_ranks.append(stats['effective_rank'])
            all_eff_rank_fracs.append(stats['effective_rank_frac'])
    print(f"\nEffective rank (all sublayers):")
    print(f"  Mean: {np.mean(all_eff_ranks):.0f} ({np.mean(all_eff_rank_fracs):.1%} of input dim)")

    # Key verdict
    overall_cv_ratio = np.mean(attn_eig_cvs + mlp_eig_cvs) / np.mean(attn_diag_cvs + mlp_diag_cvs)
    print(f"\n{'=' * 70}")
    print(f"VERDICT: Overall eigenvalue CV / diagonal CV ratio = {overall_cv_ratio:.2f}x")
    if overall_cv_ratio > 3.0:
        print("=> STRONG SIGNAL: Eigenvector rotation can substantially increase importance non-uniformity.")
        print("   The structural partition should become effective in the eigenbasis.")
    elif overall_cv_ratio > 1.5:
        print("=> MODERATE SIGNAL: Some improvement possible, but may not be sufficient alone.")
    else:
        print("=> WEAK SIGNAL: Rotation alone unlikely to help. Eigenvalues are nearly as uniform as diagonal.")
    print(f"{'=' * 70}")

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = "results/eigenvalue_hessian_diagnostic.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
