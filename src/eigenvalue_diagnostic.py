"""
Phase 5 Diagnostic: Eigenvalue spectrum of input Hessian H = E[XX^T] for each sublayer.

Measures: what fraction of total eigenvalue mass is captured by top-r eigenvectors?
If steep (top-30% captures >80%), SAQ (Spectral Adaptive Quantization) is viable.
If flat, SAQ is killed — move to SMLRQ.

Also measures: singular value spectrum of |W| (magnitude matrix) for SMLRQ viability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import json
import math
from datautils import get_loaders, set_seed
from modelutils import find_layers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL = "Qwen/Qwen3-0.6B"
NSAMPLES = 128
SEED = 0
DEVICE = "cuda:0"

def get_model(model_name):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", cache_dir="./downloads",
        attn_implementation="eager"
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    return model

@torch.no_grad()
def run_diagnostic():
    print("Loading model...")
    model = get_model(MODEL)
    model.eval()
    model.config.use_cache = False

    print("Loading calibration data...")
    set_seed(SEED)
    dataloader, _ = get_loaders("wikitext2", nsamples=NSAMPLES, seed=SEED,
                                seqlen=model.seqlen, model=MODEL)

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(DEVICE)
    model.model.norm = model.model.norm.to(DEVICE)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(DEVICE)
    layers[0] = layers[0].to(DEVICE)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (NSAMPLES, model.seqlen, model.config.hidden_size), dtype=dtype, device=DEVICE
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
            model(batch[0].to(DEVICE))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    results = {}
    # Sample layers: first, middle, late, last
    sample_layers = [0, 7, 14, 21, 27]

    for layer_idx in sample_layers:
        print(f"\n=== Layer {layer_idx} ===")
        layer = layers[layer_idx].to(DEVICE)
        subset = find_layers(layer)

        # Collect Hessian H = E[XX^T] for each sublayer
        hessians = {}
        for name in subset:
            W = subset[name].weight.data
            ncols = W.shape[1]
            hessians[name] = torch.zeros((ncols, ncols), device=DEVICE, dtype=torch.float32)

        nsamples_count = {}
        for name in subset:
            nsamples_count[name] = 0

        def make_hook(name):
            def hook_fn(_, inp, out):
                x = inp[0].data
                if len(x.shape) == 3:
                    x = x.reshape(-1, x.shape[-1])
                x = x.float()
                n = x.shape[0]
                hessians[name] *= nsamples_count[name] / (nsamples_count[name] + n)
                nsamples_count[name] += n
                x = math.sqrt(2.0 / nsamples_count[name]) * x.t()
                hessians[name] += x @ x.t()
            return hook_fn

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook(name)))
        for j in range(NSAMPLES):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        layer_results = {}
        for name in subset:
            H = hessians[name]
            W = subset[name].weight.data.float()
            k = H.shape[0]

            # Eigenvalue spectrum of H
            eigvals = torch.linalg.eigvalsh(H)
            eigvals = eigvals.flip(0)  # descending order
            eigvals = eigvals.clamp(min=0)  # numerical safety
            total_eig = eigvals.sum().item()

            cumsum = torch.cumsum(eigvals, dim=0)
            fractions = (cumsum / (total_eig + 1e-30)).cpu().tolist()

            # Compute cumulative fraction at key ranks
            ranks_to_check = [int(k * f) for f in [0.05, 0.10, 0.20, 0.30, 0.50]]
            ranks_to_check = [max(1, min(r, k-1)) for r in ranks_to_check]

            eig_report = {}
            for r in ranks_to_check:
                eig_report[f"top_{r}_of_{k}"] = {
                    "rank": r,
                    "fraction_of_k": round(r / k, 3),
                    "eigenvalue_mass_captured": round(fractions[r-1], 4),
                }

            # Condition number (ratio of largest to smallest nonzero eigenvalue)
            nonzero_eigs = eigvals[eigvals > 1e-10]
            if len(nonzero_eigs) > 1:
                cond = (nonzero_eigs[0] / nonzero_eigs[-1]).item()
            else:
                cond = float('inf')

            # Effective rank: exp(entropy of normalized eigenvalues)
            p = eigvals / (total_eig + 1e-30)
            p = p[p > 1e-10]
            entropy = -(p * torch.log(p)).sum().item()
            effective_rank = math.exp(entropy)

            # Singular values of |W| (magnitude matrix) for SMLRQ diagnostic
            W_abs = torch.abs(W)
            sv_abs = torch.linalg.svdvals(W_abs)
            sv_abs_total = (sv_abs ** 2).sum().item()
            sv_abs_cumfrac = (torch.cumsum(sv_abs ** 2, dim=0) / (sv_abs_total + 1e-30)).cpu().tolist()

            # Singular values of W (original) for comparison
            sv_W = torch.linalg.svdvals(W)
            sv_W_total = (sv_W ** 2).sum().item()
            sv_W_cumfrac = (torch.cumsum(sv_W ** 2, dim=0) / (sv_W_total + 1e-30)).cpu().tolist()

            min_dim = min(W.shape)
            sv_ranks = [int(min_dim * f) for f in [0.03, 0.05, 0.10, 0.20, 0.30]]
            sv_ranks = [max(1, min(r, min_dim-1)) for r in sv_ranks]

            sv_report = {}
            for r in sv_ranks:
                sv_report[f"rank_{r}_of_{min_dim}"] = {
                    "W_energy_captured": round(sv_W_cumfrac[r-1], 4),
                    "absW_energy_captured": round(sv_abs_cumfrac[r-1], 4),
                    "absW_advantage": round(sv_abs_cumfrac[r-1] - sv_W_cumfrac[r-1], 4),
                }

            layer_results[name] = {
                "shape": list(W.shape),
                "hessian_eigenvalue_spectrum": eig_report,
                "hessian_condition_number": round(cond, 1),
                "hessian_effective_rank": round(effective_rank, 1),
                "hessian_effective_rank_fraction": round(effective_rank / k, 3),
                "singular_value_comparison": sv_report,
            }

            # Print summary
            print(f"\n  {name} [{W.shape[0]}x{W.shape[1]}]:")
            print(f"    H condition number: {cond:.0f}, effective rank: {effective_rank:.0f}/{k} ({effective_rank/k:.1%})")
            for key, val in eig_report.items():
                print(f"    {key}: {val['eigenvalue_mass_captured']:.1%} eigenvalue mass")
            print(f"    |W| vs W singular value advantage:")
            for key, val in sv_report.items():
                print(f"      {key}: |W| {val['absW_energy_captured']:.1%} vs W {val['W_energy_captured']:.1%} (advantage: {val['absW_advantage']:+.1%})")

        results[f"layer_{layer_idx}"] = layer_results

        # Propagate through this layer
        for j in range(NSAMPLES):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[layer_idx] = layer.cpu()
        del layer, hessians
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = "results/eigenvalue_diagnostic_qwen3_0.6b.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary decision
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    # Check SAQ viability: is H spectrum steep?
    steep_count = 0
    total_count = 0
    for layer_key, layer_data in results.items():
        for name, data in layer_data.items():
            total_count += 1
            k = data["shape"][1]
            # Check if top 30% of eigenvectors capture >80% of eigenvalue mass
            for eig_key, eig_val in data["hessian_eigenvalue_spectrum"].items():
                if abs(eig_val["fraction_of_k"] - 0.3) < 0.05:  # ~30%
                    if eig_val["eigenvalue_mass_captured"] > 0.80:
                        steep_count += 1
                    break

    print(f"\nSAQ viability: {steep_count}/{total_count} sublayers have steep spectrum")
    print(f"  (top ~30% eigenvectors capture >80% mass)")
    if steep_count > total_count * 0.7:
        print("  → SAQ IS VIABLE: spectrum is steep in majority of sublayers")
    else:
        print("  → SAQ MAY NOT BE VIABLE: spectrum is not sufficiently steep")

    # Check SMLRQ viability: does |W| have better low-rank structure?
    advantage_count = 0
    total_sv = 0
    for layer_key, layer_data in results.items():
        for name, data in layer_data.items():
            for sv_key, sv_val in data["singular_value_comparison"].items():
                total_sv += 1
                if sv_val["absW_advantage"] > 0.05:  # >5% advantage
                    advantage_count += 1

    print(f"\nSMLRQ viability: {advantage_count}/{total_sv} rank checks show |W| advantage >5%")
    if advantage_count > total_sv * 0.5:
        print("  → SMLRQ IS VIABLE: |W| has meaningfully better low-rank structure")
    else:
        print("  → SMLRQ MAY NOT BE VIABLE: |W| does not have better low-rank structure")

if __name__ == "__main__":
    run_diagnostic()
