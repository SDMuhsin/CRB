"""
Phase 5: Spectral Adaptive Quantization (SAQ) Experiment

SAQ: Rotate weight matrices into the input covariance eigenbasis where columns
are independent. Allocate bits non-uniformly across columns based on eigenvalue
magnitude (reverse water-filling). In the eigenbasis, no GPTQ error correction
is needed since columns are independent.

Usage:
    source env/bin/activate
    python3 -u src/saq_experiment.py [--seed 0] [--target_bpw 2.0]
"""

import sys, os, argparse, json, time, math, gc
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ["HF_HOME"] = "./downloads"

from datautils import get_loaders, set_seed
from modelutils import find_layers
from eval_ppl_utils import qwen_eval

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda:0"

# ─── Core SAQ quantization functions ───

@torch.no_grad()
def quantize_group_perrow(W_group, nbits):
    """Quantize a group of columns using PER-ROW scales.

    W_group: (m, n_cols) — a subset of columns
    nbits: bit depth for this group

    Returns: quantized W_group (m, n_cols), same shape
    """
    if nbits == 0:
        return torch.zeros_like(W_group)

    m, n = W_group.shape
    if n == 0:
        return W_group

    if nbits == 1:
        # Binary sign quantization with per-row mean and scale
        row_mean = W_group.mean(dim=1, keepdim=True)
        centered = W_group - row_mean
        alpha = centered.abs().mean(dim=1, keepdim=True).clamp(min=1e-10)
        return torch.sign(centered) * alpha + row_mean

    if nbits == 2:
        # 2-order binary residual (per-row), equivalent to BRAQ order=2
        Q = torch.zeros_like(W_group)
        residual = W_group.clone()
        for _ in range(2):
            row_mean = residual.mean(dim=1, keepdim=True)
            centered = residual - row_mean
            alpha = centered.abs().mean(dim=1, keepdim=True).clamp(min=1e-10)
            binary = torch.sign(centered) * alpha + row_mean
            Q = Q + binary
            residual = W_group - Q
        return Q

    # For nbits >= 3: uniform symmetric quantization with per-row scale
    maxq = 2 ** (nbits - 1) - 1
    row_max = W_group.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    scale = row_max / (maxq + 0.5)
    Q = torch.clamp(torch.round(W_group / scale), -maxq, maxq) * scale
    return Q


@torch.no_grad()
def saq_quantize_matrix(W, H, target_bpw=2.0):
    """Apply SAQ to a single weight matrix.

    Rotates into eigenbasis, partitions columns into groups by eigenvalue
    magnitude, quantizes each group with per-ROW scales at the group's bit
    depth, then rotates back.

    W: (m, k) weight matrix
    H: (k, k) input second-moment matrix
    target_bpw: target bits per weight

    Returns: quantized W (m, k), info dict
    """
    m, k = W.shape
    W = W.float()

    # Step 1: Eigendecompose H
    eigenvalues, V = torch.linalg.eigh(H.float())
    eigenvalues = eigenvalues.flip(0).clamp(min=0)
    V = V.flip(1)

    # Step 2: Rotate W into eigenbasis
    W_rot = W @ V  # (m, k)

    # Step 3: Partition columns into 3 groups by eigenvalue rank
    # Group A (highest eigenvalue): 4 bits — fine quantization
    # Group B (medium eigenvalue): 1 bit — binary sign
    # Group C (lowest eigenvalue): 0 bits — zeroed
    # Constraint: 4*nA + 1*nB + 0*nC = target_bpw * k, nA + nB + nC = k
    #
    # Search over nA to minimize total weighted distortion:
    # D = sum_{A} lambda_j * 0.0094 + sum_{B} lambda_j * 0.363 + sum_{C} lambda_j * 1.0

    DISTORTION = {0: 1.0, 1: 0.363, 4: 0.0094}
    total_budget = int(target_bpw * k)

    best_distortion = float('inf')
    best_nA = 0
    best_nB = k  # fallback: all 1-bit

    # nA columns at 4 bits, nB at 1 bit, nC at 0 bits
    # 4*nA + 1*nB = total_budget, nA + nB + nC = k
    # nB = total_budget - 4*nA, nC = k - nA - nB = k - total_budget + 3*nA
    # Constraints: nB >= 0 → nA <= total_budget/4
    #              nC >= 0 → 3*nA <= k - total_budget + 3*nA ... always true if nA <= total_budget/4

    max_nA = min(total_budget // 4, k)
    # Columns sorted descending by eigenvalue already
    cum_eig = torch.cumsum(eigenvalues, dim=0).cpu()
    total_eig = cum_eig[-1].item()

    for nA in range(0, max_nA + 1, max(1, max_nA // 200)):  # ~200 candidates
        nB = total_budget - 4 * nA
        if nB < 0:
            break
        nC = k - nA - nB
        if nC < 0:
            nB = k - nA
            nC = 0
        # Distortion: A uses top nA columns, B next nB, C bottom nC
        eig_A = cum_eig[nA - 1].item() if nA > 0 else 0
        eig_B = (cum_eig[nA + nB - 1].item() if nA + nB > 0 else 0) - eig_A
        eig_C = total_eig - eig_A - eig_B
        D = eig_A * 0.0094 + eig_B * 0.363 + eig_C * 1.0
        if D < best_distortion:
            best_distortion = D
            best_nA = nA
            best_nB = nB

    nA, nB = best_nA, best_nB
    nC = k - nA - nB

    bits = torch.zeros(k, dtype=torch.int32, device=W.device)
    bits[:nA] = 4
    bits[nA:nA + nB] = 1
    # bits[nA+nB:] = 0 (already)

    # Step 4: Quantize each group with per-row scales
    Q_rot = torch.zeros_like(W_rot)
    for bit_level in [0, 1, 4]:
        col_mask = bits == bit_level
        if not col_mask.any():
            continue
        cols = col_mask.nonzero(as_tuple=True)[0]
        W_group = W_rot[:, cols]
        Q_group = quantize_group_perrow(W_group, bit_level)
        Q_rot[:, cols] = Q_group

    # Step 5: Rotate back to original basis
    Q = Q_rot @ V.T

    # Compute reconstruction error
    diff = W - Q
    recon_mse = (diff ** 2).mean().item()
    weighted_mse = (diff @ H.float() @ diff.T).trace().item() / m

    # Bit allocation stats
    bit_counts = {}
    for b in [0, 1, 4]:
        c = (bits == b).sum().item()
        if c > 0:
            bit_counts[b] = c

    info = {
        "avg_bits": bits.float().mean().item(),
        "recon_mse": recon_mse,
        "weighted_mse": weighted_mse,
        "bit_distribution": bit_counts,
        "eigenvalue_effective_rank": math.exp(-(eigenvalues / eigenvalues.sum() * torch.log(eigenvalues / eigenvalues.sum() + 1e-30)).sum().item()),
    }

    return Q, info


# ─── Model loading and evaluation infrastructure ───

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
def saq_sequential(model, dataloader, dev, target_bpw=2.0):
    """Quantize all layers of the model with SAQ."""
    print(f"SAQ quantization: target {target_bpw} bpw")

    model.config.use_cache = False
    layers = model.model.layers

    # Move embeddings to device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = len(dataloader)
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

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    all_info = {}
    total_bits = 0
    total_weights = 0

    for layer_idx in range(len(layers)):
        t0 = time.time()
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)

        # Collect Hessians for each sublayer
        hessians = {}
        nsamples_count = {}
        for name in subset:
            k = subset[name].weight.shape[1]
            hessians[name] = torch.zeros((k, k), device=dev, dtype=torch.float32)
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
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        # Quantize each sublayer with SAQ
        layer_info = {}
        for name in subset:
            W = subset[name].weight.data.float()
            H = hessians[name]

            Q, info = saq_quantize_matrix(W, H, target_bpw=target_bpw)

            # Replace weights
            subset[name].weight.data = Q.to(dtype)

            n_weights = W.numel()
            total_bits += info["avg_bits"] * n_weights
            total_weights += n_weights

            layer_info[name] = info
            print(f"  L{layer_idx} {name}: avg_bits={info['avg_bits']:.2f}, "
                  f"wmse={info['weighted_mse']:.4e}, bits={info['bit_distribution']}")

        all_info[f"layer_{layer_idx}"] = layer_info

        # Propagate through quantized layer
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[layer_idx] = layer.cpu()
        del layer, hessians
        torch.cuda.empty_cache()
        inps, outs = outs, inps

        elapsed = time.time() - t0
        print(f"  Layer {layer_idx} done in {elapsed:.1f}s")

    avg_bpw = total_bits / total_weights
    print(f"\nOverall average bits per weight: {avg_bpw:.3f}")
    all_info["overall_avg_bpw"] = avg_bpw

    return all_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_bpw", type=float, default=2.0)
    parser.add_argument("--nsamples", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading model: {MODEL_NAME}")
    model = get_model(MODEL_NAME)
    model.eval()

    print("Loading calibration data...")
    dataloader, testdata = get_loaders(
        "wikitext2", nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=MODEL_NAME
    )

    print(f"\nRunning SAQ quantization (target={args.target_bpw} bpw, seed={args.seed})...")
    info = saq_sequential(model, dataloader, DEVICE, target_bpw=args.target_bpw)

    print("\nEvaluating perplexity...")
    ppl = qwen_eval(
        model, testdata, DEVICE, "wikitext2",
        save_title=f"SAQ_qwen3_0.6b_bpw{args.target_bpw}_seed{args.seed}",
        save=True
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    result = {
        "model": MODEL_NAME,
        "method": "SAQ",
        "target_bpw": args.target_bpw,
        "actual_bpw": info["overall_avg_bpw"],
        "seed": args.seed,
        "ppl": ppl,
        "fp16_ppl": 20.97,
        "degradation_ratio": ppl / 20.97,
        "layer_info": info,
    }
    outpath = f"results/saq_qwen3_0.6b_bpw{args.target_bpw}_seed{args.seed}.json"
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {outpath}")
    print(f"PPL: {ppl:.2f} (degradation: {ppl/20.97:.1f}x)")


if __name__ == "__main__":
    main()
