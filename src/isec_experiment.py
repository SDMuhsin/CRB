#!/usr/bin/env python3
"""
ISEC: Importance-Subspace Error Correction

After standard BRAQ+GPTQ binary quantization, adds a rank-r FP16 correction
that targets quantization error in the most important input directions
(top-r Hessian eigenvectors).

The correction preserves GPTQ's error feedback (quantization is unchanged)
while correcting the errors that matter most for output quality.

At inference: y = x @ Q(W)^T + (x @ V_r) @ C^T
where V_r are top-r eigenvectors of X^T X, C = (W_orig - W_quant) @ V_r

Usage:
  source env/bin/activate
  python3 -u src/isec_experiment.py --rank 20
"""

import sys, os, json, gc, time, math, argparse, copy
import torch
import torch.nn as nn
import numpy as np

os.environ["TRANSFORMERS_CACHE"] = "./downloads"
os.environ["HF_HOME"] = "./downloads"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from binary import Binarization
from bigptq import BRAGPTQ
from modelutils import find_layers
from datautils import get_loaders

DEVICE = "cuda:0"
MODEL_NAME = "Qwen/Qwen3-0.6B"


class CorrectedLinear(nn.Module):
    """Wraps a quantized nn.Linear with a rank-r FP16 error correction."""

    def __init__(self, original_linear, V_r, C):
        super().__init__()
        self.linear = original_linear
        # V_r: (input_features, r), C: (output_features, r)
        self.register_buffer('V_r', V_r)
        self.register_buffer('C', C)

    def forward(self, x):
        # y = x @ W_quant^T + (x @ V_r) @ C^T
        base = self.linear(x)
        correction = (x @ self.V_r) @ self.C.t()
        return base + correction

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


def set_sublayer(layer, name, module):
    """Set a sublayer by dotted name (e.g. 'self_attn.q_proj')."""
    parts = name.split('.')
    obj = layer
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], module)


def get_sublayer(layer, name):
    """Get a sublayer by dotted name."""
    parts = name.split('.')
    obj = layer
    for part in parts:
        obj = getattr(obj, part)
    return obj


def compute_isec_correction(W_orig, W_quant, H, rank, dev):
    """
    Compute ISEC correction matrices.

    Args:
        W_orig: original weight (m, k) float
        W_quant: quantized weight (m, k) float
        H: Hessian matrix (k, k) float
        rank: number of eigenvectors to use
        dev: device

    Returns:
        V_r: top-r eigenvectors (k, r) float16
        C: correction matrix (m, r) float16
        eigenvalue_share: fraction of trace captured by top-r eigenvalues
    """
    # Compute quantization error
    epsilon = (W_orig - W_quant).float().to(dev)  # (m, k)

    # Eigendecompose Hessian
    H = H.float().to(dev)
    eigenvalues, eigenvectors = torch.linalg.eigh(H)  # ascending order

    # Top-r eigenvectors (last r in ascending order)
    r = min(rank, eigenvalues.shape[0])
    V_r = eigenvectors[:, -r:].contiguous()  # (k, r)
    top_eigenvalues = eigenvalues[-r:]  # top-r eigenvalues

    # Fraction of trace captured
    total_trace = eigenvalues.sum().item()
    top_trace = top_eigenvalues.sum().item()
    eigenvalue_share = top_trace / (total_trace + 1e-30)

    # Correction: C = epsilon @ V_r
    C = epsilon @ V_r  # (m, r)

    return V_r.half(), C.half(), eigenvalue_share


def load_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, cache_dir="./downloads",
        attn_implementation="eager", use_safetensors=True,
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval()
    model.config.use_cache = False
    # Ensure model is on CPU
    model = model.cpu()
    return model


def capture_inputs(model, dataloader, dev, nsamples):
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


def quantize_layer_with_isec(layer, inps, outs, layer_kwargs, nsamples, dev, rank,
                              blocksize=128, percdamp=0.01, salient_metric='magnitude'):
    """
    Quantize all sublayers in a transformer layer with BRAQ+GPTQ,
    then add ISEC corrections.

    Returns: correction stats (outs is modified in-place)
    """
    subset = find_layers(layer)

    # Create GPTQ objects
    gptq = {}
    for name in subset:
        braq_quantizer = Binarization(
            subset[name].weight,
            method='braq',
            groupsize=blocksize,
        )
        gptq[name] = BRAGPTQ(
            subset[name],
            braq_quantizer,
            salient_metric=salient_metric,
            disable_gptq=False,
        )

    # Hook phase: accumulate Hessian
    def add_batch(name):
        def tmp(_, inp, out):
            gptq[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in gptq:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    with torch.no_grad():
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
    for h in handles:
        h.remove()

    # Quantize each sublayer and compute ISEC corrections
    layer_stats = {}
    for name in sorted(gptq.keys()):
        sublayer = subset[name]

        # Save original weight and Hessian BEFORE fasterquant
        W_orig = sublayer.weight.data.clone().float()
        H_saved = gptq[name].H.clone()

        # Run standard BRAQ+GPTQ
        print(f"    {name}: quantizing...", end="", flush=True)
        info = gptq[name].fasterquant(percdamp=percdamp, blocksize=blocksize)
        gptq_error = info["error"]

        # Get quantized weight
        W_quant = sublayer.weight.data.clone().float()

        # Compute ISEC correction
        V_r, C, eig_share = compute_isec_correction(
            W_orig, W_quant, H_saved, rank, dev
        )

        # Compute correction magnitude
        epsilon = W_orig - W_quant
        total_error_norm = torch.norm(epsilon).item()
        corrected_error = epsilon - (C.float() @ V_r.float().t())
        residual_error_norm = torch.norm(corrected_error).item()
        error_reduction = 1 - (residual_error_norm / (total_error_norm + 1e-30))

        print(f" gptq_err={gptq_error:.2f}, eig_share={eig_share:.3f}, "
              f"error_reduction={error_reduction:.3f}")

        # Wrap sublayer with correction
        corrected = CorrectedLinear(sublayer, V_r.to(dev), C.to(dev))
        set_sublayer(layer, name, corrected)

        layer_stats[name] = {
            'gptq_error': gptq_error,
            'eigenvalue_share': eig_share,
            'frobenius_error_reduction': error_reduction,
            'correction_rank': rank,
            'V_r_params': V_r.numel(),
            'C_params': C.numel(),
        }

        gptq[name].free()

        del W_orig, W_quant, H_saved, epsilon, corrected_error
        torch.cuda.empty_cache()

    del gptq

    # Forward pass with corrected sublayers to propagate outputs
    with torch.no_grad():
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

    return layer_stats


def eval_ppl(model, testenc, dev):
    """Evaluate perplexity on test set."""
    print("\nEvaluating perplexity...")
    testenc_ids = testenc.input_ids
    nsamples = testenc_ids.numel() // model.seqlen

    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
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
    for i in range(nsamples):
        batch = testenc_ids[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc_ids = testenc_ids.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc_ids[:, (i * model.seqlen): ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss.float() * model.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.2f}")

    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    torch.cuda.empty_cache()

    return ppl.item()


def compute_bpw(model, rank):
    """Compute average bits per weight including ISEC overhead."""
    layers = model.model.layers
    total_binary_params = 0
    total_correction_bits = 0

    for layer in layers:
        subset = find_layers(layer)
        for name, sublayer in subset.items():
            if isinstance(sublayer, CorrectedLinear):
                w = sublayer.linear.weight
                k = w.shape[1]  # input features
                m = w.shape[0]  # output features
                total_binary_params += k * m
                # V_r: k * r * 16 bits, C: m * r * 16 bits
                r = sublayer.V_r.shape[1]
                total_correction_bits += (k * r + m * r) * 16
            else:
                total_binary_params += sublayer.weight.numel()

    # Binary weight: ~1.1 bits per param (BRAQ with default orders)
    binary_bits = total_binary_params * 1.1
    total_bits = binary_bits + total_correction_bits
    avg_bpw = total_bits / total_binary_params
    return avg_bpw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=20, help='ISEC correction rank')
    parser.add_argument('--nsamples', type=int, default=128, help='Calibration samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--blocksize', type=int, default=128, help='GPTQ block size')
    args = parser.parse_args()

    print("=" * 70)
    print(f"ISEC EXPERIMENT: rank={args.rank}")
    print(f"Model: {MODEL_NAME} | Samples: {args.nsamples} | Seed: {args.seed}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model = load_model()

    # Load calibration data
    print("[2/4] Loading calibration data...")
    dataloader, testloader = get_loaders(
        "wikitext2", nsamples=args.nsamples, seed=args.seed,
        model=MODEL_NAME, seqlen=model.seqlen
    )

    # Capture first-layer inputs
    print("[3/4] Capturing inputs and quantizing with ISEC corrections...")
    inps, layer_kwargs = capture_inputs(model, dataloader, DEVICE, args.nsamples)

    layers = model.model.layers
    n_layers = len(layers)
    all_stats = {}
    outs = torch.zeros_like(inps)

    t_start = time.time()
    for i in range(n_layers):
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"\n  Layer {i}/{n_layers-1} (GPU: {mem:.2f}GB)")
        layer = layers[i].to(DEVICE)

        layer_stats = quantize_layer_with_isec(
            layer, inps, outs, layer_kwargs, args.nsamples, DEVICE, args.rank,
            blocksize=args.blocksize,
        )

        all_stats[f"layer_{i}"] = layer_stats

        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps
        gc.collect()
        torch.cuda.empty_cache()

    t_quant = time.time() - t_start
    print(f"\nQuantization + ISEC complete in {t_quant:.1f}s")

    # Compute average bpw
    avg_bpw = compute_bpw(model, args.rank)
    print(f"Average bits per weight (with ISEC overhead): {avg_bpw:.3f}")

    # Evaluate PPL
    print("\n[4/4] Evaluating PPL...")
    ppl = eval_ppl(model, testloader, DEVICE)

    # Summary
    fp16_ppl = 20.97
    degradation = ppl / fp16_ppl
    target_ppl = 104.85
    target_met = ppl <= target_ppl

    print(f"\n{'=' * 70}")
    print(f"RESULTS: ISEC rank={args.rank}")
    print(f"{'=' * 70}")
    print(f"PPL: {ppl:.2f}")
    print(f"FP16 PPL: {fp16_ppl}")
    print(f"Degradation: {degradation:.2f}x")
    print(f"Average bpw: {avg_bpw:.3f}")
    print(f"Target PPL: {target_ppl} (5.0x)")
    print(f"TARGET MET: {'YES' if target_met else 'NO'}")
    print(f"BRAQ baseline: 1651 (78.7x)")
    print(f"Improvement over BRAQ: {1651/ppl:.2f}x")

    # Aggregate stats
    mean_eig_share = np.mean([s['eigenvalue_share']
                              for ls in all_stats.values() for s in ls.values()])
    mean_error_reduction = np.mean([s['frobenius_error_reduction']
                                    for ls in all_stats.values() for s in ls.values()])
    print(f"\nMean eigenvalue share (top-{args.rank}): {mean_eig_share:.3f}")
    print(f"Mean Frobenius error reduction: {mean_error_reduction:.3f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    result = {
        'rank': args.rank,
        'ppl': ppl,
        'degradation': degradation,
        'avg_bpw': avg_bpw,
        'target_met': target_met,
        'mean_eigenvalue_share': mean_eig_share,
        'mean_frobenius_error_reduction': mean_error_reduction,
        'quant_time_seconds': t_quant,
        'per_layer': all_stats,
    }
    output_path = f"results/isec_r{args.rank}_seed{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
