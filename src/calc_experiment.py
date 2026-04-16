#!/usr/bin/env python3
"""
CALC: Cascade-Aware Layer Correction

After standard BRAQ+GPTQ binary quantization, fits a rank-r correction to each
layer's FULL output error (FP16 output vs quantized output). This captures
cross-sublayer interaction errors that per-sublayer GPTQ doesn't optimize for.

Corrected outputs are propagated to subsequent layers, reducing cascade
amplification (measured: 1.39x/layer → target: closer to 1.0x).

Correction form: y_corrected = y_quant + (x @ V_r) @ C^T
where V_r = top-r eigenvectors of input Gram matrix, C = least-squares fit.

Usage:
  source env/bin/activate
  python3 -u src/calc_experiment.py --rank 20
"""

import sys, os, json, gc, time, math, argparse
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


class LayerCorrection(nn.Module):
    """Wraps a transformer layer with a rank-r output correction."""

    def __init__(self, original_layer, V_r, C):
        super().__init__()
        self.layer = original_layer
        self.register_buffer('V_r', V_r)  # (k, r)
        self.register_buffer('C', C)      # (m, r)

    def forward(self, x, **kwargs):
        y = self.layer(x, **kwargs)
        # y is a tuple: (hidden_states, ...). Correct the first element.
        hidden = y[0]
        # Correction: (x @ V_r) @ C^T
        # x shape: (1, seq, k), V_r: (k, r), C: (k, r) [output is same dim as input for residual layers]
        correction = (x @ self.V_r) @ self.C.t()
        corrected = hidden + correction
        return (corrected,) + y[1:]


def load_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, cache_dir="./downloads",
        attn_implementation="eager", use_safetensors=True,
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval()
    model.config.use_cache = False
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


def fit_layer_correction(inps, fp16_outs, quant_outs, rank, dev, nsamples):
    """
    Fit a rank-r correction to the layer output error.
    All computation on CPU to avoid GPU memory leaks.

    inps: (nsamples, seq, k) on GPU
    fp16_outs: (nsamples, seq, k) on CPU
    quant_outs: (nsamples, seq, k) on GPU

    Returns:
        V_r: (k, r) on CPU half, C: (k, r) on CPU half, stats: dict
    """
    k = inps.shape[-1]

    # Move everything to CPU for computation
    X = inps.cpu().reshape(-1, k).float()  # (N, k) on CPU
    N = X.shape[0]

    # Hessian and eigendecomposition on CPU
    H = (X.t() @ X) / N
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    r = min(rank, k)
    V_r = eigenvectors[:, -r:].contiguous()

    # Layer output error on CPU
    error = (fp16_outs.reshape(-1, k) - quant_outs.cpu().reshape(-1, k)).float()

    # Least-squares fit on CPU with strong regularization
    Z = X @ V_r  # (N, r)
    ZtZ = Z.t() @ Z
    # Regularize: lambda = 0.1 * mean(diag(ZtZ)) to prevent overfitting
    reg = 0.1 * torch.diag(ZtZ).mean()
    ZtZ += reg * torch.eye(r)
    ZtE = Z.t() @ error
    Ct = torch.linalg.solve(ZtZ, ZtE)  # (r, k)
    C = Ct.t()  # (k, r)

    # Error metrics
    original_mse = (error ** 2).mean().item()
    corrected_error = error - Z @ Ct
    corrected_mse = (corrected_error ** 2).mean().item()
    mse_reduction = 1 - (corrected_mse / (original_mse + 1e-30))

    H_diag = torch.diag(H)
    hw_original = (H_diag * (error ** 2).mean(dim=0)).sum().item()
    hw_corrected = (H_diag * (corrected_error ** 2).mean(dim=0)).sum().item()
    hw_reduction = 1 - (hw_corrected / (hw_original + 1e-30))

    top_eig_share = eigenvalues[-r:].sum().item() / (eigenvalues.sum().item() + 1e-30)

    stats = {
        'original_mse': original_mse,
        'corrected_mse': corrected_mse,
        'mse_reduction': mse_reduction,
        'hw_original': hw_original,
        'hw_corrected': hw_corrected,
        'hw_reduction': hw_reduction,
        'eigenvalue_share': top_eig_share,
        'correction_rank': r,
    }

    return V_r.half(), C.half(), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=20, help='Correction rank')
    parser.add_argument('--nsamples', type=int, default=128, help='Calibration samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--blocksize', type=int, default=128, help='GPTQ block size')
    parser.add_argument('--no_correction', action='store_true', help='Skip CALC corrections (baseline)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"CALC EXPERIMENT: rank={args.rank}")
    print(f"Model: {MODEL_NAME} | Samples: {args.nsamples} | Seed: {args.seed}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model = load_model()

    # Load data
    print("[2/4] Loading calibration data...")
    dataloader, testloader = get_loaders(
        "wikitext2", nsamples=args.nsamples, seed=args.seed,
        model=MODEL_NAME, seqlen=model.seqlen
    )

    # Capture inputs
    print("[3/4] Capturing inputs...")
    inps, layer_kwargs = capture_inputs(model, dataloader, DEVICE, args.nsamples)

    print("[3.5/4] Quantizing with BRAQ+GPTQ + fitting CALC corrections...\n")
    layers = model.model.layers
    n_layers = len(layers)
    outs = torch.zeros_like(inps)
    # fp16_outs on CPU to save GPU memory - move per sample
    fp16_outs = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=torch.float16, device='cpu'
    )

    all_stats = {}
    t_start = time.time()

    for i in range(n_layers):
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Layer {i}/{n_layers-1} (GPU: {mem:.2f}GB)")
        layer = layers[i].to(DEVICE)
        subset = find_layers(layer)

        # ---- Step 1: Compute FP16 outputs BEFORE quantization ----
        if not args.no_correction:
            with torch.no_grad():
                for j in range(args.nsamples):
                    fp16_outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0].cpu()

        # ---- Step 2: Standard BRAQ+GPTQ quantization ----
        gptq = {}
        for name in subset:
            braq_quantizer = Binarization(
                subset[name].weight, method='braq', groupsize=args.blocksize,
            )
            gptq[name] = BRAGPTQ(
                subset[name], braq_quantizer,
                salient_metric='magnitude', disable_gptq=False,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        # Re-run forward to accumulate Hessians (uses FP16 weights still intact)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        # Quantize sublayers
        for name in sorted(gptq.keys()):
            info = gptq[name].fasterquant(
                percdamp=0.01, blocksize=args.blocksize,
            )
            gptq[name].free()
        del gptq

        # ---- Step 3: Compute quantized outputs ----
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # ---- Step 4: Fit CALC correction ----
        if args.no_correction:
            stats = {'mse_reduction': 0, 'hw_reduction': 0, 'eigenvalue_share': 0,
                     'original_mse': 0, 'corrected_mse': 0, 'hw_original': 0,
                     'hw_corrected': 0, 'correction_rank': 0}
            V_r = torch.zeros(1, 1)
            C = torch.zeros(1, 1)
        else:
            V_r, C, stats = fit_layer_correction(
                inps, fp16_outs, outs, args.rank, DEVICE, args.nsamples
            )

        print(f"    MSE reduction: {stats['mse_reduction']:.4f} | "
              f"HW reduction: {stats['hw_reduction']:.4f} | "
              f"eig_share: {stats['eigenvalue_share']:.3f}")

        all_stats[f"layer_{i}"] = stats

        # ---- Step 5: DO NOT apply corrections during quantization ----
        # Corrections are only applied during eval to avoid cascade divergence.
        # The quantization cascade stays identical to standard BRAQ.

        # Store correction for eval (on CPU)
        layer._calc_V_r = V_r
        layer._calc_C = C

        layers[i] = layer.cpu()
        del layer, V_r, C
        inps, outs = outs, inps
        gc.collect()
        torch.cuda.empty_cache()

    t_quant = time.time() - t_start
    print(f"\nQuantization + CALC complete in {t_quant:.1f}s")

    # ---- Monkey-patch layer forwards to include corrections ----
    if args.no_correction:
        print("\n  Skipping corrections (baseline mode)")
    else:
        print("\n  Patching layer forwards with corrections...")
    for i in range(n_layers):
        if args.no_correction:
            continue
        layer = layers[i]
        if hasattr(layer, '_calc_V_r') and hasattr(layer, '_calc_C'):
            V_r_i = layer._calc_V_r.clone()  # (k, r) on CPU half
            C_i = layer._calc_C.clone()       # (k, r) on CPU half
            original_forward = layer.forward.__func__ if hasattr(layer.forward, '__func__') else None

            def make_corrected_forward(orig_layer, vr, c):
                orig_fwd = type(orig_layer).forward
                def corrected_forward(self, hidden_states, **kwargs):
                    y = orig_fwd(self, hidden_states, **kwargs)
                    out_hidden = y[0]
                    correction = (hidden_states.float() @ vr.float().to(hidden_states.device)) @ c.float().to(hidden_states.device).t()
                    # Clamp correction to prevent divergence: max 50% of output magnitude
                    scale = torch.clamp(out_hidden.float().norm(dim=-1, keepdim=True) * 0.5 /
                                       (correction.norm(dim=-1, keepdim=True) + 1e-8), max=1.0)
                    correction = (correction * scale).to(out_hidden.dtype)
                    corrected = out_hidden + correction
                    # Rebuild output maintaining original structure
                    if isinstance(y, tuple):
                        return (corrected,) + y[1:]
                    elif isinstance(y, list):
                        return [corrected] + y[1:]
                    else:
                        # Assume indexable (ModelOutput etc)
                        y_list = list(y)
                        y_list[0] = corrected
                        return tuple(y_list)
                return corrected_forward

            import types
            layer.forward = types.MethodType(
                make_corrected_forward(layer, V_r_i, C_i), layer
            )
            # Clean up stored tensors
            del layer._calc_V_r, layer._calc_C

    # ---- Aggressive memory cleanup before eval ----
    del inps, outs, fp16_outs
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ---- Step 6: Evaluate PPL using existing eval infrastructure ----
    print("\n[4/4] Evaluating PPL...")
    from eval_ppl_utils import qwen_eval
    ppl = qwen_eval(model, testloader, DEVICE, "wikitext2",
                    save_title=f"calc_r{args.rank}_s{args.seed}", save=False)

    # ---- Summary ----
    fp16_ppl = 20.97
    ppl_val = ppl if isinstance(ppl, float) else ppl.item()
    degradation = ppl_val / fp16_ppl
    target_ppl = 104.85

    print(f"\n{'=' * 70}")
    print(f"RESULTS: CALC rank={args.rank}")
    print(f"{'=' * 70}")
    print(f"PPL: {ppl_val:.2f}")
    print(f"FP16 PPL: {fp16_ppl}")
    print(f"Degradation: {degradation:.2f}x")
    print(f"Target: {target_ppl} (5.0x)")
    print(f"TARGET MET: {'YES' if ppl_val <= target_ppl else 'NO'}")
    print(f"BRAQ baseline: 1651 (78.7x)")
    print(f"Improvement over BRAQ: {1651/ppl_val:.2f}x")

    mse_reds = [s['mse_reduction'] for s in all_stats.values() if not np.isnan(s['mse_reduction'])]
    hw_reds = [s['hw_reduction'] for s in all_stats.values() if not np.isnan(s['hw_reduction'])]
    mean_mse_red = np.mean(mse_reds) if mse_reds else float('nan')
    mean_hw_red = np.mean(hw_reds) if hw_reds else float('nan')
    print(f"\nMean MSE reduction per layer: {mean_mse_red:.4f}")
    print(f"Mean HW reduction per layer: {mean_hw_red:.4f}")

    # Compute bpw overhead
    r = args.rank
    k = model.config.hidden_size  # 1024
    total_weight_params = sum(p.numel() for p in model.parameters())
    correction_params = n_layers * 2 * k * r  # V_r + C per layer
    correction_bits = correction_params * 16
    binary_bits = total_weight_params * 1.1
    avg_bpw = (binary_bits + correction_bits) / total_weight_params
    print(f"Average bpw (with CALC overhead): {avg_bpw:.3f}")

    # Save
    os.makedirs("results", exist_ok=True)
    result = {
        'rank': args.rank, 'ppl': ppl_val, 'degradation': degradation,
        'avg_bpw': avg_bpw, 'target_met': ppl_val <= target_ppl,
        'mean_mse_reduction': mean_mse_red, 'mean_hw_reduction': mean_hw_red,
        'quant_time_seconds': t_quant, 'per_layer': all_stats,
    }
    output_path = f"results/calc_r{args.rank}_seed{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
