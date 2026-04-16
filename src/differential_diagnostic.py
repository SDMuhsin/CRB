#!/usr/bin/env python3
"""
Differential Root Cause Diagnostic: OPT-1.3B vs Qwen3-0.6B

Runs the SAME measurements on both a working model (OPT-1.3B) and a
collapsing model (Qwen3-0.6B) to find what is DIFFERENT.

Measurements per layer:
  1. Per-sublayer GPTQ error (from fasterquant)
  2. Per-layer output MSE cascade (quantized vs FP16 reference)
  3. Error decomposition: attn-only MSE vs MLP-only MSE vs full MSE
  4. Weight distribution stats per sublayer (scale, kurtosis, outlier fraction)
  5. Hessian diagonal stats per sublayer (variance, max/mean ratio)

Usage:
  source env/bin/activate
  python3 -u src/differential_diagnostic.py --model opt
  python3 -u src/differential_diagnostic.py --model qwen3
"""

import sys, os, json, gc, time, argparse
import torch
import torch.nn as nn
import numpy as np
from scipy import stats as scipy_stats

os.environ["TRANSFORMERS_CACHE"] = "./downloads"
os.environ["HF_HOME"] = "./downloads"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from binary import Binarization
from bigptq import BRAGPTQ
from modelutils import find_layers
from datautils import get_loaders

DEVICE = "cuda:0"

# Model configs
MODELS = {
    'opt': {
        'name': 'facebook/opt-1.3b',
        'attn_names': {'self_attn.k_proj', 'self_attn.v_proj',
                       'self_attn.q_proj', 'self_attn.out_proj'},
        'mlp_names': {'fc1', 'fc2'},
        'arch': 'opt',
    },
    'qwen3': {
        'name': 'Qwen/Qwen3-0.6B',
        'attn_names': {'self_attn.q_proj', 'self_attn.k_proj',
                       'self_attn.v_proj', 'self_attn.o_proj'},
        'mlp_names': {'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'},
        'arch': 'qwen',
    },
}


def load_model(model_key):
    cfg = MODELS[model_key]
    if cfg['arch'] == 'opt':
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(
            cfg['name'], torch_dtype="auto", cache_dir="./downloads",
            use_safetensors=True, attn_implementation="eager"
        )
        model.seqlen = model.config.max_position_embeddings
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            cfg['name'], torch_dtype="auto", cache_dir="./downloads",
            attn_implementation="eager"
        )
        model.seqlen = min(model.config.max_position_embeddings, 2048)
    return model


def get_layers(model, model_key):
    if MODELS[model_key]['arch'] == 'opt':
        return model.model.decoder.layers
    else:
        return model.model.layers


def move_embeddings(model, model_key, dev):
    """Move embedding layers to device."""
    if MODELS[model_key]['arch'] == 'opt':
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    else:
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)


def move_embeddings_cpu(model, model_key):
    """Move embedding layers back to CPU."""
    if MODELS[model_key]['arch'] == 'opt':
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()


def capture_inputs(model, model_key, dataloader, dev, nsamples):
    """Capture first-layer inputs."""
    model.config.use_cache = False
    layers = get_layers(model, model_key)

    move_embeddings(model, model_key, dev)
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
    move_embeddings_cpu(model, model_key)
    torch.cuda.empty_cache()

    return inps, cache["layer_kwargs"]


def compute_weight_stats(weight_tensor):
    """Compute distribution statistics for a weight matrix."""
    w = weight_tensor.float().cpu().numpy().ravel()
    abs_w = np.abs(w)
    median_abs = np.median(abs_w)

    return {
        'rows': weight_tensor.shape[0],
        'cols': weight_tensor.shape[1],
        'n_params': weight_tensor.numel(),
        'mean_abs': float(np.mean(abs_w)),
        'std': float(np.std(w)),
        'kurtosis': float(scipy_stats.kurtosis(w)),
        'skewness': float(scipy_stats.skew(w)),
        'max_abs': float(np.max(abs_w)),
        'median_abs': float(median_abs),
        'outlier_frac_5x': float(np.mean(abs_w > 5 * median_abs)) if median_abs > 0 else 0.0,
        'outlier_frac_10x': float(np.mean(abs_w > 10 * median_abs)) if median_abs > 0 else 0.0,
        # Per-row scale variation
        'row_scale_mean': float(np.mean(np.mean(np.abs(weight_tensor.float().cpu().numpy()), axis=1))),
        'row_scale_std': float(np.std(np.mean(np.abs(weight_tensor.float().cpu().numpy()), axis=1))),
        'row_scale_cv': float(np.std(np.mean(np.abs(weight_tensor.float().cpu().numpy()), axis=1)) /
                               (np.mean(np.mean(np.abs(weight_tensor.float().cpu().numpy()), axis=1)) + 1e-12)),
    }


def compute_hessian_stats(H_diag):
    """Compute statistics of the Hessian diagonal."""
    h = H_diag.float().cpu().numpy()
    h_pos = h[h > 0]
    if len(h_pos) == 0:
        return {'mean': 0, 'std': 0, 'max': 0, 'cv': 0, 'max_over_mean': 0,
                'top1pct_over_mean': 0, 'zero_frac': 1.0}

    return {
        'mean': float(np.mean(h)),
        'std': float(np.std(h)),
        'max': float(np.max(h)),
        'cv': float(np.std(h) / (np.mean(h) + 1e-12)),
        'max_over_mean': float(np.max(h) / (np.mean(h) + 1e-12)),
        'top1pct_over_mean': float(np.mean(np.sort(h)[-max(1, len(h)//100):]) / (np.mean(h) + 1e-12)),
        'zero_frac': float(np.mean(h == 0)),
    }


def quantize_and_measure(layer, inps, layer_kwargs, nsamples, dev,
                         model_key, blocksize=128, percdamp=0.01, groupsize=128):
    """
    Quantize all sublayers with BRAQ+GPTQ, capturing:
    - Per-sublayer GPTQ error
    - Per-sublayer Hessian diagonal stats
    - Per-sublayer weight stats (before quantization)
    - Per-sublayer reconstruction error (Frobenius norm)

    Returns: dict with all measurements, plus quantized weights dict
    """
    subset = find_layers(layer)
    cfg = MODELS[model_key]
    results = {}

    # 1. Weight distribution stats BEFORE quantization
    for name in subset:
        w = subset[name].weight.data
        results[f'weight_stats_{name}'] = compute_weight_stats(w)

    # 2. Set up GPTQ for all sublayers and capture Hessian
    gptq = {}
    for name in subset:
        braq_quantizer = Binarization(
            subset[name].weight,
            method='braq',
            groupsize=groupsize,
            corr_damp=0.1,
            lam=1e-5,
            coupling=0.5,
        )
        gptq[name] = BRAGPTQ(
            subset[name],
            braq_quantizer,
            salient_metric='magnitude',
            disable_gptq=False,
        )

    def add_batch(name):
        def tmp(_, inp, out):
            gptq[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in gptq:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    for j in range(nsamples):
        layer(inps[j].unsqueeze(0), **layer_kwargs)
    for h in handles:
        h.remove()

    # 3. Capture Hessian diagonal stats BEFORE quantization destroys H
    for name in gptq:
        H = gptq[name].H.clone()
        h_diag = torch.diag(H)
        results[f'hessian_stats_{name}'] = compute_hessian_stats(h_diag)
        del H

    # 4. Save FP16 weights
    fp16_weights = {}
    for name in subset:
        fp16_weights[name] = subset[name].weight.data.clone()

    # 5. Run GPTQ quantization, capture per-sublayer error
    for name in gptq:
        info = gptq[name].fasterquant(percdamp=percdamp, blocksize=blocksize)
        results[f'gptq_error_{name}'] = info['error']
        gptq[name].free()

    # 6. Save quantized weights
    quantized_weights = {}
    for name in subset:
        quantized_weights[name] = subset[name].weight.data.clone()

    # 7. Per-sublayer reconstruction error (Frobenius norm, normalized)
    for name in subset:
        fp16_w = fp16_weights[name].float()
        quant_w = quantized_weights[name].float()
        frob_err = torch.norm(quant_w - fp16_w).item()
        frob_orig = torch.norm(fp16_w).item()
        results[f'recon_frob_{name}'] = frob_err
        results[f'recon_frob_rel_{name}'] = frob_err / (frob_orig + 1e-12)
        # Per-row reconstruction error
        row_err = torch.norm(quant_w - fp16_w, dim=1)
        row_orig = torch.norm(fp16_w, dim=1)
        rel_row = (row_err / (row_orig + 1e-12)).cpu().numpy()
        results[f'recon_row_rel_mean_{name}'] = float(np.mean(rel_row))
        results[f'recon_row_rel_max_{name}'] = float(np.max(rel_row))
        results[f'recon_row_rel_std_{name}'] = float(np.std(rel_row))

    del gptq
    torch.cuda.empty_cache()

    return results, fp16_weights, quantized_weights


def measure_layer_mse(layer, inps, reference, layer_kwargs, n_err):
    """Compute MSE of layer output vs reference."""
    total_mse = 0.0
    with torch.no_grad():
        for j in range(n_err):
            out = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            total_mse += ((out - reference[j]) ** 2).mean().item()
    return total_mse / n_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['opt', 'qwen3'])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_err", type=int, default=32)
    args = parser.parse_args()

    model_key = args.model
    cfg = MODELS[model_key]
    print(f"=" * 80)
    print(f"DIFFERENTIAL DIAGNOSTIC: {cfg['name']}")
    print(f"=" * 80)
    print(f"Calibration samples: {args.nsamples}")
    print(f"Error measurement samples: {args.n_err}")
    print(f"Seed: {args.seed}")
    print()

    # Load model and data
    model = load_model(model_key)
    dataloader, _ = get_loaders(
        'wikitext2', nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=cfg['name']
    )

    dev = torch.device(DEVICE)
    inps, layer_kwargs = capture_inputs(model, model_key, dataloader, dev, args.nsamples)

    layers = get_layers(model, model_key)
    n_layers = len(layers)
    n_err = min(args.n_err, args.nsamples)
    outs = torch.zeros_like(inps)

    all_results = []

    print(f"Processing {n_layers} layers...")
    print()

    for i in range(n_layers):
        t0 = time.time()
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        # Step 1: Compute FP16 reference output
        reference = torch.zeros(
            (n_err, model.seqlen, model.config.hidden_size),
            dtype=inps.dtype, device=dev
        )
        with torch.no_grad():
            for j in range(n_err):
                reference[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Step 2: Quantize and collect per-sublayer measurements
        layer_results, fp16_weights, quantized_weights = quantize_and_measure(
            layer, inps, layer_kwargs, args.nsamples, dev, model_key
        )
        layer_results['layer'] = i

        # Step 3: Error decomposition - measure MSE under different configs
        # 3a: Full (all quantized) - weights are already quantized from step 2
        full_mse = measure_layer_mse(layer, inps[:n_err], reference, layer_kwargs, n_err)
        layer_results['full_mse'] = full_mse

        # 3b: Attn-only (attention quantized, MLP fp16)
        for name in subset:
            if name in cfg['attn_names']:
                subset[name].weight.data = quantized_weights[name]
            else:
                subset[name].weight.data = fp16_weights[name]
        attn_mse = measure_layer_mse(layer, inps[:n_err], reference, layer_kwargs, n_err)
        layer_results['attn_only_mse'] = attn_mse

        # 3c: MLP-only (MLP quantized, attention fp16)
        for name in subset:
            if name in cfg['mlp_names']:
                subset[name].weight.data = quantized_weights[name]
            else:
                subset[name].weight.data = fp16_weights[name]
        mlp_mse = measure_layer_mse(layer, inps[:n_err], reference, layer_kwargs, n_err)
        layer_results['mlp_only_mse'] = mlp_mse

        # 3d: Per-sublayer MSE (one sublayer quantized at a time)
        for target_name in subset:
            for name in subset:
                if name == target_name:
                    subset[name].weight.data = quantized_weights[name]
                else:
                    subset[name].weight.data = fp16_weights[name]
            sub_mse = measure_layer_mse(layer, inps[:n_err], reference, layer_kwargs, n_err)
            layer_results[f'sublayer_mse_{target_name}'] = sub_mse

        # Derived: error shares
        if full_mse > 0:
            layer_results['attn_share'] = attn_mse / full_mse
            layer_results['mlp_share'] = mlp_mse / full_mse
            for name in subset:
                layer_results[f'sublayer_share_{name}'] = layer_results[f'sublayer_mse_{name}'] / full_mse
        layer_results['attn_mlp_ratio'] = full_mse / (attn_mse + mlp_mse) if (attn_mse + mlp_mse) > 0 else float('inf')

        # Step 4: Set all weights to quantized for cascade output
        for name in subset:
            subset[name].weight.data = quantized_weights[name]

        # Compute quantized outputs for ALL samples (feeds next layer)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        elapsed = time.time() - t0

        # Print summary
        print(f"Layer {i:2d} ({elapsed:.1f}s):")
        print(f"  full_mse={full_mse:.4e}  attn_mse={attn_mse:.4e}  mlp_mse={mlp_mse:.4e}")
        if full_mse > 0:
            print(f"  attn_share={attn_mse/full_mse:.1%}  mlp_share={mlp_mse/full_mse:.1%}")
        print(f"  GPTQ errors:", end="")
        for name in subset:
            print(f"  {name}={layer_results[f'gptq_error_{name}']:.2e}", end="")
        print()
        print(f"  Recon rel:", end="")
        for name in subset:
            print(f"  {name}={layer_results[f'recon_frob_rel_{name}']:.4f}", end="")
        print()
        print()

        all_results.append(layer_results)

        # Cleanup
        del fp16_weights, quantized_weights, reference
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # === Save results ===
    os.makedirs("results", exist_ok=True)
    out_path = f"results/differential_diagnostic_{model_key}.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # === Print aggregate summary ===
    print()
    print("=" * 80)
    print(f"AGGREGATE SUMMARY: {cfg['name']}")
    print("=" * 80)

    # Cascade growth
    first_mse = all_results[0]['full_mse']
    print(f"\nCascade growth (full_mse relative to layer 0):")
    for r in all_results:
        growth = r['full_mse'] / first_mse if first_mse > 0 else 0
        print(f"  Layer {r['layer']:2d}: {r['full_mse']:.4e}  ({growth:.1f}x)")

    # Error shares
    attn_shares = [r['attn_share'] for r in all_results if 'attn_share' in r]
    mlp_shares = [r['mlp_share'] for r in all_results if 'mlp_share' in r]
    print(f"\nError shares (mean across layers):")
    print(f"  Attention: {np.mean(attn_shares):.1%} +/- {np.std(attn_shares):.1%}")
    print(f"  MLP:       {np.mean(mlp_shares):.1%} +/- {np.std(mlp_shares):.1%}")

    # Per-sublayer GPTQ errors
    sublayer_names = list(find_layers(get_layers(model, model_key)[0]).keys())
    print(f"\nMean GPTQ error per sublayer:")
    for name in sublayer_names:
        key = f'gptq_error_{name}'
        vals = [r[key] for r in all_results]
        print(f"  {name:30s}: {np.mean(vals):.4e} +/- {np.std(vals):.4e}")

    # Weight stats
    print(f"\nMean weight kurtosis per sublayer:")
    for name in sublayer_names:
        key = f'weight_stats_{name}'
        vals = [r[key]['kurtosis'] for r in all_results]
        print(f"  {name:30s}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")

    print(f"\nMean outlier fraction (>5x median) per sublayer:")
    for name in sublayer_names:
        key = f'weight_stats_{name}'
        vals = [r[key]['outlier_frac_5x'] for r in all_results]
        print(f"  {name:30s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Hessian stats
    print(f"\nMean Hessian diagonal CV per sublayer:")
    for name in sublayer_names:
        key = f'hessian_stats_{name}'
        vals = [r[key]['cv'] for r in all_results]
        print(f"  {name:30s}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")

    print(f"\nMean Hessian max/mean ratio per sublayer:")
    for name in sublayer_names:
        key = f'hessian_stats_{name}'
        vals = [r[key]['max_over_mean'] for r in all_results]
        print(f"  {name:30s}: {np.mean(vals):.1f} +/- {np.std(vals):.1f}")

    # Reconstruction error
    print(f"\nMean relative Frobenius reconstruction error per sublayer:")
    for name in sublayer_names:
        key = f'recon_frob_rel_{name}'
        vals = [r[key] for r in all_results]
        print(f"  {name:30s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
