#!/usr/bin/env python3
"""
Phase 2.5: Empirical Error Decomposition for GLU Architectures

For each layer of Qwen3-0.6B, quantize all sublayers with BRAQ (same as
the Phase 2 baseline), then measure the layer-output MSE under different
quantization configurations to decompose the error budget.

Configurations tested (per layer):
  0. fp16 reference (all original weights)
  1. attn_only: q,k,v,o quantized; MLP fp16
  2. mlp_only: gate,up,down quantized; attention fp16
  3. gate_only: only gate_proj quantized
  4. up_only: only up_proj quantized
  5. gate_up: gate_proj + up_proj quantized
  6. down_only: only down_proj quantized
  7. full: all quantized (= baseline)

Derived quantities:
  cross_term_mse = gate_up_mse - gate_only_mse - up_only_mse
  interaction_ratio = gate_up_mse / (gate_only_mse + up_only_mse)
    < 1 -> cross-term provides cancellation (helps)
    > 1 -> cross-term amplifies error (hurts)
    = 1 -> no interaction

Usage:
  source env/bin/activate
  python3 -u src/error_decomposition.py [--nsamples 128] [--seed 0]
"""

import sys, os, json, gc, time, argparse
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

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda:0"

# Sublayer groupings
ATTN_NAMES = {'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'}
MLP_NAMES = {'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'}
GATE_NAMES = {'mlp.gate_proj'}
UP_NAMES = {'mlp.up_proj'}
DOWN_NAMES = {'mlp.down_proj'}
GATE_UP_NAMES = {'mlp.gate_proj', 'mlp.up_proj'}

CONFIGS = {
    'attn_only': ATTN_NAMES,
    'mlp_only': MLP_NAMES,
    'gate_only': GATE_NAMES,
    'up_only': UP_NAMES,
    'gate_up': GATE_UP_NAMES,
    'down_only': DOWN_NAMES,
    'full': ATTN_NAMES | MLP_NAMES,
}


def load_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", cache_dir="./downloads",
        attn_implementation="eager"
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    return model


def capture_inputs(model, dataloader, dev, nsamples):
    """Capture layer inputs using the Catcher mechanism (same as run.py)."""
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


def quantize_layer(layer, inps, layer_kwargs, nsamples, dev,
                   blocksize=128, percdamp=0.01, groupsize=128):
    """Run GPTQ+BRAQ on all Linear sublayers. Modifies weights in-place."""
    subset = find_layers(layer)
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

    for name in gptq:
        gptq[name].fasterquant(percdamp=percdamp, blocksize=blocksize)
        gptq[name].free()

    del gptq
    torch.cuda.empty_cache()


def measure_mse(layer, inps, reference, layer_kwargs, n_err):
    """Compute MSE of layer output vs reference."""
    total_mse = 0.0
    with torch.no_grad():
        for j in range(n_err):
            out = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            total_mse += ((out - reference[j]) ** 2).mean().item()
    return total_mse / n_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_err", type=int, default=32,
                        help="Number of samples for error measurement (subset of nsamples)")
    args = parser.parse_args()

    print(f"=== Phase 2.5: Error Decomposition ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Calibration samples: {args.nsamples}")
    print(f"Error measurement samples: {args.n_err}")
    print(f"Seed: {args.seed}")
    print()

    # Load model and calibration data
    model = load_model()
    dataloader, _ = get_loaders(
        'wikitext2', nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=MODEL_NAME
    )

    dev = torch.device(DEVICE)
    inps, layer_kwargs = capture_inputs(model, dataloader, dev, args.nsamples)

    layers = model.model.layers
    n_layers = len(layers)
    n_err = min(args.n_err, args.nsamples)

    outs = torch.zeros_like(inps)

    results = []

    print(f"Processing {n_layers} layers...")
    print()

    for i in range(n_layers):
        t0 = time.time()
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        # Step 1: Save FP16 weights
        fp16_weights = {}
        for name in subset:
            fp16_weights[name] = subset[name].weight.data.clone()

        # Step 2: Compute FP16 reference output (first n_err samples)
        reference = torch.zeros(
            (n_err, model.seqlen, model.config.hidden_size),
            dtype=inps.dtype, device=dev
        )
        with torch.no_grad():
            for j in range(n_err):
                reference[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Step 3: Quantize all sublayers with BRAQ (same as Phase 2 baseline)
        quantize_layer(layer, inps, layer_kwargs, args.nsamples, dev)

        # Step 4: Save quantized weights
        quantized_weights = {}
        for name in subset:
            quantized_weights[name] = subset[name].weight.data.clone()

        # Step 5: Measure error for each configuration
        layer_results = {'layer': i}

        for config_name, quantized_set in CONFIGS.items():
            # Set weights: quantized for names in quantized_set, fp16 for rest
            for name in subset:
                if name in quantized_set:
                    subset[name].weight.data = quantized_weights[name]
                else:
                    subset[name].weight.data = fp16_weights[name]

            mse = measure_mse(layer, inps[:n_err], reference, layer_kwargs, n_err)
            layer_results[config_name + '_mse'] = mse

        # Derived quantities
        gate_mse = layer_results['gate_only_mse']
        up_mse = layer_results['up_only_mse']
        gate_up_mse = layer_results['gate_up_mse']
        down_mse = layer_results['down_only_mse']
        attn_mse = layer_results['attn_only_mse']
        mlp_mse = layer_results['mlp_only_mse']
        full_mse = layer_results['full_mse']

        # Cross-term: how much the gate*up interaction adds/removes error
        cross_term = gate_up_mse - gate_mse - up_mse
        layer_results['cross_term_mse'] = cross_term

        # Interaction ratio: <1 means cross-term helps, >1 means hurts
        denom = gate_mse + up_mse
        layer_results['interaction_ratio'] = gate_up_mse / denom if denom > 0 else float('inf')

        # Superposition check: does attn + mlp = full? Or is there interaction?
        layer_results['attn_plus_mlp'] = attn_mse + mlp_mse
        layer_results['attn_mlp_ratio'] = full_mse / (attn_mse + mlp_mse) if (attn_mse + mlp_mse) > 0 else float('inf')

        # Error shares (of full layer error)
        if full_mse > 0:
            layer_results['attn_share'] = attn_mse / full_mse
            layer_results['gate_share'] = gate_mse / full_mse
            layer_results['up_share'] = up_mse / full_mse
            layer_results['cross_share'] = cross_term / full_mse
            layer_results['down_share'] = down_mse / full_mse
            layer_results['mlp_share'] = mlp_mse / full_mse

        # Step 6: Set all weights to quantized for output computation
        for name in subset:
            subset[name].weight.data = quantized_weights[name]

        # Compute quantized outputs for ALL nsamples (feeds into next layer)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        elapsed = time.time() - t0

        # Print summary
        print(f"Layer {i:2d} ({elapsed:.1f}s):")
        print(f"  attn={attn_mse:.4e}  gate={gate_mse:.4e}  up={up_mse:.4e}  "
              f"cross={cross_term:.4e}  down={down_mse:.4e}")
        print(f"  gate_up={gate_up_mse:.4e}  mlp={mlp_mse:.4e}  full={full_mse:.4e}")
        ir = layer_results['interaction_ratio']
        print(f"  interaction_ratio={ir:.4f}  "
              f"({'HELPS' if ir < 1.0 else 'HURTS' if ir > 1.0 else 'NEUTRAL'})")
        amr = layer_results['attn_mlp_ratio']
        print(f"  attn_mlp_ratio={amr:.4f}  "
              f"(full vs attn+mlp sum: {'subadditive' if amr < 1.0 else 'superadditive'})")
        if full_mse > 0:
            print(f"  Shares of full: attn={layer_results['attn_share']:.1%}  "
                  f"gate={layer_results['gate_share']:.1%}  "
                  f"up={layer_results['up_share']:.1%}  "
                  f"cross={layer_results['cross_share']:.1%}  "
                  f"down={layer_results['down_share']:.1%}  "
                  f"mlp={layer_results['mlp_share']:.1%}")
        print()

        results.append(layer_results)

        # Cleanup
        del fp16_weights, quantized_weights, reference
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # === Aggregate results ===
    print("=" * 80)
    print("AGGREGATE RESULTS (mean +/- std across layers)")
    print("=" * 80)

    # MSE values
    mse_keys = ['attn_only_mse', 'gate_only_mse', 'up_only_mse', 'gate_up_mse',
                'cross_term_mse', 'down_only_mse', 'mlp_only_mse', 'full_mse']
    for k in mse_keys:
        vals = [r[k] for r in results]
        print(f"  {k:25s}: {np.mean(vals):.4e} +/- {np.std(vals):.4e}")

    print()

    # Ratios
    ratio_keys = ['interaction_ratio', 'attn_mlp_ratio']
    for k in ratio_keys:
        vals = [r[k] for r in results]
        print(f"  {k:25s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Shares
    print()
    share_keys = ['attn_share', 'gate_share', 'up_share', 'cross_share', 'down_share', 'mlp_share']
    for k in share_keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            print(f"  {k:25s}: {np.mean(vals):.1%} +/- {np.std(vals):.1%}")

    # Cross-term direction
    helps = sum(1 for r in results if r['interaction_ratio'] < 1)
    hurts = sum(1 for r in results if r['interaction_ratio'] > 1)
    print(f"\n  Cross-term direction: HELPS in {helps}/{len(results)} layers, "
          f"HURTS in {hurts}/{len(results)} layers")

    # Dominant error source
    print("\n  Dominant error source per layer:")
    for r in results:
        sources = {
            'attn': r['attn_only_mse'],
            'gate': r['gate_only_mse'],
            'up': r['up_only_mse'],
            'down': r['down_only_mse'],
        }
        dominant = max(sources, key=sources.get)
        print(f"    Layer {r['layer']:2d}: {dominant} ({sources[dominant]:.4e})")

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = "results/error_decomposition_qwen3_0.6b.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
