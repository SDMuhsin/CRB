#!/usr/bin/env python3
"""
Fast GPTQ error comparison: CRB vs BRAQ on Layer 0 of any supported model.

For each sublayer in Layer 0, runs the full GPTQ pipeline with both methods
(same Hessian, same inputs) and compares:
  - Total GPTQ error
  - Per-column error distribution
  - Weight MSE

This is the FASTEST way to determine if CRB has lower GPTQ error on a given model.

Usage:
  python3 -u src/fast_gptq_compare.py EleutherAI/pythia-1.4b
  python3 -u src/fast_gptq_compare.py facebook/opt-1.3b
  python3 -u src/fast_gptq_compare.py bigscience/bloom-1b7
"""
import torch, torch.nn as nn, gc, sys, os, time
os.environ["TRANSFORMERS_CACHE"] = "./downloads"
os.environ["HF_HOME"] = "./downloads"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import binary
from binary import Binarization
from bigptq import BRAGPTQ
from modelutils import find_layers
from transformers import AutoModelForCausalLM
from datautils import get_loaders
import numpy as np

device = 'cuda:0'
nsamples = 32
blocksize = 128


def setup_model(model_name):
    """Load model and return (model, layers, embed_setup, embed_teardown)."""
    kwargs = dict(cache_dir='./downloads', attn_implementation='eager')

    if 'pythia' in model_name.lower():
        from transformers import GPTNeoXForCausalLM
        model = GPTNeoXForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, **kwargs)
        model.seqlen = model.config.max_position_embeddings
        layers = model.gpt_neox.layers

        def embed_on(dev):
            model.gpt_neox.embed_in.to(dev)
            if hasattr(model.gpt_neox, 'rotary_emb'):
                model.gpt_neox.rotary_emb.to(dev)

        def embed_off():
            model.gpt_neox.embed_in.cpu()
            if hasattr(model.gpt_neox, 'rotary_emb'):
                model.gpt_neox.rotary_emb.cpu()
    elif 'opt' in model_name.lower():
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                 use_safetensors=True, **kwargs)
        model.seqlen = model.config.max_position_embeddings
        layers = model.model.decoder.layers

        def embed_on(dev):
            model.model.decoder.embed_tokens.to(dev)
            model.model.decoder.embed_positions.to(dev)

        def embed_off():
            model.model.decoder.embed_tokens.cpu()
            model.model.decoder.embed_positions.cpu()
    elif 'bloom' in model_name.lower():
        from transformers import BloomForCausalLM
        model = BloomForCausalLM.from_pretrained(model_name, torch_dtype='auto', **kwargs)
        model.seqlen = 2048
        layers = model.transformer.h

        def embed_on(dev):
            model.transformer.word_embeddings.to(dev)
            model.transformer.word_embeddings_layernorm.to(dev)

        def embed_off():
            model.transformer.word_embeddings.cpu()
            model.transformer.word_embeddings_layernorm.cpu()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    model.config.use_cache = False
    return model, layers, embed_on, embed_off


def catcher_phase(model, model_name, layers, embed_on, embed_off, dataloader):
    """Capture Layer 0 input activations."""
    embed_on(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size),
                        dtype=dtype, device=device)
    cache = {'i': 0, 'layer_kwargs': {}}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m
        def __getattr__(self, name):
            if name == 'module': return super().__getattr__(name)
            try: return super().__getattr__(name)
            except AttributeError: return getattr(self.module, name)
        def forward(self, inp, **kw):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'] = kw
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    embed_off()
    torch.cuda.empty_cache()

    return inps, cache['layer_kwargs']


def compare_on_layer(layer, inps, layer_kwargs, n_layers_to_test=1, layer_idx=0):
    """Compare CRB vs BRAQ on a given layer with same Hessian and inputs."""
    subset = find_layers(layer)
    linear_names = sorted([n for n in subset if isinstance(subset[n], nn.Linear)])

    # Step 1: Gather Hessian for all sublayers
    gptq_data = {}
    for name in linear_names:
        q = Binarization(subset[name].weight, method='braq')
        gptq_data[name] = BRAGPTQ(subset[name], q, salient_metric='magnitude')

    handles = []
    for name in gptq_data:
        def make_hook(n):
            def hook(_, inp, out):
                gptq_data[n].add_batch(inp[0].data, out.data)
            return hook
        handles.append(subset[name].register_forward_hook(make_hook(name)))

    # Compute FP16 outputs
    outs = torch.zeros_like(inps)
    for j in range(nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
    for h in handles:
        h.remove()

    # Step 2: For each sublayer, compare methods
    results = {}

    for name in linear_names:
        orig = subset[name].weight.data.clone()
        H = gptq_data[name].H.clone()
        ns = gptq_data[name].nsamples
        gptq_data[name].free()

        # Get Hessian diagonal before inversion
        H_raw_diag = torch.diag(H).clone()

        method_results = {}

        for method in ['braq', 'crb']:
            subset[name].weight.data = orig.clone()
            q = Binarization(subset[name].weight, method=method)
            g = BRAGPTQ(subset[name], q, salient_metric='magnitude')
            g.H = H.clone()
            g.nsamples = ns
            info = g.fasterquant(blocksize=blocksize)

            quantized = subset[name].weight.data.clone()
            weight_err = (orig.float() - quantized.float())

            # Per-column error
            col_mse = (weight_err ** 2).mean(dim=0)  # (ic,)
            # Per-column Hessian-weighted error
            col_hessian_err = col_mse.cpu() * H_raw_diag[:col_mse.shape[0]].float().cpu()

            method_results[method] = {
                'gptq_error': info['error'],
                'weight_mse': (weight_err ** 2).mean().item(),
                'col_mse': col_mse.cpu().numpy(),
                'col_hessian_err': col_hessian_err.cpu().numpy(),
            }

            g.free()
            del g, q

        # Compare
        crb_col = method_results['crb']['col_mse']
        braq_col = method_results['braq']['col_mse']
        n_cols = len(crb_col)
        crb_wins = int(np.sum(crb_col < braq_col))

        # Hessian-weighted comparison
        crb_h = method_results['crb']['col_hessian_err']
        braq_h = method_results['braq']['col_hessian_err']
        crb_h_wins = int(np.sum(crb_h < braq_h))

        # Error magnitude ratio (CRB/BRAQ) for columns where CRB loses
        crb_worse_mask = crb_col > braq_col
        if crb_worse_mask.sum() > 0:
            crb_worse_ratio = np.mean(crb_col[crb_worse_mask] / braq_col[crb_worse_mask])
            crb_worse_excess = np.sum(crb_col[crb_worse_mask] - braq_col[crb_worse_mask])
        else:
            crb_worse_ratio = 0.0
            crb_worse_excess = 0.0

        crb_better_mask = crb_col < braq_col
        if crb_better_mask.sum() > 0:
            crb_better_saving = np.sum(braq_col[crb_better_mask] - crb_col[crb_better_mask])
        else:
            crb_better_saving = 0.0

        results[name] = {
            'crb_gptq': method_results['crb']['gptq_error'],
            'braq_gptq': method_results['braq']['gptq_error'],
            'gptq_ratio': method_results['crb']['gptq_error'] / max(method_results['braq']['gptq_error'], 1e-30),
            'crb_mse': method_results['crb']['weight_mse'],
            'braq_mse': method_results['braq']['weight_mse'],
            'mse_ratio': method_results['crb']['weight_mse'] / max(method_results['braq']['weight_mse'], 1e-30),
            'n_cols': n_cols,
            'crb_wins_cols': crb_wins,
            'crb_wins_cols_pct': 100.0 * crb_wins / n_cols,
            'crb_h_wins_cols': crb_h_wins,
            'crb_h_wins_cols_pct': 100.0 * crb_h_wins / n_cols,
            'crb_worse_excess': float(crb_worse_excess),
            'crb_better_saving': float(crb_better_saving),
            'net_saving': float(crb_better_saving - crb_worse_excess),
        }

        subset[name].weight.data = orig
        del H
        torch.cuda.empty_cache()

    # Also compute layer output MSE
    # Reset all weights to original
    # (they should already be reset from the loop above)

    # Quantize with CRB, compute output
    for name in linear_names:
        orig_w = subset[name].weight.data.clone()
        H_for_quant = gptq_data[name].H if hasattr(gptq_data[name], 'H') and gptq_data[name].H is not None else None
        # We already freed H, so we can't re-quantize here easily
        # Skip layer output MSE for now - the per-sublayer comparison is more informative

    return results, outs


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-1.4b"

    # How many layers to test
    n_test_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print("=" * 90)
    print(f"FAST GPTQ COMPARISON: CRB vs BRAQ")
    print(f"Model: {model_name}")
    print(f"Layers to test: {n_test_layers}")
    print("=" * 90)

    t0 = time.time()

    model, layers, embed_on, embed_off = setup_model(model_name)
    n_layers = len(layers)
    print(f"Model has {n_layers} layers")

    dataloader, _ = get_loaders('wikitext2', nsamples=nsamples, seed=0,
                                  model=model_name, seqlen=model.seqlen)

    inps, layer_kwargs = catcher_phase(model, model_name, layers, embed_on, embed_off, dataloader)
    outs_buf = torch.zeros_like(inps)

    all_results = {}

    for li in range(min(n_test_layers, n_layers)):
        print(f"\n--- Layer {li} ---")
        layer = layers[li].to(device)

        results, outs_buf = compare_on_layer(layer, inps, layer_kwargs, layer_idx=li)
        all_results[li] = results

        # Print results
        print(f"{'Sublayer':>30s} {'GPTQ_ratio':>10s} {'MSE_ratio':>10s} "
              f"{'ColWin%':>8s} {'HColWin%':>8s} {'CRB?':>6s}")
        print("-" * 80)
        for name, r in sorted(results.items()):
            status = "WIN" if r['gptq_ratio'] < 1.0 else "LOSE"
            print(f"{name:>30s} {r['gptq_ratio']:>10.4f} {r['mse_ratio']:>10.4f} "
                  f"{r['crb_wins_cols_pct']:>7.1f}% {r['crb_h_wins_cols_pct']:>7.1f}% "
                  f"{status:>6s}")
            print(f"{'':>30s} GPTQ: CRB={r['crb_gptq']:10.2f} BRAQ={r['braq_gptq']:10.2f}")

        # Pass through unmodified layer for next layer's inputs
        for j in range(nsamples):
            outs_buf[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[li] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs_buf = outs_buf, inps

    # Overall summary
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    total_crb_gptq = 0
    total_braq_gptq = 0
    total_crb_mse = 0
    total_braq_mse = 0
    n_sublayers = 0
    crb_gptq_wins = 0
    crb_mse_wins = 0

    for li, results in all_results.items():
        for name, r in results.items():
            total_crb_gptq += r['crb_gptq']
            total_braq_gptq += r['braq_gptq']
            total_crb_mse += r['crb_mse']
            total_braq_mse += r['braq_mse']
            n_sublayers += 1
            if r['gptq_ratio'] < 1.0:
                crb_gptq_wins += 1
            if r['mse_ratio'] < 1.0:
                crb_mse_wins += 1

    print(f"Sublayers tested: {n_sublayers}")
    print(f"CRB wins GPTQ error: {crb_gptq_wins}/{n_sublayers}")
    print(f"CRB wins weight MSE: {crb_mse_wins}/{n_sublayers}")
    print(f"Total GPTQ ratio: {total_crb_gptq / max(total_braq_gptq, 1e-30):.4f}")
    print(f"Total MSE ratio:  {total_crb_mse / max(total_braq_mse, 1e-30):.4f}")
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
