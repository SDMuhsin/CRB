#!/usr/bin/env python3
"""
Rotation GPTQ Comparison: Standard vs Eigenvector-Rotated BRAQ+GPTQ

For each sublayer in representative layers, runs BRAQ+GPTQ in two modes:
1. STANDARD: original weight + original Hessian
2. ROTATED: weight rotated to Hessian eigenbasis + diagonal Hessian (eigenvalues)

Compares GPTQ loss to determine if rotation reduces quantization error.

Usage:
  source env/bin/activate
  python3 -u src/rotation_gptq_comparison.py
"""

import sys, os, json, gc, time, copy, math
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
NSAMPLES = 128
SEED = 0
MODEL_NAME = "Qwen/Qwen3-0.6B"
BLOCKSIZE = 128
PERCDAMP = 0.01
TARGET_LAYERS = [0, 5, 10, 15, 20, 27]  # representative layers


def load_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", cache_dir="./downloads",
        attn_implementation="eager", use_safetensors=True,
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval()
    model.config.use_cache = False
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


def compute_rotation(sublayer_inputs_list):
    """
    Compute eigenvector rotation from accumulated sublayer inputs.
    Returns eigenvectors V and eigenvalues (descending order).

    sublayer_inputs_list: list of tensors, each (tokens, input_dim)
    """
    X = torch.cat(sublayer_inputs_list, dim=0).float()  # (N, k)
    k = X.shape[1]

    # Hessian = X^T X / N
    H = (X.t() @ X) / X.shape[0]

    # Eigendecompose
    eigenvalues, eigenvectors = torch.linalg.eigh(H)  # ascending order

    # Flip to descending order
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    return eigenvectors, eigenvalues


def run_braq_gptq(layer_module, sublayer_name, sublayer, inputs_list,
                   orders=(1, 1, 2), rotation_V=None):
    """
    Run BRAQ+GPTQ on a sublayer. Optionally rotate weights and inputs.

    Args:
        layer_module: the full transformer layer
        sublayer_name: name of sublayer (e.g. 'self_attn.q_proj')
        sublayer: the nn.Linear module
        inputs_list: list of input tensors for this sublayer
        orders: BRAQ orders for 3 partitions
        rotation_V: if not None, eigenvector matrix to rotate inputs and weights

    Returns:
        gptq_loss: the GPTQ quantization loss
    """
    # Save original weight
    orig_weight = sublayer.weight.data.clone()

    # Apply rotation to weight if needed
    if rotation_V is not None:
        # W is [out_features, in_features]
        # Rotate input dim: W_rot = W @ V
        W_rotated = sublayer.weight.data.float() @ rotation_V.float()
        sublayer.weight.data = W_rotated.to(sublayer.weight.data.dtype)

    # Create quantizer
    braq_q = Binarization(
        weight=sublayer.weight.data,
        method='braq',
        groupsize=BLOCKSIZE,
    )
    gptq = BRAGPTQ(sublayer, braq_q, salient_metric='magnitude', disable_gptq=False)

    # Feed inputs (optionally rotated)
    for inp_tensor in inputs_list:
        if rotation_V is not None:
            # Rotate input: X_rot = X @ V
            inp_rotated = inp_tensor.float() @ rotation_V.float()
            inp_rotated = inp_rotated.to(inp_tensor.dtype)
            # add_batch expects (batch, seq, features) or (seq, features)
            gptq.add_batch(inp_rotated, torch.zeros(1))  # out is unused for Hessian
        else:
            gptq.add_batch(inp_tensor, torch.zeros(1))

    # Run quantization
    result = gptq.fasterquant(blocksize=BLOCKSIZE, percdamp=PERCDAMP, orders=orders)
    gptq_loss = result["error"]

    # Restore original weight
    sublayer.weight.data = orig_weight

    gptq.free()
    del gptq, braq_q
    torch.cuda.empty_cache()

    return gptq_loss


def main():
    print("=" * 70)
    print("ROTATION GPTQ COMPARISON")
    print(f"Model: {MODEL_NAME} | Layers: {TARGET_LAYERS}")
    print(f"Orders: (1,1,2) | Blocksize: {BLOCKSIZE}")
    print("=" * 70)

    # Load model and data
    print("\n[1/3] Loading model and calibration data...")
    model = load_model()
    dataloader, _ = get_loaders(
        "wikitext2", nsamples=NSAMPLES, seed=SEED,
        model=MODEL_NAME, seqlen=model.seqlen
    )

    print("[2/3] Capturing first-layer inputs...")
    inps, layer_kwargs = capture_inputs(model, dataloader, DEVICE, NSAMPLES)

    print("[3/3] Running comparison...\n")
    layers = model.model.layers
    n_layers = len(layers)
    outs = torch.zeros_like(inps)

    all_results = {}

    # Input group mapping: which sublayers share the same input
    input_groups = {
        'attn_qkv': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        'attn_o': ['self_attn.o_proj'],
        'mlp_gate_up': ['mlp.gate_proj', 'mlp.up_proj'],
        'mlp_down': ['mlp.down_proj'],
    }

    for layer_idx in range(n_layers):
        layer = layers[layer_idx].to(DEVICE)

        if layer_idx in TARGET_LAYERS:
            print(f"\n{'='*60}")
            print(f"LAYER {layer_idx}")
            print(f"{'='*60}")

            subset = find_layers(layer)

            # Capture per-sublayer inputs
            sublayer_inputs = {}

            def make_hook(name):
                def hook_fn(module, inp, out):
                    if name not in sublayer_inputs:
                        sublayer_inputs[name] = []
                    sublayer_inputs[name].append(
                        inp[0].data.detach().reshape(-1, inp[0].shape[-1])
                    )
                return hook_fn

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(make_hook(name)))

            for j in range(NSAMPLES):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

            for h in handles:
                h.remove()

            layer_results = {}

            # Compute rotations for each input group
            rotations = {}
            for group_name, sublayer_names in input_groups.items():
                # Find a sublayer that exists in this layer
                representative = None
                for sn in sublayer_names:
                    if sn in sublayer_inputs:
                        representative = sn
                        break
                if representative is None:
                    continue

                V, eigenvalues = compute_rotation(sublayer_inputs[representative])
                rotations[group_name] = (V, eigenvalues)

                eig_cv = float(torch.std(eigenvalues).item() / (torch.mean(eigenvalues).item() + 1e-30))
                print(f"\n  Rotation [{group_name}]: eig_CV={eig_cv:.2f}, "
                      f"eff_rank={int((eigenvalues.sum()**2 / (eigenvalues**2).sum()).item())}/{eigenvalues.shape[0]}")

            # Run comparison for each sublayer
            for name in sorted(subset.keys()):
                if name not in sublayer_inputs:
                    continue

                # Find which rotation group this sublayer belongs to
                V = None
                for group_name, sublayer_names in input_groups.items():
                    if name in sublayer_names and group_name in rotations:
                        V = rotations[group_name][0]
                        break

                print(f"\n  {name} [W shape: {subset[name].weight.shape}]")

                # Standard BRAQ+GPTQ
                print(f"    Standard BRAQ+GPTQ...", end=" ", flush=True)
                loss_std = run_braq_gptq(layer, name, subset[name],
                                         sublayer_inputs[name], orders=(1, 1, 2))
                print(f"loss = {loss_std:.4f}")

                # Rotated BRAQ+GPTQ
                if V is not None:
                    print(f"    Rotated BRAQ+GPTQ...", end=" ", flush=True)
                    loss_rot = run_braq_gptq(layer, name, subset[name],
                                             sublayer_inputs[name], orders=(1, 1, 2),
                                             rotation_V=V)
                    ratio = loss_rot / (loss_std + 1e-30)
                    improvement = (1 - ratio) * 100
                    print(f"loss = {loss_rot:.4f} (ratio={ratio:.3f}, "
                          f"{'improvement' if improvement > 0 else 'degradation'}={abs(improvement):.1f}%)")

                    layer_results[name] = {
                        'loss_std': loss_std,
                        'loss_rot': loss_rot,
                        'ratio': ratio,
                        'improvement_pct': improvement,
                    }
                else:
                    layer_results[name] = {'loss_std': loss_std}

            all_results[f"layer_{layer_idx}"] = layer_results

            del sublayer_inputs, rotations
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # Just forward through non-target layers
            for j in range(NSAMPLES):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Propagate outputs
        inps, outs = outs, inps
        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== SUMMARY ====================
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_improvements = []
    attn_improvements = []
    mlp_improvements = []

    attn_names = {'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'}

    for layer_key, layer_data in sorted(all_results.items()):
        print(f"\n{layer_key}:")
        for name, data in sorted(layer_data.items()):
            if 'loss_rot' in data:
                imp = data['improvement_pct']
                print(f"  {name}: std={data['loss_std']:.4f} rot={data['loss_rot']:.4f} "
                      f"({'better' if imp > 0 else 'WORSE'} by {abs(imp):.1f}%)")
                all_improvements.append(imp)
                if name in attn_names:
                    attn_improvements.append(imp)
                else:
                    mlp_improvements.append(imp)

    if all_improvements:
        print(f"\nOverall: mean improvement = {np.mean(all_improvements):.1f}% "
              f"+/- {np.std(all_improvements):.1f}%")
        if attn_improvements:
            print(f"Attention: mean improvement = {np.mean(attn_improvements):.1f}% "
                  f"+/- {np.std(attn_improvements):.1f}%")
        if mlp_improvements:
            print(f"MLP: mean improvement = {np.mean(mlp_improvements):.1f}% "
                  f"+/- {np.std(mlp_improvements):.1f}%")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/rotation_gptq_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/rotation_gptq_comparison.json")


if __name__ == "__main__":
    main()
