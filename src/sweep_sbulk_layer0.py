"""S9 follow-up: sweep s_bulk to find lowest-Phi asymmetric variant.

Layer-0 q_proj only. Reuses smoke harness; sweeps s_bulk over a range and
reports per-row Phi + estimated bpw to find whether ANY s_bulk value beats
DOML reference Phi by >=1%, or whether asymmetric pruning is unconditionally
worse than DOML on the bulk partition.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from binary import sdoml_partition_quantize, lloyd_max_quantize  # noqa: E402
from datautils import get_loaders                                # noqa: E402
from modelutils import find_layers                               # noqa: E402
from bigptq import BRAGPTQ                                       # noqa: E402
from binary import Binarization                                  # noqa: E402
from utils.structure import structural_guassian_distribution     # noqa: E402


@torch.no_grad()
def main():
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype="auto", cache_dir="./downloads",
        use_safetensors=True, attn_implementation="eager",
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval(); model.config.use_cache = False

    dataloader, _ = get_loaders("wikitext2", nsamples=128, seed=0,
                                 model="Qwen/Qwen3-0.6B", seqlen=model.seqlen)

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = 128
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size),
                        dtype=dtype, device=dev)
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__(); self.module = module
        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to(dev); cache["i"] += 1
            cache["layer_kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try: model(batch[0].to(dev))
        except ValueError: pass
    layers[0] = layers[0].module

    if "past_key_values" in cache["layer_kwargs"]:
        cache["layer_kwargs"]["past_key_values"] = None
    if "use_cache" in cache["layer_kwargs"]:
        cache["layer_kwargs"]["use_cache"] = False

    layer_kwargs = cache["layer_kwargs"]
    layer = layers[0].to(dev)
    target = find_layers(layer)["self_attn.q_proj"]

    quantizer_dummy = Binarization(target.weight, method='2bit')
    bragptq = BRAGPTQ(target, quantizer_dummy, salient_metric="magnitude")
    def add_batch_hook(_, inp, out):
        bragptq.add_batch(inp[0].data, out.data)
    handle = target.register_forward_hook(add_batch_hook)
    for j in range(nsamples):
        _ = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)
    handle.remove()

    H = bragptq.H.clone()
    columns = bragptq.columns
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    percdamp = 0.01
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=dev)
    H[diag, diag] += damp
    H_chol = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(H_chol)
    H_chol_upper = torch.linalg.cholesky(H_inv, upper=True)
    Hinv = H_chol_upper
    hinv_diag = torch.diag(Hinv)
    col_weights = 1.0 / (hinv_diag ** 2 + 1e-12)

    W_full = target.weight.data.clone().float().to(dev)
    R, N = W_full.shape
    K, n_iter = 4, 20

    mask1, mask2, mask3 = structural_guassian_distribution(
        W_full, H, "magnitude", 50, orders=(1,1,2)
    )
    mask1, mask2, mask3 = (
        mask1.bool().to(dev), mask2.bool().to(dev), mask3.bool().to(dev),
    )
    partition_masks = torch.stack([mask1, mask2, mask3], dim=0)

    # DOML reference
    W_q_p1 = lloyd_max_quantize(W_full, mask1, K=K, iters=n_iter)
    W_q_p2 = lloyd_max_quantize(W_full, mask2, K=K, iters=n_iter)
    W_q_p3 = lloyd_max_quantize(W_full, mask3, K=K, iters=n_iter)
    err = W_full - (W_q_p1 + W_q_p2 + W_q_p3)
    phi_doml = (col_weights.unsqueeze(0) * err * err).sum().item()
    print(f"DOML ref Phi = {phi_doml:.4e}")
    print()

    header = f"{'s_bulk':>8}  {'Phi_asym':>14}  {'ratio_to_doml':>13}  {'bpw_est':>8}"
    print(header)
    print("-" * len(header))
    for s_bulk in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        if s_bulk == 0.0:
            phi_a = phi_doml
        else:
            W_q_a, _, _, _ = sdoml_partition_quantize(
                W_full, col_weights, partition_masks=partition_masks,
                sparsity=s_bulk, K=K, n_iter=n_iter, init="quantile",
                return_aux=True,
                per_partition_sparsity=[s_bulk, 0.0, 0.0],
            )
            err = W_full - W_q_a
            phi_a = (col_weights.unsqueeze(0) * err * err).sum().item()
        frac_bulk, frac_mid, frac_sal = 0.691, 0.262, 0.048
        keep_total = frac_bulk*(1-s_bulk) + frac_mid + frac_sal
        bpw_est = (3*K*16.0)/N + frac_bulk*1.0 + keep_total*math.log2(K)
        print(f"{s_bulk:>8.2f}  {phi_a:>14.4e}  {phi_a/phi_doml:>13.4f}  "
              f"{bpw_est:>8.4f}")


if __name__ == "__main__":
    main()
