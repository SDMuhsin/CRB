"""S9 follow-up probe: per-partition Phi contribution under DOML reference.

After S9's layer-0 smoke RED finding, this probe quantifies how much each
partition contributes to the total Phi under DOML's per-partition Lloyd-Max
(no pruning). The point: if the bulk partition (mask1) dominates Phi even
without pruning, then SDOML's joint mask + Lloyd-Max applied to the bulk has
to overcome a large fraction of total error simultaneously — and pruning
half of the bulk's representational capacity is a heavy cost the joint
optimization may not recover.

Reuses the smoke script's Hessian capture verbatim, then:
  - per-partition Phi under DOML's per-partition Lloyd-Max (no pruning)
  - per-partition Phi under SDOML-asymmetric (bulk sparse, mid+sal dense)
  - delta vs DOML at each partition
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

from binary import (                                          # noqa: E402
    sdoml_quantize, sdoml_partition_quantize, lloyd_max_quantize,
)
from datautils import get_loaders                             # noqa: E402
from modelutils import find_layers                            # noqa: E402
from bigptq import BRAGPTQ                                    # noqa: E402
from binary import Binarization                               # noqa: E402
from utils.structure import structural_guassian_distribution  # noqa: E402


def hessian_weighted_phi(W, W_q, col_weights):
    err = (W.float() - W_q.float())
    cw = col_weights.float().unsqueeze(0)
    return (cw * err * err).sum().item()


@torch.no_grad()
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    model_name = "Qwen/Qwen3-0.6B"
    downloads_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", cache_dir=downloads_dir,
        use_safetensors=True, attn_implementation="eager",
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval()
    model.config.use_cache = False

    dataloader, _ = get_loaders("wikitext2", nsamples=128, seed=0,
                                 model=model_name, seqlen=model.seqlen)

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = 128
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev,
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
            inps[cache["i"]] = inp.to(dev)
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
    if "past_key_values" in cache["layer_kwargs"]:
        cache["layer_kwargs"]["past_key_values"] = None
    if "use_cache" in cache["layer_kwargs"]:
        cache["layer_kwargs"]["use_cache"] = False

    layer_kwargs = cache["layer_kwargs"]
    SUBLAYER_NAME = "self_attn.q_proj"
    layer = layers[0].to(dev)
    subset = find_layers(layer)
    target = subset[SUBLAYER_NAME]

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
    K = 4
    n_iter = 20

    mask1, mask2, mask3 = structural_guassian_distribution(
        W_full, H, "magnitude", 50, orders=(1, 1, 2),
    )
    mask1, mask2, mask3 = mask1.bool().to(dev), mask2.bool().to(dev), mask3.bool().to(dev)

    # --- DOML reference per-partition Lloyd-Max (no pruning) ---
    W_q_p1 = lloyd_max_quantize(W_full, mask1, K=K, iters=n_iter)
    W_q_p2 = lloyd_max_quantize(W_full, mask2, K=K, iters=n_iter)
    W_q_p3 = lloyd_max_quantize(W_full, mask3, K=K, iters=n_iter)

    phi_p1 = hessian_weighted_phi(W_full * mask1.float(), W_q_p1, col_weights)
    phi_p2 = hessian_weighted_phi(W_full * mask2.float(), W_q_p2, col_weights)
    phi_p3 = hessian_weighted_phi(W_full * mask3.float(), W_q_p3, col_weights)
    phi_doml = phi_p1 + phi_p2 + phi_p3
    print("\nDOML reference per-partition Phi (no pruning, K=4 Lloyd-Max each):")
    print(f"  bulk(mask1):   Phi={phi_p1:.4e}   "
          f"({100*phi_p1/phi_doml:.1f}% of total)")
    print(f"  mid(mask2):    Phi={phi_p2:.4e}   "
          f"({100*phi_p2/phi_doml:.1f}% of total)")
    print(f"  salient(mask3): Phi={phi_p3:.4e}   "
          f"({100*phi_p3/phi_doml:.1f}% of total)")
    print(f"  TOTAL:         Phi={phi_doml:.4e}")

    # --- SDOML-asymmetric per-partition Phi ---
    partition_masks = torch.stack([mask1, mask2, mask3], dim=0)
    W_q_asym, mask_asym, _, _ = sdoml_partition_quantize(
        W_full, col_weights, partition_masks=partition_masks,
        sparsity=0.5, K=K, n_iter=n_iter, init="quantile",
        return_aux=True,
        per_partition_sparsity=[0.5, 0.0, 0.0],
    )

    # Per-partition contribution: project W_q_asym onto each partition's columns.
    W_q_asym_p1 = W_q_asym * mask1.float()
    W_q_asym_p2 = W_q_asym * mask2.float()
    W_q_asym_p3 = W_q_asym * mask3.float()

    phi_asym_p1 = hessian_weighted_phi(W_full * mask1.float(), W_q_asym_p1, col_weights)
    phi_asym_p2 = hessian_weighted_phi(W_full * mask2.float(), W_q_asym_p2, col_weights)
    phi_asym_p3 = hessian_weighted_phi(W_full * mask3.float(), W_q_asym_p3, col_weights)
    phi_asym = phi_asym_p1 + phi_asym_p2 + phi_asym_p3
    print(f"\nSDOML-asymmetric per-partition Phi (s_bulk=0.5, mid+sal dense):")
    print(f"  bulk(mask1):   Phi={phi_asym_p1:.4e}   "
          f"delta vs DOML: {phi_asym_p1 - phi_p1:+.4e} "
          f"({100*(phi_asym_p1-phi_p1)/phi_p1:+.1f}%)")
    print(f"  mid(mask2):    Phi={phi_asym_p2:.4e}   "
          f"delta vs DOML: {phi_asym_p2 - phi_p2:+.4e} "
          f"({100*(phi_asym_p2-phi_p2)/phi_p2:+.1f}%)")
    print(f"  salient(mask3): Phi={phi_asym_p3:.4e}   "
          f"delta vs DOML: {phi_asym_p3 - phi_p3:+.4e} "
          f"({100*(phi_asym_p3-phi_p3)/phi_p3:+.1f}%)")
    print(f"  TOTAL:         Phi={phi_asym:.4e}   "
          f"vs DOML {phi_doml:.4e} = {100*(phi_asym-phi_doml)/phi_doml:+.1f}%")

    # --- Sanity check: mid + salient dense should match DOML on those parts ---
    print(f"\nSanity check: mid + salient should match DOML exactly when dense:")
    print(f"  mid:    SDOML-asym Phi = {phi_asym_p2:.6e}, DOML = {phi_p2:.6e}, "
          f"match={abs(phi_asym_p2 - phi_p2) < 1e-3}")
    print(f"  salient: SDOML-asym Phi = {phi_asym_p3:.6e}, DOML = {phi_p3:.6e}, "
          f"match={abs(phi_asym_p3 - phi_p3) < 1e-3}")
    print(f"  bulk only difference = {phi_asym_p1 - phi_p1:+.4e}")
    print(f"  Conclusion: bulk pruning at s=0.5 costs {phi_asym_p1 - phi_p1:+.4e} Phi "
          f"vs DOML's no-prune bulk.")


if __name__ == "__main__":
    main()
