"""S8/S9 SDOML+partition smoke test on Qwen3-0.6B layer-0 q_proj.

Captures the real GPTQ Hessian for layer-0 q_proj on 128 wikitext2 samples
(seqlen=2048), then runs SEVEN configurations and compares Hessian-weighted Phi:

    (a) SDOML full s=0.5            — single codebook joint mask + Lloyd-Max
    (b) Magnitude->LMQ s=0.5        — magnitude-prune-then-LMQ baseline
    (c) DOML K=4 (no prune)         — single codebook, no pruning (S4 baseline)
    (d) RTN-2bit                    — round-to-nearest 2-bit (S4 baseline)
    (e) SDOML+partition s=0.5 sym   — S8: 3-way partition + uniform sparsity
                                      (HONEST-NEGATIVE: prunes salient)
    (f) DOML reference              — DOML's pipeline with 3-way partition
                                      via lloyd_max_quantize per partition
    (g) SDOML+partition s=0.5 asym  — S9 deliverable: SDOML mask + Lloyd-Max
                                      ONLY on bulk partition; mid + salient
                                      stay dense (DOML protection preserved)

Pass criteria for S9 layer-0 smoke (must be ALL true to proceed to full model):
    1. Phi(SDOML+partition asym) < Phi(DOML reference) by at least 1%
    2. Phi(SDOML+partition asym) < Phi(SDOML+partition sym)   [must beat S8]
    3. bpw(SDOML+partition asym) <= 2.20
    4. No mask leak: pruned positions in bulk exactly 0; mid + salient
       fully nonzero (every position is kept)

Reports:
    Phi_w = sum_i w_i * (W[r,i] - W_q[r,i])^2     [Hessian-weighted]
    bpw   = effective bits-per-weight
    wall  = seconds
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

# Reuse toy baselines.
sys.path.insert(0, os.path.join(_REPO, "src"))
from toy_sdoml_convergence import (                           # noqa: E402
    magnitude_prune_then_lmq,
    bpw_bitmap,
)


# ---------- helpers ----------

def hessian_weighted_phi(W, W_q, col_weights):
    """Phi_w = sum_i w_i * (W[r,i] - W_q[r,i])^2."""
    err = (W.float() - W_q.float())
    cw = col_weights.float().unsqueeze(0)
    return (cw * err * err).sum().item()


def rtn_2bit_per_row(W):
    """Per-row round-to-nearest 2-bit (4 levels), uniform grid via min/max."""
    R, N = W.shape
    bits = 2
    maxq = 2 ** bits - 1
    zero_ref = torch.zeros(R, device=W.device, dtype=W.dtype)
    row_min = torch.minimum(W.min(dim=1)[0], zero_ref)
    row_max = torch.maximum(W.max(dim=1)[0], zero_ref)
    degen = (row_min == 0) & (row_max == 0)
    row_min = torch.where(degen, torch.full_like(row_min, -1.0), row_min)
    row_max = torch.where(degen, torch.full_like(row_max, +1.0), row_max)
    scale = (row_max - row_min) / maxq
    zero = torch.round(-row_min / scale)
    sc = scale.unsqueeze(1)
    zr = zero.unsqueeze(1)
    q_int = torch.clamp(torch.round(W / sc) + zr, 0.0, float(maxq))
    out = (q_int - zr) * sc
    return out


def bpw_sdoml_partition(R, N, K, sparsity, G=3):
    """SDOML+partition bpw: G codebooks per row + bitmap + indices.

    bpw = G*K*16/N + 1 + (1-s)*log2(K)
    """
    return (G * K * 16.0) / N + 1.0 + (1.0 - sparsity) * math.log2(K)


def bpw_sdoml_partition_asym(R, N, K, sparsity, frac_bulk, frac_mid, frac_sal,
                             G=3):
    """SDOML+partition ASYMMETRIC bpw (S9): bitmap only on bulk partition.

    bpw = G*K*16/N + frac_bulk*1 + (frac_bulk*(1-s) + frac_mid + frac_sal) * log2(K)
    """
    keep_total = frac_bulk * (1.0 - sparsity) + frac_mid + frac_sal
    return (G * K * 16.0) / N + frac_bulk * 1.0 + keep_total * math.log2(K)


# ---------- main ----------

@torch.no_grad()
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    dev = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0"
    if not torch.cuda.is_available():
        dev = "cpu"
    print(f"device: {dev}")

    # --- 1. Load Qwen3-0.6B + capture activations into layer 0 ---------------
    model_name = "Qwen/Qwen3-0.6B"
    downloads_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")
    print(f"Loading {model_name} from cache_dir={downloads_dir} ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", cache_dir=downloads_dir,
        use_safetensors=True, attn_implementation="eager",
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval()
    model.config.use_cache = False

    print("Loading wikitext2 calibration set ...")
    dataloader, _ = get_loaders(
        "wikitext2", nsamples=128, seed=0, model=model_name,
        seqlen=model.seqlen,
    )

    # --- 2. Capture layer-0 q_proj input activations -------------------------
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

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    layer_kwargs = cache["layer_kwargs"]

    # --- 3. Build BRAGPTQ wrapper around layer-0 q_proj and accumulate H ----
    SUBLAYER_NAME = "self_attn.q_proj"
    layer = layers[0].to(dev)
    subset = find_layers(layer)
    target = subset[SUBLAYER_NAME]
    print(f"target sublayer: {SUBLAYER_NAME}  weight shape={tuple(target.weight.shape)}")

    quantizer_dummy = Binarization(target.weight, method='2bit')
    bragptq = BRAGPTQ(target, quantizer_dummy, salient_metric="magnitude")

    def add_batch_hook(_, inp, out):
        bragptq.add_batch(inp[0].data, out.data)

    handle = target.register_forward_hook(add_batch_hook)
    for j in range(nsamples):
        _ = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)
    handle.remove()

    # --- 4. Compute Hinv exactly like bigptq.fasterquant ---------------------
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
    col_weights_full = 1.0 / (hinv_diag ** 2 + 1e-12)
    print(f"Hinv computed; col_weights stats: "
          f"min={col_weights_full.min().item():.4e} "
          f"max={col_weights_full.max().item():.4e} "
          f"mean={col_weights_full.mean().item():.4e}")

    # --- 5. Capture pre-quantization weights ---------------------------------
    W_full = target.weight.data.clone().float().to(dev)
    R, N = W_full.shape
    print(f"W shape: ({R}, {N}); H shape: ({columns}, {columns})")

    # --- 6. Run all configurations ------------------------------------------
    K = 4
    n_iter = 20
    results = []
    sparsity = 0.5

    # --- (c) DOML K=4 single codebook, no pruning (S4 baseline) ---
    print(f"\n[c] DOML-K=4 (no pruning, full mask)")
    t0 = time.time()
    full_mask = torch.ones(R, N, dtype=torch.bool, device=dev)
    W_q_doml = lloyd_max_quantize(W_full, full_mask, K=K, iters=n_iter)
    t_doml = time.time() - t0
    phi_doml = hessian_weighted_phi(W_full, W_q_doml, col_weights_full)
    bpw_doml = (K * 16.0) / N + math.log2(K)
    print(f"  Phi={phi_doml:.4e}  bpw={bpw_doml:.4f}  wall={t_doml:.2f}s")
    results.append({"method": "(c) DOML K=4 (no prune)", "phi": phi_doml,
                    "bpw": bpw_doml, "wall": t_doml})

    # --- (d) RTN-2bit per row ---
    print(f"\n[d] RTN-2bit per row")
    t0 = time.time()
    W_q_rtn = rtn_2bit_per_row(W_full)
    t_rtn = time.time() - t0
    phi_rtn = hessian_weighted_phi(W_full, W_q_rtn, col_weights_full)
    bpw_rtn = 2.0 + 32.0 / N
    print(f"  Phi={phi_rtn:.4e}  bpw={bpw_rtn:.4f}  wall={t_rtn:.2f}s")
    results.append({"method": "(d) RTN-2bit", "phi": phi_rtn,
                    "bpw": bpw_rtn, "wall": t_rtn})

    # --- (a) SDOML full single-codebook (S4 reference) ---
    n_keep = int((1.0 - sparsity) * N)
    bpw_sd = bpw_bitmap(R, N, K, sparsity)
    print(f"\n[a] SDOML full @ s={sparsity:.2f}, n_iter={n_iter}")
    t0 = time.time()
    W_q_sd, mask_sd, codebook_sd, phi_trace = sdoml_quantize(
        W_full, col_weights_full, sparsity=sparsity, K=K,
        n_iter=n_iter, init="quantile", return_aux=True,
    )
    t_sd = time.time() - t0
    phi_sd = hessian_weighted_phi(W_full, W_q_sd, col_weights_full)
    leak_sd = (W_q_sd[~mask_sd] != 0).any().item()
    print(f"  Phi={phi_sd:.4e}  bpw={bpw_sd:.4f}  wall={t_sd:.2f}s  "
          f"n_keep_per_row={n_keep}  mask_leak={leak_sd}")
    results.append({"method": "(a) SDOML s=0.5", "phi": phi_sd,
                    "bpw": bpw_sd, "wall": t_sd})

    # --- (b) Magnitude->LMQ ---
    print(f"\n[b] Magnitude->LMQ @ s={sparsity:.2f}")
    t0 = time.time()
    W_q_b, mask_b = magnitude_prune_then_lmq(
        W_full, col_weights_full, sparsity=sparsity, K=K, lmq_iters=n_iter,
    )
    t_b = time.time() - t0
    phi_b = hessian_weighted_phi(W_full, W_q_b, col_weights_full)
    print(f"  Phi={phi_b:.4e}  bpw={bpw_sd:.4f}  wall={t_b:.2f}s")
    results.append({"method": "(b) Magnitude->LMQ", "phi": phi_b,
                    "bpw": bpw_sd, "wall": t_b})

    # --- (f) DOML reference: structural partition + per-partition Lloyd-Max ---
    print(f"\n[f] DOML reference (3-partition, per-partition Lloyd-Max)")
    t0 = time.time()
    # Match bigptq's call signature for partition=3.
    mask1, mask2, mask3 = structural_guassian_distribution(
        W_full, H, "magnitude", 50, orders=(1, 1, 2),
    )
    # Per partition: Lloyd-Max over the partition's elements (per-element mask).
    # DOML's lloyd_max_quantize takes the [R, N] x and a [R, N] boolean mask.
    # It returns a [R, N] tensor zero outside the mask.
    W_q_p1 = lloyd_max_quantize(W_full, mask1, K=K, iters=n_iter)
    W_q_p2 = lloyd_max_quantize(W_full, mask2, K=K, iters=n_iter)
    W_q_p3 = lloyd_max_quantize(W_full, mask3, K=K, iters=n_iter)
    W_q_doml_ref = W_q_p1 + W_q_p2 + W_q_p3
    t_doml_ref = time.time() - t0
    phi_doml_ref = hessian_weighted_phi(W_full, W_q_doml_ref, col_weights_full)
    bpw_doml_ref = (3 * K * 16.0) / N + math.log2(K)  # 3 codebooks × K × 16 + 2-bit indices
    print(f"  Phi={phi_doml_ref:.4e}  bpw={bpw_doml_ref:.4f}  wall={t_doml_ref:.2f}s")
    print(f"  partition fractions: m1={mask1.float().mean().item():.3f} "
          f"m2={mask2.float().mean().item():.3f} m3={mask3.float().mean().item():.3f}")
    results.append({"method": "(f) DOML reference", "phi": phi_doml_ref,
                    "bpw": bpw_doml_ref, "wall": t_doml_ref})

    # --- (e) SDOML+partition s=0.5 SYMMETRIC (S8 deliverable) ---
    print(f"\n[e] SDOML+partition SYMMETRIC @ s={sparsity:.2f}, n_iter={n_iter}")
    t0 = time.time()
    # Stack per-element partition masks into [G, R, N] shape.
    partition_masks = torch.stack([mask1, mask2, mask3], dim=0).to(dev).bool()
    W_q_sp, mask_sp_full, _cb_aux, _phi_aux = sdoml_partition_quantize(
        W_full, col_weights_full, partition_masks=partition_masks,
        sparsity=sparsity, K=K, n_iter=n_iter, init="quantile",
        return_aux=True,
    )
    t_sp = time.time() - t0
    phi_sp = hessian_weighted_phi(W_full, W_q_sp, col_weights_full)
    leak_sp = (W_q_sp[~mask_sp_full] != 0).any().item()
    bpw_sp = bpw_sdoml_partition(R, N, K, sparsity, G=3)
    print(f"  Phi={phi_sp:.4e}  bpw={bpw_sp:.4f}  wall={t_sp:.2f}s  "
          f"mask_leak={leak_sp}")
    print(f"  per-row keep rate: "
          f"{(mask_sp_full.float().sum(dim=1) / N).mean().item():.4f} "
          f"(expected ~{1 - sparsity:.4f})")
    results.append({"method": "(e) SDOML+partition s=0.5 sym", "phi": phi_sp,
                    "bpw": bpw_sp, "wall": t_sp})

    # --- (g) SDOML+partition s=0.5 ASYMMETRIC (S9 deliverable) ----------
    # Per-partition sparsity = [s, 0, 0]: bulk sparse, mid + salient dense.
    print(f"\n[g] SDOML+partition ASYMMETRIC @ s_bulk={sparsity:.2f}, "
          f"s_mid=0, s_salient=0, n_iter={n_iter}")
    t0 = time.time()
    W_q_asym, mask_asym_full, _cb_aux_a, _phi_aux_a = sdoml_partition_quantize(
        W_full, col_weights_full, partition_masks=partition_masks,
        sparsity=sparsity, K=K, n_iter=n_iter, init="quantile",
        return_aux=True,
        per_partition_sparsity=[sparsity, 0.0, 0.0],
    )
    t_asym = time.time() - t0
    phi_asym = hessian_weighted_phi(W_full, W_q_asym, col_weights_full)
    # Mask-leak check: pruned positions in bulk should be 0.
    leak_asym = (W_q_asym[~mask_asym_full] != 0).any().item()
    # Additional asymmetric check: mid + salient positions should ALL be kept.
    mid_kept = (mask_asym_full & mask2.bool().to(dev)).sum().item()
    mid_total = mask2.sum().item()
    sal_kept = (mask_asym_full & mask3.bool().to(dev)).sum().item()
    sal_total = mask3.sum().item()
    bulk_kept = (mask_asym_full & mask1.bool().to(dev)).sum().item()
    bulk_total = mask1.sum().item()
    frac_bulk = mask1.float().mean().item()
    frac_mid = mask2.float().mean().item()
    frac_sal = mask3.float().mean().item()
    bpw_asym = bpw_sdoml_partition_asym(R, N, K, sparsity, frac_bulk,
                                        frac_mid, frac_sal, G=3)
    print(f"  Phi={phi_asym:.4e}  bpw={bpw_asym:.4f}  wall={t_asym:.2f}s  "
          f"mask_leak={leak_asym}")
    print(f"  bulk:  kept {bulk_kept}/{bulk_total} = "
          f"{bulk_kept/max(bulk_total,1):.4f} (expected ~{1-sparsity:.4f})")
    print(f"  mid:   kept {mid_kept}/{mid_total} = "
          f"{mid_kept/max(mid_total,1):.4f} (expected 1.0000)")
    print(f"  sal:   kept {sal_kept}/{sal_total} = "
          f"{sal_kept/max(sal_total,1):.4f} (expected 1.0000)")
    print(f"  partition column shares: "
          f"frac_bulk={frac_bulk:.3f} frac_mid={frac_mid:.3f} "
          f"frac_sal={frac_sal:.3f}")
    results.append({"method": "(g) SDOML+partition s=0.5 asym", "phi": phi_asym,
                    "bpw": bpw_asym, "wall": t_asym})

    # --- 7. Print summary table -----------------------------------
    print("\n" + "=" * 96)
    print(f"SUMMARY TABLE — Qwen3-0.6B layer-0 q_proj (R={R} N={N})")
    print("=" * 96)
    print(f"{'method':<32}  {'Phi_w':>14}  {'bpw':>8}  {'wall_s':>7}")
    print("-" * 96)
    # Sort by Phi for easier eyeball.
    sorted_results = sorted(results, key=lambda r: r["phi"])
    for r in sorted_results:
        print(f"{r['method']:<32}  {r['phi']:>14.4e}  {r['bpw']:>8.4f}  "
              f"{r['wall']:>7.2f}")
    print("-" * 96)

    # --- 8. S9 pass criteria -------------------------------------
    print("\n" + "=" * 96)
    print(f"S9 LAYER-0 SMOKE PASS CRITERIA (must ALL be True)")
    print("=" * 96)

    # Criterion 1: SDOML+partition asym < DOML reference by >= 1%
    margin_vs_doml_ref = 100.0 * (phi_doml_ref - phi_asym) / phi_doml_ref
    pc1 = margin_vs_doml_ref >= 1.0
    print(f"  [{'PASS' if pc1 else 'FAIL'}] (1) Phi(SDOML+part asym) < "
          f"Phi(DOML ref) by ≥1%   actual={margin_vs_doml_ref:+.3f}%")
    print(f"        Phi(SDOML+part asym)={phi_asym:.4e}   "
          f"Phi(DOML ref)={phi_doml_ref:.4e}")

    # Criterion 2: SDOML+partition asym < SDOML+partition sym (must beat S8)
    margin_vs_sym = 100.0 * (phi_sp - phi_asym) / max(phi_sp, 1e-12)
    pc2 = phi_asym < phi_sp
    print(f"  [{'PASS' if pc2 else 'FAIL'}] (2) Phi(SDOML+part asym) < "
          f"Phi(SDOML+part sym)   actual={margin_vs_sym:+.3f}%")
    print(f"        Phi(SDOML+part asym)={phi_asym:.4e}   "
          f"Phi(SDOML+part sym)={phi_sp:.4e}")

    # Criterion 3: bpw <= 2.20
    pc3 = bpw_asym <= 2.20
    print(f"  [{'PASS' if pc3 else 'FAIL'}] (3) bpw(SDOML+part asym) ≤ 2.20   "
          f"actual={bpw_asym:.4f}")

    # Criterion 4: no mask leak in bulk; mid + salient fully kept.
    bulk_mask_ok = (
        (W_q_asym[mask1.bool().to(dev) & ~mask_asym_full] == 0).all().item()
    )
    mid_full_kept = (mid_kept == mid_total)
    sal_full_kept = (sal_kept == sal_total)
    pc4 = (not leak_asym) and bulk_mask_ok and mid_full_kept and sal_full_kept
    print(f"  [{'PASS' if pc4 else 'FAIL'}] (4) Mask honesty   "
          f"leak={leak_asym} bulk_zero_outside_mask={bulk_mask_ok} "
          f"mid_kept={mid_kept}/{mid_total} sal_kept={sal_kept}/{sal_total}")

    print("\n" + "=" * 96)
    if pc1 and pc2 and pc3 and pc4:
        print("OVERALL: S9 LAYER-0 SMOKE GREEN — proceed to full Qwen3-0.6B run")
        rc = 0
    else:
        print("OVERALL: S9 LAYER-0 SMOKE RED — STOP per C6, do NOT run full model")
        rc = 1
    print("=" * 96)

    return rc


if __name__ == "__main__":
    sys.exit(main())
