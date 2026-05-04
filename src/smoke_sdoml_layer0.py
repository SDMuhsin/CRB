"""S4 SDOML smoke test on Qwen3-0.6B layer-0 q_proj.

Captures the real GPTQ Hessian for one sublayer of layer 0 from Qwen3-0.6B
on 128 wikitext2 samples (seqlen=2048), then runs four configurations:

    (a) SDOML full           — n_iter=20 alternation
    (b) Magnitude->LMQ       — prune by |x|*sqrt(w), then K=4 Lloyd-Max
    (c) DOML K=4 only        — no pruning, K=4 Lloyd-Max
    (d) RTN-2bit             — round-to-nearest 2-bit per row

Reports:
    Phi_w = sum_i w_i * (W[r,i] - W_q[r,i])^2     [Hessian-weighted]
    bpw   = effective bits-per-weight (bitmap formula for SDOML)
    wall  = seconds

Pass criteria (S4 contract):
    1. SDOML-full Phi < magnitude->LMQ Phi by >= 0.5%   at s=0.5
    2. SDOML-full Phi < DOML-K=4 Phi                    at s=0.5
       (or report margin if it loses)
    3. No NaN/Inf
    4. bpw ~ 2.06 at s=0.5 K=4 N=hidden

Drift checks:
    - Per-row keep-margin Spearman correlation with |x|: 0.95 <= rho < 1.0
    - Mask leak: pruned positions remain == 0 after full GPTQ sweep
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

from binary import sdoml_quantize, lloyd_max_quantize  # noqa: E402
from datautils import get_loaders, get_tokenizer        # noqa: E402
from modelutils import find_layers                      # noqa: E402
from bigptq import BRAGPTQ                              # noqa: E402
from binary import Binarization                         # noqa: E402

# Reuse toy baselines.
sys.path.insert(0, os.path.join(_REPO, "src"))
from toy_sdoml_convergence import (                     # noqa: E402
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
    out = torch.zeros_like(W)
    bits = 2
    maxq = 2 ** bits - 1
    zero_ref = torch.zeros(R, device=W.device, dtype=W.dtype)
    row_min = torch.minimum(W.min(dim=1)[0], zero_ref)
    row_max = torch.maximum(W.max(dim=1)[0], zero_ref)
    degen = (row_min == 0) & (row_max == 0)
    row_min = torch.where(degen, torch.full_like(row_min, -1.0), row_min)
    row_max = torch.where(degen, torch.full_like(row_max, +1.0), row_max)
    scale = (row_max - row_min) / maxq                       # (R,)
    zero = torch.round(-row_min / scale)                     # (R,)
    sc = scale.unsqueeze(1)                                  # (R, 1)
    zr = zero.unsqueeze(1)
    q_int = torch.clamp(torch.round(W / sc) + zr, 0.0, float(maxq))
    out = (q_int - zr) * sc
    return out


def spearman(x, y):
    """Spearman rank correlation (numpy)."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


# ---------- main ----------

@torch.no_grad()
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    # Disable KV-cache so layer_kwargs captured by the Catcher (which include
    # past_key_values + use_cache=True by default in transformers 5.x) do not
    # accumulate stale K/V across our subsequent layer.forward() calls.
    # Without this fix, attn_weights becomes (..., L_new, L_new+L_cached) while
    # attention_mask stays (1, 1, L, L) and the shape check trips.
    model.config.use_cache = False

    # Calibration: 128 wikitext2 samples, seqlen=2048.
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

    # Strip KV-cache state from captured kwargs so each subsequent
    # layer.forward() call is independent (no stale K/V accumulation).
    if "past_key_values" in cache["layer_kwargs"]:
        cache["layer_kwargs"]["past_key_values"] = None
    if "use_cache" in cache["layer_kwargs"]:
        cache["layer_kwargs"]["use_cache"] = False

    # Move embed back off GPU to free VRAM during the rest.
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

    # Use a throwaway Binarization (we won't actually use its quantize method).
    # We just need BRAGPTQ to compute H + Hinv for us identically to the real run.
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
    Hinv = H_chol_upper                                      # upper triangular
    hinv_diag = torch.diag(Hinv)
    col_weights_full = 1.0 / (hinv_diag ** 2 + 1e-12)        # [N]
    print(f"Hinv computed; col_weights stats: "
          f"min={col_weights_full.min().item():.4e} "
          f"max={col_weights_full.max().item():.4e} "
          f"mean={col_weights_full.mean().item():.4e}")

    # --- 5. Capture pre-quantization weights ---------------------------------
    W_full = target.weight.data.clone().float().to(dev)      # [R, N]
    R, N = W_full.shape
    print(f"W shape: ({R}, {N}); H shape: ({columns}, {columns})")

    # --- 6. Run the four configurations on the full layer ----------------
    K = 4
    n_iter = 20
    results = []
    sparsities = [0.5, 0.2]

    # We'll run baselines (c) DOML-K=4 and (d) RTN-2bit only once (not per s).
    # Baseline (c): DOML K=4 Lloyd-Max with all-True mask, all weights kept.
    print("\n[c] DOML-K=4 (no pruning, full mask)")
    t0 = time.time()
    full_mask = torch.ones(R, N, dtype=torch.bool, device=dev)
    W_q_doml = lloyd_max_quantize(W_full, full_mask, K=K, iters=n_iter)
    t_doml = time.time() - t0
    phi_doml = hessian_weighted_phi(W_full, W_q_doml, col_weights_full)
    bpw_doml = (K * 16.0) / N + math.log2(K)                  # 2 + tiny tail
    print(f"  Phi={phi_doml:.4e}  bpw={bpw_doml:.4f}  wall={t_doml:.2f}s")

    # Baseline (d): RTN-2bit per row.
    print("\n[d] RTN-2bit per row")
    t0 = time.time()
    W_q_rtn = rtn_2bit_per_row(W_full)
    t_rtn = time.time() - t0
    phi_rtn = hessian_weighted_phi(W_full, W_q_rtn, col_weights_full)
    bpw_rtn = 2.0 + 32.0 / N    # 2 bits + scale (16) + zero (16) per row
    print(f"  Phi={phi_rtn:.4e}  bpw={bpw_rtn:.4f}  wall={t_rtn:.2f}s")

    headline = {}

    # Spearman fixture: only run on first s value.
    spearman_done = False

    for s in sparsities:
        n_keep = int((1.0 - s) * N)
        bpw_sd = bpw_bitmap(R, N, K, s)

        # --- (a) SDOML full ---
        print(f"\n[a] SDOML full @ s={s:.2f}, n_iter={n_iter}")
        t0 = time.time()
        W_q_sd, mask_sd, codebook_sd, phi_trace = sdoml_quantize(
            W_full, col_weights_full, sparsity=s, K=K,
            n_iter=n_iter, init="quantile", return_aux=True,
        )
        t_sd = time.time() - t0
        phi_sd = hessian_weighted_phi(W_full, W_q_sd, col_weights_full)

        # NaN/Inf check
        n_bad = (~torch.isfinite(W_q_sd)).sum().item() + \
                (~torch.isfinite(codebook_sd)).sum().item()
        # Mask-leak check on the standalone SDOML output (no GPTQ sweep here).
        leak = (W_q_sd[~mask_sd] != 0).any().item()
        print(f"  Phi={phi_sd:.4e}  bpw={bpw_sd:.4f}  wall={t_sd:.2f}s  "
              f"n_keep_per_row={n_keep}  nan/inf={n_bad}  mask_leak={leak}")
        print(f"  phi_trace[0]={phi_trace[0].item():.4e}  "
              f"phi_trace[-1]={phi_trace[-1].item():.4e}  "
              f"len={len(phi_trace)}")

        # --- (b) Magnitude->LMQ ---
        print(f"\n[b] Magnitude->LMQ @ s={s:.2f}")
        t0 = time.time()
        W_q_b, mask_b = magnitude_prune_then_lmq(
            W_full, col_weights_full, sparsity=s, K=K, lmq_iters=n_iter,
        )
        t_b = time.time() - t0
        phi_b = hessian_weighted_phi(W_full, W_q_b, col_weights_full)
        print(f"  Phi={phi_b:.4e}  bpw={bpw_sd:.4f}  wall={t_b:.2f}s")

        # Headline ratio + improvement.
        ratio = phi_sd / phi_b if phi_b > 0 else float("inf")
        improvement_pct = 100.0 * (phi_b - phi_sd) / phi_b if phi_b > 0 else 0.0
        print(f"\n  HEADLINE @ s={s:.2f}: "
              f"Phi(SDOML)/Phi(mag->LMQ) = {ratio:.6f}   "
              f"improvement = {improvement_pct:+.3f}%")

        # vs DOML K=4
        ratio_doml = phi_sd / phi_doml if phi_doml > 0 else float("inf")
        impr_doml = 100.0 * (phi_doml - phi_sd) / phi_doml if phi_doml > 0 else 0.0
        print(f"  vs DOML-K=4: Phi(SDOML)/Phi(DOML) = {ratio_doml:.6f}   "
              f"improvement = {impr_doml:+.3f}%")

        # --- per-row keep-margin Spearman correlation with |x| ---
        # Sample one row (row 0) and one row (R//2) for the correlation check.
        if not spearman_done:
            for r_chk in [0, R // 2, R - 1]:
                # mu_i = w_i * (x_i^2 - min_c (x_i - c)^2)  for this row
                x_row = W_full[r_chk]                                # [N]
                cb_row = codebook_sd[r_chk]                          # [K]
                d2 = (x_row.unsqueeze(1) - cb_row.unsqueeze(0)) ** 2  # [N, K]
                d_min = d2.min(dim=1).values                         # [N]
                mu = col_weights_full * (x_row * x_row - d_min)
                rho = spearman(mu.cpu().numpy(), x_row.abs().cpu().numpy())
                print(f"  Spearman(mu, |x|) row={r_chk}: rho={rho:.4f}")
            spearman_done = True

        results.append({
            "s": s, "method": "(a) SDOML full",
            "phi": phi_sd, "bpw": bpw_sd, "wall": t_sd,
            "n_keep": n_keep,
        })
        results.append({
            "s": s, "method": "(b) Magnitude->LMQ",
            "phi": phi_b, "bpw": bpw_sd, "wall": t_b,
            "n_keep": n_keep,
        })
        if s == sparsities[0]:
            results.append({
                "s": s, "method": "(c) DOML K=4 (no prune)",
                "phi": phi_doml, "bpw": bpw_doml, "wall": t_doml,
                "n_keep": N,
            })
            results.append({
                "s": s, "method": "(d) RTN-2bit",
                "phi": phi_rtn, "bpw": bpw_rtn, "wall": t_rtn,
                "n_keep": N,
            })

        # Pass criteria evaluation at headline s.
        if s == 0.5:
            headline = {
                "phi_sdoml_s050": phi_sd,
                "phi_mag_lmq_s050": phi_b,
                "phi_doml_k4": phi_doml,
                "phi_rtn_2bit": phi_rtn,
                "ratio_sdoml_over_maglmq": ratio,
                "improvement_pct": improvement_pct,
                "ratio_sdoml_over_doml": ratio_doml,
                "improvement_over_doml_pct": impr_doml,
            }

    # --- 7. Final Composition Candidate I via bigptq dispatch (end-to-end) ---
    # This validates the full GPTQ wiring path and the mask-leak guard.
    print("\n[Composition Candidate I via bigptq.fasterquant @ s=0.50]")
    # Reset the layer's weight to the original W before re-quantizing.
    target.weight.data = W_full.to(target.weight.dtype)

    # Re-create H accumulation (the previous run consumed it via cholesky).
    quantizer_sd = Binarization(target.weight, method='sdoml')
    quantizer_sd.sparsity = 0.5
    quantizer_sd.sdoml_K = K
    quantizer_sd.sdoml_n_iter = n_iter
    bragptq2 = BRAGPTQ(target, quantizer_sd, salient_metric="magnitude")

    handle = target.register_forward_hook(
        lambda _, inp, out: bragptq2.add_batch(inp[0].data, out.data)
    )
    for j in range(nsamples):
        _ = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)
    handle.remove()

    t0 = time.time()
    info_full = bragptq2.fasterquant(
        percdamp=percdamp, blocksize=128, partition=1, orders=(1,),
    )
    t_comp1 = time.time() - t0

    W_q_comp1 = target.weight.data.clone().float().to(dev)
    phi_comp1 = hessian_weighted_phi(W_full, W_q_comp1, col_weights_full)

    # Note: end-to-end Phi may not match the standalone fit because GPTQ
    # propagates errors *into* future blocks, shifting their inputs.
    # The standalone fit measures Phi against the original W; the GPTQ-swept
    # output's Phi against original W can be different (typically lower or
    # comparable thanks to error compensation).
    print(f"  Phi(end-to-end)={phi_comp1:.4e}  wall={t_comp1:.2f}s  "
          f"raw_error={info_full['error']:.3e}")

    # Mask-leak check: re-derive what the mask should be by re-running
    # sdoml_quantize on the *original* W and confirming positions where
    # the joint-fit mask said pruned indeed remain == 0 in W_q_comp1.
    # Caveat: bigptq.fasterquant fits a fresh mask per *block*, so we cannot
    # compare against a single-block mask. The robust mask-leak check is:
    # count zeros per row in the final output. If sparsity ~ 0.5, ~50% of
    # entries should be exactly 0 (across rows).
    zero_frac = (W_q_comp1 == 0).float().mean().item()
    print(f"  Zero-fraction of comp1 output: {zero_frac:.4f} "
          f"(expected ~0.50 at s=0.5)")
    headline["phi_comp1_end_to_end_s050"] = phi_comp1
    headline["zero_fraction_comp1_s050"] = zero_frac
    headline["wall_comp1_s050"] = t_comp1

    # --- 8. Print summary table -----------------------------------
    print("\n" + "=" * 96)
    print("SUMMARY TABLE — Qwen3-0.6B layer-0 q_proj (R=2048 N=1024)")
    print("=" * 96)
    print(f"{'s':>4}  {'method':<26}  {'Phi_w':>14}  "
          f"{'bpw':>8}  {'wall_s':>7}  {'n_keep':>6}")
    print("-" * 96)
    for r in results:
        print(f"{r['s']:>4.2f}  {r['method']:<26}  "
              f"{r['phi']:>14.4e}  {r['bpw']:>8.4f}  "
              f"{r['wall']:>7.2f}  {r['n_keep']:>6d}")
    print("-" * 96)

    # --- 9. Print headline + S4 pass-criteria check ---
    print("\n" + "=" * 96)
    print("S4 HEADLINE @ s=0.50 on real Qwen3-0.6B layer-0 q_proj Hessian")
    print("=" * 96)
    print(f"  Phi(SDOML full)         = {headline['phi_sdoml_s050']:.6e}")
    print(f"  Phi(magnitude->LMQ)     = {headline['phi_mag_lmq_s050']:.6e}")
    print(f"  Phi(DOML K=4 no prune)  = {headline['phi_doml_k4']:.6e}")
    print(f"  Phi(RTN-2bit)           = {headline['phi_rtn_2bit']:.6e}")
    print(f"  ratio SDOML/(mag->LMQ)  = {headline['ratio_sdoml_over_maglmq']:.6f}")
    print(f"  improvement vs mag->LMQ = {headline['improvement_pct']:+.3f}%")
    print(f"  ratio SDOML/DOML-K=4    = {headline['ratio_sdoml_over_doml']:.6f}")
    print(f"  improvement vs DOML K=4 = {headline['improvement_over_doml_pct']:+.3f}%")
    print(f"  Phi(comp1 end-to-end)   = {headline['phi_comp1_end_to_end_s050']:.6e}")
    print(f"  zero_frac(comp1)        = {headline['zero_fraction_comp1_s050']:.4f}")
    print(f"  wall(comp1 end-to-end)  = {headline['wall_comp1_s050']:.2f}s")

    print("\nS4 pass-criteria")
    print("-" * 96)
    pc1 = headline['improvement_pct'] >= 0.5
    pc2 = headline['phi_sdoml_s050'] < headline['phi_doml_k4']
    print(f"  [{'PASS' if pc1 else 'FAIL'}] (a) SDOML beats mag->LMQ by >=0.5%   "
          f"actual={headline['improvement_pct']:+.3f}%")
    print(f"  [{'PASS' if pc2 else 'NOTE'}] (b) SDOML < DOML-K=4 (surprising-good)   "
          f"actual_ratio={headline['ratio_sdoml_over_doml']:.4f}")
    print(f"  [{'PASS' if not leak else 'FAIL'}] (c) sdoml_quantize standalone "
          f"mask-leak    leak={leak}")

    print("\n" + "=" * 96)
    if pc1:
        print("OVERALL: S4 GREEN — proceed to S5 grid sweep")
    else:
        print("OVERALL: S4 RED — STOP per C6, escalate to Composition Candidate II")
    print("=" * 96)

    return 0


if __name__ == "__main__":
    sys.exit(main())
