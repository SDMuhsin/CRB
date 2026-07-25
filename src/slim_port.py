"""[Vendored 2026-07-25 from PRISM repo benchmarks/slim_port.py — keep in sync]

Faithful port of SLiM (Mozaffari et al., ICML 2025) core math, for use as a PRISM baseline.

SLiM = "One-shot Quantized Sparse Plus Low-rank Approximation of LLMs" (arXiv 2410.09615).
Upstream: github.com/Mohammad-Mozaffari/slim (cloned to ./temp/SLiM). This module
re-implements the *pure-math* pieces of the **SLiM-LoRA** path (the headline method =
`prune_wanda` in slim/prune.py) so they can be driven by the PRISM benchmark harness on any
LLaMA-/Qwen2-/OPT-style model, with NO dependency on the (gitignored, ephemeral) temp/SLiM clone.

The SLiM-LoRA recipe, per linear layer:
  1. Wanda prune: mask = smallest |W|*sqrt(act_norm) per row (with `shift_zeros` on act_norm).
  2. SLiM-Quant: per-MATRIX symmetric quant with an MSE-optimal clip cap chosen from the
     weight-magnitude histogram (`slim_quant_find_cap`). NOT group/block quant.
  3. Saliency low-rank adapter: SVD of the *activation-saliency-weighted* compression error
     `(W - dequant(quant(W)))*sqrt(act_norm)`, then divide the left factor by sqrt(act_norm)
     to map back to weight space. Re-prune + re-quantize `W - LR`; the fp16 adapter `LR` is
     added back at inference. Adapter rank = int(rank_ratio * min(W.shape)) (paper: r=0.1).

Each function is component-tested numerically equivalent to upstream (see
scripts/slim_component_test.py); the equivalence is exact up to bf16 rounding because we call
the same torch.svd / torch.histogram on the same inputs with the same dtype casts as upstream.

Fidelity notes / exact correspondences to temp/SLiM:
- `slim_quant_find_cap`  == slim/quantization/quantization.py::find_optimal_quantiztion_cap
                            (integrate=True branch) + compute_average_error. The cost model
                            uses 2**num_bits levels (q=num_bits), an upstream quirk we keep.
- `slim_quantize_weight` == Quantizer.quantize_weight/dequantize_absmax with slim_quant=True,
                            block_quantization=False, important_columns=None (per-matrix absmax
                            with the optimal cap; max_q = 2**(num_bits-1)-1).
- `slim_lora_decompose`  == lora.py::add_lora with slim_lora=True, separate_lora=True,
                            quantize_first=True, quantizer set, fp16 adapter (quantize_lora=False).
- `SLiMLinear` forward    == prune.py separate_lora fp16 hook (the +-=sqrt(rank) rescale cancels):
                            y = x @ Wcomp^T + (x @ lora_left) @ lora_right + bias.
"""

import os

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# SLiM-Quant: per-matrix symmetric quant with MSE-optimal clip cap            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _compute_average_error(pdf, val, index, q):
    """Exact port of quantization.py::compute_average_error.

    Quantization+clipping MSE if the abs-cap is placed at val[index]. `q` is the bit count
    used in the (uniform) level model: step = alpha / 2**q  (upstream uses q=num_bits)."""
    alpha = val[index]
    dx = val[1] - val[0]

    pdf_q = pdf[:index]
    acc_q = val[:index]
    step = alpha / (2 ** q)
    if step == 0:
        quant_q = acc_q
    else:
        quant_q = (acc_q // step) * step
    quantization_loss = torch.sum(pdf_q * (acc_q - quant_q) ** 2) * dx

    pdf_clip = pdf[index:]
    val_clip = val[index:]
    clipping_loss = torch.sum(pdf_clip * (val_clip - alpha) ** 2) * dx
    return quantization_loss + clipping_loss


@torch.no_grad()
def slim_quant_find_cap(mat, num_bits, num_bins=None, nonzero_only=False):
    """Exact port of find_optimal_quantiztion_cap (integrate=True branch).

    Builds a density histogram of |W|, scans 11 coarse cap positions for the min-MSE region,
    then refines within that bracket. Returns the optimal abs-max cap (scalar tensor on
    mat.device).

    nonzero_only=True is the sparse-aware-statistics 'correction' (Study B): the MSE-optimal
    cap is estimated only over surviving (non-zero) weights, so the pruned zeros do not enter
    the |W| histogram. Effect is expected to be small because the histogram is density-
    normalized (a uniform rescale leaves the arg-min cap unchanged) and the zero spike
    contributes ~0 quantization error — the only real change is the histogram's lower range/
    bin edges shifting from 0 up to min|survivor|."""
    _dev = mat.device
    if nonzero_only:
        mat = mat[mat != 0]
        if mat.numel() == 0:
            return torch.tensor(1e-5, device=_dev)
    if num_bins is None:
        num_bins = max(512, min(torch.numel(mat) // 1000, 20000))
    device = mat.device
    flat = mat.detach().abs().float().flatten().cpu()
    pdf, edges = torch.histogram(flat, bins=num_bins, density=True)
    pdf = pdf.to(device)
    edges = edges.to(device)
    val = (edges[:-1] + edges[1:]) / 2  # bin centers, length num_bins
    q = num_bits

    total_loss = torch.zeros(num_bins, device=device) + torch.inf
    losses = torch.zeros(11, device=device) + torch.inf
    indices = torch.zeros(11, dtype=torch.int, device=device)
    j = 0
    for i in range(0, num_bins, num_bins // 10):
        if j >= 11:
            break
        losses[j] = _compute_average_error(pdf, val, i, q)
        indices[j] = i
        j += 1

    _, turning_point = torch.min(losses, 0)
    start = int(indices[max(turning_point - 1, 0)])
    end = int(indices[min(turning_point + 1, 10)])
    for i in range(start, end):
        total_loss[i] = _compute_average_error(pdf, val, i, q)

    _, idx = torch.min(total_loss, 0)
    return val[idx].to(mat.device)


@torch.no_grad()
def slim_quantize_weight(mat, num_bits, nonzero_only=False):
    """Per-matrix SLiM-Quant: symmetric round-to-nearest with the MSE-optimal cap.

    Returns the *dequantized* weight (fake-quant) in **float32** — matching upstream's
    Quantizer.dequantize_absmax, which returns float (quantized_int / scaling_factor) and does
    NOT cast back to the weight dtype. Callers cast to the model dtype only at storage points
    (as upstream does), so the error-matrix / SVD that drives the low-rank adapter is computed
    in float32 exactly as upstream — casting to bf16 early would perturb the SVD input and
    amplify through the chained re-quantization. Equivalent to dequantize_absmax(quantize_weight(
    mat)) with slim_quant=True, no block, no important columns. Pruned zeros map to 0.

    nonzero_only=True estimates the MSE-optimal cap over surviving weights only (the Study B
    correction); the round/clamp grid is otherwise unchanged."""
    max_q = 2 ** (num_bits - 1) - 1
    cap = slim_quant_find_cap(mat, num_bits, nonzero_only=nonzero_only)
    scaling_factor = max_q / cap
    q = torch.round((mat * scaling_factor).float())
    q = torch.clamp(q, -max_q - 1, max_q)
    deq = q / scaling_factor
    return deq.float()


# --------------------------------------------------------------------------- #
# SLiM-LoRA: saliency-weighted low-rank compensation of the compression error  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def slim_lora_decompose(weight, W_mask, scaler_row, num_bits, rank_ratio,
                        quantize=True, slim_lora=True, sparse_aware_cap=False):
    """Port of lora.py::add_lora (slim_lora, separate_lora, quantize_first, fp16 adapter).

    Args:
        weight:     (out, in) weight tensor (already in the model's dtype, e.g. bf16).
        W_mask:     (out, in) bool, True where pruned (Wanda mask).
        scaler_row: (in,) Wanda activation L2^2 norm per input channel (already `shift_zeros`-ed
                    upstream when shift_zero_metrics=True).
        num_bits:   weight quant bit-width (ignored if quantize=False).
        rank_ratio: adapter rank as a fraction of min(out, in) (paper: 0.1).
        quantize:   if False, prune-only SLiM-LoRA (no weight quant; paper Table 13).
        slim_lora:  if True, saliency-weighted SVD (SLiM-LoRA); if False, plain error SVD
                    (Naive-LoRA).

    Returns (Wcomp, lora_left, lora_right):
        Wcomp:      (out, in) sparse (+ quantized) weight to store as the dense matrix.
        lora_left:  (in, rank) fp16/bf16 adapter left factor.
        lora_right: (rank, out) fp16/bf16 adapter right factor.
    The reconstructed full weight is exactly Wcomp + (lora_left @ lora_right).T."""
    dtype = weight.dtype
    use_saliency = slim_lora and not bool((scaler_row == 0).any())

    if use_saliency:
        sqrt_act = torch.sqrt(scaler_row.reshape(1, -1))  # (1, in), weight dtype
        W_metric = weight * sqrt_act
        new_weight = weight.clone()
        if quantize:
            new_weight = slim_quantize_weight(new_weight, num_bits,
                                              nonzero_only=sparse_aware_cap) * sqrt_act
        else:
            new_weight = new_weight * sqrt_act
        new_weight[W_mask] = 0
        error_mat = W_metric - new_weight
    else:
        # Fallback (matches upstream else-branch): plain (un-weighted) error matrix.
        new_weight = weight.clone()
        if quantize:
            new_weight = slim_quantize_weight(new_weight, num_bits,
                                              nonzero_only=sparse_aware_cap)
        new_weight[W_mask] = 0
        error_mat = weight - new_weight

    # SVD on the (saliency-weighted) error matrix; cast factors to weight dtype as upstream.
    U, S, V = torch.svd(error_mat.float())
    rank = int(rank_ratio * min(error_mat.shape))
    rank = max(rank, 1)

    # separate_lora factors (upstream casts U/S/V to weight dtype before the matmul).
    Sd = torch.diag_embed(S[:rank]).to(dtype)
    lora_left = (Sd @ V[:, :rank].to(dtype).T).t()  # (in, rank)
    lora_right = U[:, :rank].to(dtype).t()           # (rank, out)

    if use_saliency:
        denom = sqrt_act.to(dtype)                   # (1, in)
        lora_left = lora_left / denom.t()            # (in, rank) / (in, 1)

    low_rank_weight = lora_right.t() @ lora_left.t()  # (out, in)

    new_weight = weight - low_rank_weight
    new_weight[W_mask] = 0
    if quantize:
        # This is the call where pruned zeros are present (W - LR has been masked), so
        # nonzero_only here is the operative sparse-aware-cap correction.
        new_weight = slim_quantize_weight(new_weight, num_bits, nonzero_only=sparse_aware_cap)

    return new_weight.to(dtype), lora_left.to(dtype), lora_right.to(dtype)


# --------------------------------------------------------------------------- #
# Sparse + low-rank linear (fake-quant-in-float; mirrors SparseLinear)          #
# --------------------------------------------------------------------------- #
class SLiMLinear(nn.Module):
    """Linear with a sparse (+ optionally quantized) weight plus a dense low-rank adapter.

    Forward is exactly upstream's separate_lora fp16 path (the +-sqrt(rank) rescale cancels):
        y = x @ Wcomp^T + (x @ lora_left) @ lora_right + bias

    `weight` (Wcomp) is stored dense-with-zeros so weight-matrix sparsity stays measurable; the
    low-rank adapter (lora_left/lora_right) is the SLiM compensation term and is the source of
    SLiM's extra param budget (~rank_ratio overhead), kept SEPARATE here so that overhead is
    explicit/auditable rather than folded silently into a dense effective weight."""

    def __init__(self, weight, lora_left, lora_right, bias=None):
        super().__init__()
        self.register_buffer('weight', weight)            # (out, in)
        self.register_buffer('lora_left', lora_left)      # (in, rank)
        self.register_buffer('lora_right', lora_right)    # (rank, out)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        out = torch.matmul(x, self.weight.t())
        ll = self.lora_left.to(x.dtype)
        lr = self.lora_right.to(x.dtype)
        out = out + torch.matmul(torch.matmul(x, ll), lr)
        if self.bias is not None:
            out = out + self.bias
        return out


@torch.no_grad()
def shift_zeros(x):
    """Port of slim/utils.py::shift_zeros: add the smallest positive value to all entries so
    no activation-norm column is exactly 0 (keeps the saliency SVD path active)."""
    mp = x.clone()
    mp[mp == 0] = 1
    mp = mp.min()
    return x + mp
