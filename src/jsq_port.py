"""[Vendored 2026-07-25 from PRISM repo benchmarks/jsq_port.py — keep in sync]

Faithful port of JSQ (Guo et al., ICML 2024) core math, for use as a PRISM baseline.

JSQ = "Compressing Large Language Models by Joint Sparsification and Quantization".
Upstream: github.com/uanu2002/JSQ (cloned to ./temp/JSQ). This module re-implements the
*pure-math* pieces (activation outlier editing, the SAR sparsity metric `ss`, SmoothQuant
smoothing, and the WnAn fake-quant linear) so they can be driven by the PRISM benchmark
harness on any LLaMA-/Qwen2-style model.

Each function here is component-tested against the upstream implementation
(see tests/jsq_equivalence at the bottom / scripts) so the port is provably faithful.

Key fidelity notes:
- `quantize_weight_per_channel_absmax` / `quantize_activation_per_token_absmax` are the
  exact symmetric absmax quantizers from `temp/JSQ/jsq/fake_quant.py` (q_max = 2**(b-1)-1).
- `generate_ss` is the vectorized form of upstream's O(in_features) loop. Upstream recomputes
  `activation @ W_with_col_i_zeroed.T` for every input channel i; that equals
  `out_full - outer(act[:, i], W[:, i])`, so we compute it as a rank-1 update over channel
  chunks -> numerically identical, but tractable at 7B scale.
- `smooth_ln_fcs` is upstream's SmoothQuant migration (alpha=0.5), works for RMSNorm.
"""

import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Fake quantization (exact copy of temp/JSQ/jsq/fake_quant.py, generalized nbits) #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    """Per-output-channel symmetric absmax weight quant (in-place on `w`)."""
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_channel_absmax_sparse_aware(w, n_bits=8):
    """Per-output-channel symmetric absmax weight quant whose scale is estimated ONLY over
    surviving (non-zero) weights — the sparse-aware-statistics 'correction' grafted onto JSQ's
    RTN quantizer (Study B).

    NOTE: because the scale is an abs-MAX, and pruned positions are exactly 0 (and 0 ≤ |any
    survivor|), max(|w|) over the full row equals max(|w|) over survivors whenever a row keeps
    at least one weight. So this is mathematically identical to the standard quantizer — it is an
    exact no-op, included as a negative control demonstrating that the correction is specific to
    variance/moment-based scale estimators (Sinkhorn), not max-based RTN. Implemented honestly
    (scale computed over the masked tensor) rather than asserted."""
    survivor = (w != 0)
    masked = w.abs() * survivor
    scales = masked.max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    """Per-token symmetric absmax activation quant. Returns a NEW tensor (no in-place
    mutation of the live activation, unlike upstream which mutates in place)."""
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5) / q_max
    return (t / scales).round() * scales


class WnAnLinear(nn.Module):
    """Linear with fake-quantized weights and (optionally) fake-quantized activations.

    Generalizes upstream's `W8A8Linear` to arbitrary `nbits` and a weight-only mode
    (`act_quant=False`), which collapses to a plain dense linear with quantized weights
    (equivalent to PRISM/SparseGPT's weight-only `SparseLinear`). Pruned zeros in the
    incoming weight are preserved: 0 -> 0 under absmax quant.
    """

    def __init__(self, in_features, out_features, bias=True, nbits=8,
                 act_quant=True, quantize_output=False, sparse_aware_scale=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nbits = nbits
        self.sparse_aware_scale = sparse_aware_scale
        self.act_quant_enabled = act_quant
        self.output_quant_enabled = act_quant and quantize_output
        self.register_buffer('weight', torch.zeros(out_features, in_features, dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x):
        if self.act_quant_enabled:
            x = quantize_activation_per_token_absmax(x, self.nbits)
        y = F.linear(x, self.weight, self.bias)
        if self.output_quant_enabled:
            y = quantize_activation_per_token_absmax(y, self.nbits)
        return y

    @staticmethod
    @torch.no_grad()
    def from_float(module, nbits=8, act_quant=True, quantize_output=False, sparse_aware_scale=False):
        assert isinstance(module, nn.Linear)
        new = WnAnLinear(module.in_features, module.out_features,
                         bias=module.bias is not None, nbits=nbits,
                         act_quant=act_quant, quantize_output=quantize_output,
                         sparse_aware_scale=sparse_aware_scale)
        w = module.weight.data.clone().float()
        if sparse_aware_scale:
            quantize_weight_per_channel_absmax_sparse_aware(w, n_bits=nbits)
        else:
            quantize_weight_per_channel_absmax(w, n_bits=nbits)
        new.weight = w.to(module.weight.dtype).to(module.weight.device)
        if module.bias is not None:
            new.bias = module.bias.data.clone().to(module.weight.device)
        return new


# ----------------------------------------------------------------------------- #
# Activation outlier editing (exact copy of clip_matrix, channel=False branch)    #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def clip_matrix(matrix, use_abs=True, clip_l=0.0, clip_h=0.01):
    """Clip the most-extreme activation entries (the 'activation editor').

    With clip_h=0.01, the top 1% largest-|value| entries (over the whole matrix) are
    clamped to the 99th-percentile magnitude. Used only to compute pruning/smoothing
    statistics (matches upstream, which clips the captured input, not the forward)."""
    if clip_l == 0 and clip_h == 0:
        return matrix
    n = matrix.numel()
    flat = torch.abs(matrix).flatten() if use_abs else matrix.flatten()
    max_threshold = None
    min_threshold = None
    if clip_l != 0:
        low_index = int(clip_l * n)
        min_threshold = torch.topk(flat, largest=False, k=low_index)[0][-1]
    if clip_h != 0:
        high_index = int(clip_h * n)
        max_threshold = torch.topk(flat, largest=True, k=high_index)[0][-1]
    if use_abs:
        return torch.clamp(matrix, -max_threshold, max_threshold)
    return torch.clamp(matrix, min_threshold, max_threshold)


# ----------------------------------------------------------------------------- #
# SAR auxiliary salience metric `ss` (vectorized rank-1 form of upstream loop)     #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def generate_ss(activation, weight, max_tokens=256, mem_budget=48_000_000):
    """Per-weight quantization-range sensitivity (auxiliary salience A, paper Eq. 3).

    For each input channel i, upstream computes the output range (max-min over tokens) of
    `activation @ W` with column i zeroed, and stores it into every row of ss[:, i].
    That equals `out_full - outer(activation[:, i], W[:, i])`, so we compute the range over
    tokens for chunks of input channels via a rank-1 update -- numerically identical to
    upstream's per-channel matmul (verified component-equivalent), but far cheaper.

    Speed: the range statistic is well-estimated from a subset of token rows, so we
    uniformly subsample to `max_tokens` (set max_tokens=0 to use all tokens for an exact
    match to upstream). Input channels are chunked to keep the (T, out, chunk) temporary
    under `mem_budget` elements.

    activation: (T, in_features) mean activation
    weight:     (out_features, in_features)
    returns ss: (out_features, in_features)
    """
    activation = activation.float()
    weight = weight.float()
    T, cin = activation.shape
    if max_tokens and T > max_tokens:
        idx = torch.linspace(0, T - 1, max_tokens, device=activation.device).round().long()
        activation = activation[idx]
        T = max_tokens
    out_features = weight.shape[0]
    out_full = activation @ weight.t()                      # (T, out)
    ss = torch.empty_like(weight)                           # (out, in)
    chunk = max(1, int(mem_budget // (T * out_features)))
    for s in range(0, cin, chunk):
        e = min(s + chunk, cin)
        a = activation[:, s:e]                              # (T, c)
        w = weight[:, s:e]                                  # (out, c)
        # out_i[t, o, c] = out_full[t, o] - a[t, c] * w[o, c]
        out_i = out_full.unsqueeze(2) - a.unsqueeze(1) * w.unsqueeze(0)  # (T, out, c)
        ss[:, s:e] = out_i.amax(dim=0) - out_i.amin(dim=0)  # (out, c)
    ss = torch.where(torch.isinf(ss), torch.full_like(ss, 100.0), ss)
    return ss


# ----------------------------------------------------------------------------- #
# SmoothQuant smoothing (exact copy of upstream smooth_ln_fcs)                     #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5, unit_offset=False):
    """SmoothQuant LN<->FC scale migration.

    `unit_offset=True` handles Gemma's RMSNorm, which applies a gain of
    ``(1 + weight)`` rather than ``weight``. Naively doing ``ln.weight.div_(s)``
    there is NOT scale-preserving (it scales `w`, not the effective gain `1+w`),
    which blows the activations up into fp16 overflow -> NaN. Instead we set the
    new weight so the *effective gain* is divided by `s`:  (1 + w)/s = 1 + w_new
    =>  w_new = (1 + w)/s - 1.  Standard (Llama/OPT/Qwen) norms use w directly."""
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype)
    if unit_offset:
        w = ln.weight.data.float()
        s = scales.float()
        ln.weight.data = (((1.0 + w) / s) - 1.0).to(dtype)
    else:
        ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


# ----------------------------------------------------------------------------- #
# Per-linear calibration stat collector (clip -> Wanda norm + mean act + max)      #
# ----------------------------------------------------------------------------- #
class JSQStatWrapper:
    """Collects, per linear and over clipped inputs: Wanda L2-norm (scaler_row),
    mean activation (for `ss`), and per-channel max (act_scales for smoothing)."""

    def __init__(self, layer, clip_h=0.01, use_abs=True):
        self.dev = layer.weight.device
        self.columns = layer.weight.shape[1]
        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.inp_sum = None
        self.inp_num = 0
        self.act_max = torch.zeros(self.columns, device=self.dev)
        self.nsamples = 0
        self.clip_h = clip_h
        self.use_abs = use_abs

    @torch.no_grad()
    def add_batch(self, inp):
        inp = clip_matrix(inp.data, self.use_abs, 0.0, self.clip_h)
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])     # (tokens, in)
        inp = inp.float()
        # per-channel max (for smoothing)
        self.act_max = torch.maximum(self.act_max, inp.abs().max(dim=0)[0])
        # mean activation for ss: accumulate (tokens, in) then average by #batches
        if self.inp_sum is None:
            self.inp_sum = inp.clone()
        else:
            n = min(self.inp_sum.shape[0], inp.shape[0])
            self.inp_sum[:n] += inp[:n]
        self.inp_num += 1
        # Wanda running mean of column L2-norm^2
        t = inp.shape[0]
        self.scaler_row *= self.nsamples / (self.nsamples + t)
        self.nsamples += t
        self.scaler_row += torch.norm(inp, p=2, dim=0) ** 2 / self.nsamples

    @torch.no_grad()
    def mean_activation(self):
        return self.inp_sum / max(self.inp_num, 1)
