"""AWQ / SmoothQuant-style per-channel activation-scaling transform.

This is a *reparametrization* of the full-precision (FP) model that is applied
BEFORE DOML (or any other) weight quantization.  It migrates a positive
per-input-channel scale ``s`` (length = in_features) from the activations into
the weights of the linears that consume an RMSNorm output, while dividing the
feeding RMSNorm weight by the same ``s``:

    W  <-  W . diag(s)      (scale the weight COLUMNS / in_features axis)
    g  <-  g / s            (scale the RMSNorm weight elementwise)

For a linear ``Y = W . RMSNorm_g(h)`` this is *exactly* output-preserving on the
FP model, because RMSNorm's output channel ``j`` is ``g_j * normalized(h)_j`` and

    (W . diag(s)) . RMSNorm_{g/s}(h)
        = sum_j W[:,j] * s_j * (g_j / s_j) * normalized(h)_j
        = sum_j W[:,j] * g_j * normalized(h)_j
        = W . RMSNorm_g(h).

The RMS *denominator* depends only on ``h`` (not on ``g``), so scaling ``g`` does
not change it.  AWQ picks ``s`` to protect high-activation channels:

    s_j = a_j ** alpha,   a_j = mean_tokens |RMSNorm_output_j|,

then normalizes ``s`` so its geometric mean is 1 (``s /= s.prod()**(1/len(s))``,
computed stably in log-space so the transform folds entirely into the existing
RMSNorm weight -> ZERO extra stored parameters -> bpw unchanged).

Scope (v1): only the two RMSNorm-fed scale groups per decoder layer:
    input_layernorm           -> shared input of {q_proj, k_proj, v_proj}
    post_attention_layernorm   -> shared input of {gate_proj, up_proj}
o_proj / down_proj are intentionally NOT touched (their scale group folds into
linear *rows* with GQA head-mapping complications).  Qwen3's per-head
``q_norm`` / ``k_norm`` act on the head dimension *after* q/k projection and are
irrelevant to input scaling, so they are left alone.

Works for both Llama (``LlamaForCausalLM``) and Qwen3 (``Qwen3ForCausalLM``):
both expose ``model.model.layers[i].{self_attn.{q,k,v,o}_proj,
mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm}``.  GQA
(``num_key_value_heads < num_attention_heads``) only shrinks the *out_features*
of k/v_proj; the *in_features* (columns we scale) equal hidden_size for all of
q/k/v, so a single shared ``s`` of length hidden_size applies cleanly.
"""

import os
import argparse

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Model-structure helpers (Llama / Qwen3 share the same layout)
# ---------------------------------------------------------------------------
def _get_decoder_layers(model):
    """Return the ModuleList of decoder layers for a Llama/Qwen3 CausalLM."""
    return model.model.layers


# Each entry: (rmsnorm_attr_name, [linear_module_accessors]).  The linears all
# consume the *output* of the named RMSNorm, so they share one scale vector.
def _layer_scale_groups(layer):
    """Return the in-scope (norm_name, [linear_modules]) groups for one layer."""
    return [
        (
            "input_layernorm",
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        ),
        (
            "post_attention_layernorm",
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        ),
    ]


_GROUP_KEYS = ("input_layernorm", "post_attention_layernorm")


# ---------------------------------------------------------------------------
# Scale computation
# ---------------------------------------------------------------------------
def _awq_scale_from_activation(a, alpha):
    """s_j = a_j**alpha, normalized so the geometric mean of s is 1.

    Computed in log-space for numerical stability:
        log s = alpha * log a ;  log s -= mean(log s) ;  s = exp(log s)
    which is exactly ``s = a**alpha / (a**alpha).prod()**(1/n)``.
    """
    a = a.to(torch.float32)
    log_a = torch.log(a.clamp_min(1e-8))
    log_s = alpha * log_a
    log_s = log_s - log_s.mean()            # geometric-mean normalization
    return torch.exp(log_s)


def _as_id_batches(calib_input_ids):
    """Normalize assorted calibration containers into a list of 2D LongTensors.

    Accepts:
      * a single tensor of shape (T,) or (N, T);
      * an iterable of tensors, each (T,) or (B, T);
      * an iterable of ``(input_ids, target)`` tuples (the repo's dataloader
        format from ``datautils.get_loaders``) -> uses element 0.
    A (N, T) tensor is split into N single-row batches so the forward memory
    footprint matches the quantization pipeline (batch size 1).
    """
    if isinstance(calib_input_ids, torch.Tensor):
        t = calib_input_ids
        if t.dim() == 1:
            return [t.unsqueeze(0)]
        return [t[i : i + 1] for i in range(t.shape[0])]

    batches = []
    for item in calib_input_ids:
        inp = item[0] if isinstance(item, (tuple, list)) else item
        if not isinstance(inp, torch.Tensor):
            inp = torch.as_tensor(inp)
        batches.append(inp if inp.dim() >= 2 else inp.unsqueeze(0))
    return batches


@torch.no_grad()
def collect_awq_scales(model, calib_input_ids, alpha=0.5, device="cuda:0"):
    """Accumulate per-channel mean(|RMSNorm output|) over the calibration tokens
    and turn it into an AWQ scale vector per (layer, norm-group).

    Forward hooks on every layer's ``input_layernorm`` and
    ``post_attention_layernorm`` record ``sum(|output|)`` over all tokens; the
    running total is divided by the token count to get ``a_j``.

    Returns: ``list`` (one dict per decoder layer) of
        ``{"input_layernorm": s_vec, "post_attention_layernorm": s_vec}``,
    each ``s_vec`` a float32 tensor (length = in_features) with geometric mean 1.
    """
    layers = _get_decoder_layers(model)
    n_layers = len(layers)

    # Per-layer per-norm running sum of |activation| over tokens (float32).
    acc = [{k: None for k in _GROUP_KEYS} for _ in range(n_layers)]

    def make_hook(li, key):
        def hook(module, inp, out):
            o = out.detach()
            o = o.reshape(-1, o.shape[-1]).to(torch.float32).abs().sum(dim=0)
            if acc[li][key] is None:
                acc[li][key] = o
            else:
                acc[li][key] += o
        return hook

    handles = []
    for li, layer in enumerate(layers):
        for key in _GROUP_KEYS:
            handles.append(getattr(layer, key).register_forward_hook(make_hook(li, key)))

    batches = _as_id_batches(calib_input_ids)

    was_training = model.training
    model.eval()
    use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    total_tokens = 0
    try:
        for b in batches:
            b = b.to(device)
            if b.dim() == 1:
                b = b.unsqueeze(0)
            model(b)
            total_tokens += b.shape[0] * b.shape[1]
    finally:
        for h in handles:
            h.remove()
        if use_cache is not None:
            model.config.use_cache = use_cache
        if was_training:
            model.train()

    denom = max(total_tokens, 1)
    scales = []
    for li in range(n_layers):
        d = {}
        for key in _GROUP_KEYS:
            a = acc[li][key] / denom                    # mean |activation| per channel
            d[key] = _awq_scale_from_activation(a, alpha)
        scales.append(d)
    return scales


# ---------------------------------------------------------------------------
# Transform application (in-place, output-preserving)
# ---------------------------------------------------------------------------
@torch.no_grad()
def apply_awq_(model, scales):
    """Apply the column-scaling + norm-division transform IN PLACE.

    For each in-scope group: ``W_col_j *= s_j`` for every consuming linear and
    ``g_j /= s_j`` for the feeding RMSNorm.  No tensor is created or deleted, so
    the stored parameter count / bpw is unchanged.
    """
    layers = _get_decoder_layers(model)
    if len(scales) != len(layers):
        raise ValueError(f"scales has {len(scales)} entries but model has {len(layers)} layers")

    for li, layer in enumerate(layers):
        for key, linears in _layer_scale_groups(layer):
            s = scales[li][key]
            norm = getattr(layer, key)
            in_features = linears[0].weight.shape[1]
            if s.numel() != in_features:
                raise ValueError(
                    f"layer {li} {key}: scale length {s.numel()} != in_features {in_features}")
            if s.numel() != norm.weight.numel():
                raise ValueError(
                    f"layer {li} {key}: scale length {s.numel()} != norm weight {norm.weight.numel()}")

            for lin in linears:
                # W <- W . diag(s): scale columns (in_features axis).
                s_w = s.to(dtype=lin.weight.dtype, device=lin.weight.device)
                lin.weight.data.mul_(s_w.unsqueeze(0))     # (out,in) *= (1,in)

            # g <- g / s: fold the inverse scale into the RMSNorm weight.
            s_n = s.to(dtype=norm.weight.dtype, device=norm.weight.device)
            norm.weight.data.div_(s_n)
    return model


@torch.no_grad()
def apply_awq_from_calib(model, calib_input_ids, alpha=0.5, device="cuda:0"):
    """Convenience: collect AWQ scales from ``calib_input_ids`` then apply them.

    The model is temporarily moved to ``device`` for the forward passes if it is
    not already there, and restored to its original device afterwards (so a
    downstream layer-by-layer quantizer that expects the model on CPU is
    unaffected).

    Returns the ``scales`` list (one dict per decoder layer) so the caller can
    persist it (via ``save_scales``); the model itself is transformed in place.
    """
    orig_device = next(model.parameters()).device
    moved = str(orig_device) != str(device)
    if moved:
        model.to(device)
    try:
        scales = collect_awq_scales(model, calib_input_ids, alpha=alpha, device=device)
        apply_awq_(model, scales)
    finally:
        if moved:
            model.to(orig_device)
    return scales


# ---------------------------------------------------------------------------
# CRB_SALIENT_METRIC=actmag support (2026-07-23): compute the AWQ scales but
# do NOT apply the W <- W.diag(s) / g <- g/s transform. The scales are stashed
# on the covered linear modules as a plain attribute so DOML's salient-column
# search (utils/structure.py metric "actmag") can rank columns by
# s_j * sum_i |W_ij| while the quantizer still sees the UNMODIFIED weights.
# This isolates AWQ's partition-realignment effect from its value reshaping.
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_scales_from_calib(model, calib_input_ids, alpha=0.5, device="cuda:0"):
    """Exactly :func:`apply_awq_from_calib` minus the ``apply_awq_`` step:
    run the calibration forward pass, return the per-layer scales list
    (same structure/math), leave every weight and norm untouched."""
    orig_device = next(model.parameters()).device
    moved = str(orig_device) != str(device)
    if moved:
        model.to(device)
    try:
        scales = collect_awq_scales(model, calib_input_ids, alpha=alpha, device=device)
    finally:
        if moved:
            model.to(orig_device)
    return scales


@torch.no_grad()
def attach_selection_scales(model, scales, attr="_crb_actmag_s"):
    """Stash each covered linear's per-in-feature scale vector on the module
    as a PLAIN attribute (not a Parameter, not a buffer: it is invisible to
    state_dict/save/bpw accounting and is not moved by ``.to()`` — readers
    must ``.to()`` it themselves). Scope = the AWQ v1 norm groups
    (input_layernorm -> q/k/v, post_attention_layernorm -> gate/up); o_proj
    and down_proj get NO stash, so the selection code falls back to the plain
    magnitude score for them.

    Returns ``(n_covered, n_fallback)`` linear counts for logging."""
    layers = _get_decoder_layers(model)
    if len(scales) != len(layers):
        raise ValueError(
            f"scales has {len(scales)} entries but model has {len(layers)} layers")
    n_covered = 0
    n_fallback = 0
    for li, layer in enumerate(layers):
        for key, linears in _layer_scale_groups(layer):
            s = scales[li][key].detach().to(torch.float32).cpu().contiguous()
            for lin in linears:
                if s.numel() != lin.weight.shape[1]:
                    raise ValueError(
                        f"layer {li} {key}: scale length {s.numel()} != "
                        f"in_features {lin.weight.shape[1]}")
                setattr(lin, attr, s)
                n_covered += 1
        # v1 scope: these quantized linears are intentionally uncovered.
        n_fallback += 2                      # o_proj + down_proj
    return n_covered, n_fallback


# ---------------------------------------------------------------------------
# Scale persistence (needed because the AWQ transform folds the inverse scale
# into the RMSNorm weights, which the DPK dump — quantized LINEARS only — does
# NOT store; restore/btune/atune load a pristine FP model and must re-apply the
# ``g <- g/s`` fold from the saved ``s``).
# ---------------------------------------------------------------------------
def save_scales(scales, path):
    """Serialize the AWQ ``scales`` list to a safetensors file at ``path``.

    ``scales`` is a list (one dict per decoder layer) of
        ``{"input_layernorm": s_vec, "post_attention_layernorm": s_vec}``.
    Keys are written as ``L{li}.input_layernorm`` / ``L{li}.post_attention_layernorm``
    with each vector stored as float32 (length = hidden_size)."""
    tensors = {}
    for li, d in enumerate(scales):
        for key in _GROUP_KEYS:
            tensors[f"L{li}.{key}"] = d[key].detach().to(torch.float32).cpu().contiguous()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    save_file(tensors, path)
    return path


def load_scales(path):
    """Inverse of :func:`save_scales`: return the same list-of-dicts structure.

    The decoder-layer count is inferred from the ``L{li}.*`` key prefixes."""
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    n_layers = 0
    for k in tensors:
        li = int(k.split(".", 1)[0][1:])       # "L{li}.<key>" -> li
        n_layers = max(n_layers, li + 1)
    scales = []
    for li in range(n_layers):
        scales.append({key: tensors[f"L{li}.{key}"] for key in _GROUP_KEYS})
    return scales


# ---------------------------------------------------------------------------
# AWQ v2 (2026-07-24, opt-in via CRB_AWQ_V2=1 on top of CRB_AWQ_ALPHA):
# extend the activation scaling to the two per-layer linears v1 skips,
# o_proj and down_proj.  Unlike v1 the inverse fold does NOT go into an
# RMSNorm weight — it goes into the *rows* (out_features) of another
# QUANTIZED linear in the same block, BEFORE quantization:
#
#   down_proj:  a_j = mean|input col j|  (the SwiGLU output silu(g) ⊙ u).
#               W_down <- W_down . diag(s);  up_proj rows <- rows / s.
#               Exact because (silu(g) ⊙ u) / s = silu(g) ⊙ (u / s):
#               dividing ONLY up_proj's output rows by s divides down_proj's
#               input elementwise by s (gate untouched).
#
#   o_proj:     a_j = mean|input col j|  (concatenated attention head
#               outputs).  GQA: o_proj input columns [h*hd:(h+1)*hd) for
#               query-head h are produced by v-head h // n_rep
#               (n_rep = num_attention_heads // num_key_value_heads, HF
#               repeat_kv layout: query head h = kv*n_rep + r).  The inverse
#               fold lands in v_proj's output rows, shared across the n_rep
#               query heads, so the scale MUST be tied: pool a_j by MEAN over
#               the n_rep replicas of the same v-head column (keep per-column
#               within head_dim), s_pooled = pooled**alpha geo-mean-
#               normalized; o_proj col (h, d) uses s_pooled[h//n_rep, d];
#               v_proj row (kv, d) <- / s_pooled[kv, d].  The attention mix
#               out_h = A_h @ v_(h//n_rep) is linear in v, so scaling v rows
#               by 1/s scales o_proj input cols by 1/s exactly — output-
#               preserving for ANY attention weights.
#
# Both folds live entirely in quantized linears (o/down scaled, v/up
# inverse-scaled) inside one transformer block, so the DPK dump captures
# everything: restore/btune/atune need NO v2 fold (block_fp remains a valid
# pristine target — block outputs are preserved).  v2 scales are saved to a
# separate awq_v2_scales.safetensors for ANALYSIS ONLY; the restore-time
# norm fold (awq_scales.safetensors) stays v1-only.
# ---------------------------------------------------------------------------
_V2_GROUP_KEYS = ("o_proj", "down_proj")
_V2_SAVE_KEYS = ("o_proj", "o_proj_pooled", "down_proj")


def _gqa_dims(model):
    """(n_heads, n_kv_heads, head_dim, n_rep) from the HF config.

    ``head_dim`` prefers the explicit config field (Qwen3-1.7B: 128) and
    falls back to hidden_size // num_attention_heads (Llama)."""
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", None) or n_heads)
    head_dim = int(getattr(cfg, "head_dim", None)
                   or (cfg.hidden_size // n_heads))
    if n_heads % n_kv != 0:
        raise ValueError(
            f"num_attention_heads {n_heads} not divisible by "
            f"num_key_value_heads {n_kv}")
    return n_heads, n_kv, head_dim, n_heads // n_kv


def _v2_o_proj_scales(a, alpha, n_heads, n_kv, head_dim):
    """GQA-tied o_proj scale from mean-|input| ``a`` (length n_heads*head_dim).

    Pool ``a`` by MEAN over the n_rep query-head replicas of each v-head
    (reshape (n_kv, n_rep, head_dim) — valid because HF repeat_kv orders
    query head h = kv*n_rep + r), keep per-column within head_dim, then
    s_pooled = pooled**alpha geo-mean-normalized over the (n_kv*head_dim)
    group.  Returns ``(s_full, s_pooled_flat)`` where s_full[(h, d)] =
    s_pooled[h//n_rep, d] (length n_heads*head_dim; its geometric mean is
    also 1 since each pooled entry appears exactly n_rep times)."""
    n_rep = n_heads // n_kv
    a = a.to(torch.float32).reshape(n_kv, n_rep, head_dim)
    a_pooled = a.mean(dim=1)                              # (n_kv, head_dim)
    s_pooled = _awq_scale_from_activation(a_pooled.reshape(-1), alpha)
    s_pooled = s_pooled.reshape(n_kv, head_dim)
    s_full = (s_pooled.unsqueeze(1)
              .expand(n_kv, n_rep, head_dim)
              .reshape(n_heads * head_dim)
              .contiguous())
    return s_full, s_pooled.reshape(-1).contiguous()


@torch.no_grad()
def _v2_apply_group_(scaled_lin, inv_lin, s_cols, s_rows):
    """One v2 fold IN PLACE: ``scaled_lin`` W <- W.diag(s_cols) (columns /
    in_features axis); ``inv_lin`` output rows (and bias, if any) <- /s_rows.
    Exact output-preservation requires s_cols to be the row-scale s_rows
    broadcast onto scaled_lin's input columns (identity for down/up)."""
    if s_cols.numel() != scaled_lin.weight.shape[1]:
        raise ValueError(
            f"v2 col-scale length {s_cols.numel()} != in_features "
            f"{scaled_lin.weight.shape[1]}")
    if s_rows.numel() != inv_lin.weight.shape[0]:
        raise ValueError(
            f"v2 row-scale length {s_rows.numel()} != out_features "
            f"{inv_lin.weight.shape[0]}")
    sc = s_cols.to(dtype=scaled_lin.weight.dtype,
                   device=scaled_lin.weight.device)
    scaled_lin.weight.data.mul_(sc.unsqueeze(0))          # (out,in) *= (1,in)
    sr = s_rows.to(dtype=inv_lin.weight.dtype, device=inv_lin.weight.device)
    inv_lin.weight.data.div_(sr.unsqueeze(1))             # (out,in) /= (out,1)
    if inv_lin.bias is not None:                          # bias adds per-row
        inv_lin.bias.data.div_(sr.to(dtype=inv_lin.bias.dtype,
                                     device=inv_lin.bias.device))


@torch.no_grad()
def collect_awq_v2_scales(model, calib_input_ids, alpha=0.5, device="cuda:0"):
    """Accumulate per-channel mean|INPUT col j| of every layer's o_proj and
    down_proj over the calibration tokens (forward hooks reading ``inp[0]``)
    and turn it into the v2 scale vectors.

    Returns a list (one dict per decoder layer) of
        ``{"o_proj": s_full, "o_proj_pooled": s_pooled, "down_proj": s}``
    (float32; s_full length n_heads*head_dim, s_pooled length
    n_kv_heads*head_dim, down length intermediate_size)."""
    layers = _get_decoder_layers(model)
    n_layers = len(layers)
    n_heads, n_kv, head_dim, n_rep = _gqa_dims(model)

    acc = [{k: None for k in _V2_GROUP_KEYS} for _ in range(n_layers)]

    def make_hook(li, key):
        def hook(module, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float32).abs().sum(dim=0)
            if acc[li][key] is None:
                acc[li][key] = x
            else:
                acc[li][key] += x
        return hook

    handles = []
    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_hook(
            make_hook(li, "o_proj")))
        handles.append(layer.mlp.down_proj.register_forward_hook(
            make_hook(li, "down_proj")))

    batches = _as_id_batches(calib_input_ids)

    was_training = model.training
    model.eval()
    use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    total_tokens = 0
    try:
        for b in batches:
            b = b.to(device)
            if b.dim() == 1:
                b = b.unsqueeze(0)
            model(b)
            total_tokens += b.shape[0] * b.shape[1]
    finally:
        for h in handles:
            h.remove()
        if use_cache is not None:
            model.config.use_cache = use_cache
        if was_training:
            model.train()

    denom = max(total_tokens, 1)
    scales = []
    for li in range(n_layers):
        a_o = acc[li]["o_proj"] / denom
        if a_o.numel() != n_heads * head_dim:
            raise ValueError(
                f"layer {li}: o_proj in_features {a_o.numel()} != "
                f"n_heads*head_dim {n_heads * head_dim}")
        s_full, s_pooled = _v2_o_proj_scales(a_o, alpha, n_heads, n_kv,
                                             head_dim)
        a_d = acc[li]["down_proj"] / denom
        scales.append({
            "o_proj": s_full,
            "o_proj_pooled": s_pooled,
            "down_proj": _awq_scale_from_activation(a_d, alpha),
        })
    return scales


@torch.no_grad()
def apply_awq_v2_(model, scales):
    """Apply the v2 folds IN PLACE (o_proj/down_proj cols *= s, v_proj/up_proj
    rows /= s).  Returns the number of folded groups (2 per decoder layer)."""
    layers = _get_decoder_layers(model)
    if len(scales) != len(layers):
        raise ValueError(
            f"v2 scales has {len(scales)} entries but model has "
            f"{len(layers)} layers")
    n_groups = 0
    for li, layer in enumerate(layers):
        d = scales[li]
        _v2_apply_group_(layer.self_attn.o_proj, layer.self_attn.v_proj,
                         d["o_proj"], d["o_proj_pooled"])
        n_groups += 1
        _v2_apply_group_(layer.mlp.down_proj, layer.mlp.up_proj,
                         d["down_proj"], d["down_proj"])
        n_groups += 1
    return n_groups


@torch.no_grad()
def apply_awq_v2_from_calib(model, calib_input_ids, alpha=0.5,
                            device="cuda:0"):
    """Collect the v2 scales from a calibration forward pass (run on the
    CURRENT — i.e. already-v1-transformed, if v1 is active — model, so the
    stats match what will be quantized) then fold them in place.

    Returns ``(scales, meta)`` where meta carries the GQA mapping actually
    used (n_heads / n_kv_heads / head_dim / n_rep) and n_groups folded."""
    orig_device = next(model.parameters()).device
    moved = str(orig_device) != str(device)
    if moved:
        model.to(device)
    try:
        scales = collect_awq_v2_scales(model, calib_input_ids, alpha=alpha,
                                       device=device)
        n_groups = apply_awq_v2_(model, scales)
    finally:
        if moved:
            model.to(orig_device)
    n_heads, n_kv, head_dim, n_rep = _gqa_dims(model)
    meta = {"n_groups": n_groups, "n_heads": n_heads, "n_kv_heads": n_kv,
            "head_dim": head_dim, "n_rep": n_rep}
    return scales, meta


def save_v2_scales(scales, path):
    """Serialize the v2 ``scales`` list to safetensors at ``path`` (keys
    ``L{li}.o_proj`` / ``L{li}.o_proj_pooled`` / ``L{li}.down_proj``).
    BUILD-TIME ANALYSIS ARTIFACT ONLY: the v2 folds are fully captured by the
    dumped quantized linears, so no restore/btune/atune path may ever apply
    these (they all key on the v1-only awq_scales.safetensors filename)."""
    tensors = {}
    for li, d in enumerate(scales):
        for key in _V2_SAVE_KEYS:
            tensors[f"L{li}.{key}"] = (
                d[key].detach().to(torch.float32).cpu().contiguous())
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    save_file(tensors, path)
    return path


def load_v2_scales(path):
    """Inverse of :func:`save_v2_scales` (analysis only — never folded)."""
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    n_layers = 0
    for k in tensors:
        li = int(k.split(".", 1)[0][1:])
        n_layers = max(n_layers, li + 1)
    return [{key: tensors[f"L{li}.{key}"] for key in _V2_SAVE_KEYS}
            for li in range(n_layers)]


# ---------------------------------------------------------------------------
# --selftest CLI: proves output-preservation on the FP model
# ---------------------------------------------------------------------------
def _load_fp_model(model_name, device):
    """Load the FP model exactly like run.py:get_model (eager attn, safetensors,
    torch_dtype='auto'), reusing the repo loader when importable."""
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from run import get_model                 # same loader the repo uses
    model = get_model(model_name)
    model.eval()
    return model


def _get_calib_dataloader(model_name, model, nsamples, seed=0):
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from datautils import get_loaders        # same wikitext2 calibration path
    dataloader, _ = get_loaders(
        "wikitext2", nsamples=nsamples, seed=seed,
        model=model_name, seqlen=model.seqlen,
    )
    return dataloader


def _param_signature(model):
    """(#parameter tensors, total numel, #state_dict tensors) — must be
    invariant under the AWQ transform (bpw-neutral: no new stored tensors)."""
    n_param_tensors = sum(1 for _ in model.parameters())
    total_numel = sum(p.numel() for p in model.parameters())
    n_state = len(model.state_dict())
    return n_param_tensors, total_numel, n_state


def _selftest(model_name, alpha, device):
    print(f"\n=== AWQ selftest: {model_name}  alpha={alpha}  device={device} ===")
    model = _load_fp_model(model_name, device)

    # The transform is exactly output-preserving in real arithmetic; run the
    # gate in float32 so the numbers reflect the *math* (a bf16 store of the
    # reparametrized weights adds ~2^-8 rounding that is irrelevant once DOML
    # quantizes to ~2 bits, but would otherwise mask the correctness proof).
    orig_dtype = next(model.parameters()).dtype
    model = model.to(device=device, dtype=torch.float32)

    dataloader = _get_calib_dataloader(model_name, model, nsamples=16)
    calib = dataloader[:8]                    # calibration subset
    eval_batch = dataloader[12][0].to(device) # held-out sequence for the gate

    sig_before = _param_signature(model)

    with torch.no_grad():
        logits_before = model(eval_batch).logits.float().clone()

    apply_awq_from_calib(model, calib, alpha=alpha, device=device)

    with torch.no_grad():
        logits_after = model(eval_batch).logits.float().clone()

    sig_after = _param_signature(model)

    diff = (logits_after - logits_before).abs()
    max_abs = diff.max().item()
    denom = logits_before.abs().max().item()
    max_rel = max_abs / denom if denom > 0 else float("inf")

    print(f"  native dtype (loader)      : {orig_dtype}")
    print(f"  logits shape               : {tuple(logits_before.shape)}")
    print(f"  max |after-before|         : {max_abs:.3e}")
    print(f"  max |before|               : {denom:.3e}")
    print(f"  max relative error         : {max_rel:.3e}   (gate < 1e-3)")
    print(f"  param signature before     : {sig_before}")
    print(f"  param signature after      : {sig_after}")

    assert sig_before == sig_after, (
        f"parameter/state signature changed: {sig_before} -> {sig_after}")
    assert max_rel < 1e-3, f"relative error {max_rel:.3e} not < 1e-3"
    print(f"  RESULT                     : PASS (output-preserving, bpw-neutral)")
    return max_abs, max_rel


def main():
    parser = argparse.ArgumentParser(description="AWQ activation-scaling transform")
    parser.add_argument("--selftest", metavar="MODEL", type=str, default=None,
                        help="Model id to run the output-preservation gate on "
                             "(e.g. Qwen/Qwen3-0.6B, meta-llama/Llama-3.2-1B).")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="AWQ scaling exponent s_j = a_j**alpha (default 0.5).")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.selftest:
        _selftest(args.selftest, args.alpha, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
