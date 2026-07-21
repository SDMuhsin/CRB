"""K5b shared library — DPK W2A4 serving of Qwen3-0.6B (refit-g256).

This module is the single home of:
  * artifact paths for the ledgered serving config
    (downloads/doml_dumps/qwen3-0.6b/refit-g256: element mmode, g=256,
    3.7507 bpw aggregate, fake-quant PPL 38.152381896972656),
  * the PRE-REGISTERED A4 activation contract of
    llmdocs/cuda_kernel/03_k5_serving_design.md:
        a_s[t] = absmax(x[t, :]) / 7.5          (fp32)
        x_hat  = clamp(round(x / a_s) + 8, 0, 15)   (round-half-even, fp32)
    packed LSB-first as 8 nibbles per uint32 (doc 02 par.4). Zero-row
    guard: an all-zero token row has absmax 0; a_s := 1.0 for those rows
    (documented deviation-free choice: x/1 = 0 -> x_hat = 8 -> the kernel's
    (x_hat - 8) contract yields exactly 0 for every output),
  * DPKServeLinear — the serving module holding the container streams
    (b0/b1/m/s/cb AS LOADED; these tensors ARE the resident weight bytes,
    no unpacked [R, C] copy exists anywhere in the serving path — the
    global-dequant ban of doc 00 is upheld by construction and enforced by
    the component-split measurement in measure_dpk_serving.py),
  * RefA4Linear — the G-K5-2 torch REFERENCE module: unpacked bf16 weights
    from the SAME containers + the SAME A4 fake-quant of activations,
    computed with plain torch fp32 GEMM (no custom kernel),
  * the model builders that swap the 196 quantized sublayers (K5a pattern).

Kernel: dpk_matmul (K4b v2 build; M=1 -> GEMV, M>1 -> GEMM), contract
Y[t, r] = a_s_vec[t] * sum_j W[r, j] * (x_hat[t, j] - 8), bf16 out.

Model loading, PPL loop and nvidia-smi helpers are REUSED from
serve_common.py (K5a) — the Marlin side of that module is not touched.

No repo source file is modified by anything in kernels/serve/.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from serve_common import (  # noqa: E402
    LOG_DIR, N_QUANT_SUBLAYERS, REPO, all_layer_names, load_qwen_bf16,
)

_KCUDA = os.path.join(REPO, "kernels", "cuda")
_KPACK = os.path.join(REPO, "kernels", "pack")
for _p in (_KCUDA, _KPACK):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dpk_unpack  # noqa: E402  (K2 loader/unpacker — doc 02 normative)

DPK_DUMP_DIR = os.path.join(
    REPO, "downloads", "doml_dumps", "qwen3-0.6b", "refit-g256")
DPK_MANIFEST = os.path.join(DPK_DUMP_DIR, "manifest.json")

GATE_K51_MARKER = os.path.join(LOG_DIR, "gate_K5_1_PASS.json")
GATE_K52_MARKER = os.path.join(LOG_DIR, "gate_K5_2_PASS.json")

# Ledgered anchors (verification ledger, doc 00 / K5a §5) for reporting.
PPL_FAKEQUANT_W_ONLY = 38.152381896972656   # refit-g256, full-precision acts
PPL_MARLIN_W4A16 = 26.143260955810547       # K5a comparator
PPL_DOML_G128 = 31.0392                     # DOML anchor (g=128 artifacts)
PPL_FP16_REF = 20.9685                      # project FP16 benchmark

_EXT = None


def dpk_ext():
    """Cached JIT build of the K4b v2 kernels (kernels/cuda/build.py)."""
    global _EXT
    if _EXT is None:
        sys.path.insert(0, _KCUDA)
        from build import build_dpk
        _EXT = build_dpk()
    return _EXT


# ---------------------------------------------------------------------------
# A4 activation quantization (pre-registered contract, doc 03)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quantize_a4(x2: torch.Tensor):
    """x2: [M, C_orig] any float dtype -> (xh int32 [M, C_orig] in 0..15,
    a_s fp32 [M]).

    Contract (doc 03): per-token symmetric, a_s = absmax/7.5 fp32,
    x_hat = clamp(round(x / a_s) + 8, 0, 15), round-half-even (torch.round),
    fp32 math. The -8 code is reachable; +absmax maps to round(+7.5)=8 ->
    16 -> clamped to 15 (the inherent half-step asymmetry of excess-8,
    documented in doc 03 — not "fixed" by shrinking the range).

    Zero-row guard: rows with absmax == 0 get a_s := 1.0, so x/a_s = 0,
    x_hat = 8 everywhere, and the kernel's (x_hat - 8) yields exact zeros.
    """
    assert x2.dim() == 2
    xf = x2.to(torch.float32, copy=True)               # fp32 [M, C] (owned:
    # the in-place ops below must never touch the caller's tensor)
    a_s = xf.abs().amax(dim=1).div_(7.5)               # fp32 [M]
    a_s = torch.where(a_s == 0, torch.ones_like(a_s), a_s)
    xf.div_(a_s.unsqueeze(1)).round_().add_(8.0).clamp_(0.0, 15.0)
    return xf.to(torch.int32), a_s


def pack_a4_batch(xh: torch.Tensor, C_pad: int | None = None) -> torch.Tensor:
    """Batched LSB-first A4 nibble packing (doc 02 par.4).

    xh: int tensor [M, C_orig], values 0..15. If C_pad > C_orig, columns
    >= C_orig are padded with nibble 8 (the par.3 padding trick; x = 0).
    Returns uint32 [M, C_pad/8]; row t is bit-identical to
    kernels/ref/ref_w2a4.pack_a4(xh[t]) — property-gated in
    verify_dpk_model.py.
    """
    M, C = xh.shape
    if C_pad is None:
        C_pad = C
    assert C_pad % 8 == 0 and C_pad >= C, (C, C_pad)
    if C_pad != C:
        pad = torch.full((M, C_pad - C), 8, dtype=xh.dtype, device=xh.device)
        xh = torch.cat([xh, pad], dim=1)
    v = xh.reshape(M, C_pad // 8, 8)
    w64 = v[..., 0].to(torch.int64)
    for n in range(1, 8):
        w64 |= v[..., n].to(torch.int64) << (4 * n)
    w64 = torch.where(w64 >= 2**31, w64 - 2**32, w64)  # int32 bit pattern
    return w64.to(torch.int32).view(torch.uint32)


# ---------------------------------------------------------------------------
# Serving module (the DPK streams ARE the resident weight bytes)
# ---------------------------------------------------------------------------

class DPKServeLinear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear served by dpk_matmul.

    Buffers (exactly the container streams, doc 02 §2a — no other weight
    representation exists on this module):
      b0, b1, m : uint32 [R, C/32]   bit planes (LSB-first)
      s         : uint32 [C/32]      salient-column bitmap
      cb        : bf16   [R, NG, 3, 4] per-(row, group, partition) codebooks

    forward: bf16 activation in -> per-token A4 quantize+pack (contract
    above) -> dpk_matmul (fp32 accumulate, bf16 out) -> reshape. Dequant of
    W happens only tile-locally inside the kernel (doc 00 ban).
    """

    def __init__(self, tensors: dict, meta: dict):
        super().__init__()
        self.R = int(meta["R"])
        self.C = int(meta["C"])
        self.C_orig = int(meta["C_orig"])
        self.g = int(meta["g"])
        self.NG = int(meta["NG"])
        assert meta["mmode"] == "element", meta["mmode"]
        self.in_features = self.C_orig
        self.out_features = self.R
        for k in ("b0", "b1", "m", "s", "cb"):
            self.register_buffer(k, tensors[k].contiguous())
        # G-K5-1 capture hook: when set to a list, forward appends
        # (xhat_words, a_s_vec, y_bf16) — all detached.
        self._capture = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        xh, a_s = quantize_a4(x2)
        xw = pack_a4_batch(xh, self.C)
        del xh
        y = dpk_ext().dpk_matmul(self.b0, self.b1, self.m, self.s, self.cb,
                                 xw, a_s, self.g)
        if self._capture is not None:
            self._capture.append(
                (xw.detach(), a_s.detach(), y.detach()))
        return y.to(x.dtype).reshape(*shp[:-1], self.R)

    def stream_bytes(self) -> dict:
        return {k: getattr(self, k).nelement() * getattr(self, k).element_size()
                for k in ("b0", "b1", "m", "s", "cb")}

    def extra_repr(self) -> str:
        return (f"in={self.C_orig}, out={self.R}, g={self.g}, NG={self.NG}, "
                f"mmode=element")


# ---------------------------------------------------------------------------
# G-K5-2 reference module (same math, no custom kernel)
# ---------------------------------------------------------------------------

class RefA4Linear(nn.Module):
    """Torch reference for G-K5-2: bf16 W unpacked from the SAME container
    + the SAME A4 fake-quant of activations, plain fp32 GEMM (TF32 off is
    asserted by the verify script). NOT a serving path — exists only to
    prove end-to-end math parity of the kernel model."""

    def __init__(self, W_bf16: torch.Tensor):
        super().__init__()
        self.register_buffer("W", W_bf16.contiguous())     # [R, C_orig] bf16
        self.out_features, self.in_features = W_bf16.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        xh, a_s = quantize_a4(x2)                          # SAME contract
        xq = (xh.float() - 8.0) * a_s.unsqueeze(1)         # a_s*(x_hat-8) fp32
        y = xq @ self.W.float().t()                        # fp32 GEMM
        return y.to(torch.bfloat16).to(x.dtype).reshape(*shp[:-1],
                                                        self.out_features)


# ---------------------------------------------------------------------------
# Model builders (K5a swap pattern; embeddings/lm_head/norms stay bf16)
# ---------------------------------------------------------------------------

def dpk_path(layer_name: str, dump_dir: str = DPK_DUMP_DIR) -> str:
    return os.path.join(dump_dir, f"{layer_name}.dpk.safetensors")


def _get_parent(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]


def build_dpk_model(dump_dir: str = DPK_DUMP_DIR):
    """bf16 Qwen3-0.6B skeleton with all 196 quantized sublayers replaced by
    DPKServeLinear built from the ledgered refit-g256 containers. Returns
    (model_on_cpu, n_replaced)."""
    model = load_qwen_bf16()
    n = 0
    for lname in all_layer_names(model.config.num_hidden_layers):
        parent, leaf = _get_parent(model, lname)
        orig = getattr(parent, leaf)
        assert isinstance(orig, nn.Linear), (lname, type(orig))
        assert orig.bias is None, f"{lname} has a bias; serving module is bias-free"
        tensors, meta = dpk_unpack.load_container(dpk_path(lname, dump_dir))
        assert (orig.in_features, orig.out_features) == \
               (meta["C_orig"], meta["R"]), lname
        setattr(parent, leaf, DPKServeLinear(tensors, meta))
        n += 1
    assert n == N_QUANT_SUBLAYERS, n
    return model, n


def build_ref_a4_model(dump_dir: str = DPK_DUMP_DIR):
    """G-K5-2 reference model: same skeleton, same 196 sublayers, but each
    replaced by RefA4Linear holding the bf16 weights unpacked from the SAME
    container (truncated to C_orig; bitwise-equal to the .wq ground truth
    per the ledgered dpk_verify 196/196 run)."""
    model = load_qwen_bf16()
    n = 0
    for lname in all_layer_names(model.config.num_hidden_layers):
        parent, leaf = _get_parent(model, lname)
        orig = getattr(parent, leaf)
        assert isinstance(orig, nn.Linear), (lname, type(orig))
        tensors, meta = dpk_unpack.load_container(dpk_path(lname, dump_dir))
        W = dpk_unpack.unpack(tensors, meta)[:, :meta["C_orig"]]
        assert (orig.in_features, orig.out_features) == tuple(W.shape[::-1])
        setattr(parent, leaf, RefA4Linear(W))
        n += 1
    assert n == N_QUANT_SUBLAYERS, n
    return model, n


# ---------------------------------------------------------------------------
# Byte accounting (measure_dpk_serving.py + reconciliation)
# ---------------------------------------------------------------------------

def manifest_stream_bytes(manifest_path: str = DPK_MANIFEST):
    """Theoretical per-layer stream bytes from the ledgered dump manifest
    (packed_bits are the dpk_verify-reconciled per-layer stream sizes).
    Returns (total_bytes, total_quantized_params, aggregate_bpw)."""
    with open(manifest_path) as f:
        man = json.load(f)
    tot_bits = sum(rec["packed_bits"] for rec in man["layers"])
    tot_elems = sum(rec["R"] * rec["C"] for rec in man["layers"])
    assert tot_bits % 8 == 0
    return tot_bits // 8, tot_elems, tot_bits / tot_elems


def dpk_component_split(model):
    """Byte accounting of every parameter/buffer, deduplicated by storage
    pointer (mirrors K5a measure_marlin_baseline.component_split; DPK
    streams split per plane)."""
    seen = set()
    comp = {"dpk_b0": 0, "dpk_b1": 0, "dpk_m": 0, "dpk_s": 0, "dpk_cb": 0,
            "embed_lmhead_bf16": 0, "norms_bf16": 0, "other": 0}
    counts = dict.fromkeys(comp, 0)

    def _add(key, t):
        ptr = t.untyped_storage().data_ptr()
        if ptr in seen:
            return False
        seen.add(ptr)
        comp[key] += t.nelement() * t.element_size()
        counts[key] += 1
        return True

    for name, mod in model.named_modules():
        if isinstance(mod, DPKServeLinear):
            for k in ("b0", "b1", "m", "s", "cb"):
                _add("dpk_" + k, getattr(mod, k))

    embed_w = model.model.embed_tokens.weight
    lm_w = model.lm_head.weight
    tied = embed_w.untyped_storage().data_ptr() == lm_w.untyped_storage().data_ptr()
    _add("embed_lmhead_bf16", embed_w)
    _add("embed_lmhead_bf16", lm_w)   # no-op if tied

    for name, p in model.named_parameters():
        key = "norms_bf16" if ("norm" in name) else "other"
        _add(key, p)
    for name, b in model.named_buffers():
        if b.device.type != "cuda":
            continue
        _add("other", b)

    comp["_counts"] = counts
    comp["_lm_head_tied_to_embed"] = tied
    return comp


def assert_no_global_dequant(model):
    """Measurement-enforced ban (doc 00): the served model must not hold ANY
    [R, C]-shaped floating-point weight tensor for the 196 quantized
    sublayers. Scans every param/buffer of every DPKServeLinear."""
    for name, mod in model.named_modules():
        if not isinstance(mod, DPKServeLinear):
            continue
        for tname, t in list(mod.named_parameters(recurse=False)) + \
                list(mod.named_buffers(recurse=False)):
            if t.is_floating_point() and t.dim() >= 2 and \
                    t.shape[-1] >= mod.C_orig and t.shape[0] >= mod.R:
                raise AssertionError(
                    f"{name}.{tname}: resident {tuple(t.shape)} "
                    f"{t.dtype} tensor looks like a global dequant buffer")
    return True
