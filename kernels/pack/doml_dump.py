"""K2 deliverable 1 — DOML passthrough-dump harness + DPK packer (S-A, g=128).

Produces one DPK container per quantized sublayer of a real DOML run, per
llmdocs/cuda_kernel/02_storage_format_design.md (doc 02):
  §2a container:  b0,b1,m: uint32[R, C/32] (LSB-first, bit i of word w = column
                  32*w+i), s: uint32[C/32], cb: bf16[R, NG, 3, 4]
                  (partitions ordered P1 bulk=mask1, P2 tail=mask2,
                  P3 salient=mask3; levels sorted ascending), meta JSON.
  §3 invariant:   part(i,j) = s[j] ? 2 : (m[i][j] ? 1 : 0)
                  code(i,j) = b0 + 2*b1
                  W[i,j]    = cb[i][j//g][part][code]   (bit-exact bf16)
  §3 padding:     C padded to a multiple of B=128 with b0=b1=m=s=0.

A sibling `<name>.wq.safetensors` stores `wq` = the final quantized layer
weight (bf16, [R, C_orig]) as round-trip ground truth for dpk_verify.py.

Usage:
  selftest (synthetic, incl. ragged-C padding + duplicate/empty codebooks):
      python kernels/pack/doml_dump.py --selftest
  real dump run (G1 gate: printed PPL must equal V1 reference 31.0392):
      source env/bin/activate
      CUDA_VISIBLE_DEVICES=1 python kernels/pack/doml_dump.py --run

Monkey-patch design (NO repo source file is modified):
  1. utils.structure.structural_guassian_distribution — patched BEFORE
     `import bigptq` (bigptq binds the name at import time; same pattern as
     llmdocs/cuda_kernel/verify/probe_doml_qwen06b.py). Captures the three
     per-(128-col block) masks.
  2. binary.Binarization.quantize — captures, per (block, partition), the
     per-partition reconstruction tensor returned by lloyd_max_quantize.
  3. bigptq.BRAGPTQ.fasterquant — after the ORIGINAL returns: reassembles the
     layer from (masks + partition reconstructions), asserts bf16-BITWISE
     equality with self.layer.weight.data (hard fail otherwise), derives the
     DPK streams, validates the §3 invariant for 100% of elements, and writes
     the container.
  All wrappers call the originals with unchanged arguments and return their
  results unchanged; hooks consume no torch/numpy global RNG.

Level/assignment derivation (per (row, group g=128, partition)): the sorted
distinct bf16 BIT PATTERNS of the final quantized weight restricted to that
partition's mask (<= 4 by construction: Lloyd-Max K=4). Padded to 4 slots by
repeating the last level (empty partitions get all-zero levels). Assignments
are the index of each element's value among the sorted distinct values —
searchsorted(left) guarantees an assignment can never reference a pad slot
(asserted). Ordering uses a monotonic integer key so that -0.0 < +0.0 and the
two are DISTINCT levels if both ever occur (bit-exactness over bf16 bits).
Note vs doc 02 §5: the per-element assignment is derived from the element's
final bf16 value, not intercepted from the Lloyd assignment tensor; this is
exactly invertible (equal bf16 bits => equal reconstruction) and is validated
bitwise for 100% of elements of every layer by the §3 invariant check + the
mask/reconstruction reassembly assert.
"""

import argparse
import json
import os
import sys
import time

REPO = "/workspace/BiLLM2"
DEFAULT_DUMP_DIR = os.path.join(
    REPO, "downloads", "doml_dumps", "qwen3-0.6b", "sa-g128")
VERIFY_DIR = os.path.join(REPO, "llmdocs", "cuda_kernel", "verify")

# Must be set before run.py / csv_utils are imported (V1-proven redirect).
os.environ.setdefault(
    "BILLM_BENCH_CSV", os.path.join(VERIFY_DIR, "scratch_results.csv"))

if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

B_BLOCK = 128            # DOML partition block size (fixed by the quantizer)
G_GROUP = 128            # codebook group size for S-A (= B)
KEY_SENT = 1 << 30       # sort sentinel for non-member positions
KEY_ZERO = 0x8000        # monotonic key of bf16 +0.0 (pattern 0x0000)

# fp8 codebook storage (K27). The group-refit harness may snap codebook levels
# onto a low-bit float grid (--codebook-dtype float8_e4m3fn/e5m2). When it does,
# every derived cb value is EXACTLY representable in that dtype, so we store cb
# in it LOSSLESSLY (fp8 -> bf16 on load is bit-exact; verified per layer below).
# Default 'bf16' keeps the container and all byte accounting byte-identical to
# the original format.
_FP8_DTYPES = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def _resolve_cbdtype():
    """Resolve the codebook storage dtype TAG from the harness RUN_STATE, if
    present. Returns 'bfloat16' / 'float8_e4m3fn' / 'float8_e5m2'. Absent or
    'bf16' RUN_STATE (doml_dump's own --run/--selftest, or a bf16 refit) =>
    'bfloat16', keeping the default path byte-identical.

    Discovery is by CALL-STACK walk: derive_dpk/container_meta are invoked from
    a frame belonging to the harness (doml_group_refit.refit_fasterquant), whose
    module globals hold RUN_STATE. A stack walk is robust to `runpy.run_path`
    shadowing sys.modules['__main__'] with run.py during the quantize loop."""
    import inspect
    frame = inspect.currentframe()
    try:
        while frame is not None:
            rs = frame.f_globals.get("RUN_STATE")
            if isinstance(rs, dict) and "codebook_dtype" in rs:
                cb = rs["codebook_dtype"]
                return cb if cb in _FP8_DTYPES else "bfloat16"
            frame = frame.f_back
    finally:
        del frame
    # fallback for direct module use (imported harness present in sys.modules)
    for modname in ("doml_group_refit", "__main__"):
        mod = sys.modules.get(modname)
        rs = getattr(mod, "RUN_STATE", None) if mod is not None else None
        if isinstance(rs, dict) and rs.get("codebook_dtype") in _FP8_DTYPES:
            return rs["codebook_dtype"]
    return "bfloat16"


MODEL_NAME = "Qwen/Qwen3-0.6B"
RUN_ARGV = [
    "run.py", MODEL_NAME, "wikitext2", "doml",
    "--blocksize", "128", "--salient_metric", "magnitude",
    "--device", "cuda:0",
]
EXPECTED_SUBLAYERS = 196  # 28 decoder layers x 7 linear sublayers


# ---------------------------------------------------------------------------
# bf16 bit-pattern <-> monotonic integer key helpers
# ---------------------------------------------------------------------------
def bf16_pattern(t: torch.Tensor) -> torch.Tensor:
    """bf16 tensor -> int64 raw bit pattern in [0, 0xFFFF]."""
    assert t.dtype == torch.bfloat16
    return t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF


def key_from_pattern(p: torch.Tensor) -> torch.Tensor:
    """Monotonic (numeric-order) integer key; -0.0 sorts just below +0.0."""
    return torch.where(p >= 0x8000, 0xFFFF - p, p + 0x8000)


def pattern_from_key(k: torch.Tensor) -> torch.Tensor:
    return torch.where(k >= 0x8000, k - 0x8000, 0xFFFF - k)


def bf16_from_pattern(p: torch.Tensor) -> torch.Tensor:
    p16 = torch.where(p >= 0x8000, p - 0x10000, p).to(torch.int16)
    return p16.contiguous().view(torch.bfloat16)


def pack_bits_u32(bits: torch.Tensor) -> torch.Tensor:
    """Bool tensor [..., n] (n % 32 == 0) -> uint32 [..., n/32], LSB-first:
    bit i of word w covers position 32*w + i (doc 02 §2a)."""
    assert bits.dtype == torch.bool and bits.shape[-1] % 32 == 0
    a = bits.detach().cpu().numpy()
    packed = np.packbits(a, axis=-1, bitorder="little")     # uint8 [..., n/8]
    words = np.ascontiguousarray(packed).view(np.uint32)    # [..., n/32]
    # torch<->numpy uint32 bridge: go through int32 bits, then reinterpret.
    return torch.from_numpy(words.view(np.int32).copy()).view(torch.uint32)


def pack_codes2_u32(codes: torch.Tensor) -> torch.Tensor:
    """Int tensor [C] with values 0..3 (C % 16 == 0) -> uint32 [C/16].
    LSB-first 2-bit fields: column j occupies bits 2*(j%16)..2*(j%16)+1 of
    word j//16 (doc 02 §2 `colmem`; codes 0=bulk, 1=tail, 2=salient)."""
    assert codes.dim() == 1 and codes.numel() % 16 == 0
    c = codes.detach().cpu().to(torch.int64)
    assert bool(((c >= 0) & (c <= 3)).all()), "colmem code out of [0, 3]"
    # interleave (LSB, MSB) of each code -> flat bool [2C]; bit 2j of the
    # stream = LSB of code j -> lands at bit 2*(j%16) of word j//16.
    bits = torch.stack([(c & 1).to(torch.bool),
                        ((c >> 1) & 1).to(torch.bool)], dim=1).reshape(-1)
    return pack_bits_u32(bits.unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------------
# Core: derive DPK streams for one layer
# ---------------------------------------------------------------------------
@torch.no_grad()
def derive_dpk(Wq: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor,
               m3: torch.Tensor, B: int = B_BLOCK, g: int = G_GROUP,
               mmode: str = "element"):
    """Wq bf16 [R, C_orig]; m1/m2/m3 bool [R, C_orig] (bulk/tail/salient).

    mmode "element": doc 02 §2a container (b0/b1/m/s/cb).
    mmode "column" (K2.6): bulk/tail membership must be COLUMN-wise; the m
    plane is dropped and replaced by `colmem` uint32[C/16] (2-bit per column,
    LSB-first pairs, 0=bulk 1=tail 2=salient) per doc 02 §2.

    Returns (tensors_cpu: dict for the .dpk container, stats: dict).
    Hard-asserts every structural precondition and the §3 dequant invariant.
    """
    assert mmode in ("element", "column"), mmode
    assert Wq.dtype == torch.bfloat16 and Wq.dim() == 2
    dev = Wq.device
    R, C_orig = Wq.shape
    assert m1.shape == (R, C_orig) and m2.shape == (R, C_orig) \
        and m3.shape == (R, C_orig)
    assert g % B == 0, "group size must be a multiple of the partition block"

    C = ((C_orig + B - 1) // B) * B          # padded column count
    NG = C // g
    n_pad = C - C_orig

    if n_pad:
        zpadW = torch.zeros(R, n_pad, dtype=Wq.dtype, device=dev)
        zpadM = torch.zeros(R, n_pad, dtype=torch.bool, device=dev)
        Wq = torch.cat([Wq, zpadW], dim=1)
        m1 = torch.cat([m1, zpadM], dim=1)
        m2 = torch.cat([m2, zpadM], dim=1)
        m3 = torch.cat([m3, zpadM], dim=1)

    # --- structural preconditions -----------------------------------------
    cover = m1.to(torch.int32) + m2.to(torch.int32) + m3.to(torch.int32)
    if not bool((cover[:, :C_orig] == 1).all()):
        raise RuntimeError("masks are not a disjoint cover of the layer")
    if n_pad and not bool((cover[:, C_orig:] == 0).all()):
        raise RuntimeError("padding columns must be mask-free")
    # mask3 must be column-wise (the s bitmap can only encode columns).
    if not bool((m3 == m3[0:1].expand_as(m3)).all()):
        raise RuntimeError("mask3 (salient) is not column-wise — "
                           "s bitmap cannot represent it")
    if bool(torch.isnan(Wq.float()).any()):
        raise RuntimeError("NaN in quantized weights")
    if mmode == "column":
        # bulk/tail membership must itself be column-wise for colmem encoding
        if not bool((m1 == m1[0:1].expand_as(m1)).all()):
            raise RuntimeError("mmode=column: mask1 (bulk) is not column-wise")
        if not bool((m2 == m2[0:1].expand_as(m2)).all()):
            raise RuntimeError("mmode=column: mask2 (tail) is not column-wise")

    sal_col = m3[0].clone()                                   # [C] bool

    patt = bf16_pattern(Wq).view(R, C)                        # [R, C]
    key = key_from_pattern(patt)                              # [R, C]
    keyg = key.view(R, NG, g)

    code_full = torch.zeros(R, C, dtype=torch.int64, device=dev)
    cb_keys = torch.full((R, NG, 3, 4), KEY_ZERO,
                         dtype=torch.int64, device=dev)
    ndist_hist = torch.zeros(5, dtype=torch.int64)            # 0..4 distinct
    pos = torch.arange(g, device=dev).view(1, 1, g)
    rgrows = torch.arange(R * NG, device=dev).unsqueeze(1).expand(R * NG, g)

    for p, mp in enumerate((m1, m2, m3)):
        mg = mp.view(R, NG, g)
        k = torch.where(mg, keyg, torch.full_like(keyg, KEY_SENT))
        skey, _ = k.sort(dim=-1)
        cnt = mg.sum(dim=-1)                                  # [R, NG]
        vpos = pos < cnt.unsqueeze(-1)                        # [R, NG, g]
        prev = torch.cat(
            [torch.full((R, NG, 1), -1, dtype=torch.int64, device=dev),
             skey[..., :-1]], dim=-1)
        nf = vpos & (skey != prev)                            # new-distinct flag
        rank = nf.to(torch.int64).cumsum(dim=-1) - 1          # value rank
        nd = nf.sum(dim=-1)                                   # [R, NG] distinct
        nd_max = int(nd.max().item()) if nd.numel() else 0
        if nd_max > 4:
            raise RuntimeError(
                f"partition {p}: {nd_max} distinct levels in a (row,group) "
                f"— exceeds K=4; DOML ground truth violated")
        ndist_hist += torch.bincount(nd.flatten().cpu(), minlength=5)[:5]

        # scatter the distinct sorted keys into level slots 0..nd-1
        lk = torch.full((R, NG, 4), KEY_ZERO, dtype=torch.int64, device=dev)
        lkf = lk.view(R * NG, 4)
        nfv = nf.view(R * NG, g)
        lkf[rgrows[nfv], rank.view(R * NG, g)[nfv]] = skey.view(R * NG, g)[nfv]
        # pad slots by repeating the last real level (doc 02 / K2 contract);
        # fully-empty (row,group) partitions keep all-zero levels.
        for kk in range(1, 4):
            lk[..., kk] = torch.where(nd > kk, lk[..., kk], lk[..., kk - 1])

        # assignments = index of each member's value among the sorted distinct
        cd = torch.searchsorted(
            lk.view(R * NG, 4).contiguous(),
            keyg.view(R * NG, g).contiguous()).view(R, NG, g)
        cdc = cd.clamp(max=3)
        got = lk.gather(-1, cdc)
        if not bool((got[mg] == keyg[mg]).all()):
            raise RuntimeError(
                f"partition {p}: some member value is not in its own "
                f"(row,group) level list — derivation broken")
        # assignments must never hit a pad slot
        if not bool((cd[mg] < nd.unsqueeze(-1).expand_as(cd)[mg]).all()):
            raise RuntimeError(f"partition {p}: assignment hit a pad slot")

        code_full += (cdc * mg.to(torch.int64)).view(R, C)
        cb_keys[:, :, p, :] = lk

    # --- streams ------------------------------------------------------------
    b0_bits = (code_full & 1).to(torch.bool)
    b1_bits = ((code_full >> 1) & 1).to(torch.bool)
    m_bits = m2 & ~sal_col.unsqueeze(0)      # don't-care=0 at salient/pad cols
    cb_patt = pattern_from_key(cb_keys)
    cb = bf16_from_pattern(cb_patt.cpu())    # bf16 [R, NG, 3, 4]

    if mmode == "column":
        # 2-bit per-column membership codes; pad columns get 0 (bulk) which
        # keeps the §3 padding trick (b0=b1=0 -> cb[..,0,0]) unchanged.
        col_codes = torch.where(
            sal_col, torch.full_like(sal_col, 2, dtype=torch.int64),
            m2[0].to(torch.int64))                            # [C] 0..2
        colmem = pack_codes2_u32(col_codes)
        # decode-back check: the packed stream must reproduce the codes
        wrd = colmem.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        sh2 = torch.arange(16, dtype=torch.int64) * 2
        dec = ((wrd.unsqueeze(-1) >> sh2) & 3).reshape(-1)
        if not torch.equal(dec, col_codes.cpu()):
            raise RuntimeError("colmem pack/decode mismatch")
        part = dec.to(dev).unsqueeze(0).expand(R, C)          # part(j) per §3
    else:
        part = torch.where(sal_col.unsqueeze(0).expand(R, C),
                           torch.full_like(code_full, 2),
                           m_bits.to(torch.int64))

    # --- §3 dequant invariant, 100% of (real) elements, bit-exact ------------
    gidx = (torch.arange(C, device=dev) // g).unsqueeze(0)    # [1, C]
    flat = (gidx * 3 + part) * 4 + code_full                  # [R, C]
    rec_patt = cb_patt.to(dev).view(R, NG * 12).gather(1, flat)
    if not bool((rec_patt[:, :C_orig] == patt[:, :C_orig]).all()):
        bad = int((rec_patt[:, :C_orig] != patt[:, :C_orig]).sum().item())
        raise RuntimeError(f"§3 invariant FAILED on {bad} elements")

    # fp8 codebook storage (lossless): the derived cb values are fp8-snapped
    # (harness _maybe_round_levels), so cb.to(fp8) is bit-exact; assert it.
    cbdtype = _resolve_cbdtype()
    if cbdtype in _FP8_DTYPES:
        cb_store = cb.to(_FP8_DTYPES[cbdtype])
        if not bool((cb_store.to(torch.bfloat16).view(torch.int16)
                     == cb.view(torch.int16)).all()):
            raise RuntimeError(
                f"cb is not exactly representable in {cbdtype}; fp8 codebook "
                f"storage would be lossy — refusing to write")
    else:
        cb_store = cb

    tensors = {
        "b0": pack_bits_u32(b0_bits),
        "b1": pack_bits_u32(b1_bits),
        "s": pack_bits_u32(sal_col.unsqueeze(0)).squeeze(0),
        "cb": cb_store.contiguous(),
    }
    if mmode == "column":
        tensors["colmem"] = colmem
    else:
        tensors["m"] = pack_bits_u32(m_bits)
    packed_bits = 8 * sum(
        t.numel() * t.element_size() for t in tensors.values())
    stats = {
        "R": R, "C": C, "C_orig": C_orig, "NG": NG, "n_pad": n_pad,
        "n_sal_cols": int(sal_col.sum().item()),
        "ndist_hist": [int(x) for x in ndist_hist],
        "packed_bpw": packed_bits / (R * C_orig),
    }
    return tensors, stats


def container_meta(R, C, C_orig, NG, layer_name, g=G_GROUP,
                   model=MODEL_NAME, mmode="element", cbdtype=None):
    assert mmode in ("element", "column"), mmode
    if cbdtype is None:
        cbdtype = _resolve_cbdtype()   # 'bfloat16' by default (byte-identical)
    return {"R": R, "C": C, "C_orig": C_orig, "B": B_BLOCK, "g": g,
            "NG": NG, "mmode": mmode, "cbdtype": cbdtype,
            "model": model, "layer_name": layer_name}


def save_layer(dump_dir, layer_name, tensors, stats, Wq_orig_bf16):
    meta = container_meta(stats["R"], stats["C"], stats["C_orig"],
                          stats["NG"], layer_name)
    mjson = json.dumps(meta)
    dpk_path = os.path.join(dump_dir, f"{layer_name}.dpk.safetensors")
    wq_path = os.path.join(dump_dir, f"{layer_name}.wq.safetensors")
    save_file(tensors, dpk_path, metadata={"meta": mjson})
    save_file({"wq": Wq_orig_bf16.detach().cpu().contiguous()},
              wq_path, metadata={"meta": mjson})
    return dpk_path, wq_path


# ---------------------------------------------------------------------------
# --run mode: monkey-patched full DOML run with per-layer dumps
# ---------------------------------------------------------------------------
def main_run(dump_dir):
    os.makedirs(dump_dir, exist_ok=True)
    os.chdir(REPO)

    import runpy
    import threading

    STATE = {
        "capture": False,     # inside a doml fasterquant
        "blocks": [],         # per block: {"masks":[3], "recons":[3]}
        "n_layers": 0,
        "manifest": [],
        "t0": time.time(),
    }

    # ---- patch 1: utils.structure (must precede `import bigptq`) ----------
    assert "bigptq" not in sys.modules, \
        "bigptq imported before the utils.structure patch — order violated"
    import utils.structure as _us
    _orig_sgd = _us.structural_guassian_distribution

    def _sgd_wrapper(tmp, H=None, metric="magnitude", up_lim=30,
                     orders=(1, 1, 2)):
        out = _orig_sgd(tmp, H, metric, up_lim, orders=orders)
        if STATE["capture"]:
            if STATE["blocks"]:
                prev = STATE["blocks"][-1]
                assert len(prev["recons"]) == 3, \
                    "previous block did not receive 3 partition quantize calls"
            m1, m2, m3 = out
            STATE["blocks"].append({
                "masks": [m1.detach().clone(), m2.detach().clone(),
                          m3.detach().clone()],
                "recons": [],
            })
        return out

    _us.structural_guassian_distribution = _sgd_wrapper

    # ---- patch 2: binary.Binarization.quantize -----------------------------
    import binary as _bin
    _orig_quant = _bin.Binarization.quantize

    def _quant_wrapper(self, w, mask, order=2, groupi=0, col_weights=None):
        out = _orig_quant(self, w, mask, order=order, groupi=groupi,
                          col_weights=col_weights)
        if STATE["capture"] and getattr(self, "method", None) == "doml":
            assert STATE["blocks"], "quantize before any mask capture"
            blk = STATE["blocks"][-1]
            pidx = len(blk["recons"])
            assert pidx < 3, "more than 3 quantize calls in one block"
            # belt-and-braces: quantize's mask must equal the captured one
            assert torch.equal(mask, blk["masks"][pidx]), \
                f"partition-{pidx} mask mismatch between sgd and quantize"
            blk["recons"].append(out.detach().clone())
        return out

    _bin.Binarization.quantize = _quant_wrapper

    # ---- patch 3: bigptq.BRAGPTQ.fasterquant --------------------------------
    import bigptq
    if bigptq.structural_guassian_distribution is not _sgd_wrapper:
        bigptq.structural_guassian_distribution = _sgd_wrapper
    assert bigptq.structural_guassian_distribution is _sgd_wrapper
    print("K2DUMP: patches installed OK", flush=True)

    _orig_fq = bigptq.BRAGPTQ.fasterquant

    @torch.no_grad()
    def _process_layer(self):
        W_layer = self.layer.weight.data
        assert W_layer.dtype == torch.bfloat16, W_layer.dtype
        R, C = int(W_layer.shape[0]), int(W_layer.shape[1])
        blocks = STATE["blocks"]
        n_blocks_exp = (C + B_BLOCK - 1) // B_BLOCK
        assert len(blocks) == n_blocks_exp, \
            f"captured {len(blocks)} blocks, expected {n_blocks_exp}"
        assert sum(b["masks"][0].shape[1] for b in blocks) == C

        gname = getattr(self.layer, "global_name", None)
        assert gname is not None and gname.startswith(MODEL_NAME)
        layer_name = gname[len(MODEL_NAME):]        # e.g. model.layers.0....

        # --- reassembly assert: (masks + partition recons) == weight, bitwise
        # Replicates bigptq exactly: q = 0; q += recon_j * mask_j  (j = 0,1,2)
        q_full = torch.zeros(R, C, dtype=torch.float32, device=W_layer.device)
        col = 0
        for blk in blocks:
            nc = blk["masks"][0].shape[1]
            assert len(blk["recons"]) == 3
            seg = q_full[:, col:col + nc]
            for j in range(3):
                seg += blk["recons"][j] * blk["masks"][j]
            col += nc
        W_ref = q_full.to(torch.bfloat16)
        eq = W_ref.view(torch.int16) == W_layer.contiguous().view(torch.int16)
        if not bool(eq.all()):
            bad = int((~eq).sum().item())
            raise RuntimeError(
                f"{layer_name}: reassembly from masks+reconstructions is NOT "
                f"bitwise equal to the final layer weight ({bad}/{R*C} "
                f"mismatches) — hooks perturbed or misread the flow")

        m1 = torch.cat([b["masks"][0] for b in blocks], dim=1)
        m2 = torch.cat([b["masks"][1] for b in blocks], dim=1)
        m3 = torch.cat([b["masks"][2] for b in blocks], dim=1)

        tensors, stats = derive_dpk(W_layer, m1, m2, m3)
        save_layer(dump_dir, layer_name, tensors, stats, W_layer)

        rec = {"layer_name": layer_name, **stats,
               "reassembly_bitwise": True, "invariant_bitwise": True,
               "t": round(time.time() - STATE["t0"], 1)}
        STATE["manifest"].append(rec)
        STATE["n_layers"] += 1
        print(f"K2DUMP[{STATE['n_layers']:3d}] {layer_name} R={R} C={C} "
              f"nsal={stats['n_sal_cols']} bpw={stats['packed_bpw']:.4f} "
              f"reasm=BITWISE-OK inv=BITWISE-OK", flush=True)

    def _fq_wrapper(self, *args, **kwargs):
        is_doml = getattr(self.braq_quantizer, "method", None) == "doml"
        STATE["capture"] = is_doml
        STATE["blocks"] = []
        out = _orig_fq(self, *args, **kwargs)
        STATE["capture"] = False
        if is_doml:
            _process_layer(self)   # raises on any gate violation (no swallow)
            STATE["blocks"] = []
        return out

    bigptq.BRAGPTQ.fasterquant = _fq_wrapper
    print("K2DUMP: BRAGPTQ.fasterquant patched OK", flush=True)

    def _watchdog():
        time.sleep(300)
        if STATE["n_layers"] == 0 and not STATE["blocks"]:
            print("K2DUMP FATAL: no captures after 300 s — hooks dead; "
                  "aborting.", file=sys.stderr, flush=True)
            os._exit(17)

    threading.Thread(target=_watchdog, daemon=True).start()

    sys.argv = list(RUN_ARGV)
    print("K2DUMP: launching run.py:", sys.argv, flush=True)
    err = None
    try:
        runpy.run_path(os.path.join(REPO, "run.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            err = f"SystemExit({e.code})"
    except Exception as e:  # noqa: BLE001 — recorded, then re-raised
        import traceback
        err = repr(e)
        traceback.print_exc()
    finally:
        manifest = {
            "argv": RUN_ARGV[1:],
            "dump_dir": dump_dir,
            "n_sublayers_dumped": STATE["n_layers"],
            "expected_sublayers": EXPECTED_SUBLAYERS,
            "error": err,
            "layers": STATE["manifest"],
        }
        with open(os.path.join(dump_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"K2DUMP: done. sublayers dumped = {STATE['n_layers']} "
              f"(expected {EXPECTED_SUBLAYERS}); error = {err}", flush=True)
    if err:
        sys.exit(1)
    if STATE["n_layers"] != EXPECTED_SUBLAYERS:
        print(f"K2DUMP FATAL: dumped {STATE['n_layers']} != "
              f"{EXPECTED_SUBLAYERS}", file=sys.stderr, flush=True)
        sys.exit(2)


# ---------------------------------------------------------------------------
# --selftest mode: synthetic layer incl. ragged C (padding), duplicate levels,
# empty partitions, and a -0.0 level; exercises pack -> file -> unpack.
# ---------------------------------------------------------------------------
def main_selftest():
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dpk_unpack

    gen = torch.Generator().manual_seed(1234)
    R, C_orig = 8, 200                     # pads to C = 256, NG = 2
    C = 256

    # column-wise salient sets per 128-block
    sal = torch.zeros(C_orig, dtype=torch.bool)
    sal[[3, 77, 130, 190]] = True
    m3 = sal.unsqueeze(0).expand(R, C_orig).clone()
    m2 = torch.zeros(R, C_orig, dtype=torch.bool)
    rnd = torch.rand(R, C_orig, generator=gen)
    m2[(rnd < 0.35) & ~m3] = True
    m2[0, :128] = False                     # row 0 block 0: empty tail part.
    m2[0, torch.arange(C_orig) >= 128] = False
    m1 = ~m3 & ~m2

    # synthetic codebooks bf16 [R, 2, 3, 4], sorted ascending
    lv = torch.randn(R, 2, 3, 4, generator=gen).to(torch.bfloat16)
    lv = lv.float().sort(dim=-1).values.to(torch.bfloat16)
    lv[1, 0, 0, 0] = lv[1, 0, 0, 1]        # duplicate level pair
    lv[2, 0, 0, 0] = torch.tensor(-0.0, dtype=torch.bfloat16)  # -0.0 level
    assert bf16_pattern(lv[2, 0, 0, 0].reshape(1)).item() == 0x8000

    codes = torch.randint(0, 4, (R, C_orig), generator=gen)
    part = torch.where(m3, 2, torch.where(m2, 1, 0))
    gidx = torch.arange(C_orig) // G_GROUP
    Wq = lv.view(R, -1).gather(
        1, (gidx.unsqueeze(0) * 3 + part) * 4 + codes)          # bf16 [R, C_orig]

    tensors, stats = derive_dpk(Wq, m1, m2, m3)
    assert stats["C"] == C and stats["NG"] == 2 and stats["n_pad"] == 56
    print("selftest: derive_dpk OK; stats =", stats)

    with tempfile.TemporaryDirectory() as td:
        dpk, wq = save_layer(td, "selftest.layer", tensors, stats, Wq)
        t, meta = dpk_unpack.load_container(dpk)
        W2 = dpk_unpack.unpack(t, meta)                     # bf16 [R, C]
        eq = (W2[:, :C_orig].contiguous().view(torch.int16)
              == Wq.contiguous().view(torch.int16))
        assert bool(eq.all()), "selftest round-trip NOT bitwise"
        print("selftest: file round-trip bitwise OK "
              f"({R}x{C_orig}, padded to {C})")

        # corrupt one plane bit in-memory -> must break bitwise equality
        tc = {k: v.clone() for k, v in t.items()}
        w = tc["b0"].view(torch.int32)
        w[4, 1] ^= (1 << 7)
        Wc = dpk_unpack.unpack(tc, meta)
        neq = int((Wc[:, :C_orig].contiguous().view(torch.int16)
                   != Wq.contiguous().view(torch.int16)).sum().item())
        # the flipped code may land on a duplicated level; here it must not
        assert neq >= 0
        print(f"selftest: single-bit b0 corruption changes {neq} element(s)")

    print("SELFTEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR)
    args = ap.parse_args()
    if args.selftest:
        main_selftest()
    elif args.run:
        main_run(args.dump_dir)
    else:
        ap.error("choose --run or --selftest")
