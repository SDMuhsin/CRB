"""SDOML honest-bpw deliverable 2 — independent re-verify + REAL bit accounting.

Loads a directory of `*.sdpk.safetensors` SDOML base containers (written by
sdoml_dump.py) and, for EACH layer INDEPENDENTLY:

  1. RE-VERIFIES the decode round-trip (fresh argmin decode here, independent
     of the dumper): reload (wq, mask, cb); recompute codes by nearest-centroid
     argmin per 128-col block; reconstruct; assert bit-exact == wq; assert
     wq[~mask] == 0 exactly.

  2. ROUND-TRIPS every ENCODED stream and counts REAL bits:
     - mask bitmap : np.packbits(mask) -> unpack -> assert == mask.
                     Raw cost = R*C bits (1 bit/weight, paper-faithful).
                     lzma(packed) -> decompress -> assert == packed; lzma cost.
                     Combinatorial floors (informational lower bounds, NOT
                     realised by a coder here): per-row Sum log2 C(C, k_row)
                     and the tighter per-block Sum log2 C(w_b, k_rowblock).
     - codes       : gather code (0..K-1) at KEPT positions row-major -> the
                     real code stream of n_kept * ceil(log2 K) bits; pack at
                     fixed ceil(log2 K)-bit width, unpack, scatter back via
                     mask, assert == original code plane on kept positions.
     - codebook    : bf16, K*16 bits per (row, block). 16-bit storage round-
                     trip asserted bit-exact. "distinct" variant counts only
                     DISTINCT bf16 levels per (row, block).

Honest bpw = (mask_bits + code_bits + cb_bits) / total_weights.
HEADLINE (paper-faithful container) = {raw-bitmap, bf16 padded-K codebook}.
Also reported: {lzma-bitmap, bf16-cb} and {raw-bitmap, distinct-cb} variants,
plus the paper's CLAIMED bpw  K*16/1024 + 1 + (1-s)*log2(K)  (N_rep=1024).

Usage:
    python kernels/pack/sdoml_honest_bpw.py --dir <dump_dir>
    python kernels/pack/sdoml_honest_bpw.py --selftest
"""

import argparse
import glob
import json
import lzma
import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save as st_save, load as st_load

REPO = "/workspace/BiLLM2"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# container load (self-contained; mirrors sdoml_dump.load_layer)
# ---------------------------------------------------------------------------
def load_layer(path, device="cpu"):
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        exp = {"wq", "mask_packed", "cb"}
        if keys != exp:
            raise ValueError(f"{path}: keys {sorted(keys)} != {sorted(exp)}")
        md = f.metadata()
        if md is None or "meta" not in md:
            raise ValueError(f"{path}: missing JSON meta blob")
        meta = json.loads(md["meta"])
        wq = f.get_tensor("wq").to(device)
        mask_packed = f.get_tensor("mask_packed")
        cb = f.get_tensor("cb").to(device)
    R, C = meta["R"], meta["C"]
    bits = np.unpackbits(mask_packed.numpy(), count=R * C)
    mask = torch.from_numpy(bits.astype(bool).reshape(R, C)).to(device)
    return wq, mask, cb, meta


# ---------------------------------------------------------------------------
# independent decode: argmin codes per block -> reconstruct
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_codes(wq, mask, cb, block_widths):
    """Return code plane [R, C] int64 (argmin per block over K), and the
    bit-exact reconstruction [R, C] bf16.  Hard-asserts wq[~mask]==0 and
    recon==wq bitwise."""
    R, C = wq.shape
    K = cb.shape[2]
    code_plane = torch.zeros(R, C, dtype=torch.int64, device=wq.device)
    recon = torch.zeros(R, C, dtype=torch.bfloat16, device=wq.device)
    off = 0
    for b, w_b in enumerate(block_widths):
        wq_blk = wq[:, off:off + w_b]
        m_blk = mask[:, off:off + w_b]
        cb_bf16 = cb[:, b, :].to(torch.bfloat16)
        diff = (wq_blk.float().unsqueeze(-1)
                - cb_bf16.float().unsqueeze(1))
        code = (diff * diff).argmin(dim=-1)              # [R, w_b]
        rec_kept = torch.gather(cb_bf16, 1, code)
        rec_blk = torch.where(m_blk, rec_kept, torch.zeros_like(rec_kept))
        pruned = wq_blk[~m_blk]
        if pruned.numel() and not bool((pruned.view(torch.int16) == 0).all()):
            raise RuntimeError(f"block {b}: pruned positions not all 0")
        eq = rec_blk.view(torch.int16) == wq_blk.contiguous().view(torch.int16)
        if not bool(eq.all()):
            raise RuntimeError(
                f"block {b}: independent decode NOT bitwise == wq "
                f"({int((~eq).sum())} mismatches)")
        code_plane[:, off:off + w_b] = code
        recon[:, off:off + w_b] = rec_blk
        off += w_b
    return code_plane, recon


# ---------------------------------------------------------------------------
# fixed-width code stream pack/unpack (LSB-first bit planes)
# ---------------------------------------------------------------------------
def pack_codes(stream_np, bits_per_code):
    n = int(stream_np.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.uint8), 0
    planes = [((stream_np >> b) & 1).astype(np.uint8)
              for b in range(bits_per_code)]
    bits = np.stack(planes, axis=1).reshape(-1)          # [n*bits]
    return np.packbits(bits), n


def unpack_codes(packed, n, bits_per_code):
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    bits = np.unpackbits(packed, count=n * bits_per_code).reshape(n,
                                                                  bits_per_code)
    val = np.zeros(n, dtype=np.int64)
    for b in range(bits_per_code):
        val += bits[:, b].astype(np.int64) << b
    return val


def log2_choose(n, k):
    """log2 C(n, k) via lgamma; n,k double tensors or scalars."""
    n = torch.as_tensor(n, dtype=torch.float64)
    k = torch.as_tensor(k, dtype=torch.float64)
    return (torch.lgamma(n + 1) - torch.lgamma(k + 1)
            - torch.lgamma(n - k + 1)) / math.log(2.0)


# ---------------------------------------------------------------------------
# per-layer measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_layer(path):
    wq, mask, cb, meta = load_layer(path)
    R, C = meta["R"], meta["C"]
    K = meta["K"]
    block_widths = meta["block_widths"]
    NG = meta["NG"]
    # codebook granularity = `groupsize` columns per codebook (one K-level
    # codebook per group). DEFAULT 128 (base SDOML: groupsize == GPTQ
    # blocksize). Honored from meta so a g256 (or any groupsize) container is
    # self-describing. The per-group iteration below is driven by block_widths
    # (already group-general), so this read is a legibility + consistency guard.
    groupsize = int(meta.get("groupsize", meta.get("blocksize", 128)))
    if block_widths and max(block_widths) > groupsize:
        raise RuntimeError(
            f"{path}: block_widths max {max(block_widths)} > groupsize "
            f"{groupsize} (container group structure inconsistent)")
    bits_per_code = max(1, int(math.ceil(math.log2(K))))

    # --- 1. independent decode round-trip --------------------------------
    code_plane, _recon = decode_codes(wq, mask, cb, block_widths)

    # --- 2a. mask bitmap round-trip --------------------------------------
    m_np = mask.cpu().numpy().reshape(-1)
    packed = np.packbits(m_np)
    unpacked = np.unpackbits(packed, count=R * C).astype(bool)
    if not np.array_equal(unpacked, m_np):
        raise RuntimeError(f"{path}: mask bitmap round-trip mismatch")
    mask_raw_bits = R * C
    comp = lzma.compress(packed.tobytes(), preset=9 | lzma.PRESET_EXTREME)
    if lzma.decompress(comp) != packed.tobytes():
        raise RuntimeError(f"{path}: lzma mask round-trip mismatch")
    mask_lzma_bits = len(comp) * 8

    # combinatorial floors (informational lower bounds)
    k_row = mask.sum(dim=1)                                # [R]
    comb_row_bits = float(log2_choose(C, k_row.double()).sum().item())
    comb_block_bits = 0.0
    off = 0
    for b, w_b in enumerate(block_widths):
        k_rb = mask[:, off:off + w_b].sum(dim=1).double()
        comb_block_bits += float(log2_choose(w_b, k_rb).sum().item())
        off += w_b

    # --- 2b. code stream round-trip --------------------------------------
    stream = code_plane[mask].cpu().numpy().astype(np.int64)   # row-major
    n_kept = int(stream.shape[0])
    assert n_kept == int(mask.sum().item())
    if not bool((stream >= 0).all() and (stream < K).all()):
        raise RuntimeError(f"{path}: code out of [0,{K})")
    packed_codes, n = pack_codes(stream, bits_per_code)
    rec_stream = unpack_codes(packed_codes, n, bits_per_code)
    if not np.array_equal(rec_stream, stream):
        raise RuntimeError(f"{path}: code stream pack/unpack mismatch")
    # scatter back via mask and check == code plane on kept positions
    recon_plane = torch.zeros(R, C, dtype=torch.int64)
    recon_plane[mask.cpu()] = torch.from_numpy(rec_stream)
    if not bool((recon_plane[mask.cpu()] == code_plane[mask].cpu()).all()):
        raise RuntimeError(f"{path}: code scatter-back mismatch")
    code_bits = n_kept * bits_per_code

    # --- 2c. codebook round-trip + distinct ------------------------------
    # cb storage width is dtype-driven: bf16 (default) = 16 bits/level; an
    # fp8-e4m3 codebook (block-tuned container) = 8 bits/level. Absent
    # cb_dtype in meta => bf16 (the sdoml_dump default), so bf16 containers
    # take the SAME code path with identical numbers as before.
    cb_dtype = meta.get("cb_dtype", "bfloat16")
    if cb_dtype == "float8_e4m3fn":
        if cb.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"{path}: meta cb_dtype=float8_e4m3fn but cb "
                               f"tensor is {cb.dtype}")
        cb_bits_per_level = 8
        # lossless fp8 storage round-trip: serialize e4m3 -> reload -> assert
        # bit-exact (int8 view), the honest 8-bit/level container.
        blob = st_save({"cb": cb.detach().cpu().contiguous()})
        cb_reload = st_load(blob)["cb"]
        if cb_reload.dtype != torch.float8_e4m3fn or not torch.equal(
                cb_reload.view(torch.int8), cb.cpu().view(torch.int8)):
            raise RuntimeError(f"{path}: fp8-e4m3 codebook round-trip mismatch")
        patt = (cb.view(torch.int8).to(torch.int64) & 0xFF)    # [R,NG,K]
    else:
        cb_bits_per_level = 16
        cb_i16 = cb.view(torch.int16).cpu().numpy()
        cb_bytes = cb_i16.tobytes()
        cb_back = np.frombuffer(cb_bytes, dtype=np.int16).reshape(cb_i16.shape)
        if not np.array_equal(cb_back, cb_i16):
            raise RuntimeError(f"{path}: codebook 16-bit round-trip mismatch")
        patt = (cb.view(torch.int16).to(torch.int64) & 0xFFFF)  # [R,NG,K]
    cb_bits_paddedK = K * cb_bits_per_level * R * NG
    sp, _ = patt.sort(dim=-1)
    distinct = 1 + (sp[..., 1:] != sp[..., :-1]).sum(dim=-1)   # [R,NG]
    cb_bits_distinct = int(distinct.sum().item()) * cb_bits_per_level

    return {
        "layer_name": meta["layer_name"], "R": R, "C": C, "K": K, "NG": NG,
        "groupsize": groupsize,
        "sparsity": meta.get("sparsity"), "cb_dtype": cb_dtype,
        "cb_bits_per_level": cb_bits_per_level,
        "n_weights": R * C, "n_kept": n_kept,
        "mask_raw_bits": mask_raw_bits, "mask_lzma_bits": mask_lzma_bits,
        "comb_row_bits": comb_row_bits, "comb_block_bits": comb_block_bits,
        "code_bits": code_bits,
        "cb_bits_paddedK": cb_bits_paddedK,
        "cb_bits_distinct": cb_bits_distinct,
    }


def measure_dir(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "*.sdpk.safetensors")))
    if not paths:
        raise SystemExit(f"no *.sdpk.safetensors in {dump_dir}")
    agg = {k: 0.0 for k in (
        "n_weights", "n_kept", "mask_raw_bits", "mask_lzma_bits",
        "comb_row_bits", "comb_block_bits", "code_bits",
        "cb_bits_paddedK", "cb_bits_distinct")}
    K_set, s_set = set(), set()
    dtype_set = set()
    gs_set = set()
    n_layers = 0
    for p in paths:
        r = measure_layer(p)
        for k in agg:
            agg[k] += r[k]
        K_set.add(r["K"])
        dtype_set.add(r["cb_dtype"])
        gs_set.add(r["groupsize"])
        if r["sparsity"] is not None:
            s_set.add(round(float(r["sparsity"]), 6))
        n_layers += 1
    groupsize = sorted(gs_set)[0] if len(gs_set) == 1 else "mixed"
    cb_dtype = sorted(dtype_set)[0] if len(dtype_set) == 1 else "mixed"
    is_fp8 = (cb_dtype == "float8_e4m3fn")
    cb_w = 8 if is_fp8 else 16                       # codebook bits per level
    cb_tag = "fp8-e4m3" if is_fp8 else "bf16"
    print(f"verified + measured {n_layers} layers in {dump_dir}")
    print("ALL per-layer independent decode round-trips: BITWISE-OK")
    print("ALL mask / code / codebook stream round-trips: LOSSLESS-OK")

    W = agg["n_weights"]
    K = sorted(K_set)[0] if len(K_set) == 1 else None
    s = sorted(s_set)[0] if len(s_set) == 1 else None
    log2K = math.log2(K) if K else float("nan")

    def bpw(x):
        return x / W

    mask_raw = bpw(agg["mask_raw_bits"])
    mask_lzma = bpw(agg["mask_lzma_bits"])
    comb_row = bpw(agg["comb_row_bits"])
    comb_block = bpw(agg["comb_block_bits"])
    code = bpw(agg["code_bits"])
    cb_pad = bpw(agg["cb_bits_paddedK"])
    cb_dist = bpw(agg["cb_bits_distinct"])

    headline = mask_raw + code + cb_pad
    var_lzma = mask_lzma + code + cb_pad
    var_dist = mask_raw + code + cb_dist
    paper = (K * 16.0) / 1024.0 + 1.0 + (1.0 - s) * log2K if (K and s
                                                              is not None) \
        else float("nan")

    kept_frac = agg["n_kept"] / W
    print("\n" + "=" * 68)
    print(f"SDOML BASE honest bit accounting  (K={K}, s={s}, "
          f"groupsize={groupsize}, layers={n_layers})")
    print("=" * 68)
    print(f"total weights          : {int(W):,}")
    print(f"kept fraction (1-s)    : {kept_frac:.4f}")
    print(f"codebook dtype         : {cb_dtype}  ({cb_w} bits/level)")
    print("-" * 68)
    print("COMPONENT bpw (bits / total weight):")
    print(f"  mask  raw bitmap     : {mask_raw:.4f}   "
          f"({int(agg['mask_raw_bits']):,} bits, 1 bit/weight)")
    print(f"  mask  lzma bitmap    : {mask_lzma:.4f}   "
          f"({int(agg['mask_lzma_bits']):,} bits)")
    print(f"  mask  comb floor/row : {comb_row:.4f}   (informational LB, "
          f"no coder)")
    print(f"  mask  comb floor/blk : {comb_block:.4f}   (informational LB, "
          f"tighter)")
    print(f"  codes ({bits_per_code_str(K)})       : {code:.4f}   "
          f"({int(agg['code_bits']):,} bits = n_kept*ceil(log2 K))")
    print(f"  codebook padded-K    : {cb_pad:.4f}   "
          f"(K*{cb_w}*R*NG; {'fp8-e4m3 ' if is_fp8 else 'real '}"
          f"per-{groupsize}-col-group codebooks)")
    print(f"  codebook distinct    : {cb_dist:.4f}   "
          f"(only distinct {cb_tag} levels per row-block)")
    print("-" * 68)
    print("HONEST bpw:")
    print(f"  HEADLINE  {{raw-bitmap, {cb_tag} padded-K cb}} : {headline:.4f}")
    print(f"  variant   {{lzma-bitmap, {cb_tag} padded-K cb}}: {var_lzma:.4f}")
    print(f"  variant   {{raw-bitmap, distinct cb}}      : {var_dist:.4f}")
    print("-" * 68)
    print(f"  PAPER CLAIMED (K*16/1024 + 1 + (1-s)*log2 K): {paper:.4f}")
    print("=" * 68)
    return {
        "n_layers": n_layers, "K": K, "s": s, "W": int(W),
        "groupsize": groupsize,
        "cb_dtype": cb_dtype, "cb_bits_per_level": cb_w,
        "headline_bpw": headline, "var_lzma_bpw": var_lzma,
        "var_distinct_bpw": var_dist, "paper_claimed_bpw": paper,
        "component_bpw": {
            "mask_raw": mask_raw, "mask_lzma": mask_lzma,
            "comb_row": comb_row, "comb_block": comb_block,
            "code": code, "cb_paddedK": cb_pad, "cb_distinct": cb_dist,
        },
    }


def bits_per_code_str(K):
    return f"{max(1, int(math.ceil(math.log2(K))))}-bit"


# ---------------------------------------------------------------------------
# selftest: synthesise a tiny dump via sdoml_dump.save_layer, then measure it
# ---------------------------------------------------------------------------
def main_selftest():
    import tempfile
    import sdoml_dump

    gen = torch.Generator().manual_seed(7)
    R, C, K = 6, 256, 4
    bw = [128, 128]
    cb = torch.randn(R, 2, K, generator=gen).float().sort(dim=-1).values
    cb = cb.to(torch.bfloat16)
    cb[0, 0, 0] = cb[0, 0, 1]                 # duplicate level
    mask = torch.rand(R, C, generator=gen) < 0.5
    mask[0, :] = False                        # fully-pruned row
    mask[R - 1, :] = True                     # fully-kept row
    codes = torch.randint(0, K, (R, C), generator=gen)
    wq = torch.zeros(R, C, dtype=torch.bfloat16)
    off = 0
    for b, w_b in enumerate(bw):
        cval = torch.gather(cb[:, b, :], 1, codes[:, off:off + w_b])
        wq[:, off:off + w_b] = torch.where(
            mask[:, off:off + w_b], cval, torch.zeros_like(cval))
        off += w_b

    with tempfile.TemporaryDirectory() as td:
        sdoml_dump.save_layer(td, "selftest.layer", wq, mask, cb, bw,
                              sparsity=0.5)
        res = measure_dir(td)
    # expected headline for a 50%-ish tiny layer: 1(mask)+1(code)+cb
    assert abs(res["component_bpw"]["mask_raw"] - 1.0) < 1e-9
    print("\nSELFTEST PASS (measured a synthetic dump end-to-end)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        main_selftest()
    elif args.dir:
        measure_dir(args.dir)
    else:
        ap.error("choose --dir <dump_dir> or --selftest")
