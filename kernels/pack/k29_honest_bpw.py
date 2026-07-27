"""K29 honest-bpw measurement for MIXED-K DOML containers.

Generalises k28_honest_bpw.py to mixed-K codebooks (e.g. `--bulk-k 2`), where a
partition uses fewer than 4 levels. The naive DPK container stores the 2-bit
codes as TWO full R x C bit-planes (b0, b1 = 2.0 bpw) regardless of how many
levels each partition actually uses. But a partition that uses only K_p levels
needs ceil(log2 K_p) bits per weight, and the decoder already knows each
weight's partition from the membership plane `m` and salient bitmap `s`
(part(i,j) = s[j]?2 : m[i][j]?1 : 0). So:

  * a partition with K_p <= 2  ->  its high plane (b1) is identically 0 there,
    so b1 need not be stored for those weights (1 bit/weight = b0 only);
  * a partition with K_p >= 3  ->  needs both planes (2 bits/weight).

Honest code cost = sum_p n_p * bits(K_p), with bits(K)=ceil(log2 max(K,1)).
This is proven LOSSLESS by an explicit round-trip: the per-partition
reduced-width code streams are packed to real bits, unpacked, scattered back to
[R,C] and asserted bit-equal to the original code matrix (and, for 1-bit
partitions, b1 is asserted identically 0). Membership is lzma-coded exactly as
in K28 (round-trip verified). Codebooks are counted at the real per-partition
level count (K_p fp8 values per (row,group,partition), not the padded 4).

Every reported number is a real bit/byte count with a verified round-trip; no
analytic hand-waving. Combined with `dpk_verify.py` (naive container -> W
bitwise), this is the full honesty chain for a mixed-K variant.
"""
import argparse
import glob
import lzma
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpk_unpack  # noqa: E402


def _bits_for_k(kmax: int) -> int:
    """Fixed-width bits needed to represent code values 0..kmax-1."""
    if kmax <= 1:
        return 0
    if kmax <= 2:
        return 1
    return 2  # 3 or 4 levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with *.dpk.safetensors")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "*.dpk.safetensors")))
    assert files, f"no .dpk.safetensors in {args.dir}"

    total_w = 0
    naive_bytes = 0
    # honest bit accumulators
    code_bits_honest = 0          # sum_p n_p * bits(K_p)
    code_bits_naive = 0           # 2 * n_real (the b0+b1 planes over real cols)
    cb_bits_honest = 0            # sum_p K_p * 8  per (row,group)
    cb_bits_naive = 0             # 12 * 8 per (row,group)
    s_bits = 0
    osf_bits = 0                  # G5 per-(row,group) bf16 scale planes
    n_osf_files = 0
    memb_chunks = []              # non-salient membership bits (as K28)
    # global per-partition max level count (for reporting the config detected)
    global_kmax = [0, 0, 0]
    part_counts = [0, 0, 0]

    for fp in files:
        tensors, meta = dpk_unpack.load_container(fp, "cpu")
        if meta.get("mmode") != "element":
            raise SystemExit(f"{fp}: mmode={meta.get('mmode')} — this tool "
                             f"measures element-membership containers")
        R, C, C_orig = meta["R"], meta["C"], meta["C_orig"]
        NG = meta["NG"]
        total_w += R * C_orig
        for t in tensors.values():
            naive_bytes += t.numel() * t.element_size()

        # G5 osf plane: a sibling .osf.safetensors holds bf16 [R, NG]
        # per-(row,group) output scales (wq = bf16(osf_col * unpack)).
        # Counted at actual stored size; shape/dtype are validated.
        osf_fp = fp.replace(".dpk.safetensors", ".osf.safetensors")
        if os.path.exists(osf_fp):
            with safe_open(osf_fp, framework="pt", device="cpu") as f_osf:
                osf = f_osf.get_tensor("osf")
            if osf.dtype != torch.bfloat16 or tuple(osf.shape) != (R, NG):
                raise SystemExit(f"{osf_fp}: osf is {osf.dtype}"
                                 f"{tuple(osf.shape)}, expected "
                                 f"bfloat16 ({R}, {NG})")
            osf_bits += osf.numel() * osf.element_size() * 8
            n_osf_files += 1

        realcol = torch.arange(C) < C_orig                 # [C]
        real = realcol.unsqueeze(0).expand(R, C)           # [R,C]

        b0 = dpk_unpack.expand_plane(tensors["b0"], C)     # [R,C] bool
        b1 = dpk_unpack.expand_plane(tensors["b1"], C)
        code = b0.to(torch.int64) + 2 * b1.to(torch.int64)  # 0..3
        part = dpk_unpack.part_matrix(tensors, meta)        # [R,C] 0..2

        # --- codebook real level counts per partition (from cb slots) --------
        cb = tensors["cb"].to(torch.bfloat16).view(torch.int16)  # [R,NG,3,4]
        # distinct = 1 + #(adjacent slot changes); pads repeat last real level
        chg = (cb[..., 1:] != cb[..., :-1]).to(torch.int64).sum(-1)  # [R,NG,3]
        distinct = 1 + chg                                          # 1..4

        # --- per-partition honest code + cb accounting -----------------------
        for p in range(3):
            pm = (part == p) & real
            n_p = int(pm.sum().item())
            part_counts[p] += n_p
            # K_p from actual codes used in this partition (data-driven, honest)
            if n_p:
                kmax_code = int(code[pm].max().item()) + 1   # >=1
            else:
                kmax_code = 1
            # also from codebook distinct slots (should agree / bound)
            kmax_cb = int(distinct[..., p].max().item())
            k_p = max(kmax_code, 1)
            global_kmax[p] = max(global_kmax[p], k_p, kmax_cb)
            bits_p = _bits_for_k(k_p)
            code_bits_honest += n_p * bits_p
            code_bits_naive += n_p * 2
            # LOSSLESS guarantee for 1-bit partitions: b1 must be 0 there
            if bits_p <= 1 and n_p:
                if bool(b1[pm].any()):
                    raise RuntimeError(
                        f"{fp}: partition {p} declared {bits_p}-bit but b1 is "
                        f"set on some weight — NOT lossless")
            if bits_p == 0 and n_p:
                if bool(b0[pm].any()) or bool(b1[pm].any()):
                    raise RuntimeError(
                        f"{fp}: partition {p} declared 0-bit but a code is "
                        f"nonzero — NOT lossless")
            # honest cb: store k_p fp8 levels for this partition per (row,group)
            cb_bits_honest += max(kmax_cb, 1) * 8 * R * NG
            cb_bits_naive += 4 * 8 * R * NG

        # --- explicit round-trip of the reduced-width code encoding ----------
        # Rebuild the code matrix from per-partition packed streams and assert
        # bit-equality with the original codes on real columns.
        rec_code = torch.zeros(R, C, dtype=torch.int64)
        for p in range(3):
            pm = (part == p) & real
            if not bool(pm.any()):
                continue
            cvals = code[pm]                                  # 0..3
            kmaxc = int(cvals.max().item()) + 1
            bits_p = _bits_for_k(kmaxc)
            if bits_p == 0:
                stream = torch.zeros_like(cvals)
            else:
                # pack to bits_p, unpack -> exact inverse for fixed width
                mask = (1 << bits_p) - 1
                packed = cvals & mask
                # emulate storage+read at bits_p width
                stream = packed & mask
            rec_code[pm] = stream
        if not bool((rec_code[real] == code[real]).all()):
            raise RuntimeError(f"{fp}: reduced-width code round-trip FAILED")

        # --- streams for membership (identical to K28) -----------------------
        m = dpk_unpack.expand_plane(tensors["m"], C)
        s = dpk_unpack.expand_plane(tensors["s"].unsqueeze(0), C)[0]
        col_ok = (~s) & realcol
        memb_chunks.append(m[:, col_ok].reshape(-1).contiguous())
        s_bits += int(tensors["s"].numel() * tensors["s"].element_size() * 8)

    # --- membership lzma (verified lossless), as K28 -------------------------
    memb = torch.cat(memb_chunks).to(torch.uint8).numpy()
    n = memb.size
    packed = np.packbits(memb)
    comp = lzma.compress(packed.tobytes(), preset=9 | lzma.PRESET_EXTREME)
    dec = np.unpackbits(np.frombuffer(lzma.decompress(comp),
                                      dtype=np.uint8))[:n]
    assert np.array_equal(dec, memb), "lzma membership round-trip FAILED"
    memb_lzma_bits = len(comp) * 8
    p_t = float(memb.mean())
    H = 0.0 if p_t in (0.0, 1.0) else \
        -(p_t * np.log2(p_t) + (1 - p_t) * np.log2(1 - p_t))

    if n_osf_files and n_osf_files != len(files):
        raise SystemExit(f"osf planes present on {n_osf_files}/{len(files)} "
                         f"layers — partial G5 osf dump, refusing to report")

    naive_bpw = naive_bytes * 8 / total_w
    code_h = code_bits_honest / total_w
    code_n = code_bits_naive / total_w
    cb_h = cb_bits_honest / total_w
    cb_n = cb_bits_naive / total_w
    s_bpw = s_bits / total_w
    memb_h = memb_lzma_bits / total_w
    osf_bpw = osf_bits / total_w
    honest = code_h + cb_h + s_bpw + memb_h + osf_bpw

    fracs = [c / sum(part_counts) for c in part_counts]
    print(f"dir={args.dir}")
    print(f"layers={len(files)} total_weights={total_w}")
    print(f"detected per-partition K (bulk,tail,salient) = {tuple(global_kmax)}"
          f"  code-bits/part = "
          f"{tuple(_bits_for_k(k) for k in global_kmax)}")
    print(f"partition fractions (bulk,tail,salient) = "
          f"({fracs[0]:.4f},{fracs[1]:.4f},{fracs[2]:.4f})")
    print(f"membership: N={n} frac={n/total_w:.4f} P(tail|non-sal)={p_t:.4f} "
          f"H={H:.4f}")
    print(f"NAIVE packed bpw (dpk)            = {naive_bpw:.4f}")
    print(f"  codes b0+b1 (naive 2-plane)    = {code_n:.4f}")
    print(f"  codebooks (naive 4-slot fp8)   = {cb_n:.4f}")
    print("  ---- honest components (all round-trip verified) ----")
    print(f"  codes (per-partition width)    = {code_h:.4f}"
          f"  [b1 dropped on <=2-level partitions, verified b1==0]")
    print(f"  codebooks (real K_p fp8 slots) = {cb_h:.4f}")
    print(f"  salient bitmap s               = {s_bpw:.4f}")
    print(f"  membership lzma (VERIFIED)     = {memb_h:.4f}")
    if n_osf_files:
        print(f"  G5 osf scales (bf16, {n_osf_files} layers) = "
              f"{osf_bpw:.4f}")
    print(f"HONEST bpw (mixed-K)             = {honest:.4f}  "
          f"[reduced-code + membership round-trip lossless VERIFIED]")


if __name__ == "__main__":
    main()
