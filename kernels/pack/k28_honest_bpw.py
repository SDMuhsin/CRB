"""K28 honest-bpw measurement (DIRECTION B accounting).

The naive DPK container stores membership as a full R x C 1-bit plane (`m`,
1.0 bpw) even though only NON-salient weights carry a bulk/tail bit (salient
weights are identified by the `s` bitmap; padding columns are neutral). This
tool measures the HONEST packed size by replacing that plane with a
round-trip-verified lossless entropy code of the actual non-salient membership
bits, while keeping every OTHER stream (b0, b1, codebooks, salient bitmap) at
its real packed byte size — those are already proven bitwise-exact by
dpk_verify. The 2-bit codes (b0/b1) are kept at the full 2.0 bpw: they are
near-uniform (H(code|part) ~ 1.95) and ~incompressible, so we do NOT claim any
code-entropy saving.

Honest bpw = (b0 + b1 + cb + s bytes) * 8 / N  +  membership_coded_bits / N

The membership coder is Python stdlib `lzma` (obviously lossless) and the tool
HARD-VERIFIES decompress == original before reporting. The i.i.d. entropy
(arithmetic-coder floor) is printed for reference. No analytic hand-waving: the
core streams are real bytes, the membership size is a real compressed byte
count with a verified round-trip.
"""
import argparse
import glob
import lzma
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpk_unpack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with *.dpk.safetensors")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "*.dpk.safetensors")))
    assert files, f"no .dpk.safetensors in {args.dir}"

    total_w = 0
    naive_bytes = 0
    core_bytes = 0            # everything except the m plane
    naive_m_bytes = 0
    memb_chunks = []          # non-salient, non-pad membership bits per layer

    for fp in files:
        tensors, meta = dpk_unpack.load_container(fp, "cpu")
        if meta.get("mmode") != "element":
            raise SystemExit(f"{fp}: mmode={meta.get('mmode')} — this tool "
                             f"measures element-membership containers")
        R, C, C_orig = meta["R"], meta["C"], meta["C_orig"]
        total_w += R * C_orig
        for k, t in tensors.items():
            nb = t.numel() * t.element_size()
            naive_bytes += nb
            if k == "m":
                naive_m_bytes += nb
            else:
                core_bytes += nb
        m = dpk_unpack.expand_plane(tensors["m"], C)                 # [R,C]
        s = dpk_unpack.expand_plane(tensors["s"].unsqueeze(0), C)[0]  # [C]
        col_ok = (~s) & (torch.arange(C) < C_orig)                   # non-sal,non-pad
        memb_chunks.append(m[:, col_ok].reshape(-1).contiguous())

    memb = torch.cat(memb_chunks).to(torch.uint8).numpy()
    n = memb.size
    packed = np.packbits(memb)
    comp = lzma.compress(packed.tobytes(), preset=9 | lzma.PRESET_EXTREME)
    dec = np.unpackbits(np.frombuffer(lzma.decompress(comp),
                                      dtype=np.uint8))[:n]
    assert np.array_equal(dec, memb), "lzma membership round-trip FAILED"

    p = float(memb.mean())
    H = 0.0 if p in (0.0, 1.0) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    naive_bpw = naive_bytes * 8 / total_w
    core_bpw = core_bytes * 8 / total_w
    m_naive = naive_m_bytes * 8 / total_w
    m_lzma = len(comp) * 8 / total_w
    m_ent = H * n / total_w

    print(f"dir={args.dir}")
    print(f"layers={len(files)} total_weights={total_w}")
    print(f"non-salient membership bits N={n}  frac={n/total_w:.4f}  "
          f"P(tail|non-sal)={p:.4f}  H={H:.4f}")
    print(f"NAIVE packed bpw (dpk)         = {naive_bpw:.4f}")
    print(f"  core b0+b1+cb+s             = {core_bpw:.4f}")
    print(f"  membership naive 1-bit plane= {m_naive:.4f}")
    print(f"  membership lzma (VERIFIED)  = {m_lzma:.4f}")
    print(f"  membership i.i.d. entropy   = {m_ent:.4f}")
    print(f"HONEST bpw (lzma membership)  = {core_bpw + m_lzma:.4f}  "
          f"[round-trip lossless VERIFIED]")
    print(f"HONEST bpw (entropy floor)    = {core_bpw + m_ent:.4f}")


if __name__ == "__main__":
    main()
