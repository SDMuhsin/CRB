"""gen_synthetic.py — spec-conformant synthetic DPK test-vector generator (K3's own).

Generates random DPK-format artifacts per llmdocs/cuda_kernel/02_storage_format_design.md:
  * s bitmap built COLUMN-WISE PER 128-BLOCK (§2: "1 = salient column ... from the
    per-128-block search"), with realistic n_sal per block around the V1-measured
    f3 ~= 0.21 (median 29/block, search cap 49) — plus blocks at 0 and at the cap.
  * m plane element-wise, 0 (don't-care per §2) at salient columns, 0 at pad columns.
  * b0/b1 planes uniform random bits (zero at pad columns).
  * cb: random levels SORTED ASCENDING per (row, group, partition) (§2), bf16.
  * xhat: uniform nibbles 0..15; pad columns carry 8 (§3 padding trick / §4).
  * C_orig < C padding per §3: pad cols b0=b1=m=s=0, xhat=8.

Adversarial variants (all still §3-conformant unless flagged):
  * all_bucket_p1q0  — every weight lands in bucket (P1, code 0)
  * all_bucket_p3q3  — every real column salient with code 3 (pad cols stay P1q0)
  * empty_partitions — some groups have no salient columns (P3 empty), some
    (row, group)s have m=0 everywhere (P2 empty), one group all-salient
    (P1+P2 empty there)
  * garbage_m_at_salient — SPEC-VIOLATING m=1 under salient columns; the §3
    invariant makes m don't-care there, so kernel/reference must be insensitive.
    Used only in the robustness check, flagged in meta.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = ["gen_case", "gen_batch", "VARIANTS"]

VARIANTS = ["realistic", "all_bucket_p1q0", "all_bucket_p3q3",
            "empty_partitions", "garbage_m_at_salient"]

NSAL_MEDIAN = 29   # V1-measured median salient columns per 128-block
NSAL_CAP = 49      # V1-measured search cap


def _pack_bits_rows(bits: np.ndarray) -> torch.Tensor:
    """[..., C] 0/1 uint8 -> torch uint32 [..., C/32]; bit i of word w = col 32w+i."""
    assert bits.shape[-1] % 32 == 0
    by = np.packbits(bits.astype(np.uint8), axis=-1, bitorder="little")
    words = by.view(np.uint32) if by.flags["C_CONTIGUOUS"] else np.ascontiguousarray(by).view(np.uint32)
    return torch.from_numpy(words.copy())


def _pack_nibbles(vals: np.ndarray) -> torch.Tensor:
    """[C] 0..15 -> torch uint32 [C/8]; nibble n of word w = col 8w+n."""
    assert vals.shape[-1] % 8 == 0
    v = vals.astype(np.uint32).reshape(-1, 8)
    w = np.zeros(v.shape[0], dtype=np.uint32)
    for n in range(8):
        w |= v[:, n] << np.uint32(4 * n)
    return torch.from_numpy(w)


def _pack_nibbles_batched(vals: np.ndarray) -> torch.Tensor:
    """[M, C] 0..15 -> torch uint32 [M, C/8]; nibble n of word w = col 8w+n."""
    assert vals.shape[-1] % 8 == 0
    v = vals.astype(np.uint32).reshape(vals.shape[0], -1, 8)
    w = np.zeros(v.shape[:2], dtype=np.uint32)
    for n in range(8):
        w |= v[:, :, n] << np.uint32(4 * n)
    return torch.from_numpy(w)


def gen_batch(C_orig: int, M: int, seed: int, C: int | None = None):
    """Batched activations for the GEMM path (doc 02 par.4, K4 contract).

    Returns (Xhat uint32 [M, C/8], a_s_vec fp32 [M]). Columns >= C_orig carry
    nibble 8 (par.3 padding trick). C defaults to C_orig rounded up to 128.
    """
    rng = np.random.default_rng(seed ^ 0x5EED)   # decorrelate from gen_case
    if C is None:
        C = math.ceil(C_orig / 128) * 128
    xh = rng.integers(0, 16, (M, C)).astype(np.uint8)
    xh[:, C_orig:] = 8
    a_s = (0.01 + rng.random(M) * 0.1).astype(np.float32)
    return _pack_nibbles_batched(xh), torch.from_numpy(a_s)


def gen_case(R: int, C_orig: int, g: int, seed: int, variant: str = "realistic") -> dict:
    """Build one synthetic DPK case. All tensors on CPU; caller moves to device.

    Returns dict: b0, b1, m (uint32 [R, C/32]), s (uint32 [C/32]),
    cb (bf16 [R, NG, 3, 4]), xhat (uint32 [C/8]), a_s (float), meta.
    """
    assert variant in VARIANTS, variant
    assert g % 128 == 0 and g > 0
    rng = np.random.default_rng(seed)
    C = math.ceil(C_orig / 128) * 128
    NB = C // 128
    NG = math.ceil(C / g)

    real = np.zeros(C, dtype=bool)
    real[:C_orig] = True

    # ---- column-wise salient bitmap, built per 128-block ----
    sal = np.zeros(C, dtype=bool)
    if variant == "all_bucket_p3q3":
        sal[:C_orig] = True
    elif variant != "all_bucket_p1q0":
        for b in range(NB):
            lo, hi = b * 128, min((b + 1) * 128, C_orig)
            if hi <= lo:
                continue  # pure padding block: no salient columns
            n_cols = hi - lo
            # clipped normal around the V1 median: yields 0-salient blocks and
            # cap-saturated blocks in the tails (both observed in V1)
            n = int(np.clip(round(rng.normal(NSAL_MEDIAN, 14)), 0, min(NSAL_CAP, n_cols)))
            if n > 0:
                sal[lo + rng.choice(n_cols, size=n, replace=False)] = True

    # ---- planes ----
    if variant == "all_bucket_p1q0":
        b0 = np.zeros((R, C), dtype=np.uint8)
        b1 = np.zeros((R, C), dtype=np.uint8)
        m = np.zeros((R, C), dtype=np.uint8)
    elif variant == "all_bucket_p3q3":
        b0 = (np.ones((R, C), dtype=np.uint8) * real).astype(np.uint8)
        b1 = b0.copy()
        m = np.zeros((R, C), dtype=np.uint8)
    else:
        b0 = (rng.integers(0, 2, (R, C)) * real).astype(np.uint8)
        b1 = (rng.integers(0, 2, (R, C)) * real).astype(np.uint8)
        m = (rng.integers(0, 2, (R, C)) * real).astype(np.uint8)
        m[:, sal] = 0  # don't-care stored as 0 per §2

    if variant == "empty_partitions":
        # P3 empty: strip salient columns from even-indexed groups
        for G in range(0, NG, 2):
            sal[G * g:(G + 1) * g] = False
        # P2 empty for the first half of rows in odd-indexed groups
        for G in range(1, NG, 2):
            m[: max(1, R // 2), G * g:min((G + 1) * g, C)] = 0
        # P1+P2 empty: make group 0 all-salient (real columns only)
        sal[0:min(g, C_orig)] = True
        m[:, sal] = 0

    if variant == "garbage_m_at_salient":
        m[:, sal & real] = 1  # spec-violating; §3 makes these don't-care

    # ---- activations ----
    xh = rng.integers(0, 16, C).astype(np.uint8)
    xh[~real] = 8  # §3 padding trick

    # ---- codebooks: sorted ascending per (row, group, partition) ----
    cb = rng.standard_normal((R, NG, 3, 4)).astype(np.float32)
    cb.sort(axis=-1)
    cb_t = torch.from_numpy(cb).to(torch.bfloat16).contiguous()

    a_s = float(0.01 + rng.random() * 0.1)

    return {
        "b0": _pack_bits_rows(b0),
        "b1": _pack_bits_rows(b1),
        "m": _pack_bits_rows(m),
        "s": _pack_bits_rows(sal.astype(np.uint8).reshape(1, C))[0],
        "cb": cb_t,
        "xhat": _pack_nibbles(xh),
        "a_s": a_s,
        "meta": {
            "R": R, "C": C, "C_orig": C_orig, "B": 128, "g": g, "NG": NG,
            "mmode": "element", "cbdtype": "bf16", "variant": variant,
            "seed": seed, "f3_actual": float(sal[:C_orig].mean()) if C_orig else 0.0,
            "spec_conformant": variant != "garbage_m_at_salient",
        },
    }


if __name__ == "__main__":
    for v in VARIANTS:
        c = gen_case(64, 1000, 256, 0, v)
        mt = c["meta"]
        print(f"{v:24s} R={mt['R']} C={mt['C']} (orig {mt['C_orig']}) g={mt['g']} "
              f"NG={mt['NG']} f3={mt['f3_actual']:.3f} a_s={c['a_s']:.4f}")
