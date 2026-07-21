"""dpk_ref.py — K3's INDEPENDENT vectorized-torch reference for the DPK format.

Written from llmdocs/cuda_kernel/02_storage_format_design.md ONLY (independent
of K2's packer/reference by mandate — the PI cross-validates afterwards).

Two formulations of the same math:

1. Direct (doc 02 §3 dequant invariant):
       part(i,j) = s[j] ? 2 : (m[i][j] ? 1 : 0)
       code(i,j) = b0[i][j] + 2*b1[i][j]
       W[i,j]    = cb[i][j // g][part(i,j)][code(i,j)]
   then y = W_fp32 @ (a_s * (xhat - 8))_fp32   — a plain fp32 GEMV.

2. Bucket sums (doc 02 §7 identity):
       S[i][G][p][k] = sum_{j in G: part=p, code=k} xhat_j   (exact integers)
       N[i][G][p][k] = |{j in G: part=p, code=k}|
       y_i = a_s * sum cb[i][G][p][k] * (S - 8N)             (fp32)

Bit conventions (doc 02 §2a): bit i of word w covers column 32*w + i
(LSB-first). Activation nibbles: nibble n (bits 4n..4n+3) of word w covers
column 8*w + n (documented choice — §4 fixes 8-per-u32 but not the order; we
extend the LSB-first rule).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "unpack_bits",
    "unpack_nibbles",
    "part_code",
    "dequant_weights",
    "ref_gemv_direct",
    "ref_bucket_sums",
    "ref_gemv_bucket",
    "ref_gemm_direct",
]


def _as_i64(words: torch.Tensor) -> torch.Tensor:
    """uint32/int32 word tensor -> int64 with the same 32-bit pattern (0..2^32-1)."""
    if words.dtype == torch.uint32:
        words = words.view(torch.int32)
    if words.dtype != torch.int32:
        raise TypeError(f"expected uint32/int32 words, got {words.dtype}")
    return words.to(torch.int64) & 0xFFFFFFFF


def unpack_bits(words: torch.Tensor, C: int) -> torch.Tensor:
    """[..., C/32] words -> [..., C] bits (int64 0/1). Bit i of word w = column 32w+i."""
    w = _as_i64(words)
    shifts = torch.arange(32, device=w.device, dtype=torch.int64)
    bits = (w.unsqueeze(-1) >> shifts) & 1  # [..., CW, 32]
    return bits.reshape(*w.shape[:-1], w.shape[-1] * 32)[..., :C]


def unpack_nibbles(words: torch.Tensor, C: int) -> torch.Tensor:
    """[ceil(C/8)] words -> [C] nibble values (int64 0..15). Nibble n of word w = col 8w+n."""
    w = _as_i64(words)
    shifts = torch.arange(8, device=w.device, dtype=torch.int64) * 4
    nib = (w.unsqueeze(-1) >> shifts) & 0xF  # [..., W, 8]
    return nib.reshape(*w.shape[:-1], w.shape[-1] * 8)[..., :C]


def part_code(b0: torch.Tensor, b1: torch.Tensor, m: torch.Tensor,
              s: torch.Tensor):
    """Doc 02 §3: per-element partition (0=P1 bulk, 1=P2 tail, 2=P3 salient) and code."""
    R, CW = b0.shape
    C = CW * 32
    b0b = unpack_bits(b0, C)          # [R, C]
    b1b = unpack_bits(b1, C)
    mb = unpack_bits(m, C)
    sb = unpack_bits(s.reshape(1, CW), C)[0]  # [C]
    part = torch.where(sb.bool().unsqueeze(0).expand(R, C), 2,
                       torch.where(mb.bool(), 1, 0))
    code = b0b + 2 * b1b
    return part, code


def dequant_weights(b0, b1, m, s, cb, g: int) -> torch.Tensor:
    """Unpack per §3 -> fp32 W [R, C] (values exactly the bf16 codebook entries)."""
    R, CW = b0.shape
    C = CW * 32
    NG = cb.shape[1]
    assert NG == math.ceil(C / g), (NG, C, g)
    part, code = part_code(b0, b1, m, s)
    gidx = torch.arange(C, device=b0.device, dtype=torch.int64) // g  # [C]
    flat = gidx.unsqueeze(0) * 12 + part * 4 + code                   # [R, C]
    return torch.gather(cb.float().reshape(R, NG * 12), 1, flat)


def ref_gemv_direct(b0, b1, m, s, cb, xhat, a_s: float, g: int) -> torch.Tensor:
    """Direct fp32 GEMV of the dequantized weights. Returns fp32 y [R]."""
    C = b0.shape[1] * 32
    W = dequant_weights(b0, b1, m, s, cb, g)
    x = a_s * (unpack_nibbles(xhat, C).float() - 8.0)
    return W @ x


def ref_bucket_sums(b0, b1, m, s, xhat, g: int):
    """Exact integer bucket sums per doc 02 §7. Returns (S, N) int32 [R, NG, 3, 4].

    Values are far below int32 range: S <= 15*C <= 15*32768 < 2^19.
    """
    R, CW = b0.shape
    C = CW * 32
    NG = math.ceil(C / g)
    part, code = part_code(b0, b1, m, s)
    xh = unpack_nibbles(xhat, C)  # [C] int64
    gidx = torch.arange(C, device=b0.device, dtype=torch.int64) // g
    bidx = gidx.unsqueeze(0) * 12 + part * 4 + code  # [R, C], p*4+k matches cb layout
    S = torch.zeros(R, NG * 12, dtype=torch.int64, device=b0.device)
    S.scatter_add_(1, bidx, xh.unsqueeze(0).expand(R, C))
    N = torch.zeros(R, NG * 12, dtype=torch.int64, device=b0.device)
    N.scatter_add_(1, bidx, torch.ones_like(bidx))
    return (S.reshape(R, NG, 3, 4).to(torch.int32),
            N.reshape(R, NG, 3, 4).to(torch.int32))


def ref_gemv_bucket(b0, b1, m, s, cb, xhat, a_s: float, g: int) -> torch.Tensor:
    """Bucket formulation: y = a_s * sum cb*(S-8N) in fp32. Returns fp32 y [R]."""
    S, N = ref_bucket_sums(b0, b1, m, s, xhat, g)
    contrib = cb.float() * (S - 8 * N).float()  # [R, NG, 3, 4]
    R = cb.shape[0]
    return a_s * contrib.reshape(R, -1).sum(dim=1)


def ref_gemm_direct(b0, b1, m, s, cb, Xhat, a_s_vec, g: int) -> torch.Tensor:
    """Reference GEMM (K4 contract): dequant fp32 -> matmul -> per-token scale.

    Xhat: uint32/int32 [M, ceil(C/8)] (each row = one token's A4 excess-8
    nibbles, LSB-first per doc 02 par.4); a_s_vec: fp32 [M].
    Returns fp32 Y [M, R] with Y[t, r] = a_s_vec[t] * sum_j W[r,j]*(xhat-8).

    All inputs to the fp32 matmul are exact: W entries are bf16 codebook
    values, (xhat - 8) is an integer in [-8, 7]. TF32 must be off (torch
    default) so this is a true fp32-accumulation reference.
    """
    assert Xhat.dim() == 2, "Xhat must be [M, C/8]"
    C = b0.shape[1] * 32
    W = dequant_weights(b0, b1, m, s, cb, g)              # [R, C] fp32
    X = unpack_nibbles(Xhat, C).float() - 8.0             # [M, C]
    Y = X @ W.t()                                         # fp32 GEMM
    return a_s_vec.float().reshape(-1, 1) * Y
