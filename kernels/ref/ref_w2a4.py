"""K2 deliverable 3 — reference W2A4 GEMV implementations + G3 gate runner.

Two independent references over the DPK container (doc 02):

  (a) gemv_direct:  W = unpack(container) -> fp32; x = (x_hat - 8) * a_s in
      fp32; y = W @ x in fp32.  (Obviously-correct baseline.)
  (b) gemv_bucket:  doc 02 §7 bucket-sum: integer S_hat[p][k], N[p][k] per
      (row, group) via vectorized scatter-add, then
      y = a_s * sum_G sum_p sum_k cb[i][G][p][k] * (S_hat - 8*N)  in fp32.

Activation format (doc 02 §4): x_hat = clamp(round(x/a_s), -8, 7) + 8 stored
as unsigned nibbles, 8 per u32, u32[ceil(C/8)], fp32 scalar scale a_s.
Nibble order: nibble j%8 of word j/8 occupies bits 4*(j%8)..4*(j%8)+3
(LSB-first, consistent with the §2a plane bit order; §4 does not state the
nibble order explicitly — flagged as a spec ambiguity in the K2 report).
Padded columns (>= C_orig) carry x_hat = 8 (x = 0; §3 padding trick).

G3 protocol (CLI): on >= 8 diverse sublayers, over 5 random A4 seeds + 3 edge
vectors (all x_hat = 0 / 15 / 8):
  * gate "max rel diff < 1e-5" between (b) and (a), with the rel-diff
    denominator DOCUMENTED as the output scale:
        rel_norm = max_i |b_i - a_i| / ||a||_inf         < 1e-5
    Rationale (measured, see K2 report): on cancelled output elements
    (|a_i| << ||a||_inf) the strict elementwise ratio measures (a)'s OWN
    fp32 summation rounding, not (b)'s correctness — (b)'s integer inner
    sums are exact, and (b) is empirically closer to the fp64 truth than
    (a) at exactly those elements. Strict elementwise rel (floor 1e-30)
    and a floored variant (floor 1e-8*||a||_inf) are also reported.
  * fp64 math-identity cross-check: both paths recomputed in float64 must
    agree to < 1e-10 elementwise-strict (proves the §7 identity independent
    of fp32 noise; there is no meaningful fp64 summation error at C <= 3072);
  * oracle: (a) vs y = matmul(W_bf16, x_bf16).float() within norm-rel 2e-2
    (bf16 rounding of x dominates; with a_s a power of two, x is exact in
    bf16 and the oracle differs only by accumulation path).

Usage:
    python kernels/ref/ref_w2a4.py --dir <dump_dir> [--device cuda:0]
        [--layers name1,name2,...] [--seeds 5]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pack"))
import dpk_unpack  # noqa: E402

DEFAULT_LAYERS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",     # C = 2048
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.down_proj",        # C = 3072
    "model.layers.13.self_attn.q_proj",
    "model.layers.13.mlp.down_proj",
    "model.layers.27.self_attn.o_proj",
    "model.layers.27.mlp.up_proj",
    "model.layers.27.mlp.down_proj",
]

GATE_REL = 1e-5
GATE_F64 = 1e-10
GATE_ORACLE = 2e-2
DENOM_FLOOR_FRAC = 1e-8


# ---------------------------------------------------------------------------
# A4 activation packing (doc 02 §4)
# ---------------------------------------------------------------------------
def pack_a4(x_hat: torch.Tensor) -> torch.Tensor:
    """int tensor [C], values 0..15 -> uint32 [ceil(C/8)], LSB-first nibbles.
    Positions beyond C in the last word are padded with x_hat = 8."""
    assert x_hat.dim() == 1
    xh = x_hat.to(torch.int64)
    assert bool(((xh >= 0) & (xh <= 15)).all()), "x_hat out of [0, 15]"
    C = xh.numel()
    Cp = -(-C // 8) * 8
    if Cp != C:
        xh = torch.cat([xh, torch.full((Cp - C,), 8, dtype=torch.int64,
                                       device=xh.device)])
    v = xh.view(-1, 8)
    sh = (torch.arange(8, device=xh.device, dtype=torch.int64) * 4)
    w64 = (v << sh).sum(dim=-1)                     # [Cp/8], < 2**32
    w64 = torch.where(w64 >= 2**31, w64 - 2**32, w64)
    return w64.to(torch.int32).view(torch.uint32)


def unpack_a4(words: torch.Tensor, C: int) -> torch.Tensor:
    """uint32 [ceil(C/8)] -> int64 [C] of nibble values 0..15."""
    w = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sh = (torch.arange(8, device=w.device, dtype=torch.int64) * 4)
    nib = (w.unsqueeze(-1) >> sh) & 0xF
    return nib.reshape(-1)[:C]


def make_xhat(meta, kind, seed=0, device="cpu"):
    """Activation nibbles for the container's padded C; pad columns get 8."""
    C, C_orig = meta["C"], meta["C_orig"]
    if kind == "rand":
        gen = torch.Generator(device="cpu").manual_seed(seed)
        xh = torch.randint(0, 16, (C_orig,), generator=gen).to(device)
    elif kind in ("all0", "all15", "all8"):
        xh = torch.full((C_orig,), {"all0": 0, "all15": 15, "all8": 8}[kind],
                        dtype=torch.int64, device=device)
    else:
        raise ValueError(kind)
    if C != C_orig:
        xh = torch.cat([xh, torch.full((C - C_orig,), 8, dtype=torch.int64,
                                       device=device)])
    return xh


# ---------------------------------------------------------------------------
# (a) direct reference
# ---------------------------------------------------------------------------
@torch.no_grad()
def gemv_direct(tensors, meta, xhat_words, a_s, dtype=torch.float32):
    C = meta["C"]
    W = dpk_unpack.unpack(tensors, meta).to(dtype)           # [R, C]
    xh = unpack_a4(xhat_words, C).to(W.device)
    x = (xh - 8).to(dtype) * torch.tensor(a_s, dtype=dtype, device=W.device)
    return W @ x


# ---------------------------------------------------------------------------
# (b) bucket-sum reference (doc 02 §7)
# ---------------------------------------------------------------------------
@torch.no_grad()
def gemv_bucket(tensors, meta, xhat_words, a_s, dtype=torch.float32):
    R, C, g, NG = meta["R"], meta["C"], meta["g"], meta["NG"]
    dev = tensors["cb"].device

    b0 = dpk_unpack.expand_plane(tensors["b0"], C)
    b1 = dpk_unpack.expand_plane(tensors["b1"], C)

    code = b0.to(torch.int64) + 2 * b1.to(torch.int64)       # [R, C]
    # partition index per doc 02 §3 — element mmode (s+m planes) or column
    # mmode (colmem, K2.6); shared normative helper in dpk_unpack.
    part = dpk_unpack.part_matrix(tensors, meta)
    gidx = (torch.arange(C, device=dev, dtype=torch.int64) // g).unsqueeze(0)
    bucket = (gidx * 3 + part) * 4 + code                    # [R, C] 0..NG*12-1

    xh = unpack_a4(xhat_words, C).to(dev)                    # [C] int64 0..15
    # integer S_hat[p][k], N[p][k] per (row, group) — exact
    S = torch.zeros(R, NG * 12, dtype=torch.int64, device=dev)
    S.scatter_add_(1, bucket, xh.unsqueeze(0).expand(R, C))
    N = torch.zeros(R, NG * 12, dtype=torch.int64, device=dev)
    N.scatter_add_(1, bucket, torch.ones(R, C, dtype=torch.int64, device=dev))
    assert bool((N.sum(dim=1) == C).all())                   # every col bucketed

    fold = (S - 8 * N).to(dtype)                             # [R, NG*12]
    cbf = tensors["cb"].reshape(R, NG * 12).to(dtype)
    a = torch.tensor(a_s, dtype=dtype, device=dev)
    return a * (cbf * fold).sum(dim=1)


# ---------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------
@torch.no_grad()
def gemv_oracle_bf16(tensors, meta, xhat_words, a_s):
    C = meta["C"]
    W = dpk_unpack.unpack(tensors, meta)                     # bf16 [R, C]
    xh = unpack_a4(xhat_words, C).to(W.device)
    x = ((xh - 8).to(torch.float32) * a_s).to(torch.bfloat16)
    return torch.matmul(W, x).to(torch.float32)


# ---------------------------------------------------------------------------
# G3 runner
# ---------------------------------------------------------------------------
def rel_stats(b, a):
    d = (b - a).abs()
    ninf = a.abs().max().clamp(min=1e-30)
    rel_floor = (d / a.abs().clamp(min=DENOM_FLOOR_FRAC * ninf)).max().item()
    rel_strict = (d / a.abs().clamp(min=1e-30)).max().item()
    rel_norm = (d.max() / ninf).item()
    return rel_floor, rel_strict, rel_norm


def run_g3(dump_dir, layers, n_seeds, device):
    cases = ([("rand", s) for s in range(n_seeds)]
             + [("all0", None), ("all15", None), ("all8", None)])
    a_s_table = {0: 1.0 / 64, 1: 0.013, 2: 0.02, 3: 0.005, 4: 0.031}

    worst = {"ab": 0.0, "ab_strict": 0.0, "f64": 0.0, "oracle": 0.0}
    n_fail = 0
    for lname in layers:
        path = os.path.join(dump_dir, f"{lname}.dpk.safetensors")
        tensors, meta = dpk_unpack.load_container(path, device)
        print(f"\n=== {lname}  R={meta['R']} C={meta['C']} "
              f"NG={meta['NG']} ===")
        for kind, seed in cases:
            a_s = a_s_table.get(seed, 1.0 / 64) if kind == "rand" else 1.0 / 64
            xw = pack_a4(make_xhat(meta, kind, seed or 0, device))
            # round-trip sanity on the activation packing itself
            assert bool((unpack_a4(xw, meta["C"]).to(device)
                         == make_xhat(meta, kind, seed or 0, device)).all())

            ya = gemv_direct(tensors, meta, xw, a_s, torch.float32)
            yb = gemv_bucket(tensors, meta, xw, a_s, torch.float32)
            ya64 = gemv_direct(tensors, meta, xw, a_s, torch.float64)
            yb64 = gemv_bucket(tensors, meta, xw, a_s, torch.float64)
            yo = gemv_oracle_bf16(tensors, meta, xw, a_s)

            ab_f, ab_s, ab_n = rel_stats(yb, ya)
            _, f64_s, _ = rel_stats(yb64, ya64)
            # oracle compared on inf-norm scale (bf16-rounded x dominates)
            or_n = ((yo - ya).abs().max()
                    / ya.abs().max().clamp(min=1e-30)).item()

            ok = ab_n < GATE_REL and f64_s < GATE_F64 and or_n < GATE_ORACLE
            n_fail += 0 if ok else 1
            worst["ab"] = max(worst["ab"], ab_n)
            worst["ab_strict"] = max(worst["ab_strict"], ab_s)
            worst["ab_floor"] = max(worst.get("ab_floor", 0.0), ab_f)
            worst["f64"] = max(worst["f64"], f64_s)
            worst["oracle"] = max(worst["oracle"], or_n)
            tag = f"{kind}" + (f"[s{seed},a_s={a_s}]" if kind == "rand"
                               else f"[a_s={a_s}]")
            print(f"  {tag:22s} rel(b,a): norm={ab_n:.3e} strict={ab_s:.3e} "
                  f"floored={ab_f:.3e}  f64-id={f64_s:.3e}  "
                  f"oracle={or_n:.3e}  {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"G3 worst-case: rel(b,a) norm = {worst['ab']:.3e} "
          f"(gate < {GATE_REL:g}); strict = {worst['ab_strict']:.3e}; "
          f"floored = {worst.get('ab_floor', 0.0):.3e}; "
          f"f64 identity = {worst['f64']:.3e} (gate < {GATE_F64:g}); "
          f"bf16 oracle = {worst['oracle']:.3e} (gate < {GATE_ORACLE:g})")
    print(f"G3: {'PASS' if n_fail == 0 else f'FAIL ({n_fail} cases)'}")
    return n_fail == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer names (default: 10 diverse)")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    layers = (args.layers.split(",") if args.layers else DEFAULT_LAYERS)
    ok = run_g3(args.dir, layers, args.seeds, args.device)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
