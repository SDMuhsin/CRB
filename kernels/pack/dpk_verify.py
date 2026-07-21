"""K2 deliverable 2b — G2 gate runner: DPK round-trip verification.

For every `<name>.dpk.safetensors` in the dump directory:
  * unpack via the reference §3 invariant (dpk_unpack.py),
  * compare against the sibling `<name>.wq.safetensors` ground truth
    (final quantized layer weight) BITWISE in bf16 over 100% of the
    [R, C_orig] elements,
  * report packed bpw = 8 * (b0+b1+m+s+cb bytes) / (R * C_orig).

Exit code 0 iff every layer is bitwise-exact.

Usage:
    python kernels/pack/dpk_verify.py --dir <dump_dir> [--device cuda:0]
"""

import argparse
import glob
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpk_unpack  # noqa: E402


def load_wq(path, device="cpu"):
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor("wq").to(device)


def verify_layer(dpk_path, wq_path=None, device="cpu"):
    """Returns dict with keys: ok, n_mismatch, bpw, R, C_orig, layer_name."""
    tensors, meta = dpk_unpack.load_container(dpk_path, device)
    if wq_path is None:
        wq_path = dpk_path.replace(".dpk.safetensors", ".wq.safetensors")
    wq = load_wq(wq_path, device)
    R, C_orig = meta["R"], meta["C_orig"]
    if tuple(wq.shape) != (R, C_orig) or wq.dtype != torch.bfloat16:
        raise ValueError(f"{wq_path}: wq is {wq.dtype}{tuple(wq.shape)}, "
                         f"expected bfloat16 ({R}, {C_orig})")

    W = dpk_unpack.unpack(tensors, meta)[:, :C_orig]
    eq = (W.contiguous().view(torch.int16)
          == wq.contiguous().view(torch.int16))
    n_mismatch = int((~eq).sum().item())

    packed_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    return {
        "ok": n_mismatch == 0,
        "n_mismatch": n_mismatch,
        "n_elems": R * C_orig,
        "bpw": 8.0 * packed_bytes / (R * C_orig),
        "packed_bytes": packed_bytes,
        "R": R, "C_orig": C_orig,
        "layer_name": meta.get("layer_name", os.path.basename(dpk_path)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quiet", action="store_true",
                    help="only print failures and the aggregate")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.dpk.safetensors")))
    if not files:
        print(f"FAIL: no .dpk.safetensors files in {args.dir}")
        sys.exit(2)

    n_ok = 0
    tot_bits = 0
    tot_elems = 0
    by_type = {}   # sublayer type -> (bits, elems)
    failures = []
    for fp in files:
        r = verify_layer(fp, device=args.device)
        tot_bits += 8 * r["packed_bytes"]
        tot_elems += r["n_elems"]
        stype = r["layer_name"].split(".")[-1]
        b, e = by_type.get(stype, (0, 0))
        by_type[stype] = (b + 8 * r["packed_bytes"], e + r["n_elems"])
        status = "OK " if r["ok"] else "FAIL"
        if r["ok"]:
            n_ok += 1
        else:
            failures.append((r["layer_name"], r["n_mismatch"]))
        if not args.quiet or not r["ok"]:
            print(f"[{status}] {r['layer_name']:45s} R={r['R']:5d} "
                  f"C={r['C_orig']:5d} mismatches={r['n_mismatch']:>8d}"
                  f"/{r['n_elems']:<9d} bpw={r['bpw']:.4f}")

    print("-" * 78)
    print("per-sublayer-type packed bpw:")
    for stype in sorted(by_type):
        b, e = by_type[stype]
        print(f"  {stype:12s} {b / e:.4f}")
    print(f"AGGREGATE packed bpw = {tot_bits / tot_elems:.4f} "
          f"({tot_elems} weights over {len(files)} sublayers)")
    print(f"G2 round-trip: {n_ok}/{len(files)} sublayers bitwise-exact -> "
          + ("PASS" if n_ok == len(files) else "FAIL"))
    if failures:
        for name, nm in failures:
            print(f"  FAILED: {name} ({nm} mismatching elements)")
        sys.exit(1)


if __name__ == "__main__":
    main()
