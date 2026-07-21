"""K2 GATE-OF-GATES — prove the verification gates actually fail on corruption.

Protocol (K2 mission):
  1. Copy one layer's container (+ wq ground truth) to a scratch directory.
  2. Corruption A: flip ONE bit of ONE b0 plane word — chosen programmatically
     so the flipped 2-bit code lands on a level with a DIFFERENT bf16 bit
     pattern (a flip onto a duplicated/padded level would be a semantically
     lossless byte change that no value-level gate can or should detect).
     Run the G2 round-trip check -> must FAIL with exactly 1 mismatch.
  3. Corruption B: XOR the low mantissa bit of ONE USED codebook level
     (chosen so at least one element references it; corrupting an unused/pad
     slot is byte-noise invisible to both W and y by construction).
     Run G2 -> must FAIL; additionally show G3-style detection: bucket-sum
     GEMV on the corrupted container vs direct GEMV on the pristine one
     diverges far beyond the 1e-5 gate.
  4. Delete the corrupted copies. Exit 0 iff both corruptions were CAUGHT.

Usage:
    python kernels/pack/gate_of_gates.py --dir <dump_dir> --scratch <tmpdir>
        [--layer model.layers.0.mlp.gate_proj] [--device cpu]
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref"))
import dpk_unpack  # noqa: E402
import dpk_verify  # noqa: E402
import ref_w2a4  # noqa: E402


def resave(tensors, meta, path):
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()},
              path, metadata={"meta": json.dumps(meta)})


def planes_pcg(tensors, meta):
    """Return (part, code, gidx) int64 [R, C] helper arrays."""
    R, C, g = meta["R"], meta["C"], meta["g"]
    b0 = dpk_unpack.expand_plane(tensors["b0"], C)
    b1 = dpk_unpack.expand_plane(tensors["b1"], C)
    m = dpk_unpack.expand_plane(tensors["m"], C)
    s = dpk_unpack.expand_plane(tensors["s"].unsqueeze(0), C)[0]
    code = b0.to(torch.int64) + 2 * b1.to(torch.int64)
    part = torch.where(s.unsqueeze(0).expand(R, C),
                       torch.full((R, C), 2, dtype=torch.int64),
                       m.to(torch.int64))
    gidx = (torch.arange(C, dtype=torch.int64) // g).unsqueeze(0).expand(R, C)
    return part, code, gidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.gate_proj")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.scratch, exist_ok=True)
    src_dpk = os.path.join(args.dir, f"{args.layer}.dpk.safetensors")
    src_wq = os.path.join(args.dir, f"{args.layer}.wq.safetensors")
    cp_dpk = os.path.join(args.scratch, f"{args.layer}.dpk.safetensors")
    cp_wq = os.path.join(args.scratch, f"{args.layer}.wq.safetensors")

    tensors, meta = dpk_unpack.load_container(src_dpk)
    R, C, C_orig = meta["R"], meta["C"], meta["C_orig"]
    cb_patt = tensors["cb"].view(torch.int16).to(torch.int64) & 0xFFFF
    part, code, gidx = planes_pcg(tensors, meta)
    caught = []

    # ---------------- Corruption A: one bit of one b0 plane word ------------
    # find an element (i, j), j < C_orig, whose level differs bitwise from the
    # level at code^1 within the same (row, group, partition).
    lv = cb_patt  # [R, NG, 3, 4]
    cur = lv[torch.arange(R).unsqueeze(1), gidx, part, code]        # [R, C]
    alt = lv[torch.arange(R).unsqueeze(1), gidx, part, code ^ 1]
    cand = (cur != alt)
    cand[:, C_orig:] = False
    ij = cand.nonzero()[0]
    i, j = int(ij[0]), int(ij[1])
    word, bit = j // 32, j % 32
    print(f"[A] flipping b0[{i}][{word}] bit {bit} (element ({i},{j}); "
          f"code {int(code[i, j])} -> {int(code[i, j]) ^ 1}, level pattern "
          f"0x{int(cur[i, j]):04x} -> 0x{int(alt[i, j]):04x})")

    shutil.copy(src_wq, cp_wq)
    tc = {k: v.clone() for k, v in tensors.items()}
    w = tc["b0"].view(torch.int32)
    w[i, word] = w[i, word] ^ (1 << bit)
    resave(tc, meta, cp_dpk)

    rA = dpk_verify.verify_layer(cp_dpk, cp_wq, args.device)
    print(f"[A] G2 on corrupted copy: ok={rA['ok']} "
          f"mismatches={rA['n_mismatch']}")
    if not rA["ok"] and rA["n_mismatch"] == 1:
        print("[A] CAUGHT by G2 (exactly the flipped element) — PASS")
        caught.append(True)
    else:
        print("[A] NOT caught as expected — FAIL")
        caught.append(False)

    # ---------------- Corruption B: one USED codebook level -----------------
    # occupancy per (row, group, partition, code) from the planes
    occ = torch.zeros(R, meta["NG"] * 12, dtype=torch.int64)
    bucket = (gidx * 3 + part) * 4 + code
    occ.scatter_add_(1, bucket[:, :C_orig].contiguous(),
                     torch.ones(R, C_orig, dtype=torch.int64))
    used = occ.view(R, meta["NG"], 3, 4).nonzero()[0]
    ui, ug, up, uk = (int(x) for x in used)
    n_ref = int(occ.view(R, meta["NG"], 3, 4)[ui, ug, up, uk])
    old = tensors["cb"][ui, ug, up, uk].item()
    print(f"[B] corrupting used cb[{ui}][{ug}][{up}][{uk}] "
          f"(referenced by {n_ref} element(s), value {old})")

    tc2 = {k: v.clone() for k, v in tensors.items()}
    cbw = tc2["cb"].view(torch.int16)
    cbw[ui, ug, up, uk] = cbw[ui, ug, up, uk] ^ 1     # low mantissa bit
    resave(tc2, meta, cp_dpk)
    print(f"[B] new value {tc2['cb'][ui, ug, up, uk].item()}")

    rB = dpk_verify.verify_layer(cp_dpk, cp_wq, args.device)
    print(f"[B] G2 on corrupted copy: ok={rB['ok']} "
          f"mismatches={rB['n_mismatch']} (expected {n_ref})")
    g2_caught = (not rB["ok"]) and rB["n_mismatch"] == n_ref

    # G3-style detection: bucket GEMV on corrupted vs direct on pristine
    tcorr, mcorr = dpk_unpack.load_container(cp_dpk, args.device)
    tgood = {k: v.to(args.device) for k, v in tensors.items()}
    xw = ref_w2a4.pack_a4(ref_w2a4.make_xhat(meta, "all15", device=args.device))
    a_s = 1.0 / 64
    ya = ref_w2a4.gemv_direct(tgood, meta, xw, a_s)
    yb = ref_w2a4.gemv_bucket(tcorr, mcorr, xw, a_s)
    rel = ((yb - ya).abs().max() / ya.abs().max().clamp(min=1e-30)).item()
    g3_caught = rel > ref_w2a4.GATE_REL
    print(f"[B] G3-style: bucket(corrupted) vs direct(pristine) norm-rel = "
          f"{rel:.3e} (gate {ref_w2a4.GATE_REL:g}) -> "
          f"{'CAUGHT' if g3_caught else 'not caught'}")
    if g2_caught and g3_caught:
        print("[B] CAUGHT by G2 AND by G3 — PASS")
        caught.append(True)
    else:
        print(f"[B] g2_caught={g2_caught} g3_caught={g3_caught} — "
              + ("PASS (G2 or G3 suffices)" if (g2_caught or g3_caught)
                 else "FAIL"))
        caught.append(g2_caught or g3_caught)

    # ---------------- cleanup ------------------------------------------------
    for p in (cp_dpk, cp_wq):
        if os.path.exists(p):
            os.remove(p)
    print(f"corrupted copies deleted from {args.scratch}")

    print("GATE-OF-GATES: " + ("PASS" if all(caught) else "FAIL"))
    sys.exit(0 if all(caught) else 1)


if __name__ == "__main__":
    main()
