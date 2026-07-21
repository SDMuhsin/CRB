"""K5a GATE G-A — Marlin pack/unpack roundtrip for all 196 sublayers.

For every quantized sublayer of Qwen3-0.6B (sym INT4 g=128 artifacts from
marlin_sym_quantize.py):

  1. Rebuild the reference dequant Wq_ref = fp16((q - 8) * s)  — the exact
     fp16 semantics the Marlin kernel computes in registers.
  2. Pack with the OFFICIAL marlin.Layer.pack (patched build @1f25790) from
     (Wq_ref, s) — i.e. exactly how B1 built its layers.
  3. UNPACK the packed B and permuted s with an independent inverse
     (serve_common.unpack_marlin_B / unpack_marlin_s, permutation tables
     imported from the marlin package itself) and assert:
       (i)   unpacked codes == stored q, bitwise, all elements;
       (ii)  unpacked scales == stored s, bitwise (int16 view);
       (iii) dequant(unpacked codes, unpacked scales) == Wq_ref, bitwise
             (int16 view) — ZERO mismatches (the G-A gate).
  4. Record exact artifact bytes (B + s) => bpw must equal 4.1250.
  5. Save the packed buffers to <layer>.marlin.safetensors — these exact
     G-A-verified bytes are what the serving model loads.

Gate-of-gates: before the real run, a corrupted-code case must FAIL the
comparison (proves the check can detect errors).

Exit 0 + k5_logs/gate_A_PASS.json only if all 196 layers pass with 0
mismatches.

Usage:
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=1 python -u kernels/serve/verify_marlin_pack.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from serve_common import (  # noqa: E402
    DUMP_DIR, GATE_A_MARKER, GROUPSIZE, LOG_DIR, MODEL_NAME,
    N_QUANT_SUBLAYERS, all_layer_names, dequant_ref_fp16, load_q4_artifact,
    marlin_path, require_gpu1, unpack_marlin_B, unpack_marlin_s,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import marlin  # noqa: E402  (patched build; pack() is pure host-side code)


def pack_layer(Wq_ref: torch.Tensor, s_kg_n: torch.Tensor):
    """Official Marlin packing, exactly as kernels/bench/bench_int4_baseline
    does it. Wq_ref: (N, K) fp16 dequant weights; s_kg_n: (K/128, N) fp16.
    Returns (B int32 (K/16, N*2), s_packed fp16 (K/128, N))."""
    N, K = Wq_ref.shape
    linear = nn.Linear(K, N, bias=False)
    linear.weight.data = Wq_ref.contiguous()
    layer = marlin.Layer(K, N, groupsize=GROUPSIZE)
    layer.pack(linear, s_kg_n.t().contiguous())   # pack() expects transposed
    return layer.B.clone(), layer.s.clone()


def verify_one(q, s, layer_name):
    """Returns (record dict, packed B, packed s). Raises on gate failure."""
    N, K = q.shape
    Wq_ref = dequant_ref_fp16(q, s)                      # (N, K) fp16
    B, s_packed = pack_layer(Wq_ref, s)

    # ---- independent unpack ----
    codes_kn = unpack_marlin_B(B, K, N)                  # (K, N) int16 0..15
    s_rec = unpack_marlin_s(s_packed, K, N)              # (K/128, N) fp16

    bad_codes = int((codes_kn.t().to(torch.uint8) != q).sum().item())
    bad_scales = int((s_rec.view(torch.int16)
                      != s.view(torch.int16)).sum().item())
    Wq_unpacked = dequant_ref_fp16(codes_kn.t().to(torch.uint8).contiguous(),
                                   s_rec)
    bad_dequant = int((Wq_unpacked.view(torch.int16)
                       != Wq_ref.view(torch.int16)).sum().item())

    wbytes = B.nelement() * B.element_size() + s_packed.nelement() * s_packed.element_size()
    bpw = wbytes * 8 / (K * N)

    ok = bad_codes == 0 and bad_scales == 0 and bad_dequant == 0
    rec = {"layer": layer_name, "N": N, "K": K,
           "code_mismatches": bad_codes, "scale_mismatches": bad_scales,
           "dequant_mismatches": bad_dequant,
           "weight_bytes": wbytes, "bpw": bpw, "pass": ok}
    if not ok:
        raise RuntimeError(f"G-A FAIL {layer_name}: {rec}")
    if abs(bpw - 4.125) > 1e-12:
        raise RuntimeError(f"G-A FAIL {layer_name}: bpw {bpw} != 4.1250")
    return rec, B, s_packed


def gate_of_gates():
    """A deliberately corrupted code must be DETECTED by the same checks."""
    torch.manual_seed(0)
    N, K = 256, 256
    q = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    s = (torch.rand(K // GROUPSIZE, N) * 0.01 + 0.001).half()
    Wq_ref = dequant_ref_fp16(q, s)
    B, s_packed = pack_layer(Wq_ref, s)
    codes = unpack_marlin_B(B, K, N)
    assert torch.equal(codes.t().to(torch.uint8), q), \
        "gate-of-gates setup broken: clean roundtrip failed"
    # flip one packed bit -> exactly one code changes -> must be detected
    B_bad = B.clone()
    B_bad.view(-1)[B_bad.numel() // 2] ^= (1 << 9)
    codes_bad = unpack_marlin_B(B_bad, K, N)
    n_diff = int((codes_bad != codes).sum().item())
    assert n_diff >= 1, "gate-of-gates FAILED: corrupted bit not detected"
    print(f"gate-of-gates: 1 flipped packed bit -> {n_diff} code mismatch(es) "
          f"detected  OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DUMP_DIR)
    ap.add_argument("--allow-any-gpu", action="store_true")
    args = ap.parse_args()
    require_gpu1(args.allow_any_gpu)

    t0 = time.time()
    gate_of_gates()

    names = all_layer_names()
    assert len(names) == N_QUANT_SUBLAYERS
    records = []
    total_bytes = 0
    total_params = 0
    for i, lname in enumerate(names):
        q, s, meta = load_q4_artifact(lname, args.dump_dir)
        assert meta["layer"] == lname and meta["groupsize"] == GROUPSIZE
        rec, B, s_packed = verify_one(q, s, lname)
        records.append(rec)
        total_bytes += rec["weight_bytes"]
        total_params += rec["N"] * rec["K"]
        mmeta = {"layer": lname, "model": MODEL_NAME, "K": rec["K"],
                 "N": rec["N"], "groupsize": GROUPSIZE,
                 "format": "marlin sym g=128 (patched build @1f25790)",
                 "gate_A": "pass"}
        save_file({"B": B.contiguous(), "s": s_packed.contiguous()},
                  marlin_path(lname, args.dump_dir),
                  metadata={"meta": json.dumps(mmeta)})
        if (i + 1) % 28 == 0:
            print(f"G-A [{i+1:3d}/196] ... {lname}: 0 mismatches, "
                  f"bpw={rec['bpw']:.4f}", flush=True)

    agg_bpw = total_bytes * 8 / total_params
    out = {
        "gate": "G-A (Marlin pack/unpack/dequant roundtrip, bitwise)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dump_dir": os.path.abspath(args.dump_dir),
        "n_layers": len(records),
        "total_code_mismatches": sum(r["code_mismatches"] for r in records),
        "total_scale_mismatches": sum(r["scale_mismatches"] for r in records),
        "total_dequant_mismatches": sum(r["dequant_mismatches"] for r in records),
        "total_weight_bytes_B_plus_s": total_bytes,
        "total_quantized_params": total_params,
        "aggregate_bpw": agg_bpw,
        "elapsed_s": round(time.time() - t0, 1),
        "layers": records,
        "pass": all(r["pass"] for r in records) and len(records) == N_QUANT_SUBLAYERS,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(GATE_A_MARKER if out["pass"]
              else os.path.join(LOG_DIR, "gate_A_FAIL.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nG-A: {len(records)}/196 layers, dequant mismatches = "
          f"{out['total_dequant_mismatches']}, aggregate bpw = {agg_bpw:.4f}, "
          f"bytes(B+s) = {total_bytes:,}")
    print(f"GATE G-A: {'PASS' if out['pass'] else 'FAIL'}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
