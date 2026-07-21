"""bench_dpk_gemm.py — DPK W2A4 tensor-core GEMM vs the Marlin INT4 baseline.

For each Qwen3-relevant shape (K=infeatures=C, N=outfeatures=R), batch
M in {16, 128, 2048} and codebook group size g in {512, C}:
  1. Generates spec-conformant synthetic DPK artifacts + batched A4
     activations (kernels/cuda/gen_synthetic.py).
  2. CERTIFIES correctness first (fp32 norm-rel vs reference GEMM + bf16
     mismatch + determinism) — numbers are only reported for a kernel that
     passes its gates in-run.
  3. Measures: packed weight bytes -> bpw (must be IDENTICAL to K3's GEMV
     numbers — same streams, asserted against results_dpk_gemv.json), peak
     CUDA memory delta during the call (must equal the Y tensor exactly: NO
     hidden workspaces, the global-dequant ban is enforced by measurement),
     latency (time_kernel).
  4. RE-RUNS Marlin (patched build, sym g=128) at the SAME M on the same
     shape — same recipe as bench_int4_baseline.py — and reports side by
     side (weight bytes, peak delta, latency).

Activation-side accounting (Req 3 => Req 1 margin): our packed A4 input is
M*C/2 bytes vs Marlin's fp16 X at M*C*2 bytes — a 4x input-activation saving,
reported per config (it is real resident-memory margin at model scale).

REQUIREMENT-1 CHECK at GEMM scale: packed weight bytes < Marlin weight bytes
AND peak delta <= Marlin peak delta (both outputs are M*N*2 B). g=128 is the
K0-1 expected-fail bootstrap config and is not benched here (K3 already
reports it; g in {512, C} are the Req-1 serving candidates).

Run:
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python /workspace/BiLLM2/kernels/bench/bench_dpk_gemm.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(_THIS, "..", "cuda"))

from bench_utils import measure_peak, time_kernel, weight_bytes  # noqa: E402
import dpk_ref as ref  # noqa: E402
from build import build_dpk  # noqa: E402
from gen_synthetic import gen_batch, gen_case  # noqa: E402
from bench_int4_baseline import build_marlin_layer  # noqa: E402
import marlin  # noqa: E402

SEED = 0
DEV = torch.device("cuda:0")
SHAPES = [(4096, 4096), (4096, 14336)]  # (K=C infeatures, N=R outfeatures)
MS = [2, 16, 128, 2048]  # K4b adds M=2 (batched-decode regime)
GROUPS = [512, "C"]
GEMV_JSON = os.path.join(_THIS, "results_dpk_gemv.json")
OUT_JSON = os.path.join(_THIS, "results_dpk_gemm.json")

NORM_REL_GATE = 1e-5
BF16_MISMATCH_FRAC_GATE = 1e-3
N_DET = 3
ALLOC_GRAN = 512  # torch caching-allocator block granularity


def round_gran(b):
    return ((b + ALLOC_GRAN - 1) // ALLOC_GRAN) * ALLOC_GRAN


def load_k3_weight_bytes():
    """K3 GEMV weight bytes per (K, N, g) — ours must be identical (same streams)."""
    out = {}
    if os.path.exists(GEMV_JSON):
        with open(GEMV_JSON) as f:
            for r in json.load(f)["records"]:
                out[(r["K"], r["N"], r["g"])] = r["weight_bytes_total"]
    return out


def main():
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(DEV)
    cc = torch.cuda.get_device_capability(DEV)
    ext = build_dpk()
    k3_bytes = load_k3_weight_bytes()

    meta = {
        "kernel": "DPK W2A4 tensor-core GEMM v1 (kernels/cuda/dpk_gemm.cu)",
        "format": "DPK element mmode: b0/b1/m planes + s bitmap + bf16 cb[R,NG,3,4]",
        "activations": "A4 excess-8 nibbles [M, C/8], per-token fp32 scales [M]",
        "gpu": gpu,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_contract": "kernels/bench/bench_utils.py",
        "comparator": "Marlin FP16xINT4 sym g=128 (patched) RE-RUN at the same M "
                      "in this script (recipe of bench_int4_baseline.py)",
        "req1_rule": "weight bytes < Marlin AND peak delta <= Marlin peak "
                     "(outputs both M*N*2 B); activation-side 4x saving reported",
    }
    print(f"# DPK GEMM v1 vs Marlin on {gpu} (SM{cc[0]}{cc[1]})")

    records, rows = [], []
    gates_pass = True
    req1_pass = True
    worst_ratio = 0.0
    for (K, N) in SHAPES:
        # ---- Marlin comparator at each M (built once per shape) ----
        layer, _ref_w = build_marlin_layer(K, N, 128)
        B, s_m, ws_m = layer.B, layer.s, layer.workspace
        marlin_wbytes = weight_bytes(B, s_m)
        marlin_at = {}
        for M in MS:
            A = torch.randn((M, K), dtype=torch.half, device=DEV)

            def marlin_call():
                C = torch.empty((M, N), dtype=torch.half, device=DEV)
                marlin.mul(A, B, C, s_m, ws_m)
                return C

            _, mpeak, _ = measure_peak(marlin_call)
            mlat = time_kernel(marlin_call, iters=100, warmup=25)
            marlin_at[M] = {"peak_mem_delta_bytes": mpeak,
                            "median_ms": mlat["median_ms"],
                            "min_ms": mlat["min_ms"],
                            "act_bytes_fp16": M * K * 2}
            print(f"  [marlin] K={K} N={N} M={M:5d}: "
                  f"{mlat['median_ms']:.4f} ms, peak {mpeak:,} B")
            del A
            torch.cuda.empty_cache()
        del layer, _ref_w, B, s_m, ws_m
        torch.cuda.empty_cache()

        for gspec in GROUPS:
            g = K if gspec == "C" else gspec
            case = gen_case(R=N, C_orig=K, g=g, seed=SEED, variant="realistic")
            c = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                 for k, v in case.items()}
            b0, b1, m, s, cb = c["b0"], c["b1"], c["m"], c["s"], c["cb"]

            wbytes = weight_bytes(b0, b1, m, s, cb)
            bpw = wbytes * 8 / (K * N)
            k3 = k3_bytes.get((K, N, g))
            same_streams = (k3 is None) or (wbytes == k3)
            gates_pass &= same_streams

            for M in MS:
                Xh, asv = gen_batch(K, M, seed=SEED + M)
                Xh, asv = Xh.to(DEV), asv.to(DEV)

                # ---- correctness certification (gates before numbers) ----
                y_f32 = ext.dpk_gemm(b0, b1, m, s, cb, Xh, asv, g, out_fp32=True)
                y_ref = ref.ref_gemm_direct(b0, b1, m, s, cb, Xh, asv, g)
                nr = ((y_f32 - y_ref).double().norm()
                      / y_ref.double().norm().clamp_min(1e-30)).item()
                y_bf = ext.dpk_gemm(b0, b1, m, s, cb, Xh, asv, g)
                mism = (y_bf.view(torch.int16)
                        != y_ref.to(torch.bfloat16).view(torch.int16)
                        ).float().mean().item()
                det = all(torch.equal(
                    ext.dpk_gemm(b0, b1, m, s, cb, Xh, asv, g), y_bf)
                    for _ in range(N_DET - 1))
                cert = (nr <= NORM_REL_GATE and mism <= BF16_MISMATCH_FRAC_GATE
                        and det and same_streams)
                gates_pass &= cert
                del y_f32, y_ref, y_bf
                torch.cuda.empty_cache()

                # ---- peak memory + latency ----
                def call():
                    return ext.dpk_gemm(b0, b1, m, s, cb, Xh, asv, g)

                _, peak, base = measure_peak(call)
                out_bytes = M * N * 2  # bf16 [M, N]
                exp_peak = round_gran(out_bytes)
                peak_ok = peak <= exp_peak
                gates_pass &= peak_ok  # global dequant buffers are BANNED
                lat = time_kernel(call, iters=100, warmup=25)
                mrec = marlin_at[M]
                ratio = lat["median_ms"] / mrec["median_ms"]
                worst_ratio = max(worst_ratio, ratio)

                verdict = ("PASS" if (wbytes < marlin_wbytes
                                      and peak <= mrec["peak_mem_delta_bytes"])
                           else "FAIL")
                req1_pass &= verdict == "PASS"

                act_ours = M * (K // 2)   # A4 nibbles: M * C/2 bytes
                act_marlin = mrec["act_bytes_fp16"]

                rec = {
                    "K": K, "N": N, "g": g, "g_spec": str(gspec), "M": M,
                    "weight_bytes_total": wbytes, "bpw": bpw,
                    "weight_bytes_equal_k3_gemv": same_streams,
                    "correctness": {"f32_norm_rel": nr,
                                    "bf16_mismatch_frac": mism,
                                    "bitwise_deterministic": det,
                                    "certified": cert},
                    "peak_mem_delta_bytes": peak,
                    "expected_peak_bytes": exp_peak,
                    "peak_is_output_only": peak_ok,
                    "baseline_allocated_bytes": base,
                    "latency_ms": {k: lat[k] for k in
                                   ("mean_ms", "median_ms", "min_ms", "max_ms",
                                    "p90_ms", "iters", "warmup")},
                    "marlin": {"weight_bytes": marlin_wbytes,
                               "bpw": marlin_wbytes * 8 / (K * N), **mrec},
                    "weight_bytes_vs_marlin": wbytes / marlin_wbytes,
                    "latency_vs_marlin": ratio,
                    "activation_bytes": {"ours_a4": act_ours,
                                         "marlin_fp16": act_marlin,
                                         "ratio": act_ours / act_marlin},
                    "req1_verdict": verdict,
                }
                records.append(rec)
                rows.append((f"{K}x{N}", g, M, f"{bpw:.4f}", wbytes,
                             marlin_wbytes, peak,
                             mrec["peak_mem_delta_bytes"],
                             f"{lat['median_ms']:.4f}",
                             f"{mrec['median_ms']:.4f}", f"{ratio:.2f}x",
                             f"{act_ours:,}/{act_marlin:,}",
                             "PASS" if cert else "FAIL", verdict))
                print(f"K={K} N={N:5d} g={g:5d} M={M:5d}: "
                      f"lat={lat['median_ms']:.4f} ms (marlin "
                      f"{mrec['median_ms']:.4f}, {ratio:.2f}x) "
                      f"peak={peak:,} B (out-only={peak_ok}) "
                      f"nr={nr:.2e} mism={mism:.2e} "
                      f"cert={'PASS' if cert else 'FAIL'} req1={verdict}")
                del Xh, asv
                torch.cuda.empty_cache()
            del c, b0, b1, m, s, cb, case
            torch.cuda.empty_cache()

    # ---- 1000-call warm-allocator loop (Req-1 evidence): zero growth ----
    # Largest config: 4096x14336, g=C, M=2048. Any hidden per-call
    # allocation or workspace would show up as reserved-memory growth.
    case = gen_case(R=14336, C_orig=4096, g=4096, seed=SEED, variant="realistic")
    c = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
         for k, v in case.items()}
    Xh, asv = gen_batch(4096, 2048, seed=SEED)
    Xh, asv = Xh.to(DEV), asv.to(DEV)
    for _ in range(10):
        ext.dpk_gemm(c["b0"], c["b1"], c["m"], c["s"], c["cb"], Xh, asv, 4096)
    torch.cuda.synchronize()
    res0 = torch.cuda.memory_reserved(DEV)
    for _ in range(1000):
        ext.dpk_gemm(c["b0"], c["b1"], c["m"], c["s"], c["cb"], Xh, asv, 4096)
    torch.cuda.synchronize()
    res_growth = torch.cuda.memory_reserved(DEV) - res0
    warm_loop_ok = res_growth == 0
    gates_pass &= warm_loop_ok
    print(f"\n1000-call warm-allocator loop (4096x14336 g=C M=2048): "
          f"reserved growth = {res_growth} B [{'PASS' if warm_loop_ok else 'FAIL'}]")
    del c, Xh, asv, case
    torch.cuda.empty_cache()

    hdr = ("| shape KxN | g | M | bpw | weight bytes | marlin bytes | "
           "peak delta (B) | marlin peak (B) | dpk med ms | marlin med ms | "
           "lat ratio | act bytes ours/marlin | correctness | Req-1 |")
    sep = "|" + "---|" * 14
    lines = [hdr, sep] + ["| " + " | ".join(str(x) for x in r) + " |"
                          for r in rows]
    table = "\n".join(lines)
    print("\n" + table)

    summary = {
        "all_correctness_certified": gates_pass,
        "req1_all_pass": req1_pass,
        "warm_loop_1000_calls_reserved_growth_bytes": int(res_growth),
        "worst_latency_ratio_vs_marlin": worst_ratio,
        "activation_note": "A4 input activations are M*C/2 bytes vs Marlin's "
                           "fp16 M*C*2 bytes (4x smaller) — real Req-1 margin "
                           "at model scale, legitimate under Req 3 (W2A4).",
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "records": records, "summary": summary,
                   "markdown_table": table}, f, indent=2)
    print(f"\nJSON written to {OUT_JSON}")
    print(f"CORRECTNESS CERTIFICATION: {'PASS' if gates_pass else 'FAIL'}")
    print(f"REQ-1 (weight bytes < Marlin AND peak <= Marlin): "
          f"{'PASS' if req1_pass else 'FAIL'}")
    print(f"worst latency ratio vs Marlin: {worst_ratio:.2f}x")
    return 0 if (gates_pass and req1_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
