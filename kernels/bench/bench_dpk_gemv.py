"""bench_dpk_gemv.py — DPK W2A4 bucket-popcount GEMV (M=1) vs the Marlin INT4 baseline.

For each Qwen3-relevant shape (K=infeatures=C, N=outfeatures=R) and codebook
group size g in {128, 512, C}:
  1. Generates spec-conformant synthetic DPK artifacts (kernels/cuda/gen_synthetic.py,
     realistic variant: f3 ~ 0.21 column-salient bitmap per 128-block).
  2. CERTIFIES correctness first (exact-integer S/N vs reference + bf16 output
     check) — numbers are only reported for a kernel that passes its gates.
  3. Measures: resident packed weight bytes -> bpw (actual tensor bytes),
     peak CUDA memory delta during the call (measure_peak; must be ~= the
     output tensor: NO hidden workspaces), latency (time_kernel).
  4. Loads the Marlin M=1 record for the same (K, N) from
     results_int4_baseline.json and reports side by side.

REQUIREMENT-1 CHECK at matmul scale: packed weight bytes < Marlin weight bytes
must hold for every g >= 256 config. g=128 exceeding Marlin (4.5 vs 4.125 bpw)
is EXPECTED — Finding K0-1: the existing S-A artifacts' codebook granularity
is too fine; g=128 is served as a correctness bootstrap only. It is reported
as expected-fail, never hidden.

Run:
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python /workspace/BiLLM2/kernels/bench/bench_dpk_gemv.py
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

from bench_utils import measure_peak, rel_err, time_kernel, weight_bytes  # noqa: E402
import dpk_ref as ref  # noqa: E402
from build import build_dpk  # noqa: E402
from gen_synthetic import gen_case  # noqa: E402

SEED = 0
DEV = torch.device("cuda:0")
# (K=C infeatures, N=R outfeatures) — same shapes as the INT4 baseline.
SHAPES = [(1024, 1024), (2048, 2048), (4096, 4096), (4096, 14336)]
GROUPS = [128, 512, "C"]
MARLIN_JSON = os.path.join(_THIS, "results_int4_baseline.json")
OUT_JSON = os.path.join(_THIS, "results_dpk_gemv.json")

ALLOC_GRAN = 512  # torch caching-allocator block granularity


def load_marlin_m1():
    with open(MARLIN_JSON) as f:
        data = json.load(f)
    out = {}
    for r in data["records"]:
        if r["M"] == 1:
            out[(r["K"], r["N"])] = r
    return out


def main():
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(DEV)
    cc = torch.cuda.get_device_capability(DEV)
    ext = build_dpk()
    marlin = load_marlin_m1()

    meta = {
        "kernel": "DPK W2A4 bucket-popcount GEMV v1 (kernels/cuda/dpk_gemv.cu)",
        "format": "DPK element mmode: b0/b1/m planes + s bitmap + bf16 cb[R,NG,3,4]",
        "activations": "A4 excess-8 nibbles, per-tensor fp32 scale (doc 02 par.4)",
        "M": 1,
        "gpu": gpu,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_contract": "kernels/bench/bench_utils.py",
        "comparator": "Marlin FP16xINT4 sym g=128 (patched), M=1 records from "
                      "results_int4_baseline.json",
        "req1_rule": "packed weight bytes < Marlin weight bytes for g >= 256; "
                     "g=128 expected-fail (Finding K0-1, S-A bootstrap config)",
    }
    print(f"# DPK GEMV v1 vs Marlin M=1 on {gpu} (SM{cc[0]}{cc[1]})")

    records, rows = [], []
    gates_pass = True
    req1_pass = True
    for (K, N) in SHAPES:
        for gspec in GROUPS:
            g = K if gspec == "C" else gspec
            case = gen_case(R=N, C_orig=K, g=g, seed=SEED, variant="realistic")
            c = {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                 for k, v in case.items()}
            b0, b1, m, s, cb, xhat, a_s = (c["b0"], c["b1"], c["m"], c["s"],
                                           c["cb"], c["xhat"], c["a_s"])

            # ---- correctness certification (gates before numbers) ----
            S_k, N_k = ext.dpk_gemv_debug(b0, b1, m, s, cb, xhat, a_s, g)
            S_r, N_r = ref.ref_bucket_sums(b0, b1, m, s, xhat, g)
            int_exact = torch.equal(S_k, S_r) and torch.equal(N_k, N_r)
            y_f32 = ext.dpk_gemv(b0, b1, m, s, cb, xhat, a_s, g, out_fp32=True)
            y_ref = ref.ref_gemv_bucket(b0, b1, m, s, cb, xhat, a_s, g)
            err = rel_err(y_f32, y_ref)
            y_bf = ext.dpk_gemv(b0, b1, m, s, cb, xhat, a_s, g)
            mism = (y_bf.view(torch.int16)
                    != y_ref.to(torch.bfloat16).view(torch.int16)).float().mean().item()
            det = all(torch.equal(ext.dpk_gemv(b0, b1, m, s, cb, xhat, a_s, g), y_bf)
                      for _ in range(4))
            cert = int_exact and err["rms_rel"] <= 1e-5 and mism <= 0.02 and det
            gates_pass &= cert
            del S_k, N_k, S_r, N_r, y_f32, y_ref, y_bf
            torch.cuda.empty_cache()

            # ---- weight footprint (actual resident tensor bytes) ----
            wbytes = weight_bytes(b0, b1, m, s, cb)
            bpw = wbytes * 8 / (K * N)
            mrec = marlin[(K, N)]
            mbytes = mrec["weight_bytes_total"]

            # ---- Requirement-1 verdict ----
            smaller = wbytes < mbytes
            if g >= 256:
                verdict = "PASS" if smaller else "FAIL"
                req1_pass &= smaller
            else:
                verdict = ("expected-fail (K0-1)" if not smaller
                           else "PASS (unexpected!)")

            # ---- peak memory + latency ----
            def call():
                return ext.dpk_gemv(b0, b1, m, s, cb, xhat, a_s, g)

            _, peak, base = measure_peak(call)
            out_bytes = 2 * N  # bf16 [N]
            exp_peak = ((out_bytes + ALLOC_GRAN - 1) // ALLOC_GRAN) * ALLOC_GRAN
            peak_ok = peak <= exp_peak
            gates_pass &= peak_ok  # no hidden workspaces allowed
            lat = time_kernel(call, iters=100, warmup=25)
            ratio = lat["median_ms"] / mrec["latency_ms"]["median_ms"]

            rec = {
                "K": K, "N": N, "g": g, "g_spec": str(gspec), "M": 1,
                "NG": c["cb"].shape[1],
                "artifacts": {
                    "b0/b1/m": {"shape": [N, K // 32], "dtype": "uint32",
                                "bytes_each": N * (K // 32) * 4},
                    "s": {"shape": [K // 32], "dtype": "uint32", "bytes": (K // 32) * 4},
                    "cb": {"shape": list(c["cb"].shape), "dtype": "bfloat16",
                           "bytes": c["cb"].nelement() * 2},
                },
                "weight_bytes_total": wbytes,
                "bpw": bpw,
                "correctness": {
                    "int_SN_exact": int_exact,
                    "f32_rms_rel": err["rms_rel"],
                    "f32_cos_sim": err["cos_sim"],
                    "bf16_mismatch_frac": mism,
                    "bitwise_deterministic": det,
                    "certified": cert,
                },
                "peak_mem_delta_bytes": peak,
                "expected_peak_bytes": exp_peak,
                "peak_is_output_only": peak_ok,
                "baseline_allocated_bytes": base,
                "latency_ms": {k: lat[k] for k in
                               ("mean_ms", "median_ms", "min_ms", "max_ms",
                                "p90_ms", "iters", "warmup")},
                "marlin_m1": {
                    "weight_bytes": mbytes,
                    "bpw": mrec["bpw"],
                    "peak_mem_delta_bytes": mrec["peak_mem_delta_bytes"],
                    "median_ms": mrec["latency_ms"]["median_ms"],
                },
                "weight_bytes_vs_marlin": wbytes / mbytes,
                "latency_vs_marlin": ratio,
                "req1_verdict": verdict,
            }
            records.append(rec)
            rows.append((f"{K}x{N}", g, f"{bpw:.4f}", wbytes, mbytes,
                         f"{wbytes / mbytes:.3f}", peak,
                         mrec["peak_mem_delta_bytes"],
                         f"{lat['median_ms']:.4f}",
                         f"{mrec['latency_ms']['median_ms']:.4f}",
                         f"{ratio:.2f}x",
                         "PASS" if cert else "FAIL", verdict))
            print(f"K={K:5d} N={N:5d} g={g:5d}: bpw={bpw:.4f} "
                  f"bytes={wbytes:>10,} (marlin {mbytes:>10,}, x{wbytes/mbytes:.3f}) "
                  f"peak={peak} B (out={exp_peak}, ok={peak_ok}) "
                  f"lat={lat['median_ms']:.4f} ms (marlin {mrec['latency_ms']['median_ms']:.4f}, "
                  f"{ratio:.2f}x) cert={'PASS' if cert else 'FAIL'} req1={verdict}")

            del c, b0, b1, m, s, cb, xhat, case
            torch.cuda.empty_cache()

    hdr = ("| shape KxN | g | bpw | weight bytes | marlin bytes | ratio | "
           "peak delta (B) | marlin peak (B) | dpk med ms | marlin med ms | "
           "lat ratio | correctness | Req-1 |")
    sep = "|" + "---|" * 13
    lines = [hdr, sep] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    table = "\n".join(lines)
    print("\n" + table)

    summary = {
        "all_correctness_certified": gates_pass,
        "req1_all_g_ge_256_pass": req1_pass,
        "g128_expected_fail_note": "g=128 (S-A artifacts) bpw 4.5 > 4.125: "
                                   "Finding K0-1, by design; bootstrap config only",
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "records": records, "summary": summary,
                   "markdown_table": table}, f, indent=2)
    print(f"\nJSON written to {OUT_JSON}")
    print(f"CORRECTNESS CERTIFICATION: {'PASS' if gates_pass else 'FAIL'}")
    print(f"REQ-1 (g>=256 weight bytes < Marlin): {'PASS' if req1_pass else 'FAIL'}")
    return 0 if (gates_pass and req1_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
