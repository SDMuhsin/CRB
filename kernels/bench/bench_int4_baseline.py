"""INT4 GEMM baseline benchmark for the DOML CUDA kernel project.

Official strong INT4 comparator: Marlin (IST-DASLab/marlin @ 1f25790),
FP16xINT4 tensor-core kernel, symmetric quantization, groupsize=128, SM80+
(A40=SM86), built with ONE local patch (kernels/third_party/
marlin_racefix.patch): the cp.async fetch in global_reduce is replaced by
ld.volatile.global.v4 loads. The unpatched kernel has a measured data race
(nondeterministic outputs, rms_rel up to 3e-2) on A40/CUDA-12.5 for M >= 512
whenever column slices are k-split across threadblocks; see
llmdocs/cuda_kernel/baselines/B1_int4_baseline.md. Patched source is vendored
at kernels/third_party/marlin/. The patch changes memory-ordering only, not
math; latency is at parity with the unpatched build.

For each Qwen3-relevant GEMM shape (K, N) and batch M this script:
  1. Builds real Marlin INT4 weight artifacts (packed int4 weights + fp16
     group scales) from a random fp16 matrix, and records their EXACT
     storage bytes -> implied bits-per-weight (expected 4.125 for sym g=128).
  2. Correctness: compares the Marlin kernel output against a float32
     reference GEMM of the DEQUANTIZED int4 weights (i.e. the reference is
     exactly what the kernel is supposed to compute, not the original
     unquantized weights). Reports max/mean/rms relative error + cosine sim.
  3. Measures peak CUDA memory delta of the matmul call (measure_peak),
     CUDA-event latency (time_kernel), and weight storage bytes
     (weight_bytes) -- all via the project-wide contract in bench_utils.py.

Also measures a plain fp16 cuBLAS GEMM (torch.matmul on the dequantized
weights) on the same shapes as context, NOT as the baseline.

Outputs: markdown table on stdout + raw JSON in
kernels/bench/results_int4_baseline.json.

Run (non-interactive):
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python /workspace/BiLLM2/kernels/bench/bench_int4_baseline.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import measure_peak, rel_err, time_kernel, weight_bytes

import marlin  # noqa: E402  (built from IST-DASLab/marlin @ 1f25790)

SEED = 0
DEV = torch.device("cuda:0")
GROUPSIZE = 128
# Qwen3-relevant GEMM shapes (K = infeatures, N = outfeatures).
# Marlin constraints: K % 128 == 0, N % 256 == 0 -- all satisfied.
SHAPES = [(1024, 1024), (2048, 2048), (4096, 4096), (4096, 14336)]
MS = [1, 16, 2048]
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_int4_baseline.json")

# Correctness gates (vs float32 reference of the dequantized weights; the
# kernel computes in fp16/fp32-accum and stores fp16 outputs, so ~1e-3
# relative error is the expected floor).
GATE_RMS_REL = 5e-3
GATE_COS = 0.9999
# Determinism gate: N_DET repeated calls on identical inputs must be bitwise
# identical. This is what catches the upstream global_reduce race (the racy
# build fails this at M >= 512 on shapes where column slices are k-split).
N_DET = 5


def build_marlin_layer(k: int, n: int, groupsize: int):
    """Quantize a random fp16 (k, n) weight to sym INT4 g=`groupsize` and pack
    into a real marlin.Layer. Returns (layer, ref) where `ref` is the exact
    fp16 dequantized weight matrix the kernel should reproduce.

    Quantization recipe identical to gen_quant4 in marlin/test.py
    (symmetric max-abs per group, 16 levels, zero-point 8).
    """
    maxq = 2 ** 4 - 1
    w = torch.randn((k, n), dtype=torch.half, device=DEV)
    # per-group max-abs scales
    wg = w.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
    s = torch.max(torch.abs(wg), 0, keepdim=True)[0]
    s *= 2 / maxq
    q = torch.round(wg / s).int() + (maxq + 1) // 2
    q = torch.clamp(q, 0, maxq)
    ref = (q - (maxq + 1) // 2).half() * s
    ref = ref.reshape((groupsize, -1, n)).permute(1, 0, 2).reshape((k, n)).contiguous()
    s = s.reshape((-1, n)).contiguous()  # (k // groupsize, n)

    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t().contiguous()  # nn.Linear stores (out, in)
    layer = marlin.Layer(k, n, groupsize=groupsize)
    layer = layer.to(DEV)
    layer.pack(linear, s.t())  # pack() expects scales transposed
    return layer, ref


def main():
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(DEV)
    cc = torch.cuda.get_device_capability(DEV)

    meta = {
        "baseline_kernel": "Marlin FP16xINT4 (sym, groupsize=128) + race-fix patch",
        "marlin_repo": "https://github.com/IST-DASLab/marlin",
        "marlin_commit": "1f25790bdd49fba53106164a24666dade68d7c90",
        "local_patch": "kernels/third_party/marlin_racefix.patch "
                       "(global_reduce: cp.async -> ld.volatile.global.v4; "
                       "fixes measured nondeterminism at M>=512 on A40/CUDA 12.5)",
        "vendored_source": "kernels/third_party/marlin/",
        "gpu": gpu,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": SEED,
        "groupsize": GROUPSIZE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_contract": "kernels/bench/bench_utils.py",
        "reference": "float32 GEMM of the dequantized INT4 weights",
    }
    print(f"# INT4 baseline: Marlin on {gpu} (SM{cc[0]}{cc[1]}), torch {torch.__version__}")

    records = []
    rows = []
    all_pass = True
    for (K, N) in SHAPES:
        layer, ref_w = build_marlin_layer(K, N, GROUPSIZE)
        B, s, ws = layer.B, layer.s, layer.workspace
        wbytes = weight_bytes(B, s)
        bpw = wbytes * 8 / (K * N)
        shape_info = {
            "K": K, "N": N, "groupsize": GROUPSIZE,
            "packed_B": {"shape": list(B.shape), "dtype": str(B.dtype),
                         "bytes": B.nelement() * B.element_size()},
            "scales_s": {"shape": list(s.shape), "dtype": str(s.dtype),
                         "bytes": s.nelement() * s.element_size()},
            "workspace": {"shape": list(ws.shape), "dtype": str(ws.dtype),
                          "bytes": ws.nelement() * ws.element_size()},
            "weight_bytes_total": wbytes,  # B + s (persistent weight artifacts)
            "bpw": bpw,
            "fp16_weight_bytes": K * N * 2,
        }
        print(f"\n## K={K} N={N}: weight bytes={wbytes} "
              f"(B {tuple(B.shape)} {B.dtype} = {B.nelement()*B.element_size()} B, "
              f"s {tuple(s.shape)} {s.dtype} = {s.nelement()*s.element_size()} B) "
              f"-> bpw={bpw:.4f} | workspace {ws.nelement()*ws.element_size()} B")

        for M in MS:
            A = torch.randn((M, K), dtype=torch.half, device=DEV)
            # float32 reference of what the kernel SHOULD compute
            C_ref32 = torch.matmul(A.float(), ref_w.float())

            def marlin_call():
                # Mirrors marlin.Layer.forward: allocate output inside the
                # measured call so the peak delta includes it (same contract
                # every kernel in this project is held to).
                C = torch.empty((M, N), dtype=torch.half, device=DEV)
                marlin.mul(A, B, C, s, ws)
                return C

            # correctness first
            C = marlin_call()
            torch.cuda.synchronize()
            err = rel_err(C, C_ref32.half())
            # determinism: repeated calls must be bitwise identical
            det_diff = 0.0
            for _ in range(N_DET - 1):
                C2 = marlin_call()
                torch.cuda.synchronize()
                det_diff = max(det_diff, (C - C2).abs().max().item())
                del C2
            deterministic = det_diff == 0.0
            passed = (err["rms_rel"] < GATE_RMS_REL and err["cos_sim"] > GATE_COS
                      and deterministic)
            all_pass &= passed

            # fp16 cuBLAS context numbers (NOT the baseline; dequant GEMM)
            def fp16_call():
                return torch.matmul(A, ref_w)

            del C, C_ref32
            torch.cuda.empty_cache()

            _, peak, base = measure_peak(marlin_call)
            lat = time_kernel(marlin_call, iters=100, warmup=25)
            _, peak_fp16, _ = measure_peak(fp16_call)
            lat_fp16 = time_kernel(fp16_call, iters=100, warmup=25)

            rec = {
                **{k2: shape_info[k2] for k2 in ("K", "N", "groupsize", "weight_bytes_total", "bpw")},
                "artifacts": {kk: shape_info[kk] for kk in ("packed_B", "scales_s", "workspace")},
                "M": M,
                "correctness": {**err, "pass": passed,
                                "determinism_max_abs_diff": det_diff,
                                "determinism_reps": N_DET,
                                "gate": {"rms_rel<": GATE_RMS_REL, "cos_sim>": GATE_COS,
                                         "bitwise_deterministic": True}},
                "peak_mem_delta_bytes": peak,
                "baseline_allocated_bytes": base,
                "latency_ms": {k2: lat[k2] for k2 in ("mean_ms", "median_ms", "min_ms", "max_ms", "p90_ms", "iters", "warmup")},
                "fp16_context": {
                    "peak_mem_delta_bytes": peak_fp16,
                    "median_ms": lat_fp16["median_ms"],
                    "note": "torch.matmul(A, dequantized fp16 W); context only, not the baseline",
                },
            }
            records.append(rec)
            rows.append((f"{K}x{N}", M, wbytes, f"{bpw:.4f}", peak,
                         f"{lat['median_ms']:.4f}", f"{lat_fp16['median_ms']:.4f}",
                         f"{err['max_rel']:.3g}", f"{err['mean_rel']:.3g}",
                         f"{err['rms_rel']:.2e}", f"{err['cos_sim']:.6f}",
                         "yes" if deterministic else f"NO ({det_diff:g})",
                         "PASS" if passed else "FAIL"))
            print(f"  M={M:5d}  peak_delta={peak:>12,} B  marlin={lat['median_ms']:.4f} ms  "
                  f"fp16GEMM={lat_fp16['median_ms']:.4f} ms  rms_rel={err['rms_rel']:.2e}  "
                  f"cos={err['cos_sim']:.6f}  det={'yes' if deterministic else det_diff}  "
                  f"[{'PASS' if passed else 'FAIL'}]")
            del A
        del layer, ref_w, B, s, ws
        torch.cuda.empty_cache()

    # markdown table
    hdr = ("| shape KxN | M | weight bytes | bpw | peak delta (B) | marlin med ms | "
           "fp16 GEMM med ms | max rel | mean rel | rms rel | cos sim | bitwise det | gate |")
    sep = "|" + "---|" * 13
    lines = [hdr, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "records": records, "markdown_table": table}, f, indent=2)
    print(f"\nJSON written to {OUT_JSON}")
    print(f"ALL CORRECTNESS GATES: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
