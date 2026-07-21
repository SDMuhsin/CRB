# bench_ik P2a validation — 2026-07-15

Harness: `kernels/cpu/bench_ik/bench_ik.cpp` linked against
`temp/ik_llama.cpp/build/ggml/src/libggml.so` (commit 7937465ff15a, CPU-only
Release, `-DGGML_CUDA=OFF -DGGML_NATIVE=ON`, built 2026-07-14; verified no
source file newer than the lib). Build: `make` in this directory — one
warning-free g++ invocation, see README for the exact link line.

Raw logs: `llmdocs/cpu_kernel_rnd/verify/p2a/validate_20260715_195257.log`,
`llmdocs/cpu_kernel_rnd/verify/p2a/smoke_bench_20260715_195409.log`.

## 1. Correctness (`./bench_ik --validate`) — ALL 24 CASES PASS

Reference = ik's own `dequantize_row_{iq2_kl,q2_K,q2_k_r4,q8_k_r16}` x
dequantized Q8_K activations, double accumulation. Errors normalized by
reference RMS (elementwise rel is meaningless near ref~0).

Path routing sanity (via exported `iqk_dequant_type`):
IQ2_KL@ny512 -> 397 (Q8_K_R16), Q2_K@ny512 -> 397, Q2_K_R4@ny512 -> itself.
Matches IK_KERNEL_ANATOMY.md §2.2/§4.3.

| type | path | cases | max_err/ref_rms | rms_rel |
|---|---|---|---|---|
| iq2_kl | direct (ny=1,16) | 3 | 7.1e-07 .. 9.3e-07 | ~1.6e-07 |
| iq2_kl | convert (ny=512) | 3 | 2.2e-02 .. 2.4e-02 | ~4.5e-03 |
| q2_k | direct (ny=1,16) | 3 | 5.6e-07 .. 7.4e-07 | ~1.4e-07 |
| q2_k | convert (ny=512) | 3 | 1.9e-02 .. 2.1e-02 | ~4.0e-03 |
| q2_k_r4 | direct (all ny) | 6 | 3.9e-07 .. 1.5e-06 | ~1.2e-07 |
| q8_k_r16 | direct (all ny) | 6 | 3.8e-07 .. 1.0e-06 | ~1.1e-07 |

Cases per type: (2048x1024) x ny{1,16,512} x nth{1,24} + (1024x3072) ny=512
nth=24. Interpretation:

- **Direct paths agree with the dequantized-weight reference at fp32
  accumulation level (~1e-7 RMS)** — the kernels' integer arithmetic is exact
  on the same values; plumbing (strides, row-interleave, threading, C layout)
  is correct.
- **Convert paths (iq2_kl/q2_k at ny>=32) show ~0.45% RMS** — this is the
  engine's deliberate second int8 rounding when it re-quantizes weight panels
  to Q8_K_R16 for the VNNI GEMM (anatomy §4.3/§7.4), not a harness error.
  It is the format's true inference behavior.

Quantized row sizes validated against `ggml_row_size` (asserted == quantize
return): iq2_kl 346 B @K=1024 (2.7031 bpw), q2_k/q2_k_r4 336 B (2.6250),
q8_k_r16 1032 B (8.0625), Q8_K activations 1184 B/row @K=1024.

## 2. API findings

- **No imatrix required**: `quantize_iq2_kl(src, dst, nrows, n_per_row, NULL,
  NULL)` works — `QHelper::row_weights` returns NULL when imatrix is NULL and
  the impl falls back to sigma-based weighting. Same for the other three.
  (We did NOT need the uniform-imatrix fallback.)
- **No init required**: no `ggml_quantize_init`-style call exists for iqk
  types; all LUTs are static/ctor-built. `GGML_FP16_TO_FP32` is hardware
  (F16C) on native builds. Harness calls `ggml_init` once defensively anyway.
- **Q8_K_R16 as typeA works directly from `quantize_q8_k_r16`** (fp32 ->
  interleaved r16 in one call, `nrows%16==0`); no separate repack step needed.
  `MulMat::prepare` accepts it for all Ny (this isolates the raw VNNI GEMM
  without the IQ2_KL->r16 conversion cost).
- `iqk_mul_mat` interleaved-type strides are *per nominal row*
  (`ggml_row_size(type, ne00)`), with the engine internally addressing 4/16-row
  groups; the harness's first-touch slicing replicates
  `MulMat::num_rows` + the contiguous-slice split of `iqk_mul_mat.cpp:501-610`.
- `iqk_dequant_type` is exported — used to detect the convert path at runtime
  instead of hardcoding the Ny>=32 table.

## 3. Smoke bench — **SMOKE ONLY, NON-CITABLE**

Box was NOT idle (load avg 9.8, concurrent build agent). Purpose: prove the
timing plumbing, calibration, buffer rotation and placement evidence work.
2048x1024, reps=5, auto nbuf (>=384 MB weight working set):

| config | median | weight BW | GMAC/s | placement |
|---|---|---|---|---|
| iq2_kl ny=1 nth=1 | 138.3 us/call | 5.12 GB/s | 15.2 | 100% node0 |
| q2_k_r4 ny=1 nth=1 | 79.1 us/call | 8.70 GB/s | 26.5 | 100% node0 |
| q2_k ny=1 nth=1 | 82.4 us/call | 8.35 GB/s | 25.5 | 100% node0 |
| q8_k_r16 ny=512 nth=1 | 9.78 ms/call | — | 109.8 | 100% node0 |
| iq2_kl ny=512 nth=24 | 646 us/call | — | 1662 | 50.8% node0 |

Even these smoke numbers are direction-consistent with the anatomy doc:
q2_k_r4 GEMV ~1.75x faster than iq2_kl GEMV single-core (unpack-dominated
regime), r16 GEMM ~110 GMAC/s/core, and 24T placement splits ~50/50 across
nodes as designed. **Do not cite; rerun on the idle box.**

## 4. Idle-box baseline sweep (for the PI to trigger)

```bash
cd /workspace/BiLLM2/kernels/cpu/roofline && ./hog 102400   # free page cache first
cd /workspace/BiLLM2/kernels/cpu/bench_ik && \
  ./bench_ik --sweep --reps 9 2>&1 | tee /workspace/BiLLM2/llmdocs/cpu_kernel_rnd/verify/p2a/sweep_$(date +%Y%m%d_%H%M%S).log
```

80 configs (4 types x 5 unique shapes x ny{1,512} x threads{1,24}), ~10-20 min.
Check `PLACEMENT` lines (expect ~100%/~50% node0) and 1-min load <= 1 before
starting.
