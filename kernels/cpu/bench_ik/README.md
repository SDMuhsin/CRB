# bench_ik — competitor-kernel microbenchmark harness (ik_llama.cpp iqk_mul_mat)

Standalone, placement-fair microbenchmark for the ik_llama.cpp CPU matmul engine,
so future DOML kernels can be compared per-shape against the strongest existing
CPU kernels. Drives the single exported entry point

```c
bool iqk_mul_mat(long Nx, long Ny, long ne00,
                 int typeA, const void *A, long strideA,
                 int typeB, const void *B, long strideB,
                 float *C, long stride_C, int ith, int nth);
```

exactly the way `ggml_compute_forward_mul_mat` does (every thread calls it with
its `(ith, nth)`; one barrier per call group). See
`llmdocs/cpu_kernel_rnd/IK_KERNEL_ANATOMY.md` §6 for the extraction plan this
implements, and `kernels/cpu/roofline/README.md` for the placement discipline
it follows.

## Weight types supported end-to-end

| `--type` | ggml type (enum) | bpw @ K=1024 | Ny=1 path | Ny=512 path |
|---|---|---|---|---|
| `iq2_kl` | `GGML_TYPE_IQ2_KL` (157) | 2.7031 | direct GEMV (LUT unpack) | on-the-fly convert to Q8_K_R16 + VNNI GEMM |
| `q2_k` | `GGML_TYPE_Q2_K` (10) | 2.6250 | direct GEMV | on-the-fly convert to Q8_K_R16 + VNNI GEMM |
| `q2_k_r4` | `GGML_TYPE_Q2_K_R4` (210) | 2.6250 | direct 4-row-interleaved GEMV | direct (no conversion; R4 kernels serve all Ny) |
| `q8_k_r16` | `GGML_TYPE_Q8_K_R16` (397) | 8.0625 | direct | direct — this is the *raw* r16 VNNI GEMM, i.e. the Ny>=32 IQ2_KL/Q2_K pipeline **without** the conversion cost |

- Activation type is `GGML_TYPE_Q8_K` (15) for **all four** weight types
  (verified against `expected_type_B` in `iqk_gemm_kquants.cpp` /
  `iqk_gemm_iqk_quants.cpp`). Activations are quantized once with
  `iqk_quantize_row_q8_K` **outside the timed region**;
  `strideB = ggml_row_size(GGML_TYPE_Q8_K, ne00)`.
- Weights: `quantize_iq2_kl` / `quantize_q2_K` / `quantize_q2_k_r4` /
  `quantize_q8_k_r16` with `imatrix = NULL` (allowed: ik's `QHelper` then
  passes NULL weights through and the per-row impl falls back to its
  sigma-based weighting; quality is irrelevant for speed benching, format and
  sizes are exact). Returned byte counts are asserted against
  `nrows * ggml_row_size(type, ne00)`.
- `Q2_K_R4` requires `dout % 4 == 0`, `Q8_K_R16` requires `dout % 16 == 0`,
  every type requires `din % 256 == 0` (all Qwen3-0.6B shapes qualify).
- No init calls are required for the iqk entry points (all LUTs are static or
  built per call). `ggml_init` is still called once at startup, defensively.

## Build

Requires the CPU-only Release build of `temp/ik_llama.cpp` @ 7937465ff15a
(`cmake -B build -DGGML_CUDA=OFF -DGGML_NATIVE=ON`, as
`cpu_baselines/scripts/build_ikllamacpp.sh` does — `libggml.so` must exist
under `temp/ik_llama.cpp/build/ggml/src/`). Then:

```bash
cd kernels/cpu/bench_ik && make
```

Exact link line (from the Makefile):

```
g++ -O2 -g -march=native -std=c++17 -fopenmp \
    -I../../../temp/ik_llama.cpp/ggml/include -I../../../temp/ik_llama.cpp/ggml/src \
    -o bench_ik bench_ik.cpp \
    -L../../../temp/ik_llama.cpp/build/ggml/src -lggml \
    -Wl,-rpath,/workspace/BiLLM2/temp/ik_llama.cpp/build/ggml/src -fopenmp -pthread
```

All symbols (`iqk_mul_mat`, `iqk_dequant_type`, `quantize_*`,
`dequantize_row_*`, `iqk_quantize_row_q8_K`, `ggml_row_size`) come from the
shared `libggml.so`.

## Run

```bash
# correctness matrix (all 4 types x ny {1,16,512} x threads {1,24}); exit != 0 on FAIL
./bench_ik --validate

# single config: correctness check and/or bench
./bench_ik --type iq2_kl --dout 2048 --din 1024 --ny 1 --threads 24 --check
./bench_ik --type q2_k_r4 --dout 2048 --din 1024 --ny 1 --threads 24 --bench --reps 9

# FULL BASELINE SWEEP (idle box only!): all Qwen3-0.6B shapes x 4 types x
# ny {1,512} x threads {1,24}; CSV on stdout, SUMMARY lines on stderr
./hog_first   # <- run kernels/cpu/roofline/hog 102400 first (see below)
./bench_ik --sweep --reps 9 2>&1 | tee sweep.log
```

Options: `--reps R` (timed repetitions, default 9), `--target-ms M` (minimum
rep duration, default 60), `--nbuf N` (weight copies to cycle; default auto =
enough copies for a >=384 MB working set so decode streams from DRAM like a
real 28-layer model instead of re-reading an L3-resident matrix), `--seed S`.

Qwen3-0.6B shapes covered by `--sweep` (7 matmuls, 5 unique shapes):
2048x1024 (q), 1024x1024 (k,v x2), 1024x2048 (o), 3072x1024 (gate,up x2),
1024x3072 (down).

## Placement discipline (matches kernels/cpu/roofline)

- Thread `t` is pinned to CPU `t`: CPUs 0..23 are the 24 distinct physical
  cores alternating node0 (even) / node1 (odd). `--threads 1` = CPU 0 = node0.
- The quantized weight slab is **first-touched by the exact thread that reads
  it inside `iqk_mul_mat`**: the harness replicates the engine's contiguous
  row-slice split, including the Ny>=32 convert path whose slicing uses
  16-row groups (`iqk_mul_mat.cpp:501-610`, `MulMat::num_rows`). Weights are
  therefore split across both nodes the way the threads will read them.
- Activation rows are first-touched striped (`row % nth`), mimicking ggml's
  cooperative activation quantization.
- Placement is *verified* per run via `/proc/self/numa_maps`
  (`PLACEMENT ... node0=... node1=...` stderr line, same method as
  `roofline/stream_read`). Expect ~100% node0 at 1 thread, ~50/50 at 24.
- `numactl --membind` is EPERM in this container; before any *citable* run,
  free the page cache with `kernels/cpu/roofline/hog 102400` or first-touch
  can silently land on the wrong node — then confirm via the PLACEMENT line.

## Metrics

Per rep (CSV): `secs`, `ns_call` (per call-group, includes the one omp
barrier that ggml also pays per node), `weight_GBps` (= Nx*strideA bytes of
*quantized weights* streamed per call / time — the decode-relevant number),
`GMACs` (= Nx*ne00*Ny MAC / time — the prefill-relevant number). SUMMARY line
reports median [min,max] over reps and calls/s ("GEMV/s" at ny=1).

## Correctness model (what --validate proves)

Reference = weights dequantized by ik's **own** `dequantize_row_*` x Q8_K
activations dequantized as `d*qs`, accumulated in double. This validates
harness plumbing (strides, types, threading), not kernel quality.

- Direct paths (ny=1/16 for all; all ny for `q2_k_r4`/`q8_k_r16`): the kernel
  does exact integer arithmetic on those very values -> observed max err
  ~1e-6 of ref RMS (pure fp32 accumulation-order noise).
- Convert paths (`iq2_kl`/`q2_k` at ny>=32): the engine re-quantizes weights
  to Q8_K_R16 on the fly (a second int8 rounding, by design — anatomy doc
  §4.3) -> observed ~0.45% RMS rel err. Expected, and matches the format's
  real inference behavior.

## Caveats

- rep 0 occasionally runs slow on a loaded box (frequency ramp); the citable
  statistic is the median. Do not cite anything measured with load avg > ~1.
- `stride_C = Nx` (contiguous fp32 output rows), C first-touch is approximate
  at page granularity (C is small and write-mostly).
- The harness measures the mul_mat op only: real end-to-end decode also pays
  activation quantization (excluded here by design) and ~2 spin barriers per
  node (1 is included per call).
