# DOML prefill GEMM (P3) — slab-convert + VNNI

`C[y][r] = sum_j W[r,j] * X[y,j]` for ny=512 activation rows, with `W` in the
FROZEN v2 i8 fused row-slab (`kernels/cpu/gemv2/`, 2.6535 bpw resident —
reused via includes, not forked). This is ik_llama.cpp's own prefill duality
(decode reads the packed format, prefill converts to int8 panels + runs a
`vpdpbusd` GEMM) with a **rounding-free** conversion: the slab already stores
the int8 levels (u8 = q+128, pack-time, gate-checked), so the panel bytes are
bitwise the slab's cb-record levels selected by the decoded idx stream —
no second weight rounding (ik re-rounds IQ2_KL → Q8_K_R16 on the fly, ~0.45%
extra RMS in their shipped prefill numerics).

Pipeline per call (all inside the timed region):

1. **Activation quant** (y-sliced over threads): fp32 → s8 per 256-group,
   `id = 127/amax`, RNE, plus `dx` and `128*dx*bsum` fixup terms. Stored in
   the panel-matched k-step order — which is the NATURAL dword order of
   `packs_epi32`+`packs_epi16` (both are the same 4x4 dword transpose), so
   the quantizer needs no shuffle. One pool barrier.
2. **Panel conversion** (16-row-group slices, strip-blocked to stay in L2,
   default cap 256 KB/thread — env `DOML3_STRIP_KB`): the P2c idx decode
   (pdep b1/m expansion, merge-mask idx, cb record = the vpermb table), but
   the vpermb result is 4x16-dword-transposed and stored as Q8_K_R16-style
   16-row interleaved lines + fp32 scales. Transient WORKING MEMORY, never
   resident (bytes printed in every SETUP line).
3. **GEMM** per strip: micro-kernel = 4 row-groups x 4 columns (flagship,
   `--mk 1`, default) or 2 x 8 (`--mk 0`); per 4-weight k-step: NRB 64-B line loads +
   NY `vpbroadcastd` (load ports, not port 5) + NRB*NY `vpdpbusd`. Integer
   accumulation per group is exact; per-group fp fixup
   `C += sc_rg*(cvt(I)*dx_yg - 128*dx*bsum_yg)` with fp accumulators in an
   L1 stack tile. Outputs are bitwise identical across micro-kernels and
   thread counts.

## Build

```
cd /workspace/BiLLM2
make -C kernels/cpu     # adds gemm/{gemm_test,gemm_bench}; check make's exit code
```

## Gates

```
./kernels/cpu/gemm/gemm_test           # 15 real tensors x ny=512 (1t):
                                       #  G-NUM-P3 (total rms <= 1.2e-2 fixed
                                       #  pre-bench + level/act breakdown),
                                       #  G-XQ, G-MK, G-UNIQUE
./kernels/cpu/gemm/gemm_test --mt      # + 24t/48t full-call bitwise == 1t
                                       #   (free box only)
./kernels/cpu/gemm/gemm_test --derive  # G-DERIVE-P3: all 196 tensors, every
                                       #   panel byte == slab cb level at the
                                       #   spec position (independent walk)
./kernels/cpu/gemm/gemm_test --bpw     # G-BPW-P3: resident re-assert (2.6535)
                                       #   + transient bytes table
```

exit code 0 iff the selected gates pass. All gates use REAL tensors from
`downloads/cpu_kernel_rnd/qwen3-0.6b-k31.dpka`.

## Microbenchmark (bench_ik protocol at ny=512)

Cycled weight copies >= 384 MB, >= 9 reps, medians, per-rep CSV, PLACEMENT
evidence, pinned pool threads, pack first-touch by the CONVERT ownership
(16-row-group slices), inline fp64 correctness check. fp32 activations are
generated outside; the int8 activation quantization runs INSIDE the timed
region every call (bench_ik quantizes its Q8_K activations OUTSIDE its timed
region — the paired comparison is conservative against us by that cost; the
`--split` diagnostic reports the quant/convert/GEMM phase medians).

```
./kernels/cpu/gemm/gemm_bench --dout 2048 --din 1024 --ny 512 --threads 24
./kernels/cpu/gemm/gemm_bench --dout 2048 --din 1024 --ny 512 --threads 24 --split
./kernels/cpu/gemm/gemm_bench --sweep        # 5 shapes x ny=512 x {24,48}t
```

## Citable-run protocol (shared box — numbers only from logs)

Paired same-window rounds, alternating order, hog 102400 before first-touch
every round, evidence (uptime + 10-s busy% + pgrep) per round:

```
./kernels/cpu/gemm/run_p3_checkpoint.sh  # brief's mid-build checkpoint:
                                         # 2048x1024 ny=512 24t vs bench_ik
                                         # iq2_kl (+ --split diagnostic)
./kernels/cpu/gemm/run_p3_paired.sh      # full G-SPEED-P3 matrix: 5 shapes x
                                         # {24,48}t vs {iq2_kl, q8_k_r16}
python3 kernels/cpu/gemm/parse_p3.py <log>
```

Logs land in `llmdocs/cpu_kernel_rnd/verify/p3/`. SMOKE numbers are never
citable.

## Files

| file | role |
|---|---|
| `doml_gemm.h` | panel/activation layout spec + API |
| `doml_gemm.c` | activation quantizer, slab->panel converter, VNNI micro-kernels, threaded call glue |
| `gemm_test.c` | gate driver (default / `--mt` / `--derive` / `--bpw`) |
| `gemm_bench.c` | µbench (`--split` = phase diagnostic) |
| `run_p3_checkpoint.sh` | paired mid-build checkpoint runner |
| `run_p3_paired.sh` | paired full-matrix runner |
| `parse_p3.py` | log -> median-of-round-medians tables + within-round ratios |
