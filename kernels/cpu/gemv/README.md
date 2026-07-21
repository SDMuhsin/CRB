# DOML GEMV decode microkernel (P2b)

`y = W·x` with `W` in the DOML R-B resident layout (2.4992 bpw), consuming the
five planes (`b0` / packed `b1` / packed `m` / `s` / 10-slot fp8 `cb`)
directly. Two value paths:

* **fp** — container-exact: fp8→bf16 (exact widening) → fp32 FMA. Only the
  accumulation order differs from the reference decode (gate: max rel err
  ≤ 2e-5 vs fp64 reference).
* **i8** — per-(row,group) on-the-fly quantization of the ≤10 levels to
  u8 + fp32 scale, activations int8 per 256-group, `vpdpbusd`, per-(row,group)
  fixup. Carries level+activation rounding (~0.7% RMS class, reported by the
  gate with a breakdown).

Each path also has an `*_mf` variant (`--mfull`): the packed m plane is
expanded at pack/load time to the full R×C/8 bit plane (bit-identical to the
container m plane), which deletes the per-chunk m bit-stream walk from the
inner loop at +0.213 bpw of streamed weight bytes. Pure R-B (packed m) and
mf are both benchmarked; outputs are bitwise identical (gate G-MFULL).

## Build

```
cd /workspace/BiLLM2
make -C kernels/cpu          # builds tests/dpka_test, gemv/gemv_test, gemv/gemv_bench
```

## Tests (gates G-NUM-FP, G-NUM-I8, G-24T, G-MFULL, G-UNIQUE)

All gates run on REAL tensors from the DPKA artifact (layers 0/14/27 × all 5
shapes), no synthetic weights:

```
./kernels/cpu/gemv/gemv_test [downloads/cpu_kernel_rnd/qwen3-0.6b-k31.dpka]
# exit code 0 iff all gates pass; ~4 min (scalar fp64 references)
```

* `G-NUM-FP`: fp path vs (C reference R-B decode → fp64 dot), gate
  max|Δ|/rms(ref) ≤ 2e-5.
* `G-NUM-I8`: total rel err + breakdown (level-rounding only /
  activation-rounding only), each vs the exact fp64 reference.
* `G-24T`: 24-thread output bitwise equal to 1-thread.
* `G-MFULL`: mf variant bitwise equal to packed R-B.
* `G-UNIQUE`: zeroing the packed m plane, then the s bitmap, changes outputs
  (kernel provably consumes membership + salience planes).

## Microbenchmark (mirrors kernels/cpu/bench_ik protocol)

Cycled weight copies ≥ 384 MB (cap 512 copies), ≥9 reps, medians, per-rep CSV
rows, PLACEMENT numa_maps evidence, threads pinned t→CPU t (node0 = even,
node1 = odd), weights first-touched by the owning thread's row slice,
activations prepped once outside the timed region (same liberty bench_ik
takes for its Q8_K quantization), inline correctness check vs the C
reference before timing.

```
# single config
./kernels/cpu/gemv/gemv_bench --variant i8 --dout 2048 --din 1024 --threads 24
./kernels/cpu/gemv/gemv_bench --variant fp --mfull --dout 1024 --din 3072 --threads 1

# full sweep: {fp,i8} x {packed,mfull} x 5 shapes x threads {1,24}
./kernels/cpu/gemv/gemv_bench --sweep
```

Options: `--layer L` (default 0: which layer's real tensor to use),
`--reps R` (9), `--nbuf N` (auto ≥384 MB), `--target-ms M` (60), `--seed S`,
`--artifact PATH`.

Env knobs (bake-off; defaults are the shipped configuration):
`DOML_TILE_FP` / `DOML_TILE_I8` ∈ {1,2,4} — row-tile height (default 4).

### Citable-run protocol (numbers only from logs)

```
cd /workspace/BiLLM2
./kernels/cpu/roofline/hog 102400        # page-cache purge (~75 s) so first-touch places node-local
uptime                                    # 1-min load must be <= 1
pgrep -f 'bench_ik|python|hog'            # must be empty
nohup ./kernels/cpu/gemv/gemv_bench --sweep \
  > llmdocs/cpu_kernel_rnd/verify/p2b/sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Every log carries LOADAVG at start of each config (the bench prints a
NOT-citable warning if 1-min load > 1) and a PLACEMENT line after setup.

## Files

| file | role |
|---|---|
| `doml_gemv.h` | kernel + pack + thread-pool API |
| `doml_gemv.c` | slab packing (NUMA first-touch via `doml_gemv_pack_rows` from pinned threads), on-the-fly per-(row,group) table prep (vectorized), fp/i8 row-tile kernels, pinned spin-barrier pool |
| `gemv_test.c` | gate driver (see above) |
| `gemv_bench.c` | µbench main (bench_ik protocol) |
