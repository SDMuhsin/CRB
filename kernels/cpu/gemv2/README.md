# DOML GEMV v2 (P2c) — fused row-slab resident + sync redesign

`y = W·x` with `W` in the v2 FUSED-SLAB resident: per (4-row tile, 256-column
group) every byte the kernel consumes is contiguous in consumption order —

```
block := [hdr: u8 b1len[4] | cb: 4 x rec | b0: 4 x 32 B | m: 4 x mlen_g | b1: var]
rec    = 14 B (i8 slab: 12 u8 levels q+128 in container 12-slot order + bf16 scale)
       = 12 B (fp slab: the container 12-slot fp8 table, bit-identical to DPK cb)
```

so each thread's tile range is ONE contiguous forward stream (per-tile u32
offsets give random access for stealing / first-touch). This deletes P2b's
per-row b1 popcount-offset table, the 4-disjoint-plane prefetch restarts, and
the entire on-the-fly i8 table prep (the i8 record IS the vpermb table — one
unaligned 16 B broadcast load; slots 12..15 land in the next record and are
never indexed because idx = b0|b1<<1|m<<2|s<<3 = part*4+code ∈ [0,12)).

Everything is losslessly derived from the frozen DPKA artifact (G-DERIVE);
`kernels/cpu/gemv/` (P2b), `fmt/`, `ref/` are untouched.

Value paths: **fp** (container-exact bf16 levels, fp32 FMA, gate ≤2e-5) and
**i8** (levels quantized at PACK time to u8+bf16 scale — deterministic:
scale = bf16-RNE(max|level|/127), q = clamp(rint(level/scale)); activations
int8 per 256-group; `vpdpbusd`; ~0.7% RMS class, reported with breakdown).
`--mfull` ablation stores the full m plane (32 B/row/group) — OVER the 2.70
bpw budget (2.86), measured as ablation only.

Threading: pinned pool (thread t → CPU t; node0 = even, node1 = odd; env
`DOML2_PIN=cpu|node|none` -- `node` widens each thread's mask to its whole
NUMA node keeping first-touch locality, `none` unpins entirely; P2d bake-off:
cpu == node at 24t, both degrade at 48t vs cpu, none is always worst --
default `cpu`), radix-2/4 dissemination barrier (env `DOML2_RADIX`, default 2), every wait =
bounded spin (`DOML2_SPIN0` plain loads + `DOML2_SPIN` pauses, defaults
4096/2048) then TIMED futex sleep (`DOML2_FUTEX_US`, default 50); signals are
plain release stores unless a sleeper registered (sleep bit → xchg +
`FUTEX_WAKE`). ONE barrier per GEMV call (bench_ik pays one omp barrier).
The spin budget must exceed barrier RTT + wake latency so the pool
re-converges to pure spin after any sleep — see comments in `doml_gemv2.c`
for the two measured failure modes (syscall storm / sleep cascade).
Work distribution: static contiguous tile slices (flagship) or chunked
work-stealing (`--steal [--chunk N]`, atomic per-thread cursors,
double-buffered by call parity, same-NUMA victims first).

## Resident budget (G-BPW-V2, hard ≤2.70 bpw model-aggregate)

i8-slab (flagship) ≈ **2.66 bpw** aggregate counting every allocated byte
(headers, byte-align pads, 64 B slab align, 512 B tail slack, shared tile
offsets); fp-slab ≈ 2.60. Exact per-component table: `gemv2_test --bpw`.

## Build

```
cd /workspace/BiLLM2
make -C kernels/cpu     # builds P1 + P2b + gemv2/{gemv2_test,gemv2_bench}
```

## Gates

```
./kernels/cpu/gemv2/gemv2_test             # 15 real tensors: G-NUM-FP (<=2e-5),
                                           # G-NUM-I8 (+level/act breakdown),
                                           # G-24T, G-48T, G-STEAL, G-MF, G-UNIQUE
./kernels/cpu/gemv2/gemv2_test --derive    # G-DERIVE, all 196 tensors:
                                           # fp slab bitwise decode-equality vs P1
                                           # reference; i8 slab plane round-trip +
                                           # cb records vs independent re-derivation
./kernels/cpu/gemv2/gemv2_test --bpw       # G-BPW-V2 component table + <=2.70 check
```

exit code 0 iff the selected gates pass. All gates use REAL tensors from
`downloads/cpu_kernel_rnd/qwen3-0.6b-k31.dpka`.

## Microbenchmark (bench_ik protocol)

Cycled weight copies ≥384 MB, ≥9 reps, medians, per-rep CSV, PLACEMENT
numa_maps evidence, pinned threads, per-thread-range first-touch (copy 0
derived by owners, other copies owner-memcpy'd), activations prepped outside
the timed region (bench_ik liberty), inline correctness check vs the C
reference, ONE barrier per timed call.

```
./kernels/cpu/gemv2/gemv2_bench --variant i8 --dout 2048 --din 1024 --threads 24
./kernels/cpu/gemv2/gemv2_bench --variant i8 --steal --chunk 8 --dout 2048 --din 1024 --threads 24
./kernels/cpu/gemv2/gemv2_bench --sweep            # {i8,fp} x 5 shapes x {1,24}t
./kernels/cpu/gemv2/gemv2_bench --curve            # i8 2048x1024 x {1,6,12,24,48}t
./kernels/cpu/gemv2/gemv2_bench --barrier-bench    # null-kernel barrier ns:
                                                   # v2 (futex) vs v1 (P2b spin),
                                                   # {6,12,24,48}t
```

### Citable-run protocol (shared box — numbers only from logs)

Paired same-window rounds, alternating order, hog before first-touch,
evidence (uptime + 10-s busy% + pgrep) per round:

```
./kernels/cpu/gemv2/run_checkpoint.sh      # brief's mid-build checkpoint
                                           # (2048x1024, {1,24}t vs bench_ik iq2_kl)
./kernels/cpu/gemv2/run_paired_sweeps.sh   # full 5-shape paired sweep + curve
```

Logs land in `llmdocs/cpu_kernel_rnd/verify/p2c/`.

P2d (see P2C_REPORT.md addendum): v2 microbench numbers are sensitive to
AGED physical-page state (ratchets when the box goes un-hogged; up to +93%
at 6t, 0% at 1t; ik immune; `hog 102400` restores). Always run speed
numbers through the hog-per-round runners above; never quote direct
one-off invocations. `run_p2d_pins.sh` re-runs the pin-mode bake-off.

## Files

| file | role |
|---|---|
| `doml_gemv2.h` | format, packer, kernel, pool, stealing API |
| `doml_gemv2.c` | slab size/fill passes (scalar, gate-checked), fused-block fp/i8 kernels (chunk-unrolled, absolute b1 bit-cursors, merge-mask idx), pool (dissemination + bounded-spin/timed-futex), stealing |
| `gemv2_test.c` | gate driver (default / `--derive` / `--bpw`) |
| `gemv2_bench.c` | µbench + `--barrier-bench` |
| `run_checkpoint.sh` | paired mid-build checkpoint runner |
| `run_paired_sweeps.sh` | paired full-sweep runner (adapted from P2b) |
