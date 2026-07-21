# CPU roofline microbenchmarks (dual Xeon Silver 4310, Ice Lake-SP)

Purpose: measure the machine's actual limits to set physics-based expectations for
(a) a weight-streaming GEMV decode kernel at ~2.2-3.4 bits/weight and
(b) an int8 VNNI GEMM prefill kernel.
Report: `llmdocs/cpu_kernel_rnd/ROOFLINE.md`. Raw logs: `llmdocs/cpu_kernel_rnd/verify/roofline/`.

## Box (verified via lscpu / numactl / sysfs)

- 2 x Xeon Silver 4310: 12 physical cores/socket, SMT2 -> 48 CPUs.
  CPU numbering: node0 = even CPUs, node1 = odd; SMT sibling of CPU n is CPU n+24
  (so CPUs 0-23 are 24 distinct physical cores alternating between sockets).
- L1d 48 KB/core, L2 1.25 MB/core, L3 18 MB/socket. Base 2.1 GHz, max turbo 3.3 GHz.
- AVX-512 F/BW/DQ/VL/CD/IFMA/VNNI/VBMI/VBMI2/BITALG/VPOPCNTDQ. No AMX, no AVX512-BF16.
- Container quirks: `numactl --membind` -> EPERM; `/proc/sys/vm/drop_caches` read-only;
  no `perf`. NUMA placement is done by first-touch from pinned threads and *verified*
  per-run via `/proc/self/numa_maps` (the `PLACEMENT` stderr line).

## Binaries (build: `make`, flags `-O3 -march=icelake-server -mprefer-vector-width=512`)

| binary | measures |
|---|---|
| `stream_read` | multithreaded DRAM streaming READ (AVX-512 loads + accumulate). Modes: `a` all-node0, `b` split node-local, `c` split threads reading node0-resident memory (worst case / page-cache scenario). Prints per-rep CSV + placement evidence. |
| `triad` | STREAM triad `a[i]=b[i]+s*c[i]`, threads pinned to physical cores, node-local slices. GB/s uses STREAM convention (3x8xN/t; RFO write-allocate means true bus traffic is ~4/3 of reported). |
| `cache_bw` | single-core streaming read, working sets 16K/256K/1M/8M/32M. |
| `instr_tp` | single-core zmm throughput: vpermb, vpshufb, vpdpbusd, vpdpwssd, vpopcntq, vfmadd231ps, and a mixed decode loop (load+vpermb+vpdpbusd). Effective AVX-512 frequency from a dependent vpaddd-zmm chain (1 cycle latency/add on ICL), cross-checked against sysfs `scaling_cur_freq`. |
| `vnni_gemm` | straightforward blocked int8 GEMM (u8 x s8 -> i32, vpdpbusd, 4x4-tile micro-kernel, 16 zmm accumulators), threads pinned to physical cores. Verifies 256 random C entries vs scalar reference. Also measures all-core AVX-512 frequency. |
| `mmap_read` | 24 *unpinned* threads reading a page-cache-warm 2 GiB file via mmap (the "mmap'd GGUF" scenario). Dumps numa_maps placement of the mapping. |
| `hog` | touches N MB anon memory then exits; forces global page-cache reclaim (drop_caches is blocked). **Must be run (~100 GB) before the campaign** or first-touch falls back to the wrong node (observed: only 17% of "node0" pages actually on node0 before hog). |

## Anti-elision measures (all verified with objdump; see git history for the bugs)

- Loads feed `vpaddq` accumulators reduced into a `volatile` sink.
- Throughput loops use 12 chains with **distinct** initial values - identical inits
  let GCC CSE all 12 chains into one (caught: only 7 vpshufb in the binary).
- The frequency chain interleaves an empty `asm volatile("" : "+v"(x))` between adds -
  otherwise GCC folds the 16 dependent adds (reported "36 GHz").
- FMA multiplier is loaded at runtime - a literal `1.0f` got simplified to `vaddps`
  (0 fma instructions in the binary), which *doubled* apparent FMA throughput since
  vaddps issues on 2 ports.

## Running

```
./hog 102400                      # once, frees page cache for correct first-touch
./run_all.sh stream               # ~5 min
./run_all.sh triad
./run_all.sh cache
./run_all.sh instr
./run_all.sh vnni
./run_all.sh mmap
```

Each phase checks 1-min load <= 1.0 before starting and logs to
`llmdocs/cpu_kernel_rnd/verify/roofline/<phase>_<timestamp>.log`; parsed CSVs land in
`results/`. GB/s everywhere = 1e9 bytes/s.

## Known caveats

- `cache_bw` at 16 KB working set: the read-pass function is re-entered every 16 KB,
  so per-pass call/reduction overhead depresses the L1 number by ~10-15%. The DRAM
  points (the decode-relevant ones) are unaffected.
- `vnni_gemm` is deliberately *not* tuned (GCC emits some register rotation + 2 spills
  per k-step in the inner loop); it bounds what a straightforward implementation gets,
  not the machine peak.
- Run-to-run spread is reported as min/median/max over >=5 reps in every SUMMARY line.
