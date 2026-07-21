// bench_ik.cpp - standalone microbenchmark harness for ik_llama.cpp CPU matmul
// kernels (iqk_mul_mat), placement-fair, for per-shape comparison against
// future DOML kernels.
//
// Weight types supported end-to-end: IQ2_KL, Q2_K, Q2_K_R4, Q8_K_R16.
// Activation type: Q8_K for all four (verified against iqk_set_kernels_* in
// iqk_gemm_kquants.cpp / iqk_gemm_iqk_quants.cpp: expected_type_B == GGML_TYPE_Q8_K).
// Activations are quantized ONCE outside the timed region.
//
// NUMA/pinning discipline (matches kernels/cpu/roofline): thread t is pinned
// to CPU t (node0 = even CPUs, node1 = odd; CPUs 0..23 are the 24 distinct
// physical cores alternating between sockets). Weights are first-touched by
// the exact thread that will read them inside iqk_mul_mat (the row-slice
// logic of iqk_mul_mat.cpp is replicated here, including the Ny>=32
// convert-to-Q8_K_R16 path whose group size is 16 rows). Placement evidence
// is printed from /proc/self/numa_maps (same method as roofline/stream_read).
//
// Correctness: reference output = (weights dequantized with ik's own
// dequantize_row_* functions) x (activations dequantized from Q8_K), dot in
// double. For the direct (non-convert) paths the kernel does exact integer
// arithmetic on the same values -> max rel err ~1e-6 (fp32 accumulation
// order only). For the Ny>=32 convert path (IQ2_KL, Q2_K) the engine
// re-quantizes weights to Q8_K_R16 on the fly (second int8 rounding),
// so rel err ~1e-3 is expected and correct behavior.
//
// Build/run: see README.md in this directory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>

#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>

#include "ggml.h"                 // ggml_type, ggml_row_size, ggml_init
#include "iqk/iqk_mul_mat.h"      // iqk_mul_mat, iqk_dequant_type
#include "iqk/iqk_quantize.h"     // quantize_iq2_kl, quantize_q2_k_r4, quantize_q8_k_r16,
                                  // dequantize_row_*, iqk_quantize_row_q8_K
#include "ggml-quants.h"          // quantize_q2_K, dequantize_row_q2_K

// ---------------------------------------------------------------- utilities

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void pin_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set)) {
        perror("sched_setaffinity");
        exit(1);
    }
}

// thread t -> cpu t: CPUs 0..23 are 24 distinct physical cores alternating
// node0(even)/node1(odd). Same as roofline stream_read mode b.
static inline int cpu_for_thread(int t) { return t; }

static void *alloc_anon(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    return p;
}

// Placement evidence: sum N0=/N1= pages over large anonymous mappings
// (the weight slab is >= 4 MB; everything else large is transient).
static void dump_anon_numa(const char *tag) {
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f) { fprintf(stderr, "numa_maps unavailable\n"); return; }
    char line[8192];
    long n0 = 0, n1 = 0;
    while (fgets(line, sizeof(line), f)) {
        if (!strstr(line, "anon=")) continue;
        char *p = strstr(line, "anon=");
        long anon = p ? atol(p + 5) : 0;
        if (anon < 1000) continue; // skip mappings < ~4 MB
        p = strstr(line, "N0=");
        if (p) n0 += atol(p + 3);
        p = strstr(line, "N1=");
        if (p) n1 += atol(p + 3);
    }
    fclose(f);
    fprintf(stderr, "PLACEMENT %s large-anon pages node0=%ld node1=%ld (%.1f%% on node0)\n",
            tag, n0, n1, 100.0 * (double)n0 / (double)(n0 + n1 ? n0 + n1 : 1));
}

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// deterministic pseudo-random fp32 in [-1,1), independent of threading
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline float det_val(uint64_t seed, uint64_t idx) {
    uint64_t h = splitmix64(seed ^ (0x100000001b3ULL * (idx + 1)));
    return (float)((double)(h >> 11) * (2.0 / 9007199254740992.0) - 1.0);
}

// ---------------------------------------------------------------- type table

typedef size_t (*quant_fn)(const float *, void *, int64_t, int64_t,
                           const float *, const struct quantize_user_data *);

struct TypeInfo {
    const char *name;
    ggml_type   type;
    quant_fn    quantize;     // fp32[nrows][n_per_row] -> quantized, imatrix may be NULL
    int         row_mult;     // nrows must be a multiple of this (interleave group)
    // dequantize one interleave group (row_mult rows, k = row_mult*n_per_row)
    void (*deq_group)(const void *src, float *dst, int64_t k);
};

static void deq_iq2_kl(const void *s, float *d, int64_t k)   { dequantize_row_iq2_kl ((const block_iq2_kl  *)s, d, k); }
static void deq_q2_K(const void *s, float *d, int64_t k)     { dequantize_row_q2_K   ((const block_q2_K    *)s, d, k); }
static void deq_q2_k_r4(const void *s, float *d, int64_t k)  { dequantize_row_q2_k_r4((const block_q2_k_r4 *)s, d, k); }
static void deq_q8_k_r16(const void *s, float *d, int64_t k) { dequantize_row_q8_k_r16((const block_q8_k_r16*)s, d, k); }

static const TypeInfo k_types[] = {
    { "iq2_kl",    GGML_TYPE_IQ2_KL,   quantize_iq2_kl,    1,  deq_iq2_kl    },
    { "q2_k",      GGML_TYPE_Q2_K,     quantize_q2_K,      1,  deq_q2_K      },
    { "q2_k_r4",   GGML_TYPE_Q2_K_R4,  quantize_q2_k_r4,   4,  deq_q2_k_r4   },
    { "q8_k_r16",  GGML_TYPE_Q8_K_R16, quantize_q8_k_r16,  16, deq_q8_k_r16  },
};
static const int k_ntypes = (int)(sizeof(k_types) / sizeof(k_types[0]));

static const TypeInfo *find_type(const char *name) {
    for (int i = 0; i < k_ntypes; ++i)
        if (strcmp(k_types[i].name, name) == 0) return &k_types[i];
    return nullptr;
}

// interleave-group size used by iqk_mul_mat row splitting (MulMat::num_rows,
// HAVE_FANCY_SIMD branch, iqk_mul_mat.cpp:327). Only the types we can reach.
static int iqk_num_rows(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q2_K_R4:  return 4;
        case GGML_TYPE_Q8_K_R16: return 16;
        default:                 return 1;
    }
}

// Replicates the thread->row-slice logic of iqk_mul_mat (iqk_mul_mat.cpp:501-610)
// so we can first-touch each thread's weight rows on its own node.
// Returns [r0, r1) rows of A read by thread ith. Returns false if the config
// falls into the shared tile-queue path (Nx/nth < 32) where slicing is
// round-robin per tile (placement can then only be approximate).
static bool thread_row_slice(ggml_type typeA, long Nx, long Ny, int ith, int nth,
                             long *r0, long *r1) {
    *r0 = *r1 = 0;
    if (Nx / nth < 32) return false; // tile-queue path
    long npt = (Nx + nth - 1) / nth;
    ggml_type dt = (ggml_type)iqk_dequant_type((int)typeA, (int)Ny);
    int nr;
    if (npt >= 16 && dt != typeA && Nx % iqk_num_rows(dt) == 0) {
        nr = iqk_num_rows(dt);   // on-the-fly convert path (e.g. IQ2_KL -> Q8_K_R16)
    } else {
        nr = iqk_num_rows(typeA);
    }
    long ngroups = Nx / nr;
    long npg = (ngroups + nth - 1) / nth;
    long g0 = (long)ith * npg;
    long g1 = g0 + npg;
    if (g0 > ngroups) g0 = ngroups;
    if (g1 > ngroups) g1 = ngroups;
    *r0 = g0 * nr;
    *r1 = g1 * nr;
    return true;
}

// ---------------------------------------------------------------- buffers

struct Config {
    const TypeInfo *ti;
    long dout, din, ny;
    int  nth;
    int  nbuf;
    uint64_t seed;
};

struct Bufs {
    uint8_t *slabA = nullptr;   // nbuf weight copies
    size_t   slab_bytes = 0;
    size_t   copy_stride = 0;   // page-rounded bytes per copy
    size_t   strideA = 0;       // bytes per weight row
    uint8_t *B = nullptr;       // Q8_K activations, quantized once
    size_t   B_bytes = 0;
    size_t   strideB = 0;
    float   *C = nullptr;       // output Ny x Nx
    size_t   C_bytes = 0;
    std::vector<float> srcW;    // transient; freed at end of setup so the
    std::vector<float> srcB;    // numa_maps placement evidence is clean

    const uint8_t *A(int i) const { return slabA + (size_t)i * copy_stride; }
    void release() {
        if (slabA) munmap(slabA, slab_bytes);
        if (B)     munmap(B, B_bytes);
        if (C)     munmap(C, C_bytes);
        slabA = nullptr; B = nullptr; C = nullptr;
    }
};

static void setup_buffers(const Config &cfg, Bufs &b, bool verbose) {
    const long Nx = cfg.dout, K = cfg.din, Ny = cfg.ny;
    const TypeInfo &ti = *cfg.ti;

    if (Nx % ti.row_mult) {
        fprintf(stderr, "FATAL: dout=%ld not a multiple of %d (required by %s)\n",
                Nx, ti.row_mult, ti.name);
        exit(1);
    }
    if (K % 256) {
        fprintf(stderr, "FATAL: din=%ld not a multiple of QK_K=256\n", K);
        exit(1);
    }

    b.strideA = ggml_row_size(ti.type, K);
    b.strideB = ggml_row_size(GGML_TYPE_Q8_K, K);

    // 1. deterministic fp32 weights + activations (same source for all types)
    b.srcW.resize((size_t)Nx * K);
    b.srcB.resize((size_t)Ny * K);
    #pragma omp parallel for schedule(static)
    for (long r = 0; r < Nx; ++r)
        for (long j = 0; j < K; ++j)
            b.srcW[(size_t)r * K + j] = det_val(cfg.seed, (uint64_t)r * K + j);
    #pragma omp parallel for schedule(static)
    for (long r = 0; r < Ny; ++r)
        for (long j = 0; j < K; ++j)
            b.srcB[(size_t)r * K + j] = det_val(cfg.seed + 0x5eedULL, (uint64_t)r * K + j);

    // 2. quantize weights once into a temp buffer; verify the reported size
    size_t w_bytes = (size_t)Nx * b.strideA;
    std::vector<uint8_t> qtmp(w_bytes);
    size_t ret = ti.quantize(b.srcW.data(), qtmp.data(), Nx, K, nullptr, nullptr);
    if (ret != w_bytes) {
        fprintf(stderr, "FATAL: quantize_%s returned %zu bytes, expected %zu (= %ld * ggml_row_size=%zu)\n",
                ti.name, ret, w_bytes, Nx, b.strideA);
        exit(1);
    }

    // 3. weight slab: nbuf copies, first-touched per iqk_mul_mat's row slices
    b.copy_stride = (w_bytes + 4095) & ~(size_t)4095;
    b.slab_bytes  = b.copy_stride * (size_t)cfg.nbuf;
    b.slabA = (uint8_t *)alloc_anon(b.slab_bytes);

    bool sliced_ok = true;
    #pragma omp parallel num_threads(cfg.nth) reduction(&&:sliced_ok)
    {
        int ith = omp_get_thread_num();
        pin_cpu(cpu_for_thread(ith));
        long r0, r1;
        bool ok = thread_row_slice(ti.type, Nx, Ny, ith, cfg.nth, &r0, &r1);
        if (ok && r1 > r0) {
            for (int c = 0; c < cfg.nbuf; ++c) {
                memcpy(b.slabA + (size_t)c * b.copy_stride + (size_t)r0 * b.strideA,
                       qtmp.data() + (size_t)r0 * b.strideA,
                       (size_t)(r1 - r0) * b.strideA);
            }
        }
        sliced_ok = sliced_ok && ok;
    }
    if (!sliced_ok) {
        // tile-queue fallback: touch round-robin from all threads (approximate)
        fprintf(stderr, "WARN: Nx/nth < 32 -> iqk uses the shared tile queue; "
                        "first-touch placement is only approximate for this config\n");
        #pragma omp parallel num_threads(cfg.nth)
        {
            int ith = omp_get_thread_num();
            pin_cpu(cpu_for_thread(ith));
            for (long r = ith * 32; r < Nx; r += (long)cfg.nth * 32) {
                long rows = std::min<long>(32, Nx - r);
                for (int c = 0; c < cfg.nbuf; ++c)
                    memcpy(b.slabA + (size_t)c * b.copy_stride + (size_t)r * b.strideA,
                           qtmp.data() + (size_t)r * b.strideA, (size_t)rows * b.strideA);
            }
        }
    }

    // 4. activations: quantize to Q8_K once, rows striped across threads
    //    (thread ith touches+quantizes rows ith, ith+nth, ... like ggml's
    //    cooperative activation quantization, ggml.c:17298)
    b.B_bytes = (size_t)Ny * b.strideB;
    b.B = (uint8_t *)alloc_anon(b.B_bytes);
    #pragma omp parallel num_threads(cfg.nth)
    {
        int ith = omp_get_thread_num();
        pin_cpu(cpu_for_thread(ith));
        for (long r = ith; r < Ny; r += cfg.nth)
            iqk_quantize_row_q8_K(b.srcB.data() + (size_t)r * K, b.B + (size_t)r * b.strideB, K);
    }

    // 5. output: first-touch by each thread's column slice (approximate at
    //    page granularity; C is small and write-mostly)
    b.C_bytes = (size_t)Ny * Nx * sizeof(float);
    b.C = (float *)alloc_anon(b.C_bytes);
    #pragma omp parallel num_threads(cfg.nth)
    {
        int ith = omp_get_thread_num();
        pin_cpu(cpu_for_thread(ith));
        long r0, r1;
        if (thread_row_slice(ti.type, Nx, Ny, ith, cfg.nth, &r0, &r1) && r1 > r0) {
            for (long y = 0; y < Ny; ++y)
                memset(b.C + (size_t)y * Nx + r0, 0, (size_t)(r1 - r0) * sizeof(float));
        } else if (ith == 0) {
            memset(b.C, 0, b.C_bytes);
        }
    }

    // free the fp32 sources: they are large anon mappings that would otherwise
    // pollute the /proc/self/numa_maps placement evidence
    b.srcW = std::vector<float>();
    b.srcB = std::vector<float>();

    if (verbose) {
        double bpw = 8.0 * (double)b.strideA / (double)K;
        ggml_type dt = (ggml_type)iqk_dequant_type((int)ti.type, (int)Ny);
        fprintf(stderr,
                "SETUP type=%s dout=%ld din=%ld ny=%ld nth=%d nbuf=%d strideA=%zu (%.4f bpw) "
                "strideB=%zu path=%s\n",
                ti.name, Nx, K, Ny, cfg.nth, cfg.nbuf, b.strideA, bpw, b.strideB,
                dt != ti.type && (Nx + cfg.nth - 1) / cfg.nth >= 16 && Nx / cfg.nth >= 32
                    ? "convert-to-q8_k_r16" : "direct");
    }
}

// ------------------------------------------------------------ kernel driving

// One multi-threaded iqk_mul_mat call group: every thread calls with its
// (ith, nth), exactly like ggml_compute_forward_mul_mat does per node.
// Returns false if ANY thread's call returned false (unsupported combo).
static bool run_once(const Config &cfg, const Bufs &b, int buf_idx) {
    std::atomic<bool> ok{true};
    #pragma omp parallel num_threads(cfg.nth)
    {
        int ith = omp_get_thread_num();
        pin_cpu(cpu_for_thread(ith));
        if (!iqk_mul_mat(cfg.dout, cfg.ny, cfg.din,
                         (int)cfg.ti->type, b.A(buf_idx), (long)b.strideA,
                         (int)GGML_TYPE_Q8_K, b.B, (long)b.strideB,
                         b.C, cfg.dout, ith, cfg.nth))
            ok.store(false, std::memory_order_relaxed);
    }
    return ok.load();
}

// Timed repetition: iters back-to-back call groups inside one parallel region,
// one omp barrier between calls (mirrors ggml's one-barrier-per-node model).
// Weight copies are rotated to control the cache working set.
static double run_rep(const Config &cfg, const Bufs &b, int iters, int *buf_rot,
                      std::atomic<bool> *ok) {
    double t0 = 0, t1 = 0;
    int rot = *buf_rot;
    #pragma omp parallel num_threads(cfg.nth)
    {
        int ith = omp_get_thread_num();
        pin_cpu(cpu_for_thread(ith));
        #pragma omp barrier
        #pragma omp master
        t0 = now_sec();
        #pragma omp barrier
        for (int it = 0; it < iters; ++it) {
            const uint8_t *A = b.A((rot + it) % cfg.nbuf);
            if (!iqk_mul_mat(cfg.dout, cfg.ny, cfg.din,
                             (int)cfg.ti->type, A, (long)b.strideA,
                             (int)GGML_TYPE_Q8_K, b.B, (long)b.strideB,
                             b.C, cfg.dout, ith, cfg.nth))
                ok->store(false, std::memory_order_relaxed);
            #pragma omp barrier
        }
        #pragma omp master
        t1 = now_sec();
    }
    *buf_rot = (rot + iters) % cfg.nbuf;
    return t1 - t0;
}

// ------------------------------------------------------------ correctness

struct CheckResult {
    double max_nrm;     // max |C-ref| / rms(ref)  (elementwise rel is misleading
                        //  near ref~0, so errors are normalized by the ref RMS)
    double rms_rel;     // ||C-ref|| / ||ref||
};

static CheckResult check_output(const Config &cfg, const Bufs &b) {
    const long Nx = cfg.dout, K = cfg.din, Ny = cfg.ny;
    const TypeInfo &ti = *cfg.ti;

    // dequantize weights with ik's own row functions (per interleave group)
    std::vector<float> W((size_t)Nx * K);
    const int rm = ti.row_mult;
    #pragma omp parallel for schedule(static)
    for (long g = 0; g < Nx / rm; ++g)
        ti.deq_group(b.A(0) + (size_t)g * rm * b.strideA,
                     W.data() + (size_t)g * rm * K, (int64_t)rm * K);

    // dequantize the Q8_K activations (d * qs per 256-block; block_q8_K layout)
    std::vector<float> Y((size_t)Ny * K);
    for (long r = 0; r < Ny; ++r) {
        const uint8_t *row = b.B + (size_t)r * b.strideB;
        for (long ib = 0; ib < K / 256; ++ib) {
            const block_q8_K *blk = (const block_q8_K *)(row + ib * sizeof(block_q8_K));
            for (int j = 0; j < 256; ++j)
                Y[(size_t)r * K + ib * 256 + j] = blk->d * (float)blk->qs[j];
        }
    }

    // reference in double + compare
    double ref_ss = 0;
    #pragma omp parallel for schedule(static) reduction(+:ref_ss)
    for (long y = 0; y < Ny; ++y) {
        for (long x = 0; x < Nx; ++x) {
            double acc = 0;
            const float *w = W.data() + (size_t)x * K;
            const float *v = Y.data() + (size_t)y * K;
            for (long j = 0; j < K; ++j) acc += (double)w[j] * (double)v[j];
            ref_ss += acc * acc;
        }
    }
    double ref_rms = std::sqrt(ref_ss / ((double)Nx * Ny));

    double max_abs = 0, err_ss = 0;
    #pragma omp parallel for schedule(static) reduction(max:max_abs) reduction(+:err_ss)
    for (long y = 0; y < Ny; ++y) {
        for (long x = 0; x < Nx; ++x) {
            double acc = 0;
            const float *w = W.data() + (size_t)x * K;
            const float *v = Y.data() + (size_t)y * K;
            for (long j = 0; j < K; ++j) acc += (double)w[j] * (double)v[j];
            double c = (double)b.C[(size_t)y * Nx + x];
            double e = std::fabs(c - acc);
            err_ss += e * e;
            if (e > max_abs) max_abs = e;
        }
    }
    return { max_abs / ref_rms, std::sqrt(err_ss / ref_ss) };
}

// ------------------------------------------------------------ bench driver

struct BenchLine {
    double ns_call, gbps_w, gmacs, calls_s;
};

static BenchLine bench_config(const Config &cfg, const Bufs &b, int reps,
                              double target_ms, FILE *csv) {
    std::atomic<bool> ok{true};
    int rot = 0;

    // warmup (page tables, thread_local convert buffers, AVX-512 license)
    run_rep(cfg, b, 2, &rot, &ok);

    // calibrate iters so a rep lasts >= target_ms
    int iters = 1;
    for (int tries = 0; tries < 4; ++tries) {
        double t = run_rep(cfg, b, iters, &rot, &ok);
        if (t >= target_ms * 1e-3 || iters >= (1 << 22)) break;
        double scale = (target_ms * 1.2e-3) / std::max(t, 1e-7);
        iters = (int)std::min<double>((double)iters * std::max(scale, 2.0), 1 << 22);
    }

    std::vector<double> secs(reps);
    for (int r = 0; r < reps; ++r) secs[r] = run_rep(cfg, b, iters, &rot, &ok);

    if (!ok.load()) {
        fprintf(stderr, "FATAL: iqk_mul_mat returned false for type=%s ny=%ld nth=%d "
                        "(unsupported combo)\n", cfg.ti->name, cfg.ny, cfg.nth);
        exit(1);
    }

    double w_bytes = (double)cfg.dout * (double)b.strideA;
    double macs    = (double)cfg.dout * (double)cfg.din * (double)cfg.ny;
    std::vector<double> nsc(reps), gbw(reps), gmc(reps);
    for (int r = 0; r < reps; ++r) {
        double per_call = secs[r] / iters;
        nsc[r] = per_call * 1e9;
        gbw[r] = w_bytes / per_call / 1e9;
        gmc[r] = macs / per_call / 1e9;
        if (csv)
            fprintf(csv, "bench_ik,%s,%ld,%ld,%ld,%d,%d,%d,%d,%.6f,%.1f,%.3f,%.3f\n",
                    cfg.ti->name, cfg.dout, cfg.din, cfg.ny, cfg.nth, cfg.nbuf,
                    r, iters, secs[r], nsc[r], gbw[r], gmc[r]);
    }
    BenchLine out;
    out.ns_call = median_of(nsc);
    out.gbps_w  = median_of(gbw);
    out.gmacs   = median_of(gmc);
    out.calls_s = 1e9 / out.ns_call;
    fprintf(stderr,
            "SUMMARY bench_ik type=%-9s dout=%-4ld din=%-4ld ny=%-3ld nth=%-2d nbuf=%-3d iters=%d "
            "median=%.1f ns/call [%.1f,%.1f]  weightBW=%.2f GB/s  %.2f GMAC/s  %.1f calls/s\n",
            cfg.ti->name, cfg.dout, cfg.din, cfg.ny, cfg.nth, cfg.nbuf, iters,
            out.ns_call, *std::min_element(nsc.begin(), nsc.end()),
            *std::max_element(nsc.begin(), nsc.end()), out.gbps_w, out.gmacs, out.calls_s);
    return out;
}

// ------------------------------------------------------------ modes

static int auto_nbuf(const Config &cfg, size_t strideA_hint) {
    // enough weight copies that the cycled working set exceeds combined L3
    // (2 x 18 MB) by a wide margin -> DRAM-streaming decode like a real model
    size_t w = (size_t)cfg.dout * strideA_hint;
    size_t target = 384ull << 20;
    long n = (long)((target + w - 1) / w);
    return (int)std::max(1L, std::min(n, 512L));
}

// Qwen3-0.6B linear shapes: {dout, din, count-in-model, label}
struct Shape { long dout, din; int count; const char *label; };
static const Shape k_shapes[] = {
    { 2048, 1024, 1, "q_proj"    },
    { 1024, 1024, 2, "k/v_proj"  },
    { 1024, 2048, 1, "o_proj"    },
    { 3072, 1024, 2, "gate/up"   },
    { 1024, 3072, 1, "down_proj" },
};
static const int k_nshapes = (int)(sizeof(k_shapes) / sizeof(k_shapes[0]));

static int do_validate(uint64_t seed) {
    fprintf(stderr, "=== VALIDATE: correctness vs ik's own dequant + double reference ===\n");
    // sanity: convert-path routing is what the anatomy doc says
    fprintf(stderr, "iqk_dequant_type(IQ2_KL, 512) = %d (expect %d = GGML_TYPE_Q8_K_R16)\n",
            iqk_dequant_type(GGML_TYPE_IQ2_KL, 512), (int)GGML_TYPE_Q8_K_R16);
    fprintf(stderr, "iqk_dequant_type(Q2_K,   512) = %d (expect %d)\n",
            iqk_dequant_type(GGML_TYPE_Q2_K, 512), (int)GGML_TYPE_Q8_K_R16);
    fprintf(stderr, "iqk_dequant_type(Q2_K_R4,512) = %d (expect %d = itself, no convert)\n",
            iqk_dequant_type(GGML_TYPE_Q2_K_R4, 512), (int)GGML_TYPE_Q2_K_R4);

    struct Case { long dout, din; long ny; int nth; };
    const Case cases[] = {
        { 2048, 1024,   1,  1 }, { 2048, 1024,   1, 24 },
        { 2048, 1024,  16, 24 },                          // func16 ladder
        { 2048, 1024, 512,  1 }, { 2048, 1024, 512, 24 },
        { 1024, 3072, 512, 24 },                          // widest K
    };
    int nfail = 0;
    printf("type,dout,din,ny,nth,path,max_err_over_ref_rms,rms_rel_err,verdict\n");
    for (int t = 0; t < k_ntypes; ++t) {
        for (const auto &cs : cases) {
            Config cfg { &k_types[t], cs.dout, cs.din, cs.ny, cs.nth, 1, seed };
            Bufs b;
            setup_buffers(cfg, b, false);
            if (!run_once(cfg, b, 0)) {
                fprintf(stderr, "FATAL: iqk_mul_mat returned false: type=%s ny=%ld nth=%d\n",
                        cfg.ti->name, cs.ny, cs.nth);
                exit(1);
            }
            CheckResult cr = check_output(cfg, b);
            ggml_type dt = (ggml_type)iqk_dequant_type((int)cfg.ti->type, (int)cs.ny);
            bool convert = dt != cfg.ti->type && (cs.dout + cs.nth - 1) / cs.nth >= 16
                           && cs.dout / cs.nth >= 32;
            // direct paths do exact int math on the dequantized values ->
            // fp32-accumulation-level error; the convert path re-rounds the
            // weights to int8 (Q8_K_R16) on the fly -> ~0.5% RMS is expected
            double tol_max = convert ? 1e-1 : 1e-4;
            double tol_rms = convert ? 2e-2 : 1e-5;
            bool pass = cr.max_nrm < tol_max && cr.rms_rel < tol_rms;
            if (!pass) ++nfail;
            printf("%s,%ld,%ld,%ld,%d,%s,%.3e,%.3e,%s\n",
                   cfg.ti->name, cs.dout, cs.din, cs.ny, cs.nth,
                   convert ? "convert" : "direct", cr.max_nrm, cr.rms_rel,
                   pass ? "PASS" : "FAIL");
            fflush(stdout);
            b.release();
        }
    }
    fprintf(stderr, nfail ? "=== VALIDATE: %d FAILURES ===\n" : "=== VALIDATE: all PASS ===\n", nfail);
    return nfail ? 1 : 0;
}

static void do_sweep(int reps, double target_ms, uint64_t seed) {
    fprintf(stderr, "=== SWEEP: Qwen3-0.6B shapes x {%d types} x ny {1,512} x threads {1,24} ===\n",
            k_ntypes);
    printf("kind,type,dout,din,ny,nth,nbuf,rep,iters,secs,ns_call,weight_GBps,GMACs\n");
    const long nys[] = { 1, 512 };
    const int  nts[] = { 1, 24 };
    for (int t = 0; t < k_ntypes; ++t) {
        for (int s = 0; s < k_nshapes; ++s) {
            for (long ny : nys) {
                for (int nth : nts) {
                    Config cfg { &k_types[t], k_shapes[s].dout, k_shapes[s].din,
                                 ny, nth, 0, seed };
                    cfg.nbuf = auto_nbuf(cfg, ggml_row_size(cfg.ti->type, cfg.din));
                    Bufs b;
                    setup_buffers(cfg, b, true);
                    dump_anon_numa(cfg.ti->name);
                    bench_config(cfg, b, reps, target_ms, stdout);
                    fflush(stdout);
                    b.release();
                }
            }
        }
    }
}

// ------------------------------------------------------------ main

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s --validate [--seed S]\n"
        "  %s --type {iq2_kl|q2_k|q2_k_r4|q8_k_r16} --dout N --din K --ny NY --threads T\n"
        "        [--check] [--bench] [--reps R] [--nbuf N] [--target-ms M] [--seed S]\n"
        "  %s --sweep [--reps R] [--target-ms M] [--seed S]\n"
        "notes: din must be a multiple of 256; dout a multiple of 16 keeps all types valid.\n"
        "       threads are pinned to CPUs 0..T-1 (node0=even, node1=odd).\n",
        argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    // Not strictly required for the iqk entry points on an F16C build (all
    // iqk LUTs are static/ctor-built, fp16 conversion is hardware), but init
    // ggml once defensively: it fills the global fp16/gelu tables used by
    // non-F16C code paths and costs ~ms.
    {
        struct ggml_init_params ip = { 1u << 20, nullptr, true };
        struct ggml_context *ctx = ggml_init(ip);
        if (ctx) ggml_free(ctx);
    }

    const TypeInfo *ti = nullptr;
    long dout = 0, din = 0, ny = 1;
    int nth = 1, reps = 9, nbuf = 0;
    double target_ms = 60.0;
    uint64_t seed = 20260715ULL;
    bool mode_validate = false, mode_sweep = false, mode_check = false, mode_bench = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(1); }
            return argv[++i];
        };
        if      (a == "--validate")  mode_validate = true;
        else if (a == "--sweep")     mode_sweep = true;
        else if (a == "--check")     mode_check = true;
        else if (a == "--bench")     mode_bench = true;
        else if (a == "--type")      { const char *n = next("--type"); ti = find_type(n);
                                       if (!ti) { fprintf(stderr, "unknown type '%s'\n", n); return 1; } }
        else if (a == "--dout")      dout = atol(next("--dout"));
        else if (a == "--din")       din  = atol(next("--din"));
        else if (a == "--ny")        ny   = atol(next("--ny"));
        else if (a == "--threads")   nth  = atoi(next("--threads"));
        else if (a == "--reps")      reps = atoi(next("--reps"));
        else if (a == "--nbuf")      nbuf = atoi(next("--nbuf"));
        else if (a == "--target-ms") target_ms = atof(next("--target-ms"));
        else if (a == "--seed")      seed = strtoull(next("--seed"), nullptr, 0);
        else { usage(argv[0]); return 1; }
    }

    if (mode_validate) return do_validate(seed);
    if (mode_sweep)    { do_sweep(reps, target_ms, seed); return 0; }

    if (!ti || dout <= 0 || din <= 0 || ny <= 0 || nth <= 0) { usage(argv[0]); return 1; }
    if (!mode_check && !mode_bench) mode_check = true;

    Config cfg { ti, dout, din, ny, nth, nbuf, seed };
    if (cfg.nbuf <= 0)
        cfg.nbuf = mode_bench ? auto_nbuf(cfg, ggml_row_size(ti->type, din)) : 1;

    Bufs b;
    setup_buffers(cfg, b, true);
    dump_anon_numa(ti->name);

    if (mode_check) {
        if (!run_once(cfg, b, 0)) {
            fprintf(stderr, "FATAL: iqk_mul_mat returned false (unsupported combo)\n");
            return 1;
        }
        CheckResult cr = check_output(cfg, b);
        printf("check,%s,%ld,%ld,%ld,%d,max_err_over_ref_rms=%.3e,rms_rel=%.3e\n",
               ti->name, dout, din, ny, nth, cr.max_nrm, cr.rms_rel);
    }
    if (mode_bench) {
        printf("kind,type,dout,din,ny,nth,nbuf,rep,iters,secs,ns_call,weight_GBps,GMACs\n");
        bench_config(cfg, b, reps, target_ms, stdout);
    }
    b.release();
    return 0;
}
