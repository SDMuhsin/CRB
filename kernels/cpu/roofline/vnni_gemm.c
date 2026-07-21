// vnni_gemm.c - straightforward blocked int8 GEMM using vpdpbusd (AVX512-VNNI)
// to bound the achievable fraction of theoretical VNNI peak. Not a tuned kernel.
//
// C[M][N] (int32) = A[M][K] (uint8) * B[K][N] (int8)
// B is pre-packed: Bp[ntile][kq][64] where byte c*4+j = B[kq*4+j][ntile*16+c]
// Micro-kernel: 4 rows x 4 ntiles (64 cols) = 16 zmm accumulators;
// per kq: 4 B-vector loads + 4 A dword broadcasts + 16 vpdpbusd.
//
// Usage: vnni_gemm <nthreads> <M> <K> <N> <gemms_per_rep> <reps>
// Threads pinned to cpus 0..nthreads-1 (0..23 = 24 physical cores, both sockets).
// Also measures all-core effective frequency under AVX-512 via a dependent
// vpaddd chain run simultaneously on all threads (latency 1c on Ice Lake).
//
// CSV: vnni_gemm,<threads>,<M>,<K>,<N>,<rep>,<gemms>,<secs>,<GMACps>
//      vnni_freq,<threads>,<tid>,<freqGHz>
#define _GNU_SOURCE
#include "common.h"

static int M, K, N, NT, GEMMS, REPS;
static uint8_t *A;      // M x K
static int8_t *B;       // K x N (row-major, only for packing + reference)
static int8_t *Bp;      // packed
static int32_t *C;      // M x N
static pthread_barrier_t bar;
static double *rep_secs;
static double *thread_freq;

static void pack_B(void) {
    int ntiles = N / 16, kq_n = K / 4;
    for (int t = 0; t < ntiles; t++)
        for (int kq = 0; kq < kq_n; kq++) {
            int8_t *dst = Bp + ((size_t)t * kq_n + kq) * 64;
            for (int c = 0; c < 16; c++)
                for (int j = 0; j < 4; j++)
                    dst[c * 4 + j] = B[(size_t)(kq * 4 + j) * N + t * 16 + c];
        }
}

static inline void kernel_4x4(int m0, int t0) {
    int kq_n = K / 4;
    __m512i acc[4][4];
    for (int r = 0; r < 4; r++)
        for (int v = 0; v < 4; v++) acc[r][v] = _mm512_setzero_si512();
    const int8_t *bp0 = Bp + (size_t)(t0 + 0) * kq_n * 64;
    const int8_t *bp1 = Bp + (size_t)(t0 + 1) * kq_n * 64;
    const int8_t *bp2 = Bp + (size_t)(t0 + 2) * kq_n * 64;
    const int8_t *bp3 = Bp + (size_t)(t0 + 3) * kq_n * 64;
    const uint8_t *a0 = A + (size_t)(m0 + 0) * K;
    const uint8_t *a1 = A + (size_t)(m0 + 1) * K;
    const uint8_t *a2 = A + (size_t)(m0 + 2) * K;
    const uint8_t *a3 = A + (size_t)(m0 + 3) * K;
    for (int kq = 0; kq < kq_n; kq++) {
        __m512i b0 = _mm512_load_si512((const __m512i *)(bp0 + (size_t)kq * 64));
        __m512i b1 = _mm512_load_si512((const __m512i *)(bp1 + (size_t)kq * 64));
        __m512i b2 = _mm512_load_si512((const __m512i *)(bp2 + (size_t)kq * 64));
        __m512i b3 = _mm512_load_si512((const __m512i *)(bp3 + (size_t)kq * 64));
        __m512i av;
        av = _mm512_set1_epi32(*(const int32_t *)(a0 + kq * 4));
        acc[0][0] = _mm512_dpbusd_epi32(acc[0][0], av, b0);
        acc[0][1] = _mm512_dpbusd_epi32(acc[0][1], av, b1);
        acc[0][2] = _mm512_dpbusd_epi32(acc[0][2], av, b2);
        acc[0][3] = _mm512_dpbusd_epi32(acc[0][3], av, b3);
        av = _mm512_set1_epi32(*(const int32_t *)(a1 + kq * 4));
        acc[1][0] = _mm512_dpbusd_epi32(acc[1][0], av, b0);
        acc[1][1] = _mm512_dpbusd_epi32(acc[1][1], av, b1);
        acc[1][2] = _mm512_dpbusd_epi32(acc[1][2], av, b2);
        acc[1][3] = _mm512_dpbusd_epi32(acc[1][3], av, b3);
        av = _mm512_set1_epi32(*(const int32_t *)(a2 + kq * 4));
        acc[2][0] = _mm512_dpbusd_epi32(acc[2][0], av, b0);
        acc[2][1] = _mm512_dpbusd_epi32(acc[2][1], av, b1);
        acc[2][2] = _mm512_dpbusd_epi32(acc[2][2], av, b2);
        acc[2][3] = _mm512_dpbusd_epi32(acc[2][3], av, b3);
        av = _mm512_set1_epi32(*(const int32_t *)(a3 + kq * 4));
        acc[3][0] = _mm512_dpbusd_epi32(acc[3][0], av, b0);
        acc[3][1] = _mm512_dpbusd_epi32(acc[3][1], av, b1);
        acc[3][2] = _mm512_dpbusd_epi32(acc[3][2], av, b2);
        acc[3][3] = _mm512_dpbusd_epi32(acc[3][3], av, b3);
    }
    for (int r = 0; r < 4; r++)
        for (int v = 0; v < 4; v++)
            _mm512_storeu_si512((__m512i *)(C + (size_t)(m0 + r) * N + (t0 + v) * 16),
                                acc[r][v]);
}

typedef struct { int tid; } targ_t;

static void gemm_thread_part(int tid) {
    int rowblocks = M / 4, tgroups = N / 64;
    for (int rb = tid; rb < rowblocks; rb += NT)
        for (int tg = 0; tg < tgroups; tg++)
            kernel_4x4(rb * 4, tg * 4);
}

static void *worker(void *arg) {
    int tid = ((targ_t *)arg)->tid;
    pin_cpu(tid);
    // warmup
    pthread_barrier_wait(&bar);
    for (int w = 0; w < 10; w++) { gemm_thread_part(tid); pthread_barrier_wait(&bar); }
    // all-core AVX-512 frequency: dependent vpaddd chain, all threads at once
    pthread_barrier_wait(&bar);
    {
        // empty asm blocks GCC from folding the dependent chain (1c latency/add)
#define CHAIN_BARRIER(v) __asm__ volatile("" : "+v"(v))
        __m512i x = _mm512_set1_epi32(tid + 1), one = _mm512_set1_epi32(1);
        long iters = 40 * 1000 * 1000;
        double t0 = now_sec();
        for (long i = 0; i < iters; i++) {
            x = _mm512_add_epi32(x, one); CHAIN_BARRIER(x);
            x = _mm512_add_epi32(x, one); CHAIN_BARRIER(x);
            x = _mm512_add_epi32(x, one); CHAIN_BARRIER(x);
            x = _mm512_add_epi32(x, one); CHAIN_BARRIER(x);
            x = _mm512_add_epi32(x, one); CHAIN_BARRIER(x);
            x = _mm512_add_epi32(x, one); CHAIN_BARRIER(x);
            x = _mm512_add_epi32(x, one); CHAIN_BARRIER(x);
            x = _mm512_add_epi32(x, one); CHAIN_BARRIER(x);
        }
        double dt = now_sec() - t0;
        g_sink = (uint64_t)_mm512_reduce_add_epi64(x);
        thread_freq[tid] = 8.0 * (double)iters / dt / 1e9;
    }
    pthread_barrier_wait(&bar);
    for (int r = 0; r < REPS; r++) {
        pthread_barrier_wait(&bar);
        double t0 = now_sec();
        for (int gg = 0; gg < GEMMS; gg++) {
            gemm_thread_part(tid);
            pthread_barrier_wait(&bar);
        }
        if (tid == 0) rep_secs[r] = now_sec() - t0;
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <nthreads> <M> <K> <N> <gemms_per_rep> <reps>\n", argv[0]);
        return 1;
    }
    NT = atoi(argv[1]); M = atoi(argv[2]); K = atoi(argv[3]); N = atoi(argv[4]);
    GEMMS = atoi(argv[5]); REPS = atoi(argv[6]);
    if (M % 4 || K % 4 || N % 64) { fprintf(stderr, "need M%%4==0 K%%4==0 N%%64==0\n"); return 1; }

    A = alloc_anon((size_t)M * K);
    B = alloc_anon((size_t)K * N);
    Bp = alloc_anon((size_t)K * N);
    C = alloc_anon((size_t)M * N * 4);
    srandom(999);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (uint8_t)(random() % 16);
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (int8_t)((random() % 16) - 8);
    pack_B();

    // correctness check: one single-threaded GEMM, verify 256 random entries
    {
        int save_nt = NT; NT = 1;
        gemm_thread_part(0);
        NT = save_nt;
        for (int chk = 0; chk < 256; chk++) {
            int m = (int)(random() % M), n = (int)(random() % N);
            int64_t ref = 0;
            for (int k = 0; k < K; k++) ref += (int64_t)A[(size_t)m * K + k] * B[(size_t)k * N + n];
            if (C[(size_t)m * N + n] != (int32_t)ref) {
                fprintf(stderr, "MISMATCH C[%d][%d]=%d ref=%lld\n", m, n,
                        C[(size_t)m * N + n], (long long)ref);
                return 2;
            }
        }
        fprintf(stderr, "correctness: 256/256 sampled entries match reference\n");
    }

    pthread_barrier_init(&bar, NULL, NT);
    rep_secs = calloc(REPS, sizeof(double));
    thread_freq = calloc(NT, sizeof(double));
    pthread_t *th = calloc(NT, sizeof(pthread_t));
    targ_t *ta = calloc(NT, sizeof(targ_t));
    for (int t = 0; t < NT; t++) { ta[t].tid = t; pthread_create(&th[t], NULL, worker, &ta[t]); }
    for (int t = 0; t < NT; t++) pthread_join(th[t], NULL);

    for (int t = 0; t < NT; t++)
        printf("vnni_freq,%d,%d,%.4f\n", NT, t, thread_freq[t]);
    double fmed = median_of(thread_freq, NT); // note: sorts in place
    double macs_per_gemm = (double)M * K * N;
    double *gmacs = calloc(REPS, sizeof(double));
    for (int r = 0; r < REPS; r++) {
        gmacs[r] = macs_per_gemm * GEMMS / rep_secs[r] / 1e9;
        printf("vnni_gemm,%d,%d,%d,%d,%d,%d,%.6f,%.3f\n",
               NT, M, K, N, r, GEMMS, rep_secs[r], gmacs[r]);
    }
    fprintf(stderr, "SUMMARY vnni_gemm threads=%d M=%d K=%d N=%d median=%.2f GMAC/s "
            "min=%.2f max=%.2f | allcore-512 freq median=%.3f GHz\n",
            NT, M, K, N, median_of(gmacs, REPS), min_of(gmacs, REPS), max_of(gmacs, REPS), fmed);
    return 0;
}
