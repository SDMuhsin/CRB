// instr_tp.c - single-core AVX-512 (zmm) instruction throughput + effective
// frequency under sustained 512-bit work.
//
// Usage: instr_tp <cpu> <reps>
//
// Frequency: a dependent vpaddd-zmm chain (latency = 1 cycle on Ice Lake-SP)
// executes exactly 1 add/cycle, so freq = chain_len / elapsed. This runs in the
// same AVX-512 "license" as the throughput loops. We also sample
// /sys/.../scaling_cur_freq (or /proc/cpuinfo) for the pinned CPU mid-run.
//
// Throughput loops: 12 independent chains, each op depends only on its own
// accumulator (chain latency <= 5 cycles for all tested ops, so 12 chains
// saturate any throughput >= 1/5 per cycle). 24 ops per loop iteration.
//
// CSV: instr_tp,<cpu>,<instr>,<rep>,<ops>,<secs>,<GOPS>,<freqGHz_chain>,<ops_per_cycle>,<sysfs_MHz>
#define _GNU_SOURCE
#include "common.h"

static volatile uint8_t v_seed[64 * 4];
static volatile float v_fseed[16 * 3];

static int g_cpu = 2;
static volatile int g_sampling = 0;
static double g_mhz_samples[4096];
static int g_nsamples = 0;

static double read_cpu_mhz(int cpu) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *f = fopen(path, "r");
    if (f) {
        long khz = 0;
        if (fscanf(f, "%ld", &khz) == 1) { fclose(f); return khz / 1000.0; }
        fclose(f);
    }
    // fallback: /proc/cpuinfo
    f = fopen("/proc/cpuinfo", "r");
    if (!f) return -1;
    char line[512];
    int cur = -1;
    double mhz = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "processor : %d", &cur) == 1) continue;
        double m;
        if (cur == cpu && sscanf(line, "cpu MHz : %lf", &m) == 1) { mhz = m; break; }
    }
    fclose(f);
    return mhz;
}

static void *sampler(void *arg) {
    (void)arg;
    while (g_sampling) {
        double m = read_cpu_mhz(g_cpu);
        if (m > 0 && g_nsamples < 4096) g_mhz_samples[g_nsamples++] = m;
        struct timespec ts = { 0, 20 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    return NULL;
}

// Dependent vpaddd zmm chain: 16 dependent adds per iteration, latency 1c each.
// The empty asm forces x through a register at every step so GCC cannot
// reassociate/fold the chain (it did: reported "36 GHz" before this fix).
#define CHAIN_BARRIER(x) __asm__ volatile("" : "+v"(x))
static double chain_freq_ghz(long iters) {
    __m512i x = _mm512_loadu_si512((const void *)v_seed);
    __m512i one = _mm512_set1_epi32(1);
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
    return 16.0 * (double)iters / dt / 1e9;
}

// NOTE: each chain gets a DISTINCT init value; identical inits let GCC merge
// the 12 chains into one via CSE (verified with objdump before this fix).
#define DECL12(init) __m512i a0=init, \
    a1=_mm512_add_epi8(a0,_mm512_set1_epi8(1)),  a2=_mm512_add_epi8(a0,_mm512_set1_epi8(2)), \
    a3=_mm512_add_epi8(a0,_mm512_set1_epi8(3)),  a4=_mm512_add_epi8(a0,_mm512_set1_epi8(4)), \
    a5=_mm512_add_epi8(a0,_mm512_set1_epi8(5)),  a6=_mm512_add_epi8(a0,_mm512_set1_epi8(6)), \
    a7=_mm512_add_epi8(a0,_mm512_set1_epi8(7)),  a8=_mm512_add_epi8(a0,_mm512_set1_epi8(8)), \
    a9=_mm512_add_epi8(a0,_mm512_set1_epi8(9)),  aa=_mm512_add_epi8(a0,_mm512_set1_epi8(10)), \
    ab=_mm512_add_epi8(a0,_mm512_set1_epi8(11))
#define APPLY12(OP) OP(a0);OP(a1);OP(a2);OP(a3);OP(a4);OP(a5); \
    OP(a6);OP(a7);OP(a8);OP(a9);OP(aa);OP(ab)
#define SINK12() do { \
    __m512i s = _mm512_add_epi64(a0,a1); s=_mm512_add_epi64(s,a2); s=_mm512_add_epi64(s,a3); \
    s=_mm512_add_epi64(s,a4); s=_mm512_add_epi64(s,a5); s=_mm512_add_epi64(s,a6); \
    s=_mm512_add_epi64(s,a7); s=_mm512_add_epi64(s,a8); s=_mm512_add_epi64(s,a9); \
    s=_mm512_add_epi64(s,aa); s=_mm512_add_epi64(s,ab); \
    g_sink = (uint64_t)_mm512_reduce_add_epi64(s); } while (0)

typedef struct { double secs; double ops; } res_t;

static res_t test_vpermb(long iters) {
    __m512i tbl = _mm512_loadu_si512((const void *)(v_seed + 64));
    DECL12(_mm512_loadu_si512((const void *)v_seed));
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
#define OP(x) x = _mm512_permutexvar_epi8(x, tbl)
        APPLY12(OP); APPLY12(OP);
#undef OP
    }
    double dt = now_sec() - t0;
    SINK12();
    return (res_t){ dt, 24.0 * (double)iters };
}

static res_t test_vpshufb(long iters) {
    __m512i sel = _mm512_and_si512(_mm512_loadu_si512((const void *)(v_seed + 64)),
                                   _mm512_set1_epi8(0x0f));
    DECL12(_mm512_loadu_si512((const void *)v_seed));
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
#define OP(x) x = _mm512_shuffle_epi8(x, sel)
        APPLY12(OP); APPLY12(OP);
#undef OP
    }
    double dt = now_sec() - t0;
    SINK12();
    return (res_t){ dt, 24.0 * (double)iters };
}

static res_t test_vpdpbusd(long iters) {
    __m512i u = _mm512_loadu_si512((const void *)(v_seed + 64));
    __m512i s = _mm512_loadu_si512((const void *)(v_seed + 128));
    DECL12(_mm512_loadu_si512((const void *)v_seed));
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
#define OP(x) x = _mm512_dpbusd_epi32(x, u, s)
        APPLY12(OP); APPLY12(OP);
#undef OP
    }
    double dt = now_sec() - t0;
    SINK12();
    return (res_t){ dt, 24.0 * (double)iters };
}

static res_t test_vpdpwssd(long iters) {
    __m512i u = _mm512_loadu_si512((const void *)(v_seed + 64));
    __m512i s = _mm512_loadu_si512((const void *)(v_seed + 128));
    DECL12(_mm512_loadu_si512((const void *)v_seed));
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
#define OP(x) x = _mm512_dpwssd_epi32(x, u, s)
        APPLY12(OP); APPLY12(OP);
#undef OP
    }
    double dt = now_sec() - t0;
    SINK12();
    return (res_t){ dt, 24.0 * (double)iters };
}

static res_t test_vpopcntq(long iters) {
    DECL12(_mm512_loadu_si512((const void *)v_seed));
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
#define OP(x) x = _mm512_popcnt_epi64(x)
        APPLY12(OP); APPLY12(OP);
#undef OP
    }
    double dt = now_sec() - t0;
    SINK12();
    return (res_t){ dt, 24.0 * (double)iters };
}

static res_t test_vfmadd(long iters) {
    // multiplier must be a RUNTIME value: a constant 1.0f let GCC simplify
    // fmadd(x,1.0,t) -> vaddps (0 fma instructions in the binary before fix).
    // v_fseed values are >= 1.0, so accumulators grow to +inf and stay there
    // (no denormal slowdown).
    __m512 one = _mm512_set1_ps(((volatile float *)v_fseed)[0]);
    __m512 tiny = _mm512_set1_ps(1e-7f);
    __m512 f0 = _mm512_loadu_ps((const void *)v_fseed);
    __m512 f1 = _mm512_add_ps(f0, _mm512_set1_ps(0.001f));
    __m512 f2 = _mm512_add_ps(f0, _mm512_set1_ps(0.002f));
    __m512 f3 = _mm512_add_ps(f0, _mm512_set1_ps(0.003f));
    __m512 f4 = _mm512_add_ps(f0, _mm512_set1_ps(0.004f));
    __m512 f5 = _mm512_add_ps(f0, _mm512_set1_ps(0.005f));
    __m512 f6 = _mm512_add_ps(f0, _mm512_set1_ps(0.006f));
    __m512 f7 = _mm512_add_ps(f0, _mm512_set1_ps(0.007f));
    __m512 f8 = _mm512_add_ps(f0, _mm512_set1_ps(0.008f));
    __m512 f9 = _mm512_add_ps(f0, _mm512_set1_ps(0.009f));
    __m512 fa = _mm512_add_ps(f0, _mm512_set1_ps(0.010f));
    __m512 fb = _mm512_add_ps(f0, _mm512_set1_ps(0.011f));
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
#define OPF(x) x = _mm512_fmadd_ps(x, one, tiny)
        OPF(f0);OPF(f1);OPF(f2);OPF(f3);OPF(f4);OPF(f5);
        OPF(f6);OPF(f7);OPF(f8);OPF(f9);OPF(fa);OPF(fb);
        OPF(f0);OPF(f1);OPF(f2);OPF(f3);OPF(f4);OPF(f5);
        OPF(f6);OPF(f7);OPF(f8);OPF(f9);OPF(fa);OPF(fb);
#undef OPF
    }
    double dt = now_sec() - t0;
    __m512 s = _mm512_add_ps(f0, f1);
    s = _mm512_add_ps(s, f2); s = _mm512_add_ps(s, f3); s = _mm512_add_ps(s, f4);
    s = _mm512_add_ps(s, f5); s = _mm512_add_ps(s, f6); s = _mm512_add_ps(s, f7);
    s = _mm512_add_ps(s, f8); s = _mm512_add_ps(s, f9); s = _mm512_add_ps(s, fa);
    s = _mm512_add_ps(s, fb);
    g_sink = (uint64_t)_mm512_reduce_add_ps(s);
    return (res_t){ dt, 24.0 * (double)iters };
}

// mixed "decode" loop: streaming load from an L1-resident buffer,
// vpermb (LUT decode), vpdpbusd (dot with activations).
// One "group" = load + vpermb + vpdpbusd.
static res_t test_decode(long outer) {
    static uint8_t buf[240 * 64] __attribute__((aligned(64)));
    for (size_t i = 0; i < sizeof(buf); i++) buf[i] = (uint8_t)(v_seed[i & 255] + i);
    const __m512i *p = (const __m512i *)buf;
    __m512i tbl = _mm512_loadu_si512((const void *)(v_seed + 64));
    __m512i act = _mm512_loadu_si512((const void *)(v_seed + 128));
    DECL12(_mm512_setzero_si512());
    double t0 = now_sec();
    for (long o = 0; o < outer; o++) {
        for (int k = 0; k < 240; k += 12) {
#define DEC(x, j) { __m512i v = _mm512_load_si512(p + k + j); \
                    v = _mm512_permutexvar_epi8(v, tbl); \
                    x = _mm512_dpbusd_epi32(x, v, act); }
            DEC(a0, 0) DEC(a1, 1) DEC(a2, 2) DEC(a3, 3) DEC(a4, 4) DEC(a5, 5)
            DEC(a6, 6) DEC(a7, 7) DEC(a8, 8) DEC(a9, 9) DEC(aa, 10) DEC(ab, 11)
#undef DEC
        }
    }
    double dt = now_sec() - t0;
    SINK12();
    return (res_t){ dt, 240.0 * (double)outer };
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <cpu> <reps>\n", argv[0]); return 1; }
    g_cpu = atoi(argv[1]);
    int reps = atoi(argv[2]);
    pin_cpu(g_cpu);
    srandom(12345);
    for (int i = 0; i < 64 * 4; i++) v_seed[i] = (uint8_t)(random() & 0x7f);
    for (int i = 0; i < 48; i++) v_fseed[i] = 1.0f + 1e-3f * (float)(random() % 100);

    g_sampling = 1;
    pthread_t st;
    pthread_create(&st, NULL, sampler, NULL);

    // warm up the AVX-512 license / frequency
    chain_freq_ghz(40 * 1000 * 1000);

    struct { const char *name; res_t (*fn)(long); long iters; } tests[] = {
        { "vpermb",    test_vpermb,   30 * 1000 * 1000 },
        { "vpshufb",   test_vpshufb,  30 * 1000 * 1000 },
        { "vpdpbusd",  test_vpdpbusd, 30 * 1000 * 1000 },
        { "vpdpwssd",  test_vpdpwssd, 30 * 1000 * 1000 },
        { "vpopcntq",  test_vpopcntq, 30 * 1000 * 1000 },
        { "vfmadd231ps", test_vfmadd, 30 * 1000 * 1000 },
        { "decode_mix", test_decode,   3 * 1000 * 1000 },
    };
    int ntests = (int)(sizeof(tests) / sizeof(tests[0]));

    for (int r = 0; r < reps; r++) {
        double f_run = chain_freq_ghz(80 * 1000 * 1000);
        printf("instr_tp,%d,chain_freq,%d,0,0,0,%.4f,1.000,%.1f\n",
               g_cpu, r, f_run, read_cpu_mhz(g_cpu));
        for (int t = 0; t < ntests; t++) {
            double f0 = chain_freq_ghz(20 * 1000 * 1000);
            res_t res = tests[t].fn(tests[t].iters);
            double f1 = chain_freq_ghz(20 * 1000 * 1000);
            double freq = 0.5 * (f0 + f1);
            double gops = res.ops / res.secs / 1e9;
            double opc = gops / freq;
            printf("instr_tp,%d,%s,%d,%.0f,%.6f,%.4f,%.4f,%.4f,%.1f\n",
                   g_cpu, tests[t].name, r, res.ops, res.secs, gops, freq, opc,
                   read_cpu_mhz(g_cpu));
        }
    }
    g_sampling = 0;
    pthread_join(st, NULL);
    if (g_nsamples > 0) {
        double med = median_of(g_mhz_samples, g_nsamples);
        fprintf(stderr, "SUMMARY instr_tp cpu=%d sysfs MHz samples n=%d median=%.1f min=%.1f max=%.1f\n",
                g_cpu, g_nsamples, med, min_of(g_mhz_samples, g_nsamples),
                max_of(g_mhz_samples, g_nsamples));
    } else {
        fprintf(stderr, "SUMMARY instr_tp cpu=%d NO sysfs/procfs MHz samples available\n", g_cpu);
    }
    return 0;
}
