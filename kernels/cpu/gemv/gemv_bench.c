/* P2b µbench for the DOML GEMV kernel — protocol mirrors kernels/cpu/bench_ik:
 *   - REAL tensors from the DPKA artifact (no synthetic weights)
 *   - cycled weight copies >= 384 MB so ny=1 streams from DRAM
 *   - threads pinned t -> CPU t (node0 = even, node1 = odd), weights
 *     first-touched by the exact thread that reads them (row slices)
 *   - >= 9 reps, medians, per-rep CSV rows, PLACEMENT numa_maps evidence
 *   - activations prepped ONCE outside the timed region (bench_ik does the
 *     same for its Q8_K quantization)
 *   - inline correctness check vs the C reference decode + fp64 dot
 *
 * usage:
 *   gemv_bench --variant {fp|i8} --dout N --din K --threads T
 *              [--layer L] [--reps R] [--nbuf N] [--target-ms M] [--seed S]
 *              [--artifact PATH]
 *   gemv_bench --sweep [--reps R] [--target-ms M] [--layer L] [--artifact PATH]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "../fmt/dpka.h"
#include "../ref/ref_decode.h"
#include "doml_gemv.h"

#define DEF_ARTIFACT "downloads/cpu_kernel_rnd/qwen3-0.6b-k31.dpka"

static inline double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline float det_val(uint64_t seed, uint64_t idx)
{
    uint64_t h = splitmix64(seed ^ (0x100000001b3ULL * (idx + 1)));
    return (float)((double)(h >> 11) * (2.0 / 9007199254740992.0) - 1.0);
}

static void print_loadavg(void)
{
    FILE *f = fopen("/proc/loadavg", "r");
    char buf[256] = { 0 };
    if (f) {
        if (fgets(buf, sizeof(buf), f)) { /* keep */ }
        fclose(f);
    }
    fprintf(stderr, "LOADAVG %s", buf);
    double l1 = atof(buf);
    if (l1 > 1.0)
        fprintf(stderr, "WARN: 1-min load %.2f > 1 — numbers from this run "
                        "are NOT citable\n", l1);
}

/* Placement evidence: sum N0=/N1= pages over large anonymous mappings
 * (same method + threshold as bench_ik / roofline). */
static void dump_anon_numa(const char *tag)
{
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f) { fprintf(stderr, "numa_maps unavailable\n"); return; }
    char line[8192];
    long n0 = 0, n1 = 0;
    while (fgets(line, sizeof(line), f)) {
        if (!strstr(line, "anon=")) continue;
        char *p = strstr(line, "anon=");
        long anon = p ? atol(p + 5) : 0;
        if (anon < 1000) continue;
        p = strstr(line, "N0=");
        if (p) n0 += atol(p + 3);
        p = strstr(line, "N1=");
        if (p) n1 += atol(p + 3);
    }
    fclose(f);
    fprintf(stderr,
            "PLACEMENT %s large-anon pages node0=%ld node1=%ld (%.1f%% on node0)\n",
            tag, n0, n1, 100.0 * (double)n0 / (double)(n0 + n1 ? n0 + n1 : 1));
}

static double median_of(double *v, int n)
{
    for (int i = 1; i < n; i++) /* insertion sort, n small */
        for (int j = i; j > 0 && v[j] < v[j - 1]; j--) {
            double t = v[j]; v[j] = v[j - 1]; v[j - 1] = t;
        }
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void *xaligned(size_t n)
{
    void *p = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

/* Qwen3-0.6B linear shapes -> artifact tensor names */
static const struct {
    long dout, din;
    const char *fmt;
    const char *label;
} k_shapes[5] = {
    { 2048, 1024, "model.layers.%d.self_attn.q_proj", "q_proj" },
    { 1024, 1024, "model.layers.%d.self_attn.k_proj", "k/v_proj" },
    { 1024, 2048, "model.layers.%d.self_attn.o_proj", "o_proj" },
    { 3072, 1024, "model.layers.%d.mlp.gate_proj", "gate/up" },
    { 1024, 3072, "model.layers.%d.mlp.down_proj", "down_proj" },
};

typedef enum { VAR_FP = 0, VAR_I8 = 1 } Variant;

typedef struct {
    Variant var;
    int m_full; /* 1: m plane expanded at pack time (streams +0.213 bpw) */
    long dout, din;
    int layer, nth, nbuf, reps;
    double target_ms;
    uint64_t seed;
    const char *artifact;
} Config;

static const char *var_name_full(Variant v, int m_full)
{
    if (m_full) return v == VAR_FP ? "fp_mf" : "i8_mf";
    return v == VAR_FP ? "fp" : "i8";
}

/* ------------------------------------------------------------ pool jobs --- */

typedef struct {
    const DpkaResB *rb;
    const DomlGemvW *wv;
    int nbuf;
} PackJob;

static void pack_job(void *arg, int ith, int nth)
{
    PackJob *pj = (PackJob *)arg;
    uint32_t r0, r1;
    doml_gemv_slice(pj->rb->R, ith, nth, &r0, &r1);
    for (int c = 0; c < pj->nbuf; c++)
        doml_gemv_pack_rows(pj->rb, &pj->wv[c], r0, r1);
}

typedef struct {
    const DomlGemvW *wv;
    int nbuf;
    Variant var;
    const float *xperm;
    const DomlQx *qx;
    float *y;
    int iters, rot;
    DomlPool *pool;
    double t0, t1;
} RepJob;

static void rep_job(void *arg, int ith, int nth)
{
    RepJob *j = (RepJob *)arg;
    doml_pool_barrier(j->pool, ith);
    if (ith == 0) j->t0 = now_sec();
    doml_pool_barrier(j->pool, ith);
    for (int it = 0; it < j->iters; it++) {
        const DomlGemvW *w = &j->wv[(j->rot + it) % j->nbuf];
        uint32_t r0, r1;
        doml_gemv_slice(w->R, ith, nth, &r0, &r1);
        if (j->var == VAR_FP)
            doml_gemv_fp_rows(w, j->xperm, j->y, r0, r1);
        else
            doml_gemv_i8_rows(w, j->qx, j->y, r0, r1);
        doml_pool_barrier(j->pool, ith);
    }
    if (ith == 0) j->t1 = now_sec();
}

static double run_rep(RepJob *j, int iters, int *rot)
{
    j->iters = iters;
    j->rot = *rot;
    doml_pool_run(j->pool, rep_job, j);
    *rot = (*rot + iters) % j->nbuf;
    return j->t1 - j->t0;
}

/* -------------------------------------------------------------- bench ----- */

static int bench_one(const Config *cfg, FILE *csv)
{
    print_loadavg();
    int si = -1;
    for (int i = 0; i < 5; i++)
        if (k_shapes[i].dout == cfg->dout && k_shapes[i].din == cfg->din) si = i;
    if (si < 0) {
        fprintf(stderr, "FATAL: no artifact tensor with shape %ldx%ld\n",
                cfg->dout, cfg->din);
        return 1;
    }
    char name[128];
    snprintf(name, sizeof(name), k_shapes[si].fmt, cfg->layer);

    DpkaFile *f = dpka_open(cfg->artifact);
    int tidx = dpka_find(f, name);
    if (tidx < 0) { fprintf(stderr, "FATAL: tensor %s missing\n", name); return 1; }
    DpkaResB *rb = dpka_build_rb(f, tidx);
    const uint32_t R = rb->R, C = rb->C_orig;

    size_t slab_sz = doml_gemv_slab_bytes(rb, cfg->m_full);
    size_t stride = (slab_sz + 4095) & ~(size_t)4095;
    int nbuf = cfg->nbuf;
    if (nbuf <= 0) {
        size_t target = 384ull << 20;
        long n = (long)((target + stride - 1) / stride);
        nbuf = (int)(n < 1 ? 1 : (n > 512 ? 512 : n));
    }
    uint8_t *slab = (uint8_t *)mmap(NULL, stride * (size_t)nbuf,
                                    PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (slab == MAP_FAILED) { perror("mmap"); return 1; }

    DomlGemvW *wv = (DomlGemvW *)malloc(sizeof(DomlGemvW) * (size_t)nbuf);
    for (int c = 0; c < nbuf; c++)
        doml_gemv_pack_init(rb, slab + (size_t)c * stride, &wv[c], cfg->m_full);

    DomlPool *pool = doml_pool_create(cfg->nth);
    PackJob pj = { rb, wv, nbuf };
    doml_pool_run(pool, pack_job, &pj); /* first-touch by owning threads */

    /* activations + output (prep once, outside the timed region) */
    float *x = (float *)xaligned((size_t)C * 4);
    float *xperm = (float *)xaligned((size_t)C * 4);
    for (uint32_t j = 0; j < C; j++) x[j] = det_val(cfg->seed, j);
    doml_gemv_prep_x_fp(x, C, xperm);
    DomlQx qx;
    qx.q = (int8_t *)xaligned(C);
    qx.dx = (float *)xaligned(sizeof(float) * 16);
    qx.c128 = (float *)xaligned(sizeof(float) * 16);
    doml_gemv_prep_x_i8(x, C, &qx);
    float *y = (float *)xaligned((size_t)R * 4);
    memset(y, 0, (size_t)R * 4);

    double bpw = 8.0 * (double)wv[0].weight_bytes / ((double)R * C);
    fprintf(stderr,
            "SETUP doml_gemv variant=%s tensor=%s dout=%u din=%u ny=1 nth=%d "
            "nbuf=%d weight_bytes=%zu (%.4f bpw resident) slab_stride=%zu\n",
            var_name_full(cfg->var, cfg->m_full), name, R, C, cfg->nth, nbuf,
            wv[0].weight_bytes, bpw, stride);

    /* inline correctness check (1 call) vs C reference decode + fp64 dot */
    RepJob rj = { wv, nbuf, cfg->var, xperm, &qx, y, 0, 0, pool, 0, 0 };
    int rot = 0;
    run_rep(&rj, 1, &rot);
    {
        uint16_t tab[256];
        dpka_fp8e4m3_to_bf16_table(tab);
        float *wrow = (float *)xaligned((size_t)C * 4);
        double ss_ref = 0, ss_err = 0, max_abs = 0;
        for (uint32_t r = 0; r < R; r++) {
            dpka_ref_decode_row_rb(rb, tab, r, wrow);
            double acc = 0;
            for (uint32_t j = 0; j < C; j++)
                acc += (double)wrow[j] * (double)x[j];
            double e = fabs((double)y[r] - acc);
            ss_ref += acc * acc;
            ss_err += e * e;
            if (e > max_abs) max_abs = e;
        }
        double rms_ref = sqrt(ss_ref / (double)R);
        double max_nrm = max_abs / (rms_ref > 0 ? rms_ref : 1.0);
        double rms_rel = sqrt(ss_err / (ss_ref > 0 ? ss_ref : 1.0));
        /* sanity guard only — the numeric gates live in gemv_test. i8 max_nrm
         * on real tensors measures up to ~7e-2 (single-output tails of the
         * ~0.7% RMS class), so the guard sits at 1.5e-1. */
        double tol = cfg->var == VAR_FP ? 2e-5 : 1.5e-1;
        fprintf(stderr, "CHECK variant=%s max_nrm=%.3e rms_rel=%.3e (tol %.0e) %s\n",
                var_name_full(cfg->var, cfg->m_full), max_nrm, rms_rel, tol,
                max_nrm <= tol ? "PASS" : "FAIL");
        free(wrow);
        if (max_nrm > tol) {
            fprintf(stderr, "FATAL: inline correctness check failed\n");
            exit(1);
        }
    }

    dpka_free_rb(rb); /* large malloc'd planes released before placement dump */
    rb = NULL;
    dpka_close(f);
    dump_anon_numa(var_name_full(cfg->var, cfg->m_full));

    /* warmup + calibrate so one rep lasts >= target_ms (bench_ik logic) */
    run_rep(&rj, 2, &rot);
    int iters = 1;
    for (int tries = 0; tries < 4; tries++) {
        double t = run_rep(&rj, iters, &rot);
        if (t >= cfg->target_ms * 1e-3 || iters >= (1 << 22)) break;
        double scale = (cfg->target_ms * 1.2e-3) / (t > 1e-7 ? t : 1e-7);
        double ni = (double)iters * (scale > 2.0 ? scale : 2.0);
        iters = (int)(ni < (double)(1 << 22) ? ni : (double)(1 << 22));
    }

    double nsc[64] = { 0 }, gbw[64] = { 0 }, gmc[64] = { 0 };
    int reps = cfg->reps > 64 ? 64 : (cfg->reps < 1 ? 1 : cfg->reps);
    for (int r = 0; r < reps; r++) {
        double secs = run_rep(&rj, iters, &rot);
        double per_call = secs / iters;
        nsc[r] = per_call * 1e9;
        gbw[r] = (double)wv[0].weight_bytes / per_call / 1e9;
        gmc[r] = (double)R * C / per_call / 1e9;
        if (csv)
            fprintf(csv, "doml_gemv,%s,%u,%u,1,%d,%d,%d,%d,%.6f,%.1f,%.3f,%.3f\n",
                    var_name_full(cfg->var, cfg->m_full), R, C, cfg->nth, nbuf, r, iters, secs,
                    nsc[r], gbw[r], gmc[r]);
    }
    double mn = nsc[0], mx = nsc[0];
    for (int r = 1; r < reps; r++) {
        if (nsc[r] < mn) mn = nsc[r];
        if (nsc[r] > mx) mx = nsc[r];
    }
    double med_ns = median_of(nsc, reps);
    double med_gb = median_of(gbw, reps);
    double med_gm = median_of(gmc, reps);
    fprintf(stderr,
            "SUMMARY doml_gemv type=%-3s dout=%-4u din=%-4u ny=1   nth=%-2d nbuf=%-3d iters=%d "
            "median=%.1f ns/call [%.1f,%.1f]  weightBW=%.2f GB/s  %.2f GMAC/s  %.1f calls/s\n",
            var_name_full(cfg->var, cfg->m_full), R, C, cfg->nth, nbuf, iters, med_ns, mn, mx,
            med_gb, med_gm, 1e9 / med_ns);

    doml_pool_destroy(pool);
    munmap(slab, stride * (size_t)nbuf);
    free(wv);
    free(x); free(xperm); free(qx.q); free(qx.dx); free(qx.c128); free(y);
    return 0;
}

/* --------------------------------------------------------------- main ----- */

static void usage(const char *a0)
{
    fprintf(stderr,
            "usage:\n"
            "  %s --variant {fp|i8} [--mfull] --dout N --din K --threads T [--layer L]\n"
            "        [--reps R] [--nbuf N] [--target-ms M] [--seed S] [--artifact P]\n"
            "  %s --sweep [--reps R] [--target-ms M] [--layer L] [--artifact P]\n",
            a0, a0);
}

int main(int argc, char **argv)
{
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.var = VAR_FP;
    cfg.layer = 0;
    cfg.nth = 1;
    cfg.nbuf = 0;
    cfg.reps = 9;
    cfg.target_ms = 60.0;
    cfg.seed = 20260715ULL;
    cfg.artifact = DEF_ARTIFACT;
    int sweep = 0, have_var = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char *)0))
        if (!strcmp(a, "--sweep")) sweep = 1;
        else if (!strcmp(a, "--variant")) {
            const char *v = NEXT();
            if (!strcmp(v, "fp")) cfg.var = VAR_FP;
            else if (!strcmp(v, "i8")) cfg.var = VAR_I8;
            else { fprintf(stderr, "unknown variant %s\n", v); return 1; }
            have_var = 1;
        }
        else if (!strcmp(a, "--mfull")) cfg.m_full = 1;
        else if (!strcmp(a, "--dout")) cfg.dout = atol(NEXT());
        else if (!strcmp(a, "--din")) cfg.din = atol(NEXT());
        else if (!strcmp(a, "--threads")) cfg.nth = atoi(NEXT());
        else if (!strcmp(a, "--layer")) cfg.layer = atoi(NEXT());
        else if (!strcmp(a, "--reps")) cfg.reps = atoi(NEXT());
        else if (!strcmp(a, "--nbuf")) cfg.nbuf = atoi(NEXT());
        else if (!strcmp(a, "--target-ms")) cfg.target_ms = atof(NEXT());
        else if (!strcmp(a, "--seed")) cfg.seed = strtoull(NEXT(), NULL, 0);
        else if (!strcmp(a, "--artifact")) cfg.artifact = NEXT();
        else { usage(argv[0]); return 1; }
#undef NEXT
    }

    doml_gemv_init();
    printf("kind,type,dout,din,ny,nth,nbuf,rep,iters,secs,ns_call,weight_GBps,GMACs\n");
    fflush(stdout);

    if (sweep) {
        fprintf(stderr, "=== SWEEP doml_gemv: 5 shapes x {fp,i8} x threads {1,24} (layer %d) ===\n",
                cfg.layer);
        const int nts[2] = { 1, 24 };
        for (int mf = 0; mf < 2; mf++)
            for (int v = 0; v < 2; v++)
                for (int s = 0; s < 5; s++)
                    for (int t = 0; t < 2; t++) {
                        Config c = cfg;
                        c.var = (Variant)v;
                        c.m_full = mf;
                        c.dout = k_shapes[s].dout;
                        c.din = k_shapes[s].din;
                        c.nth = nts[t];
                        c.nbuf = 0;
                        if (bench_one(&c, stdout)) return 1;
                        fflush(stdout);
                    }
        return 0;
    }

    if (!have_var || cfg.dout <= 0 || cfg.din <= 0 || cfg.nth <= 0) {
        usage(argv[0]);
        return 1;
    }
    return bench_one(&cfg, stdout);
}
