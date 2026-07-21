/* P2c µbench for the v2 fused-slab GEMV — protocol mirrors kernels/cpu/bench_ik
 * (and P2b's gemv_bench):
 *   - REAL tensors from the DPKA artifact (no synthetic weights)
 *   - cycled weight copies >= 384 MB so ny=1 streams from DRAM
 *   - threads pinned t -> CPU t (node0 = even, node1 = odd); each slab copy's
 *     tile range is first-touched by the exact thread that reads it
 *   - >= 9 reps, medians, per-rep CSV rows, PLACEMENT numa_maps evidence
 *   - activations prepped ONCE outside the timed region (bench_ik liberty)
 *   - inline correctness check vs the C reference decode before timing
 *   - ONE barrier per timed call (bench_ik pays one omp barrier per call)
 *
 * usage:
 *   gemv2_bench --variant {fp|i8} [--mfull] [--steal [--chunk N]]
 *               --dout N --din K --threads T [--layer L] [--reps R]
 *               [--nbuf N] [--target-ms M] [--seed S] [--artifact P]
 *   gemv2_bench --sweep  [--reps R] [--target-ms M] [--layer L] [--steal] ...
 *   gemv2_bench --curve  [same knobs; i8 2048x1024 x threads {1,6,12,24,48}]
 *   gemv2_bench --barrier-bench   null-kernel barrier round-trip, pool v2
 *               (bounded spin + futex) vs pool v1 (P2b pure spin), at
 *               threads {6,12,24,48}
 */
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "../fmt/dpka.h"
#include "../ref/ref_decode.h"
#include "doml_gemv2.h"

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
}

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
    for (int i = 1; i < n; i++)
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

#define N_SHAPES 9
static const struct {
    long dout, din;
    const char *fmt;
    const char *label;
} k_shapes[N_SHAPES] = {
    /* Qwen3-0.6B */
    { 2048, 1024, "model.layers.%d.self_attn.q_proj", "q_proj" },
    { 1024, 1024, "model.layers.%d.self_attn.k_proj", "k/v_proj" },
    { 1024, 2048, "model.layers.%d.self_attn.o_proj", "o_proj" },
    { 3072, 1024, "model.layers.%d.mlp.gate_proj", "gate/up" },
    { 1024, 3072, "model.layers.%d.mlp.down_proj", "down_proj" },
    /* [H17-C] Qwen3-1.7B (q/o share 2048x2048 -> one row; 1024x2048 also
     * names the 0.6B o_proj row above -> resolution is artifact-aware) */
    { 2048, 2048, "model.layers.%d.self_attn.q_proj", "q/o_proj" },
    { 1024, 2048, "model.layers.%d.self_attn.k_proj", "k/v_proj" },
    { 6144, 2048, "model.layers.%d.mlp.gate_proj", "gate/up" },
    { 2048, 6144, "model.layers.%d.mlp.down_proj", "down_proj" },
};

/* [H17-C] resolve requested dims against the artifact: the same (dout,din)
 * can map to different tensors per model (0.6B o_proj vs 1.7B k/v are both
 * 1024x2048), so a row is valid only if its named tensor exists in the
 * artifact with exactly the requested dims. Returns the row index and fills
 * name[128], or -1. */
static int shape_resolve(const DpkaFile *f, long dout, long din, int layer,
                         char name[128])
{
    for (int i = 0; i < N_SHAPES; i++) {
        if (k_shapes[i].dout != dout || k_shapes[i].din != din) continue;
        snprintf(name, 128, k_shapes[i].fmt, layer);
        int ti = dpka_find(f, name);
        if (ti < 0) continue;
        if (f->toc[ti].R != (uint32_t)dout ||
            f->toc[ti].C_orig != (uint32_t)din) continue;
        return i;
    }
    return -1;
}

typedef struct {
    Doml2Var var;
    int m_full, steal, chunk;
    long dout, din;
    int layer, nth, nbuf, reps;
    double target_ms;
    uint64_t seed;
    const char *artifact;
} Config;

static const char *var_name(Doml2Var v, int mf, int steal)
{
    static char buf[32];
    snprintf(buf, sizeof(buf), "%s%s%s", v == DOML2_VAR_FP ? "fp" : "i8",
             mf ? "_mf" : "", steal ? "_st" : "");
    return buf;
}

/* ------------------------------------------------------------ pool jobs --- */

typedef struct {
    const DpkaResB *rb;
    Doml2W *wv;
    uint8_t *slab0;
    size_t stride;
    int nbuf;
} PackJob;

/* copy 0 is derived from the R-B planes by its owning thread; the other
 * cycled copies are byte-identical -> owner memcpy (same first-touch) */
static void pack_job(void *arg, int ith, int nth)
{
    PackJob *pj = (PackJob *)arg;
    uint32_t t0, t1;
    doml2_slice(pj->wv[0].ntiles, ith, nth, &t0, &t1);
    doml2_pack_tiles(pj->rb, &pj->wv[0], t0, t1);
    const size_t head = (size_t)(pj->wv[0].blocks - pj->wv[0].s);
    const size_t off0 = head + pj->wv[0].tileoff[t0];
    const size_t off1 = head + pj->wv[0].tileoff[t1];
    for (int c = 1; c < pj->nbuf; c++)
        memcpy(pj->slab0 + (size_t)c * pj->stride + off0,
               pj->slab0 + off0, off1 - off0);
}

typedef struct {
    const Doml2W *wv;
    int nbuf;
    Doml2Var var;
    int steal, chunk;
    const float *xperm;
    const DomlQx *qx;
    float *y;
    int iters, rot;
    Doml2Pool *pool;
    Doml2Steal *ws;
    /* static mode: remainder-tile pool. ntiles % nth tiles would otherwise
     * make some threads one tile taller (10.7 -> 11 tiles at 1024 rows /
     * 24t = ~5-8%% tail); instead every thread gets floor(ntiles/nth) and
     * the remainder is grabbed 1 tile at a time from an atomic cursor,
     * double-buffered by call parity (each thread re-arms the OTHER parity
     * before the completion barrier). Outputs are bitwise identical
     * regardless of executor. */
    _Atomic uint32_t rem[2];
    double t0, t1;
} RepJob;

static void rep_job(void *arg, int ith, int nth)
{
    RepJob *j = (RepJob *)arg;
    doml2_pool_barrier(j->pool, ith);
    if (ith == 0) j->t0 = now_sec();
    doml2_pool_barrier(j->pool, ith);
    for (int it = 0; it < j->iters; it++) {
        const int call = j->rot + it;
        const Doml2W *w = &j->wv[call % j->nbuf];
        if (j->steal) {
            doml2_steal_gemv(j->ws, call & 1, j->chunk, w, j->xperm, j->qx,
                             j->y, ith, nth);
        } else {
            /* static slices MUST be doml2_slice: identical to the ranges the
             * pack phase first-touched, or up to min(t,rem) tiles per thread
             * are read cross-socket every call. (P2d root cause: a
             * "remainder-pool" dispatch used [t*base,(t+1)*base) here while
             * pack kept doml2_slice ownership; the +8-tile shear sent ~25%
             * of the slab stream remote and cost +2 us/call at 24t --
             * masked at first by a silently failed relative-path make.) */
            uint32_t t0, t1;
            doml2_slice(w->ntiles, ith, nth, &t0, &t1);
            if (j->var == DOML2_VAR_FP)
                doml2_gemv_fp_tiles(w, j->xperm, j->y, t0, t1);
            else
                doml2_gemv_i8_tiles(w, j->qx, j->y, t0, t1);
        }
        doml2_pool_barrier(j->pool, ith); /* ONE barrier per call */
    }
    if (ith == 0) j->t1 = now_sec();
}

static double run_rep(RepJob *j, int iters, int *rot)
{
    j->iters = iters;
    j->rot = *rot;
    doml2_pool_run(j->pool, rep_job, j);
    *rot += iters; /* NOT wrapped: steal parity must keep alternating */
    return j->t1 - j->t0;
}

/* -------------------------------------------------------------- bench ----- */

static int bench_one(const Config *cfg, FILE *csv)
{
    print_loadavg();
    DpkaFile *f = dpka_open(cfg->artifact);
    char name[128];
    int si = shape_resolve(f, cfg->dout, cfg->din, cfg->layer, name);
    if (si < 0) {
        fprintf(stderr, "FATAL: no artifact tensor with shape %ldx%ld\n",
                cfg->dout, cfg->din);
        return 1;
    }
    int tidx = dpka_find(f, name);
    if (tidx < 0) { fprintf(stderr, "FATAL: tensor %s missing\n", name); return 1; }
    DpkaResB *rb = dpka_build_rb(f, tidx);
    const uint32_t R = rb->R, C = rb->C_orig;
    const uint32_t ntiles = R / DOML2_TILE;

    uint32_t *toff = (uint32_t *)xaligned(((size_t)ntiles + 1) * 4);
    size_t slab_sz = doml2_slab_bytes(rb, cfg->var, cfg->m_full, toff, NULL);
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

    Doml2W *wv = (Doml2W *)malloc(sizeof(Doml2W) * (size_t)nbuf);
    for (int c = 0; c < nbuf; c++)
        doml2_pack_init(rb, cfg->var, cfg->m_full,
                        slab + (size_t)c * stride, toff, &wv[c]);

    Doml2Pool *pool = doml2_pool_create(cfg->nth);
    PackJob pj = { rb, wv, slab, stride, nbuf };
    doml2_pool_run(pool, pack_job, &pj); /* first-touch by owning threads */

    float *x = (float *)xaligned((size_t)C * 4);
    float *xperm = (float *)xaligned((size_t)C * 4);
    for (uint32_t j = 0; j < C; j++) x[j] = det_val(cfg->seed, j);
    doml_gemv_prep_x_fp(x, C, xperm);
    DomlQx qx;
    qx.q = (int8_t *)xaligned(C);
    /* [H17-C] NG floats needed (24 at 1.7B down_proj); +4 slack as in the
     * fork's stack arrays. Was a fixed 16. */
    qx.dx = (float *)xaligned(sizeof(float) * (DOML2_NG_MAX + 4));
    qx.c128 = (float *)xaligned(sizeof(float) * (DOML2_NG_MAX + 4));
    doml_gemv_prep_x_i8(x, C, &qx);
    float *y = (float *)xaligned((size_t)R * 4);
    memset(y, 0, (size_t)R * 4);

    Doml2Steal *ws = (Doml2Steal *)xaligned(sizeof(Doml2Steal));
    doml2_steal_init(ws, ntiles, cfg->nth);

    double bpw_w = 8.0 * (double)wv[0].weight_bytes / ((double)R * C);
    double bpw_a = 8.0 * (double)wv[0].slab_bytes / ((double)R * C);
    fprintf(stderr,
            "SETUP doml2_gemv variant=%s tensor=%s dout=%u din=%u ny=1 nth=%d "
            "nbuf=%d weight_bytes=%zu (%.4f bpw consumed, %.4f bpw allocated) "
            "slab_stride=%zu chunk=%d\n",
            var_name(cfg->var, cfg->m_full, cfg->steal), name, R, C, cfg->nth,
            nbuf, wv[0].weight_bytes, bpw_w, bpw_a, stride, cfg->chunk);

    /* inline correctness check (1 call) vs C reference decode + fp64 dot */
    RepJob rj;
    memset(&rj, 0, sizeof(rj));
    rj.wv = wv;
    rj.nbuf = nbuf;
    rj.var = cfg->var;
    rj.steal = cfg->steal;
    rj.chunk = cfg->chunk;
    rj.xperm = xperm;
    rj.qx = &qx;
    rj.y = y;
    rj.pool = pool;
    rj.ws = ws;
    {
        uint32_t start = (ntiles / (uint32_t)cfg->nth) * (uint32_t)cfg->nth;
        atomic_store(&rj.rem[0], start);
        atomic_store(&rj.rem[1], start);
    }
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
        double tol = cfg->var == DOML2_VAR_FP ? 2e-5 : 1.5e-1;
        fprintf(stderr, "CHECK variant=%s max_nrm=%.3e rms_rel=%.3e (tol %.0e) %s\n",
                var_name(cfg->var, cfg->m_full, cfg->steal), max_nrm, rms_rel,
                tol, max_nrm <= tol ? "PASS" : "FAIL");
        free(wrow);
        if (max_nrm > tol) {
            fprintf(stderr, "FATAL: inline correctness check failed\n");
            exit(1);
        }
    }

    dpka_free_rb(rb);
    rb = NULL;
    dpka_close(f);
    dump_anon_numa(var_name(cfg->var, cfg->m_full, cfg->steal));

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
            fprintf(csv, "doml2_gemv,%s,%u,%u,1,%d,%d,%d,%d,%.6f,%.1f,%.3f,%.3f\n",
                    var_name(cfg->var, cfg->m_full, cfg->steal), R, C,
                    cfg->nth, nbuf, r, iters, secs, nsc[r], gbw[r], gmc[r]);
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
            "SUMMARY doml2_gemv type=%-6s dout=%-4u din=%-4u ny=1   nth=%-2d nbuf=%-3d iters=%d "
            "median=%.1f ns/call [%.1f,%.1f]  weightBW=%.2f GB/s  %.2f GMAC/s  %.1f calls/s\n",
            var_name(cfg->var, cfg->m_full, cfg->steal), R, C, cfg->nth, nbuf,
            iters, med_ns, mn, mx, med_gb, med_gm, 1e9 / med_ns);

    doml2_pool_destroy(pool);
    munmap(slab, stride * (size_t)nbuf);
    free(wv);
    free(toff);
    free(ws);
    free(x); free(xperm); free(qx.q); free(qx.dx); free(qx.c128); free(y);
    return 0;
}

/* ------------------------------------------------------- barrier bench ---- */

typedef struct {
    Doml2Pool *p2;
    DomlPool *p1;
    int iters;
    double t0, t1;
} BarJob;

static void bar_job2(void *arg, int ith, int nth)
{
    (void)nth;
    BarJob *j = (BarJob *)arg;
    doml2_pool_barrier(j->p2, ith);
    if (ith == 0) j->t0 = now_sec();
    doml2_pool_barrier(j->p2, ith);
    for (int it = 0; it < j->iters; it++) doml2_pool_barrier(j->p2, ith);
    if (ith == 0) j->t1 = now_sec();
}

static void bar_job1(void *arg, int ith, int nth)
{
    (void)nth;
    BarJob *j = (BarJob *)arg;
    doml_pool_barrier(j->p1, ith);
    if (ith == 0) j->t0 = now_sec();
    doml_pool_barrier(j->p1, ith);
    for (int it = 0; it < j->iters; it++) doml_pool_barrier(j->p1, ith);
    if (ith == 0) j->t1 = now_sec();
}

static void barrier_bench(int reps, double target_ms)
{
    static const int nts[4] = { 6, 12, 24, 48 };
    printf("pool,nth,rep,iters,secs,ns_barrier\n");
    for (int pi = 0; pi < 2; pi++) {
        for (int ti = 0; ti < 4; ti++) {
            const int nth = nts[ti];
            print_loadavg();
            BarJob j;
            memset(&j, 0, sizeof(j));
            if (pi == 0)
                j.p2 = doml2_pool_create(nth);
            else
                j.p1 = doml_pool_create(nth);
            int iters = 16;
            for (int tries = 0; tries < 6; tries++) {
                j.iters = iters;
                if (pi == 0)
                    doml2_pool_run(j.p2, bar_job2, &j);
                else
                    doml_pool_run(j.p1, bar_job1, &j);
                double t = j.t1 - j.t0;
                if (t >= target_ms * 1e-3 || iters >= (1 << 22)) break;
                iters *= 4;
            }
            double ns[64] = { 0 };
            int nr = reps > 64 ? 64 : (reps < 1 ? 1 : reps);
            for (int r = 0; r < nr; r++) {
                j.iters = iters;
                if (pi == 0)
                    doml2_pool_run(j.p2, bar_job2, &j);
                else
                    doml_pool_run(j.p1, bar_job1, &j);
                ns[r] = (j.t1 - j.t0) / iters * 1e9;
                printf("%s,%d,%d,%d,%.6f,%.1f\n",
                       pi == 0 ? "v2_futex" : "v1_spin", nth, r, iters,
                       j.t1 - j.t0, ns[r]);
            }
            double mn = ns[0], mx = ns[0];
            for (int r = 1; r < nr; r++) {
                if (ns[r] < mn) mn = ns[r];
                if (ns[r] > mx) mx = ns[r];
            }
            fprintf(stderr,
                    "SUMMARY barrier pool=%s nth=%-2d iters=%d median=%.1f "
                    "ns/barrier [%.1f,%.1f]\n",
                    pi == 0 ? "v2_futex" : "v1_spin", nth, iters,
                    median_of(ns, nr), mn, mx);
            if (pi == 0)
                doml2_pool_destroy(j.p2);
            else
                doml_pool_destroy(j.p1);
        }
    }
}

/* --------------------------------------------------------------- main ----- */

static void usage(const char *a0)
{
    fprintf(stderr,
            "usage:\n"
            "  %s --variant {fp|i8} [--mfull] [--steal] [--chunk N] --dout N --din K\n"
            "        --threads T [--layer L] [--reps R] [--nbuf N] [--target-ms M]\n"
            "        [--seed S] [--artifact P]\n"
            "  %s --sweep [--reps R] [--target-ms M] [--layer L] [--steal] [--artifact P]\n"
            "  %s --curve [--variant V] [--steal] [--reps R] [--artifact P]\n"
            "  %s --barrier-bench [--reps R]\n",
            a0, a0, a0, a0);
}

int main(int argc, char **argv)
{
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.var = DOML2_VAR_I8;
    cfg.chunk = 8;
    cfg.layer = 0;
    cfg.nth = 1;
    cfg.nbuf = 0;
    cfg.reps = 9;
    cfg.target_ms = 60.0;
    cfg.seed = 20260715ULL;
    cfg.artifact = DEF_ARTIFACT;
    int sweep = 0, curve = 0, barrier = 0, have_var = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char *)0))
        if (!strcmp(a, "--sweep")) sweep = 1;
        else if (!strcmp(a, "--curve")) curve = 1;
        else if (!strcmp(a, "--barrier-bench")) barrier = 1;
        else if (!strcmp(a, "--variant")) {
            const char *v = NEXT();
            if (!strcmp(v, "fp")) cfg.var = DOML2_VAR_FP;
            else if (!strcmp(v, "i8")) cfg.var = DOML2_VAR_I8;
            else { fprintf(stderr, "unknown variant %s\n", v); return 1; }
            have_var = 1;
        }
        else if (!strcmp(a, "--mfull")) cfg.m_full = 1;
        else if (!strcmp(a, "--steal")) cfg.steal = 1;
        else if (!strcmp(a, "--chunk")) cfg.chunk = atoi(NEXT());
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

    doml2_init();
    doml_gemv_init();

    if (barrier) {
        barrier_bench(cfg.reps, 20.0);
        return 0;
    }

    printf("kind,type,dout,din,ny,nth,nbuf,rep,iters,secs,ns_call,weight_GBps,GMACs\n");
    fflush(stdout);

    if (sweep) {
        fprintf(stderr,
                "=== SWEEP doml2_gemv: artifact shapes x {i8,fp} x threads {1,24}%s (layer %d) ===\n",
                cfg.steal ? " [steal]" : "", cfg.layer);
        const int nts[2] = { 1, 24 };
        const Doml2Var vars[2] = { DOML2_VAR_I8, DOML2_VAR_FP };
        DpkaFile *af = dpka_open(cfg.artifact);
        for (int v = 0; v < 2; v++)
            for (int s = 0; s < N_SHAPES; s++) {
                /* [H17-C] skip rows absent from this artifact (and shadowed
                 * duplicates: resolve must land on row s itself) */
                char nbuf[128];
                if (shape_resolve(af, k_shapes[s].dout, k_shapes[s].din,
                                  cfg.layer, nbuf) != s)
                    continue;
                for (int t = 0; t < 2; t++) {
                    Config c = cfg;
                    c.var = vars[v];
                    c.dout = k_shapes[s].dout;
                    c.din = k_shapes[s].din;
                    c.nth = nts[t];
                    c.nbuf = 0;
                    if (c.nth == 1) c.steal = 0; /* stealing needs >1 thread */
                    if (bench_one(&c, stdout)) return 1;
                    fflush(stdout);
                }
            }
        dpka_close(af);
        return 0;
    }

    if (curve) {
        fprintf(stderr,
                "=== CURVE doml2_gemv: %s 2048x1024 x threads {1,6,12,24,48}%s ===\n",
                cfg.var == DOML2_VAR_FP ? "fp" : "i8",
                cfg.steal ? " [steal]" : "");
        const int nts[5] = { 1, 6, 12, 24, 48 };
        for (int t = 0; t < 5; t++) {
            Config c = cfg;
            c.dout = 2048;
            c.din = 1024;
            c.nth = nts[t];
            c.nbuf = 0;
            if (c.nth == 1) c.steal = 0;
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
