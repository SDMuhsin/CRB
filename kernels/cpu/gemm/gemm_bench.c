/* P3 µbench for the DOML prefill GEMM — protocol mirrors kernels/cpu/bench_ik
 * at ny=512 (and gemv2_bench):
 *   - REAL tensors from the DPKA artifact (no synthetic weights)
 *   - cycled weight copies >= 384 MB (the conversion streams a fresh slab
 *     copy per call, like a real 28-layer prefill pass)
 *   - threads pinned t -> CPU t (pool v2); each slab copy's tile range is
 *     first-touched by the exact thread that CONVERTS it (16-row-group
 *     slices — the GEMM ownership, not the GEMV tile slices)
 *   - fp32 activations generated outside; int8 quantization of the
 *     activations runs INSIDE the timed region every call (brief: prefill
 *     activations change every call). NOTE bench_ik quantizes its Q8_K
 *     activations OUTSIDE its timed region — the paired comparison is
 *     therefore conservative against us by the quant cost (reported by
 *     --split).
 *   - >= 9 reps, medians, per-rep CSV rows, PLACEMENT numa_maps evidence
 *   - inline correctness check vs the C reference decode -> fp64 GEMM
 *   - two pool barriers per call (post-quant + completion; ik's engine pays
 *     a post-quant barrier + the node barrier as well)
 *
 * usage:
 *   gemm_bench --dout N --din K [--ny 512] --threads T [--mk 0|1] [--split]
 *              [--layer L] [--reps R] [--nbuf N] [--target-ms M] [--seed S]
 *              [--artifact P]
 *   gemm_bench --sweep [--reps R] [--target-ms M] ...   (5 shapes x {24,48}t)
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
#include "doml_gemm.h"

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
} k_shapes[N_SHAPES] = {
    /* Qwen3-0.6B */
    { 2048, 1024, "model.layers.%d.self_attn.q_proj" },
    { 1024, 1024, "model.layers.%d.self_attn.k_proj" },
    { 1024, 2048, "model.layers.%d.self_attn.o_proj" },
    { 3072, 1024, "model.layers.%d.mlp.gate_proj" },
    { 1024, 3072, "model.layers.%d.mlp.down_proj" },
    /* [H17-C] Qwen3-1.7B (q/o share 2048x2048 -> one row; 1024x2048 also
     * names the 0.6B o_proj row above -> resolution is artifact-aware) */
    { 2048, 2048, "model.layers.%d.self_attn.q_proj" },
    { 1024, 2048, "model.layers.%d.self_attn.k_proj" },
    { 6144, 2048, "model.layers.%d.mlp.gate_proj" },
    { 2048, 6144, "model.layers.%d.mlp.down_proj" },
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
    long dout, din, ny;
    int layer, nth, nbuf, reps, mk, split;
    double target_ms;
    uint64_t seed;
    const char *artifact;
} Config;

/* ------------------------------------------------------------ pool jobs --- */

typedef struct {
    const DpkaResB *rb;
    Doml2W *wv;
    uint8_t *slab0;
    size_t stride;
    int nbuf;
} PackJob;

/* first-touch by the GEMM/convert ownership: 16-row-group slices (NOT the
 * GEMV 4-row-tile slices — a shear between pack and compute ownership sends
 * boundary tiles cross-socket every call; P2d root cause) */
static void pack_job(void *arg, int ith, int nth)
{
    PackJob *pj = (PackJob *)arg;
    uint32_t rg0, rg1;
    doml2_slice(pj->wv[0].R / 16, ith, nth, &rg0, &rg1);
    doml2_pack_tiles(pj->rb, &pj->wv[0], rg0 * 4, rg1 * 4);
    const size_t head = (size_t)(pj->wv[0].blocks - pj->wv[0].s);
    const size_t off0 = head + pj->wv[0].tileoff[rg0 * 4];
    const size_t off1 = head + pj->wv[0].tileoff[rg1 * 4];
    for (int c = 1; c < pj->nbuf; c++)
        memcpy(pj->slab0 + (size_t)c * pj->stride + off0,
               pj->slab0 + off0, off1 - off0);
}

typedef struct {
    float *X;
    float *C;
    int8_t *xq;
    uint32_t ny, K, R;
    uint64_t seed;
} TouchJob;

/* first-touch X/xq by activation-slice owner, C by row-slice owner */
static void touch_job(void *arg, int ith, int nth)
{
    TouchJob *tj = (TouchJob *)arg;
    uint32_t ya, yb;
    doml2_slice(tj->ny, ith, nth, &ya, &yb);
    for (uint32_t y = ya; y < yb; y++) {
        for (uint32_t j = 0; j < tj->K; j++)
            tj->X[(size_t)y * tj->K + j] =
                det_val(tj->seed + 0x5eedULL, (uint64_t)y * tj->K + j);
        memset(tj->xq + (size_t)y * tj->K, 0, tj->K);
    }
    uint32_t rg0, rg1;
    doml2_slice(tj->R / 16, ith, nth, &rg0, &rg1);
    for (uint32_t y = 0; y < tj->ny; y++)
        memset(tj->C + (size_t)y * tj->R + rg0 * 16, 0,
               (size_t)(rg1 - rg0) * 16 * 4);
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
    const uint32_t R = rb->R, K = rb->C_orig, NG = rb->NG;
    const uint32_t ny = (uint32_t)cfg->ny;
    const uint32_t ntiles = R / DOML2_TILE;

    uint32_t *toff = (uint32_t *)xaligned(((size_t)ntiles + 1) * 4);
    size_t slab_sz = doml2_slab_bytes(rb, DOML2_VAR_I8, 0, toff, NULL);
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
        doml2_pack_init(rb, DOML2_VAR_I8, 0,
                        slab + (size_t)c * stride, toff, &wv[c]);

    Doml2Pool *pool = doml2_pool_create(cfg->nth);
    PackJob pj = { rb, wv, slab, stride, nbuf };
    doml2_pool_run(pool, pack_job, &pj);

    /* activations / outputs / panels */
    float *X = (float *)xaligned((size_t)ny * K * 4);
    float *C = (float *)xaligned((size_t)ny * R * 4);
    Doml3X qx;
    doml3_x_alloc(&qx, ny, K);
    TouchJob tj = { X, C, qx.q, ny, K, R, cfg->seed };
    doml2_pool_run(pool, touch_job, &tj);

    uint32_t strip = doml3_strip_rows(K);
    if (cfg->split) {
        uint32_t n16 = R / 16;
        uint32_t maxg = n16 / (uint32_t)cfg->nth + (n16 % cfg->nth ? 1 : 0);
        strip = ((maxg * 16 + 31) / 32) * 32;
    }
    Doml3Panel *pans =
        (Doml3Panel *)xaligned(sizeof(Doml3Panel) * (size_t)cfg->nth);
    for (int t = 0; t < cfg->nth; t++)
        doml3_panel_alloc(&pans[t], strip, K, NG);

    double bpw_w = 8.0 * (double)wv[0].weight_bytes / ((double)R * K);
    size_t xq_bytes = (size_t)ny * K + 2 * (size_t)ny * NG * 4;
    fprintf(stderr,
            "SETUP doml3_gemm mk=%d tensor=%s dout=%u din=%u ny=%u nth=%d "
            "nbuf=%d weight_bytes=%zu (%.4f bpw resident) slab_stride=%zu "
            "panel=%zu B/thread xq=%zu B C=%zu B split=%d\n",
            cfg->mk, name, R, K, ny, cfg->nth, nbuf, wv[0].weight_bytes,
            bpw_w, stride, pans[0].bytes, xq_bytes, (size_t)ny * R * 4,
            cfg->split);

    Doml3Job job;
    memset(&job, 0, sizeof(job));
    job.wv = wv;
    job.nbuf = nbuf;
    job.X = X;
    job.qx = &qx;
    job.C = C;
    job.ldc = R;
    job.panels = pans;
    job.pool = pool;
    job.mk = cfg->mk;
    job.split = cfg->split;

    int rot = 0;
    job.iters = 1;
    job.rot = rot;
    doml2_pool_run(pool, doml3_job_exec, &job);
    rot += 1;

    /* inline correctness: fp64 reference GEMM on 3 activation rows */
    {
        uint16_t tab[256];
        dpka_fp8e4m3_to_bf16_table(tab);
        float *wrow = (float *)xaligned((size_t)K * 4);
        const uint32_t ys[3] = { 0, ny / 2, ny - 1 };
        double ss_ref = 0, ss_err = 0, max_abs = 0;
        for (uint32_t r = 0; r < R; r++) {
            dpka_ref_decode_row_rb(rb, tab, r, wrow);
            for (int yi = 0; yi < 3; yi++) {
                const float *xr = X + (size_t)ys[yi] * K;
                double acc = 0;
                for (uint32_t j = 0; j < K; j++)
                    acc += (double)wrow[j] * (double)xr[j];
                double e = fabs((double)C[(size_t)ys[yi] * R + r] - acc);
                ss_ref += acc * acc;
                ss_err += e * e;
                if (e > max_abs) max_abs = e;
            }
        }
        double rms_ref = sqrt(ss_ref / (double)(R * 3));
        double max_nrm = max_abs / (rms_ref > 0 ? rms_ref : 1.0);
        double rms_rel = sqrt(ss_err / (ss_ref > 0 ? ss_ref : 1.0));
        fprintf(stderr,
                "CHECK doml3_gemm mk=%d rms_rel=%.3e max_nrm=%.3e "
                "(plumbing bar 2e-2; the binding gate is gemm_test) %s\n",
                cfg->mk, rms_rel, max_nrm, rms_rel <= 2e-2 ? "PASS" : "FAIL");
        free(wrow);
        if (rms_rel > 2e-2) {
            fprintf(stderr, "FATAL: inline correctness check failed\n");
            exit(1);
        }
    }
    dpka_free_rb(rb);
    rb = NULL;
    dpka_close(f);
    dump_anon_numa("doml3");

    /* warmup + calibrate so one rep lasts >= target_ms */
    job.iters = 2;
    job.rot = rot;
    doml2_pool_run(pool, doml3_job_exec, &job);
    rot += 2;
    int iters = 1;
    for (int tries = 0; tries < 4; tries++) {
        job.iters = iters;
        job.rot = rot;
        job.tq = job.tc = job.tg = 0;
        doml2_pool_run(pool, doml3_job_exec, &job);
        rot += iters;
        double t = job.t1 - job.t0;
        if (t >= cfg->target_ms * 1e-3 || iters >= (1 << 20)) break;
        double scale = (cfg->target_ms * 1.2e-3) / (t > 1e-7 ? t : 1e-7);
        double ni = (double)iters * (scale > 2.0 ? scale : 2.0);
        iters = (int)(ni < (double)(1 << 20) ? ni : (double)(1 << 20));
    }

    double nsc[64] = { 0 }, gmc[64] = { 0 }, gbw[64] = { 0 };
    double pq[64] = { 0 }, pc[64] = { 0 }, pg[64] = { 0 };
    int reps = cfg->reps > 64 ? 64 : (cfg->reps < 1 ? 1 : cfg->reps);
    for (int r = 0; r < reps; r++) {
        job.iters = iters;
        job.rot = rot;
        job.tq = job.tc = job.tg = 0;
        doml2_pool_run(pool, doml3_job_exec, &job);
        rot += iters;
        double secs = job.t1 - job.t0;
        double per_call = secs / iters;
        nsc[r] = per_call * 1e9;
        gbw[r] = (double)wv[0].weight_bytes / per_call / 1e9;
        gmc[r] = (double)R * K * ny / per_call / 1e9;
        pq[r] = job.tq / iters * 1e9;
        pc[r] = job.tc / iters * 1e9;
        pg[r] = job.tg / iters * 1e9;
        if (csv)
            fprintf(csv,
                    "doml3_gemm,i8p_mk%d,%u,%u,%u,%d,%d,%d,%d,%.6f,%.1f,%.3f,%.3f\n",
                    cfg->mk, R, K, ny, cfg->nth, nbuf, r, iters, secs, nsc[r],
                    gbw[r], gmc[r]);
    }
    double mn = nsc[0], mx = nsc[0];
    for (int r = 1; r < reps; r++) {
        if (nsc[r] < mn) mn = nsc[r];
        if (nsc[r] > mx) mx = nsc[r];
    }
    fprintf(stderr,
            "SUMMARY doml3_gemm type=i8p_mk%d dout=%-4u din=%-4u ny=%-3u nth=%-2d "
            "nbuf=%-3d iters=%d median=%.1f ns/call [%.1f,%.1f]  "
            "weightBW=%.2f GB/s  %.2f GMAC/s  %.1f calls/s\n",
            cfg->mk, R, K, ny, cfg->nth, nbuf, iters, median_of(nsc, reps),
            mn, mx, median_of(gbw, reps), median_of(gmc, reps),
            1e9 / median_of(nsc, reps));
    if (cfg->split)
        fprintf(stderr,
                "SPLIT doml3_gemm dout=%u din=%u ny=%u nth=%d medians: "
                "quant=%.1f ns/call convert=%.1f ns/call gemm=%.1f ns/call "
                "(split-mode total pays one extra barrier)\n",
                R, K, ny, cfg->nth, median_of(pq, reps), median_of(pc, reps),
                median_of(pg, reps));

    doml2_pool_destroy(pool);
    munmap(slab, stride * (size_t)nbuf);
    for (int t = 0; t < cfg->nth; t++) doml3_panel_free(&pans[t]);
    free(pans);
    free(wv);
    free(toff);
    doml3_x_free(&qx);
    free(X);
    free(C);
    return 0;
}

/* --------------------------------------------------------------- main ----- */

static void usage(const char *a0)
{
    fprintf(stderr,
            "usage:\n"
            "  %s --dout N --din K [--ny 512] --threads T [--mk 0|1] [--split]\n"
            "        [--layer L] [--reps R] [--nbuf N] [--target-ms M] [--seed S]\n"
            "        [--artifact P]\n"
            "  %s --sweep [--reps R] [--target-ms M] [--mk 0|1] [--artifact P]\n",
            a0, a0);
}

int main(int argc, char **argv)
{
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.ny = 512;
    cfg.mk = 1; /* flagship 4x4 (SMOKE bake-off: beats 2x8 at 24t and 48t) */
    cfg.layer = 0;
    cfg.nth = 0;
    cfg.nbuf = 0;
    cfg.reps = 9;
    cfg.target_ms = 60.0;
    cfg.seed = 20260716ULL;
    cfg.artifact = DEF_ARTIFACT;
    int sweep = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (usage(argv[0]), exit(1), (char *)0))
        if (!strcmp(a, "--sweep")) sweep = 1;
        else if (!strcmp(a, "--split")) cfg.split = 1;
        else if (!strcmp(a, "--mk")) cfg.mk = atoi(NEXT());
        else if (!strcmp(a, "--dout")) cfg.dout = atol(NEXT());
        else if (!strcmp(a, "--din")) cfg.din = atol(NEXT());
        else if (!strcmp(a, "--ny")) cfg.ny = atol(NEXT());
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
    if (cfg.ny % 8 || cfg.ny < 8) {
        fprintf(stderr, "ny must be a positive multiple of 8\n");
        return 1;
    }

    doml2_init();
    doml3_init();

    printf("kind,type,dout,din,ny,nth,nbuf,rep,iters,secs,ns_call,weight_GBps,GMACs\n");
    fflush(stdout);

    if (sweep) {
        fprintf(stderr,
                "=== SWEEP doml3_gemm: artifact shapes x ny=%ld x threads {24,48} mk=%d (layer %d) ===\n",
                cfg.ny, cfg.mk, cfg.layer);
        const int nts[2] = { 24, 48 };
        DpkaFile *af = dpka_open(cfg.artifact);
        for (int s = 0; s < N_SHAPES; s++) {
            /* [H17-C] skip rows absent from this artifact (and shadowed
             * duplicates: resolve must land on row s itself) */
            char nbuf[128];
            if (shape_resolve(af, k_shapes[s].dout, k_shapes[s].din,
                              cfg.layer, nbuf) != s)
                continue;
            for (int t = 0; t < 2; t++) {
                Config c = cfg;
                c.dout = k_shapes[s].dout;
                c.din = k_shapes[s].din;
                c.nth = nts[t];
                c.nbuf = 0;
                if (bench_one(&c, stdout)) return 1;
                fflush(stdout);
            }
        }
        dpka_close(af);
        return 0;
    }

    if (cfg.dout <= 0 || cfg.din <= 0 || cfg.nth <= 0) {
        usage(argv[0]);
        return 1;
    }
    return bench_one(&cfg, stdout);
}
