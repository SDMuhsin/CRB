/* P2b gate driver: G-NUM-FP, G-NUM-I8 (with level/activation breakdown),
 * G-UNIQUE, plus a 24t-vs-1t bitwise consistency check.
 * All gates run on REAL tensors from the DPKA artifact (no synthetic data).
 *
 * usage: gemv_test [artifact.dpka]
 * exit code 0 iff all gates pass.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../fmt/dpka.h"
#include "../ref/ref_decode.h"
#include "doml_gemv.h"

#define DEF_ARTIFACT "downloads/cpu_kernel_rnd/qwen3-0.6b-k31.dpka"
#define NG_MAX_TEST 12 /* NG <= 12 for every tensor in this artifact */

/* deterministic pseudo-random fp32 in [-1,1) (bench_ik's generator) */
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

static const struct {
    const char *fmt;
    const char *label;
} k_shapes[5] = {
    { "model.layers.%d.self_attn.q_proj", "q_proj  2048x1024" },
    { "model.layers.%d.self_attn.k_proj", "k_proj  1024x1024" },
    { "model.layers.%d.self_attn.o_proj", "o_proj  1024x2048" },
    { "model.layers.%d.mlp.gate_proj",    "gate    3072x1024" },
    { "model.layers.%d.mlp.down_proj",    "down    1024x3072" },
};
static const int k_layers[3] = { 0, 14, 27 };

typedef struct {
    double max_nrm; /* max |y-ref| / rms(ref) */
    double rms_rel; /* ||y-ref|| / ||ref||    */
} Err;

static Err err_vs_ref(const double *ref, const double *y, long n)
{
    double ss_ref = 0, ss_err = 0, max_abs = 0;
    for (long i = 0; i < n; i++) {
        double e = fabs(y[i] - ref[i]);
        ss_err += e * e;
        ss_ref += ref[i] * ref[i];
        if (e > max_abs) max_abs = e;
    }
    double rms_ref = sqrt(ss_ref / (double)n);
    Err r;
    r.max_nrm = max_abs / (rms_ref > 0 ? rms_ref : 1.0);
    r.rms_rel = sqrt(ss_err / (ss_ref > 0 ? ss_ref : 1.0));
    return r;
}

/* scalar R-B walk emitting the kernel's table slot index per column
 * (idx = part*4 + code = b0 | b1<<1 | m<<2 | s<<3) — used to rebuild the
 * level-quantized weights for the G-NUM-I8 breakdown. */
static void rb_row_idx(const DpkaResB *b, uint32_t r, uint8_t *idx_out)
{
    const size_t pitch = b->C / 8;
    const uint8_t *b0r = b->b0 + (size_t)r * pitch;
    const uint8_t *b1seg = b->b1 + b->b1_rowoff[r];
    const uint8_t *mrow = b->m + (size_t)r * b->m_pitch;
    size_t b1pos = 0, mpos = 0;
    for (uint32_t j = 0; j < b->C_orig; j++) {
        int part;
        if ((b->s[j >> 3] >> (j & 7)) & 1) {
            part = 2;
        } else {
            part = (mrow[mpos >> 3] >> (mpos & 7)) & 1;
            mpos++;
        }
        int code = (b0r[j >> 3] >> (j & 7)) & 1;
        if (part != 0) {
            code |= ((b1seg[b1pos >> 3] >> (b1pos & 7)) & 1) << 1;
            b1pos++;
        }
        idx_out[j] = (uint8_t)(part * 4 + code);
    }
}

typedef struct {
    const DomlGemvW *w;
    const float *xperm;
    const DomlQx *qx;
    float *y_fp, *y_i8;
} PoolJob;

static void pool_job(void *arg, int ith, int nth)
{
    PoolJob *j = (PoolJob *)arg;
    uint32_t r0, r1;
    doml_gemv_slice(j->w->R, ith, nth, &r0, &r1);
    doml_gemv_fp_rows(j->w, j->xperm, j->y_fp, r0, r1);
    doml_gemv_i8_rows(j->w, j->qx, j->y_i8, r0, r1);
}

static void *xaligned(size_t n)
{
    void *p = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : DEF_ARTIFACT;
    DpkaFile *f = dpka_open(path);
    doml_gemv_init();
    uint16_t tab[256];
    dpka_fp8e4m3_to_bf16_table(tab);

    int nfail = 0;
    double worst_fp = 0;
    DomlPool *pool = doml_pool_create(24);

    for (int li = 0; li < 3; li++) {
        for (int si = 0; si < 5; si++) {
            char name[128];
            snprintf(name, sizeof(name), k_shapes[si].fmt, k_layers[li]);
            int idx = dpka_find(f, name);
            if (idx < 0) { fprintf(stderr, "missing tensor %s\n", name); return 1; }
            DpkaResB *rb = dpka_build_rb(f, idx);
            const uint32_t R = rb->R, C = rb->C_orig;

            uint8_t *slab = (uint8_t *)xaligned(doml_gemv_slab_bytes(rb, 0));
            DomlGemvW w;
            doml_gemv_pack_init(rb, slab, &w, 0);
            doml_gemv_pack_rows(rb, &w, 0, R);

            float *x = (float *)xaligned((size_t)C * 4);
            float *xperm = (float *)xaligned((size_t)C * 4);
            for (uint32_t j = 0; j < C; j++)
                x[j] = det_val(20260715ULL + (uint64_t)li * 100 + (uint64_t)si, j);
            doml_gemv_prep_x_fp(x, C, xperm);
            DomlQx qx;
            qx.q = (int8_t *)xaligned(C);
            qx.dx = (float *)xaligned(sizeof(float) * 16);
            qx.c128 = (float *)xaligned(sizeof(float) * 16);
            doml_gemv_prep_x_i8(x, C, &qx);

            float *y_fp = (float *)xaligned((size_t)R * 4);
            float *y_i8 = (float *)xaligned((size_t)R * 4);
            doml_gemv_fp_rows(&w, xperm, y_fp, 0, R);
            doml_gemv_i8_rows(&w, &qx, y_i8, 0, R);

            /* 24t consistency: same per-row arithmetic => bitwise equal */
            float *y_fp24 = (float *)xaligned((size_t)R * 4);
            float *y_i824 = (float *)xaligned((size_t)R * 4);
            PoolJob pj = { &w, xperm, &qx, y_fp24, y_i824 };
            doml_pool_run(pool, pool_job, &pj);
            int t24_ok = memcmp(y_fp, y_fp24, (size_t)R * 4) == 0 &&
                         memcmp(y_i8, y_i824, (size_t)R * 4) == 0;
            if (!t24_ok) nfail++;

            /* m_full variant decodes the identical weights => bitwise equal */
            uint8_t *slab_mf = (uint8_t *)xaligned(doml_gemv_slab_bytes(rb, 1));
            DomlGemvW wmf;
            doml_gemv_pack_init(rb, slab_mf, &wmf, 1);
            doml_gemv_pack_rows(rb, &wmf, 0, R);
            float *y_mf = (float *)xaligned((size_t)R * 4);
            doml_gemv_fp_rows(&wmf, xperm, y_mf, 0, R);
            int mf_ok = memcmp(y_fp, y_mf, (size_t)R * 4) == 0;
            doml_gemv_i8_rows(&wmf, &qx, y_mf, 0, R);
            mf_ok = mf_ok && memcmp(y_i8, y_mf, (size_t)R * 4) == 0;
            if (!mf_ok) nfail++;
            free(slab_mf);
            free(y_mf);

            /* reference: C decode (R-B) -> fp64 dot; plus the two
             * single-liberty variants for the I8 breakdown */
            double *ref = (double *)malloc(sizeof(double) * R);
            double *y_lvl = (double *)malloc(sizeof(double) * R);
            double *y_act = (double *)malloc(sizeof(double) * R);
            double *yfp_d = (double *)malloc(sizeof(double) * R);
            double *yi8_d = (double *)malloc(sizeof(double) * R);
            float *wrow = (float *)xaligned((size_t)C * 4);
            uint8_t *irow = (uint8_t *)malloc(C);
            uint8_t ut[NG_MAX_TEST][16];
            float sw[NG_MAX_TEST];
            for (uint32_t r = 0; r < R; r++) {
                dpka_ref_decode_row_rb(rb, tab, r, wrow);
                double a_ref = 0, a_act = 0;
                for (uint32_t j = 0; j < C; j++) {
                    a_ref += (double)wrow[j] * (double)x[j];
                    a_act += (double)wrow[j] *
                             ((double)qx.dx[j >> 8] * (double)qx.q[j]);
                }
                ref[r] = a_ref;
                y_act[r] = a_act;
                /* level-rounding only: kernel's own quantized levels, exact x */
                doml_gemv_prep_row_i8(&w, r, ut, sw);
                rb_row_idx(rb, r, irow);
                double a_lvl = 0;
                for (uint32_t j = 0; j < C; j++) {
                    double wq = (double)sw[j >> 8] *
                                ((double)ut[j >> 8][irow[j]] - 128.0);
                    a_lvl += wq * (double)x[j];
                }
                y_lvl[r] = a_lvl;
                yfp_d[r] = (double)y_fp[r];
                yi8_d[r] = (double)y_i8[r];
            }

            Err e_fp = err_vs_ref(ref, yfp_d, R);
            Err e_i8 = err_vs_ref(ref, yi8_d, R);
            Err e_lvl = err_vs_ref(ref, y_lvl, R);
            Err e_act = err_vs_ref(ref, y_act, R);
            int fp_pass = e_fp.max_nrm <= 2e-5;
            if (!fp_pass) nfail++;
            if (e_fp.max_nrm > worst_fp) worst_fp = e_fp.max_nrm;

            printf("[G-NUM-FP] L%-2d %s  max_nrm=%.3e rms_rel=%.3e  (gate 2e-5)  %s\n",
                   k_layers[li], k_shapes[si].label, e_fp.max_nrm, e_fp.rms_rel,
                   fp_pass ? "PASS" : "FAIL");
            printf("[G-NUM-I8] L%-2d %s  total rms=%.3e max=%.3e | level-only rms=%.3e max=%.3e | act-only rms=%.3e max=%.3e\n",
                   k_layers[li], k_shapes[si].label,
                   e_i8.rms_rel, e_i8.max_nrm, e_lvl.rms_rel, e_lvl.max_nrm,
                   e_act.rms_rel, e_act.max_nrm);
            printf("[G-24T   ] L%-2d %s  fp+i8 bitwise 24t==1t: %s\n",
                   k_layers[li], k_shapes[si].label, t24_ok ? "PASS" : "FAIL");
            printf("[G-MFULL ] L%-2d %s  fp+i8 bitwise m_full==packed: %s\n",
                   k_layers[li], k_shapes[si].label, mf_ok ? "PASS" : "FAIL");
            fflush(stdout);

            /* G-UNIQUE ablation on layer 0 q_proj (the 3-line test):
             * zeroing the m plane or the s bitmap must change outputs */
            if (li == 0 && si == 0) {
                float *ya = (float *)xaligned((size_t)R * 4);
                memset((void *)(uintptr_t)w.m, 0, rb->bytes_m);
                doml_gemv_fp_rows(&w, xperm, ya, 0, R);
                long d_m = 0;
                for (uint32_t r = 0; r < R; r++) d_m += (ya[r] != y_fp[r]);
                doml_gemv_pack_rows(rb, &w, 0, R); /* restore */
                memset((void *)(uintptr_t)w.s, 0, rb->bytes_s);
                doml_gemv_fp_rows(&w, xperm, ya, 0, R);
                long d_s = 0;
                for (uint32_t r = 0; r < R; r++) d_s += (ya[r] != y_fp[r]);
                doml_gemv_pack_init(rb, slab, &w, 0); /* restore s */
                doml_gemv_pack_rows(rb, &w, 0, R);
                int u_ok = d_m > 0 && d_s > 0;
                if (!u_ok) nfail++;
                printf("[G-UNIQUE] zero-m plane: %ld/%u outputs changed; zero-s bitmap: %ld/%u outputs changed  %s\n",
                       d_m, R, d_s, R, u_ok ? "PASS" : "FAIL");
                free(ya);
            }

            free(ref); free(y_lvl); free(y_act); free(yfp_d); free(yi8_d);
            free(wrow); free(irow);
            free(x); free(xperm); free(qx.q); free(qx.dx); free(qx.c128);
            free(y_fp); free(y_i8); free(y_fp24); free(y_i824);
            free(slab);
            dpka_free_rb(rb);
        }
    }
    doml_pool_destroy(pool);
    dpka_close(f);

    printf("================ P2b gate summary ================\n");
    printf("  G-NUM-FP worst max_nrm = %.3e (gate 2e-5)\n", worst_fp);
    printf("  %s\n", nfail ? "GATES: FAIL" : "GATES: ALL PASS");
    return nfail ? 1 : 0;
}
