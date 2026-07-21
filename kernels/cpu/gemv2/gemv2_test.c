/* P2c gate driver for the v2 fused-slab GEMV.
 *
 * default mode (15 real tensors: layers 0/14/27 x 5 shapes):
 *   G-NUM-FP  fp path vs (C reference R-B decode -> fp64 dot), max_nrm <= 2e-5
 *   G-NUM-I8  total + level-only + activation-only breakdown
 *   G-24T     24t (pool v2, static) bitwise == 1t, both paths
 *   G-48T     48t bitwise == 1t (sync correctness under oversubscription)
 *   G-STEAL   work-stealing (24t, chunk 8, both cursor parities) bitwise == static
 *   G-MF      m-full ablation slab bitwise == packed slab, both paths
 *   G-UNIQUE  zeroing the m sections / the s bitmap changes outputs
 *
 * --derive  (G-DERIVE, all 196 tensors):
 *   fp slab: scalar walk of the v2 slab decodes bitwise-identical to the P1
 *            reference decode (dpka_ref_decode_row_rb) — full decode-equality.
 *   i8 slab: plane round-trip — the idx stream (b0/b1/m/s) recovered from the
 *            v2 slab equals a scalar R-B walk bitwise, and every cb record
 *            equals an independent scalar re-derivation (max/127 bf16-RNE
 *            scale + rounded levels) from the resident fp8 slots.
 *
 * --bpw     (G-BPW-V2, all 196 tensors): per-component byte table; PASS iff
 *           flagship (i8, packed m) aggregate <= 2.70 bpw counting EVERY
 *           allocated byte (headers, alignment, tail slack, offsets).
 *
 * usage: gemv2_test [--derive|--bpw] [artifact.dpka]
 * exit code 0 iff the selected gates pass.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../fmt/dpka.h"
#include "../ref/ref_decode.h"
#include "doml_gemv2.h"

#define DEF_ARTIFACT "downloads/cpu_kernel_rnd/qwen3-0.6b-k31.dpka"

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

static void *xaligned(size_t n)
{
    void *p = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static inline int bit_at(const uint8_t *bytes, size_t j)
{
    return (bytes[j >> 3] >> (j & 7)) & 1;
}

static inline float bf16f_(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* ---------------------------------------------------------- slab helper --- */

typedef struct {
    Doml2W w;
    uint8_t *slab;
    uint32_t *toff;
} Slab;

static Slab build_slab(const DpkaResB *rb, Doml2Var var, int mf)
{
    Slab s;
    uint32_t ntiles = rb->R / DOML2_TILE;
    s.toff = (uint32_t *)xaligned(((size_t)ntiles + 1) * 4);
    size_t sz = doml2_slab_bytes(rb, var, mf, s.toff, NULL);
    s.slab = (uint8_t *)xaligned(sz);
    doml2_pack_init(rb, var, mf, s.slab, s.toff, &s.w);
    doml2_pack_tiles(rb, &s.w, 0, ntiles);
    return s;
}

static void free_slab(Slab *s)
{
    free(s->slab);
    free(s->toff);
}

/* -------------------------------------------- scalar v2-slab row walker --- */
/* Independent scalar decode of one row from the fused blocks: emits the
 * table slot index (= part*4 + code) per column and, for the fp slab, the
 * decoded fp32 value (bf16 pattern widened). Shares NO code with the kernel
 * or the packer beyond the header geometry constants. */
static void v2_row_walk(const Doml2W *w, uint32_t r, const uint16_t tab[256],
                        uint8_t *idx_out, float *val_out)
{
    const unsigned rec = w->variant == DOML2_VAR_I8 ? 14u : 12u;
    const uint32_t t = r / DOML2_TILE, n = r % DOML2_TILE;
    const uint8_t *p = w->blocks + w->tileoff[t];
    for (uint32_t gi = 0; gi < w->NG; gi++) {
        const unsigned l[4] = { p[0], p[1], p[2], p[3] };
        const uint8_t *cbp = p + 4;
        const uint8_t *b0s = cbp + 4u * rec;
        const unsigned mlen = w->m_full ? 32u : w->mlen[gi];
        const uint8_t *ms = b0s + 128;
        const uint8_t *b1s = ms + 4u * mlen;
        const uint8_t *myrec = cbp + (size_t)n * rec;
        const uint8_t *b0r = b0s + (size_t)n * 32;
        const uint8_t *mr = ms + (size_t)n * mlen;
        const uint8_t *b1r = b1s;
        for (uint32_t k = 0; k < n; k++) b1r += l[k];
        unsigned mpos = 0, b1pos = 0;
        for (uint32_t jj = 0; jj < 256; jj++) {
            const uint32_t j = gi * 256 + jj;
            int part;
            if (bit_at(w->s, j)) {
                part = 2;
            } else if (w->m_full) {
                part = bit_at(mr, jj);
            } else {
                part = bit_at(mr, mpos);
                mpos++;
            }
            int code = bit_at(b0r, jj);
            if (part != 0) {
                code |= bit_at(b1r, b1pos) << 1;
                b1pos++;
            }
            const int idx = part * 4 + code;
            idx_out[j] = (uint8_t)idx;
            if (val_out) val_out[j] = bf16f_(tab[myrec[idx]]);
        }
        p = b1s + l[0] + l[1] + l[2] + l[3];
    }
}

/* scalar R-B walk emitting the same slot index (P2b gate machinery) */
static void rb_row_idx(const DpkaResB *b, uint32_t r, uint8_t *idx_out)
{
    const size_t pitch = b->C / 8;
    const uint8_t *b0r = b->b0 + (size_t)r * pitch;
    const uint8_t *b1seg = b->b1 + b->b1_rowoff[r];
    const uint8_t *mrow = b->m + (size_t)r * b->m_pitch;
    size_t b1pos = 0, mpos = 0;
    for (uint32_t j = 0; j < b->C_orig; j++) {
        int part;
        if (bit_at(b->s, j)) {
            part = 2;
        } else {
            part = bit_at(mrow, mpos);
            mpos++;
        }
        int code = bit_at(b0r, j);
        if (part != 0) {
            code |= bit_at(b1seg, b1pos) << 1;
            b1pos++;
        }
        idx_out[j] = (uint8_t)(part * 4 + code);
    }
}

/* independent scalar re-derivation of one i8 cb record (spec: container
 * 12-slot order, q = rint(level/scale) clamped to ±127 stored +128, scale =
 * bf16-RNE(max|level|/127)) — written against the spec, not the packer */
static void ref_quant_cb10(const uint8_t cb10[10], const uint16_t tab[256],
                           uint8_t rec[14])
{
    float lv[12];
    lv[0] = bf16f_(tab[cb10[0]]);
    lv[1] = bf16f_(tab[cb10[1]]);
    lv[2] = lv[1]; /* pad slots replicate bulk1 */
    lv[3] = lv[1];
    for (int k = 0; k < 8; k++) lv[4 + k] = bf16f_(tab[cb10[2 + k]]);
    float maxa = 0.f;
    for (int k = 0; k < 12; k++)
        if (fabsf(lv[k]) > maxa) maxa = fabsf(lv[k]);
    uint16_t sb = 0;
    if (maxa > 0.f) {
        float scf = maxa / 127.0f;
        uint32_t u;
        memcpy(&u, &scf, 4);
        u += 0x7FFFu + ((u >> 16) & 1u); /* RNE to bf16 */
        sb = (uint16_t)(u >> 16);
    }
    float s = bf16f_(sb);
    for (int k = 0; k < 12; k++) {
        long q = 0;
        if (s > 0.f) {
            q = lrintf(lv[k] / s);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
        }
        rec[k] = (uint8_t)(q + 128);
    }
    rec[12] = (uint8_t)(sb & 0xFF);
    rec[13] = (uint8_t)(sb >> 8);
}

/* --------------------------------------------------------------- shapes --- */

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
    double max_nrm, rms_rel;
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

/* ------------------------------------------------------------ pool jobs --- */

typedef struct {
    const Doml2W *wfp, *wi8;
    const float *xperm;
    const DomlQx *qx;
    float *y_fp, *y_i8;
} PoolJob;

static void pool_job(void *arg, int ith, int nth)
{
    PoolJob *j = (PoolJob *)arg;
    uint32_t t0, t1;
    doml2_slice(j->wfp->ntiles, ith, nth, &t0, &t1);
    doml2_gemv_fp_tiles(j->wfp, j->xperm, j->y_fp, t0, t1);
    doml2_gemv_i8_tiles(j->wi8, j->qx, j->y_i8, t0, t1);
}

typedef struct {
    Doml2Steal *ws;
    int parity, chunk;
    const Doml2W *w;
    const float *xperm;
    const DomlQx *qx;
    float *y;
} StealJob;

static void steal_job(void *arg, int ith, int nth)
{
    StealJob *j = (StealJob *)arg;
    doml2_steal_gemv(j->ws, j->parity, j->chunk, j->w, j->xperm, j->qx, j->y,
                     ith, nth);
}

/* --------------------------------------------------------------- G-BPW ---- */

static int run_bpw(DpkaFile *f)
{
    Doml2Stats tot_i8, tot_fp;
    memset(&tot_i8, 0, sizeof(tot_i8));
    memset(&tot_fp, 0, sizeof(tot_fp));
    size_t slab_i8 = 0, slab_fp = 0;
    uint64_t tot_w = 0;
    for (uint32_t i = 0; i < f->n_tensors; i++) {
        DpkaResB *rb = dpka_build_rb(f, (int)i);
        uint32_t ntiles = rb->R / DOML2_TILE;
        uint32_t *toff = (uint32_t *)xaligned(((size_t)ntiles + 1) * 4);
        Doml2Stats st;
        slab_i8 += doml2_slab_bytes(rb, DOML2_VAR_I8, 0, toff, &st);
#define ADD(F) tot_i8.F += st.F
        ADD(b0); ADD(cb); ADD(hdr); ADD(m); ADD(b1); ADD(s); ADD(tileoff);
        ADD(align_pad); ADD(tail_pad); ADD(b1_pad_bits); ADD(m_pad_bits);
#undef ADD
        slab_fp += doml2_slab_bytes(rb, DOML2_VAR_FP, 0, toff, &st);
#define ADD(F) tot_fp.F += st.F
        ADD(b0); ADD(cb); ADD(hdr); ADD(m); ADD(b1); ADD(s); ADD(tileoff);
        ADD(align_pad); ADD(tail_pad); ADD(b1_pad_bits); ADD(m_pad_bits);
#undef ADD
        tot_w += (uint64_t)rb->R * rb->C_orig;
        free(toff);
        dpka_free_rb(rb);
        if ((i + 1) % 28 == 0)
            fprintf(stderr, "  ..%u/%u tensors\n", i + 1, f->n_tensors);
    }
    double dw = (double)tot_w;
    printf("G-BPW-V2 component table (all %u tensors, %llu weights)\n",
           f->n_tensors, (unsigned long long)tot_w);
    printf("%-28s %14s %10s  |  %14s %10s\n", "component", "i8-slab B", "bpw",
           "fp-slab B", "bpw");
#define ROW(NAME, F)                                                          \
    printf("%-28s %14zu %10.4f  |  %14zu %10.4f\n", NAME, tot_i8.F,           \
           8.0 * (double)tot_i8.F / dw, tot_fp.F, 8.0 * (double)tot_fp.F / dw)
    ROW("b0 plane", b0);
    ROW("cb records", cb);
    ROW("block headers", hdr);
    ROW("m sections (incl align)", m);
    ROW("b1 sections (incl align)", b1);
    ROW("s bitmap", s);
    ROW("tile offsets (shared)", tileoff);
    ROW("slab align pad", align_pad);
    ROW("slab tail slack", tail_pad);
#undef ROW
    printf("%-28s %14zu %10.4f  |  %14zu %10.4f   (subset of sections above)\n",
           "  of which b1 align bits", tot_i8.b1_pad_bits / 8,
           (double)tot_i8.b1_pad_bits / dw, tot_fp.b1_pad_bits / 8,
           (double)tot_fp.b1_pad_bits / dw);
    printf("%-28s %14zu %10.4f  |  %14zu %10.4f   (subset of sections above)\n",
           "  of which m align bits", tot_i8.m_pad_bits / 8,
           (double)tot_i8.m_pad_bits / dw, tot_fp.m_pad_bits / 8,
           (double)tot_fp.m_pad_bits / dw);
    double bpw_i8 = 8.0 * (double)slab_i8 / dw;
    double bpw_fp = 8.0 * (double)slab_fp / dw;
    /* per-tensor mlen tables (NG bytes) are shared host-side metadata */
    printf("TOTAL allocated (every byte)  i8-slab %.4f bpw | fp-slab %.4f bpw\n",
           bpw_i8, bpw_fp);
    printf("deltas: artifact 2.2482 -> i8 %+0.4f; honest 2.2299 -> i8 %+0.4f\n",
           bpw_i8 - 2.2482, bpw_i8 - 2.2299);
    int pass = bpw_i8 <= 2.70;
    printf("G-BPW-V2 flagship (i8, packed m) %.4f <= 2.70: %s\n", bpw_i8,
           pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

/* ------------------------------------------------------------- G-DERIVE --- */

static int run_derive(DpkaFile *f)
{
    uint16_t tab[256];
    dpka_fp8e4m3_to_bf16_table(tab);
    long bad_fp = 0, bad_idx = 0, bad_cb = 0;
    for (uint32_t i = 0; i < f->n_tensors; i++) {
        DpkaResB *rb = dpka_build_rb(f, (int)i);
        Slab sfp = build_slab(rb, DOML2_VAR_FP, 0);
        Slab si8 = build_slab(rb, DOML2_VAR_I8, 0);
        const uint32_t R = rb->R, C = rb->C_orig;
        float *ref = (float *)xaligned((size_t)C * 4);
        float *val = (float *)xaligned((size_t)C * 4);
        uint8_t *idx_v2 = (uint8_t *)malloc(C);
        uint8_t *idx_v2b = (uint8_t *)malloc(C);
        uint8_t *idx_rb = (uint8_t *)malloc(C);
        uint8_t rec[14], rrec[14];
        for (uint32_t r = 0; r < R; r++) {
            dpka_ref_decode_row_rb(rb, tab, r, ref);
            v2_row_walk(&sfp.w, r, tab, idx_v2, val);
            if (memcmp(ref, val, (size_t)C * 4) != 0) bad_fp++;
            v2_row_walk(&si8.w, r, tab, idx_v2b, NULL);
            rb_row_idx(rb, r, idx_rb);
            if (memcmp(idx_rb, idx_v2, C) != 0 ||
                memcmp(idx_rb, idx_v2b, C) != 0)
                bad_idx++;
            /* every i8 cb record vs independent scalar re-derivation */
            for (uint32_t gi = 0; gi < rb->NG; gi++) {
                const uint32_t t = r / DOML2_TILE, n = r % DOML2_TILE;
                const uint8_t *p = si8.w.blocks + si8.w.tileoff[t];
                for (uint32_t g2 = 0; g2 < gi; g2++) {
                    unsigned sz = 4u + 4u * 14u + 128u +
                                  4u * (unsigned)si8.w.mlen[g2];
                    sz += p[0] + p[1] + p[2] + p[3];
                    p += sz;
                }
                memcpy(rec, p + 4 + (size_t)n * 14, 14);
                ref_quant_cb10(rb->cb + ((size_t)r * rb->NG + gi) * 10, tab,
                               rrec);
                if (memcmp(rec, rrec, 14) != 0) bad_cb++;
            }
        }
        free(ref); free(val); free(idx_v2); free(idx_v2b); free(idx_rb);
        free_slab(&sfp);
        free_slab(&si8);
        dpka_free_rb(rb);
        if ((i + 1) % 28 == 0)
            fprintf(stderr, "  ..%u/%u tensors (bad rows fp=%ld idx=%ld cb=%ld)\n",
                    i + 1, f->n_tensors, bad_fp, bad_idx, bad_cb);
    }
    int pass = bad_fp == 0 && bad_idx == 0 && bad_cb == 0;
    printf("G-DERIVE  fp-slab decode-equality: %ld bad rows; "
           "i8-slab plane round-trip: %ld bad rows; cb records: %ld bad  -> %s "
           "(%u tensors)\n",
           bad_fp, bad_idx, bad_cb, pass ? "PASS" : "FAIL", f->n_tensors);
    return pass ? 0 : 1;
}

/* ------------------------------------------------------------- numeric ---- */

int main(int argc, char **argv)
{
    const char *path = DEF_ARTIFACT;
    int mode_derive = 0, mode_bpw = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--derive")) mode_derive = 1;
        else if (!strcmp(argv[i], "--bpw")) mode_bpw = 1;
        else path = argv[i];
    }
    DpkaFile *f = dpka_open(path);
    doml2_init();
    uint16_t tab[256];
    dpka_fp8e4m3_to_bf16_table(tab);

    if (mode_bpw) { int rc = run_bpw(f); dpka_close(f); return rc; }
    if (mode_derive) { int rc = run_derive(f); dpka_close(f); return rc; }

    int nfail = 0;
    double worst_fp = 0;
    Doml2Pool *pool24 = doml2_pool_create(24);
    Doml2Pool *pool48 = NULL; /* created after 24t gates to limit thread mix */

    for (int li = 0; li < 3; li++) {
        for (int si = 0; si < 5; si++) {
            char name[128];
            snprintf(name, sizeof(name), k_shapes[si].fmt, k_layers[li]);
            int idx = dpka_find(f, name);
            if (idx < 0) { fprintf(stderr, "missing tensor %s\n", name); return 1; }
            DpkaResB *rb = dpka_build_rb(f, idx);
            const uint32_t R = rb->R, C = rb->C_orig;

            Slab sfp = build_slab(rb, DOML2_VAR_FP, 0);
            Slab si8 = build_slab(rb, DOML2_VAR_I8, 0);

            float *x = (float *)xaligned((size_t)C * 4);
            float *xperm = (float *)xaligned((size_t)C * 4);
            for (uint32_t j = 0; j < C; j++)
                x[j] = det_val(20260715ULL + (uint64_t)li * 100 + (uint64_t)si, j);
            doml_gemv_prep_x_fp(x, C, xperm);
            DomlQx qx;
            qx.q = (int8_t *)xaligned(C);
            /* [H17-C] NG floats needed (24 at 1.7B down_proj); +4 slack as
             * in the fork's stack arrays. Was a fixed 16. */
            qx.dx = (float *)xaligned(sizeof(float) * (DOML2_NG_MAX + 4));
            qx.c128 = (float *)xaligned(sizeof(float) * (DOML2_NG_MAX + 4));
            doml_gemv_prep_x_i8(x, C, &qx);

            float *y_fp = (float *)xaligned((size_t)R * 4);
            float *y_i8 = (float *)xaligned((size_t)R * 4);
            doml2_gemv_fp_tiles(&sfp.w, xperm, y_fp, 0, sfp.w.ntiles);
            doml2_gemv_i8_tiles(&si8.w, &qx, y_i8, 0, si8.w.ntiles);

            /* 24t bitwise consistency (static slices) */
            float *ya = (float *)xaligned((size_t)R * 4);
            float *yb = (float *)xaligned((size_t)R * 4);
            PoolJob pj = { &sfp.w, &si8.w, xperm, &qx, ya, yb };
            doml2_pool_run(pool24, pool_job, &pj);
            int t24_ok = memcmp(y_fp, ya, (size_t)R * 4) == 0 &&
                         memcmp(y_i8, yb, (size_t)R * 4) == 0;
            if (!t24_ok) nfail++;

            /* steal bitwise == static (both parities) */
            Doml2Steal *ws = (Doml2Steal *)xaligned(sizeof(Doml2Steal));
            doml2_steal_init(ws, si8.w.ntiles, 24);
            memset(ya, 0, (size_t)R * 4);
            StealJob sj = { ws, 0, 8, &si8.w, NULL, &qx, ya };
            doml2_pool_run(pool24, steal_job, &sj);
            sj.parity = 1;
            memset(yb, 0, (size_t)R * 4);
            sj.y = yb;
            doml2_pool_run(pool24, steal_job, &sj);
            int steal_ok = memcmp(y_i8, ya, (size_t)R * 4) == 0 &&
                           memcmp(y_i8, yb, (size_t)R * 4) == 0;
            if (!steal_ok) nfail++;
            free(ws);

            /* m-full ablation slab: identical outputs, both paths */
            Slab sfm = build_slab(rb, DOML2_VAR_FP, 1);
            Slab sim = build_slab(rb, DOML2_VAR_I8, 1);
            float *y_mf = (float *)xaligned((size_t)R * 4);
            doml2_gemv_fp_tiles(&sfm.w, xperm, y_mf, 0, sfm.w.ntiles);
            int mf_ok = memcmp(y_fp, y_mf, (size_t)R * 4) == 0;
            doml2_gemv_i8_tiles(&sim.w, &qx, y_mf, 0, sim.w.ntiles);
            mf_ok = mf_ok && memcmp(y_i8, y_mf, (size_t)R * 4) == 0;
            if (!mf_ok) nfail++;
            free_slab(&sfm);
            free_slab(&sim);
            free(y_mf);

            /* reference: C decode (R-B) -> fp64 dot; + the two
             * single-liberty variants for the I8 breakdown */
            double *ref = (double *)malloc(sizeof(double) * R);
            double *y_lvl = (double *)malloc(sizeof(double) * R);
            double *y_act = (double *)malloc(sizeof(double) * R);
            double *yfp_d = (double *)malloc(sizeof(double) * R);
            double *yi8_d = (double *)malloc(sizeof(double) * R);
            float *wrow = (float *)xaligned((size_t)C * 4);
            uint8_t *irow = (uint8_t *)malloc(C);
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
                /* level-rounding only: the slab's own (q,scale) levels */
                v2_row_walk(&si8.w, r, tab, irow, NULL);
                double a_lvl = 0;
                {
                    const uint32_t t = r / DOML2_TILE, n = r % DOML2_TILE;
                    const uint8_t *p = si8.w.blocks + si8.w.tileoff[t];
                    for (uint32_t gi = 0; gi < si8.w.NG; gi++) {
                        const uint8_t *rc14 = p + 4 + (size_t)n * 14;
                        uint16_t sb;
                        memcpy(&sb, rc14 + 12, 2);
                        double sc = (double)bf16f_(sb);
                        for (uint32_t jj = 0; jj < 256; jj++) {
                            uint32_t j = gi * 256 + jj;
                            double wq =
                                sc * ((double)rc14[irow[j]] - 128.0);
                            a_lvl += wq * (double)x[j];
                        }
                        unsigned sz = 4u + 56u + 128u +
                                      4u * (si8.w.m_full ? 32u
                                                         : si8.w.mlen[gi]);
                        sz += p[0] + p[1] + p[2] + p[3];
                        p += sz;
                    }
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
            printf("[G-STEAL ] L%-2d %s  i8 steal(24t,chunk8,par0+1)==static: %s\n",
                   k_layers[li], k_shapes[si].label, steal_ok ? "PASS" : "FAIL");
            printf("[G-MF    ] L%-2d %s  fp+i8 bitwise m_full==packed: %s\n",
                   k_layers[li], k_shapes[si].label, mf_ok ? "PASS" : "FAIL");
            fflush(stdout);

            /* G-UNIQUE + 48t on layer 0 q_proj only */
            if (li == 0 && si == 0) {
                /* zero every m section of the fp slab */
                uint8_t *blocks = (uint8_t *)(uintptr_t)sfp.w.blocks;
                for (uint32_t t = 0; t < sfp.w.ntiles; t++) {
                    uint8_t *p = blocks + sfp.w.tileoff[t];
                    for (uint32_t gi = 0; gi < sfp.w.NG; gi++) {
                        unsigned mlen = sfp.w.mlen[gi];
                        unsigned sz = 4u + 48u + 128u + 4u * mlen;
                        unsigned b1sz = p[0] + p[1] + p[2] + p[3];
                        memset(p + 4 + 48 + 128, 0, 4u * mlen);
                        p += sz + b1sz;
                    }
                }
                doml2_gemv_fp_tiles(&sfp.w, xperm, ya, 0, sfp.w.ntiles);
                long d_m = 0;
                for (uint32_t r = 0; r < R; r++) d_m += (ya[r] != y_fp[r]);
                doml2_pack_tiles(rb, &sfp.w, 0, sfp.w.ntiles); /* restore */
                memset((void *)(uintptr_t)sfp.w.s, 0, C / 8);
                doml2_gemv_fp_tiles(&sfp.w, xperm, ya, 0, sfp.w.ntiles);
                long d_s = 0;
                for (uint32_t r = 0; r < R; r++) d_s += (ya[r] != y_fp[r]);
                memcpy((void *)(uintptr_t)sfp.w.s, rb->s, C / 8); /* restore */
                int u_ok = d_m > 0 && d_s > 0;
                if (!u_ok) nfail++;
                printf("[G-UNIQUE] zero-m sections: %ld/%u outputs changed; zero-s bitmap: %ld/%u outputs changed  %s\n",
                       d_m, R, d_s, R, u_ok ? "PASS" : "FAIL");
                /* sanity: restored slab still matches */
                doml2_gemv_fp_tiles(&sfp.w, xperm, ya, 0, sfp.w.ntiles);
                if (memcmp(ya, y_fp, (size_t)R * 4) != 0) {
                    printf("[G-UNIQUE] RESTORE FAILED\n");
                    nfail++;
                }
                /* 48t bitwise (futex-fallback sync under HT oversubscription) */
                pool48 = doml2_pool_create(48);
                PoolJob pj48 = { &sfp.w, &si8.w, xperm, &qx, ya, yb };
                doml2_pool_run(pool48, pool_job, &pj48);
                int t48_ok = memcmp(y_fp, ya, (size_t)R * 4) == 0 &&
                             memcmp(y_i8, yb, (size_t)R * 4) == 0;
                if (!t48_ok) nfail++;
                printf("[G-48T   ] fp+i8 bitwise 48t==1t: %s\n",
                       t48_ok ? "PASS" : "FAIL");
                doml2_pool_destroy(pool48);
                pool48 = NULL;
            }

            free(ref); free(y_lvl); free(y_act); free(yfp_d); free(yi8_d);
            free(wrow); free(irow);
            free(x); free(xperm); free(qx.q); free(qx.dx); free(qx.c128);
            free(y_fp); free(y_i8); free(ya); free(yb);
            free_slab(&sfp);
            free_slab(&si8);
            dpka_free_rb(rb);
        }
    }
    doml2_pool_destroy(pool24);
    dpka_close(f);

    printf("================ P2c gate summary ================\n");
    printf("  G-NUM-FP worst max_nrm = %.3e (gate 2e-5)\n", worst_fp);
    printf("  %s\n", nfail ? "GATES: FAIL" : "GATES: ALL PASS");
    return nfail ? 1 : 0;
}
