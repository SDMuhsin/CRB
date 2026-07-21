/* P3 gate driver for the DOML prefill GEMM (slab-convert + VNNI).
 *
 * default mode (15 real tensors: layers 0/14/27 x 5 shapes, ny=512):
 *   G-NUM-P3  GEMM output vs (P1 C reference decode -> fp64 GEMM), per
 *             tensor: total rms_rel <= 1.2e-2 (tolerance fixed in the brief
 *             BEFORE benching) + level-only / act-only breakdown
 *   G-XQ      kernel activation quantizer bitwise == scalar spec (values,
 *             panel-order permutation, dx, b2), all (y, j)
 *   G-MK      4x4 micro-kernel output bitwise == 2x8 flagship
 *   G-UNIQUE  zeroing the slab's m sections / s bitmap changes GEMM outputs;
 *             slab restore re-verified bitwise
 *
 * --mt      adds 24t and 48t full-call (quant+convert+GEMM via the pool)
 *           bitwise == 1t. RUN ONLY ON A FREE BOX (protocol: no multi-thread
 *           work while a paired sweep is live).
 *
 * --derive  (G-DERIVE-P3, all 196 tensors): every panel byte bitwise == the
 *           slab's cb-record level selected by an independent scalar R-B
 *           plane walk, at the position given by the header's layout
 *           formula; panel scales bitwise == the record's bf16 scale.
 *
 * --bpw     (G-BPW-P3): resident re-assert (the i8 slab aggregate over all
 *           196 tensors, must equal P2c's 2.6535 <= 2.70) + transient
 *           working-memory table (panel/xq/C bytes — per call, reused
 *           scratch, never weight-persistent).
 *
 * usage: gemm_test [--mt] [--derive|--bpw] [--ny N] [artifact.dpka]
 * exit code 0 iff the selected gates pass.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../fmt/dpka.h"
#include "../ref/ref_decode.h"
#include "doml_gemm.h"

#define DEF_ARTIFACT "downloads/cpu_kernel_rnd/qwen3-0.6b-k31.dpka"

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

static Slab build_slab(const DpkaResB *rb)
{
    Slab s;
    uint32_t ntiles = rb->R / DOML2_TILE;
    s.toff = (uint32_t *)xaligned(((size_t)ntiles + 1) * 4);
    size_t sz = doml2_slab_bytes(rb, DOML2_VAR_I8, 0, s.toff, NULL);
    s.slab = (uint8_t *)xaligned(sz);
    doml2_pack_init(rb, DOML2_VAR_I8, 0, s.slab, s.toff, &s.w);
    doml2_pack_tiles(rb, &s.w, 0, ntiles);
    return s;
}

static void free_slab(Slab *s)
{
    free(s->slab);
    free(s->toff);
}

/* block pointer of (tile t, group gi) inside the fused slab (independent
 * walk over the self-delimiting headers; test-local) */
static const uint8_t *slab_block(const Doml2W *w, uint32_t t, uint32_t gi)
{
    const uint8_t *p = w->blocks + w->tileoff[t];
    for (uint32_t g2 = 0; g2 < gi; g2++) {
        unsigned sz = 4u + 4u * 14u + 128u + 4u * (unsigned)w->mlen[g2];
        sz += p[0] + p[1] + p[2] + p[3];
        p += sz;
    }
    return p;
}

/* scalar R-B plane walk emitting the 12-slot index per column (independent
 * of the packer/kernel; same machinery as the P2b/P2c gates) */
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

/* layout formulas from doml_gemm.h (written against the spec, not the
 * converter): stored line t within a 16-line chunk holds original dword
 * D = 4*(t%4) + t/4; the map is an involution, so t(D) = 4*(D%4) + D/4. */
static inline size_t panel_pos(uint32_t K, uint32_t r, uint32_t j)
{
    uint32_t rg = r / 16, rl = r % 16;
    uint32_t g = j >> 8, ci = (j >> 6) & 3, D = (j >> 2) & 15, o = j & 3;
    uint32_t t = 4 * (D % 4) + D / 4;
    uint32_t s = g * 64 + ci * 16 + t;
    return ((size_t)rg * (K / 4) + s) * 64 + 4u * rl + o;
}
static inline size_t xq_pos(uint32_t K, uint32_t j)
{
    uint32_t g = j >> 8, ci = (j >> 6) & 3, D = (j >> 2) & 15, o = j & 3;
    uint32_t t = 4 * (D % 4) + D / 4;
    (void)K;
    return (size_t)g * 256 + (size_t)ci * 64 + (size_t)t * 4 + o;
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

/* ------------------------------------------------------------- G-DERIVE --- */

static int run_derive(DpkaFile *f)
{
    long bad_lvl = 0, bad_sc = 0;
    for (uint32_t i = 0; i < f->n_tensors; i++) {
        DpkaResB *rb = dpka_build_rb(f, (int)i);
        Slab sl = build_slab(rb);
        const uint32_t R = rb->R, C = rb->C_orig, NG = rb->NG;
        Doml3Panel pan;
        doml3_panel_alloc(&pan, R, C, NG);
        doml3_convert(&sl.w, 0, R / 16, &pan);
        uint8_t *irow = (uint8_t *)malloc(C);
        for (uint32_t r = 0; r < R; r++) {
            rb_row_idx(rb, r, irow);
            for (uint32_t gi = 0; gi < NG; gi++) {
                const uint8_t *bp = slab_block(&sl.w, r / 4, gi);
                const uint8_t *rec = bp + 4 + (size_t)(r % 4) * 14;
                for (uint32_t jj = 0; jj < 256; jj++) {
                    uint32_t j = gi * 256 + jj;
                    uint8_t want = rec[irow[j]];
                    uint8_t got = pan.pu8[panel_pos(C, r, j)];
                    if (want != got) bad_lvl++;
                }
                uint16_t sb;
                memcpy(&sb, rec + 12, 2);
                float wsc = bf16f_(sb);
                float gsc = pan.sc[(size_t)gi * pan.srows_cap + r];
                if (memcmp(&wsc, &gsc, 4) != 0) bad_sc++;
            }
        }
        free(irow);
        doml3_panel_free(&pan);
        free_slab(&sl);
        dpka_free_rb(rb);
        if ((i + 1) % 28 == 0)
            fprintf(stderr, "  ..%u/%u tensors (bad levels=%ld scales=%ld)\n",
                    i + 1, f->n_tensors, bad_lvl, bad_sc);
    }
    int pass = bad_lvl == 0 && bad_sc == 0;
    printf("G-DERIVE-P3  panel levels vs slab cb records (R-B idx walk): "
           "%ld bad bytes; scales: %ld bad  -> %s (%u tensors)\n",
           bad_lvl, bad_sc, pass ? "PASS" : "FAIL", f->n_tensors);
    return pass ? 0 : 1;
}

/* --------------------------------------------------------------- G-BPW ---- */

static int run_bpw(DpkaFile *f, uint32_t ny)
{
    size_t slab_i8 = 0;
    uint64_t tot_w = 0;
    for (uint32_t i = 0; i < f->n_tensors; i++) {
        DpkaResB *rb = dpka_build_rb(f, (int)i);
        uint32_t ntiles = rb->R / DOML2_TILE;
        uint32_t *toff = (uint32_t *)xaligned(((size_t)ntiles + 1) * 4);
        slab_i8 += doml2_slab_bytes(rb, DOML2_VAR_I8, 0, toff, NULL);
        tot_w += (uint64_t)rb->R * rb->C_orig;
        free(toff);
        dpka_free_rb(rb);
    }
    double bpw = 8.0 * (double)slab_i8 / (double)tot_w;
    printf("G-BPW-P3 resident: i8-slab aggregate %.4f bpw over %u tensors "
           "(%llu weights) — unchanged from P2c, gate <= 2.70: %s\n",
           bpw, f->n_tensors, (unsigned long long)tot_w,
           bpw <= 2.70 ? "PASS" : "FAIL");
    printf("transient working memory (per GEMM call, reused scratch, never "
           "weight-persistent), ny=%u:\n", ny);
    printf("%-22s %10s %14s %16s %12s\n", "shape", "K", "panel B/thread",
           "xq B (shared)", "C B (shared)");
    static const struct { uint32_t R, K; const char *lbl; } sh[5] = {
        { 2048, 1024, "q_proj 2048x1024" },
        { 1024, 1024, "k/v    1024x1024" },
        { 1024, 2048, "o_proj 1024x2048" },
        { 3072, 1024, "gate   3072x1024" },
        { 1024, 3072, "down   1024x3072" },
    };
    for (int i = 0; i < 5; i++) {
        uint32_t strip = doml3_strip_rows(sh[i].K);
        uint32_t NG = sh[i].K >> 8;
        size_t pb = (size_t)strip * sh[i].K + (size_t)NG * strip * 4;
        size_t xb = (size_t)ny * sh[i].K + 2 * (size_t)ny * NG * 4;
        size_t cb = (size_t)ny * sh[i].R * 4;
        printf("%-22s %10u %14zu %16zu %12zu\n", sh[i].lbl, sh[i].K, pb, xb,
               cb);
    }
    return bpw <= 2.70 ? 0 : 1;
}

/* ------------------------------------------------------------- numeric ---- */

typedef struct {
    double rms_rel, max_nrm;
} Err;

int main(int argc, char **argv)
{
    const char *path = DEF_ARTIFACT;
    int mode_derive = 0, mode_bpw = 0, mode_mt = 0;
    uint32_t ny = 512;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--derive")) mode_derive = 1;
        else if (!strcmp(argv[i], "--bpw")) mode_bpw = 1;
        else if (!strcmp(argv[i], "--mt")) mode_mt = 1;
        else if (!strcmp(argv[i], "--ny") && i + 1 < argc)
            ny = (uint32_t)atoi(argv[++i]);
        else path = argv[i];
    }
    if (ny % 8) { fprintf(stderr, "ny must be a multiple of 8\n"); return 1; }
    DpkaFile *f = dpka_open(path);
    doml2_init();
    doml3_init();
    uint16_t tab[256];
    dpka_fp8e4m3_to_bf16_table(tab);

    if (mode_bpw) { int rc = run_bpw(f, ny); dpka_close(f); return rc; }
    if (mode_derive) { int rc = run_derive(f); dpka_close(f); return rc; }

    int nfail = 0;
    double worst_tot = 0;
    Doml2Pool *pool24 = NULL, *pool48 = NULL;

    for (int li = 0; li < 3; li++) {
        for (int si = 0; si < 5; si++) {
            char name[128];
            snprintf(name, sizeof(name), k_shapes[si].fmt, k_layers[li]);
            int idx = dpka_find(f, name);
            if (idx < 0) { fprintf(stderr, "missing tensor %s\n", name); return 1; }
            DpkaResB *rb = dpka_build_rb(f, idx);
            const uint32_t R = rb->R, C = rb->C_orig, NG = rb->NG;
            Slab sl = build_slab(rb);

            /* activations + quant (1t, all rows) */
            float *X = (float *)xaligned((size_t)ny * C * 4);
            for (uint32_t y = 0; y < ny; y++)
                for (uint32_t j = 0; j < C; j++)
                    X[(size_t)y * C + j] = det_val(
                        20260716ULL + (uint64_t)li * 100 + (uint64_t)si,
                        (uint64_t)y * C + j);
            Doml3X qx;
            doml3_x_alloc(&qx, ny, C);
            doml3_quant_x_rows(X, C, &qx, 0, ny);

            /* G-XQ: scalar re-derivation of the quantizer, bitwise */
            long xq_bad = 0;
            for (uint32_t y = 0; y < ny; y++) {
                for (uint32_t g = 0; g < NG; g++) {
                    const float *xg = X + (size_t)y * C + (size_t)g * 256;
                    float amax = 0.f;
                    for (int j = 0; j < 256; j++)
                        if (fabsf(xg[j]) > amax) amax = fabsf(xg[j]);
                    float id = amax > 0.f ? 127.0f / amax : 0.0f;
                    float dx = amax > 0.f ? amax / 127.0f : 0.0f;
                    long bsum = 0;
                    for (int j = 0; j < 256; j++) {
                        long q = lrintf(xg[j] * id);
                        bsum += q;
                        int8_t got = qx.q[(size_t)y * C + (size_t)g * 256 +
                                          xq_pos(C, (uint32_t)j)];
                        if ((int8_t)q != got) xq_bad++;
                    }
                    float b2 = 128.0f * dx * (float)bsum;
                    if (memcmp(&dx, &qx.dx[(size_t)y * NG + g], 4) != 0)
                        xq_bad++;
                    if (memcmp(&b2, &qx.b2[(size_t)y * NG + g], 4) != 0)
                        xq_bad++;
                }
            }
            if (xq_bad) nfail++;

            /* full-tensor panel + both micro-kernels (1t) */
            Doml3Panel pan;
            doml3_panel_alloc(&pan, R, C, NG);
            doml3_convert(&sl.w, 0, R / 16, &pan);
            float *C0 = (float *)xaligned((size_t)ny * R * 4);
            float *C1 = (float *)xaligned((size_t)ny * R * 4);
            doml3_gemm_strip(&pan, R, 0, &qx, C0, R, 0);
            doml3_gemm_strip(&pan, R, 0, &qx, C1, R, 1);
            int mk_ok = memcmp(C0, C1, (size_t)ny * R * 4) == 0;
            if (!mk_ok) nfail++;

            /* fp64 references: exact decode, slab levels, dequant acts */
            float *xdq = (float *)xaligned((size_t)ny * C * 4);
            for (uint32_t y = 0; y < ny; y++)
                for (uint32_t j = 0; j < C; j++)
                    xdq[(size_t)y * C + j] =
                        qx.dx[(size_t)y * NG + (j >> 8)] *
                        (float)qx.q[(size_t)y * C + (j & ~255u) +
                                    xq_pos(C, j & 255u)];
            float *wref = (float *)xaligned((size_t)C * 4);
            float *wlvl = (float *)xaligned((size_t)C * 4);
            uint8_t *irow = (uint8_t *)malloc(C);
            double ss_ref = 0, ss_tot = 0, ss_lvl = 0, ss_act = 0;
            double mx_tot = 0, mx_lvl = 0, mx_act = 0;
            for (uint32_t r = 0; r < R; r++) {
                dpka_ref_decode_row_rb(rb, tab, r, wref);
                rb_row_idx(rb, r, irow);
                for (uint32_t gi = 0; gi < NG; gi++) {
                    const uint8_t *bp = slab_block(&sl.w, r / 4, gi);
                    const uint8_t *rec = bp + 4 + (size_t)(r % 4) * 14;
                    uint16_t sb;
                    memcpy(&sb, rec + 12, 2);
                    float sc = bf16f_(sb);
                    for (uint32_t jj = 0; jj < 256; jj++) {
                        uint32_t j = gi * 256 + jj;
                        wlvl[j] = sc * (float)((int)rec[irow[j]] - 128);
                    }
                }
                for (uint32_t y = 0; y < ny; y++) {
                    const float *xr = X + (size_t)y * C;
                    const float *xd = xdq + (size_t)y * C;
                    double aref = 0, alvl = 0, aact = 0;
                    for (uint32_t j = 0; j < C; j++) {
                        aref += (double)wref[j] * (double)xr[j];
                        alvl += (double)wlvl[j] * (double)xr[j];
                        aact += (double)wref[j] * (double)xd[j];
                    }
                    double c = (double)C0[(size_t)y * R + r];
                    double et = fabs(c - aref), el = fabs(alvl - aref),
                           ea = fabs(aact - aref);
                    ss_ref += aref * aref;
                    ss_tot += et * et;
                    ss_lvl += el * el;
                    ss_act += ea * ea;
                    if (et > mx_tot) mx_tot = et;
                    if (el > mx_lvl) mx_lvl = el;
                    if (ea > mx_act) mx_act = ea;
                }
            }
            double rms_ref = sqrt(ss_ref / ((double)R * ny));
            double nrm = rms_ref > 0 ? rms_ref : 1.0;
            Err e_tot = { sqrt(ss_tot / (ss_ref > 0 ? ss_ref : 1.0)),
                          mx_tot / nrm };
            Err e_lvl = { sqrt(ss_lvl / (ss_ref > 0 ? ss_ref : 1.0)),
                          mx_lvl / nrm };
            Err e_act = { sqrt(ss_act / (ss_ref > 0 ? ss_ref : 1.0)),
                          mx_act / nrm };
            int num_ok = e_tot.rms_rel <= 1.2e-2;
            if (!num_ok) nfail++;
            if (e_tot.rms_rel > worst_tot) worst_tot = e_tot.rms_rel;

            printf("[G-NUM-P3] L%-2d %s ny=%u  total rms=%.3e max=%.3e | "
                   "level rms=%.3e | act rms=%.3e  (gate rms<=1.2e-2)  %s\n",
                   k_layers[li], k_shapes[si].label, ny, e_tot.rms_rel,
                   e_tot.max_nrm, e_lvl.rms_rel, e_act.rms_rel,
                   num_ok ? "PASS" : "FAIL");
            printf("[G-XQ    ] L%-2d %s  quantizer bitwise==scalar spec: "
                   "%ld bad  %s\n",
                   k_layers[li], k_shapes[si].label, xq_bad,
                   xq_bad == 0 ? "PASS" : "FAIL");
            printf("[G-MK    ] L%-2d %s  mk4x4 bitwise == mk2x8: %s\n",
                   k_layers[li], k_shapes[si].label, mk_ok ? "PASS" : "FAIL");
            fflush(stdout);

            /* G-UNIQUE + MT gates on layer-0 q_proj only */
            if (li == 0 && si == 0) {
                float *CA = (float *)xaligned((size_t)ny * R * 4);
                uint8_t *blocks = (uint8_t *)(uintptr_t)sl.w.blocks;
                for (uint32_t t = 0; t < sl.w.ntiles; t++) {
                    uint8_t *p = blocks + sl.w.tileoff[t];
                    for (uint32_t gi = 0; gi < sl.w.NG; gi++) {
                        unsigned mlen = sl.w.mlen[gi];
                        unsigned sz = 4u + 56u + 128u + 4u * mlen;
                        unsigned b1sz = p[0] + p[1] + p[2] + p[3];
                        memset(p + 4 + 56 + 128, 0, 4u * mlen);
                        p += sz + b1sz;
                    }
                }
                doml3_convert(&sl.w, 0, R / 16, &pan);
                doml3_gemm_strip(&pan, R, 0, &qx, CA, R, 0);
                long d_m = 0;
                for (size_t k = 0; k < (size_t)ny * R; k++)
                    d_m += (CA[k] != C0[k]);
                doml2_pack_tiles(rb, &sl.w, 0, sl.w.ntiles); /* restore */
                memset((void *)(uintptr_t)sl.w.s, 0, C / 8);
                doml3_convert(&sl.w, 0, R / 16, &pan);
                doml3_gemm_strip(&pan, R, 0, &qx, CA, R, 0);
                long d_s = 0;
                for (size_t k = 0; k < (size_t)ny * R; k++)
                    d_s += (CA[k] != C0[k]);
                memcpy((void *)(uintptr_t)sl.w.s, rb->s, C / 8); /* restore */
                doml3_convert(&sl.w, 0, R / 16, &pan);
                doml3_gemm_strip(&pan, R, 0, &qx, CA, R, 0);
                int rest_ok = memcmp(CA, C0, (size_t)ny * R * 4) == 0;
                int u_ok = d_m > 0 && d_s > 0 && rest_ok;
                if (!u_ok) nfail++;
                printf("[G-UNIQUE] zero-m sections: %ld/%zu outputs changed; "
                       "zero-s bitmap: %ld/%zu; restore bitwise: %s  %s\n",
                       d_m, (size_t)ny * R, d_s, (size_t)ny * R,
                       rest_ok ? "OK" : "BROKEN", u_ok ? "PASS" : "FAIL");

                if (mode_mt) {
                    /* full threaded call (quant striped + convert + GEMM) */
                    uint32_t strip = doml3_strip_rows(C);
                    for (int tc = 0; tc < 2; tc++) {
                        int nth = tc == 0 ? 24 : 48;
                        Doml2Pool *pool = doml2_pool_create(nth);
                        if (tc == 0) pool24 = pool; else pool48 = pool;
                        Doml3Panel *pans = (Doml3Panel *)xaligned(
                            sizeof(Doml3Panel) * (size_t)nth);
                        for (int t = 0; t < nth; t++)
                            doml3_panel_alloc(&pans[t], strip, C, NG);
                        Doml3X qmt;
                        doml3_x_alloc(&qmt, ny, C);
                        memset(CA, 0, (size_t)ny * R * 4);
                        Doml3Job job;
                        memset(&job, 0, sizeof(job));
                        job.wv = &sl.w;
                        job.nbuf = 1;
                        job.X = X;
                        job.qx = &qmt;
                        job.C = CA;
                        job.ldc = R;
                        job.panels = pans;
                        job.pool = pool;
                        job.iters = 1;
                        job.mk = 0;
                        doml2_pool_run(pool, doml3_job_exec, &job);
                        int ok = memcmp(CA, C0, (size_t)ny * R * 4) == 0 &&
                                 memcmp(qmt.q, qx.q, (size_t)ny * C) == 0;
                        if (!ok) nfail++;
                        printf("[G-%dT-P3] full call bitwise == 1t: %s\n",
                               nth, ok ? "PASS" : "FAIL");
                        for (int t = 0; t < nth; t++)
                            doml3_panel_free(&pans[t]);
                        free(pans);
                        doml3_x_free(&qmt);
                        doml2_pool_destroy(pool);
                        if (tc == 0) pool24 = NULL; else pool48 = NULL;
                    }
                }
                free(CA);
            }

            free(wref); free(wlvl); free(irow); free(xdq);
            free(X); free(C0); free(C1);
            doml3_x_free(&qx);
            doml3_panel_free(&pan);
            free_slab(&sl);
            dpka_free_rb(rb);
        }
    }
    (void)pool24; (void)pool48;
    dpka_close(f);

    printf("================ P3 gate summary ================\n");
    printf("  G-NUM-P3 worst total rms_rel = %.3e (gate 1.2e-2)\n", worst_tot);
    printf("  %s\n", nfail ? "GATES: FAIL" : "GATES: ALL PASS");
    return nfail ? 1 : 0;
}
