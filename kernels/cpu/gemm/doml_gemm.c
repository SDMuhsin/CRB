/* DOML prefill GEMM (P3) — slab-convert + VNNI. See doml_gemm.h.
 *
 * Three kernels:
 *   1. doml3_quant_x_rows: fp32 -> s8 per-256-group activation quantization
 *      (id = 127/amax, RNE, |q| <= 127), emitted in the panel-matched
 *      k-step order — which is the NATURAL order of packs_epi32+packs_epi16
 *      (both are the same 4x4 dword transpose), so no shuffle is needed.
 *   2. doml3_convert: v2 i8 slab -> transient u8 panel. Decode idx stream
 *      exactly as the P2c GEMV (pdep b1/m expansion, merge-mask idx build,
 *      cb record = the vpermb table) but the vpermb result is TRANSPOSED
 *      (4x16 dword unpack) and stored as 16-row interleaved lines instead of
 *      being dotted. Panel bytes are bitwise the slab's stored levels
 *      (u8 = q+128): zero additional rounding (G-DERIVE-P3).
 *   3. doml3_gemm_strip: dense u8s8 vpdpbusd GEMM over the panel.
 *      Micro-kernels: (2 row-groups x 8 cols) flagship and (4 x 4) alt;
 *      per k-step 4 weights: NRB line loads + NY vpbroadcastd (load ports,
 *      NOT port 5) + NRB*NY vpdpbusd -> dpbusd-bound by design. Integer
 *      accumulation per 256-group is exact; per-group fp fixup
 *      C += sc_rg * (cvt(I)*dx_yg - 128*dx*bsum_yg) with fp accumulators in
 *      an L1 stack tile (int accs own the registers).
 *
 * The idx-decode expressions mirror kernels/cpu/gemv2/doml_gemv2.c (same
 * slab, same invariants); they are re-instantiated here because the panel
 * writer needs the level VECTOR, not the dot product. The slab format and
 * the gemv2 sources are untouched.
 */
#include "doml_gemm.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------------------------------------------------------- utils --- */

static int g3_init_done = 0;
static uint32_t g3_strip_kb = 256; /* env DOML3_STRIP_KB */

void doml3_init(void)
{
    if (g3_init_done) return;
    const char *e = getenv("DOML3_STRIP_KB");
    if (e) {
        long v = atol(e);
        if (v >= 8 && v <= 16384) g3_strip_kb = (uint32_t)v;
    }
    g3_init_done = 1;
}

static void die3(const char *msg)
{
    fprintf(stderr, "doml3: FATAL: %s\n", msg);
    abort();
}

static void *xaligned3(size_t n)
{
    void *p = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!p) die3("OOM");
    return p;
}

static inline uint64_t load_u64_3(const void *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* next k bits of an LSB-first bit stream (same contract as gemv2; reads up
 * to 9 bytes — the slab's 512 B tail slack covers the last tile) */
static inline uint64_t take_bits3(const uint8_t *p, unsigned sh, unsigned k)
{
    uint64_t lo = load_u64_3(p) >> sh;
    if (__builtin_expect(sh + k > 64, 0))
        lo |= ((uint64_t)p[8] << (63 ^ sh)) << 1;
    return lo;
}

static inline float bf16f3(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline double now3(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

uint32_t doml3_strip_rows(uint32_t K)
{
    doml3_init();
    uint32_t r = (g3_strip_kb * 1024u) / K;
    r &= ~31u;
    if (r < 32) r = 32;
    return r;
}

/* --------------------------------------------------------- activations ---- */

void doml3_x_alloc(Doml3X *qx, uint32_t ny, uint32_t K)
{
    if (K % 256) die3("K % 256 != 0");
    qx->ny = ny;
    qx->K = K;
    qx->NG = K >> 8;
    qx->q = (int8_t *)xaligned3((size_t)ny * K);
    qx->dx = (float *)xaligned3((size_t)ny * qx->NG * 4);
    qx->b2 = (float *)xaligned3((size_t)ny * qx->NG * 4);
}

void doml3_x_free(Doml3X *qx)
{
    free(qx->q);
    free(qx->dx);
    free(qx->b2);
    memset(qx, 0, sizeof(*qx));
}

void doml3_quant_x_rows(const float *X, uint32_t K, Doml3X *qx,
                        uint32_t y0, uint32_t y1)
{
    const uint32_t NG = qx->NG;
    for (uint32_t y = y0; y < y1; y++) {
        const float *xr = X + (size_t)y * K;
        int8_t *qr = qx->q + (size_t)y * K;
        for (uint32_t g = 0; g < NG; g++) {
            const float *xg = xr + (size_t)g * 256;
            __m512 am = _mm512_setzero_ps();
            for (int c = 0; c < 16; c++)
                am = _mm512_max_ps(am,
                                   _mm512_abs_ps(_mm512_loadu_ps(xg + 16 * c)));
            float amax = _mm512_reduce_max_ps(am);
            float id = amax > 0.f ? 127.0f / amax : 0.0f;
            float dx = amax > 0.f ? amax / 127.0f : 0.0f;
            const __m512 idv = _mm512_set1_ps(id);
            __m512i bs = _mm512_setzero_si512();
            for (int ci = 0; ci < 4; ci++) {
                const float *xc = xg + 64 * ci;
                __m512i v0 = _mm512_cvtps_epi32(
                    _mm512_mul_ps(_mm512_loadu_ps(xc + 0), idv));
                __m512i v1 = _mm512_cvtps_epi32(
                    _mm512_mul_ps(_mm512_loadu_ps(xc + 16), idv));
                __m512i v2 = _mm512_cvtps_epi32(
                    _mm512_mul_ps(_mm512_loadu_ps(xc + 32), idv));
                __m512i v3 = _mm512_cvtps_epi32(
                    _mm512_mul_ps(_mm512_loadu_ps(xc + 48), idv));
                bs = _mm512_add_epi32(
                    bs, _mm512_add_epi32(_mm512_add_epi32(v0, v1),
                                         _mm512_add_epi32(v2, v3)));
                /* packs_epi32+packs_epi16 emit dword order
                 * d -> 4*(d%4) + d/4 == the panel k-step permutation */
                __m512i p01 = _mm512_packs_epi32(v0, v1);
                __m512i p23 = _mm512_packs_epi32(v2, v3);
                _mm512_storeu_si512((void *)(qr + (size_t)g * 256 + 64 * ci),
                                    _mm512_packs_epi16(p01, p23));
            }
            int32_t bsum = _mm512_reduce_add_epi32(bs);
            qx->dx[(size_t)y * NG + g] = dx;
            qx->b2[(size_t)y * NG + g] = 128.0f * dx * (float)bsum;
        }
    }
}

/* --------------------------------------------------------------- panel ---- */

void doml3_panel_alloc(Doml3Panel *pan, uint32_t srows_cap, uint32_t K,
                       uint32_t NG)
{
    if (srows_cap % 32) die3("srows_cap % 32 != 0");
    pan->srows_cap = srows_cap;
    pan->K = K;
    pan->NG = NG;
    size_t b1 = (size_t)srows_cap * K;
    size_t b2 = (size_t)NG * srows_cap * 4;
    pan->pu8 = (uint8_t *)xaligned3(b1);
    pan->sc = (float *)xaligned3(b2);
    pan->bytes = b1 + b2;
}

void doml3_panel_free(Doml3Panel *pan)
{
    free(pan->pu8);
    free(pan->sc);
    memset(pan, 0, sizeof(*pan));
}

/* ------------------------------------------------------------ converter --- */

#define BCAST16_3(p) \
    _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(p)))

/* decode one row's chunk to a u8 level vector (mirrors gemv2 IDX2_ROW) */
#define CV_ROW(n, ci)                                                         \
    uint64_t m64##n = _pdep_u64(take_bits3(msb + mo##n, msh, kns), ns64);     \
    uint64_t nb64##n = s64 + m64##n; /* disjoint: '+' == '|' */               \
    unsigned pc##n = (unsigned)__builtin_popcountll(nb64##n);                 \
    uint64_t b164##n = _pdep_u64(                                             \
        take_bits3((const uint8_t *)(uintptr_t)(bq##n >> 3),                  \
                   (unsigned)bq##n & 7, pc##n),                               \
        nb64##n);                                                             \
    bq##n += pc##n;                                                           \
    __mmask64 kb0##n = _load_mask64(                                          \
        (__mmask64 *)(uintptr_t)(b0s + 32u * (n) + 8u * (ci)));               \
    __m512i idx##n = _mm512_ternarylogic_epi64(                               \
        _mm512_maskz_mov_epi8(kb0##n, vone),                                  \
        _mm512_maskz_mov_epi8((__mmask64)b164##n, vtwo),                      \
        _mm512_mask_mov_epi8(v8s, (__mmask64)m64##n, vfour), 0xFE);           \
    __m512i wv##n = _mm512_permutexvar_epi8(idx##n, tb##n)

/* 4x16 dword transpose of the 4 row vectors + 16 line-piece stores.
 * u_q lane l == k-step dword 4l+q -> stored line t = q*4 + l, i.e. stored
 * line t holds original dword 4*(t%4) + t/4 (the header's wj formula). */
#define CV_STORE_Q(uq, q)                                                     \
    do {                                                                      \
        _mm_store_si128((__m128i *)(dl + ((q) * 4 + 0) * 64),                 \
                        _mm512_castsi512_si128(uq));                          \
        _mm_store_si128((__m128i *)(dl + ((q) * 4 + 1) * 64),                 \
                        _mm512_extracti32x4_epi32(uq, 1));                    \
        _mm_store_si128((__m128i *)(dl + ((q) * 4 + 2) * 64),                 \
                        _mm512_extracti32x4_epi32(uq, 2));                    \
        _mm_store_si128((__m128i *)(dl + ((q) * 4 + 3) * 64),                 \
                        _mm512_extracti32x4_epi32(uq, 3));                    \
    } while (0)

#define CV_CHUNK(ci)                                                          \
    do {                                                                      \
        const uint64_t s64 = load_u64_3(sp + (size_t)gi * 32 + 8u * (ci));    \
        const uint64_t ns64 = ~s64;                                           \
        const unsigned kns = (unsigned)__builtin_popcountll(ns64);            \
        const __m512i v8s = _mm512_maskz_mov_epi8((__mmask64)s64, veight);    \
        const uint8_t *msb = ms + (mpos >> 3);                                \
        const unsigned msh = (unsigned)mpos & 7;                              \
        CV_ROW(0, ci);                                                        \
        CV_ROW(1, ci);                                                        \
        CV_ROW(2, ci);                                                        \
        CV_ROW(3, ci);                                                        \
        __m512i t0 = _mm512_unpacklo_epi32(wv0, wv1);                         \
        __m512i t1 = _mm512_unpackhi_epi32(wv0, wv1);                         \
        __m512i t2 = _mm512_unpacklo_epi32(wv2, wv3);                         \
        __m512i t3 = _mm512_unpackhi_epi32(wv2, wv3);                         \
        __m512i u0 = _mm512_unpacklo_epi64(t0, t2);                           \
        __m512i u1 = _mm512_unpackhi_epi64(t0, t2);                           \
        __m512i u2 = _mm512_unpacklo_epi64(t1, t3);                           \
        __m512i u3 = _mm512_unpackhi_epi64(t1, t3);                           \
        uint8_t *dl = dst + ((size_t)gi * 64u + (ci) * 16u) * 64u;            \
        CV_STORE_Q(u0, 0);                                                    \
        CV_STORE_Q(u1, 1);                                                    \
        CV_STORE_Q(u2, 2);                                                    \
        CV_STORE_Q(u3, 3);                                                    \
        mpos += kns;                                                          \
    } while (0)

/* one slab tile (4 rows) -> panel lines at dst (already offset by the
 * tile's 16-byte lane position) + scales at scp[gi*S + n] */
static void conv_tile(const Doml2W *w, uint32_t t, uint8_t *dst, float *scp,
                      uint32_t S)
{
    const uint32_t NG = w->NG;
    const uint8_t *sp = w->s;
    const __m512i vone = _mm512_set1_epi8(1);
    const __m512i vtwo = _mm512_set1_epi8(2);
    const __m512i vfour = _mm512_set1_epi8(4);
    const __m512i veight = _mm512_set1_epi8(8);
    const uint8_t *p = w->blocks + w->tileoff[t];
    for (uint32_t gi = 0; gi < NG; gi++) {
        const unsigned l0 = p[0], l1 = p[1], l2 = p[2], l3 = p[3];
        const uint8_t *cbp = p + 4;
        const uint8_t *b0s = cbp + 4u * 14u;
        const unsigned mlen = w->mlen[gi];
        const size_t mo0 = 0, mo1 = mlen, mo2 = 2u * mlen, mo3 = 3u * mlen;
        const uint8_t *ms = b0s + 128;
        const uint8_t *b1p0 = ms + 4u * mlen;
        p = b1p0 + l0 + l1 + l2 + l3;
        uint64_t bq0 = (uint64_t)(uintptr_t)b1p0 * 8;
        uint64_t bq1 = bq0 + 8u * l0;
        uint64_t bq2 = bq1 + 8u * l1;
        uint64_t bq3 = bq2 + 8u * l2;
        unsigned mpos = 0;
        _mm_prefetch((const char *)p + 384, _MM_HINT_T0);
        _mm_prefetch((const char *)p + 448, _MM_HINT_T0);
        _mm_prefetch((const char *)p + 512, _MM_HINT_T0);
        _mm_prefetch((const char *)p + 576, _MM_HINT_T0);
        _mm_prefetch((const char *)p + 640, _MM_HINT_T0);
        const __m512i tb0 = BCAST16_3(cbp + 0 * 14);
        const __m512i tb1 = BCAST16_3(cbp + 1 * 14);
        const __m512i tb2 = BCAST16_3(cbp + 2 * 14);
        const __m512i tb3 = BCAST16_3(cbp + 3 * 14);
        uint16_t sb;
        memcpy(&sb, cbp + 0 * 14 + 12, 2);
        scp[(size_t)gi * S + 0] = bf16f3(sb);
        memcpy(&sb, cbp + 1 * 14 + 12, 2);
        scp[(size_t)gi * S + 1] = bf16f3(sb);
        memcpy(&sb, cbp + 2 * 14 + 12, 2);
        scp[(size_t)gi * S + 2] = bf16f3(sb);
        memcpy(&sb, cbp + 3 * 14 + 12, 2);
        scp[(size_t)gi * S + 3] = bf16f3(sb);
        (void)mo0; (void)mo1; (void)mo2; (void)mo3;
        CV_CHUNK(0);
        CV_CHUNK(1);
        CV_CHUNK(2);
        CV_CHUNK(3);
    }
}

void doml3_convert(const Doml2W *w, uint32_t rg0, uint32_t rg1,
                   Doml3Panel *pan)
{
    if (w->variant != DOML2_VAR_I8) die3("gemm requires the i8 slab");
    if (w->m_full) die3("gemm supports the flagship packed-m slab only");
    const uint32_t K4 = pan->K >> 2;
    const uint32_t S = pan->srows_cap;
    if ((rg1 - rg0) * 16 > S) die3("strip exceeds panel capacity");
    for (uint32_t t = rg0 * 4; t < rg1 * 4; t++) {
        uint32_t rg_rel = t / 4 - rg0, a = t % 4;
        uint8_t *dst = pan->pu8 + (size_t)rg_rel * K4 * 64 + (size_t)a * 16;
        float *scp = pan->sc + (size_t)rg_rel * 16 + (size_t)a * 4;
        conv_tile(w, t, dst, scp, S);
    }
}

/* ----------------------------------------------------------------- GEMM --- */

/* flagship micro-kernel: NRB in {1,2} row-groups (16 rows each) x 8 columns.
 * Integer accs (16 zmm) own the registers; fp accs live in an L1 stack tile
 * updated once per 256-group. */
static inline __attribute__((always_inline)) void
mk8_g(const Doml3Panel *pan, uint32_t rg, uint32_t row0, const Doml3X *qx,
      uint32_t y0, float *C, long ldc, const int NRB)
{
    const uint32_t K = pan->K, K4 = K >> 2, NG = pan->NG, S = pan->srows_cap;
    const uint8_t *pw0 = pan->pu8 + (size_t)rg * K4 * 64;
    const uint8_t *pw1 = pw0 + (size_t)K4 * 64;
    const int8_t *xq0 = qx->q + (size_t)(y0 + 0) * K;
    const int8_t *xq1 = qx->q + (size_t)(y0 + 1) * K;
    const int8_t *xq2 = qx->q + (size_t)(y0 + 2) * K;
    const int8_t *xq3 = qx->q + (size_t)(y0 + 3) * K;
    const int8_t *xq4 = qx->q + (size_t)(y0 + 4) * K;
    const int8_t *xq5 = qx->q + (size_t)(y0 + 5) * K;
    const int8_t *xq6 = qx->q + (size_t)(y0 + 6) * K;
    const int8_t *xq7 = qx->q + (size_t)(y0 + 7) * K;
    float fb0[8][16] __attribute__((aligned(64)));
    float fb1[8][16] __attribute__((aligned(64)));
    {
        const __m512 z = _mm512_setzero_ps();
        for (int u = 0; u < 8; u++) {
            _mm512_store_ps(fb0[u], z);
            if (NRB == 2) _mm512_store_ps(fb1[u], z);
        }
    }
    size_t s4 = 0;
    for (uint32_t g = 0; g < NG; g++) {
        __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
        __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
        __m512i a4 = _mm512_setzero_si512(), a5 = _mm512_setzero_si512();
        __m512i a6 = _mm512_setzero_si512(), a7 = _mm512_setzero_si512();
        __m512i b0 = _mm512_setzero_si512(), b1 = _mm512_setzero_si512();
        __m512i b2 = _mm512_setzero_si512(), b3 = _mm512_setzero_si512();
        __m512i b4 = _mm512_setzero_si512(), b5 = _mm512_setzero_si512();
        __m512i b6 = _mm512_setzero_si512(), b7 = _mm512_setzero_si512();
        for (int kk = 0; kk < 16; kk++) {
#define KS8(off)                                                              \
    do {                                                                      \
        const __m512i w0 =                                                    \
            _mm512_load_si512((const void *)(pw0 + (off) * 16));              \
        __m512i w1v = _mm512_setzero_si512();                                 \
        if (NRB == 2)                                                         \
            w1v = _mm512_load_si512((const void *)(pw1 + (off) * 16));        \
        __m512i xb;                                                           \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq0 + s4 + (off)));         \
        a0 = _mm512_dpbusd_epi32(a0, w0, xb);                                 \
        if (NRB == 2) b0 = _mm512_dpbusd_epi32(b0, w1v, xb);                  \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq1 + s4 + (off)));         \
        a1 = _mm512_dpbusd_epi32(a1, w0, xb);                                 \
        if (NRB == 2) b1 = _mm512_dpbusd_epi32(b1, w1v, xb);                  \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq2 + s4 + (off)));         \
        a2 = _mm512_dpbusd_epi32(a2, w0, xb);                                 \
        if (NRB == 2) b2 = _mm512_dpbusd_epi32(b2, w1v, xb);                  \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq3 + s4 + (off)));         \
        a3 = _mm512_dpbusd_epi32(a3, w0, xb);                                 \
        if (NRB == 2) b3 = _mm512_dpbusd_epi32(b3, w1v, xb);                  \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq4 + s4 + (off)));         \
        a4 = _mm512_dpbusd_epi32(a4, w0, xb);                                 \
        if (NRB == 2) b4 = _mm512_dpbusd_epi32(b4, w1v, xb);                  \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq5 + s4 + (off)));         \
        a5 = _mm512_dpbusd_epi32(a5, w0, xb);                                 \
        if (NRB == 2) b5 = _mm512_dpbusd_epi32(b5, w1v, xb);                  \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq6 + s4 + (off)));         \
        a6 = _mm512_dpbusd_epi32(a6, w0, xb);                                 \
        if (NRB == 2) b6 = _mm512_dpbusd_epi32(b6, w1v, xb);                  \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq7 + s4 + (off)));         \
        a7 = _mm512_dpbusd_epi32(a7, w0, xb);                                 \
        if (NRB == 2) b7 = _mm512_dpbusd_epi32(b7, w1v, xb);                  \
    } while (0)
            _mm_prefetch((const char *)(pw0 + 1024), _MM_HINT_T0);
            if (NRB == 2)
                _mm_prefetch((const char *)(pw1 + 1024), _MM_HINT_T0);
            KS8(0);
            KS8(4);
            KS8(8);
            KS8(12);
#undef KS8
            pw0 += 256;
            if (NRB == 2) pw1 += 256;
            s4 += 16;
        }
        const float *scp = pan->sc + (size_t)g * S + (size_t)rg * 16;
        const __m512 vs0 = _mm512_load_ps(scp);
        __m512 vs1 = _mm512_setzero_ps();
        if (NRB == 2) vs1 = _mm512_load_ps(scp + 16);
        const float *dxp = qx->dx + (size_t)(y0) * NG + g;
        const float *b2p = qx->b2 + (size_t)(y0) * NG + g;
#define FIX8(u, ACC, BCC)                                                     \
    do {                                                                      \
        const __m512 dxv = _mm512_set1_ps(dxp[(size_t)(u) * NG]);             \
        const __m512 b2v = _mm512_set1_ps(b2p[(size_t)(u) * NG]);             \
        __m512 t = _mm512_fmsub_ps(_mm512_cvtepi32_ps(ACC), dxv, b2v);        \
        __m512 f = _mm512_load_ps(fb0[u]);                                    \
        _mm512_store_ps(fb0[u], _mm512_fmadd_ps(t, vs0, f));                  \
        if (NRB == 2) {                                                       \
            __m512 t2 = _mm512_fmsub_ps(_mm512_cvtepi32_ps(BCC), dxv, b2v);   \
            __m512 f2 = _mm512_load_ps(fb1[u]);                               \
            _mm512_store_ps(fb1[u], _mm512_fmadd_ps(t2, vs1, f2));            \
        }                                                                     \
    } while (0)
        FIX8(0, a0, b0);
        FIX8(1, a1, b1);
        FIX8(2, a2, b2);
        FIX8(3, a3, b3);
        FIX8(4, a4, b4);
        FIX8(5, a5, b5);
        FIX8(6, a6, b6);
        FIX8(7, a7, b7);
#undef FIX8
    }
    for (int u = 0; u < 8; u++) {
        float *cp = C + (size_t)(y0 + u) * ldc + row0 + (size_t)rg * 16;
        _mm512_storeu_ps(cp, _mm512_load_ps(fb0[u]));
        if (NRB == 2)
            _mm512_storeu_ps(cp + 16, _mm512_load_ps(fb1[u]));
    }
}

/* alternate micro-kernel: NRB in {1..4} row-groups x 4 columns */
static inline __attribute__((always_inline)) void
mk4_g(const Doml3Panel *pan, uint32_t rg, uint32_t row0, const Doml3X *qx,
      uint32_t y0, float *C, long ldc, const int NRB)
{
    const uint32_t K = pan->K, K4 = K >> 2, NG = pan->NG, S = pan->srows_cap;
    const size_t lstr = (size_t)K4 * 64;
    const uint8_t *pw0 = pan->pu8 + (size_t)rg * lstr;
    const uint8_t *pw1 = pw0 + lstr;
    const uint8_t *pw2 = pw0 + 2 * lstr;
    const uint8_t *pw3 = pw0 + 3 * lstr;
    const int8_t *xq0 = qx->q + (size_t)(y0 + 0) * K;
    const int8_t *xq1 = qx->q + (size_t)(y0 + 1) * K;
    const int8_t *xq2 = qx->q + (size_t)(y0 + 2) * K;
    const int8_t *xq3 = qx->q + (size_t)(y0 + 3) * K;
    float fb[4][4][16] __attribute__((aligned(64)));
    {
        const __m512 z = _mm512_setzero_ps();
        for (int i = 0; i < NRB; i++)
            for (int u = 0; u < 4; u++) _mm512_store_ps(fb[i][u], z);
    }
    size_t s4 = 0;
    for (uint32_t g = 0; g < NG; g++) {
        __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
        __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
        __m512i b0 = _mm512_setzero_si512(), b1 = _mm512_setzero_si512();
        __m512i b2 = _mm512_setzero_si512(), b3 = _mm512_setzero_si512();
        __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
        __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
        __m512i d0 = _mm512_setzero_si512(), d1 = _mm512_setzero_si512();
        __m512i d2 = _mm512_setzero_si512(), d3 = _mm512_setzero_si512();
        for (int kk = 0; kk < 16; kk++) {
#define KS4(off)                                                              \
    do {                                                                      \
        const __m512i w0 =                                                    \
            _mm512_load_si512((const void *)(pw0 + (off) * 16));              \
        __m512i w1v = _mm512_setzero_si512();                                 \
        __m512i w2v = _mm512_setzero_si512();                                 \
        __m512i w3v = _mm512_setzero_si512();                                 \
        if (NRB > 1)                                                          \
            w1v = _mm512_load_si512((const void *)(pw1 + (off) * 16));        \
        if (NRB > 2)                                                          \
            w2v = _mm512_load_si512((const void *)(pw2 + (off) * 16));        \
        if (NRB > 3)                                                          \
            w3v = _mm512_load_si512((const void *)(pw3 + (off) * 16));        \
        __m512i xb;                                                           \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq0 + s4 + (off)));         \
        a0 = _mm512_dpbusd_epi32(a0, w0, xb);                                 \
        if (NRB > 1) b0 = _mm512_dpbusd_epi32(b0, w1v, xb);                   \
        if (NRB > 2) c0 = _mm512_dpbusd_epi32(c0, w2v, xb);                   \
        if (NRB > 3) d0 = _mm512_dpbusd_epi32(d0, w3v, xb);                   \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq1 + s4 + (off)));         \
        a1 = _mm512_dpbusd_epi32(a1, w0, xb);                                 \
        if (NRB > 1) b1 = _mm512_dpbusd_epi32(b1, w1v, xb);                   \
        if (NRB > 2) c1 = _mm512_dpbusd_epi32(c1, w2v, xb);                   \
        if (NRB > 3) d1 = _mm512_dpbusd_epi32(d1, w3v, xb);                   \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq2 + s4 + (off)));         \
        a2 = _mm512_dpbusd_epi32(a2, w0, xb);                                 \
        if (NRB > 1) b2 = _mm512_dpbusd_epi32(b2, w1v, xb);                   \
        if (NRB > 2) c2 = _mm512_dpbusd_epi32(c2, w2v, xb);                   \
        if (NRB > 3) d2 = _mm512_dpbusd_epi32(d2, w3v, xb);                   \
        xb = _mm512_set1_epi32(*(const int32_t *)(xq3 + s4 + (off)));         \
        a3 = _mm512_dpbusd_epi32(a3, w0, xb);                                 \
        if (NRB > 1) b3 = _mm512_dpbusd_epi32(b3, w1v, xb);                   \
        if (NRB > 2) c3 = _mm512_dpbusd_epi32(c3, w2v, xb);                   \
        if (NRB > 3) d3 = _mm512_dpbusd_epi32(d3, w3v, xb);                   \
    } while (0)
            _mm_prefetch((const char *)(pw0 + 1024), _MM_HINT_T0);
            if (NRB > 1) _mm_prefetch((const char *)(pw1 + 1024), _MM_HINT_T0);
            if (NRB > 2) _mm_prefetch((const char *)(pw2 + 1024), _MM_HINT_T0);
            if (NRB > 3) _mm_prefetch((const char *)(pw3 + 1024), _MM_HINT_T0);
            KS4(0);
            KS4(4);
            KS4(8);
            KS4(12);
#undef KS4
            pw0 += 256;
            if (NRB > 1) pw1 += 256;
            if (NRB > 2) pw2 += 256;
            if (NRB > 3) pw3 += 256;
            s4 += 16;
        }
        const float *scp = pan->sc + (size_t)g * S + (size_t)rg * 16;
        const float *dxp = qx->dx + (size_t)(y0) * NG + g;
        const float *b2p = qx->b2 + (size_t)(y0) * NG + g;
#define FIX4(u, A0, A1, A2, A3)                                               \
    do {                                                                      \
        const __m512 dxv = _mm512_set1_ps(dxp[(size_t)(u) * NG]);             \
        const __m512 b2v = _mm512_set1_ps(b2p[(size_t)(u) * NG]);             \
        __m512 t;                                                             \
        t = _mm512_fmsub_ps(_mm512_cvtepi32_ps(A0), dxv, b2v);                \
        _mm512_store_ps(fb[0][u],                                             \
                        _mm512_fmadd_ps(t, _mm512_load_ps(scp),               \
                                        _mm512_load_ps(fb[0][u])));           \
        if (NRB > 1) {                                                        \
            t = _mm512_fmsub_ps(_mm512_cvtepi32_ps(A1), dxv, b2v);            \
            _mm512_store_ps(fb[1][u],                                         \
                            _mm512_fmadd_ps(t, _mm512_load_ps(scp + 16),      \
                                            _mm512_load_ps(fb[1][u])));       \
        }                                                                     \
        if (NRB > 2) {                                                        \
            t = _mm512_fmsub_ps(_mm512_cvtepi32_ps(A2), dxv, b2v);            \
            _mm512_store_ps(fb[2][u],                                         \
                            _mm512_fmadd_ps(t, _mm512_load_ps(scp + 32),      \
                                            _mm512_load_ps(fb[2][u])));       \
        }                                                                     \
        if (NRB > 3) {                                                        \
            t = _mm512_fmsub_ps(_mm512_cvtepi32_ps(A3), dxv, b2v);            \
            _mm512_store_ps(fb[3][u],                                         \
                            _mm512_fmadd_ps(t, _mm512_load_ps(scp + 48),      \
                                            _mm512_load_ps(fb[3][u])));       \
        }                                                                     \
    } while (0)
        FIX4(0, a0, b0, c0, d0);
        FIX4(1, a1, b1, c1, d1);
        FIX4(2, a2, b2, c2, d2);
        FIX4(3, a3, b3, c3, d3);
#undef FIX4
    }
    for (int u = 0; u < 4; u++) {
        float *cp = C + (size_t)(y0 + u) * ldc + row0 + (size_t)rg * 16;
        for (int i = 0; i < NRB; i++)
            _mm512_storeu_ps(cp + 16 * i, _mm512_load_ps(fb[i][u]));
    }
}

void doml3_gemm_strip(const Doml3Panel *pan, uint32_t nrows, uint32_t row0,
                      const Doml3X *qx, float *C, long ldc, int mk)
{
    if (nrows % 16) die3("nrows % 16 != 0");
    const uint32_t nrg = nrows / 16;
    const uint32_t ny = qx->ny;
    if (mk == 0) {
        if (ny % 8) die3("mk 2x8 requires ny % 8 == 0");
        for (uint32_t y0 = 0; y0 < ny; y0 += 8) {
            uint32_t rg = 0;
            for (; rg + 2 <= nrg; rg += 2)
                mk8_g(pan, rg, row0, qx, y0, C, ldc, 2);
            if (rg < nrg) mk8_g(pan, rg, row0, qx, y0, C, ldc, 1);
        }
    } else {
        if (ny % 4) die3("mk 4x4 requires ny % 4 == 0");
        for (uint32_t y0 = 0; y0 < ny; y0 += 4) {
            uint32_t rg = 0;
            for (; rg + 4 <= nrg; rg += 4)
                mk4_g(pan, rg, row0, qx, y0, C, ldc, 4);
            switch (nrg - rg) {
            case 3: mk4_g(pan, rg, row0, qx, y0, C, ldc, 3); break;
            case 2: mk4_g(pan, rg, row0, qx, y0, C, ldc, 2); break;
            case 1: mk4_g(pan, rg, row0, qx, y0, C, ldc, 1); break;
            default: break;
            }
        }
    }
}

/* --------------------------------------------------- threaded call glue --- */

void doml3_job_exec(void *arg, int ith, int nth)
{
    Doml3Job *j = (Doml3Job *)arg;
    Doml2Pool *pool = j->pool;
    const uint32_t ny = j->qx->ny;
    const uint32_t K = j->qx->K;
    doml2_pool_barrier(pool, ith);
    if (ith == 0) j->t0 = now3();
    doml2_pool_barrier(pool, ith);
    double mark = 0.0;
    for (int it = 0; it < j->iters; it++) {
        const Doml2W *w = &j->wv[(j->rot + it) % j->nbuf];
        if (ith == 0) mark = now3();
        uint32_t ya, yb;
        doml2_slice(ny, ith, nth, &ya, &yb);
        doml3_quant_x_rows(j->X, K, j->qx, ya, yb);
        doml2_pool_barrier(pool, ith); /* xq ready for everyone */
        if (ith == 0) {
            j->tq += now3() - mark;
            mark = now3();
        }
        uint32_t rg0, rg1;
        doml2_slice(w->R / 16, ith, nth, &rg0, &rg1);
        Doml3Panel *pan = &j->panels[ith];
        if (j->split) {
            doml3_convert(w, rg0, rg1, pan);
            doml2_pool_barrier(pool, ith);
            if (ith == 0) {
                j->tc += now3() - mark;
                mark = now3();
            }
            doml3_gemm_strip(pan, (rg1 - rg0) * 16, rg0 * 16, j->qx, j->C,
                             j->ldc, j->mk);
            doml2_pool_barrier(pool, ith); /* call complete */
            if (ith == 0) j->tg += now3() - mark;
        } else {
            const uint32_t step = pan->srows_cap / 16;
            for (uint32_t rs = rg0; rs < rg1; rs += step) {
                uint32_t re = rs + step > rg1 ? rg1 : rs + step;
                doml3_convert(w, rs, re, pan);
                doml3_gemm_strip(pan, (re - rs) * 16, rs * 16, j->qx, j->C,
                                 j->ldc, j->mk);
            }
            doml2_pool_barrier(pool, ith); /* call complete */
            if (ith == 0) j->tg += now3() - mark;
        }
    }
    if (ith == 0) j->t1 = now3();
}
