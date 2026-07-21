/* DOML GEMV decode microkernel — implementation. See doml_gemv.h.
 *
 * Inner loop per (row, 64-column chunk):
 *   s64   : 8 B of the salient bitmap (shared across rows)
 *   m64   : pdep(next m bits, ~s64)          -- membership expansion
 *   nb64  : s64 | m64                        -- non-bulk mask
 *   b164  : pdep(next b1 bits, nb64)         -- high-code expansion
 *   b064  : 8 B of the full b0 plane
 *   idx   : byte lane j = b0 | b1<<1 | m<<2 | s<<3   in [0,12)
 *           (4 masked byte-adds from the four 64-bit masks)
 *   value : vpermb against the 16-slot per-(row,group) table
 *           built on the fly from the 10 resident fp8 bytes.
 *
 * The m bit-stream position advances by popcount(~s64) per chunk — IDENTICAL
 * for every row (salience is column-wise) — so it is tracked once per tile.
 * The b1 position advances by popcount(nb64), per row.
 *
 * FP path : two byte tables (bf16 lo/hi) -> vpunpcklbw/hbw -> bf16 words ->
 *           vpunpcklwd/hwd with zero -> fp32 (=bf16<<16) -> FMA against a
 *           once-per-call permuted copy of x (dot products are order-
 *           invariant; the permutation makes the unpack order line up).
 * I8 path : one u8 table (level -> q+128) -> vpdpbusd against int8 x,
 *           per-(row,group) fixup y += s_rg*dx_g*(acc - 128*sum_qx_g).
 *
 * On-the-fly per-(row,group) table prep is fully vectorized (the fp8->bf16
 * widening is a 128-entry vpermi2b LUT split into lo/hi byte tables; the I8
 * level quantization exploits that the bf16 bit pattern of |v| is monotone
 * in |v|, so the per-group max is an integer vpmaxuw reduction).
 *
 * Row tiles are hand-unrolled via macros: gcc 11 does NOT fully unroll the
 * equivalent `for (i < NR)` loops (measured: it keeps zmm state in stack
 * arrays -> 3-6x slowdown), so per-row state lives in named locals.
 */
#include "doml_gemv.h"
#include "../ref/ref_decode.h"

#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NG_MAX 12

/* ------------------------------------------------------------- tables ----- */

static uint16_t g_fp8bf16[256];
/* fp8->bf16 byte LUTs over the low 7 bits (sign handled separately):
 * two 64-byte halves each for vpermi2b */
static uint8_t g_lut_lo[128] __attribute__((aligned(64)));
static uint8_t g_lut_hi[128] __attribute__((aligned(64)));
/* scatter pattern: 10 resident cb slots -> 16-slot kernel table, 4 groups
 * per zmm */
static uint8_t g_expand_pat[64] __attribute__((aligned(64)));
static int g_init_done = 0;
static int g_tile_fp = 4, g_tile_i8 = 4; /* env DOML_TILE_FP / DOML_TILE_I8 */

/* resident slot k (of [bulk0,bulk1,tail0..3,sal0..3]) -> kernel table slot */
static const uint8_t k_slotpos[10] = { 0, 1, 4, 5, 6, 7, 8, 9, 10, 11 };
#define RB16_MASK 0x0FF30FF30FF30FF3ULL /* valid kernel slots, 4 groups */

void doml_gemv_init(void)
{
    if (g_init_done) return;
    dpka_fp8e4m3_to_bf16_table(g_fp8bf16);
    for (int b = 0; b < 128; b++) {
        g_lut_lo[b] = (uint8_t)(g_fp8bf16[b] & 0xFF);
        g_lut_hi[b] = (uint8_t)(g_fp8bf16[b] >> 8); /* sign 0 -> bit7 == 0 */
    }
    /* inverse of k_slotpos; unreachable slots (2,3,12..15) are masked off */
    uint8_t inv[16] = { 0 };
    for (int k = 0; k < 10; k++) inv[k_slotpos[k]] = (uint8_t)k;
    for (int j = 0; j < 64; j++)
        g_expand_pat[j] = (uint8_t)((j >> 4) * 10 + inv[j & 15]);
    /* row-tile height bake-off knobs (default 4; see P2B_REPORT.md) */
    {
        const char *e = getenv("DOML_TILE_FP");
        if (e) g_tile_fp = atoi(e);
        e = getenv("DOML_TILE_I8");
        if (e) g_tile_i8 = atoi(e);
    }
    g_init_done = 1;
}

static inline uint64_t load_u64(const void *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* Next k bits of an LSB-first bit stream from byte pointer p, bit shift sh.
 * The consumer (pdep) only uses the low k bits, so a single unaligned load
 * suffices whenever sh + k <= 64; the 9th-byte merge is a rarely-taken
 * branch (m side: shared per chunk; b1 side: needs >57 non-bulk of 64).
 * (63 ^ sh) == 63 - sh for sh in [0,7]; the extra <<1 makes sh==0 safe.
 * Reads up to 9 bytes; callers guarantee tail slack (slab padding). */
static inline uint64_t take_bits(const uint8_t *p, unsigned sh, unsigned k)
{
    uint64_t lo = load_u64(p) >> sh;
    if (__builtin_expect(sh + k > 64, 0))
        lo |= ((uint64_t)p[8] << (63 ^ sh)) << 1;
    return lo;
}

/* --------------------------------------------------------------- slab ----- */

#define AL 64u
#define PAD_STREAM 512u /* tail slack after the two bit-stream planes: the
                         * 16B unaligned loads may read past the last row, and
                         * the G-UNIQUE ablation (s or m zeroed) shifts stream
                         * consumption by up to C bits for the last row. */

typedef struct {
    size_t b0, b1, m, cb, off, s, total;
} SlabOffs;

static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

static SlabOffs slab_offs(const DpkaResB *rb, int m_full)
{
    size_t m_bytes = m_full ? (size_t)rb->R * (rb->C / 8) : rb->bytes_m;
    SlabOffs o;
    size_t p = 0;
    o.b0 = p;  p = align_up(p + rb->bytes_b0, AL);
    o.b1 = p;  p = align_up(p + rb->bytes_b1 + PAD_STREAM, AL);
    o.m  = p;  p = align_up(p + m_bytes + PAD_STREAM, AL);
    o.cb = p;  p = align_up(p + rb->bytes_cb, AL);
    o.off = p; p = align_up(p + rb->bytes_b1off, AL);
    o.s  = p;  p = align_up(p + rb->bytes_s, AL);
    o.total = p + AL;
    return o;
}

size_t doml_gemv_slab_bytes(const DpkaResB *rb, int m_full)
{
    return slab_offs(rb, m_full).total;
}

void doml_gemv_pack_init(const DpkaResB *rb, uint8_t *slab, DomlGemvW *w,
                         int m_full)
{
    SlabOffs o = slab_offs(rb, m_full);
    w->R = rb->R;
    w->C = rb->C;
    w->NG = rb->NG;
    w->g = rb->g;
    w->n_sal = rb->n_sal;
    w->m_full = m_full ? 1u : 0u;
    w->m_pitch = m_full ? rb->C / 8 : rb->m_pitch;
    w->b0 = slab + o.b0;
    w->b1 = slab + o.b1;
    w->m = slab + o.m;
    w->cb = slab + o.cb;
    w->b1_rowoff = (const uint32_t *)(slab + o.off);
    w->s = slab + o.s;
    /* exact resident bytes the kernel streams (BW accounting; pads excluded) */
    w->weight_bytes = rb->bytes_b0 + rb->bytes_b1 + rb->bytes_b1off +
                      (size_t)w->m_pitch * rb->R + rb->bytes_s + rb->bytes_cb;
    /* shared small section: salient bitmap (row sections are first-touched
     * per thread by doml_gemv_pack_rows) */
    memcpy((uint8_t *)(uintptr_t)w->s, rb->s, rb->bytes_s);
    if (w->C % 256 != 0 || w->g != 256 || w->NG > NG_MAX || w->NG % 4 != 0) {
        fprintf(stderr,
                "doml_gemv: unsupported C=%u g=%u NG=%u (need C%%256==0, "
                "g=256, NG%%4==0, NG<=%d)\n",
                w->C, w->g, w->NG, NG_MAX);
        abort();
    }
}

void doml_gemv_pack_rows(const DpkaResB *rb, const DomlGemvW *w,
                         uint32_t r0, uint32_t r1)
{
    if (r1 <= r0) return;
    const size_t pitch = rb->C / 8;
    memcpy((uint8_t *)(uintptr_t)w->b0 + (size_t)r0 * pitch,
           rb->b0 + (size_t)r0 * pitch, (size_t)(r1 - r0) * pitch);
    if (w->m_full) {
        /* expand packed non-salient bits to the full plane (== R-A m);
         * bounce the last rows so take_bits' 9-byte reads stay inside
         * rb->m (the malloc'd source has no tail slack) */
        uint8_t bounce[3072 / 8 + 16];
        for (uint32_t r = r0; r < r1; r++) {
            const uint8_t *src = rb->m + (size_t)r * rb->m_pitch;
            if ((size_t)(r + 1) * rb->m_pitch + 16 > rb->bytes_m) {
                memcpy(bounce, src, rb->m_pitch);
                memset(bounce + rb->m_pitch, 0, 16);
                src = bounce;
            }
            uint8_t *dst =
                (uint8_t *)(uintptr_t)w->m + (size_t)r * pitch;
            uint64_t mbit = 0;
            for (uint32_t cch = 0; cch < rb->C / 64; cch++) {
                uint64_t s64 = load_u64(rb->s + (size_t)cch * 8);
                uint64_t ns64 = ~s64;
                unsigned kns = (unsigned)__builtin_popcountll(ns64);
                uint64_t v = _pdep_u64(
                    take_bits(src + (mbit >> 3), (unsigned)mbit & 7, kns),
                    ns64);
                memcpy(dst + (size_t)cch * 8, &v, 8);
                mbit += kns;
            }
        }
    } else {
        memcpy((uint8_t *)(uintptr_t)w->m + (size_t)r0 * rb->m_pitch,
               rb->m + (size_t)r0 * rb->m_pitch,
               (size_t)(r1 - r0) * rb->m_pitch);
    }
    memcpy((uint8_t *)(uintptr_t)w->cb + (size_t)r0 * rb->NG * 10,
           rb->cb + (size_t)r0 * rb->NG * 10, (size_t)(r1 - r0) * rb->NG * 10);
    memcpy((uint8_t *)(uintptr_t)w->b1 + rb->b1_rowoff[r0],
           rb->b1 + rb->b1_rowoff[r0],
           rb->b1_rowoff[r1] - rb->b1_rowoff[r0]);
    /* rowoff slice incl. the shared fencepost (overlap writes are benign) */
    memcpy((uint32_t *)(uintptr_t)w->b1_rowoff + r0, rb->b1_rowoff + r0,
           ((size_t)(r1 - r0) + 1) * sizeof(uint32_t));
}

void doml_gemv_slice(uint32_t R, int ith, int nth, uint32_t *r0, uint32_t *r1)
{
    uint32_t base = R / (uint32_t)nth, rem = R % (uint32_t)nth;
    uint32_t u = (uint32_t)ith;
    uint32_t lo = u * base + (u < rem ? u : rem);
    *r0 = lo;
    *r1 = lo + base + (u < rem ? 1u : 0u);
}

/* --------------------------------------------------------- activations ---- */

void doml_gemv_prep_x_fp(const float *x, uint32_t C, float *xperm)
{
    /* Chunk-local unpack order (see FP_ROW): output vector k in {0..3} of
     * chunk c holds column 16*l + 8*(k>>1) + 2*d + (k&1) at lane 4*l + d
     * (vpunpck{l,h}bw then vpslld-16 / vpandd-hi16 word split). */
    for (uint32_t c = 0; c < C / 64; c++) {
        const float *xs = x + (size_t)c * 64;
        float *xd = xperm + (size_t)c * 64;
        for (int k = 0; k < 4; k++)
            for (int l = 0; l < 4; l++)
                for (int d = 0; d < 4; d++)
                    xd[16 * k + 4 * l + d] =
                        xs[16 * l + 8 * (k >> 1) + 2 * d + (k & 1)];
    }
}

void doml_gemv_prep_x_i8(const float *x, uint32_t C, DomlQx *qx)
{
    uint32_t NG = C / 256;
    qx->C = C;
    qx->NG = NG;
    for (uint32_t g = 0; g < NG; g++) {
        const float *xs = x + (size_t)g * 256;
        float maxa = 0.f;
        for (int j = 0; j < 256; j++) {
            float a = fabsf(xs[j]);
            if (a > maxa) maxa = a;
        }
        float dx = maxa > 0 ? maxa / 127.f : 0.f;
        float inv = maxa > 0 ? 127.f / maxa : 0.f;
        int sum = 0;
        for (int j = 0; j < 256; j++) {
            int q = (int)lrintf(xs[j] * inv);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qx->q[(size_t)g * 256 + j] = (int8_t)q;
            sum += q;
        }
        qx->dx[g] = dx;
        qx->c128[g] = 128.f * dx * (float)sum;
    }
}

/* -------------------------------------------------- per-row table prep ---- */

void doml_gemv_prep_row_fp(const DomlGemvW *w, uint32_t r,
                           uint8_t (*tlo)[16], uint8_t (*thi)[16])
{
    const uint8_t *cbr = w->cb + (size_t)r * w->NG * 10;
    const __m512i P = _mm512_load_si512(g_expand_pat);
    const __m512i L0 = _mm512_load_si512(g_lut_lo);
    const __m512i L1 = _mm512_load_si512(g_lut_lo + 64);
    const __m512i H0 = _mm512_load_si512(g_lut_hi);
    const __m512i H1 = _mm512_load_si512(g_lut_hi + 64);
    const __m512i m7f = _mm512_set1_epi8(0x7F);
    const __m512i m80 = _mm512_set1_epi8((char)0x80);
    for (uint32_t b = 0; b < w->NG; b += 4) {
        /* 64B load covers the 40 cb bytes of 4 groups (overread stays inside
         * the cb section / slab pad) */
        __m512i raw = _mm512_loadu_si512(cbr + (size_t)b * 10);
        __m512i cb16 =
            _mm512_maskz_permutexvar_epi8((__mmask64)RB16_MASK, P, raw);
        __m512i i7 = _mm512_and_si512(cb16, m7f);
        __m512i sg = _mm512_and_si512(cb16, m80);
        __m512i lo = _mm512_permutex2var_epi8(L0, i7, L1);
        __m512i hi =
            _mm512_or_si512(_mm512_permutex2var_epi8(H0, i7, H1), sg);
        _mm512_storeu_si512(tlo[b], lo);
        _mm512_storeu_si512(thi[b], hi);
    }
}

void doml_gemv_prep_row_i8(const DomlGemvW *w, uint32_t r,
                           uint8_t (*ut)[16], float *sw)
{
    const uint8_t *cbr = w->cb + (size_t)r * w->NG * 10;
    const __m512i P = _mm512_load_si512(g_expand_pat);
    const __m512i L0 = _mm512_load_si512(g_lut_lo);
    const __m512i L1 = _mm512_load_si512(g_lut_lo + 64);
    const __m512i H0 = _mm512_load_si512(g_lut_hi);
    const __m512i H1 = _mm512_load_si512(g_lut_hi + 64);
    const __m512i m7f = _mm512_set1_epi8(0x7F);
    const __m512i m80 = _mm512_set1_epi8((char)0x80);
    const __m512i m7fff = _mm512_set1_epi16(0x7FFF);
    const __m512i mffff = _mm512_set1_epi32(0xFFFF);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i lane0 = _mm512_set_epi32(12, 12, 12, 12, 8, 8, 8, 8,
                                           4, 4, 4, 4, 0, 0, 0, 0);
    const __m512 f127 = _mm512_set1_ps(127.f);
    const __m512 r127 = _mm512_set1_ps(1.f / 127.f);
    for (uint32_t b = 0; b < w->NG; b += 4) {
        __m512i raw = _mm512_loadu_si512(cbr + (size_t)b * 10);
        __m512i cb16 =
            _mm512_maskz_permutexvar_epi8((__mmask64)RB16_MASK, P, raw);
        __m512i i7 = _mm512_and_si512(cb16, m7f);
        __m512i sg = _mm512_and_si512(cb16, m80);
        __m512i lo = _mm512_permutex2var_epi8(L0, i7, L1);
        __m512i hi =
            _mm512_or_si512(_mm512_permutex2var_epi8(H0, i7, H1), sg);
        /* bf16 words per group lane: slots 0-7 in ulo, 8-15 in uhi */
        __m512i ulo = _mm512_unpacklo_epi8(lo, hi);
        __m512i uhi = _mm512_unpackhi_epi8(lo, hi);
        /* per-group max |level|: |bf16| bit pattern is monotone in |v| */
        __m512i ab = _mm512_max_epu16(_mm512_and_si512(ulo, m7fff),
                                      _mm512_and_si512(uhi, m7fff));
        ab = _mm512_max_epu16(ab, _mm512_bsrli_epi128(ab, 8));
        ab = _mm512_max_epu16(ab, _mm512_bsrli_epi128(ab, 4));
        ab = _mm512_max_epu16(ab, _mm512_bsrli_epi128(ab, 2));
        __m512i mxd = _mm512_and_si512(ab, mffff); /* dword0/lane = max bits */
        __m512i mxb = _mm512_permutexvar_epi32(lane0, mxd);
        __m512 mxf = _mm512_castsi512_ps(_mm512_slli_epi32(mxb, 16));
        __mmask16 kk =
            _mm512_cmp_ps_mask(mxf, _mm512_setzero_ps(), _CMP_GT_OQ);
        /* inv = 127/max via rcp14 (rel err <= 2^-14): |q| <= 127*(1+2^-14)
         * < 127.5, so the nearest-int quantization still lands in [-127,127].
         * (vdivps zmm has ~16-cycle throughput and dominated prep cost.) */
        __m512 invb = _mm512_mul_ps(_mm512_maskz_rcp14_ps(kk, mxf), f127);
        __m512 scb = _mm512_mul_ps(mxf, r127);
        float sc[16] __attribute__((aligned(64)));
        _mm512_store_ps(sc, scb);
        sw[b + 0] = sc[0];
        sw[b + 1] = sc[4];
        sw[b + 2] = sc[8];
        sw[b + 3] = sc[12];
        /* widen to fp32 (slots 0-3/4-7/8-11/12-15 per lane), quantize */
        __m512 f0 = _mm512_castsi512_ps(_mm512_unpacklo_epi16(zero, ulo));
        __m512 f1 = _mm512_castsi512_ps(_mm512_unpackhi_epi16(zero, ulo));
        __m512 f2 = _mm512_castsi512_ps(_mm512_unpacklo_epi16(zero, uhi));
        __m512 f3 = _mm512_castsi512_ps(_mm512_unpackhi_epi16(zero, uhi));
        __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, invb));
        __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, invb));
        __m512i q2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, invb));
        __m512i q3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, invb));
        /* in-lane saturating packs land lane l = group l slots 0..15 in
         * order (q0..q3 hold slots 0-3/4-7/8-11/12-15 per lane); +128 */
        __m512i qq = _mm512_packs_epi16(_mm512_packs_epi32(q0, q1),
                                        _mm512_packs_epi32(q2, q3));
        qq = _mm512_add_epi8(qq, m80); /* +128 mod 256 */
        _mm512_storeu_si512(ut[b], qq);
    }
}

/* ------------------------------------------------------------ FP kernel --- */

#define BCAST16(p) \
    _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)(p)))

/* shared decode of one row's 64-column chunk into the idx byte vector.
 * b1 walks via a combined bit address bq##n (= ptr*8 + bit), b0 loads
 * straight into a mask register; the s mask ks is hoisted per chunk. */
#define IDX_ROW(n, MF)                                                        \
    uint64_t m64##n = (MF) ? load_u64(mr##n + (size_t)c * 8)                  \
                           : _pdep_u64(take_bits(mr##n + mb, msh, kns),       \
                                       ns64);                                 \
    /* m64 subset of ~s64 -> disjoint: '+' == '|', and gcc cannot lift an    \
     * add into the k-domain (a korq+kmovq round trip costs 4 cycles on the  \
     * pdep/popcnt chain) */                                                  \
    uint64_t nb64##n = s64 + m64##n;                                          \
    unsigned pc##n = (unsigned)__builtin_popcountll(nb64##n);                 \
    uint64_t b164##n = _pdep_u64(                                             \
        take_bits((const uint8_t *)(uintptr_t)(bq##n >> 3),                   \
                  (unsigned)bq##n & 7, pc##n),                                \
        nb64##n);                                                             \
    bq##n += pc##n;                                                           \
    __mmask64 kb0##n = _load_mask64(                                          \
        (__mmask64 *)(uintptr_t)(b0r##n + (size_t)c * 8));                    \
    /* independent maskz terms + OR3 ternlog: no destructive masked merges   \
     * (those made gcc emit a zmm-copy per step), latency 2 instead of 4 */  \
    __m512i idx##n = _mm512_ternarylogic_epi64(                               \
        _mm512_maskz_mov_epi8(kb0##n, vone),                                  \
        _mm512_maskz_mov_epi8((__mmask64)b164##n, vtwo),                      \
        _mm512_maskz_mov_epi8((__mmask64)m64##n, vfour), 0xFE);               \
    idx##n = _mm512_or_si512(idx##n, v8s)

/* one row's decode+FMA for one 64-column chunk (locals via token pasting) */
/* bf16 words w of a dword pair expand to fp32 as (u << 16) for the even
 * word and (u & 0xFFFF0000) for the odd word — vpslld/vpandd instead of
 * vpunpck*wd keeps these off port 5 (which the vpermb/vpunpckbw already
 * saturate); the resulting order change is absorbed by the x permutation. */
#define FP_ROW(n, MF)                                                         \
    do {                                                                      \
        IDX_ROW(n, MF);                                                       \
        __m512i lo = _mm512_permutexvar_epi8(idx##n, vl##n);                  \
        __m512i hi = _mm512_permutexvar_epi8(idx##n, vh##n);                  \
        __m512i ulo = _mm512_unpacklo_epi8(lo, hi);                           \
        __m512i uhi = _mm512_unpackhi_epi8(lo, hi);                           \
        acA##n = _mm512_fmadd_ps(                                             \
            _mm512_castsi512_ps(_mm512_slli_epi32(ulo, 16)), x0, acA##n);     \
        acB##n = _mm512_fmadd_ps(                                             \
            _mm512_castsi512_ps(_mm512_and_si512(ulo, vhi16)), x1, acB##n);   \
        acA##n = _mm512_fmadd_ps(                                             \
            _mm512_castsi512_ps(_mm512_slli_epi32(uhi, 16)), x2, acA##n);     \
        acB##n = _mm512_fmadd_ps(                                             \
            _mm512_castsi512_ps(_mm512_and_si512(uhi, vhi16)), x3, acB##n);   \
    } while (0)

#define FP_ROW_DECLS(n)                                                      \
    const uint8_t *b0r##n = b0 + (size_t)(r + n) * pitch;                    \
    const uint8_t *mr##n = mp + (size_t)(r + n) * m_pitch;                   \
    uint64_t bq##n = ((uint64_t)(uintptr_t)(b1 + roff[r + n])) * 8;          \
    __m512 acA##n = _mm512_setzero_ps(), acB##n = _mm512_setzero_ps()

#define FP_COMMON_DECLS                                                     \
    const uint32_t NG = w->NG;                                              \
    const size_t pitch = w->C / 8;                                          \
    const uint32_t m_pitch = w->m_pitch;                                    \
    const uint8_t *b0 = w->b0, *mp = w->m, *b1 = w->b1, *sp = w->s;         \
    const uint32_t *roff = w->b1_rowoff;                                    \
    const __m512i vone = _mm512_set1_epi8(1);                               \
    const __m512i vtwo = _mm512_set1_epi8(2);                               \
    const __m512i vfour = _mm512_set1_epi8(4);                              \
    const __m512i veight = _mm512_set1_epi8(8);                             \
    const __m512i vhi16 = _mm512_set1_epi32((int)0xFFFF0000)

static inline __attribute__((always_inline)) void
fp_tile4_g(const DomlGemvW *w, const float *xperm, float *y, uint32_t r,
           const int MF)
{
    FP_COMMON_DECLS;
    uint8_t tlo[4][NG_MAX][16] __attribute__((aligned(64)));
    uint8_t thi[4][NG_MAX][16] __attribute__((aligned(64)));
    doml_gemv_prep_row_fp(w, r + 0, tlo[0], thi[0]);
    doml_gemv_prep_row_fp(w, r + 1, tlo[1], thi[1]);
    doml_gemv_prep_row_fp(w, r + 2, tlo[2], thi[2]);
    doml_gemv_prep_row_fp(w, r + 3, tlo[3], thi[3]);
    FP_ROW_DECLS(0);
    FP_ROW_DECLS(1);
    FP_ROW_DECLS(2);
    FP_ROW_DECLS(3);
    uint64_t mbit = 0;
    for (uint32_t gi = 0; gi < NG; gi++) {
        const __m512i vl0 = BCAST16(tlo[0][gi]), vh0 = BCAST16(thi[0][gi]);
        const __m512i vl1 = BCAST16(tlo[1][gi]), vh1 = BCAST16(thi[1][gi]);
        const __m512i vl2 = BCAST16(tlo[2][gi]), vh2 = BCAST16(thi[2][gi]);
        const __m512i vl3 = BCAST16(tlo[3][gi]), vh3 = BCAST16(thi[3][gi]);
        for (uint32_t ci = 0; ci < 4; ci++) {
            const uint32_t c = gi * 4 + ci;
            const uint64_t s64 = load_u64(sp + (size_t)c * 8);
            const uint64_t ns64 = ~s64;
            const __mmask64 ks = (__mmask64)s64;
            const __m512i v8s = _mm512_maskz_mov_epi8(ks, veight);
            const uint64_t mb = mbit >> 3;
            const unsigned msh = (unsigned)mbit & 7;
            const unsigned kns = (unsigned)__builtin_popcountll(ns64);
            const float *xp = xperm + (size_t)c * 64;
            const __m512 x0 = _mm512_load_ps(xp);
            const __m512 x1 = _mm512_load_ps(xp + 16);
            const __m512 x2 = _mm512_load_ps(xp + 32);
            const __m512 x3 = _mm512_load_ps(xp + 48);
            FP_ROW(0, MF);
            FP_ROW(1, MF);
            FP_ROW(2, MF);
            FP_ROW(3, MF);
            mbit += kns;
        }
    }
    y[r + 0] = _mm512_reduce_add_ps(_mm512_add_ps(acA0, acB0));
    y[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(acA1, acB1));
    y[r + 2] = _mm512_reduce_add_ps(_mm512_add_ps(acA2, acB2));
    y[r + 3] = _mm512_reduce_add_ps(_mm512_add_ps(acA3, acB3));
}

static inline __attribute__((always_inline)) void
fp_tile2_g(const DomlGemvW *w, const float *xperm, float *y, uint32_t r,
           const int MF)
{
    FP_COMMON_DECLS;
    uint8_t tlo[2][NG_MAX][16] __attribute__((aligned(64)));
    uint8_t thi[2][NG_MAX][16] __attribute__((aligned(64)));
    doml_gemv_prep_row_fp(w, r + 0, tlo[0], thi[0]);
    doml_gemv_prep_row_fp(w, r + 1, tlo[1], thi[1]);
    FP_ROW_DECLS(0);
    FP_ROW_DECLS(1);
    uint64_t mbit = 0;
    for (uint32_t gi = 0; gi < NG; gi++) {
        const __m512i vl0 = BCAST16(tlo[0][gi]), vh0 = BCAST16(thi[0][gi]);
        const __m512i vl1 = BCAST16(tlo[1][gi]), vh1 = BCAST16(thi[1][gi]);
        for (uint32_t ci = 0; ci < 4; ci++) {
            const uint32_t c = gi * 4 + ci;
            const uint64_t s64 = load_u64(sp + (size_t)c * 8);
            const uint64_t ns64 = ~s64;
            const __mmask64 ks = (__mmask64)s64;
            const __m512i v8s = _mm512_maskz_mov_epi8(ks, veight);
            const uint64_t mb = mbit >> 3;
            const unsigned msh = (unsigned)mbit & 7;
            const unsigned kns = (unsigned)__builtin_popcountll(ns64);
            const float *xp = xperm + (size_t)c * 64;
            const __m512 x0 = _mm512_load_ps(xp);
            const __m512 x1 = _mm512_load_ps(xp + 16);
            const __m512 x2 = _mm512_load_ps(xp + 32);
            const __m512 x3 = _mm512_load_ps(xp + 48);
            FP_ROW(0, MF);
            FP_ROW(1, MF);
            mbit += kns;
        }
    }
    y[r + 0] = _mm512_reduce_add_ps(_mm512_add_ps(acA0, acB0));
    y[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(acA1, acB1));
}

static inline __attribute__((always_inline)) void
fp_tile1_g(const DomlGemvW *w, const float *xperm, float *y, uint32_t r,
           const int MF)
{
    FP_COMMON_DECLS;
    uint8_t tlo[1][NG_MAX][16] __attribute__((aligned(64)));
    uint8_t thi[1][NG_MAX][16] __attribute__((aligned(64)));
    doml_gemv_prep_row_fp(w, r, tlo[0], thi[0]);
    FP_ROW_DECLS(0);
    uint64_t mbit = 0;
    for (uint32_t gi = 0; gi < NG; gi++) {
        const __m512i vl0 = BCAST16(tlo[0][gi]), vh0 = BCAST16(thi[0][gi]);
        for (uint32_t ci = 0; ci < 4; ci++) {
            const uint32_t c = gi * 4 + ci;
            const uint64_t s64 = load_u64(sp + (size_t)c * 8);
            const uint64_t ns64 = ~s64;
            const __mmask64 ks = (__mmask64)s64;
            const __m512i v8s = _mm512_maskz_mov_epi8(ks, veight);
            const uint64_t mb = mbit >> 3;
            const unsigned msh = (unsigned)mbit & 7;
            const unsigned kns = (unsigned)__builtin_popcountll(ns64);
            const float *xp = xperm + (size_t)c * 64;
            const __m512 x0 = _mm512_load_ps(xp);
            const __m512 x1 = _mm512_load_ps(xp + 16);
            const __m512 x2 = _mm512_load_ps(xp + 32);
            const __m512 x3 = _mm512_load_ps(xp + 48);
            FP_ROW(0, MF);
            mbit += kns;
        }
    }
    y[r] = _mm512_reduce_add_ps(_mm512_add_ps(acA0, acB0));
}

void doml_gemv_fp_rows(const DomlGemvW *w, const float *xperm, float *y,
                       uint32_t r0, uint32_t r1)
{
    uint32_t r = r0;
    if (w->m_full) {
        if (g_tile_fp >= 4)
            for (; r + 4 <= r1; r += 4) fp_tile4_g(w, xperm, y, r, 1);
        if (g_tile_fp >= 2)
            for (; r + 2 <= r1; r += 2) fp_tile2_g(w, xperm, y, r, 1);
        for (; r < r1; r++) fp_tile1_g(w, xperm, y, r, 1);
    } else {
        if (g_tile_fp >= 4)
            for (; r + 4 <= r1; r += 4) fp_tile4_g(w, xperm, y, r, 0);
        if (g_tile_fp >= 2)
            for (; r + 2 <= r1; r += 2) fp_tile2_g(w, xperm, y, r, 0);
        for (; r < r1; r++) fp_tile1_g(w, xperm, y, r, 0);
    }
}

/* ------------------------------------------------------------ I8 kernel --- */

#define I8_ROW(n, MF)                                                         \
    do {                                                                      \
        IDX_ROW(n, MF);                                                       \
        __m512i wv = _mm512_permutexvar_epi8(idx##n, tb##n);                  \
        ia##n = _mm512_dpbusd_epi32(ia##n, wv, qv);                           \
    } while (0)

#define I8_ROW_DECLS(n)                                                      \
    const uint8_t *b0r##n = b0 + (size_t)(r + n) * pitch;                    \
    const uint8_t *mr##n = mp + (size_t)(r + n) * m_pitch;                   \
    uint64_t bq##n = ((uint64_t)(uintptr_t)(b1 + roff[r + n])) * 8;          \
    __m512 ya##n = _mm512_setzero_ps();                                      \
    float corr##n = 0.f

#define I8_FIXUP(n)                                                          \
    do {                                                                     \
        const float s_ = sw[n][gi];                                          \
        corr##n += s_ * c128g;                                               \
        ya##n = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia##n),                   \
                                _mm512_set1_ps(s_ * dxg), ya##n);            \
    } while (0)

static inline __attribute__((always_inline)) void
i8_tile4_g(const DomlGemvW *w, const DomlQx *qx, float *y, uint32_t r,
           const int MF)
{
    FP_COMMON_DECLS;
    (void)vhi16;
    uint8_t ut[4][NG_MAX][16] __attribute__((aligned(64)));
    float sw[4][NG_MAX];
    doml_gemv_prep_row_i8(w, r + 0, ut[0], sw[0]);
    doml_gemv_prep_row_i8(w, r + 1, ut[1], sw[1]);
    doml_gemv_prep_row_i8(w, r + 2, ut[2], sw[2]);
    doml_gemv_prep_row_i8(w, r + 3, ut[3], sw[3]);
    I8_ROW_DECLS(0);
    I8_ROW_DECLS(1);
    I8_ROW_DECLS(2);
    I8_ROW_DECLS(3);
    const int8_t *qp = qx->q;
    uint64_t mbit = 0;
    for (uint32_t gi = 0; gi < NG; gi++) {
        const __m512i tb0 = BCAST16(ut[0][gi]);
        const __m512i tb1 = BCAST16(ut[1][gi]);
        const __m512i tb2 = BCAST16(ut[2][gi]);
        const __m512i tb3 = BCAST16(ut[3][gi]);
        __m512i ia0 = _mm512_setzero_si512(), ia1 = _mm512_setzero_si512();
        __m512i ia2 = _mm512_setzero_si512(), ia3 = _mm512_setzero_si512();
        for (uint32_t ci = 0; ci < 4; ci++) {
            const uint32_t c = gi * 4 + ci;
            const uint64_t s64 = load_u64(sp + (size_t)c * 8);
            const uint64_t ns64 = ~s64;
            const __mmask64 ks = (__mmask64)s64;
            const __m512i v8s = _mm512_maskz_mov_epi8(ks, veight);
            const uint64_t mb = mbit >> 3;
            const unsigned msh = (unsigned)mbit & 7;
            const unsigned kns = (unsigned)__builtin_popcountll(ns64);
            const __m512i qv =
                _mm512_load_si512((const void *)(qp + (size_t)c * 64));
            I8_ROW(0, MF);
            I8_ROW(1, MF);
            I8_ROW(2, MF);
            I8_ROW(3, MF);
            mbit += kns;
        }
        const float dxg = qx->dx[gi];
        const float c128g = qx->c128[gi];
        I8_FIXUP(0);
        I8_FIXUP(1);
        I8_FIXUP(2);
        I8_FIXUP(3);
    }
    y[r + 0] = _mm512_reduce_add_ps(ya0) - corr0;
    y[r + 1] = _mm512_reduce_add_ps(ya1) - corr1;
    y[r + 2] = _mm512_reduce_add_ps(ya2) - corr2;
    y[r + 3] = _mm512_reduce_add_ps(ya3) - corr3;
}

static inline __attribute__((always_inline)) void
i8_tile2_g(const DomlGemvW *w, const DomlQx *qx, float *y, uint32_t r,
           const int MF)
{
    FP_COMMON_DECLS;
    (void)vhi16;
    uint8_t ut[2][NG_MAX][16] __attribute__((aligned(64)));
    float sw[2][NG_MAX];
    doml_gemv_prep_row_i8(w, r + 0, ut[0], sw[0]);
    doml_gemv_prep_row_i8(w, r + 1, ut[1], sw[1]);
    I8_ROW_DECLS(0);
    I8_ROW_DECLS(1);
    const int8_t *qp = qx->q;
    uint64_t mbit = 0;
    for (uint32_t gi = 0; gi < NG; gi++) {
        const __m512i tb0 = BCAST16(ut[0][gi]);
        const __m512i tb1 = BCAST16(ut[1][gi]);
        __m512i ia0 = _mm512_setzero_si512(), ia1 = _mm512_setzero_si512();
        for (uint32_t ci = 0; ci < 4; ci++) {
            const uint32_t c = gi * 4 + ci;
            const uint64_t s64 = load_u64(sp + (size_t)c * 8);
            const uint64_t ns64 = ~s64;
            const __mmask64 ks = (__mmask64)s64;
            const __m512i v8s = _mm512_maskz_mov_epi8(ks, veight);
            const uint64_t mb = mbit >> 3;
            const unsigned msh = (unsigned)mbit & 7;
            const unsigned kns = (unsigned)__builtin_popcountll(ns64);
            const __m512i qv =
                _mm512_load_si512((const void *)(qp + (size_t)c * 64));
            I8_ROW(0, MF);
            I8_ROW(1, MF);
            mbit += kns;
        }
        const float dxg = qx->dx[gi];
        const float c128g = qx->c128[gi];
        I8_FIXUP(0);
        I8_FIXUP(1);
    }
    y[r + 0] = _mm512_reduce_add_ps(ya0) - corr0;
    y[r + 1] = _mm512_reduce_add_ps(ya1) - corr1;
}

static inline __attribute__((always_inline)) void
i8_tile1_g(const DomlGemvW *w, const DomlQx *qx, float *y, uint32_t r,
           const int MF)
{
    FP_COMMON_DECLS;
    (void)vhi16;
    uint8_t ut[1][NG_MAX][16] __attribute__((aligned(64)));
    float sw[1][NG_MAX];
    doml_gemv_prep_row_i8(w, r, ut[0], sw[0]);
    I8_ROW_DECLS(0);
    const int8_t *qp = qx->q;
    uint64_t mbit = 0;
    for (uint32_t gi = 0; gi < NG; gi++) {
        const __m512i tb0 = BCAST16(ut[0][gi]);
        __m512i ia0 = _mm512_setzero_si512();
        for (uint32_t ci = 0; ci < 4; ci++) {
            const uint32_t c = gi * 4 + ci;
            const uint64_t s64 = load_u64(sp + (size_t)c * 8);
            const uint64_t ns64 = ~s64;
            const __mmask64 ks = (__mmask64)s64;
            const __m512i v8s = _mm512_maskz_mov_epi8(ks, veight);
            const uint64_t mb = mbit >> 3;
            const unsigned msh = (unsigned)mbit & 7;
            const unsigned kns = (unsigned)__builtin_popcountll(ns64);
            const __m512i qv =
                _mm512_load_si512((const void *)(qp + (size_t)c * 64));
            I8_ROW(0, MF);
            mbit += kns;
        }
        const float dxg = qx->dx[gi];
        const float c128g = qx->c128[gi];
        I8_FIXUP(0);
    }
    y[r] = _mm512_reduce_add_ps(ya0) - corr0;
}

void doml_gemv_i8_rows(const DomlGemvW *w, const DomlQx *qx, float *y,
                       uint32_t r0, uint32_t r1)
{
    uint32_t r = r0;
    if (w->m_full) {
        if (g_tile_i8 >= 4)
            for (; r + 4 <= r1; r += 4) i8_tile4_g(w, qx, y, r, 1);
        if (g_tile_i8 >= 2)
            for (; r + 2 <= r1; r += 2) i8_tile2_g(w, qx, y, r, 1);
        for (; r < r1; r++) i8_tile1_g(w, qx, y, r, 1);
    } else {
        if (g_tile_i8 >= 4)
            for (; r + 4 <= r1; r += 4) i8_tile4_g(w, qx, y, r, 0);
        if (g_tile_i8 >= 2)
            for (; r + 2 <= r1; r += 2) i8_tile2_g(w, qx, y, r, 0);
        for (; r < r1; r++) i8_tile1_g(w, qx, y, r, 0);
    }
}

/* ---------------------------------------------------------- thread pool --- */

#define POOL_MAX 48

#define POOL_ROUNDS 6 /* ceil(log2(POOL_MAX)) */

struct DomlPool {
    int nth;
    _Atomic int stop;
    void (*volatile fn)(void *, int, int);
    void *volatile arg;
    _Atomic uint64_t seq;
    _Atomic int done;
    /* dissemination barrier: in round k thread i signals (i+2^k)%n and
     * waits for (i-2^k)%n; per-(round,thread) epoch slots, 64B padded.
     * ~log2(n) cross-core hops instead of n serialized RMWs on one line. */
    _Atomic uint32_t dflag[POOL_ROUNDS][POOL_MAX * 16];
    uint32_t epoch[POOL_MAX * 16]; /* owner-written only, padded */
    pthread_t th[POOL_MAX];
};

static void pool_pin(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) {
        perror("pthread_setaffinity_np");
        exit(1);
    }
}

void doml_pool_barrier(DomlPool *p, int ith)
{
    const int n = p->nth;
    if (n == 1) return;
    uint32_t e = ++p->epoch[ith * 16];
    for (int k = 0, d = 1; d < n; k++, d <<= 1) {
        int to = ith + d;
        if (to >= n) to -= n;
        atomic_store_explicit(&p->dflag[k][to * 16], e, memory_order_release);
        while ((int32_t)(atomic_load_explicit(&p->dflag[k][ith * 16],
                                              memory_order_acquire) -
                         e) < 0)
            _mm_pause();
    }
}

typedef struct {
    DomlPool *p;
    int ith;
} WorkerArg;

static void *pool_worker(void *v)
{
    WorkerArg *wa = (WorkerArg *)v;
    DomlPool *p = wa->p;
    int ith = wa->ith;
    free(wa);
    pool_pin(ith); /* thread t -> CPU t: node0 = even, node1 = odd */
    uint64_t last = 0;
    for (;;) {
        while (atomic_load_explicit(&p->seq, memory_order_acquire) == last) {
            if (atomic_load_explicit(&p->stop, memory_order_acquire))
                return NULL;
            _mm_pause();
        }
        last = atomic_load_explicit(&p->seq, memory_order_acquire);
        p->fn(p->arg, ith, p->nth);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_acq_rel);
    }
}

DomlPool *doml_pool_create(int nth)
{
    if (nth < 1 || nth > POOL_MAX) {
        fprintf(stderr, "doml_pool: bad nth=%d\n", nth);
        exit(1);
    }
    DomlPool *p = (DomlPool *)calloc(1, sizeof(DomlPool));
    if (!p) exit(1);
    p->nth = nth;
    pool_pin(0); /* caller acts as thread 0 */
    for (int t = 1; t < nth; t++) {
        WorkerArg *wa = (WorkerArg *)malloc(sizeof(WorkerArg));
        wa->p = p;
        wa->ith = t;
        if (pthread_create(&p->th[t], NULL, pool_worker, wa)) {
            perror("pthread_create");
            exit(1);
        }
    }
    return p;
}

void doml_pool_run(DomlPool *p, void (*fn)(void *, int, int), void *arg)
{
    if (p->nth == 1) {
        fn(arg, 0, 1);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->seq, 1, memory_order_release);
    fn(arg, 0, p->nth);
    while (atomic_load_explicit(&p->done, memory_order_acquire) != p->nth - 1)
        _mm_pause();
}

void doml_pool_destroy(DomlPool *p)
{
    if (!p) return;
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    for (int t = 1; t < p->nth; t++) pthread_join(p->th[t], NULL);
    free(p);
}

int doml_pool_nth(const DomlPool *p) { return p->nth; }
