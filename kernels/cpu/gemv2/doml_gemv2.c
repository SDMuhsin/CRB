/* DOML GEMV v2 — fused row-slab resident + futex-fallback sync. See header.
 *
 * Inner loop per (4-row tile, 256-column group) block:
 *   hdr   : 4 x u8 b1 segment lengths -> section pointers (3 adds); the
 *           next-block pointer is available right after the header loads,
 *           ~one full chunk loop ahead of its first use.
 *   cb    : i8: table = ONE unaligned 16 B broadcast load per row (the
 *           stored record IS the vpermb table; slots 12..15 read into the
 *           next record and are never indexed). fp: 12 fp8 bytes ->
 *           vpermi2b lo/hi LUT, in-loop, per (row,group).
 *   chunks: per 64-column chunk, per row:
 *             m64  = pdep(next m bits, ~s64)   [packed]  or 8 B load [mf]
 *             nb64 = s64 + m64 ; pc = popcount(nb64)
 *             b164 = pdep(next b1 bits, nb64)  [bit pos local to the block]
 *             kb0  = 8 B mask load from the block's b0 section
 *             idx  = ternlog-OR3(b0,b1<<1,m<<2) | s<<3 ; vpermb ; vpdpbusd
 *           The m bit position is SHARED by the 4 rows (salience is
 *           column-wise) and tracked once per chunk.
 *
 * All bytes of a thread's tile range form one contiguous forward stream —
 * no per-row plane pointers, no b1 row-offset table, no prefetch restarts.
 *
 * Thread pool: dissemination barrier; every wait is bounded spin (plain
 * loads, then _mm_pause loop) with futex fallback, so oversubscription
 * (48t, co-tenant spikes) sleeps instead of death-spiraling.
 */
#include "doml_gemv2.h"
#include "../ref/ref_decode.h"

#include <immintrin.h>
#include <limits.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

/* ------------------------------------------------------------- tables ----- */

static uint16_t g2_fp8bf16[256];
static uint8_t g2_lut_lo[128] __attribute__((aligned(64)));
static uint8_t g2_lut_hi[128] __attribute__((aligned(64)));
static int g2_init_done = 0;
/* Spin budget before sleeping: must comfortably exceed one barrier RTT plus
 * one futex wake-to-run latency, so that after any sleep event the barrier
 * RE-CONVERGES to pure-spin steady state instead of cascading (a budget of
 * ~5 us measurably locks 12-48t barriers into a permanent sleep-skew-sleep
 * loop at 20-100x the spin-only cost). ~4096 plain loads (~2 us) + 2048
 * pauses (~100 us) keeps the futex path reserved for genuine stalls
 * (dispatch gaps, co-tenant preemption, oversubscription). */
static int g2_spin0 = 4096;          /* plain-load spins  (env DOML2_SPIN0) */
static int g2_spin1 = 2048;          /* _mm_pause spins   (env DOML2_SPIN)  */
static long g2_futex_ns = 50 * 1000; /* sleep-poll bound  (env DOML2_FUTEX_US) */
static int g2_radix = 2;             /* dissemination radix 2 or 4 (env DOML2_RADIX) */
static int g2_pin_mode = 0;          /* 0=cpu 1=node 2=none (env DOML2_PIN) */

void doml2_init(void)
{
    if (g2_init_done) return;
    dpka_fp8e4m3_to_bf16_table(g2_fp8bf16);
    for (int b = 0; b < 128; b++) {
        g2_lut_lo[b] = (uint8_t)(g2_fp8bf16[b] & 0xFF);
        g2_lut_hi[b] = (uint8_t)(g2_fp8bf16[b] >> 8);
    }
    const char *e = getenv("DOML2_SPIN0");
    if (e) g2_spin0 = atoi(e);
    e = getenv("DOML2_SPIN");
    if (e) g2_spin1 = atoi(e);
    e = getenv("DOML2_FUTEX_US");
    if (e) g2_futex_ns = atol(e) * 1000;
    e = getenv("DOML2_RADIX");
    if (e) g2_radix = atoi(e) == 4 ? 4 : 2;
    e = getenv("DOML2_PIN");
    if (e) {
        if (!strcmp(e, "node")) g2_pin_mode = 1;
        else if (!strcmp(e, "none")) g2_pin_mode = 2;
        else g2_pin_mode = 0;
    }
    g2_init_done = 1;
}

static inline uint64_t load_u64(const void *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* Next k bits of an LSB-first bit stream (identical contract to P2b):
 * consumer pdep uses only the low k bits; single unaligned load suffices
 * whenever sh + k <= 64; reads up to 9 bytes (slab tail slack covers it). */
static inline uint64_t take_bits(const uint8_t *p, unsigned sh, unsigned k)
{
    uint64_t lo = load_u64(p) >> sh;
    if (__builtin_expect(sh + k > 64, 0))
        lo |= ((uint64_t)p[8] << (63 ^ sh)) << 1;
    return lo;
}

/* round-to-nearest-even fp32 -> bf16 bit pattern */
static inline uint16_t bf16_rne(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static inline float bf16_f(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* i8 cb record from the 10 resident fp8 slots [bulk0,bulk1,tail0..3,sal0..3]:
 * container 12-slot order, u8 = q+128, then bf16 scale (RNE of max/127). */
void doml2_quant_cb10(const uint8_t cb10[10], uint8_t rec14[14])
{
    if (!g2_init_done) doml2_init();
    float lv[12];
    lv[0] = bf16_f(g2_fp8bf16[cb10[0]]);
    lv[1] = bf16_f(g2_fp8bf16[cb10[1]]);
    lv[2] = lv[1]; /* pad slots replicate bulk1 (never indexed) */
    lv[3] = lv[1];
    for (int k = 0; k < 8; k++) lv[4 + k] = bf16_f(g2_fp8bf16[cb10[2 + k]]);
    float maxa = 0.f;
    for (int k = 0; k < 12; k++) {
        float a = fabsf(lv[k]);
        if (a > maxa) maxa = a;
    }
    uint16_t sb = 0;
    if (maxa > 0.f) sb = bf16_rne(maxa / 127.0f);
    float s = bf16_f(sb);
    for (int k = 0; k < 12; k++) {
        long q = 0;
        if (s > 0.f) {
            q = lrintf(lv[k] / s);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
        }
        rec14[k] = (uint8_t)(q + 128);
    }
    rec14[12] = (uint8_t)(sb & 0xFF);
    rec14[13] = (uint8_t)(sb >> 8);
}

/* --------------------------------------------------------------- pack ----- */

#define CBREC(var) ((var) == DOML2_VAR_I8 ? 14u : 12u)
/* [H17-C] row bounce bounds follow max C = DOML2_NG_MAX * 256 (were 3072/8) */
#define MROW_MAX (DOML2_NG_MAX * 256 / 8 + 16)
#define B1ROW_MAX (DOML2_NG_MAX * 256 / 8 + 16)

static void die2(const char *msg)
{
    fprintf(stderr, "doml2: FATAL: %s\n", msg);
    abort();
}

/* copy row streams into padded stack buffers so take_bits' 9-byte reads
 * never leave the malloc'd R-B planes (packer-only; not on the hot path) */
static void bounce_row(uint8_t *dst, const uint8_t *src, size_t n)
{
    memcpy(dst, src, n);
    memset(dst + n, 0, 16);
}

/* non-bulk bit count of row r per group (drives b1 section sizes) */
static void row_group_nb(const DpkaResB *rb, uint32_t r,
                         uint16_t nbb[DOML2_NG_MAX])
{
    uint8_t mbuf[MROW_MAX];
    bounce_row(mbuf, rb->m + (size_t)r * rb->m_pitch, rb->m_pitch);
    uint64_t mbit = 0;
    for (uint32_t gi = 0; gi < rb->NG; gi++) {
        unsigned nb = 0;
        for (uint32_t ci = 0; ci < 4; ci++) {
            uint64_t s64 = load_u64(rb->s + (size_t)gi * 32 + ci * 8);
            uint64_t ns64 = ~s64;
            unsigned kns = (unsigned)__builtin_popcountll(ns64);
            uint64_t m64 = _pdep_u64(
                take_bits(mbuf + (mbit >> 3), (unsigned)mbit & 7, kns), ns64);
            nb += (unsigned)__builtin_popcountll(s64 + m64);
            mbit += kns;
        }
        nbb[gi] = (uint16_t)nb;
    }
}

static void geom_check(const DpkaResB *rb)
{
    if (rb->C % 256 != 0 || rb->g != 256 || rb->NG > DOML2_NG_MAX ||
        rb->R % DOML2_TILE != 0)
        die2("unsupported geometry (need C%256==0, g=256, NG<=24, R%4==0)");
}

size_t doml2_slab_bytes(const DpkaResB *rb, Doml2Var var, int m_full,
                        uint32_t *tileoff, Doml2Stats *st)
{
    geom_check(rb);
    const uint32_t NG = rb->NG, ntiles = rb->R / DOML2_TILE;
    const unsigned rec = CBREC(var);
    Doml2Stats z;
    memset(&z, 0, sizeof(z));
    /* per-group m section bytes (shared across rows: salience column-wise) */
    unsigned mlen[DOML2_NG_MAX];
    for (uint32_t gi = 0; gi < NG; gi++) {
        unsigned ns = 0;
        for (uint32_t ci = 0; ci < 4; ci++)
            ns += (unsigned)__builtin_popcountll(
                ~load_u64(rb->s + (size_t)gi * 32 + ci * 8));
        mlen[gi] = m_full ? 32u : (ns + 7) / 8;
        z.m_pad_bits += m_full ? 0 : (size_t)rb->R * (8 * mlen[gi] - ns);
    }
    uint16_t nbb[DOML2_NG_MAX];
    uint32_t off = 0;
    for (uint32_t t = 0; t < ntiles; t++) {
        tileoff[t] = off;
        uint32_t bsz[DOML2_NG_MAX];
        for (uint32_t gi = 0; gi < NG; gi++)
            bsz[gi] = 4u + 4u * rec + 128u + 4u * mlen[gi];
        for (uint32_t n = 0; n < DOML2_TILE; n++) {
            row_group_nb(rb, t * DOML2_TILE + n, nbb);
            for (uint32_t gi = 0; gi < NG; gi++) {
                unsigned b = ((unsigned)nbb[gi] + 7) / 8;
                bsz[gi] += b;
                z.b1 += b;
                z.b1_pad_bits += 8 * b - nbb[gi];
            }
        }
        for (uint32_t gi = 0; gi < NG; gi++) off += bsz[gi];
    }
    tileoff[ntiles] = off;
    z.b0 = (size_t)rb->R * rb->C / 8;
    z.cb = (size_t)rb->R * NG * rec;
    z.hdr = (size_t)ntiles * NG * 4;
    z.m = 0;
    for (uint32_t gi = 0; gi < NG; gi++) z.m += (size_t)rb->R * mlen[gi];
    z.s = rb->C / 8;
    z.tileoff = ((size_t)ntiles + 1) * 4;
    size_t s_sec = (z.s + 63) & ~(size_t)63; /* blocks start 64B-aligned */
    z.align_pad = s_sec - z.s;
    z.tail_pad = DOML2_TAIL_PAD;
    if (st) *st = z;
    return s_sec + off + DOML2_TAIL_PAD;
}

void doml2_pack_init(const DpkaResB *rb, Doml2Var var, int m_full,
                     uint8_t *slab, const uint32_t *tileoff, Doml2W *w)
{
    geom_check(rb);
    memset(w, 0, sizeof(*w));
    w->R = rb->R;
    w->C = rb->C;
    w->NG = rb->NG;
    w->g = rb->g;
    w->n_sal = rb->n_sal;
    w->ntiles = rb->R / DOML2_TILE;
    w->variant = (uint32_t)var;
    w->m_full = m_full ? 1u : 0u;
    w->s = slab;
    w->blocks = slab + (((size_t)rb->C / 8 + 63) & ~(size_t)63);
    w->tileoff = tileoff;
    for (uint32_t gi = 0; gi < rb->NG; gi++) {
        unsigned ns = 0;
        for (uint32_t ci = 0; ci < 4; ci++)
            ns += (unsigned)__builtin_popcountll(
                ~load_u64(rb->s + (size_t)gi * 32 + ci * 8));
        w->mlen[gi] = (uint8_t)(m_full ? 32u : (ns + 7) / 8);
    }
    memcpy((uint8_t *)(uintptr_t)w->s, rb->s, rb->C / 8);
    /* resident bytes consumed per call: s + blocks + shared tileoff + mlen */
    w->weight_bytes = (size_t)rb->C / 8 + tileoff[w->ntiles] +
                      ((size_t)w->ntiles + 1) * 4 + rb->NG;
    w->slab_bytes = (size_t)(w->blocks - w->s) + tileoff[w->ntiles] +
                    DOML2_TAIL_PAD;
}

/* LSB-first bit appender (packer only) */
typedef struct {
    uint8_t *p;
    uint64_t acc;
    unsigned n;
} BitApp;

static inline void app_put(BitApp *a, uint64_t v, unsigned k)
{
    if (k == 0) return;
    if (k < 64) v &= (1ULL << k) - 1;
    a->acc |= v << a->n;
    unsigned t = a->n + k;
    if (t >= 64) {
        memcpy(a->p, &a->acc, 8);
        a->p += 8;
        a->acc = a->n ? (v >> (64 - a->n)) : 0;
        t -= 64;
    }
    a->n = t;
}

static inline void app_flush(BitApp *a)
{
    while (a->n > 0) {
        *a->p++ = (uint8_t)a->acc;
        a->acc >>= 8;
        a->n = a->n >= 8 ? a->n - 8 : 0;
    }
}

void doml2_pack_tiles(const DpkaResB *rb, const Doml2W *w,
                      uint32_t t0, uint32_t t1)
{
    const uint32_t NG = rb->NG;
    const unsigned rec = CBREC((Doml2Var)w->variant);
    const size_t pitch = rb->C / 8;
    uint8_t *blocks = (uint8_t *)(uintptr_t)w->blocks;
    uint16_t nbb[DOML2_TILE][DOML2_NG_MAX];
    uint8_t mbuf[MROW_MAX], b1buf[B1ROW_MAX];

    for (uint32_t t = t0; t < t1; t++) {
        for (uint32_t n = 0; n < DOML2_TILE; n++)
            row_group_nb(rb, t * DOML2_TILE + n, nbb[n]);
        /* per-group block offsets within this tile */
        uint32_t boff[DOML2_NG_MAX + 1];
        boff[0] = w->tileoff[t];
        for (uint32_t gi = 0; gi < NG; gi++) {
            uint32_t sz = 4u + 4u * rec + 128u + 4u * (uint32_t)w->mlen[gi];
            for (uint32_t n = 0; n < DOML2_TILE; n++)
                sz += ((uint32_t)nbb[n][gi] + 7) / 8;
            boff[gi + 1] = boff[gi] + sz;
        }
        if (boff[NG] != w->tileoff[t + 1]) die2("tile size mismatch");
        /* headers + cb + b0 */
        for (uint32_t gi = 0; gi < NG; gi++) {
            uint8_t *bp = blocks + boff[gi];
            for (uint32_t n = 0; n < DOML2_TILE; n++)
                bp[n] = (uint8_t)(((uint32_t)nbb[n][gi] + 7) / 8);
            for (uint32_t n = 0; n < DOML2_TILE; n++) {
                const uint32_t r = t * DOML2_TILE + n;
                const uint8_t *c10 = rb->cb + ((size_t)r * NG + gi) * 10;
                uint8_t *dst = bp + 4 + (size_t)n * rec;
                if (w->variant == DOML2_VAR_I8) {
                    doml2_quant_cb10(c10, dst);
                } else {
                    dst[0] = c10[0];
                    dst[1] = c10[1];
                    dst[2] = c10[1];
                    dst[3] = c10[1];
                    memcpy(dst + 4, c10 + 2, 8);
                }
                memcpy(bp + 4 + 4 * rec + (size_t)n * 32,
                       rb->b0 + (size_t)r * pitch + (size_t)gi * 32, 32);
            }
        }
        /* m + b1 sections: walk each row once across all groups */
        for (uint32_t n = 0; n < DOML2_TILE; n++) {
            const uint32_t r = t * DOML2_TILE + n;
            bounce_row(mbuf, rb->m + (size_t)r * rb->m_pitch, rb->m_pitch);
            bounce_row(b1buf, rb->b1 + rb->b1_rowoff[r],
                       rb->b1_rowoff[r + 1] - rb->b1_rowoff[r]);
            uint64_t mbit = 0, b1bit = 0;
            for (uint32_t gi = 0; gi < NG; gi++) {
                uint8_t *bp = blocks + boff[gi];
                uint8_t *msec = bp + 4 + 4 * rec + 128;
                uint8_t *b1sec = msec + 4u * (uint32_t)w->mlen[gi];
                for (uint32_t k = 0; k < n; k++)
                    b1sec += ((uint32_t)nbb[k][gi] + 7) / 8;
                BitApp am = { msec + (size_t)n * w->mlen[gi], 0, 0 };
                BitApp ab = { b1sec, 0, 0 };
                for (uint32_t ci = 0; ci < 4; ci++) {
                    uint64_t s64 =
                        load_u64(rb->s + (size_t)gi * 32 + ci * 8);
                    uint64_t ns64 = ~s64;
                    unsigned kns = (unsigned)__builtin_popcountll(ns64);
                    uint64_t mv = take_bits(mbuf + (mbit >> 3),
                                            (unsigned)mbit & 7, kns);
                    if (!w->m_full) {
                        app_put(&am, mv, kns);
                    } else {
                        uint64_t m64 = _pdep_u64(mv, ns64);
                        app_put(&am, m64, 64);
                    }
                    uint64_t m64 = _pdep_u64(mv, ns64);
                    unsigned pc =
                        (unsigned)__builtin_popcountll(s64 + m64);
                    app_put(&ab, take_bits(b1buf + (b1bit >> 3),
                                           (unsigned)b1bit & 7, pc),
                            pc);
                    mbit += kns;
                    b1bit += pc;
                }
                app_flush(&am);
                app_flush(&ab);
                if (am.p != msec + (size_t)(n + 1) * w->mlen[gi])
                    die2("m section fill mismatch");
                if (ab.p != b1sec + ((uint32_t)nbb[n][gi] + 7) / 8)
                    die2("b1 section fill mismatch");
            }
        }
    }
}

void doml2_slice(uint32_t ntiles, int ith, int nth, uint32_t *t0, uint32_t *t1)
{
    uint32_t base = ntiles / (uint32_t)nth, rem = ntiles % (uint32_t)nth;
    uint32_t u = (uint32_t)ith;
    uint32_t lo = u * base + (u < rem ? u : rem);
    *t0 = lo;
    *t1 = lo + base + (u < rem ? 1u : 0u);
}

/* -------------------------------------------------------------- kernels --- */

#define BCAST16(p) \
    _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(p)))

/* one row's (m, b1, b0) -> idx byte vector for chunk ci of the current
 * block. Locals per row via token pasting (gcc 11 spills zmm state in the
 * equivalent loops — P2b measured 3-6x). MF and ci are compile-time
 * constants (the 4-chunk loop is fully unrolled so 32*n + 8*ci folds into
 * the addressing immediates and per-row pointers stay out of GPRs).
 * b1 walks via an absolute bit cursor bq = ptr*8 + bit (P2b trick: one live
 * register per row instead of pointer + offset).
 * idx build: maskz(b0,1) OR3 maskz(b1,2) OR3 mask_mov(v8s, m, 4) — the
 * m-term merges into the chunk-shared s-term (m AND s == 0 invariant),
 * saving the separate OR. */
#define IDX2_ROW(n, ci, MF)                                                   \
    uint64_t m64##n =                                                         \
        (MF) ? load_u64(ms + 32u * (n) + 8u * (ci))                           \
             : _pdep_u64(take_bits(msb + mo##n, msh, kns), ns64);             \
    uint64_t nb64##n = s64 + m64##n; /* disjoint: '+' == '|' */               \
    unsigned pc##n = (unsigned)__builtin_popcountll(nb64##n);                 \
    uint64_t b164##n = _pdep_u64(                                             \
        take_bits((const uint8_t *)(uintptr_t)(bq##n >> 3),                   \
                  (unsigned)bq##n & 7, pc##n),                                \
        nb64##n);                                                             \
    bq##n += pc##n;                                                           \
    __mmask64 kb0##n = _load_mask64(                                          \
        (__mmask64 *)(uintptr_t)(b0s + 32u * (n) + 8u * (ci)));               \
    __m512i idx##n = _mm512_ternarylogic_epi64(                               \
        _mm512_maskz_mov_epi8(kb0##n, vone),                                  \
        _mm512_maskz_mov_epi8((__mmask64)b164##n, vtwo),                      \
        _mm512_mask_mov_epi8(v8s, (__mmask64)m64##n, vfour), 0xFE)

#define COMMON_DECLS                                                          \
    const uint32_t NG = w->NG;                                                \
    const uint8_t *sp = w->s;                                                 \
    const __m512i vone = _mm512_set1_epi8(1);                                 \
    const __m512i vtwo = _mm512_set1_epi8(2);                                 \
    const __m512i vfour = _mm512_set1_epi8(4);                                \
    const __m512i veight = _mm512_set1_epi8(8)

/* section pointers of the current block + advance p to the next block */
#define BLOCK_SECTIONS(REC)                                                   \
    const unsigned l0 = p[0], l1 = p[1], l2 = p[2], l3 = p[3];                \
    const uint8_t *cbp = p + 4;                                               \
    const uint8_t *b0s = cbp + 4u * (REC);                                    \
    const unsigned mlen = MF ? 32u : w->mlen[gi];                             \
    const size_t mo0 = 0, mo1 = mlen, mo2 = 2u * mlen, mo3 = 3u * mlen;       \
    const uint8_t *ms = b0s + 128;                                            \
    const uint8_t *b1p0 = ms + 4u * mlen;                                     \
    p = b1p0 + l0 + l1 + l2 + l3;                                             \
    uint64_t bq0 = (uint64_t)(uintptr_t)b1p0 * 8;                             \
    uint64_t bq1 = bq0 + 8u * l0;                                             \
    uint64_t bq2 = bq1 + 8u * l1;                                             \
    uint64_t bq3 = bq2 + 8u * l2;                                             \
    unsigned mpos = 0;                                                        \
    /* pull the block after next into L2 while this one is processed (the   \
     * stream is forward but restarts at a fresh copy every call, so the    \
     * HW streamer's ramp is paid per call without this) */                  \
    _mm_prefetch((const char *)p + 384, _MM_HINT_T0);                         \
    _mm_prefetch((const char *)p + 448, _MM_HINT_T0);                         \
    _mm_prefetch((const char *)p + 512, _MM_HINT_T0);                         \
    _mm_prefetch((const char *)p + 576, _MM_HINT_T0);                         \
    _mm_prefetch((const char *)p + 640, _MM_HINT_T0);                         \
    (void)mo0; (void)mo1; (void)mo2; (void)mo3;                               \
    (void)bq0; (void)bq1; (void)bq2; (void)bq3; (void)mpos

#define CHUNK_SHARED(ci)                                                      \
    const uint64_t s64 = load_u64(sp + (size_t)gi * 32 + 8u * (ci));          \
    const uint64_t ns64 = ~s64;                                               \
    const unsigned kns = (unsigned)__builtin_popcountll(ns64);                \
    const __m512i v8s = _mm512_maskz_mov_epi8((__mmask64)s64, veight);        \
    const uint8_t *msb = ms + (mpos >> 3);                                    \
    const unsigned msh = (unsigned)mpos & 7;                                  \
    (void)msb; (void)msh

/* ---- I8 ---- */

#define I8_ROW(n, ci, MF)                                                     \
    do {                                                                      \
        IDX2_ROW(n, ci, MF);                                                  \
        __m512i wv = _mm512_permutexvar_epi8(idx##n, tb##n);                  \
        ia##n = _mm512_dpbusd_epi32(ia##n, wv, qv);                           \
    } while (0)

#define I8_CHUNK(ci, MF)                                                      \
    do {                                                                      \
        CHUNK_SHARED(ci);                                                     \
        const __m512i qv = _mm512_load_si512(                                 \
            (const void *)(qp + (size_t)gi * 256 + 64u * (ci)));              \
        I8_ROW(0, ci, MF);                                                    \
        I8_ROW(1, ci, MF);                                                    \
        I8_ROW(2, ci, MF);                                                    \
        I8_ROW(3, ci, MF);                                                    \
        mpos += kns;                                                          \
    } while (0)

static inline __attribute__((always_inline)) void
i8_tile_g(const Doml2W *w, const DomlQx *qx, float *y, uint32_t t,
          const int MF)
{
    COMMON_DECLS;
    const uint8_t *p = w->blocks + w->tileoff[t];
    const int8_t *qp = qx->q;
    __m512 ya0 = _mm512_setzero_ps(), ya1 = _mm512_setzero_ps();
    __m512 ya2 = _mm512_setzero_ps(), ya3 = _mm512_setzero_ps();
    float corr0 = 0.f, corr1 = 0.f, corr2 = 0.f, corr3 = 0.f;
    for (uint32_t gi = 0; gi < NG; gi++) {
        BLOCK_SECTIONS(14u);
        const __m512i tb0 = BCAST16(cbp + 0 * 14);
        const __m512i tb1 = BCAST16(cbp + 1 * 14);
        const __m512i tb2 = BCAST16(cbp + 2 * 14);
        const __m512i tb3 = BCAST16(cbp + 3 * 14);
        __m512i ia0 = _mm512_setzero_si512(), ia1 = _mm512_setzero_si512();
        __m512i ia2 = _mm512_setzero_si512(), ia3 = _mm512_setzero_si512();
        I8_CHUNK(0, MF);
        I8_CHUNK(1, MF);
        I8_CHUNK(2, MF);
        I8_CHUNK(3, MF);
        const float dxg = qx->dx[gi];
        const float c128g = qx->c128[gi];
        uint16_t sb;
#define I8_FIXUP(n)                                                           \
        do {                                                                  \
            memcpy(&sb, cbp + (n) * 14 + 12, 2);                              \
            const float sc = bf16_f(sb);                                      \
            corr##n += sc * c128g;                                            \
            ya##n = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia##n),                \
                                    _mm512_set1_ps(sc * dxg), ya##n);         \
        } while (0)
        I8_FIXUP(0);
        I8_FIXUP(1);
        I8_FIXUP(2);
        I8_FIXUP(3);
#undef I8_FIXUP
    }
    y[t * 4 + 0] = _mm512_reduce_add_ps(ya0) - corr0;
    y[t * 4 + 1] = _mm512_reduce_add_ps(ya1) - corr1;
    y[t * 4 + 2] = _mm512_reduce_add_ps(ya2) - corr2;
    y[t * 4 + 3] = _mm512_reduce_add_ps(ya3) - corr3;
}

/* [TP17] Split the two MF variants into separate NOINLINE functions.
 * Both were inlined into one body: objdump measured 1322 instructions for
 * doml2_gemv_i8_tiles, i.e. ~41 per (row,chunk) — and the loop is ISSUE-bound
 * (4.3 IPC of a 5-wide machine, TP17 §2b), so the dead variant's code is pure
 * front-end pressure (Ice Lake DSB is 4K uops; the hot loop should stay in it).
 * m_full is false for every shipped tensor, so MF=1 is cold code.
 * Pure code layout: same arithmetic, same order, outputs bitwise identical. */
static __attribute__((noinline)) void
i8_tiles_mf0(const Doml2W *w, const DomlQx *qx, float *y, uint32_t t0,
             uint32_t t1)
{
    for (uint32_t t = t0; t < t1; t++) i8_tile_g(w, qx, y, t, 0);
}

static __attribute__((noinline)) void
i8_tiles_mf1(const Doml2W *w, const DomlQx *qx, float *y, uint32_t t0,
             uint32_t t1)
{
    for (uint32_t t = t0; t < t1; t++) i8_tile_g(w, qx, y, t, 1);
}

void doml2_gemv_i8_tiles(const Doml2W *w, const DomlQx *qx, float *y,
                         uint32_t t0, uint32_t t1)
{
    if (w->m_full) i8_tiles_mf1(w, qx, y, t0, t1);
    else           i8_tiles_mf0(w, qx, y, t0, t1);
}

/* ---- FP ---- */

#define FP_ROW(n, ci, MF)                                                     \
    do {                                                                      \
        IDX2_ROW(n, ci, MF);                                                  \
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

/* fp8 rec -> broadcast bf16 lo/hi byte tables (broadcast load first, so the
 * LUT results stay lane-replicated) */
#define FP_TABLES(n)                                                          \
    __m512i raw##n = BCAST16(cbp + (n) * 12);                                 \
    __m512i i7##n = _mm512_and_si512(raw##n, m7f);                            \
    __m512i vl##n = _mm512_permutex2var_epi8(L0, i7##n, L1);                  \
    __m512i vh##n = _mm512_or_si512(_mm512_permutex2var_epi8(H0, i7##n, H1),  \
                                    _mm512_and_si512(raw##n, m80))

static inline __attribute__((always_inline)) void
fp_tile_g(const Doml2W *w, const float *xperm, float *y, uint32_t t,
          const int MF)
{
    COMMON_DECLS;
    const __m512i vhi16 = _mm512_set1_epi32((int)0xFFFF0000);
    const __m512i m7f = _mm512_set1_epi8(0x7F);
    const __m512i m80 = _mm512_set1_epi8((char)0x80);
    const __m512i L0 = _mm512_load_si512(g2_lut_lo);
    const __m512i L1 = _mm512_load_si512(g2_lut_lo + 64);
    const __m512i H0 = _mm512_load_si512(g2_lut_hi);
    const __m512i H1 = _mm512_load_si512(g2_lut_hi + 64);
    const uint8_t *p = w->blocks + w->tileoff[t];
    __m512 acA0 = _mm512_setzero_ps(), acB0 = _mm512_setzero_ps();
    __m512 acA1 = _mm512_setzero_ps(), acB1 = _mm512_setzero_ps();
    __m512 acA2 = _mm512_setzero_ps(), acB2 = _mm512_setzero_ps();
    __m512 acA3 = _mm512_setzero_ps(), acB3 = _mm512_setzero_ps();
#define FP_CHUNK(ci, MF)                                                      \
    do {                                                                      \
        CHUNK_SHARED(ci);                                                     \
        const float *xp = xperm + (size_t)gi * 256 + 64u * (ci);              \
        const __m512 x0 = _mm512_load_ps(xp);                                 \
        const __m512 x1 = _mm512_load_ps(xp + 16);                            \
        const __m512 x2 = _mm512_load_ps(xp + 32);                            \
        const __m512 x3 = _mm512_load_ps(xp + 48);                            \
        FP_ROW(0, ci, MF);                                                    \
        FP_ROW(1, ci, MF);                                                    \
        FP_ROW(2, ci, MF);                                                    \
        FP_ROW(3, ci, MF);                                                    \
        mpos += kns;                                                          \
    } while (0)
    for (uint32_t gi = 0; gi < NG; gi++) {
        BLOCK_SECTIONS(12u);
        FP_TABLES(0);
        FP_TABLES(1);
        FP_TABLES(2);
        FP_TABLES(3);
        FP_CHUNK(0, MF);
        FP_CHUNK(1, MF);
        FP_CHUNK(2, MF);
        FP_CHUNK(3, MF);
    }
#undef FP_CHUNK
    y[t * 4 + 0] = _mm512_reduce_add_ps(_mm512_add_ps(acA0, acB0));
    y[t * 4 + 1] = _mm512_reduce_add_ps(_mm512_add_ps(acA1, acB1));
    y[t * 4 + 2] = _mm512_reduce_add_ps(_mm512_add_ps(acA2, acB2));
    y[t * 4 + 3] = _mm512_reduce_add_ps(_mm512_add_ps(acA3, acB3));
}

void doml2_gemv_fp_tiles(const Doml2W *w, const float *xperm, float *y,
                         uint32_t t0, uint32_t t1)
{
    if (w->m_full)
        for (uint32_t t = t0; t < t1; t++) fp_tile_g(w, xperm, y, t, 1);
    else
        for (uint32_t t = t0; t < t1; t++) fp_tile_g(w, xperm, y, t, 0);
}

/* ---------------------------------------------------------- thread pool --- */

#define POOL2_MAX 64
#define POOL2_ROUNDS 6 /* ceil(log2(POOL2_MAX)) */

/* Wait protocol: the signal side is a PLAIN release store (identical cost to
 * the P2b pure-spin pool — no RMW, no wake syscall, so a late thread cannot
 * start a syscall storm). The wait side spins bounded (plain loads, then
 * _mm_pause), then sleeps in a TIMED futex wait: a signal that races the
 * sleep entry is caught by the futex value check (EAGAIN), a signal that
 * lands after costs at most one timeout (default 50 us, env DOML2_FUTEX_US).
 * The futex path only ever engages under real oversubscription (48t,
 * co-tenant preemption), where sleeping frees the core / HT sibling instead
 * of death-spiraling. Epoch compares are wraparound-safe. */
/* Slot word = (epoch << 1) | sleep-bit. Signal = xchg (clears the sleep bit,
 * observes it atomically) + FUTEX_WAKE only if a sleeper registered — the
 * hot path never syscalls. Wait = bounded spin, then register the sleep bit
 * and TIMED futex wait (the timeout is a lost-wake backstop; wakes normally
 * arrive in ~us, so after a sleep event the next rounds fall back inside the
 * spin budget and the barrier re-converges to pure spinning). Epoch compares
 * are wraparound-safe; the sleep bit makes the observed value at most
 * epoch*2+1, which still satisfies (>= epoch*2). */
static inline int slot_reached(_Atomic uint32_t *word, uint32_t tgt2)
{
    return (int32_t)(atomic_load_explicit(word, memory_order_acquire) -
                     tgt2) >= 0;
}

static void slot_wait(_Atomic uint32_t *word, uint32_t tgt2, int nwake)
{
    (void)nwake;
    for (int i = 0; i < g2_spin0; i++)
        if (slot_reached(word, tgt2)) return;
    for (;;) {
        for (int i = 0; i < g2_spin1; i++) {
            if (slot_reached(word, tgt2)) return;
            _mm_pause();
        }
        uint32_t old =
            atomic_fetch_or_explicit(word, 1u, memory_order_acq_rel);
        if ((int32_t)(old - tgt2) >= 0) return;
        struct timespec ts = { 0, g2_futex_ns };
        syscall(SYS_futex, (uint32_t *)word, FUTEX_WAIT_PRIVATE, old | 1u,
                &ts, NULL, 0);
        if (slot_reached(word, tgt2)) return;
    }
}

static inline void slot_signal(_Atomic uint32_t *word, uint32_t tgt2,
                               int nwake)
{
    /* fast path: no sleeper registered -> plain release store (no RMW, no
     * fence; measured ~1 us cheaper per 24t barrier than unconditional
     * xchg). A waiter registering between the load and the store loses the
     * wake, but its TIMED futex wait self-recovers (<= one poll period) and
     * the following rounds re-converge to spinning. */
    if (__builtin_expect(
            atomic_load_explicit(word, memory_order_relaxed) & 1u, 0)) {
        uint32_t old =
            atomic_exchange_explicit(word, tgt2, memory_order_acq_rel);
        if (old & 1u)
            syscall(SYS_futex, (uint32_t *)word, FUTEX_WAKE_PRIVATE, nwake,
                    NULL, NULL, 0);
    } else {
        atomic_store_explicit(word, tgt2, memory_order_release);
    }
}

#define POOL2_RADIX_MAX 4
#define POOL2_ROUNDS_MAX 6 /* ceil(log2(POOL2_MAX)) covers radix 2 and 4 */

struct Doml2Pool {
    int nth;
    _Atomic int stop;
    void (*volatile fn)(void *, int, int);
    void *volatile arg;
    _Atomic uint32_t seq; /* call number */
    uint32_t seq_epoch;
    _Atomic uint32_t dflag[POOL2_ROUNDS_MAX][POOL2_RADIX_MAX - 1][POOL2_MAX * 16];
    uint32_t epoch[POOL2_MAX * 16]; /* owner-written only, 64B padded */
    pthread_t th[POOL2_MAX];
};

/* DOML2_PIN=cpu  (default): thread t -> exactly CPU t.
 * DOML2_PIN=node: thread t -> ALL CPUs of its NUMA node (node0 = even CPUs,
 * node1 = odd). Same node assignment as cpu mode, so per-node first-touch
 * placement is unchanged; the scheduler is free to dodge co-tenant-occupied
 * cores inside the node. Rationale (P2d, PI-measured): with exact pinning a
 * co-tenant at ~16% busy on ONE pinned core gates every barrier on that
 * straggler (9.5 -> 11.3+ us at 24t, = 9.5/(1-0.16)); ik's unpinned omp
 * threads migrate around the hot core under the same load. */
static void pool2_pin(int t)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    if (g2_pin_mode == 2) return; /* none: scheduler-managed, like ik/omp */
    if (g2_pin_mode == 1) {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        if (ncpu > CPU_SETSIZE) ncpu = CPU_SETSIZE;
        for (long c = t & 1; c < ncpu; c += 2) CPU_SET(c, &set);
    } else {
        CPU_SET(t, &set);
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) {
        perror("pthread_setaffinity_np");
        exit(1);
    }
}

/* Radix-4 dissemination: round k, thread i signals i + j*4^k (j = 1..3) and
 * waits on its 3 (round, j) slots — ceil(log4 n) rounds (24t: 3 rounds vs 5
 * binary; measured ~2x cheaper). Thread t is pinned to CPU t (node = t & 1),
 * so even distances stay on-socket; the j-loop signals all partners before
 * any wait, letting the cross-socket hops overlap. Monotone epoch compares
 * make signal reordering across barriers safe (a fast thread may re-signal a
 * slot for epoch e+1 before the waiter consumed e). */
void doml2_pool_barrier(Doml2Pool *p, int ith)
{
    const int n = p->nth;
    if (n == 1) return;
    const int r = g2_radix;
    /* socket-aware role permutation: thread t is pinned to CPU t and node =
     * t & 1, so with identity roles every ODD dissemination distance is a
     * cross-socket hop. Roles group same-node threads contiguously (node0 =
     * roles [0, half), node1 = [half, n)), keeping small distances on-node;
     * slots are indexed by ROLE so no inverse map is needed. */
    const int half = (n + 1) >> 1;
    const int ri = (ith >> 1) + ((ith & 1) ? half : 0);
    uint32_t e = ++p->epoch[ith * 16];
    for (int k = 0, d = 1; d < n; k++, d *= r) {
        int nj = 0;
        for (int j = 1; j < r && j * d < n; j++) {
            int to = ri + j * d;
            if (to >= n) to -= n;
            slot_signal(&p->dflag[k][j - 1][to * 16], e << 1, 1);
            nj = j;
        }
        for (int j = 1; j <= nj; j++)
            slot_wait(&p->dflag[k][j - 1][ri * 16], e << 1, 1);
    }
}

typedef struct {
    Doml2Pool *p;
    int ith;
} Worker2Arg;

static void *pool2_worker(void *v)
{
    Worker2Arg *wa = (Worker2Arg *)v;
    Doml2Pool *p = wa->p;
    int ith = wa->ith;
    free(wa);
    pool2_pin(ith); /* thread t -> CPU t: node0 = even, node1 = odd */
    uint32_t last = 0;
    for (;;) {
        slot_wait(&p->seq, (last + 1) << 1, INT_MAX);
        if (atomic_load_explicit(&p->stop, memory_order_acquire)) return NULL;
        last++;
        p->fn(p->arg, ith, p->nth);
        doml2_pool_barrier(p, ith);
    }
}

Doml2Pool *doml2_pool_create(int nth)
{
    if (nth < 1 || nth > POOL2_MAX) {
        fprintf(stderr, "doml2_pool: bad nth=%d\n", nth);
        exit(1);
    }
    doml2_init();
    Doml2Pool *p = (Doml2Pool *)calloc(1, sizeof(Doml2Pool));
    if (!p) exit(1);
    p->nth = nth;
    pool2_pin(0); /* caller acts as thread 0 */
    for (int t = 1; t < nth; t++) {
        Worker2Arg *wa = (Worker2Arg *)malloc(sizeof(Worker2Arg));
        wa->p = p;
        wa->ith = t;
        if (pthread_create(&p->th[t], NULL, pool2_worker, wa)) {
            perror("pthread_create");
            exit(1);
        }
    }
    return p;
}

void doml2_pool_run(Doml2Pool *p, void (*fn)(void *, int, int), void *arg)
{
    if (p->nth == 1) {
        fn(arg, 0, 1);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    p->seq_epoch++;
    slot_signal(&p->seq, p->seq_epoch << 1, INT_MAX);
    fn(arg, 0, p->nth);
    doml2_pool_barrier(p, 0);
}

void doml2_pool_destroy(Doml2Pool *p)
{
    if (!p) return;
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    p->seq_epoch++;
    slot_signal(&p->seq, p->seq_epoch << 1, INT_MAX);
    for (int t = 1; t < p->nth; t++) pthread_join(p->th[t], NULL);
    free(p);
}

int doml2_pool_nth(const Doml2Pool *p) { return p->nth; }

/* ------------------------------------------------- work distribution ------ */

void doml2_steal_init(Doml2Steal *ws, uint32_t ntiles, int nth)
{
    for (int par = 0; par < 2; par++)
        for (int t = 0; t < nth; t++) {
            uint32_t t0, t1;
            doml2_slice(ntiles, t, nth, &t0, &t1);
            atomic_store_explicit(&ws->cur[par][t * 16], t0,
                                  memory_order_relaxed);
        }
}

static inline void steal_run(const Doml2W *w, const float *xperm,
                             const DomlQx *qx, float *y, uint32_t a,
                             uint32_t b)
{
    if (w->variant == DOML2_VAR_I8)
        doml2_gemv_i8_tiles(w, qx, y, a, b);
    else
        doml2_gemv_fp_tiles(w, xperm, y, a, b);
}

void doml2_steal_gemv(Doml2Steal *ws, int parity, int chunk, const Doml2W *w,
                      const float *xperm, const DomlQx *qx, float *y,
                      int ith, int nth)
{
    const uint32_t ntiles = w->ntiles;
    if (chunk < 1) chunk = 1;
    /* victim order: self (d=0), then even offsets (same NUMA parity when
     * nth is even: thread t -> CPU t, node = t & 1), then odd offsets */
    for (int pass = 0; pass < 2; pass++) {
        for (int d = pass; d < nth; d += 2) {
            int v = ith + d;
            if (v >= nth) v -= nth;
            uint32_t v0, v1;
            doml2_slice(ntiles, v, nth, &v0, &v1);
            (void)v0;
            _Atomic uint32_t *cur = &ws->cur[parity][v * 16];
            for (;;) {
                uint32_t c = atomic_fetch_add_explicit(
                    cur, (uint32_t)chunk, memory_order_relaxed);
                if (c >= v1) break;
                uint32_t e =
                    c + (uint32_t)chunk < v1 ? c + (uint32_t)chunk : v1;
                steal_run(w, xperm, qx, y, c, e);
            }
        }
    }
    /* re-arm the OTHER parity for the next call BEFORE the caller's
     * completion barrier — race-free double buffering */
    uint32_t t0, t1;
    doml2_slice(ntiles, ith, nth, &t0, &t1);
    (void)t1;
    atomic_store_explicit(&ws->cur[parity ^ 1][ith * 16], t0,
                          memory_order_release);
}
