/* DPKA v1 loader: mmap the artifact, rANS-decode the m plane, and build the
 * resident layouts R-A (raw container planes) and R-B (packed).
 *
 * Correctness code: every structural expectation is checked with die()-style
 * asserts that abort loudly. No intrinsics; plain C11 + libc.
 */
#include "dpka.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void die(const char *msg, const char *ctx)
{
    fprintf(stderr, "dpka: FATAL: %s (%s)\n", msg, ctx ? ctx : "-");
    abort();
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory", NULL);
    return p;
}

static void *xcalloc(size_t n)
{
    void *p = calloc(1, n ? n : 1);
    if (!p) die("out of memory", NULL);
    return p;
}

static inline uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t le64(const uint8_t *p)
{
    return (uint64_t)le32(p) | ((uint64_t)le32(p + 4) << 32);
}

static inline int bit_get(const uint8_t *bytes, size_t j)
{
    return (bytes[j >> 3] >> (j & 7)) & 1;
}

static inline void bit_set(uint8_t *bytes, size_t j)
{
    bytes[j >> 3] |= (uint8_t)(1u << (j & 7));
}

/* ---------------------------------------------------------------- open ---- */

DpkaFile *dpka_open(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open artifact", path);
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat failed", path);
    size_t fsize = (size_t)st.st_size;
    if (fsize < DPKA_TOC_OFF) die("file too small", path);
    void *base = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) die("mmap failed", path);
    const uint8_t *b = (const uint8_t *)base;

    if (memcmp(b, DPKA_MAGIC, 8) != 0) die("bad magic", path);
    uint32_t version = le32(b + 0x08);
    if (version != DPKA_VERSION) die("unsupported version", path);
    uint32_t n_tensors = le32(b + 0x0C);
    uint64_t total_weights = le64(b + 0x10);
    uint64_t file_size = le64(b + 0x18);
    uint64_t toc_off = le64(b + 0x20);
    if (file_size != fsize) die("header file_size mismatch", path);
    if (toc_off != DPKA_TOC_OFF) die("unexpected toc_off", path);
    if (toc_off + (uint64_t)n_tensors * DPKA_REC_SIZE > fsize)
        die("TOC out of bounds", path);
    _Static_assert(sizeof(DpkaTensorRec) == DPKA_REC_SIZE,
                   "DpkaTensorRec must be 256 bytes");

    DpkaFile *f = xmalloc(sizeof(DpkaFile));
    f->fd = fd;
    f->base = b;
    f->fsize = fsize;
    f->n_tensors = n_tensors;
    f->total_weights = total_weights;
    f->toc = (const DpkaTensorRec *)(b + toc_off);

    /* validate every record's bounds once */
    for (uint32_t i = 0; i < n_tensors; i++) {
        const DpkaTensorRec *r = &f->toc[i];
        if (r->name[127] != '\0') die("unterminated tensor name", path);
        if (r->C != r->C_orig) die("pad columns unsupported", r->name);
        if (r->g == 0 || r->C % r->g != 0) die("short group unsupported", r->name);
        if (r->NG != r->C / r->g) die("NG mismatch", r->name);
        if (r->C % 32 != 0) die("C not multiple of 32", r->name);
        if (r->f1 < 1 || r->f1 >= DPKA_PROB_M) die("bad rANS f1", r->name);
        if (r->n_m_bits != (uint64_t)r->R * (r->C_orig - r->n_sal))
            die("n_m_bits mismatch", r->name);
        for (int p = 0; p < DPKA_NPLANES; p++)
            if (r->off[p] + r->size[p] > fsize)
                die("plane out of bounds", r->name);
        if (r->size[DPKA_PL_B0] != (uint64_t)r->R * r->C / 8)
            die("b0 size mismatch", r->name);
        if (r->size[DPKA_PL_S] != r->C / 8) die("s size mismatch", r->name);
        if (r->size[DPKA_PL_CB] != (uint64_t)r->R * r->NG * 10)
            die("cb size mismatch", r->name);
    }
    return f;
}

void dpka_close(DpkaFile *f)
{
    if (!f) return;
    munmap((void *)f->base, f->fsize);
    close(f->fd);
    free(f);
}

int dpka_find(const DpkaFile *f, const char *name)
{
    for (uint32_t i = 0; i < f->n_tensors; i++)
        if (strncmp(f->toc[i].name, name, 128) == 0) return (int)i;
    return -1;
}

/* ------------------------------------------------- rANS m-plane decode ---- */
/* Mirror of rans.py decode_bits (see its docstring for the normative spec). */

typedef struct {
    uint32_t x;
    const uint8_t *p, *end;
    uint32_t f0, f1;
    const char *ctx;
} RansDec;

static void rans_init(RansDec *d, const uint8_t *stream, size_t n,
                      uint32_t f1, const char *ctx)
{
    if (n < 4) die("rANS stream too short", ctx);
    d->x = le32(stream);
    d->p = stream + 4;
    d->end = stream + n;
    d->f1 = f1;
    d->f0 = DPKA_PROB_M - f1;
    d->ctx = ctx;
}

static inline int rans_get_bit(RansDec *d)
{
    uint32_t slot = d->x & (DPKA_PROB_M - 1);
    int bit = slot >= d->f0;
    d->x = bit ? d->f1 * (d->x >> DPKA_PROB_BITS) + slot - d->f0
               : d->f0 * (d->x >> DPKA_PROB_BITS) + slot;
    while (d->x < DPKA_RANS_L) {
        if (d->p >= d->end) die("rANS stream underrun", d->ctx);
        d->x = (d->x << 8) | *d->p++;
    }
    return bit;
}

static void rans_finish(const RansDec *d)
{
    if (d->x != DPKA_RANS_L || d->p != d->end)
        die("rANS termination invariant violated", d->ctx);
}

uint8_t *dpka_decode_m_plane(const DpkaFile *f, int idx)
{
    const DpkaTensorRec *r = &f->toc[idx];
    const uint32_t R = r->R, C = r->C;
    const size_t pitch = C / 8;
    uint8_t *m = xcalloc((size_t)R * pitch);
    const uint8_t *s = f->base + r->off[DPKA_PL_S];

    RansDec d;
    rans_init(&d, f->base + r->off[DPKA_PL_M], r->size[DPKA_PL_M],
              r->f1, r->name);
    uint64_t n_dec = 0;
    for (uint32_t row = 0; row < R; row++) {
        uint8_t *mrow = m + (size_t)row * pitch;
        for (uint32_t j = 0; j < C; j++) {
            if (bit_get(s, j)) continue;        /* salient: m forced 0 */
            if (rans_get_bit(&d)) bit_set(mrow, j);
            n_dec++;
        }
    }
    if (n_dec != r->n_m_bits) die("decoded m bit count mismatch", r->name);
    rans_finish(&d);
    return m;
}

/* --------------------------------------------------------------- R-A ------ */

DpkaResA *dpka_build_ra(const DpkaFile *f, int idx)
{
    const DpkaTensorRec *r = &f->toc[idx];
    const uint32_t R = r->R, C = r->C, NG = r->NG;
    const size_t pitch = C / 8;

    DpkaResA *a = xmalloc(sizeof(DpkaResA));
    a->R = R; a->C = C; a->C_orig = r->C_orig; a->g = r->g; a->NG = NG;
    a->bytes_b0 = a->bytes_b1 = a->bytes_m = (size_t)R * pitch;
    a->bytes_s = pitch;
    a->bytes_cb = (size_t)R * NG * 12;

    a->b0 = xmalloc(a->bytes_b0);
    memcpy(a->b0, f->base + r->off[DPKA_PL_B0], a->bytes_b0);
    a->s = xmalloc(a->bytes_s);
    memcpy(a->s, f->base + r->off[DPKA_PL_S], a->bytes_s);

    a->m = dpka_decode_m_plane(f, idx);

    /* b1: expand row-byte-aligned non-bulk segments to a full plane.
     * non-bulk(row, j) = s[j] | m[row][j]; bulk positions stay 0. */
    a->b1 = xcalloc(a->bytes_b1);
    const uint8_t *src = f->base + r->off[DPKA_PL_B1];
    size_t src_off = 0;
    for (uint32_t row = 0; row < R; row++) {
        const uint8_t *mrow = a->m + (size_t)row * pitch;
        uint8_t *dst = a->b1 + (size_t)row * pitch;
        const uint8_t *seg = src + src_off;
        size_t bitpos = 0;
        for (uint32_t j = 0; j < C; j++) {
            if (!(bit_get(a->s, j) || bit_get(mrow, j))) continue;
            if (bit_get(seg, bitpos)) bit_set(dst, j);
            bitpos++;
        }
        src_off += (bitpos + 7) / 8;
    }
    if (src_off != r->size[DPKA_PL_B1])
        die("b1 payload size mismatch on expand", r->name);

    /* cb: 10 real slots -> 12-slot container tables; bulk pads replicate
     * slot 1 (spec §2.3 pad-replication contract, asserted at export). */
    a->cb = xmalloc(a->bytes_cb);
    const uint8_t *c10 = f->base + r->off[DPKA_PL_CB];
    for (size_t rg = 0; rg < (size_t)R * NG; rg++) {
        const uint8_t *in = c10 + rg * 10;
        uint8_t *out = a->cb + rg * 12;
        out[0] = in[0]; out[1] = in[1]; out[2] = in[1]; out[3] = in[1];
        memcpy(out + 4, in + 2, 8);            /* tail0..3, sal0..3 */
    }
    return a;
}

void dpka_free_ra(DpkaResA *a)
{
    if (!a) return;
    free(a->b0); free(a->b1); free(a->m); free(a->s); free(a->cb);
    free(a);
}

/* --------------------------------------------------------------- R-B ------ */

DpkaResB *dpka_build_rb(const DpkaFile *f, int idx)
{
    const DpkaTensorRec *r = &f->toc[idx];
    const uint32_t R = r->R, C = r->C, NG = r->NG, g = r->g;
    const size_t pitch = C / 8;

    DpkaResB *b = xmalloc(sizeof(DpkaResB));
    b->R = R; b->C = C; b->C_orig = r->C_orig; b->g = g; b->NG = NG;
    b->n_sal = r->n_sal;

    b->bytes_b0 = (size_t)R * pitch;
    b->b0 = xmalloc(b->bytes_b0);
    memcpy(b->b0, f->base + r->off[DPKA_PL_B0], b->bytes_b0);

    b->bytes_s = pitch;
    b->s = xmalloc(b->bytes_s);
    memcpy(b->s, f->base + r->off[DPKA_PL_S], b->bytes_s);

    b->bytes_cb = (size_t)R * NG * 10;
    b->cb = xmalloc(b->bytes_cb);
    memcpy(b->cb, f->base + r->off[DPKA_PL_CB], b->bytes_cb);

    /* full m plane (temporary) drives packing + offset derivation */
    uint8_t *mfull = dpka_decode_m_plane(f, idx);

    /* b1 payload: artifact layout == resident layout (row-byte-aligned) */
    b->bytes_b1 = r->size[DPKA_PL_B1];
    b->b1 = xmalloc(b->bytes_b1);
    memcpy(b->b1, f->base + r->off[DPKA_PL_B1], b->bytes_b1);

    /* per-row byte offsets from popcounts of the non-bulk mask (s | m) */
    b->bytes_b1off = ((size_t)R + 1) * sizeof(uint32_t);
    b->b1_rowoff = xmalloc(b->bytes_b1off);
    uint32_t off = 0;
    for (uint32_t row = 0; row < R; row++) {
        b->b1_rowoff[row] = off;
        const uint8_t *mrow = mfull + (size_t)row * pitch;
        uint32_t nb = 0;
        for (size_t byte = 0; byte < pitch; byte++)
            nb += (uint32_t)__builtin_popcount(
                (unsigned)(mrow[byte] | b->s[byte]));
        off += (nb + 7) / 8;
    }
    b->b1_rowoff[R] = off;
    if (off != b->bytes_b1) die("b1 row offsets != payload size", r->name);

    /* m: non-salient bits packed per row, row-byte-aligned */
    uint32_t c_ns = r->C_orig - r->n_sal;
    b->m_pitch = (c_ns + 7) / 8;
    b->bytes_m = (size_t)R * b->m_pitch;
    b->m = xcalloc(b->bytes_m);
    for (uint32_t row = 0; row < R; row++) {
        const uint8_t *mrow = mfull + (size_t)row * pitch;
        uint8_t *dst = b->m + (size_t)row * b->m_pitch;
        size_t bitpos = 0;
        for (uint32_t j = 0; j < C; j++) {
            if (bit_get(b->s, j)) continue;
            if (bit_get(mrow, j)) bit_set(dst, bitpos);
            bitpos++;
        }
        if (bitpos != c_ns) die("non-salient column count mismatch", r->name);
    }
    free(mfull);

    /* shared per-group non-salient prefix (m expansion / random access) */
    b->bytes_aux = ((size_t)NG + 1) * sizeof(uint32_t);
    b->ns_prefix = xmalloc(b->bytes_aux);
    uint32_t ns = 0;
    b->ns_prefix[0] = 0;
    for (uint32_t j = 0; j < C; j++) {
        if (!bit_get(b->s, j)) ns++;
        if ((j + 1) % g == 0) b->ns_prefix[(j + 1) / g] = ns;
    }
    return b;
}

void dpka_free_rb(DpkaResB *b)
{
    if (!b) return;
    free(b->b0); free(b->b1); free(b->b1_rowoff); free(b->m);
    free(b->s); free(b->cb); free(b->ns_prefix);
    free(b);
}
