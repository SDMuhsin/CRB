/* DOML GEMV decode microkernel (P2b) — consumes the R-B resident planes
 * (b0 / packed b1 / packed m / s / 10-slot fp8 cb) DIRECTLY.
 *
 * Two value paths (P2B_DESIGN_BRIEF.md):
 *   FP : container-exact bf16 levels widened to fp32, fp32 FMA accumulation.
 *        Only accumulation order differs from the reference decode.
 *   I8 : per-(row,group) on-the-fly quantization of the <=10 fp8 levels to
 *        u8 (=q+128) + fp32 scale; activations pre-quantized to int8 per
 *        256-column group; vpdpbusd accumulation with per-(row,group) fixup
 *        y += scale_rg*dx_g*(acc - 128*sum_qx_g).
 *
 * Bit addressing everywhere is LSB-first (DPK spec 2.1):
 *   bit(bytes, j) = (bytes[j>>3] >> (j&7)) & 1.
 *
 * Decode semantics (DPK spec 3):
 *   part = s[j] ? 2 : m[i][j];  code = b0 + 2*b1 (b1 stored non-bulk only);
 *   W[i,j] = cb[i][j>>8][part][code]   (10-slot base {0,2,6} per partition).
 *
 * The kernel index byte is idx = b0 | b1<<1 | m<<2 | s<<3 in [0,12)
 * (m AND s == 0 is a verified container invariant), looked up in a 16-slot
 * per-(row,group) table built on the fly from the 10 resident fp8 bytes:
 *   T[0..1] = bulk0..1, T[4..7] = tail0..3, T[8..11] = sal0..3, rest unused.
 */
#ifndef DOML_GEMV_H
#define DOML_GEMV_H

#include <stddef.h>
#include <stdint.h>

#include "../fmt/dpka.h"

/* ------------------------------------------------------------- weights ---- */
/* Kernel view of one tensor's R-B planes. Pointers normally point into a
 * packed slab (doml_gemv_pack_*) so that (a) every plane has tail slack for
 * the 16-byte unaligned bit-stream loads and (b) each thread's row slice can
 * be first-touched on its own NUMA node. Byte layouts are IDENTICAL to
 * DpkaResB (P1) — the kernel consumes the R-B format, not a re-encoding. */
typedef struct {
    uint32_t R, C, NG, g, n_sal;
    uint32_t m_pitch;          /* ceil((C - n_sal)/8) bytes per row; C/8 in
                                  m_full mode                                */
    uint32_t m_full;           /* 0: packed non-salient m bits (pure R-B);
                                  1: m expanded at pack time to the full
                                  R*C/8 bit plane (container-identical to
                                  the R-A m plane) — kills the per-chunk
                                  m bit-stream walk at +0.213 bpw streamed  */
    const uint8_t  *b0;        /* R * C/8 full bit plane                     */
    const uint8_t  *b1;        /* packed non-bulk bits, row-byte-aligned     */
    const uint32_t *b1_rowoff; /* R+1 byte offsets into b1                   */
    const uint8_t  *m;         /* R * m_pitch membership bits                */
    const uint8_t  *s;         /* C/8 salient column bitmap                  */
    const uint8_t  *cb;        /* R * NG * 10 fp8 [bulk0,bulk1,t0..3,s0..3]  */
    size_t weight_bytes;       /* exact resident bytes (BW accounting)       */
} DomlGemvW;

/* Slab: one contiguous allocation holding all planes + alignment + slack.
 * doml_gemv_slab_bytes() returns the size to allocate; doml_gemv_pack_init()
 * fills *w with pointers into slab and copies the SHARED small sections
 * (s, b1_rowoff); doml_gemv_pack_rows() copies rows [r0,r1) of b0/m/cb and
 * the b1 byte range — call it FROM THE PINNED THREAD that will later read
 * those rows so first-touch places them node-local. */
size_t doml_gemv_slab_bytes(const DpkaResB *rb, int m_full);
void   doml_gemv_pack_init(const DpkaResB *rb, uint8_t *slab, DomlGemvW *w,
                           int m_full);
void   doml_gemv_pack_rows(const DpkaResB *rb, const DomlGemvW *w,
                           uint32_t r0, uint32_t r1);

/* Contiguous row slice of thread ith among nth (same split used for
 * first-touch and for compute — keep them consistent). */
void doml_gemv_slice(uint32_t R, int ith, int nth, uint32_t *r0, uint32_t *r1);

/* ---------------------------------------------------------- activations --- */
/* FP path: x permuted once per call into the kernel's unpack order
 * (shared by every row/tile). xperm must hold C floats, 64B-aligned. */
void doml_gemv_prep_x_fp(const float *x, uint32_t C, float *xperm);

/* I8 path: x quantized once per call to int8 per 256-column group. */
typedef struct {
    int8_t *q;        /* C int8, 64B-aligned                                */
    float  *dx;       /* NG group scales: x ~ dx[g]*q                       */
    float  *c128;     /* NG: 128.0f * dx[g] * sum(q in group)  (fixup term) */
    uint32_t C, NG;
} DomlQx;
void doml_gemv_prep_x_i8(const float *x, uint32_t C, DomlQx *qx);

/* --------------------------------------------------------------- kernels -- */
void doml_gemv_init(void);   /* build fp8->bf16 LUT once (idempotent) */

/* Per-row on-the-fly 16-slot table prep (exposed for the numeric gates:
 * G-NUM-I8's level-rounding breakdown must use the kernel's own levels).
 * tlo/thi/ut are [NG][16]; sw is [NG]. */
void doml_gemv_prep_row_fp(const DomlGemvW *w, uint32_t r,
                           uint8_t (*tlo)[16], uint8_t (*thi)[16]);
void doml_gemv_prep_row_i8(const DomlGemvW *w, uint32_t r,
                           uint8_t (*ut)[16], float *sw);

void doml_gemv_fp_rows(const DomlGemvW *w, const float *xperm, float *y,
                       uint32_t r0, uint32_t r1);
void doml_gemv_i8_rows(const DomlGemvW *w, const DomlQx *qx, float *y,
                       uint32_t r0, uint32_t r1);

/* ------------------------------------------------------------ thread pool - */
/* Minimal pinned pool: thread t runs on CPU t (node0 = even CPUs, node1 =
 * odd — the roofline/bench_ik convention). Caller participates as ith=0. */
typedef struct DomlPool DomlPool;
DomlPool *doml_pool_create(int nth);
void      doml_pool_run(DomlPool *p, void (*fn)(void *, int, int), void *arg);
void      doml_pool_barrier(DomlPool *p, int ith); /* usable inside fn */
void      doml_pool_destroy(DomlPool *p);
int       doml_pool_nth(const DomlPool *p);

#endif /* DOML_GEMV_H */
