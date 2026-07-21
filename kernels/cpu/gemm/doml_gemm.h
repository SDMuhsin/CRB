/* DOML prefill GEMM (P3) — slab-convert + VNNI, consuming the FROZEN v2 i8
 * fused row-slab (kernels/cpu/gemv2/, 2.6535 bpw resident) UNCHANGED.
 *
 * Architecture (P3_DESIGN_BRIEF.md): ik's own prefill duality — decode reads
 * the packed format directly (P2c GEMV), prefill converts each thread's row
 * slice into TRANSIENT int8 panels and runs a dense u8s8 vpdpbusd GEMM — but
 * our conversion is rounding-free: the v2 i8 slab already stores the int8
 * levels (u8 = q+128, pack-time, gate-checked), so the panel bytes are
 * BITWISE the slab's cb-record levels selected by the decoded idx stream
 * (G-DERIVE-P3). No second weight rounding (ik re-rounds IQ2_KL -> Q8_K_R16).
 *
 * PANEL LAYOUT (transient working memory, never resident):
 *   For a strip of S rows (S % 16 == 0) starting at row-group rg0, K weights:
 *     pu8[((rg - rg0) * (K/4) + s) * 64 + 4*rl + o]
 *   holds the u8 level (q+128) of row rg*16 + rl (rl in [0,16)), weight
 *     wj(s, o) = g*256 + ci*64 + (4*(t%4) + t/4)*4 + o
 *   where g = s/64, ci = (s%64)/16, t = s%16, o in [0,4).  I.e. one 64-B
 *   line = 16 rows x 4 consecutive weights (Q8_K_R16-style interleave), and
 *   within each 16-line chunk the k-step order is the 4x4 dword transpose
 *   t -> 4*(t%4) + t/4 (falls out of the converter's unpack transpose).
 *   Scales: sc[g * srows_cap + (r - rg0*16)] = fp32 of the bf16 scale in the
 *   slab's cb record for (row r, group g).
 *
 * ACTIVATIONS: int8 per-256-group (same scheme/class as the decode i8 path):
 *   id = 127/amax_g, q = RNE(x*id) in [-127,127], dx = amax_g/127,
 *   b2 = 128*dx*bsum_g.  Stored with the SAME per-16-dword permutation as
 *   the panel (q[y*K + s*4 + o] = quant of x[y][wj(s,o)]) — which is exactly
 *   the natural output order of packs_epi32+packs_epi16, so the quantizer
 *   needs no extra shuffle and the GEMM streams both sides linearly.
 *
 * GEMM: C[y][r] += sum_g sc_rg * dx_yg * (Idot_rg,y - 128*bsum_yg) with
 *   Idot accumulated by vpdpbusd (u8 weights x s8 activations), integer-
 *   exact per group.  Micro-kernels: 4 row-groups x 4 columns (mk=1,
 *   flagship by SMOKE bake-off) and 2 x 8 (mk=0, alternate); outputs are
 *   bitwise identical between micro-kernels and across thread counts
 *   (per-(row,y) arithmetic is executor-independent).
 *
 * Everything here reuses the gemv2 slab/pack/pool API via includes; the
 * gemv2/gemv/fmt/ref sources are untouched.
 */
#ifndef DOML_GEMM_H
#define DOML_GEMM_H

#include <stddef.h>
#include <stdint.h>

#include "../gemv2/doml_gemv2.h" /* Doml2W slab + packer + pool (reused) */

/* ------------------------------------------------------- activations ------ */
typedef struct {
    int8_t *q;    /* [ny][K] s8, panel-permuted order, 64B-aligned          */
    float *dx;    /* [ny][NG] group scales                                  */
    float *b2;    /* [ny][NG] 128 * dx * bsum  (u8-bias fixup term)         */
    uint32_t ny, K, NG;
} Doml3X;

/* allocate/free the buffers (q zeroed so pad rows are inert) */
void doml3_x_alloc(Doml3X *qx, uint32_t ny, uint32_t K);
void doml3_x_free(Doml3X *qx);

/* quantize fp32 rows [y0,y1) of X (row stride K) into qx */
void doml3_quant_x_rows(const float *X, uint32_t K, Doml3X *qx,
                        uint32_t y0, uint32_t y1);

/* ------------------------------------------------------------- panel ------ */
typedef struct {
    uint8_t *pu8;        /* [srows_cap * K] interleaved u8 levels           */
    float *sc;           /* [NG][srows_cap] fp32 scales                     */
    uint32_t srows_cap;  /* strip capacity in rows (multiple of 32)         */
    uint32_t K, NG;
    size_t bytes;        /* total allocated (working-memory accounting)     */
} Doml3Panel;

void doml3_panel_alloc(Doml3Panel *pan, uint32_t srows_cap, uint32_t K,
                       uint32_t NG);
void doml3_panel_free(Doml3Panel *pan);

/* Convert row-groups [rg0, rg1) (16 rows each) of the i8 slab into pan
 * (strip-relative; (rg1-rg0)*16 <= srows_cap). Panel bytes are bitwise the
 * slab's cb-record levels; scales are bf16->fp32 of the record scales. */
void doml3_convert(const Doml2W *w, uint32_t rg0, uint32_t rg1,
                   Doml3Panel *pan);

/* GEMM over a converted strip: rows [rg0*16, rg0*16 + nrows) (nrows multiple
 * of 16), activation columns [0, qx->ny), C row-major per y with leading
 * dimension ldc (C[y*ldc + row]).  mk: 0 = 2x8 flagship (ny%8==0),
 * 1 = 4x4 alternate (ny%4==0). */
void doml3_gemm_strip(const Doml3Panel *pan, uint32_t nrows, uint32_t row0,
                      const Doml3X *qx, float *C, long ldc, int mk);

/* --------------------------------------------------- threaded call glue --- */
/* One full GEMM call = { quantize activations (y striped over threads);
 * barrier; per-thread row slice: convert strip -> GEMM strip }.  Designed to
 * be dispatched via doml2_pool_run with iters calls back-to-back (bench) or
 * once (tests).  Phase timestamps for the --split diagnostic are taken by
 * thread 0 around pool barriers. */
typedef struct {
    const Doml2W *wv;    /* nbuf cycled weight copies (bench) or 1          */
    int nbuf;
    const float *X;      /* fp32 activations [ny][K]                        */
    Doml3X *qx;          /* shared, pre-allocated                           */
    float *C;
    long ldc;
    Doml3Panel *panels;  /* per-thread panels [nth]                         */
    Doml2Pool *pool;
    int iters, rot, mk;
    int split;           /* 1: extra barriers + phase timing (diagnostic)   */
    double t0, t1;       /* filled by thread 0 (fused wall)                 */
    double tq, tc, tg;   /* cumulative phase times (split mode)             */
} Doml3Job;

/* pool job function: doml2_pool_run(pool, doml3_job_exec, &job) */
void doml3_job_exec(void *arg, int ith, int nth);

/* strip cap in rows for (K, env DOML3_STRIP_KB default 256), multiple of 32 */
uint32_t doml3_strip_rows(uint32_t K);

void doml3_init(void); /* idempotent (env knobs) */

#endif /* DOML_GEMM_H */
