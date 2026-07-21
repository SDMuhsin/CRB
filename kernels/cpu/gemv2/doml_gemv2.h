/* DOML GEMV v2 (P2c) — fused row-slab resident + sync redesign.
 *
 * Differences vs P2b (kernels/cpu/gemv/):
 *
 * 1. FUSED ROW-SLAB: per (4-row tile, 256-column group) every byte the kernel
 *    consumes is laid out contiguously in consumption order:
 *
 *      block := [ hdr: u8 b1len[4] ]                          4 B
 *               [ cb : 4 x CBREC   ]  CBREC = 12 (fp) / 14 (i8)
 *               [ b0 : 4 x 32 B    ]  full bit plane, row-major within block
 *               [ m  : 4 x mlen_g  ]  packed non-salient bits (mlen_g shared
 *                                     across rows; 32 B in the m-full ablation)
 *               [ b1 : b1len[0]+..+b1len[3] ]  packed non-bulk bits,
 *                                     byte-aligned per (row,group)
 *
 *    Each thread's tile range is ONE contiguous forward stream (per-tile u32
 *    offsets give random access for work stealing / first-touch). This
 *    deletes the P2b per-row b1 popcount-offset walk (offsets implicit) and
 *    the 4-disjoint-plane prefetch restarts.
 *
 * 2. CB RECORDS ARE THE KERNEL TABLES (approved liberty #2):
 *    - i8 slab: CBREC=14 = [12 x u8 (level q+128, container 12-slot order
 *      [bulk0,bulk1,pad,pad,tail0..3,sal0..3])][bf16 scale]. Table prep is a
 *      single unaligned 16-byte broadcast load (slots 12..15 are never
 *      indexed); kills P2b's 13.9 ns/row on-the-fly i8 quantization.
 *    - fp slab: CBREC=12 = the container 12-slot fp8 table (bit-identical to
 *      the DPK cb plane, R-A form). Container-exact levels; fp8->bf16 via
 *      in-loop vpermi2b LUT.
 *    Kernel index idx = b0 | b1<<1 | m<<2 | s<<3 = part*4 + code in [0,12).
 *
 * 3. SYNC: dissemination barrier with bounded spin -> futex fallback
 *    (graceful at 48t); ONE barrier per GEMV call; static tile slices or
 *    chunked work stealing (atomic per-thread cursors, NUMA-aware probing).
 *
 * Everything is losslessly derived from the DPKA artifact (G-DERIVE); the
 * artifact and the P1/P2b code are untouched.
 *
 * Bit addressing everywhere is LSB-first (DPK spec 2.1).
 */
#ifndef DOML_GEMV2_H
#define DOML_GEMV2_H

#include <stddef.h>
#include <stdint.h>

#include "../fmt/dpka.h"
#include "../gemv/doml_gemv.h" /* DomlQx + activation prep + v1 pool (reused) */

#define DOML2_NG_MAX 24 /* [H17-C] 24 groups = C 6144 (1.7B down_proj); was 12 */
#define DOML2_TILE 4u          /* rows per tile — fixed by the block format  */
#define DOML2_TAIL_PAD 512u    /* slab tail slack for 16 B unaligned loads
                                  (counted in slab_bytes / G-BPW)            */

typedef enum { DOML2_VAR_FP = 0, DOML2_VAR_I8 = 1 } Doml2Var;

/* ------------------------------------------------------------- weights ---- */
typedef struct {
    uint32_t R, C, NG, g, n_sal;
    uint32_t ntiles;          /* R / 4 (R % 4 == 0 enforced)                 */
    uint32_t variant;         /* Doml2Var — fixes CBREC / block geometry     */
    uint32_t m_full;          /* 1: m sections are full 32 B/row (ablation)  */
    const uint8_t *s;         /* slab head: C/8 salient bitmap               */
    const uint8_t *blocks;    /* slab: ntiles*NG fused blocks, tile-major    */
    const uint32_t *tileoff;  /* shared: ntiles+1 byte offsets into blocks   */
    uint8_t mlen[DOML2_NG_MAX]; /* per-group m-section bytes per row         */
    /* byte accounting (see doml2_pack_stats for the component split) */
    size_t weight_bytes;      /* resident bytes consumed per call            */
    size_t slab_bytes;        /* allocated slab incl. align pad + tail slack */
} Doml2W;

/* per-tensor component byte accounting for G-BPW-V2 (filled by size pass) */
typedef struct {
    size_t b0, cb, hdr, m, b1, s, tileoff, align_pad, tail_pad;
    size_t b1_pad_bits, m_pad_bits; /* byte-align padding inside b1/m       */
} Doml2Stats;

/* Size pass: returns total slab bytes (s head + blocks + tail slack) and
 * fills tileoff[0..ntiles] (byte offsets of each tile's first block) and,
 * optionally, the component stats. tileoff must hold R/4 + 1 entries. */
size_t doml2_slab_bytes(const DpkaResB *rb, Doml2Var var, int m_full,
                        uint32_t *tileoff, Doml2Stats *st);

/* Fill pass: init sets pointers + copies the shared small sections (s) and
 * fills w->mlen; pack_tiles derives the fused blocks for tiles [t0,t1) from
 * the R-B planes — call it FROM THE PINNED THREAD that will read those tiles
 * so first-touch places them node-local. tileoff must be the array filled by
 * doml2_slab_bytes for the same (rb, var, m_full). */
void doml2_pack_init(const DpkaResB *rb, Doml2Var var, int m_full,
                     uint8_t *slab, const uint32_t *tileoff, Doml2W *w);
void doml2_pack_tiles(const DpkaResB *rb, const Doml2W *w,
                      uint32_t t0, uint32_t t1);

/* Contiguous tile slice of thread ith among nth (used for first-touch and
 * for static compute slices — keep them consistent). */
void doml2_slice(uint32_t ntiles, int ith, int nth, uint32_t *t0, uint32_t *t1);

/* Reference re-derivation of one i8 cb record from the 10 resident fp8
 * bytes (exposed so gates check the packer's quantization independently):
 * rec14 = [12 x u8 q+128][bf16 scale, RNE from max|level|/127]. */
void doml2_quant_cb10(const uint8_t cb10[10], uint8_t rec14[14]);

/* --------------------------------------------------------------- kernels -- */
void doml2_init(void); /* build fp8->bf16 LUTs once (idempotent) */

void doml2_gemv_fp_tiles(const Doml2W *w, const float *xperm, float *y,
                         uint32_t t0, uint32_t t1);
void doml2_gemv_i8_tiles(const Doml2W *w, const DomlQx *qx, float *y,
                         uint32_t t0, uint32_t t1);

/* ------------------------------------------------------------ thread pool - */
/* Pinned pool, thread t -> CPU t (node0 = even, node1 = odd). Waits are
 * bounded spin (DOML2_SPIN pauses, default 240) then futex sleep, so 48t /
 * co-tenant oversubscription degrades gracefully instead of death-spiraling.
 * doml2_pool_run(fn) = dispatch + fn(0) + one dissemination barrier; fn may
 * call doml2_pool_barrier itself (bench iterations: ONE barrier per call). */
typedef struct Doml2Pool Doml2Pool;
Doml2Pool *doml2_pool_create(int nth);
void       doml2_pool_run(Doml2Pool *p, void (*fn)(void *, int, int), void *arg);
void       doml2_pool_barrier(Doml2Pool *p, int ith);
void       doml2_pool_destroy(Doml2Pool *p);
int        doml2_pool_nth(const Doml2Pool *p);

/* ------------------------------------------------- work distribution ------ */
/* Chunked work stealing over tiles: per-thread atomic cursors over the SAME
 * static slices used for first-touch; a thread that exhausts its slice
 * probes victims (same NUMA parity first) with atomic chunk grabs. Cursors
 * are double-buffered by call parity so repeated calls need only the single
 * completion barrier (each thread re-arms the OTHER parity before the
 * barrier). Outputs are bitwise identical to static slicing (same per-row
 * arithmetic regardless of executor). */
typedef struct {
    _Atomic uint32_t cur[2][64 * 16]; /* [parity][thread*16], 64B padded */
} Doml2Steal;

void doml2_steal_init(Doml2Steal *ws, uint32_t ntiles, int nth);
/* one call's worth of work for thread ith; chunk = tiles per grab */
void doml2_steal_gemv(Doml2Steal *ws, int parity, int chunk, const Doml2W *w,
                      const float *xperm, const DomlQx *qx, float *y,
                      int ith, int nth);

#endif /* DOML_GEMV2_H */
