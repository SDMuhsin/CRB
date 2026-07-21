/* DPKA v1 — DOML packed-kernel artifact: format definition + loader API.
 *
 * The artifact is ONE mmap-able file per model (written by dpka_export.py;
 * layout documented in that file's docstring and in P1_REPORT.md).
 * The loader mmaps it and builds malloc'd RESIDENT layouts:
 *
 *   R-A ("raw")    : the five container planes rebuilt exactly as in the
 *                    original DPK containers (b0/b1/m full bit planes, s,
 *                    cb 12 fp8 slots per (row,group)).   ~3.376 bpw.
 *   R-B ("packed") : b0 full; b1 only for non-bulk elements (row-byte-
 *                    aligned segments + per-row u32 byte offsets); m only
 *                    for non-salient columns (row-byte-aligned, expansion
 *                    control = the shared s bitmap + per-group non-salient
 *                    prefix table); s; cb 10 real fp8 slots per (row,group)
 *                    ordered [bulk0,bulk1,tail0..3,sal0..3].   ~2.50 bpw.
 *
 * Bit addressing everywhere is LSB-first (spec §2.1):
 *   bit(plane_row, j) = (row_bytes[j >> 3] >> (j & 7)) & 1
 */
#ifndef DPKA_H
#define DPKA_H

#include <stddef.h>
#include <stdint.h>

#define DPKA_MAGIC "DPKART01"
#define DPKA_VERSION 1u
#define DPKA_TOC_OFF 64u
#define DPKA_REC_SIZE 256u

/* rANS coder constants (must match kernels/cpu/fmt/rans.py) */
#define DPKA_PROB_BITS 15u
#define DPKA_PROB_M (1u << DPKA_PROB_BITS)
#define DPKA_RANS_L (1u << 23)

enum { DPKA_PL_B0 = 0, DPKA_PL_B1, DPKA_PL_M, DPKA_PL_S, DPKA_PL_CB, DPKA_NPLANES };

/* On-disk TOC record: 256 bytes, little-endian, naturally aligned. */
typedef struct {
    char name[128];
    uint32_t R, C, C_orig, g, NG, n_sal;
    uint64_t n_nonbulk;   /* total tail+salient elements (= stored b1 bits) */
    uint64_t n_m_bits;    /* R * (C_orig - n_sal) coded membership bits */
    uint32_t f1;          /* rANS freq of bit=1 (tail), out of 2^15 */
    uint32_t reserved0;
    uint64_t off[DPKA_NPLANES];   /* absolute file offsets: b0,b1,m,s,cb */
    uint64_t size[DPKA_NPLANES];  /* payload sizes in bytes               */
} DpkaTensorRec;

typedef struct {
    int fd;
    const uint8_t *base;   /* mmap base */
    size_t fsize;
    uint32_t n_tensors;
    uint64_t total_weights;
    const DpkaTensorRec *toc;  /* points into the mapping */
} DpkaFile;

/* ---- resident layout R-A: raw container planes -------------------------- */
typedef struct {
    uint32_t R, C, C_orig, g, NG;
    uint8_t *b0;   /* R * C/8, full bit plane */
    uint8_t *b1;   /* R * C/8, full bit plane (0 at bulk) */
    uint8_t *m;    /* R * C/8, full bit plane (0 at salient) */
    uint8_t *s;    /* C/8 */
    uint8_t *cb;   /* R * NG * 12 fp8 bytes: [3 partitions][4 slots] */
    /* byte accounting (malloc'd sizes) */
    size_t bytes_b0, bytes_b1, bytes_m, bytes_s, bytes_cb;
} DpkaResA;

/* ---- resident layout R-B: packed ---------------------------------------- */
typedef struct {
    uint32_t R, C, C_orig, g, NG, n_sal;
    uint8_t *b0;          /* R * C/8, full bit plane */
    uint8_t *b1;          /* non-bulk bits, LSB-first, per-row byte-aligned */
    uint32_t *b1_rowoff;  /* R+1 byte offsets into b1 (row r: [off[r],off[r+1])) */
    uint8_t *m;           /* R * m_pitch: non-salient bits, LSB-first */
    uint32_t m_pitch;     /* ceil((C_orig - n_sal)/8) bytes per row */
    uint8_t *s;           /* C/8 (the shared m/b1 expansion control) */
    uint8_t *cb;          /* R * NG * 10 fp8: [bulk0,bulk1,tail0..3,sal0..3] */
    uint32_t *ns_prefix;  /* NG+1: non-salient columns before group boundary
                             (shared across rows; random access helper) */
    /* byte accounting (malloc'd sizes) */
    size_t bytes_b0, bytes_b1, bytes_b1off, bytes_m, bytes_s, bytes_cb,
           bytes_aux;
} DpkaResB;

/* base index of each partition's slots inside the 10-entry R-B table */
static const uint8_t DPKA_RB_CB_BASE[3] = { 0u, 2u, 6u };

DpkaFile *dpka_open(const char *path);
void dpka_close(DpkaFile *f);
int dpka_find(const DpkaFile *f, const char *name);   /* -1 if absent */

/* rANS-decode tensor idx's m plane into a freshly malloc'd FULL bit plane
 * (R * C/8 bytes, salient columns 0). Aborts on any integrity violation. */
uint8_t *dpka_decode_m_plane(const DpkaFile *f, int idx);

DpkaResA *dpka_build_ra(const DpkaFile *f, int idx);
DpkaResB *dpka_build_rb(const DpkaFile *f, int idx);
void dpka_free_ra(DpkaResA *a);
void dpka_free_rb(DpkaResB *b);

#endif /* DPKA_H */
