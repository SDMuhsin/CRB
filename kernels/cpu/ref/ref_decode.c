/* Scalar reference decoder for the DPKA resident layouts (R-A and R-B).
 *
 * Semantics (DPK_FORMAT_SPEC.md §3):
 *   part(i,j) = s[j] ? 2 : (m[i][j] ? 1 : 0)      // 0 bulk, 1 tail, 2 salient
 *   code(i,j) = b0[i][j] + 2*b1[i][j]             // 0..3; bulk: b1 absent => 0..1
 *   W[i,j]    = cb[i][j/g][part][code]            // fp8 -> bf16 -> fp32 (<<16)
 *
 * Bit addressing is LSB-first: bit(bytes, j) = (bytes[j>>3] >> (j&7)) & 1.
 * This file is CORRECTNESS code — clarity over speed, no intrinsics.
 */
#include "ref_decode.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int bit_get(const uint8_t *bytes, size_t j)
{
    return (bytes[j >> 3] >> (j & 7)) & 1;
}

void dpka_fp8e4m3_to_bf16_table(uint16_t tab[256])
{
    for (int byte = 0; byte < 256; byte++) {
        int sign = byte >> 7;
        int e = (byte >> 3) & 0xF;
        int mant = byte & 0x7;
        if (e == 0xF && mant == 0x7) {      /* NaN: never referenced (asserted) */
            tab[byte] = 0x7FC0;
            continue;
        }
        /* exact value: normal (1+m/8)*2^(e-7) = (8+m)*2^(e-10);
         * subnormal (e==0): (m/8)*2^-6 = m*2^-9 */
        float mag = e ? ldexpf((float)(8 + mant), e - 10)
                      : ldexpf((float)mant, -9);
        float v = sign ? -mag : mag;        /* -0.0f preserved for byte 0x80 */
        uint32_t bits;
        memcpy(&bits, &v, 4);
        if (bits & 0xFFFFu) {               /* provably zero; belt-and-braces */
            fprintf(stderr, "fp8 table: value not exactly bf16 (byte %d)\n",
                    byte);
            abort();
        }
        tab[byte] = (uint16_t)(bits >> 16);
    }
}

static inline float bf16_pattern_to_f32(uint16_t pat)
{
    uint32_t bits = (uint32_t)pat << 16;
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

void dpka_ref_decode_row_ra(const DpkaResA *a, const uint16_t tab[256],
                            uint32_t r, float *out)
{
    const size_t pitch = a->C / 8;
    const uint8_t *b0r = a->b0 + (size_t)r * pitch;
    const uint8_t *b1r = a->b1 + (size_t)r * pitch;
    const uint8_t *mr = a->m + (size_t)r * pitch;
    const uint8_t *cbr = a->cb + (size_t)r * a->NG * 12;

    for (uint32_t j = 0; j < a->C_orig; j++) {
        int sal = bit_get(a->s, j);
        int tail = bit_get(mr, j);
        int part = sal ? 2 : tail;                     /* salient overrides m */
        int code = bit_get(b0r, j) | (bit_get(b1r, j) << 1);
        uint32_t gi = j / a->g;
        uint8_t fp8 = cbr[(gi * 3 + (uint32_t)part) * 4 + (uint32_t)code];
        out[j] = bf16_pattern_to_f32(tab[fp8]);
    }
}

void dpka_ref_decode_row_rb(const DpkaResB *b, const uint16_t tab[256],
                            uint32_t r, float *out)
{
    const size_t pitch = b->C / 8;
    const uint8_t *b0r = b->b0 + (size_t)r * pitch;
    const uint8_t *b1seg = b->b1 + b->b1_rowoff[r];    /* this row's non-bulk bits */
    const uint8_t *mrow = b->m + (size_t)r * b->m_pitch; /* non-salient bits */
    const uint8_t *cbr = b->cb + (size_t)r * b->NG * 10;

    size_t b1pos = 0;   /* running index into the row's non-bulk b1 bits */
    size_t mpos = 0;    /* running index into the row's non-salient m bits */
    for (uint32_t j = 0; j < b->C_orig; j++) {
        int part;
        if (bit_get(b->s, j)) {
            part = 2;                                  /* salient column */
        } else {
            part = bit_get(mrow, mpos);                /* 0 bulk / 1 tail */
            mpos++;
        }
        int code = bit_get(b0r, j);
        if (part != 0) {                               /* non-bulk: b1 stored */
            code |= bit_get(b1seg, b1pos) << 1;
            b1pos++;
        }
        uint32_t gi = j / b->g;
        uint8_t fp8 = cbr[gi * 10 + DPKA_RB_CB_BASE[part] + (uint32_t)code];
        out[j] = bf16_pattern_to_f32(tab[fp8]);
    }
    /* row-level consistency: consumed exactly the stored bits */
    if (mpos != b->C_orig - b->n_sal) {
        fprintf(stderr, "dpka ref rb: row %u consumed %zu m bits, expected %u\n",
                r, mpos, b->C_orig - b->n_sal);
        abort();
    }
    size_t seg_bytes = b->b1_rowoff[r + 1] - b->b1_rowoff[r];
    if ((b1pos + 7) / 8 != seg_bytes) {
        fprintf(stderr, "dpka ref rb: row %u consumed %zu b1 bits but segment "
                        "is %zu bytes\n", r, b1pos, seg_bytes);
        abort();
    }
}
