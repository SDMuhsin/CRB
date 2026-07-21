/* Scalar reference decode of DPKA resident layouts to fp32 rows.
 * Correctness tier: readable, no intrinsics (P1 brief). */
#ifndef DPKA_REF_DECODE_H
#define DPKA_REF_DECODE_H

#include <stdint.h>

#include "../fmt/dpka.h"

/* Fill tab[256] with the bf16 bit pattern of every fp8-e4m3fn byte.
 * Bit-exact vs torch float8_e4m3fn -> bfloat16 (every finite e4m3fn value
 * is exactly bf16-representable; asserted). NaN codes (0x7F/0xFF) map to
 * 0x7FC0 but are never referenced in this dump (no-NaN asserted at export
 * and at decode time via the cb bytes). */
void dpka_fp8e4m3_to_bf16_table(uint16_t tab[256]);

/* Decode row r (C_orig fp32 values, bf16 patterns widened <<16) */
void dpka_ref_decode_row_ra(const DpkaResA *a, const uint16_t tab[256],
                            uint32_t r, float *out);
void dpka_ref_decode_row_rb(const DpkaResB *b, const uint16_t tab[256],
                            uint32_t r, float *out);

#endif
