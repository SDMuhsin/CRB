// dpk_gemm.cu — W2A4 tensor-core GEMM (M >= 1) on the DPK format. (K4b, v2)
//
// Implements the doc 02 §8 GEMM path (Marlin-style tiles) with the PI-decided
// K4 contract:
//
//   Y[t, r] = a_s_vec[t] * sum_j W[r, j] * (xhat[t, j] - 8)
//   W[r, j] = cb[r][j // g][part(r, j)][code(r, j)]      (doc 02 §3 invariant)
//
// Per k-tile, the weight tile is decoded from the DPK streams (b0/b1/m planes,
// s bitmap, per-(row, group) 12-level bf16 codebooks) into SHARED MEMORY as
// bf16; the activation tile's excess-8 nibbles are expanded to bf16(x - 8) in
// shared; then bf16 tensor-core wmma (16x16x16, fp32 accumulate) consumes both.
// The per-token fp32 scale is applied once in the epilogue.
//
// NO global-memory dequant buffer exists anywhere: the only global allocation
// per call is the output tensor Y (accountability protocol of
// 00_OBJECTIVE_AND_REQUIREMENTS.md — peak delta must equal Y's bytes).
//
// Decode-to-shared conventions (all normative per doc 02 §§2a/3/4):
//   * bit i of plane word w covers column 32*w + i (LSB-first)
//   * nibble n of activation word w covers column 8*w + n (LSB-first)
//   * lut index = s<<3 | (m & ~s)<<2 | b1<<1 | b0  in [0, 12); this equals
//     part*4 + code, i.e. exactly the flattened cb[..][3][4] order. Entries
//     12..15 are unreachable (s=1 forces the m' bit to 0) and hold 0.
//   * weight rows >= R and token rows >= M decode to exact zeros (pad nibble
//     0x8 -> bf16 0), so no output-tile edge branches are needed in the mma.
//
// Tiling:
//   BK = 64 columns per k-step. g is a multiple of 128 and k-tiles are
//   64-aligned, so a k-tile NEVER straddles a codebook group; the group's
//   16-entry per-row LUT lives in shared, double-buffered (buffer G & 1),
//   prefetched one group ahead. Shared tiles use leading dim 72 (= BK + 8,
//   multiple of 8 as wmma requires).
//
// K4b (v2) performance features vs v1 — SAME storage format, SAME math, same
// per-k-step wmma accumulation order (v1's FLUSH numerics analysis carries
// over unchanged). Each was kept only after measuring a win on the A40 bench
// grid (losers — cp.async shared staging of the planes, per-k-step register
// prefetch, an L-size double-buffered pipeline — are documented in
// llmdocs/cuda_kernel/K4b_gemm_v2.md and were removed):
//   1. UNIFIED decode unit space: X-tile expansion (2*BM units) and W-tile
//      decode (2*BN units) are one flat unit list served by ALL threads.
//      v1 ran them as two back-to-back loops, each engaging only a subset
//      of the CTA (config 64x64: half the threads idled through decode).
//   2. WST — super-tile REGISTER staging of the bit-planes: at BK = 64 a
//      per-k-step (row, plane) read is 8 B, so a warp's 32 lanes touch 16
//      scattered 32 B DRAM sectors per plane load (~4x overfetch; measured
//      as ~50% of total time at 4096x14336, M = 16). With WST, each W
//      thread owns one (row, half) unit and, once per 4 k-steps, loads one
//      uint4 (16 B) per plane — lane pairs cover 32 B, every sector fully
//      used — plus the matching s-bitmap words. Sub-steps assemble their
//      two plane words with one __shfl_xor from the lane partner. No shared
//      traffic, ~16 registers per W thread.
//   3. KS — CTA-internal k-split: the warps divide into KS groups; group q
//      processes k-step range [q*ST, (q+1)*ST) into its OWN tiles/LUTs and
//      accumulates independently; partials are combined IN SHARED in a
//      fixed order in the epilogue (bitwise deterministic, no global
//      workspace — the peak == Y contract holds). Fills the SM at small M
//      where the output grid alone underfills the GPU (4096^2, M = 16: 64
//      CTAs on 84 SMs).
//   4. DBT — double-buffered decoded tiles, ONE barrier per k-step: each
//      iteration issues the WST loads for step t+1, the mma of step t
//      (buffer t & 1), the LUT prefetch, then the decode of step t+1
//      (buffer ~t & 1). The barrier between decode and mma disappears, so
//      the LSU-heavy decode overlaps the tensor-core phase; the single
//      end-of-iteration barrier orders buffer reuse. Doubles the tile
//      shared memory, so the L config keeps DBT off (it would drop to
//      1 CTA/SM — measured loss).
//
// Config table (host dispatch, thresholds measured on A40):
//   L:   128x128, 256 thr (4x2 warps, 32x64/warp), FLUSH=0, WST
//        — M > 64, grid >= 96 CTAs, C <= 4096 (no-flush depth bound)
//   Mnf:  64x 64, 256 thr (2x4 warps, 32x16/warp), FLUSH=0, WST+DBT, C<=4096
//   Mf:   64x 64, 256 thr, FLUSH=16, WST+DBT — the C > 4096 fallback
//   S:    16x 64, 128 thr (1x4 warps, 16x16/warp), FLUSH=16, WST+DBT
//        — M <= 16 on grids >= 168 CTAs
//   S4:   16x 64, 512 thr (4 k-slices x 4 warps), FLUSH=16, WST, KS=4
//        — M <= 16 on small grids

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

#define DPK_BK 64
#define DPK_LDT 72  // shared tile leading dim (bf16 elems): BK + 8, multiple of 8

// FLUSH: accumulator-flush period in k-steps (0 = disabled). The wmma k-loop
// adds one fp32 product-sum fragment per 16 columns SEQUENTIALLY into acc, so
// plain accumulation has summation depth C/16 (2048 at C=32768) and its
// rounding error grows past the documented 1e-5 gate for very large C.
// With FLUSH > 0, acc is drained every FLUSH k-steps into a separate fp32
// running total (element-wise fragment add — valid: the opaque fragment
// layout is element-consistent), bounding the depth at 4*FLUSH + NK/FLUSH.
// Costs 8*FM*FN registers, so it is enabled only for the configs with
// register headroom; the big-tile config L keeps FLUSH=0 and is only
// dispatched for C <= 4096 (depth <= 256, measured error ~6e-7 << gate).
// KS > 1 only SHORTENS each slice's sequential depth (each slice drains its
// own range; the cross-slice combine adds KS-1 ordered fp32 adds), and DBT
// does not change the accumulation order at all, so the v1 bound holds.
template <int BM, int BN, int TPB, int WGM, int WGN, int FLUSH, int KS,
          bool WST, bool DBT = false>
__global__ void __launch_bounds__(TPB) dpk_gemm_kernel(
    const uint32_t *__restrict__ b0p,   // [R, CW] LSB of level code
    const uint32_t *__restrict__ b1p,   // [R, CW] MSB of level code
    const uint32_t *__restrict__ mp,    // [R, CW] 1 = tail P2 (don't-care at salient)
    const uint32_t *__restrict__ sp,    // [CW]    1 = salient column P3
    const __nv_bfloat16 *__restrict__ cb,  // [R, NG, 3, 4]
    const uint32_t *__restrict__ xh,    // [M, 4*CW] excess-8 nibbles, LSB-first
    const float *__restrict__ a_s_vec,  // [M] per-token fp32 scales
    int M, int R, int CW, int NG, int GC,  // GC = g/32 chunks per group
    __nv_bfloat16 *__restrict__ y_bf16,    // [M, R] (nullptr in fp32 mode)
    float *__restrict__ y_f32)             // [M, R] optional fp32 output
{
    constexpr int NWARP = TPB / 32;
    constexpr int NWPS = NWARP / KS;      // warps per k-slice
    constexpr int TPS = TPB / KS;         // threads per k-slice
    constexpr int FM = BM / (WGM * 16);   // wmma tiles per warp, M dir
    constexpr int FN = BN / (WGN * 16);   // wmma tiles per warp, N dir
    static_assert(NWPS * KS == NWARP, "k-slices must partition the warps");
    static_assert(WGM * WGN == NWPS, "warp grid must cover one k-slice");
    static_assert(BM % (WGM * 16) == 0 && BN % (WGN * 16) == 0, "tile split");
    constexpr int UT = 2 * BN + 2 * BM;   // decode units per slice per k-step
    constexpr int UPT = (UT + TPS - 1) / TPS;  // decode units per thread
    constexpr int DB = DBT ? 2 : 1;       // decoded-tile buffers
    // epilogue staging (NWARP x 256 fp32) must fit in the dead tile region
    static_assert(NWARP * 256 * 4 <= DB * KS * (BM + BN) * DPK_LDT * 2,
                  "staging must fit in tiles");
    // WST needs every W unit in decode slot 0 (a thread's plane registers
    // belong to exactly one (row, half) unit) and full W warps for the
    // __shfl_xor exchange (2*BN is a multiple of 32).
    static_assert(!WST || (TPS >= 2 * BN && (2 * BN) % 32 == 0),
                  "WST unit mapping");

    const int XW = CW * 4;               // activation words per token row (C/8)
    const int n0 = blockIdx.x * BN;      // weight-row block origin
    const int m0 = blockIdx.y * BM;      // token block origin

    extern __shared__ uint8_t smem[];
    __nv_bfloat16 *Xs = reinterpret_cast<__nv_bfloat16 *>(smem);  // [DB][KS][BM][LDT]
    __nv_bfloat16 *Ws = Xs + DB * KS * BM * DPK_LDT;              // [DB][KS][BN][LDT]
    __nv_bfloat16 *lut = Ws + DB * KS * BN * DPK_LDT;             // [KS][2][BN][16]
    uint32_t *xlut = reinterpret_cast<uint32_t *>(lut + KS * 2 * BN * 16);  // [256]
    float *as_sh = reinterpret_cast<float *>(xlut + 256);                   // [BM]

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int q = wid / NWPS;            // this warp's k-slice
    const int wsl = wid % NWPS;          // warp index within the slice
    const int lt = tid - q * TPS;        // thread index within the slice
    const int warp_m0 = (wsl % WGM) * (FM * 16);
    const int warp_n0 = (wsl / WGM) * (FN * 16);

    // this slice's tiles (of double buffer b = step & 1 when DBT) and LUTs
    auto Xtile = [&](int step) {
        return Xs + ((size_t)(DBT ? step & 1 : 0) * KS + q) * BM * DPK_LDT;
    };
    auto Wtile = [&](int step) {
        return Ws + ((size_t)(DBT ? step & 1 : 0) * KS + q) * BN * DPK_LDT;
    };
    __nv_bfloat16 *lutq = lut + q * 2 * BN * 16;

    // ---- one-time staging: nibble-pair LUT + per-token scales ----
    // xlut[b] = packed bf16x2 { bf16((b & 15) - 8), bf16((b >> 4) - 8) }:
    // byte q of an activation word holds columns (8w + 2q, 8w + 2q + 1),
    // low nibble = first column = low bf16 half (little-endian order).
    for (int i = tid; i < 256; i += TPB) {
        __nv_bfloat162 v = __floats2bfloat162_rn((float)(i & 15) - 8.0f,
                                                 (float)(i >> 4) - 8.0f);
        xlut[i] = *reinterpret_cast<uint32_t *>(&v);
    }
    for (int i = tid; i < BM; i += TPB) {
        int t = m0 + i;
        as_sh[i] = (t < M) ? a_s_vec[t] : 0.0f;
    }

    // ---- codebook group LUT loader (slice-local, into buffer G & 1) ----
    // lut[q][buf][r][0..11] = cb[n0+r][G][*][*] (flattened = part*4 + code),
    // [12..15] = 0 (unreachable indices); rows >= R decode to all-zeros.
    auto load_lut = [&](int G) {
        __nv_bfloat16 *dst = lutq + (size_t)(G & 1) * BN * 16;
        for (int r = lt; r < BN; r += TPS) {
            int row = n0 + r;
            uint2 v0 = make_uint2(0u, 0u), v1 = v0, v2 = v0;
            if (row < R) {
                const uint2 *src = reinterpret_cast<const uint2 *>(
                    cb + ((size_t)row * NG + G) * 12);
                v0 = __ldg(src);
                v1 = __ldg(src + 1);
                v2 = __ldg(src + 2);
            }
            uint2 *d = reinterpret_cast<uint2 *>(dst + r * 16);
            d[0] = v0;
            d[1] = v1;
            d[2] = v2;
            d[3] = make_uint2(0u, 0u);
        }
    };

    // ---- raw-word loader for one decode unit at k-step ks ----
    // W unit (u < 2*BN, non-WST configs): {b0, b1, m, s} words for one
    // (row, 32-column chunk); rows >= R load zeros (decode to 0 via the
    // zeroed LUT). X unit: one uint4 = 4 activation words (32 columns);
    // token rows >= M load the pad nibble pattern (0x8 -> bf16 0).
    auto load_unit = [&](int u, int ks) -> uint4 {
        if (u < 2 * BN) {
            const int r = u >> 1, ch = u & 1, row = n0 + r;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (row < R) {
                const size_t off = (size_t)row * CW + ks * 2 + ch;
                v.x = __ldg(b0p + off);
                v.y = __ldg(b1p + off);
                v.z = __ldg(mp + off);
                v.w = __ldg(sp + ks * 2 + ch);
            }
            return v;
        }
        const int t = (u - 2 * BN) >> 1, qw = (u - 2 * BN) & 1;
        const int gt = m0 + t;
        if (gt < M)
            return *reinterpret_cast<const uint4 *>(
                xh + (size_t)gt * XW + ks * 8 + qw * 4);
        return make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
    };

    // ---- decoder for one unit's raw words into the given tile buffers ----
    auto decode_unit = [&](int u, uint4 w, const __nv_bfloat16 *lg,
                           __nv_bfloat16 *Xb, __nv_bfloat16 *Wb) {
        if (u < 2 * BN) {
            // W unit: decode planes via the group LUT -> bf16 in shared
            const int r = u >> 1, ch = u & 1;
            const uint32_t w0 = w.x, w1 = w.y, wsb = w.w, wm2 = w.z & ~w.w;
            const uint16_t *L = reinterpret_cast<const uint16_t *>(lg + r * 16);
            uint32_t *dst =
                reinterpret_cast<uint32_t *>(Wb + r * DPK_LDT + ch * 32);
#pragma unroll
            for (int j = 0; j < 32; j += 2) {
                uint32_t i0 = ((w0 >> j) & 1u) | (((w1 >> j) & 1u) << 1) |
                              (((wm2 >> j) & 1u) << 2) |
                              (((wsb >> j) & 1u) << 3);
                uint32_t i1 = ((w0 >> (j + 1)) & 1u) |
                              (((w1 >> (j + 1)) & 1u) << 1) |
                              (((wm2 >> (j + 1)) & 1u) << 2) |
                              (((wsb >> (j + 1)) & 1u) << 3);
                dst[j >> 1] = (uint32_t)L[i0] | ((uint32_t)L[i1] << 16);
            }
        } else {
            // X unit: expand nibbles -> bf16(x - 8) in shared
            const int t = (u - 2 * BN) >> 1, qw = (u - 2 * BN) & 1;
            __nv_bfloat16 *dst = Xb + t * DPK_LDT + qw * 32;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t v = (&w.x)[k];
                uint4 o;
                o.x = xlut[v & 255u];
                o.y = xlut[(v >> 8) & 255u];
                o.z = xlut[(v >> 16) & 255u];
                o.w = xlut[v >> 24];
                *reinterpret_cast<uint4 *>(dst + k * 8) = o;
            }
        }
    };

    using AccFrag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    AccFrag acc[FM][FN];
    constexpr int NELEM = AccFrag::num_elements;
    float grand[FLUSH > 0 ? FM * FN * NELEM : 1];  // fp32 running totals
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    if (FLUSH > 0)
#pragma unroll
        for (int i = 0; i < FM * FN * NELEM; ++i) grand[i] = 0.0f;

    const int NK = CW / 2;              // k-steps of BK=64 columns
    const int SPG = GC / 2;             // k-steps per codebook group
    // k-steps per slice (uniform across slices -> uniform barrier counts).
    // With WST and KS > 1, ST is rounded up to even so every slice starts on
    // an even k-step: the 16 B plane loads (word offset 2*ks0 + 4h + 8m)
    // then stay 16-byte aligned. Costs at most one inactive tail step.
    const int ST0 = (NK + KS - 1) / KS;
    const int ST = (WST && KS > 1) ? (ST0 + 1) & ~1 : ST0;
    const int ks0 = q * ST;             // this slice's first k-step

    // WST plane registers: this thread's 16 B half-super-tile of each plane
    // + the matching s-bitmap words. s is row-independent but rides the same
    // (offset, shuffle) scheme, which removes the last per-k-step scattered
    // global load from the W decode. Lane pairs (row, h = 0/1) load adjacent
    // 16 B chunks -> full 32 B DRAM sectors. Out-of-range (row >= R or
    // past-CW ragged chunk) loads zero-fill; rows >= R decode through an
    // all-zero LUT anyway, so the result is deterministic either way.
    uint4 pw0 = make_uint4(0u, 0u, 0u, 0u), pw1 = pw0, pw2 = pw0, pw3 = pw0;
    auto wst_load = [&](int kb) {
        const int row = n0 + (lt >> 1);
        const int off = kb * 2 + (lt & 1) * 4;
        if (row < R && off < CW) {
            const size_t o = (size_t)row * CW + off;
            pw0 = *reinterpret_cast<const uint4 *>(b0p + o);
            pw1 = *reinterpret_cast<const uint4 *>(b1p + o);
            pw2 = *reinterpret_cast<const uint4 *>(mp + o);
            pw3 = *reinterpret_cast<const uint4 *>(sp + off);
        } else {
            pw0 = pw1 = pw2 = pw3 = make_uint4(0u, 0u, 0u, 0u);
        }
    };

    // ---- decode one k-step's tiles (X + W in one flat unit space) ----
    auto decode_tiles = [&](int step) {
        const int ksd = ks0 + step;
        const int Gd = ksd / SPG;
        const __nv_bfloat16 *lg = lutq + (size_t)(Gd & 1) * BN * 16;
        __nv_bfloat16 *Xb = Xtile(step);
        __nv_bfloat16 *Wb = Wtile(step);
        const int j = step & 3;        // WST sub-step within the super-tile
        const int hsrc = j >> 1;
        const bool odd = (j & 1) != 0;
#pragma unroll
        for (int k = 0; k < UPT; ++k) {
            const int u = lt + k * TPS;
            if (u >= UT) continue;
            uint4 w;
            if (WST && u < 2 * BN) {
                // W unit: assemble this sub-step's plane words from the
                // super-tile registers. Sub-step j needs global words
                // 2j + ch; the holder half is hsrc = j >> 1, its local
                // pair is words 2(j&1), 2(j&1)+1. Each thread keeps the
                // word for its own ch and ships the other to its lane
                // partner (unit u ^ 1 == lane ^ 1) via one __shfl_xor.
                const int h = u & 1;
                auto pick = [&](const uint4 &pw) -> uint32_t {
                    const uint32_t lo = odd ? pw.z : pw.x;
                    const uint32_t hi = odd ? pw.w : pw.y;
                    const uint32_t mine = h ? hi : lo;
                    const uint32_t other = h ? lo : hi;
                    const uint32_t got =
                        __shfl_xor_sync(0xFFFFFFFFu, other, 1);
                    return (h == hsrc) ? mine : got;
                };
                w.x = pick(pw0);
                w.y = pick(pw1);
                w.z = pick(pw2);
                w.w = pick(pw3);
            } else {
                w = load_unit(u, ksd);
            }
            decode_unit(u, w, lg, Xb, Wb);
        }
    };

    // ---- tensor-core mma over one k-tile (4 sub-steps of k=16) ----
    auto mma_tiles = [&](int step) {
        const __nv_bfloat16 *Xb = Xtile(step);
        const __nv_bfloat16 *Wb = Wtile(step);
#pragma unroll
        for (int kk = 0; kk < DPK_BK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                           wmma::row_major> af[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                           wmma::col_major> bf[FN];
#pragma unroll
            for (int i2 = 0; i2 < FM; ++i2)
                wmma::load_matrix_sync(
                    af[i2], Xb + (warp_m0 + i2 * 16) * DPK_LDT + kk * 16,
                    DPK_LDT);
#pragma unroll
            for (int j = 0; j < FN; ++j)
                wmma::load_matrix_sync(
                    bf[j], Wb + (warp_n0 + j * 16) * DPK_LDT + kk * 16,
                    DPK_LDT);
#pragma unroll
            for (int i2 = 0; i2 < FM; ++i2)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    wmma::mma_sync(acc[i2][j], af[i2], bf[j], acc[i2][j]);
        }
    };

    // ---- fp32 running-total drain (bounds summation depth) ----
    auto flush_acc = [&](int i) {
        if (FLUSH > 0 &&
            ((i + 1) % (FLUSH > 0 ? FLUSH : 1) == 0 || i == ST - 1)) {
#pragma unroll
            for (int i2 = 0; i2 < FM; ++i2)
#pragma unroll
                for (int j = 0; j < FN; ++j)
#pragma unroll
                    for (int e = 0; e < NELEM; ++e) {
                        grand[(i2 * FN + j) * NELEM + e] += acc[i2][j].x[e];
                        acc[i2][j].x[e] = 0.0f;
                    }
        }
    };

    if (DBT) {
        // ---- single-barrier pipeline: decode(t + 1) overlaps mma(t) ----
        // Prologue loads BOTH halves of the LUT double buffer (the in-loop
        // prefetch of G + 1 triggers when the DECODED step enters group G,
        // >= SPG - 1 >= 1 barriers before its first consumer, and >= 1
        // barrier after the last reader of the buffer it overwrites).
        if (ks0 < NK) {
            const int G0 = ks0 / SPG;
            load_lut(G0);
            if (G0 + 1 < NG) load_lut(G0 + 1);
            if (WST && lt < 2 * BN) wst_load(ks0);
        }
        __syncthreads();               // luts + xlut + as_sh visible
        if (ks0 < NK) decode_tiles(0);
        __syncthreads();               // tile buffer 0 ready

        for (int i = 0; i < ST; ++i) {
            const int ks = ks0 + i;
            const bool act = ks < NK;  // inactive tail steps still barrier
            const int id = i + 1;      // step decoded this iteration
            const int ksd = ks0 + id;
            const bool dact = (id < ST) && (ksd < NK);
            // WST plane loads for the decoded step first: pure LDGs whose
            // latency is covered by the mma phase issued right after
            if (WST && dact && (id & 3) == 0 && lt < 2 * BN) wst_load(ksd);
            if (act) mma_tiles(i);
            if (dact) {
                const int Gd = ksd / SPG;
                if (ksd == Gd * SPG && Gd + 1 < NG) load_lut(Gd + 1);
                decode_tiles(id);
            }
            flush_acc(i);
            __syncthreads();  // decode(id) ready; mma(i) done CTA-wide, so
                              // buffer i & 1 is writable next iteration
        }
    } else {
        // ---- two-barrier loop (configs without tile headroom for DBT) ----
        if (ks0 < NK) load_lut(ks0 / SPG);  // visible after the first barrier

        for (int i = 0; i < ST; ++i) {
            const int ks = ks0 + i;
            const bool act = ks < NK;   // inactive tail steps still barrier
            const int G = act ? ks / SPG : 0;
            __syncthreads();  // previous mma done -> tiles writable

            // WST: refill the plane registers at each super-tile boundary
            if (WST && act && (i & 3) == 0 && lt < 2 * BN) wst_load(ks);

            // Prefetch the NEXT group's LUT during this slice's first
            // k-step of each group. A slice may START mid-group
            // (ks0 % SPG != 0): its prologue loaded that group, and i == 0
            // prefetches the next one; the two writes go to adjacent
            // (distinct) halves of the double buffer. The buffer being
            // overwritten was last read >= 1 barrier earlier (as in v1).
            if (act && (i == 0 || ks == G * SPG) && G + 1 < NG)
                load_lut(G + 1);

            if (act) decode_tiles(i);
            __syncthreads();  // tiles ready -> mma
            if (act) mma_tiles(i);
            flush_acc(i);
        }
    }
    if (FLUSH > 0)  // move the totals back so the epilogue stays uniform
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j)
#pragma unroll
                for (int e = 0; e < NELEM; ++e)
                    acc[i][j].x[e] = grand[(i * FN + j) * NELEM + e];

    // ---- epilogue: per-warp 16x16 fp32 staging (reuses the dead tile
    // region). For KS > 1 the k-slices' partial fragments are combined IN
    // SHARED in a fixed order (slice 0 += slice 1, 2, ... — bitwise
    // deterministic), then slice 0's warps apply the per-token scale and
    // store. Lane <-> staging mapping: lane reads exactly the elements
    // (32*e + lane) it also summed, so no extra intra-warp sync is needed
    // beyond the v1 __syncwarp() pattern. ----
    __syncthreads();  // all mma done; Xs/Ws reusable as staging
    float *stg = reinterpret_cast<float *>(smem) + wid * 256;
#pragma unroll
    for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            wmma::store_matrix_sync(stg, acc[i][j], 16, wmma::mem_row_major);
            if (KS > 1) {
                __syncthreads();  // all slices' fragment (i, j) staged
                if (q == 0) {
                    float *base = reinterpret_cast<float *>(smem);
#pragma unroll
                    for (int x = lane; x < 256; x += 32) {
                        float v = stg[x];
#pragma unroll
                        for (int qq = 1; qq < KS; ++qq)
                            v += base[(size_t)(qq * NWPS + wsl) * 256 + x];
                        stg[x] = v;
                    }
                }
            }
            __syncwarp();
            if (q == 0) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    int li = 2 * e + (lane >> 4);
                    int lj = lane & 15;
                    int lm = warp_m0 + i * 16 + li;   // row within the CTA tile
                    int t = m0 + lm;
                    int rr = n0 + warp_n0 + j * 16 + lj;
                    if (t < M && rr < R) {
                        float v = stg[li * 16 + lj] * as_sh[lm];
                        if (y_f32)
                            y_f32[(size_t)t * R + rr] = v;
                        else
                            y_bf16[(size_t)t * R + rr] = __float2bfloat16(v);
                    }
                }
            }
            if (KS > 1)
                __syncthreads();  // staging slots reusable for the next frag
            else
                __syncwarp();
        }
    }
}

// -------- host launcher: config dispatch by M (no allocations here) --------

template <int BM, int BN, int TPB, int WGM, int WGN, int FLUSH, int KS,
          bool WST, bool DBT = false>
static void launch_cfg(const uint32_t *b0, const uint32_t *b1,
                       const uint32_t *m, const uint32_t *s, const void *cb,
                       const uint32_t *xhat, const float *a_s_vec, int M,
                       int R, int CW, int NG, int GC, void *y_bf16,
                       float *y_f32, cudaStream_t stream) {
    dim3 grid((R + BN - 1) / BN, (M + BM - 1) / BM);
    size_t shmem = (size_t)(DBT ? 2 : 1) * KS * (BM + BN) * DPK_LDT * 2  // tiles
                   + (size_t)KS * 2 * BN * 16 * 2  // double-buffered cb LUT
                   + 256 * 4                       // nibble-pair LUT
                   + (size_t)BM * 4;               // per-token scales
    auto *kfn = dpk_gemm_kernel<BM, BN, TPB, WGM, WGN, FLUSH, KS, WST, DBT>;
    if (shmem > 48 * 1024) {  // SM86 needs an opt-in past 48 KB (once per cfg)
        static bool once = [&] {
            cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)shmem);
            return true;
        }();
        (void)once;
    }
    kfn<<<grid, TPB, shmem, stream>>>(
        b0, b1, m, s, reinterpret_cast<const __nv_bfloat16 *>(cb), xhat,
        a_s_vec, M, R, CW, NG, GC,
        reinterpret_cast<__nv_bfloat16 *>(y_bf16), y_f32);
}

void dpk_gemm_launch(const uint32_t *b0, const uint32_t *b1, const uint32_t *m,
                     const uint32_t *s, const void *cb, const uint32_t *xhat,
                     const float *a_s_vec, int M, int R, int CW, int NG,
                     int GC, void *y_bf16, float *y_f32, cudaStream_t stream) {
    // Config choice (all thresholds measured on the A40 bench grid, K4b):
    //  * M <= 16: S (16x64) — with a 4-way CTA-internal k-split (S4) when
    //    the output grid alone would underfill the GPU (< 168 CTAs), else
    //    the single-barrier DBT pipeline flavor.
    //  * M > 64 on large grids at C <= 4096: L (128x128, FLUSH=0 -> its
    //    summation depth C/16 <= 256 holds the 1e-5 gate; measured 6e-7).
    //    The 96-CTA threshold: at 112 CTAs (4096x14336, M=128) L-WST beats
    //    the 64x64 DBT config 0.373 vs 0.518 ms; at 32 CTAs (4096^2,
    //    M=128) it loses 0.249 vs 0.173 ms.
    //  * otherwise: 64x64 DBT pipeline; FLUSH=0 up to C = 4096 (same depth
    //    bound as L), FLUSH=16 beyond.
    auto ctas = [&](int bm, int bn) {
        return (int64_t)((M + bm - 1) / bm) * ((R + bn - 1) / bn);
    };
    if (M <= 16) {
        if (ctas(16, 64) >= 168)
            launch_cfg<16, 64, 128, 1, 4, 16, 1, true, true>(
                b0, b1, m, s, cb, xhat, a_s_vec, M, R, CW, NG, GC, y_bf16,
                y_f32, stream);
        else
            launch_cfg<16, 64, 512, 1, 4, 16, 4, true>(
                b0, b1, m, s, cb, xhat, a_s_vec, M, R, CW, NG, GC, y_bf16,
                y_f32, stream);
    } else if (M > 64 && ctas(128, 128) >= 96 && CW <= 128) {
        launch_cfg<128, 128, 256, 4, 2, 0, 1, true>(
            b0, b1, m, s, cb, xhat, a_s_vec, M, R, CW, NG, GC, y_bf16, y_f32,
            stream);
    } else if (CW <= 128) {
        launch_cfg<64, 64, 256, 2, 4, 0, 1, true, true>(
            b0, b1, m, s, cb, xhat, a_s_vec, M, R, CW, NG, GC, y_bf16, y_f32,
            stream);
    } else {
        launch_cfg<64, 64, 256, 2, 4, 16, 1, true, true>(
            b0, b1, m, s, cb, xhat, a_s_vec, M, R, CW, NG, GC, y_bf16, y_f32,
            stream);
    }
}
