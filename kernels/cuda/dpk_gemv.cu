// dpk_gemv.cu — W2A4 bucket-popcount GEMV (M=1) on the DPK format.
//
// Implements EXACTLY the algorithm of llmdocs/cuda_kernel/02_storage_format_design.md §7:
//
//   y_i = a_s * sum_G sum_p sum_k cb[i][G][p][k] * (S[p][k] - 8*N[p][k])
//   S[p][k] = sum_{j in G: part=p, code=k} xhat_j ;  N[p][k] = |{j in G: part=p, code=k}|
//
// Bit conventions (normative, doc 02 §2a/§3):
//   * bit i of plane word w covers column 32*w + i (LSB-first)
//   * code(i,j) = b0 + 2*b1 ; part(i,j) = s ? 2 : (m ? 1 : 0)   [0=bulk P1, 1=tail P2, 2=salient P3]
//   * activations: unsigned nibbles xhat = clamp(round(x/a_s),-8,7)+8, 8 per u32,
//     nibble n (bits 4n..4n+3) of word w covers column 8*w + n (LSB-first, documented
//     choice consistent with §2a — §4 does not spell out nibble order)
//   * cb layout: cbdtype[R][NG][3][4], partitions ordered (P1,P2,P3), levels ascending
//
// Thread mapping (doc 02 §7):
//   * CTA = 8 warps, warp = one output row, grid-stride over rows
//   * shared: 4 activation bit-planes xp[t][CW] built once per CTA from the nibble
//     words + the s bitmap; CW = C/32 chunk words. 5*CW*4 bytes (20 KB at C=32768).
//   * lane t handles chunk base+t (coalesced 128 B per plane per warp-iteration)
//   * 12 (S,N) int32 accumulator pairs per lane; fold with the group's 12 bf16
//     codebook entries happens PER LANE whenever the lane's chunk crosses a group
//     boundary (the fold is linear, so lanes may sit in different groups); one
//     warp shuffle reduction of the scalar y at row end.
//   * g (columns per codebook group) is a runtime parameter: GC = g/32 chunks.
//
// NO global memory is allocated or written except the output tensor(s); dequant
// never materializes — everything happens in registers/shared (accountability
// protocol of 00_OBJECTIVE_AND_REQUIREMENTS.md).
//
// Debug entry: same accumulation, but folds the exact int32 S/N per (row,group)
// bucket into global int32 tensors [R,NG,3,4] via atomicAdd, for exact-integer
// comparison against a reference. (Perf-irrelevant path.)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DPK_WARPS 8

// Compress bit t of each of the 8 nibbles of u into contiguous bits 0..7.
// (bits t, t+4, ..., t+28  ->  bits 0..7)
__device__ __forceinline__ uint32_t nib_bit_compress(uint32_t u, int t) {
    uint32_t v = (u >> t) & 0x11111111u;   // bit of nibble n at position 4n
    v = (v | (v >> 3)) & 0x03030303u;      // 2 bits per byte
    v = (v | (v >> 6)) & 0x000F000Fu;      // 4 bits per halfword
    v = (v | (v >> 12)) & 0x000000FFu;     // 8 contiguous bits
    return v;
}

// u32 holding two consecutive bf16 codebook entries -> float2 (x = low half)
__device__ __forceinline__ float2 bf16x2_to_f2(uint32_t u) {
    __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162 *>(&u);
    return __bfloat1622float2(h);
}

// Fold the 12 (S,N) accumulators of one codebook group into the scalar fp32 y
// and reset them. cbg points at cb[row][grp][0][0] (12 contiguous bf16 = 24 B,
// 8-byte aligned since 24 | offset). int->float conversions are exact:
// |S| <= 15*32768 < 2^24, |8N| <= 8*32768 < 2^24.
__device__ __forceinline__ void fold_group(float &yacc, int *S, int *N,
                                           const __nv_bfloat16 *cbg) {
    const uint2 *c2 = reinterpret_cast<const uint2 *>(cbg);
    uint2 u0 = c2[0], u1 = c2[1], u2 = c2[2];
    uint32_t w[6] = {u0.x, u0.y, u1.x, u1.y, u2.x, u2.y};
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        float2 c = bf16x2_to_f2(w[i]);
        yacc += c.x * (float)(S[2 * i] - 8 * N[2 * i]);
        yacc += c.y * (float)(S[2 * i + 1] - 8 * N[2 * i + 1]);
        S[2 * i] = 0;      N[2 * i] = 0;
        S[2 * i + 1] = 0;  N[2 * i + 1] = 0;
    }
}

// Debug fold: exact int32 S/N per (row, group, partition, level) via atomicAdd.
__device__ __forceinline__ void fold_debug(int row, int NG, int grp, int *S, int *N,
                                           int32_t *S_out, int32_t *N_out) {
    size_t base = ((size_t)row * NG + grp) * 12;
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        if (S[i]) atomicAdd(S_out + base + i, S[i]);
        if (N[i]) atomicAdd(N_out + base + i, N[i]);
        S[i] = 0;
        N[i] = 0;
    }
}

template <bool DEBUG>
__global__ void __launch_bounds__(32 * DPK_WARPS) dpk_gemv_kernel(
    const uint32_t *__restrict__ b0p,   // [R, CW] LSB of code
    const uint32_t *__restrict__ b1p,   // [R, CW] MSB of code
    const uint32_t *__restrict__ mp,    // [R, CW] 1 = tail P2 (don't-care at salient)
    const uint32_t *__restrict__ sp,    // [CW]    1 = salient column P3
    const __nv_bfloat16 *__restrict__ cb,  // [R, NG, 3, 4]
    const uint32_t *__restrict__ xhat,  // [4*CW] unsigned nibbles
    float a_s,                          // fp32 per-tensor activation scale
    const float *__restrict__ a_s_ptr,  // optional device scale (overrides a_s;
                                        // used by the dpk_matmul M=1 dispatch)
    int R, int CW, int NG, int GC,      // GC = g / 32 (chunks per group)
    __nv_bfloat16 *__restrict__ y_bf16, // [R] output (nullptr in debug / fp32 mode)
    float *__restrict__ y_f32,          // [R] optional fp32 output (gate diagnostics)
    int32_t *__restrict__ S_out,        // [R, NG, 3, 4] debug only
    int32_t *__restrict__ N_out)        // [R, NG, 3, 4] debug only
{
    extern __shared__ uint32_t smem[];
    uint32_t *xp0 = smem;            // activation bit-plane t=0
    uint32_t *xp1 = xp0 + CW;
    uint32_t *xp2 = xp1 + CW;
    uint32_t *xp3 = xp2 + CW;
    uint32_t *ssh = xp3 + CW;        // salient bitmap

    // ---- stage x-hat bit-planes + s bitmap (once per CTA) ----
    for (int wc = threadIdx.x; wc < CW; wc += blockDim.x) {
        uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {   // 4 nibble words cover one 32-col chunk
            uint32_t u = xhat[4 * wc + q];
            r0 |= nib_bit_compress(u, 0) << (8 * q);
            r1 |= nib_bit_compress(u, 1) << (8 * q);
            r2 |= nib_bit_compress(u, 2) << (8 * q);
            r3 |= nib_bit_compress(u, 3) << (8 * q);
        }
        xp0[wc] = r0;
        xp1[wc] = r1;
        xp2[wc] = r2;
        xp3[wc] = r3;
        ssh[wc] = sp[wc];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    for (int row = blockIdx.x * DPK_WARPS + wid; row < R;
         row += gridDim.x * DPK_WARPS) {
        const uint32_t *b0r = b0p + (size_t)row * CW;
        const uint32_t *b1r = b1p + (size_t)row * CW;
        const uint32_t *mr = mp + (size_t)row * CW;

        float yacc = 0.0f;
        int S[12], N[12];
#pragma unroll
        for (int i = 0; i < 12; ++i) { S[i] = 0; N[i] = 0; }
        int cur_grp = -1;

        for (int chunk = lane; chunk < CW; chunk += 32) {
            int grp = chunk / GC;
            if (grp != cur_grp) {           // group-aligned: fold before switching
                if (cur_grp >= 0) {
                    if (DEBUG)
                        fold_debug(row, NG, cur_grp, S, N, S_out, N_out);
                    else
                        fold_group(yacc, S, N, cb + ((size_t)row * NG + cur_grp) * 12);
                }
                cur_grp = grp;
            }
            uint32_t w0 = __ldg(b0r + chunk);
            uint32_t w1 = __ldg(b1r + chunk);
            uint32_t wm = __ldg(mr + chunk);
            uint32_t ws = ssh[chunk];
            uint32_t x0 = xp0[chunk], x1 = xp1[chunk];
            uint32_t x2 = xp2[chunk], x3 = xp3[chunk];

            // level masks (doc 02 §7, LOP3-friendly)
            uint32_t q0m = ~(w0 | w1);
            uint32_t q1m = w0 & ~w1;
            uint32_t q2m = w1 & ~w0;
            uint32_t q3m = w0 & w1;
            // partition masks
            uint32_t p1m = ~wm & ~ws;
            uint32_t p2m = wm & ~ws;
            uint32_t p3m = ws;

            // bucket index = p*4 + k  (matches cb[..][p][k] layout)
#define DPK_ACC(idx, mexpr)                                                     \
    {                                                                           \
        const uint32_t mk = (mexpr);                                            \
        N[idx] += __popc(mk);                                                   \
        S[idx] += __popc(x0 & mk) + (__popc(x1 & mk) << 1) +                    \
                  (__popc(x2 & mk) << 2) + (__popc(x3 & mk) << 3);              \
    }
            DPK_ACC(0, q0m & p1m)
            DPK_ACC(1, q1m & p1m)
            DPK_ACC(2, q2m & p1m)
            DPK_ACC(3, q3m & p1m)
            DPK_ACC(4, q0m & p2m)
            DPK_ACC(5, q1m & p2m)
            DPK_ACC(6, q2m & p2m)
            DPK_ACC(7, q3m & p2m)
            DPK_ACC(8, q0m & p3m)
            DPK_ACC(9, q1m & p3m)
            DPK_ACC(10, q2m & p3m)
            DPK_ACC(11, q3m & p3m)
#undef DPK_ACC
        }
        if (cur_grp >= 0) {                 // trailing fold
            if (DEBUG)
                fold_debug(row, NG, cur_grp, S, N, S_out, N_out);
            else
                fold_group(yacc, S, N, cb + ((size_t)row * NG + cur_grp) * 12);
        }

        if (!DEBUG) {
            // single warp reduction of the scalar y (deterministic shuffle tree)
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                yacc += __shfl_down_sync(0xffffffffu, yacc, off);
            if (lane == 0) {
                float out = (a_s_ptr ? __ldg(a_s_ptr) : a_s) * yacc;
                if (y_f32)
                    y_f32[row] = out;
                else
                    y_bf16[row] = __float2bfloat16(out);
            }
        }
    }
}

// -------- host launchers (no allocations, no synchronization here) --------

void dpk_gemv_launch(const uint32_t *b0, const uint32_t *b1, const uint32_t *m,
                     const uint32_t *s, const void *cb, const uint32_t *xhat,
                     float a_s, const float *a_s_ptr, int R, int CW, int NG,
                     int GC, void *y_bf16, float *y_f32, cudaStream_t stream) {
    dim3 block(32 * DPK_WARPS);
    int need = (R + DPK_WARPS - 1) / DPK_WARPS;
    int grid = need < 65535 ? need : 65535;   // grid-stride covers the rest
    size_t shmem = (size_t)5 * CW * sizeof(uint32_t);
    dpk_gemv_kernel<false><<<grid, block, shmem, stream>>>(
        b0, b1, m, s, reinterpret_cast<const __nv_bfloat16 *>(cb), xhat, a_s,
        a_s_ptr, R, CW, NG, GC, reinterpret_cast<__nv_bfloat16 *>(y_bf16),
        y_f32, nullptr, nullptr);
}

void dpk_gemv_debug_launch(const uint32_t *b0, const uint32_t *b1,
                           const uint32_t *m, const uint32_t *s, const void *cb,
                           const uint32_t *xhat, float a_s, int R, int CW,
                           int NG, int GC, int32_t *S_out, int32_t *N_out,
                           cudaStream_t stream) {
    dim3 block(32 * DPK_WARPS);
    int need = (R + DPK_WARPS - 1) / DPK_WARPS;
    int grid = need < 65535 ? need : 65535;
    size_t shmem = (size_t)5 * CW * sizeof(uint32_t);
    dpk_gemv_kernel<true><<<grid, block, shmem, stream>>>(
        b0, b1, m, s, reinterpret_cast<const __nv_bfloat16 *>(cb), xhat, a_s,
        nullptr, R, CW, NG, GC, nullptr, nullptr, S_out, N_out);
}
