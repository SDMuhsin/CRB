// dpk_ext.cpp — torch extension boundary for the DPK W2A4 GEMV kernel.
//
// Exposes:
//   dpk_gemv(b0, b1, m, s, cb, xhat, a_s, g, out_fp32=False) -> y [R]
//       bf16 output by default (the format's layer dtype); out_fp32=True
//       returns the pre-bf16-rounding fp32 accumulator values (used by the
//       correctness gates to isolate summation-order error from bf16
//       rounding — same kernel math, only the final store differs).
//   dpk_gemv_debug(b0, b1, m, s, cb, xhat, a_s, g) -> (S, N) int32 [R, NG, 3, 4]
//       exact integer bucket sums for gate comparison.
//
// The ONLY device allocation made per call is the output tensor(s) — the
// kernel has no workspace (accountability protocol: peak delta == output).
//
// g is an explicit runtime argument (columns per codebook group, multiple of
// 128, ceil(C/g) must equal cb.shape[1]). It cannot be inferred from NG alone:
// e.g. C=2560, NG=4 is satisfied by both g=640 and g=768, which map columns to
// different groups. The DPK container header carries g; callers pass it.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

void dpk_gemv_launch(const uint32_t *b0, const uint32_t *b1, const uint32_t *m,
                     const uint32_t *s, const void *cb, const uint32_t *xhat,
                     float a_s, const float *a_s_ptr, int R, int CW, int NG,
                     int GC, void *y_bf16, float *y_f32, cudaStream_t stream);
void dpk_gemm_launch(const uint32_t *b0, const uint32_t *b1, const uint32_t *m,
                     const uint32_t *s, const void *cb, const uint32_t *xhat,
                     const float *a_s_vec, int M, int R, int CW, int NG,
                     int GC, void *y_bf16, float *y_f32, cudaStream_t stream);
void dpk_gemv_debug_launch(const uint32_t *b0, const uint32_t *b1,
                           const uint32_t *m, const uint32_t *s, const void *cb,
                           const uint32_t *xhat, float a_s, int R, int CW,
                           int NG, int GC, int32_t *S_out, int32_t *N_out,
                           cudaStream_t stream);

namespace {

const uint32_t *u32_ptr(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kUInt32 || t.scalar_type() == at::kInt,
                name, " must be uint32 (or int32 bit-view), got ", t.scalar_type());
    return reinterpret_cast<const uint32_t *>(t.data_ptr());
}

struct Args {
    const uint32_t *b0, *b1, *m, *s, *xhat;
    const void *cb;
    float a_s;
    int64_t R, C, CW, NG, GC;
};

Args validate(const torch::Tensor &b0, const torch::Tensor &b1,
              const torch::Tensor &m, const torch::Tensor &s,
              const torch::Tensor &cb, const torch::Tensor &xhat, double a_s,
              int64_t g) {
    Args a;
    TORCH_CHECK(b0.dim() == 2, "b0 must be [R, C/32]");
    a.R = b0.size(0);
    a.CW = b0.size(1);
    a.C = a.CW * 32;
    TORCH_CHECK(a.R >= 1, "R must be >= 1");
    TORCH_CHECK(a.C % 128 == 0, "C must be a multiple of 128 (padded), got C=", a.C);
    TORCH_CHECK(a.C <= 32768, "C up to 32768 supported in v1, got ", a.C);
    TORCH_CHECK(b1.sizes() == b0.sizes() && m.sizes() == b0.sizes(),
                "b0/b1/m must have identical shapes");
    TORCH_CHECK(s.numel() == a.CW, "s must have C/32 words");
    TORCH_CHECK(xhat.numel() == a.CW * 4, "xhat must have ceil(C/8) = C/8 words");

    TORCH_CHECK(cb.is_cuda() && cb.is_contiguous(), "cb must be CUDA contiguous");
    TORCH_CHECK(cb.scalar_type() == at::kBFloat16,
                "cb must be bf16 (Qwen3 layer dtype; fp16 models are a later variant)");
    TORCH_CHECK(cb.dim() == 4 && cb.size(0) == a.R && cb.size(2) == 3 && cb.size(3) == 4,
                "cb must be [R, NG, 3, 4]");
    a.NG = cb.size(1);

    TORCH_CHECK(g > 0 && g % 128 == 0, "g must be a positive multiple of 128, got ", g);
    TORCH_CHECK((a.C + g - 1) / g == a.NG,
                "ceil(C/g) = ", (a.C + g - 1) / g, " does not match cb NG = ", a.NG);
    a.GC = g / 32;

    a.b0 = u32_ptr(b0, "b0");
    a.b1 = u32_ptr(b1, "b1");
    a.m = u32_ptr(m, "m");
    a.s = u32_ptr(s, "s");
    a.xhat = u32_ptr(xhat, "xhat");
    a.cb = cb.data_ptr();
    a.a_s = static_cast<float>(a_s);

    auto dev = b0.device();
    for (const auto *t : {&b1, &m, &s, &cb, &xhat})
        TORCH_CHECK(t->device() == dev, "all tensors must be on the same device");
    return a;
}

}  // namespace

torch::Tensor dpk_gemv(torch::Tensor b0, torch::Tensor b1, torch::Tensor m,
                       torch::Tensor s, torch::Tensor cb, torch::Tensor xhat,
                       double a_s, int64_t g, bool out_fp32) {
    Args a = validate(b0, b1, m, s, cb, xhat, a_s, g);
    const at::cuda::OptionalCUDAGuard guard(b0.device());
    auto opts = cb.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16);
    auto y = torch::empty({a.R}, opts);   // the ONLY allocation of this call
    dpk_gemv_launch(a.b0, a.b1, a.m, a.s, a.cb, a.xhat, a.a_s, nullptr,
                    (int)a.R, (int)a.CW, (int)a.NG, (int)a.GC,
                    out_fp32 ? nullptr : y.data_ptr(),
                    out_fp32 ? y.data_ptr<float>() : nullptr,
                    at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ---- GEMM (M >= 1): Y[t, r] = a_s_vec[t] * sum_j W[r,j] * (xhat[t,j] - 8) ----

torch::Tensor dpk_gemm(torch::Tensor b0, torch::Tensor b1, torch::Tensor m,
                       torch::Tensor s, torch::Tensor cb, torch::Tensor xhat,
                       torch::Tensor a_s_vec, int64_t g, bool out_fp32) {
    TORCH_CHECK(xhat.dim() == 2, "Xhat must be [M, ceil(C/8)] for dpk_gemm");
    const int64_t M = xhat.size(0);
    TORCH_CHECK(M >= 1, "M must be >= 1");
    // validate() checks xhat.numel() == 4*CW*... only for the 1-D case; do the
    // 2-D checks here, then reuse validate() for everything else.
    TORCH_CHECK(b0.dim() == 2, "b0 must be [R, C/32]");
    const int64_t CW = b0.size(1);
    TORCH_CHECK(xhat.size(1) == CW * 4,
                "Xhat must have C/8 = ", CW * 4, " words per row, got ",
                xhat.size(1));
    TORCH_CHECK(a_s_vec.is_cuda() && a_s_vec.is_contiguous() &&
                a_s_vec.scalar_type() == at::kFloat && a_s_vec.dim() == 1 &&
                a_s_vec.size(0) == M,
                "a_s_vec must be a contiguous fp32 CUDA tensor [M]");
    TORCH_CHECK(a_s_vec.device() == b0.device(),
                "a_s_vec must be on b0's device");
    // validate() expects a single activation row; row 0 shares Xhat's base
    // pointer, and the [M, C/8] shape was already checked above.
    Args a = validate(b0, b1, m, s, cb, xhat.narrow(0, 0, 1).reshape({-1}),
                      0.0, g);
    const at::cuda::OptionalCUDAGuard guard(b0.device());
    auto opts = cb.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16);
    auto y = torch::empty({M, a.R}, opts);  // the ONLY allocation of this call
    dpk_gemm_launch(a.b0, a.b1, a.m, a.s, a.cb, a.xhat,
                    a_s_vec.data_ptr<float>(), (int)M, (int)a.R, (int)a.CW,
                    (int)a.NG, (int)a.GC,
                    out_fp32 ? nullptr : y.data_ptr(),
                    out_fp32 ? y.data_ptr<float>() : nullptr,
                    at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ---- dispatch wrapper: GEMV kernel at M == 1, GEMM tiles otherwise ----

torch::Tensor dpk_matmul(torch::Tensor b0, torch::Tensor b1, torch::Tensor m,
                         torch::Tensor s, torch::Tensor cb, torch::Tensor xhat,
                         torch::Tensor a_s_vec, int64_t g, bool out_fp32) {
    TORCH_CHECK(xhat.dim() == 2, "Xhat must be [M, ceil(C/8)] for dpk_matmul");
    if (xhat.size(0) != 1)
        return dpk_gemm(b0, b1, m, s, cb, xhat, a_s_vec, g, out_fp32);
    TORCH_CHECK(a_s_vec.is_cuda() && a_s_vec.is_contiguous() &&
                a_s_vec.scalar_type() == at::kFloat && a_s_vec.numel() == 1,
                "a_s_vec must be a contiguous fp32 CUDA tensor [1]");
    Args a = validate(b0, b1, m, s, cb, xhat.reshape({-1}), 0.0, g);
    const at::cuda::OptionalCUDAGuard guard(b0.device());
    auto opts = cb.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16);
    auto y = torch::empty({1, a.R}, opts);  // the ONLY allocation of this call
    dpk_gemv_launch(a.b0, a.b1, a.m, a.s, a.cb, a.xhat, /*a_s=*/1.0f,
                    a_s_vec.data_ptr<float>(), (int)a.R, (int)a.CW, (int)a.NG,
                    (int)a.GC, out_fp32 ? nullptr : y.data_ptr(),
                    out_fp32 ? y.data_ptr<float>() : nullptr,
                    at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::vector<torch::Tensor> dpk_gemv_debug(torch::Tensor b0, torch::Tensor b1,
                                          torch::Tensor m, torch::Tensor s,
                                          torch::Tensor cb, torch::Tensor xhat,
                                          double a_s, int64_t g) {
    Args a = validate(b0, b1, m, s, cb, xhat, a_s, g);
    const at::cuda::OptionalCUDAGuard guard(b0.device());
    auto opts = b0.options().dtype(at::kInt);
    auto S = torch::zeros({a.R, a.NG, 3, 4}, opts);
    auto N = torch::zeros({a.R, a.NG, 3, 4}, opts);
    dpk_gemv_debug_launch(a.b0, a.b1, a.m, a.s, a.cb, a.xhat, a.a_s, (int)a.R,
                          (int)a.CW, (int)a.NG, (int)a.GC, S.data_ptr<int32_t>(),
                          N.data_ptr<int32_t>(),
                          at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {S, N};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("dpk_gemv", &dpk_gemv,
            "DPK W2A4 bucket-popcount GEMV (M=1): y[R] = a_s * sum cb*(S-8N)",
            py::arg("b0"), py::arg("b1"), py::arg("m"), py::arg("s"),
            py::arg("cb"), py::arg("xhat"), py::arg("a_s"), py::arg("g"),
            py::arg("out_fp32") = false);
    mod.def("dpk_gemv_debug", &dpk_gemv_debug,
            "Debug: exact int32 bucket sums (S, N), each [R, NG, 3, 4]",
            py::arg("b0"), py::arg("b1"), py::arg("m"), py::arg("s"),
            py::arg("cb"), py::arg("xhat"), py::arg("a_s"), py::arg("g"));
    mod.def("dpk_gemm", &dpk_gemm,
            "DPK W2A4 tensor-core GEMM: Y[M,R] = diag(a_s_vec) * (X-8) @ W^T",
            py::arg("b0"), py::arg("b1"), py::arg("m"), py::arg("s"),
            py::arg("cb"), py::arg("xhat"), py::arg("a_s_vec"), py::arg("g"),
            py::arg("out_fp32") = false);
    mod.def("dpk_matmul", &dpk_matmul,
            "Dispatch: GEMV kernel at M == 1, GEMM tiles otherwise; Y [M, R]",
            py::arg("b0"), py::arg("b1"), py::arg("m"), py::arg("s"),
            py::arg("cb"), py::arg("xhat"), py::arg("a_s_vec"), py::arg("g"),
            py::arg("out_fp32") = false);
}
