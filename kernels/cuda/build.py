"""Cached JIT build of the DPK CUDA kernels (torch.utils.cpp_extension).

Usage:
    from build import build_dpk          # (kernels/cuda on sys.path)
    ext = build_dpk()
    y = ext.dpk_gemv(b0, b1, m, s, cb, xhat, a_s, g)

Build artifacts are cached in kernels/cuda/build/ (ninja skips recompilation
when sources are unchanged). Target arch: SM86 (A40), -O3. No fast-math: the
kernel's float math is plain FMA/adds and must stay IEEE for the documented
tolerance analysis.
"""

import os

from torch.utils.cpp_extension import load

_THIS = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(_THIS, "build")


def build_dpk(verbose: bool = False):
    os.makedirs(_BUILD, exist_ok=True)
    return load(
        name="dpk_kernels",
        sources=[
            os.path.join(_THIS, "dpk_ext.cpp"),
            os.path.join(_THIS, "dpk_gemv.cu"),
            os.path.join(_THIS, "dpk_gemm.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_86,code=sm_86",
            "-lineinfo",
        ],
        build_directory=_BUILD,
        verbose=verbose,
    )


if __name__ == "__main__":
    ext = build_dpk(verbose=True)
    print("built OK:", ext.__file__)
