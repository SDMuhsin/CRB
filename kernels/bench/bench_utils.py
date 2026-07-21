"""Project-wide measurement contract for the DOML CUDA kernel project.

Every kernel benchmark in this project (INT4 baselines, DOML W2A4 kernels,
end-to-end model runs) MUST use these helpers so that numbers are comparable
across milestones. See llmdocs/cuda_kernel/00_OBJECTIVE_AND_REQUIREMENTS.md:
Requirement 1 is checked with torch.cuda.max_memory_allocated(); this module
is the single implementation of that check.

Conventions
-----------
* All measurements are taken on the tensors' device (must be a CUDA device).
* "peak delta" = (max_memory_allocated during fn) - (memory_allocated just
  before fn). It counts every allocation fn makes -- output tensors AND
  temporaries -- but not pre-existing allocations (weights, inputs). This is
  the quantity Requirement 1 compares between kernels, together with
  weight_bytes() of the persistent weight artifacts.
* Latency uses CUDA events, never wall-clock.
"""

from __future__ import annotations

import torch

__all__ = ["measure_peak", "weight_bytes", "rel_err", "time_kernel"]


def measure_peak(fn, *args, device=None, **kwargs):
    """Run ``fn(*args, **kwargs)`` and measure its peak CUDA memory delta.

    Semantics (the project-wide contract):

    1. ``torch.cuda.synchronize(device)`` -- flush pending work so prior
       allocations/frees are settled.
    2. ``baseline = torch.cuda.memory_allocated(device)`` -- bytes held by
       live tensors *before* fn (weights, inputs, caches).
    3. ``torch.cuda.reset_peak_memory_stats(device)`` -- so the recorded peak
       reflects only what happens during fn.
    4. run fn, then ``torch.cuda.synchronize(device)`` again (kernels are
       async; the allocator peak is only final after sync).
    5. ``peak_delta = torch.cuda.max_memory_allocated(device) - baseline``.

    Returns ``(result, peak_bytes_delta, baseline_allocated_bytes)``.

    ``peak_bytes_delta`` is the extra allocator high-water mark attributable
    to fn: output tensors + all temporaries/workspaces allocated through the
    PyTorch caching allocator. It does NOT include memory allocated outside
    the caching allocator (e.g. cudaMalloc inside a third-party lib); for
    milestones, cross-check with nvidia-smi when a kernel is suspected of
    side-allocations.

    Note: ``peak_delta`` can be 0 for an op that writes into preallocated
    buffers, and is never negative (peak >= current at reset time).
    """
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = fn(*args, **kwargs)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return result, peak - baseline, baseline


def weight_bytes(*tensors) -> int:
    """Total storage bytes of the given tensors: sum(nelement * element_size).

    Use this for weight-footprint accounting: pass *every* persistent
    artifact the kernel needs at inference time (packed weights, scales,
    codebooks, masks, per-layer workspaces...). bpw is then
    ``weight_bytes(...) * 8 / (K * N)`` for a K x N layer.

    Accepts tensors and/or (nested) lists/tuples of tensors.
    """
    total = 0
    stack = list(tensors)
    while stack:
        t = stack.pop()
        if isinstance(t, (list, tuple)):
            stack.extend(t)
        elif isinstance(t, torch.Tensor):
            total += t.nelement() * t.element_size()
        else:
            raise TypeError(f"weight_bytes: expected Tensor or list/tuple, got {type(t)}")
    return total


def rel_err(out: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6):
    """Correctness-gate metrics between a kernel output and its reference.

    Both tensors are flattened and compared in float32 (upcast) so fp16
    rounding of the *comparison itself* does not pollute the metric.

    Returns a dict:
      max_rel  : max_i |out_i - ref_i| / (|ref_i| + eps)   (element-wise; eps
                 guards zero entries -- for entries where |ref| ~ 0 this term
                 degenerates to |out|/eps, so read max_rel together with
                 mean_rel and cos_sim rather than in isolation)
      mean_rel : mean of the same element-wise ratio
      rms_rel  : ||out - ref||_2 / (||ref||_2 + eps)  (global, scale-robust;
                 the most stable single number for GEMM correctness)
      cos_sim  : cosine similarity of the flattened tensors
    """
    if out.shape != ref.shape:
        raise ValueError(f"shape mismatch: out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    o = out.detach().reshape(-1).float()
    r = ref.detach().reshape(-1).float()
    diff = (o - r).abs()
    denom = r.abs() + eps
    ratio = diff / denom
    cos = torch.nn.functional.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item()
    return {
        "max_rel": ratio.max().item(),
        "mean_rel": ratio.mean().item(),
        "rms_rel": (diff.pow(2).sum().sqrt() / (r.pow(2).sum().sqrt() + eps)).item(),
        "cos_sim": cos,
    }


def time_kernel(fn, iters: int = 100, warmup: int = 25, device=None) -> dict:
    """CUDA-event-based latency of ``fn()`` (fn takes no args; close over them).

    Runs ``warmup`` untimed calls, synchronizes, then times ``iters`` calls
    individually with cudaEvent pairs (elapsed_time, milliseconds).

    Returns a dict with mean/median/min/max/p90 latency in **milliseconds**
    plus the raw per-iteration list. Use ``median_ms`` as the headline number
    (robust to clock-boost transients); report ``min_ms`` as best-case.
    """
    if device is None:
        device = torch.cuda.current_device()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize(device)
    times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    ts = sorted(times)
    n = len(ts)
    return {
        "mean_ms": sum(ts) / n,
        "median_ms": ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2]),
        "min_ms": ts[0],
        "max_ms": ts[-1],
        "p90_ms": ts[min(n - 1, int(round(0.9 * (n - 1))))],
        "iters": iters,
        "warmup": warmup,
        "raw_ms": times,
    }
