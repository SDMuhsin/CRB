"""SDOML toy convergence trace.

Verifies the four pass criteria from the S2 contract on a synthetic
[R=64, N=1024] Gaussian row-block with uniform col_weights:

    (a) SDOML full        — n_iter=20 alternation
    (b) SDOML-1pass       — n_iter=1 (single Step A + Step B)
    (c) Magnitude-prune-then-LMQ — forbidden baseline (prune-first)
    (d) Quantize-then-prune-by-mu — forbidden baseline (quantize-first)

Pass criteria:
    Phi(SDOML full)  <=  Phi(SDOML 1pass)
    Phi(SDOML full)  <   Phi(magnitude-prune-then-LMQ)
    Phi(SDOML full)  <   Phi(quantize-then-prune)
    Per-iter trace monotone non-increasing for >= 90% of iterations
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

# Make sure we can import binary.py from the repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from binary import sdoml_quantize, lloyd_max_quantize  # noqa: E402


# -------------------- helpers --------------------

def hessian_weighted_phi(W, W_q, mask, col_weights):
    """Phi = sum_i w_i * (x_i - m_i * q_i)^2 over the full row-block."""
    recon = mask.float() * W_q if mask.dtype == torch.bool else W_q
    # If W_q is already pre-zeroed at pruned positions, recon == W_q.
    err = W.float() - W_q.float()
    cw = col_weights.float().unsqueeze(0)
    return (cw * err * err).sum().item()


def sdoml_full(W, col_weights, sparsity, K=4, n_iter=20):
    """SDOML alternation, return (W_q, mask, codebook, phi_trace)."""
    return sdoml_quantize(W, col_weights, sparsity=sparsity, K=K,
                          n_iter=n_iter, init="quantile", return_aux=True)


def sdoml_1pass(W, col_weights, sparsity, K=4):
    """SDOML with n_iter=1 — single Step A + Step B after init."""
    return sdoml_quantize(W, col_weights, sparsity=sparsity, K=K,
                          n_iter=1, init="quantile", return_aux=True)


def magnitude_prune_then_lmq(W, col_weights, sparsity, K=4, lmq_iters=20):
    """Forbidden baseline: prune by w_i * x_i^2, then Lloyd-Max on survivors.

    Step 1: drop n_drop = sparsity * N smallest |x_i| * sqrt(w_i) per row.
            (This is the standard SparseGPT / magnitude-pruning heuristic.)
    Step 2: run DOML's `lloyd_max_quantize` on the surviving entries.
    """
    R, N = W.shape
    n_keep = int((1.0 - sparsity) * N)
    n_keep = max(1, min(N, n_keep))
    sqrt_w = col_weights.to(W.device).to(W.dtype).sqrt().unsqueeze(0)
    score = W.abs() * sqrt_w
    _, keep_idx = score.topk(n_keep, dim=1, largest=True)
    mask = torch.zeros(R, N, dtype=torch.bool, device=W.device)
    mask.scatter_(1, keep_idx, True)
    # Lloyd-Max on the *survivors only* with K levels.
    W_q = lloyd_max_quantize(W, mask, K=K, iters=lmq_iters)
    return W_q, mask


def quantize_then_prune(W, col_weights, sparsity, K=4, lmq_iters=20):
    """Forbidden baseline: DOML K=4 Lloyd-Max first, then prune by mu_i.

    Step 1: lloyd_max_quantize over all weights with all-True mask.
    Step 2: keep top n_keep by mu_i = w_i * (x_i^2 - (x_i - c_assigned)^2).
    """
    R, N = W.shape
    n_keep = int((1.0 - sparsity) * N)
    n_keep = max(1, min(N, n_keep))
    full_mask = torch.ones(R, N, dtype=torch.bool, device=W.device)
    W_q_full = lloyd_max_quantize(W, full_mask, K=K, iters=lmq_iters)
    # Compute mu_i using the assigned centroid value (a_assigned = W_q_full).
    cw = col_weights.to(W.device).to(W.dtype).unsqueeze(0)
    diff = W - W_q_full
    mu = cw * (W * W - diff * diff)
    _, keep_idx = mu.topk(n_keep, dim=1, largest=True)
    mask = torch.zeros(R, N, dtype=torch.bool, device=W.device)
    mask.scatter_(1, keep_idx, True)
    # Apply mask: pruned positions become 0.
    W_q = W_q_full * mask.float()
    return W_q, mask


def bpw_bitmap(R, N, K, sparsity):
    """Effective bits-per-weight under bitmap mask encoding.

    bpw = (K * 16) / N  +  1  +  (1 - sparsity) * log2(K)
        ^ codebook       ^ bitmap   ^ index per kept weight
    """
    return (K * 16.0) / N + 1.0 + (1.0 - sparsity) * math.log2(K)


def fmt_phi(p):
    return f"{p:14.4f}"


# -------------------- main toy --------------------

def main():
    torch.manual_seed(42)
    R, N = 64, 1024
    K = 4
    n_iter = 20
    device = "cpu"

    print(f"\nSDOML toy convergence — R={R}, N={N}, K={K}, "
          f"n_iter={n_iter}, seed=42")
    print("=" * 88)

    W = torch.randn(R, N, dtype=torch.float32, device=device)
    col_weights = torch.ones(N, dtype=torch.float32, device=device)

    sparsities = [0.2, 0.5]

    rows = []
    traces = {}

    pass_results = {}

    for s in sparsities:
        n_keep = int((1.0 - s) * N)
        bpw = bpw_bitmap(R, N, K, s)

        # (a) SDOML full
        t0 = time.time()
        W_q_a, mask_a, cb_a, trace_a = sdoml_full(W, col_weights, s, K=K,
                                                   n_iter=n_iter)
        t_a = time.time() - t0
        phi_a = trace_a[-1].item()

        # (b) SDOML-1pass
        t0 = time.time()
        W_q_b, mask_b, cb_b, trace_b = sdoml_1pass(W, col_weights, s, K=K)
        t_b = time.time() - t0
        phi_b = trace_b[-1].item()

        # (c) Magnitude-prune-then-LMQ
        t0 = time.time()
        W_q_c, mask_c = magnitude_prune_then_lmq(W, col_weights, s, K=K)
        t_c = time.time() - t0
        phi_c = hessian_weighted_phi(W, W_q_c, mask_c, col_weights)

        # (d) Quantize-then-prune
        t0 = time.time()
        W_q_d, mask_d = quantize_then_prune(W, col_weights, s, K=K)
        t_d = time.time() - t0
        phi_d = hessian_weighted_phi(W, W_q_d, mask_d, col_weights)

        traces[s] = trace_a.cpu().numpy()

        rows.append((s, "SDOML full",        phi_a, bpw, t_a, n_keep))
        rows.append((s, "SDOML 1pass",       phi_b, bpw, t_b, n_keep))
        rows.append((s, "Magnitude->LMQ",    phi_c, bpw, t_c, n_keep))
        rows.append((s, "Quantize->prune",   phi_d, bpw, t_d, n_keep))

        # Pass criteria.
        pc1 = phi_a <= phi_b + 1e-6
        pc2 = phi_a < phi_c
        pc3 = phi_a < phi_d
        # Monotone trace check.
        diffs = np.diff(trace_a.cpu().numpy())
        non_inc = (diffs <= max(1e-5, 1e-6 * abs(phi_a))).sum()
        total_diffs = len(diffs)
        pc4 = (non_inc / total_diffs) >= 0.9

        pass_results[s] = {
            "(a) SDOML full <= (b) SDOML 1pass":    (pc1, phi_a, phi_b),
            "(a) SDOML full <  (c) magnitude->LMQ": (pc2, phi_a, phi_c),
            "(a) SDOML full <  (d) quant->prune":   (pc3, phi_a, phi_d),
            "trace >= 90% non-increasing":           (pc4,
                                                       float(non_inc),
                                                       float(total_diffs)),
        }

    # ---- print stdout table ----
    print(f"{'s':>4}  {'method':<22}  {'Phi':>14}  "
          f"{'bpw_bitmap':>10}  {'wall_s':>7}  {'n_keep':>6}")
    print("-" * 88)
    for s, name, phi, bpw, t, nk in rows:
        print(f"{s:>4.2f}  {name:<22}  {fmt_phi(phi)}  "
              f"{bpw:>10.4f}  {t:>7.3f}  {nk:>6d}")

    print("-" * 88)
    print("\nPass-criteria check")
    print("-" * 88)
    all_pass = True
    for s, results in pass_results.items():
        print(f"\n  s = {s:.2f}")
        for label, (passed, a, b) in results.items():
            mark = "PASS" if passed else "FAIL"
            print(f"    [{mark}] {label}   ({a:.4f}, {b:.4f})")
            if not passed:
                all_pass = False

    print("\n" + "=" * 88)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
    print("=" * 88)

    # ---- save trace ----
    out_dir = os.path.join(_REPO, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sdoml_toy_convergence.npz")
    np.savez(
        out_path,
        sparsity_02_phi_trace=traces[0.2],
        sparsity_05_phi_trace=traces[0.5],
        R=np.array(R), N=np.array(N), K=np.array(K),
        n_iter=np.array(n_iter), seed=np.array(42),
    )
    print(f"\nSaved per-iter Phi traces to {out_path}")
    print(f"  s=0.20: trace shape {traces[0.2].shape}")
    print(f"  s=0.50: trace shape {traces[0.5].shape}")

    # Print s=0.5 trace explicitly (21 numbers).
    print(f"\nPer-iter Phi trace for SDOML-full at s=0.50 ({len(traces[0.5])} values):")
    for i, v in enumerate(traces[0.5]):
        print(f"  iter {i:2d}: {v:.6f}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
