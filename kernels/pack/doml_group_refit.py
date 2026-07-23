"""K2.5 deliverable — DOML group-refit quantizer harness (PPL-vs-g study).

Re-runs the real DOML quantization (Qwen/Qwen3-0.6B wikitext2, same args as the
K2 dump run) with codebook group size g = k*128 as the ONLY change:

  * At each GROUP start (column index multiple of g), on the CURRENT W (which
    includes GPTQ feedback from all previous groups):
      - the per-128-block 3-way masks are computed for ALL k blocks of the
        group via the UNMODIFIED `structural_guassian_distribution`;
      - ONE codebook per (row, partition) is fit over the group's union of
        that partition's elements via the UNMODIFIED `lloyd_max_quantize`
        on the [R, g] slice (same K=4, iters=20).
  * The k blocks are then processed exactly as the original bigptq legacy
    loop (bigptq.py:491-520): pre-quantize each block by snapping to the
    FROZEN group codebook per element's partition, column sweep (W1 is never
    mutated inside a block, faithful to the legacy path), inter-block
    feedback `W[:, col_ed:] -= Err1 @ Hinv[col_st:col_ed, col_ed:]` unchanged
    — including feedback INTO later blocks of the same group before they are
    snapped.

  KNOWN, DOCUMENTED DEVIATION vs the original at g > 128: later blocks' masks
  are computed from group-start W instead of feedback-updated W; codebooks
  are frozen at group start. At g = 128 the procedure is IDENTICAL to the
  original (gate G0: bitwise weight equality vs K2's sa-g128 dumps + PPL
  31.0392 reproduced).

Implementation notes (bitwise-fidelity critical):
  * `bigptq.BRAGPTQ.fasterquant` is monkey-patched (NO repo source modified)
    with a reimplementation of the doml/partition=3 branch; every other
    method/partition delegates to the original.
  * Block 0 of every group uses the `lloyd_max_quantize` reconstruction
    directly (its final step IS the nearest-level snap) — at g=128 every
    block is block 0, so the whole path reduces to the original ops.
  * The internal sorted `levels` [R, 4] are captured via a scoped
    torch.Tensor.gather hook (the only .gather inside lloyd_max_quantize is
    `levels.gather(1, assignments)`); the hook calls the original gather with
    unchanged args (non-perturbing). Per group x partition we HARD-ASSERT
    that `_snap_to_levels(Wg, mask, levels)` equals the lloyd reconstruction
    bitwise — proving the snap function replicates lloyd's own final
    assignment before it is ever used on later blocks.

K2.6 EXTENSIONS (membership-axis study; same gate discipline):

  * --mmode column (E1 "S-C128" / E3 "S-C256"): the per-128-block bulk/tail
    split of the NON-salient columns becomes COLUMN-wise. mask3 (salient) is
    EXACTLY the unmodified `structural_searching` output per block. The
    remaining columns are split by per-column score c_j = mean_i |W[i,j]|
    (block-local, non-salient columns only); candidate thresholds = the 81
    quantiles (0.10..0.90) of {c_j}; each threshold is scored with the SAME
    proxy objective as utils/autosearch.py lines 70-80 (high_order_residual
    order=1 on bulk and tail + the order-2 salient group3, total MSE), with
    the column masks expanded to all rows; argmin threshold wins. Degenerate
    guard: empty tail or bulk -> median split of {c_j} (-> index split if
    still degenerate; counted in the manifest). Codebooks/GPTQ unchanged.
    Containers are emitted with mmode=column (colmem stream, NO m plane).
  * --fresh-masks (E2): at g > 128 the per-128-block masks used for SNAPPING
    (and stored) are recomputed at block time on the CURRENT feedback-updated
    W — exactly the original DOML mask timing; ONLY the codebook fit at group
    start uses the group-start masks/weights. At g=128 the flag is inert by
    construction (every block is block 0), so the g=128 gate run must stay
    bitwise-identical to the original artifacts.

Usage:
  synthetic selftest (refit@128 == original bitwise; g=256/global round-trip;
  fresh-masks property gates; column-mmode round-trip + colmem decode):
      CUDA_VISIBLE_DEVICES=1 python kernels/pack/doml_group_refit.py --selftest
  gate G0 (g=128 refit must reproduce K2's dump bitwise + PPL 31.0392):
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/doml_group_refit.py --run \
          --g 128 --gate-dir downloads/doml_dumps/qwen3-0.6b/sa-g128
  refit runs (add --dump-dir to emit DPK containers + wq ground truth):
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/doml_group_refit.py --run --g 256
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/doml_group_refit.py --run --g 512 \
          --dump-dir downloads/doml_dumps/qwen3-0.6b/refit-g512
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/doml_group_refit.py --run --g global \
          --dump-dir downloads/doml_dumps/qwen3-0.6b/refit-gC
  K2.6 runs:
      # E1 S-C128 (column membership, per-block codebooks) + dump
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/doml_group_refit.py --run \
          --g 128 --mmode column --tag sc128 \
          --dump-dir downloads/doml_dumps/qwen3-0.6b/sc128
      # E2 hard gate (fresh-masks plumbing must not disturb g=128)
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/doml_group_refit.py --run \
          --g 128 --fresh-masks --tag fresh-g128-gate \
          --gate-dir downloads/doml_dumps/qwen3-0.6b/sa-g128
      # E2 fresh-masks g=256 (stats only)
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/doml_group_refit.py --run \
          --g 256 --fresh-masks --stats-only --tag fresh-g256
"""

import argparse
import json
import math
import os
import sys
import time

REPO = "/workspace/BiLLM2"
VERIFY_DIR = os.path.join(REPO, "llmdocs", "cuda_kernel", "verify")

# Must be set before run.py / csv_utils are imported (V1/K2-proven redirect).
os.environ.setdefault(
    "BILLM_BENCH_CSV", os.path.join(VERIFY_DIR, "scratch_results.csv"))

if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import transformers  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import bigptq  # noqa: E402  (binds structural_guassian_distribution at import)
from binary import lloyd_max_quantize, Binarization  # noqa: E402
from binary import high_order_residual  # noqa: E402  (K2.6 column search)
from utils.autosearch import structural_searching  # noqa: E402  (K2.6)
from doml_dump import derive_dpk, container_meta  # noqa: E402  (K2 packer)

B_BLOCK = 128
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MODEL_NAME = DEFAULT_MODEL          # retargeted by set_model() (H17-A)


def _run_argv(model_name):
    return [
        "run.py", model_name, "wikitext2", "doml",
        "--blocksize", "128", "--salient_metric", "magnitude",
        "--device", "cuda:0",
    ]


RUN_ARGV = _run_argv(MODEL_NAME)


def _model_n_layers(model_name):
    """num_hidden_layers from the cached HF config (no weights load)."""
    from transformers import AutoConfig
    cache_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")
    return AutoConfig.from_pretrained(
        model_name, cache_dir=cache_dir).num_hidden_layers


def set_model(model_name):
    """H17-A: retarget the harness at `model_name` and rebuild RUN_ARGV.
    Default (0.6B) keeps the pre-flag behavior identical. 2026-07-20:
    EXPECTED_SUBLAYERS is derived from the model config (7 linears per
    decoder block) — 0.6B and 1.7B both have 28 blocks, so the old 196
    constant was a coincidence that 4B (36 blocks) breaks."""
    global MODEL_NAME, RUN_ARGV, EXPECTED_SUBLAYERS
    MODEL_NAME = model_name
    RUN_ARGV = _run_argv(model_name)
    EXPECTED_SUBLAYERS = 7 * _model_n_layers(model_name)


def manifest_model(dump_dir):
    """Model recorded in <dump_dir>/manifest.json: explicit 'model' field,
    with argv[0] (which has always been the model name) as the legacy
    fallback. None if the manifest or both fields are missing."""
    mp = os.path.join(dump_dir, "manifest.json")
    if not os.path.exists(mp):
        return None
    with open(mp) as f:
        man = json.load(f)
    m = man.get("model")
    if not m:
        argv = man.get("argv") or []
        m = argv[0] if argv else None
    return m


EXPECTED_SUBLAYERS = 196

_ORIG_FQ = bigptq.BRAGPTQ.fasterquant  # captured once, used for delegation

# Global run configuration (set by main); None fields disable that feature.
RUN_STATE = {
    "g": None,           # int or "global"
    "gate_dir": None,    # per-layer bitwise vs <gate_dir>/<name>.wq.safetensors
    "dump_dir": None,    # emit DPK containers + wq ground truth
    "stats_only": False,  # run derive_dpk for stats/invariant, discard tensors
    "mmode": "element",  # "element" (m plane) or "column" (colmem; K2.6 E1/E3)
    "fresh_masks": False,  # E2: recompute snap masks at block time (K2.6)
    # K27 quality probes (all OFF by default => default path byte-identical):
    "codebook_dtype": "bf16",  # round codebook levels through this dtype
    "two_pass": "none",        # "global": refit one codebook/(row,part) on W
    "merge_tail": False,       # merge bulk+tail into one non-salient partition
    "partition_k": (4, 4, 4),  # K27 probe 4: per-partition #levels (bulk,tail,salient)
    "cb_weight": "none",       # K28 (B): codebook-fit weighting {none,hdiag,gptq}
    "cb_weight_pow": 1.0,      # K28 (B): exponent p on the weight (w^p)
    "intra_block": False,      # K28 (B): proper intra-block GPTQ error feedback
    "refit_iters": 1,          # K31 (B'): outer joint codebook<->GPTQ iterations
                               #   per group. 1 => current path (byte-identical);
                               #   M>1 requires --intra-block-gptq.
    "bulk_frac": None,         # K30: re-split non-salient into bulk/tail by |W|
                               #      quantile β (None => original split, untouched)
    "rd_split": None,          # K32: λ for the K-aware RATE-aware per-block
                               #   bulk/tail split search (None => original
                               #   split path completely untouched). λ=0 =>
                               #   pure K-aware distortion-optimal split.
    "rd_iters": 8,             # K32: Lloyd iterations for the SEARCH proxy
                               #   fits only (final codebook fit stays 20).
    "bulk_k_map": None,        # K30 mixed-K: {sublayer_name -> bulk K} per-
                               #   sublayer override of partition_k[0]. Names as
                               #   in the manifests ("model.layers.N.mod.proj").
                               #   None (default) => global partition_k for all.
    "manifest": [],
    "n_refit_layers": 0,
    "t0": None,
}


class _GatherCapture:
    """Scoped torch.Tensor.gather hook: records the `self` tensor of every
    gather call in the window, forwarding to the original with unchanged
    args (gather never mutates its inputs -> non-perturbing)."""

    def __enter__(self):
        self.tensors = []
        self._orig = torch.Tensor.gather
        book, orig = self.tensors, self._orig

        def _hook(t, *args, **kwargs):
            book.append(t)
            return orig(t, *args, **kwargs)

        torch.Tensor.gather = _hook
        return self

    def __exit__(self, *exc):
        torch.Tensor.gather = self._orig
        return False


def _snap_to_levels(x, mask, levels):
    """Verbatim replica of lloyd_max_quantize's FINAL assignment step
    (binary.py:3625-3633): nearest-level snap of x*mask to the sorted
    per-row levels, zero where mask is False."""
    masked_x = x * mask.float()
    x_expanded = masked_x.unsqueeze(2)              # [rows, cols, 1]
    levels_expanded = levels.unsqueeze(1)           # [rows, 1, K]
    dists = (x_expanded - levels_expanded) ** 2
    dists = dists + (~mask).unsqueeze(2).float() * 1e30
    assignments = dists.argmin(dim=2)
    return levels.gather(1, assignments) * mask.float()


def _snap_to_levels_assign(x, mask, levels):
    """Identical computation to `_snap_to_levels` but ALSO returns the per-row
    argmin level index used (K31 assignment capture). The returned recon is
    produced by exactly the same ops (masked_x -> squared dists -> argmin ->
    gather -> mask), so substituting this for `_snap_to_levels` on the
    intra-block path is byte-for-byte identical on the reconstruction."""
    masked_x = x * mask.float()
    x_expanded = masked_x.unsqueeze(2)              # [rows, cols, 1]
    levels_expanded = levels.unsqueeze(1)           # [rows, 1, K]
    dists = (x_expanded - levels_expanded) ** 2
    dists = dists + (~mask).unsqueeze(2).float() * 1e30
    assignments = dists.argmin(dim=2)               # [rows, cols]
    recon = levels.gather(1, assignments) * mask.float()
    return recon, assignments


@torch.no_grad()
def _refit_levels_from_assign(Wg, union_masks, assign_full, levels_prev,
                              col_w, part_k):
    """K31 (DIRECTION B') — RE-FIT each partition's codebook levels as the
    (col_w-weighted) mean of the ORIGINAL group-start `Wg` values grouped by
    the assignment indices `assign_full[p]` captured from the PREVIOUS
    intra-block GPTQ sweep. This is the codebook half of the joint
    codebook<->GPTQ alternation: assignments are feedback-aware (from the GPTQ
    sweep), and levels are re-centered on the group-start values under those
    assignments (NOT on the snapped values — those are a fixed point).

    Per (row, partition p) and level k:
        new_level[k] = Σ_j col_w[j]·Wg[row,j]·1[assign=k] /
                       Σ_j col_w[j]·1[assign=k]
    over that partition's columns (masked by union_masks[p]). Empty clusters
    keep the previous level. Levels are kept sorted and fp8-rounded when the
    codebook dtype is fp8. Storage/structure is unchanged (still K_p sorted
    levels per (row,group,partition), nearest-level decode) — only the level
    VALUES move, so packed bpw is identical.
    """
    R, Gcols = Wg.shape
    fp8 = RUN_STATE.get("codebook_dtype", "bf16") != "bf16"
    xf = Wg.to(torch.float32)
    if col_w is None:
        wcol = torch.ones((1, Gcols), device=Wg.device, dtype=torch.float32)
    else:
        wcol = col_w.to(torch.float32).clamp(min=0).view(1, Gcols)
    new_levels = []
    for p in range(3):
        K_p = part_k[p]
        mask = union_masks[p]
        if int(mask.sum().item()) == 0:
            new_levels.append(levels_prev[p])
            continue
        assign = assign_full[p]                          # [R, Gcols] long
        lev = levels_prev[p].to(torch.float32).clone()   # [R, K_p] guard base
        for k in range(K_p):
            km = (assign == k) & mask                    # [R, Gcols]
            wk = wcol * km.float()                        # weighted membership
            k_wsum = wk.sum(dim=1)                         # [R]
            k_xsum = (wk * xf).sum(dim=1)                # [R]
            valid = k_wsum > 0
            lev[valid, k] = k_xsum[valid] / k_wsum[valid]
        lev, _ = lev.sort(dim=1)
        if fp8:
            lev = _maybe_round_levels(lev)
        new_levels.append(lev)
    return new_levels


def _maybe_round_levels(lev):
    """K27 probe 1: optionally snap codebook levels onto the grid of a low-bit
    float dtype. Default 'bf16' => identity (byte-identical default path)."""
    cb = RUN_STATE.get("codebook_dtype", "bf16")
    if cb == "bf16":
        return lev
    fp8 = (torch.float8_e4m3fn if cb == "float8_e4m3fn"
           else torch.float8_e5m2)
    return lev.to(fp8).to(lev.dtype)


@torch.no_grad()
def _weighted_lloyd_max_quantize(x, mask, col_w, K=4, iters=20):
    """K28 (DIRECTION B) — per-row *importance-weighted* Lloyd-Max.

    Minimizes the column-weighted MSE  sum_{i,j} mask*col_w[j]*(x_ij - lvl)^2
    per row instead of the unweighted sum used by `binary.lloyd_max_quantize`.
    The ASSIGNMENT step is unchanged (each weight snaps to its nearest level —
    correct for weighted MSE too); only the CENTROID update becomes a
    col_w-weighted mean. This does NOT change the storage structure at all:
    still K levels per (row, group, partition), still nearest-level decode
    (`_snap_to_levels`), still the same membership planes. Only the *values*
    of the levels move — so the packed bpw is identical to the unweighted
    config at the same (g, codebook_dtype, mmode); only PPL changes.

    `col_w`: [cols] non-negative per-column importance weights, broadcast over
    rows. Uniform col_w reproduces the unweighted fit up to FP associativity.
    Returns (reconstruction, sorted_levels[R,K]).
    """
    rows, cols = x.shape
    mf = mask.float()
    w = col_w.to(torch.float32).clamp(min=0).view(1, cols)   # [1, cols]
    xf = x.to(torch.float32)
    wm = w * mf                                              # weight*mask
    # weighted per-row init statistics (fall back to tiny denom if a row has
    # zero total weight over its masked columns).
    wsum = wm.sum(dim=1, keepdim=True).clamp(min=1e-30)
    row_mean = (wm * xf).sum(dim=1, keepdim=True) / wsum
    row_var = (wm * (xf - row_mean) ** 2).sum(dim=1, keepdim=True) / wsum
    row_std = row_var.sqrt().clamp(min=1e-8)

    if K == 4:
        init_positions = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104],
                                      device=x.device, dtype=torch.float32)
    elif K == 3:
        init_positions = torch.tensor([-1.2247, 0.0, 1.2247],
                                      device=x.device, dtype=torch.float32)
    elif K == 2:
        init_positions = torch.tensor([-0.7979, 0.7979],
                                      device=x.device, dtype=torch.float32)
    else:
        quantiles = (torch.arange(K, device=x.device, dtype=torch.float32)
                     + 0.5) / K
        init_positions = torch.distributions.Normal(0.0, 1.0).icdf(quantiles)
    levels = row_mean + row_std * init_positions.unsqueeze(0)  # [rows, K] fp32

    masked_x = xf * mf
    for _ in range(iters):
        x_expanded = masked_x.unsqueeze(2)          # [rows, cols, 1]
        levels_expanded = levels.unsqueeze(1)       # [rows, 1, K]
        dists = (x_expanded - levels_expanded) ** 2
        dists = dists + (~mask).unsqueeze(2).float() * 1e30
        assignments = dists.argmin(dim=2)           # [rows, cols]
        new_levels = torch.zeros_like(levels)
        for k in range(K):
            k_mask = (assignments == k) & mask
            wk = wm * k_mask.float()                 # weighted membership
            k_wsum = wk.sum(dim=1).clamp(min=1e-30)
            k_xsum = (wk * xf).sum(dim=1)
            new_levels[:, k] = k_xsum / k_wsum
        if torch.allclose(new_levels, levels, atol=1e-6):
            levels = new_levels
            break
        levels = new_levels

    levels, _ = levels.sort(dim=1)
    # keep levels in fp32 (as the unweighted lloyd_max_quantize does — its
    # row_mean/row_std are fp32); the harness weight W is float() so this is
    # the same dtype the frozen-codebook snapping and dump path already use.
    rec = _snap_to_levels(x, mask, levels)
    return rec, levels


def _layer_part_k(self_):
    """K30 --bulk-k-map: per-sublayer effective (bulk, tail, salient) level
    counts. Returns the global RUN_STATE['partition_k'] unless the sublayer's
    manifest name (layer.global_name minus the MODEL_NAME prefix) is a key in
    RUN_STATE['bulk_k_map'], in which case ONLY the BULK entry (idx0) is
    overridden by the mapped K. Map absent (None, default) => exactly the
    global tuple, so the default code path is byte-identical."""
    part_k = RUN_STATE.get("partition_k", (4, 4, 4))
    kmap = RUN_STATE.get("bulk_k_map", None)
    if not kmap:
        return part_k
    gname = getattr(self_.layer, "global_name", None)
    if gname is None:
        return part_k
    lname = gname[len(MODEL_NAME):] if gname.startswith(MODEL_NAME) else gname
    if lname in kmap:
        return (int(kmap[lname]), part_k[1], part_k[2])
    return part_k


def _col_weights(H_diag_raw, Hinv):
    """Per-column importance weights for the weighted codebook fit.
      - 'hdiag': the (damped) diagonal of the Gram matrix H = X Xᵀ, i.e. each
        input dim's activation energy (LeanQuant-style saliency).
      - 'gptq' : 1/d_j² where d_j = Hinv[j,j] is the diagonal of the upper
        Cholesky of H⁻¹ — this is EXACTLY the per-column loss coefficient GPTQ
        itself sums (bigptq.py `Losses1 = (w-q)²/d²`), so weighting the fit by
        it makes the codebook minimize the same objective GPTQ optimizes.
    Returns None for 'none' (weighting disabled)."""
    mode = RUN_STATE.get("cb_weight", "none")
    if mode == "none":
        return None
    if mode == "hdiag":
        w = H_diag_raw.to(torch.float32).clamp(min=0)
    elif mode == "gptq":
        d = Hinv.diagonal().to(torch.float32)
        w = 1.0 / (d * d).clamp(min=1e-30)
    else:
        raise ValueError(mode)
    # optional exponent: w <- w^p (p=1 default). p<1 tempers the weighting,
    # p>1 sharpens it. Applied before mean-normalization (scale-invariant).
    p = float(RUN_STATE.get("cb_weight_pow", 1.0))
    if p != 1.0:
        w = w.clamp(min=0) ** p
    # normalize to mean 1 for numerical conditioning (scale-invariant anyway).
    w = w / w.mean().clamp(min=1e-30)
    return w


@torch.no_grad()
def _column_block_masks(Wblk, Hblk, metric, orders):
    """K2.6 E1 column-membership mask search for ONE 128-column block.

    mask3 (salient): EXACTLY the unmodified `structural_searching` output.
    bulk/tail: COLUMN-wise split of the non-salient columns by per-column
    score c_j = mean_i |target[i,j]|, threshold chosen among the 81 quantiles
    (0.10..0.90) of {c_j} by the same proxy objective as autosearch lines
    70-80 (order-1 high_order_residual on bulk+tail plus the order-2 salient
    group3, total MSE), with column masks expanded to all rows.

    Returns ((m1, m2, m3) element-expanded bool [R, nc], info dict).
    Hard-asserts: masks column-wise, disjoint, covering.
    """
    R, nc = Wblk.shape
    # metric dispatch — verbatim semantics of utils/structure.py
    if metric == "hessian":
        target = Wblk ** 2 / (torch.diag(Hblk).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target = Wblk
    else:
        raise NotImplementedError(metric)

    # salient columns: unmodified DOML search (up_lim=50 as in bigptq)
    _elem_split, mask3 = structural_searching(target, 50, orders=orders)
    sal_col = mask3[0]
    assert torch.equal(mask3, sal_col.unsqueeze(0).expand_as(mask3)), \
        "structural_searching returned a non-column-wise mask3"
    nonsal = ~sal_col
    n_nonsal = int(nonsal.sum().item())
    assert n_nonsal >= 2, f"only {n_nonsal} non-salient columns in block"

    c = target.abs().mean(dim=0)                       # [nc] per-column score
    c_ns = c[nonsal]
    qs = torch.linspace(0.10, 0.90, 81)
    thresholds = torch.tensor(
        np.quantile(c_ns.detach().cpu().numpy(), q=qs.cpu().numpy(),
                    axis=None, keepdims=False)).to(Wblk.device)

    group3 = high_order_residual(target, mask3, order=2)

    def _cols(t):
        tail = nonsal & (c >= t)
        return tail, nonsal & ~tail

    best_err, best_t = float("inf"), None
    for t in thresholds:
        tail_col, bulk_col = _cols(t)
        m1e = bulk_col.unsqueeze(0).expand(R, nc)
        m2e = tail_col.unsqueeze(0).expand(R, nc)
        g1 = high_order_residual(target, m1e, order=1)
        g2 = high_order_residual(target, m2e, order=1)
        err = torch.mean((target - (g1 + g2 + group3)) ** 2).item()
        if err < best_err:
            best_err, best_t = err, t

    tail_col, bulk_col = _cols(best_t)
    guard = 0
    if int(tail_col.sum()) == 0 or int(bulk_col.sum()) == 0:
        guard = 1                       # degenerate argmin -> median split
        t_med = torch.median(c_ns)
        tail_col, bulk_col = _cols(t_med)
        if int(tail_col.sum()) == 0 or int(bulk_col.sum()) == 0:
            guard = 2                   # still degenerate (ties) -> index split
            ns_idx = nonsal.nonzero(as_tuple=True)[0]
            order_idx = ns_idx[torch.argsort(c[ns_idx], descending=True,
                                             stable=True)]
            n_tail = max(1, n_nonsal // 2)
            tail_col = torch.zeros_like(nonsal)
            tail_col[order_idx[:n_tail]] = True
            bulk_col = nonsal & ~tail_col
    assert int(tail_col.sum()) > 0 and int(bulk_col.sum()) > 0

    m1 = bulk_col.unsqueeze(0).expand(R, nc).clone()
    m2 = tail_col.unsqueeze(0).expand(R, nc).clone()
    m3 = mask3.clone()
    # hard gates: column-wise, disjoint, covering
    for mm in (m1, m2, m3):
        assert torch.equal(mm, mm[0:1].expand_as(mm)), "mask not column-wise"
    cover = m1.to(torch.int32) + m2.to(torch.int32) + m3.to(torch.int32)
    assert bool((cover == 1).all()), "masks not a disjoint cover"

    info = {"guard": guard, "n_sal": int(sal_col.sum()),
            "n_tail": int(tail_col.sum()), "n_bulk": int(bulk_col.sum())}
    return (m1, m2, m3), info


@torch.no_grad()
def _resplit_bulk_frac(W_blk, masks):
    """K30 --bulk-frac β: RE-SPLIT the non-salient elements (bulk ∪ tail) of a
    128-col block into a new bulk/tail so that ≈β of those elements — the
    smallest-|W| ones — become the NEW bulk (mask1) and the rest the NEW tail
    (mask2). mask3 (salient) is kept EXACTLY. Purely element-wise, per block.

    The threshold is the β-quantile of the block's non-salient |W| values, so
    exactly ≈β of the non-salient elements fall into the bulk. Disjoint-cover
    (bulk ⊕ tail ⊕ salient = all) is preserved because {new_bulk, new_tail}
    partition the same non-salient set that {mask1, mask2} did.

    When RUN_STATE['bulk_frac'] is None this returns `masks` UNCHANGED, so the
    default code path is byte-identical to the original."""
    beta = RUN_STATE.get("bulk_frac", None)
    if beta is None:
        return masks
    m_bulk, m_tail, m_sal = masks
    nonsal = m_bulk | m_tail                       # element-wise non-salient
    absW = W_blk.abs().to(torch.float32)
    vals = absW[nonsal]
    if vals.numel() == 0:
        return masks
    thr = torch.quantile(vals, float(beta))        # β-quantile of |W| over nonsal
    new_bulk = nonsal & (absW <= thr)              # smallest-|W| fraction β
    new_tail = nonsal & ~new_bulk
    # hard invariant: {new_bulk, new_tail, salient} is a disjoint cover of all.
    cover = (new_bulk.to(torch.int32) + new_tail.to(torch.int32)
             + m_sal.to(torch.int32))
    assert bool((cover == 1).all()), "bulk-frac re-split broke disjoint cover"
    return (new_bulk, new_tail, m_sal)


RD_GRID = tuple(round(0.40 + 0.05 * i, 2) for i in range(10))  # 0.40..0.85


def _bits_for_k(kmax):
    """Fixed-width bits for code values 0..kmax-1 (same rule as
    k29_honest_bpw._bits_for_k, so the rate model matches the honest
    accounting): K<=1 -> 0, K=2 -> 1, K in {3,4} -> 2."""
    if kmax <= 1:
        return 0
    if kmax <= 2:
        return 1
    return 2


@torch.no_grad()
def _rd_split_masks(W_blk, masks, col_w_blk, part_k):
    """K32 --rd-split λ: K-aware, RATE-aware per-block bulk/tail re-split of
    the NON-salient elements of one 128-col block. mask3 (salient) is kept
    EXACTLY; only the bulk/tail border of the non-salient set moves.

    Candidate borders: the |W| β-quantile thresholds over the block's
    non-salient elements for β in RD_GRID (0.40..0.85, step .05) — the same
    threshold rule as `_resplit_bulk_frac`, swept instead of fixed. Each
    candidate is scored with the RD Lagrangian

        cost(β) = D(β) + λ · R(β)

      D(β): col_w-weighted SSE of the REAL K-model — a K_bulk-level Lloyd fit
        on the candidate bulk plus a K_tail-level Lloyd fit on the candidate
        tail, via the harness's own `_weighted_lloyd_max_quantize`.
        DOCUMENTED PROXY: the search fits are per (row, BLOCK) with
        RUN_STATE['rd_iters'] (default 8) Lloyd iterations and no fp8 level
        rounding, while the final codebooks stay the untouched full path
        (per (row, GROUP) union, 20 iters, fp8 rounding, refit-iters
        alternation). All candidates are evaluated in ONE batched Lloyd call
        (candidates stacked along the row axis; rows are independent in the
        primitive, so this is exactly the per-candidate fit).
      R(β): code + membership bits over the block's non-salient elements:
        n_bulk·bits(K_bulk) + n_tail·bits(K_tail) + n_ns·H2(n_bulk/n_ns)
        (binary entropy) — matching k29_honest_bpw's honest accounting
        (reduced-width code planes + lzma-coded membership).
      λ: weighted-squared-error per bit. λ=0 => pure K-aware
        distortion-optimal split. One global λ enforces the equal-slope
        condition across ALL blocks and layers.

    col_w_blk None (cb_weight 'none') => uniform column weights in D.
    Degenerate candidates (empty bulk or tail after ties) are skipped; if all
    candidates are degenerate the original masks are returned (rd_guard).
    The per-block D/R curves over the candidate grid are returned in `info`
    so any λ can be re-simulated offline from a λ=0 run's manifest.

    When RUN_STATE['rd_split'] is None the caller never invokes this — the
    default path is byte-identical to the original."""
    lam = float(RUN_STATE["rd_split"])
    m_bulk, m_tail, m_sal = masks
    nonsal = m_bulk | m_tail
    n_ns = int(nonsal.sum().item())
    R, nc = W_blk.shape
    if n_ns < 8:
        return masks, {"rd_guard": 1}
    if part_k is None:
        part_k = RUN_STATE.get("partition_k", (4, 4, 4))
    K_b, K_t = int(part_k[0]), int(part_k[1])
    bits_b, bits_t = _bits_for_k(K_b), _bits_for_k(K_t)

    absW = W_blk.abs().to(torch.float32)
    vals = absW[nonsal]
    qs = torch.tensor(RD_GRID, dtype=torch.float32, device=W_blk.device)
    thrs = torch.quantile(vals, qs)                       # [n_grid]
    cand_bulks, cand_tails, keep = [], [], []
    for ci in range(len(RD_GRID)):
        cb_ = nonsal & (absW <= thrs[ci])
        nb = int(cb_.sum().item())
        if nb == 0 or nb == n_ns:
            continue                                      # degenerate (ties)
        keep.append(ci)
        cand_bulks.append(cb_)
        cand_tails.append(nonsal & ~cb_)
    if not keep:
        return masks, {"rd_guard": 2}
    ncand = len(keep)

    if col_w_blk is None:
        cw = torch.ones(nc, device=W_blk.device, dtype=torch.float32)
    else:
        cw = col_w_blk.to(torch.float32).clamp(min=0)
    it = int(RUN_STATE.get("rd_iters", 8))
    Xrep = W_blk.repeat(ncand, 1)                         # [ncand*R, nc]
    Bstack = torch.cat(cand_bulks, dim=0)
    Tstack = torch.cat(cand_tails, dim=0)
    rec_b, _ = _weighted_lloyd_max_quantize(Xrep, Bstack, cw, K=K_b, iters=it)
    rec_t, _ = _weighted_lloyd_max_quantize(Xrep, Tstack, cw, K=K_t, iters=it)
    Xf = Xrep.to(torch.float32)
    err = ((Xf - rec_b) ** 2 * Bstack.float()
           + (Xf - rec_t) ** 2 * Tstack.float()) * cw.view(1, nc)
    D = err.view(ncand, R, nc).sum(dim=(1, 2))            # [ncand] wSSE
    nb_c = Bstack.view(ncand, R, nc).sum(dim=(1, 2)).to(torch.float32)
    p_c = nb_c / float(n_ns)
    H2 = torch.zeros_like(p_c)
    inner = (p_c > 0) & (p_c < 1)
    pi = p_c[inner]
    H2[inner] = -(pi * torch.log2(pi) + (1.0 - pi) * torch.log2(1.0 - pi))
    Rbits = nb_c * bits_b + (float(n_ns) - nb_c) * bits_t + float(n_ns) * H2
    cost = D + lam * Rbits
    best = int(torch.argmin(cost).item())

    new_bulk, new_tail = cand_bulks[best], cand_tails[best]
    # hard invariant: {new_bulk, new_tail, salient} is a disjoint cover.
    cover = (new_bulk.to(torch.int32) + new_tail.to(torch.int32)
             + m_sal.to(torch.int32))
    assert bool((cover == 1).all()), "rd-split broke disjoint cover"
    info = {
        "rd_beta": RD_GRID[keep[best]],
        "rd_frac": round(float(p_c[best].item()), 5),
        "rd_cands": [RD_GRID[k] for k in keep],
        "rd_D": [float(f"{v:.6g}") for v in D.tolist()],
        "rd_R": [float(f"{v:.6g}") for v in Rbits.tolist()],
        "rd_fracs": [round(float(v), 5) for v in p_c.tolist()],
    }
    return (new_bulk, new_tail, m_sal), info


def _compute_block_masks(W, st, ed, Hinv, self_, orders, col_w_blk=None,
                         part_k=None):
    """Per-128-block 3-way masks on the CURRENT W (unmodified primitive in
    element mmode; the E1 column-membership search in column mmode)."""
    if RUN_STATE["mmode"] == "column":
        masks, info = _column_block_masks(W[:, st:ed], Hinv[st:ed, st:ed],
                                          self_.salient_metric, orders)
    else:
        m1b, m2b, m3b = bigptq.structural_guassian_distribution(
            W[:, st:ed], Hinv[st:ed, st:ed], self_.salient_metric, 50,
            orders=orders)
        if RUN_STATE.get("rd_split", None) is not None:
            # K32 --rd-split: K-aware RD sweep of the bulk/tail border
            # (mutually exclusive with --bulk-frac; argparse enforces it).
            masks, info = _rd_split_masks(W[:, st:ed], (m1b, m2b, m3b),
                                          col_w_blk, part_k)
        else:
            # K30 --bulk-frac: re-split the non-salient elements by |W|
            # quantile (element mmode only; None => masks untouched).
            masks = _resplit_bulk_frac(W[:, st:ed], (m1b, m2b, m3b))
            info = {}
    if RUN_STATE.get("merge_tail", False):
        # masks are (bulk, tail, salient) — idx0/idx1/idx2. K27 probe 3:
        # merge bulk+tail into partition 0, empty partition 1, keep salient
        # at partition 2. Disjoint cover is preserved (2 non-empty parts).
        m_bulk, m_tail, m_sal = masks
        masks = (m_bulk | m_tail, torch.zeros_like(m_tail), m_sal)
    return masks, info


def _fit_group_codebooks(Wg, union_masks, rows, col_w=None, part_k=None):
    """Call the UNMODIFIED lloyd_max_quantize per partition on the [R, g]
    group slice; capture the sorted levels [R, 4]; hard-assert that
    _snap_to_levels reproduces the lloyd reconstruction bitwise.

    When RUN_STATE['cb_weight'] != 'none' (K28, DIRECTION B), the per-partition
    fit is the importance-weighted Lloyd-Max (`_weighted_lloyd_max_quantize`)
    with per-column weights `col_w` (aligned to Wg's columns) instead. Storage
    is unchanged (K sorted levels + nearest-level snap); only level VALUES move,
    so packed bpw is identical to the unweighted config at the same g."""
    recons, levels_g = [], []
    fp8 = RUN_STATE.get("codebook_dtype", "bf16") != "bf16"
    if part_k is None:                       # K30: caller may pass per-layer K
        part_k = RUN_STATE.get("partition_k", (4, 4, 4))
    weighted = RUN_STATE.get("cb_weight", "none") != "none"
    for p in range(3):
        K_p = part_k[p]
        if int(union_masks[p].sum().item()) == 0:
            # K27 probe 3 (--merge-tail) empties a partition; emit a zero
            # codebook + zero reconstruction so downstream indexing is safe
            # (snap on an all-False mask already returns zeros).
            recons.append(torch.zeros_like(Wg))
            levels_g.append(torch.zeros((rows, 4), device=Wg.device,
                                        dtype=Wg.dtype))
            continue
        if weighted:
            assert col_w is not None, "cb_weight set but col_w not threaded"
            rec, lev = _weighted_lloyd_max_quantize(
                Wg, union_masks[p], col_w, K=K_p, iters=20)
            # cheap gate: the returned levels must reproduce the returned
            # reconstruction under the exact decode used at serve time.
            snap = _snap_to_levels(Wg, union_masks[p], lev)
            if not torch.equal(snap, rec):
                bad = int((snap != rec).sum().item())
                raise RuntimeError(
                    f"partition {p}: weighted snap-to-levels mismatch "
                    f"({bad}) — snap replica broken")
        else:
            with _GatherCapture() as cap:
                rec = lloyd_max_quantize(Wg, union_masks[p], K=K_p, iters=20)
            if len(cap.tensors) != 1:
                raise RuntimeError(
                    f"expected exactly 1 gather inside lloyd_max_quantize, "
                    f"saw {len(cap.tensors)}")
            lev = cap.tensors[0]
            if lev.shape != (rows, K_p):
                raise RuntimeError(f"captured levels shape {tuple(lev.shape)} "
                                   f"!= ({rows}, {K_p})")
            snap = _snap_to_levels(Wg, union_masks[p], lev)
            if not torch.equal(snap, rec):
                bad = int((snap != rec).sum().item())
                raise RuntimeError(
                    f"partition {p}: snap-to-levels does not reproduce "
                    f"lloyd_max_quantize reconstruction ({bad} mismatches) — "
                    f"level capture or snap replica broken")
        if fp8:
            # K27 probe 1: snap levels onto the fp8 grid and RE-DERIVE the
            # reconstruction for ALL blocks (incl. block 0) so decode ==
            # nearest-fp8-level everywhere and stays self-consistent.
            lev = _maybe_round_levels(lev)
            rec = _snap_to_levels(Wg, union_masks[p], lev)
        recons.append(rec)
        levels_g.append(lev)
    return recons, levels_g


@torch.no_grad()
def refit_fasterquant(self, blocksize=128, percdamp=0.01, partition=3,
                      orders=(1, 1, 2), global_scale=False):
    """Reimplementation of BRAGPTQ.fasterquant for the doml/partition=3 GPTQ
    path with group-refit codebooks (group size from RUN_STATE['g']).
    All other configurations delegate to the ORIGINAL fasterquant."""
    method = getattr(self.braq_quantizer, "method", None)
    if method != "doml" or partition != 3 or self.disable_gptq or global_scale:
        return _ORIG_FQ(self, blocksize=blocksize, percdamp=percdamp,
                        partition=partition, orders=orders,
                        global_scale=global_scale)
    K_doml = int(getattr(self.braq_quantizer, "codebook_K", 4))
    assert K_doml == 4, f"group refit assumes K=4, got {K_doml}"
    assert isinstance(self.layer, nn.Linear), type(self.layer)

    # ---------------- preamble: verbatim replica of bigptq.py:63-129 -------
    W = self.layer.weight.data.clone()
    if isinstance(self.layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()
    W = W.float()
    tick = time.time()

    # doml never takes the global_scale branch; replicate the clearing branch
    if hasattr(self.braq_quantizer, "global_scale"):
        self.braq_quantizer.global_scale = None
    if hasattr(self.braq_quantizer, "global_zero"):
        self.braq_quantizer.global_zero = None

    H = self.H
    del self.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    Losses = torch.zeros(self.rows, device=self.dev)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.columns, device=self.dev)
    H[diag, diag] += damp
    H_diag_raw = torch.diag(H).clone()  # noqa: F841  (parity with original)
    for _retry in range(10):
        try:
            H_chol = torch.linalg.cholesky(H)
            break
        except torch._C._LinAlgError:
            extra_damp = 1e-3 * torch.mean(torch.diag(H))
            if extra_damp == 0:
                extra_damp = 1e-6
            H[diag, diag] += extra_damp
    else:
        H_chol = torch.diag(torch.sqrt(torch.diag(H).clamp(min=1e-8)))
    H = torch.cholesky_inverse(H_chol)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    # K28 (DIRECTION B): per-column importance weights for the codebook fit,
    # computed once from the damped Gram diagonal / the Cholesky sensitivity.
    # None when cb_weight == 'none' (default path untouched).
    col_w_full = _col_weights(H_diag_raw, Hinv)

    # ---------------- group-refit main loop ---------------------------------
    C, R = self.columns, self.rows
    g_cfg = RUN_STATE["g"]
    assert g_cfg is not None, "RUN_STATE['g'] not configured"
    g_eff = C if g_cfg == "global" else min(int(g_cfg), C)
    assert g_eff % blocksize == 0, (g_eff, blocksize)
    assert self.braq_quantizer.groupsize % blocksize == 0

    masks_cols = ([], [], [])          # full-layer SNAP masks, for dump/derive
    minfos = []                        # per-block column-search info (K2.6)
    n_groups = 0

    # K31 (DIRECTION B') outer joint codebook<->GPTQ iterations per group.
    intra = RUN_STATE.get("intra_block", False)
    M_refit = int(RUN_STATE.get("refit_iters", 1))
    if M_refit > 1 and not intra:
        raise RuntimeError(
            "--refit-iters > 1 requires --intra-block-gptq (the joint "
            "codebook<->GPTQ alternation is defined only for that path)")
    # K30 --bulk-k-map: per-sublayer effective level counts (== the global
    # partition_k tuple whenever the map is absent or misses this sublayer).
    part_k = _layer_part_k(self)

    for grp_st in range(0, C, g_eff):
        grp_ed = min(grp_st + g_eff, C)
        n_groups += 1

        # -- per-128-block 3-way masks for ALL blocks of the group, computed
        #    on the CURRENT W (group-start state; unmodified primitive in
        #    element mmode, E1 column search in column mmode). These masks
        #    always drive the GROUP CODEBOOK FIT; they also drive snapping
        #    and storage unless --fresh-masks recomputes them per block.
        blocks = []
        for st in range(grp_st, grp_ed, blocksize):
            ed = min(st + blocksize, grp_ed)
            bm, minfo = _compute_block_masks(
                W, st, ed, Hinv, self, orders,
                col_w_blk=(None if col_w_full is None
                           else col_w_full[st:ed]),
                part_k=part_k)
            blocks.append((st, ed, bm, minfo))

        # -- group-start slice (FIXED across refit iterations) + union masks.
        Wg = W[:, grp_st:grp_ed].clone()
        union_masks = [torch.cat([b[2][p] for b in blocks], dim=1)
                       for p in range(3)]
        col_w_g = None if col_w_full is None else col_w_full[grp_st:grp_ed]
        Gcols = grp_ed - grp_st

        # K31: snapshot the downstream columns so each refit iteration re-runs
        # the group's inter-block feedback from the SAME group-start state;
        # only the FINAL iteration's feedback survives into downstream columns.
        Wdown_start = W[:, grp_ed:].clone() if M_refit > 1 else None

        levels_g = None
        recons = None
        assign_full = None                   # per-partition [R, Gcols] indices
        for it in range(M_refit):
            final_it = (it == M_refit - 1)
            if M_refit > 1:
                # re-run the whole group from group-start: reset the group's
                # own columns AND the downstream columns (undo this group's
                # inter-block feedback so it is not accumulated across iters).
                W[:, grp_st:grp_ed] = Wg
                W[:, grp_ed:] = Wdown_start

            if it == 0:
                # -- ONE codebook per (row, partition) over the group union
                #    mask (hdiag-weighted Lloyd on cb_weight != none). Bit-for-
                #    bit the original fit at M=1.
                recons, levels_g = _fit_group_codebooks(
                    Wg, union_masks, R, col_w_g, part_k=part_k)
            else:
                # -- RE-FIT levels: (col_w-weighted) mean of the group-start Wg
                #    values grouped by the PREVIOUS sweep's GPTQ assignments.
                levels_g = _refit_levels_from_assign(
                    Wg, union_masks, assign_full, levels_g, col_w_g, part_k)

            capture = (M_refit > 1)
            if capture:
                assign_full = [torch.zeros((R, Gcols), dtype=torch.long,
                                           device=W.device) for _ in range(3)]

            # -- process the k blocks exactly as the original legacy loop does.
            for bi, (col_st, col_ed, bmasks_fit, minfo_fit) in enumerate(
                    blocks):
                n_cols = col_ed - col_st
                if RUN_STATE["fresh_masks"] and bi > 0:
                    # E2: snap/storage masks recomputed at block time on the
                    # CURRENT feedback-updated W (original DOML mask timing);
                    # the group codebooks stay frozen from the group start.
                    bmasks, minfo = _compute_block_masks(
                        W, col_st, col_ed, Hinv, self, orders,
                        col_w_blk=(None if col_w_full is None
                                   else col_w_full[col_st:col_ed]),
                        part_k=part_k)
                else:
                    # block 0's W is unchanged since the group start, so the
                    # fit masks ARE the block-time masks; likewise all blocks
                    # in the frozen-mask (K2.5) mode.
                    bmasks, minfo = bmasks_fit, minfo_fit
                if final_it:            # store masks/minfos ONCE (final result)
                    for p in range(3):
                        masks_cols[p].append(bmasks[p])
                    minfos.append(minfo)
                mask = torch.stack(list(bmasks), dim=0)  # [3, R, n_cols] bool

                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                if intra:
                    # K28/K31 (DIRECTION B/B') — PROPER intra-block GPTQ
                    # feedback: quantize column-by-column, snapping each
                    # feedback-updated column to the FROZEN group codebook per
                    # its partition, then propagate the residual within THIS
                    # block via Hinv1. Storage is unchanged (final values are
                    # still the frozen group levels). W1 IS mutated in-block.
                    # K31: also CAPTURE the per-weight assignment index so the
                    # next refit iteration re-centers levels on those feedback-
                    # aware assignments.
                    gcol0 = col_st - grp_st
                    for i in range(n_cols):
                        w = W1[:, i]
                        d = Hinv1[i, i]
                        q = torch.zeros_like(w)
                        for j in range(3):
                            mij = mask[j, :, i]
                            if bool(mij.any()):
                                rec_ji, asg_ji = _snap_to_levels_assign(
                                    w.unsqueeze(1), mij.unsqueeze(1),
                                    levels_g[j])
                                q = q + rec_ji.squeeze(1)
                                if capture:
                                    assign_full[j][:, gcol0 + i] = \
                                        asg_ji.squeeze(1)
                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / d**2
                        err1 = (w - q) / d
                        Err1[:, i] = err1
                        if i + 1 < n_cols:
                            W1[:, i + 1:] -= (err1.unsqueeze(1)
                                              * Hinv1[i, i + 1:].unsqueeze(0))
                else:
                    if bi == 0:
                        # Block 0's values are unchanged since the fit -> the
                        # lloyd reconstruction restricted to these columns IS
                        # the snap (proven bitwise per group by
                        # _fit_group_codebooks). At g=128 every block is block
                        # 0 => exactly the original op.
                        q_part_groups = [recons[p][:, :n_cols]
                                         for p in range(3)]
                    else:
                        # Later blocks received inter-block feedback -> snap
                        # the CURRENT values to the FROZEN group codebook.
                        q_part_groups = [
                            _snap_to_levels(W1, bmasks[p], levels_g[p])
                            for p in range(3)
                        ]

                    # Column sweep — verbatim replica of bigptq.py:505-520
                    # (legacy path: W1 is never mutated inside the block).
                    for i in range(n_cols):
                        w = W1[:, i]
                        d = Hinv1[i, i]

                        q = torch.zeros_like(w)
                        for j in range(mask.shape[0]):
                            q += q_part_groups[j][:, i] * mask[j, :, i]

                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / d**2
                        err1 = (w - q) / d
                        Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                if final_it:            # accumulate GPTQ loss ONCE (final)
                    Losses += torch.sum(Losses1, 1) / 2
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
        del Wg

    # ---------------- K27 probe 2: optional global second-pass codebooks ----
    # Replace the per-block/per-group codebooks with ONE global codebook per
    # (row, partition): refit on the FINAL quantized W over each partition's
    # full-width union mask, then re-snap that partition's elements to the
    # global levels. Runs on float32 W BEFORE the final cast below.
    if RUN_STATE.get("two_pass", "none") == "global":
        for p in range(3):
            if not masks_cols[p]:
                continue
            union_mask_p = torch.cat(masks_cols[p], dim=1)
            if int(union_mask_p.sum().item()) == 0:
                continue                    # e.g. emptied partition (merge)
            with _GatherCapture() as cap:
                lloyd_max_quantize(W, union_mask_p, K=4, iters=20)
            lev_g = _maybe_round_levels(cap.tensors[0])
            snapped = _snap_to_levels(W, union_mask_p, lev_g)
            W = torch.where(union_mask_p, snapped, W)

    # ---------------- tail: verbatim replica of bigptq.py:528-547 ----------
    torch.cuda.synchronize()
    print("time %.2f" % (time.time() - tick))
    print("error", torch.sum(Losses).item())

    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()
    self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
        self.layer.weight.data.dtype
    )

    err_total = torch.sum(Losses).item()

    # ---------------- post-layer: G0 gate compare and/or DPK dump ----------
    _post_layer(self, masks_cols, g_eff, n_groups, err_total, minfos)

    del W1, Q1, W, Err1, Losses1, Hinv1
    del H, Hinv
    torch.cuda.empty_cache()
    return {"error": err_total}


def _post_layer(self, masks_cols, g_eff, n_groups, err_total, minfos=None):
    st = RUN_STATE
    if st["t0"] is None:                # selftest mode: no gate/dump/manifest
        return
    W_layer = self.layer.weight.data
    R, C = int(W_layer.shape[0]), int(W_layer.shape[1])
    gname = getattr(self.layer, "global_name", None)
    assert gname is not None and gname.startswith(MODEL_NAME), gname
    layer_name = gname[len(MODEL_NAME):]

    rec = {"layer_name": layer_name, "R": R, "C": C, "g_eff": g_eff,
           "n_groups": n_groups, "gptq_error": err_total,
           "mmode": st["mmode"], "fresh_masks": st["fresh_masks"]}
    if st.get("bulk_k_map"):
        rec["part_k"] = list(_layer_part_k(self))   # K30: effective per-layer K
    if st["mmode"] == "column" and minfos:
        infos = [m for m in minfos if m]
        rec["col_guard_median"] = sum(1 for m in infos if m["guard"] == 1)
        rec["col_guard_index"] = sum(1 for m in infos if m["guard"] == 2)
        rec["col_n_sal_mean"] = round(
            sum(m["n_sal"] for m in infos) / max(1, len(infos)), 2)
        rec["col_n_tail_mean"] = round(
            sum(m["n_tail"] for m in infos) / max(1, len(infos)), 2)
    if st.get("rd_split", None) is not None and minfos:
        hits = [m for m in minfos if m and "rd_beta" in m]
        rec["rd_guards"] = sum(1 for m in minfos if m and "rd_guard" in m)
        if hits:
            fr = np.array([m["rd_frac"] for m in hits], dtype=np.float64)
            rec["rd_n_blocks"] = len(hits)
            rec["rd_frac_mean"] = round(float(fr.mean()), 5)
            rec["rd_frac_std"] = round(float(fr.std()), 5)
            rec["rd_beta_hist"] = {
                str(b): sum(1 for m in hits if m["rd_beta"] == b)
                for b in sorted({m["rd_beta"] for m in hits})}
            # full per-block RD curves — lets any λ be re-simulated offline
            rec["rd_blocks"] = [
                {"beta": m["rd_beta"], "cands": m["rd_cands"],
                 "D": m["rd_D"], "R": m["rd_R"], "fracs": m["rd_fracs"]}
                for m in hits]

    gate_msg = ""
    if st["gate_dir"]:
        ref_path = os.path.join(st["gate_dir"],
                                f"{layer_name}.wq.safetensors")
        with safe_open(ref_path, framework="pt", device="cpu") as f:
            wq_ref = f.get_tensor("wq")
        wq_ref = wq_ref.to(W_layer.device)
        assert wq_ref.dtype == torch.bfloat16 and wq_ref.shape == W_layer.shape
        eq = (wq_ref.contiguous().view(torch.int16)
              == W_layer.contiguous().view(torch.int16))
        n_bad = int((~eq).sum().item())
        rec["gate_bitwise_ok"] = (n_bad == 0)
        rec["gate_mismatches"] = n_bad
        if n_bad:
            idx = (~eq).nonzero()[:5].tolist()
            raise RuntimeError(
                f"G0 FAIL: {layer_name}: {n_bad}/{R*C} weights differ "
                f"bitwise from {ref_path}; first mismatch indices {idx}")
        gate_msg = " gate=BITWISE-OK"

    if st["dump_dir"] or st["stats_only"]:
        m1 = torch.cat(masks_cols[0], dim=1)
        m2 = torch.cat(masks_cols[1], dim=1)
        m3 = torch.cat(masks_cols[2], dim=1)
        # derive_dpk also hard-validates the doc-02 §3 invariant (bit-exact
        # reconstruction) and <=4 distinct levels per (row, group, partition);
        # in column mmode additionally: column-wise m1/m2 + colmem decode-back.
        tensors, stats = derive_dpk(W_layer, m1, m2, m3, B=B_BLOCK, g=g_eff,
                                    mmode=st["mmode"])
        if st["dump_dir"]:
            meta = container_meta(stats["R"], stats["C"], stats["C_orig"],
                                  stats["NG"], layer_name, g=g_eff,
                                  mmode=st["mmode"])
            mjson = json.dumps(meta)
            save_file(tensors,
                      os.path.join(st["dump_dir"],
                                   f"{layer_name}.dpk.safetensors"),
                      metadata={"meta": mjson})
            save_file({"wq": W_layer.detach().cpu().contiguous()},
                      os.path.join(st["dump_dir"],
                                   f"{layer_name}.wq.safetensors"),
                      metadata={"meta": mjson})
        rec.update({k: stats[k] for k in
                    ("NG", "n_sal_cols", "ndist_hist", "packed_bpw")})
        rec["packed_bits"] = int(round(stats["packed_bpw"] * R * C))
        gate_msg += f" bpw={stats['packed_bpw']:.4f} inv=BITWISE-OK"

    st["manifest"].append(rec)
    st["n_refit_layers"] += 1
    print(f"K25REFIT[{st['n_refit_layers']:3d}] {layer_name} R={R} C={C} "
          f"g={g_eff} NG={n_groups}{gate_msg} "
          f"t={time.time() - st['t0']:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# --run mode
# ---------------------------------------------------------------------------
def main_run(args):
    set_model(args.model or DEFAULT_MODEL)
    print(f"K25REFIT: model = {MODEL_NAME}", flush=True)
    g_tag = args.g
    RUN_STATE["g"] = "global" if g_tag == "global" else int(g_tag)
    RUN_STATE["gate_dir"] = args.gate_dir
    RUN_STATE["dump_dir"] = args.dump_dir
    RUN_STATE["stats_only"] = args.stats_only
    RUN_STATE["mmode"] = args.mmode
    RUN_STATE["fresh_masks"] = args.fresh_masks
    RUN_STATE["codebook_dtype"] = args.codebook_dtype
    RUN_STATE["two_pass"] = args.two_pass
    RUN_STATE["merge_tail"] = args.merge_tail
    RUN_STATE["partition_k"] = (args.bulk_k, args.tail_k, args.salient_k)
    RUN_STATE["cb_weight"] = args.cb_weight
    RUN_STATE["cb_weight_pow"] = args.cb_weight_pow
    RUN_STATE["intra_block"] = args.intra_block_gptq
    RUN_STATE["bulk_frac"] = args.bulk_frac
    RUN_STATE["rd_split"] = args.rd_split
    RUN_STATE["rd_iters"] = args.rd_iters
    RUN_STATE["refit_iters"] = args.refit_iters
    if args.rd_split is not None:
        if args.bulk_frac is not None:
            raise SystemExit("--rd-split and --bulk-frac are mutually "
                             "exclusive (both re-split the bulk/tail border)")
        if args.mmode != "element" or args.merge_tail:
            raise SystemExit("--rd-split requires element mmode without "
                             "--merge-tail")
    if args.bulk_k_map:
        with open(args.bulk_k_map) as f:
            RUN_STATE["bulk_k_map"] = {str(k): int(v)
                                       for k, v in json.load(f).items()}
        print(f"K30: bulk-k-map loaded from {args.bulk_k_map}: "
              f"{len(RUN_STATE['bulk_k_map'])} sublayers overridden", flush=True)
    else:
        RUN_STATE["bulk_k_map"] = None
    if args.refit_iters > 1 and not args.intra_block_gptq:
        raise SystemExit(
            "--refit-iters > 1 requires --intra-block-gptq")
    RUN_STATE["t0"] = time.time()
    tag = args.tag or f"g{g_tag}"
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    os.chdir(REPO)

    bigptq.BRAGPTQ.fasterquant = refit_fasterquant
    print(f"K25REFIT: fasterquant patched OK (g={RUN_STATE['g']}, "
          f"mmode={args.mmode}, fresh_masks={args.fresh_masks}, tag={tag}, "
          f"gate_dir={args.gate_dir}, dump_dir={args.dump_dir})", flush=True)

    import runpy
    import threading

    def _watchdog():
        time.sleep(300)
        if RUN_STATE["n_refit_layers"] == 0:
            print("K25REFIT FATAL: no layers after 300 s — patch dead; "
                  "aborting.", file=sys.stderr, flush=True)
            os._exit(17)

    threading.Thread(target=_watchdog, daemon=True).start()

    sys.argv = list(RUN_ARGV)
    print("K25REFIT: launching run.py:", sys.argv, flush=True)
    err = None
    try:
        runpy.run_path(os.path.join(REPO, "run.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            err = f"SystemExit({e.code})"
    except Exception as e:  # noqa: BLE001 — recorded, then re-raised via exit
        import traceback
        err = repr(e)
        traceback.print_exc()
    finally:
        manifest = {
            "model": MODEL_NAME,
            "argv": RUN_ARGV[1:],
            "g": g_tag,
            "mmode": args.mmode,
            "fresh_masks": args.fresh_masks,
            "bulk_k_map": args.bulk_k_map,
            "rd_split": args.rd_split,
            "rd_iters": args.rd_iters,
            "tag": tag,
            "gate_dir": args.gate_dir,
            "dump_dir": args.dump_dir,
            "n_sublayers_refit": RUN_STATE["n_refit_layers"],
            "expected_sublayers": EXPECTED_SUBLAYERS,
            "error": err,
            "layers": RUN_STATE["manifest"],
        }
        log_sub = ("k25_logs" if (args.mmode == "element"
                                  and not args.fresh_masks
                                  and args.tag is None) else "k26_logs")
        mpath = (os.path.join(args.dump_dir, "manifest.json") if args.dump_dir
                 else os.path.join(VERIFY_DIR, log_sub,
                                   f"refit_{tag}_manifest.json"))
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"K25REFIT: done. layers refit = {RUN_STATE['n_refit_layers']} "
              f"(expected {EXPECTED_SUBLAYERS}); error = {err}; "
              f"manifest = {mpath}", flush=True)
    if err:
        sys.exit(1)
    if RUN_STATE["n_refit_layers"] != EXPECTED_SUBLAYERS:
        print(f"K25REFIT FATAL: refit {RUN_STATE['n_refit_layers']} != "
              f"{EXPECTED_SUBLAYERS}", file=sys.stderr, flush=True)
        sys.exit(2)


# ---------------------------------------------------------------------------
# --restore-dpk mode (K33): restore a dump's wq weights instead of quantizing,
# then let the UNTOUCHED run.py finish with its standard eval. The patched
# fasterquant loads <dump_dir>/<layer_name>.wq.safetensors (shape/dtype
# checked) into self.layer.weight.data. NOTHING is written; the run ends with
# run.py's standard wikitext2 seed-0 PPL eval (plus c4/ptb if
# --eval-extra-ppl). A restored dump must reproduce its own run's PPL exactly
# (bitwise-identical weights).
# ---------------------------------------------------------------------------
def main_restore(args):
    import glob

    dump_dir = os.path.abspath(args.restore_dpk)
    if not os.path.isdir(dump_dir):
        raise SystemExit(f"K33RESTORE FATAL: {dump_dir} is not a directory")
    n_wq = len(glob.glob(os.path.join(dump_dir, "*.wq.safetensors")))
    print(f"K33RESTORE: dump dir {dump_dir} has {n_wq} wq sublayer files",
          flush=True)
    man_model = manifest_model(dump_dir)
    if args.model and man_model and args.model != man_model:
        raise SystemExit(
            f"K33RESTORE FATAL: --model {args.model} != dump manifest model "
            f"{man_model} — refusing to mix models")
    set_model(args.model or man_model or DEFAULT_MODEL)
    print(f"K33RESTORE: model = {MODEL_NAME} (manifest={man_model}, "
          f"--model={args.model})", flush=True)
    # count check AFTER set_model so EXPECTED_SUBLAYERS matches this model
    if n_wq != EXPECTED_SUBLAYERS:
        raise SystemExit(f"K33RESTORE FATAL: {n_wq} wq files != "
                         f"{EXPECTED_SUBLAYERS} expected sublayers")
    os.chdir(REPO)

    state = {"n": 0, "t0": time.time()}

    def restore_fasterquant(self, blocksize=128, percdamp=0.01, partition=3,
                            orders=(1, 1, 2), global_scale=False):
        method = getattr(self.braq_quantizer, "method", None)
        if method != "doml":
            raise RuntimeError(
                f"K33RESTORE: expected a doml sublayer, got method={method} "
                f"— restore mode only supports the standard doml run")
        gname = getattr(self.layer, "global_name", None)
        assert gname is not None and gname.startswith(MODEL_NAME), gname
        layer_name = gname[len(MODEL_NAME):]
        wq_path = os.path.join(dump_dir, f"{layer_name}.wq.safetensors")
        if not os.path.exists(wq_path):
            raise FileNotFoundError(f"K33RESTORE: missing {wq_path}")
        with safe_open(wq_path, framework="pt", device="cpu") as f:
            wq = f.get_tensor("wq")
        W = self.layer.weight
        if wq.dtype != W.dtype or tuple(wq.shape) != tuple(W.shape):
            raise RuntimeError(
                f"K33RESTORE: {layer_name}: dump wq is "
                f"{wq.dtype}{tuple(wq.shape)}, layer weight is "
                f"{W.dtype}{tuple(W.shape)}")
        self.layer.weight.data = wq.to(W.device)
        self.H = None                      # parity with original's `del self.H`
        state["n"] += 1
        print(f"K33RESTORE[{state['n']:3d}] {layer_name} "
              f"R={wq.shape[0]} C={wq.shape[1]} restored "
              f"t={time.time() - state['t0']:.1f}s", flush=True)
        return {"error": float("nan"), "restored": True}

    bigptq.BRAGPTQ.fasterquant = restore_fasterquant
    print(f"K33RESTORE: fasterquant patched to RESTORE from {dump_dir}",
          flush=True)

    # Hard pre-eval guard: PPL must never be computed/logged on a partially
    # restored model. run.py's `from eval_utils import evaluate_and_log_all`
    # resolves against sys.modules at its (runpy) import time, so patching the
    # module attribute here is picked up by run.py.
    sys.path.insert(0, os.path.join(REPO, "src"))
    import eval_utils
    _orig_eval = eval_utils.evaluate_and_log_all

    def _guarded_eval(*a, **kw):
        if state["n"] != EXPECTED_SUBLAYERS:
            print(f"K33RESTORE FATAL: eval reached with only {state['n']}/"
                  f"{EXPECTED_SUBLAYERS} sublayers restored — aborting "
                  f"before any PPL is computed.", file=sys.stderr, flush=True)
            os._exit(3)
        print(f"K33RESTORE: all {state['n']}/{EXPECTED_SUBLAYERS} sublayers "
              f"restored — proceeding to the standard eval.", flush=True)
        return _orig_eval(*a, **kw)

    eval_utils.evaluate_and_log_all = _guarded_eval

    import runpy
    import threading

    def _watchdog():
        time.sleep(300)
        if state["n"] == 0:
            print("K33RESTORE FATAL: no restores after 300 s — patch dead; "
                  "aborting.", file=sys.stderr, flush=True)
            os._exit(17)

    threading.Thread(target=_watchdog, daemon=True).start()

    sys.argv = list(RUN_ARGV)
    if args.eval_extra_ppl:
        sys.argv.append("--eval_extra_ppl")
    if args.eval_arc:
        sys.argv.append("--eval_arc")
    if args.eval_mmlu:
        sys.argv.append("--eval_mmlu")
    if args.eval_hellaswag:
        sys.argv.append("--eval_hellaswag")
    print("K33RESTORE: launching run.py:", sys.argv, flush=True)
    err = None
    try:
        runpy.run_path(os.path.join(REPO, "run.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            err = f"SystemExit({e.code})"
    except Exception as e:  # noqa: BLE001 — recorded, then re-raised via exit
        import traceback
        err = repr(e)
        traceback.print_exc()
    print(f"K33RESTORE: done. sublayers restored = {state['n']} "
          f"(expected {EXPECTED_SUBLAYERS}); error = {err}", flush=True)
    if err:
        sys.exit(1)
    if state["n"] != EXPECTED_SUBLAYERS:
        print(f"K33RESTORE FATAL: restored {state['n']} != "
              f"{EXPECTED_SUBLAYERS}", file=sys.stderr, flush=True)
        sys.exit(2)


# ---------------------------------------------------------------------------
# --selftest mode: synthetic layer; refit@g=128 must equal ORIGINAL
# fasterquant bitwise; refit@g=256/global must round-trip through derive_dpk.
# ---------------------------------------------------------------------------
def _make_case(seed, R, C, device):
    gen = torch.Generator().manual_seed(seed)
    W0 = (torch.randn(R, C, generator=gen) * 0.05).to(torch.bfloat16)
    X = torch.randn(8, 64, C, generator=gen) * 0.7
    return W0, X


def _build_braq(W0, X, device):
    lin = nn.Linear(W0.shape[1], W0.shape[0], bias=False)
    lin.weight.data = W0.clone().to(device)
    lin = lin.to(device)
    q = Binarization(lin.weight, method="doml", groupsize=-1)
    q.codebook_K = 4
    br = bigptq.BRAGPTQ(lin, q, salient_metric="magnitude")
    for j in range(X.shape[0]):
        br.add_batch(X[j].to(device).float(), None)
    return lin, br


def main_selftest():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    R, C = 64, 512
    W0, X = _make_case(1234, R, C, device)

    # (a) refit at g=128 == ORIGINAL fasterquant, bitwise
    lin_a, br_a = _build_braq(W0, X, device)
    out_a = _ORIG_FQ(br_a, blocksize=128, percdamp=0.01, partition=3,
                     orders=(1, 1, 1))
    RUN_STATE["g"] = 128
    lin_b, br_b = _build_braq(W0, X, device)
    out_b = refit_fasterquant(br_b, blocksize=128, percdamp=0.01,
                              partition=3, orders=(1, 1, 1))
    eq = (lin_a.weight.data.view(torch.int16)
          == lin_b.weight.data.view(torch.int16))
    n_bad = int((~eq).sum().item())
    assert n_bad == 0, f"selftest (a) FAILED: {n_bad}/{R*C} weights differ " \
                       f"between original and refit@g=128"
    assert out_a["error"] == out_b["error"], (out_a, out_b)
    print(f"selftest (a): refit@g=128 == original fasterquant BITWISE "
          f"({R}x{C}, gptq_error={out_a['error']:.6f})  PASS")

    # (b) refit at g=256 and g=global: <=4 distinct values per
    # (row, group, partition) and bitwise DPK round-trip via K2's packer.
    import dpk_unpack

    cap_masks = {}
    _orig_post = globals()["_post_layer"]

    def _cap_post(self_, masks_cols, g_eff, n_groups, err_total,
                  minfos=None):
        cap_masks["m"] = [torch.cat(mc, dim=1) for mc in masks_cols]
        cap_masks["g_eff"] = g_eff
        cap_masks["minfos"] = minfos

    def _run_captured(br):
        globals()["_post_layer"] = _cap_post
        try:
            refit_fasterquant(br, blocksize=128, percdamp=0.01,
                              partition=3, orders=(1, 1, 1))
        finally:
            globals()["_post_layer"] = _orig_post

    W_frozen256 = None
    for g_cfg, g_num in ((256, 256), ("global", C)):
        RUN_STATE["g"] = g_cfg
        lin_c, br_c = _build_braq(W0, X, device)
        _run_captured(br_c)

        m1, m2, m3 = cap_masks["m"]
        assert cap_masks["g_eff"] == g_num
        Wq = lin_c.weight.data
        if g_cfg == 256:
            W_frozen256 = Wq.detach().clone()
        tensors, stats = derive_dpk(Wq, m1, m2, m3, B=128, g=g_num)
        assert stats["NG"] == C // g_num
        meta = container_meta(stats["R"], stats["C"], stats["C_orig"],
                              stats["NG"], "selftest", g=g_num)
        W2 = dpk_unpack.unpack(tensors, meta)
        assert torch.equal(
            W2[:, :C].contiguous().view(torch.int16).to(device),
            Wq.contiguous().view(torch.int16)), \
            f"selftest (b) g={g_cfg}: DPK round-trip NOT bitwise"
        print(f"selftest (b): refit@g={g_cfg} (g_eff={g_num}) "
              f"NG={stats['NG']} bpw={stats['packed_bpw']:.4f} "
              f"DPK round-trip BITWISE  PASS")

        # sanity: refit at coarser g must actually differ from g=128 output
        neq = int((lin_c.weight.data.view(torch.int16)
                   != lin_b.weight.data.view(torch.int16)).sum().item())
        print(f"selftest (b): refit@g={g_cfg} differs from g=128 on "
              f"{neq}/{R*C} weights (expected > 0)")
        assert neq > 0

    # ------------------------------------------------------------------
    # (c) K2.6 E2 fresh-masks property gates
    # ------------------------------------------------------------------
    # (c1) fresh-masks at g=128 is inert: bitwise == original fasterquant
    RUN_STATE["g"] = 128
    RUN_STATE["fresh_masks"] = True
    lin_f1, br_f1 = _build_braq(W0, X, device)
    refit_fasterquant(br_f1, blocksize=128, percdamp=0.01, partition=3,
                      orders=(1, 1, 1))
    n_bad = int((lin_f1.weight.data.view(torch.int16)
                 != lin_a.weight.data.view(torch.int16)).sum().item())
    assert n_bad == 0, f"selftest (c1) FAILED: fresh@g=128 differs from " \
                       f"original on {n_bad}/{R*C} weights"
    print("selftest (c1): fresh-masks@g=128 == original BITWISE  PASS")

    # (c2) diagonal Hessian => zero GPTQ feedback => block-time W equals
    # group-start W => fresh masks == fit masks => fresh == frozen BITWISE.
    outs = {}
    for fresh in (False, True):
        RUN_STATE["g"] = 256
        RUN_STATE["fresh_masks"] = fresh
        lin_d, br_d = _build_braq(W0, X, device)
        br_d.H = torch.eye(C, device=device) * 3.0    # kill feedback
        refit_fasterquant(br_d, blocksize=128, percdamp=0.01, partition=3,
                          orders=(1, 1, 1))
        outs[fresh] = lin_d.weight.data.detach().clone()
    n_bad = int((outs[True].view(torch.int16)
                 != outs[False].view(torch.int16)).sum().item())
    assert n_bad == 0, f"selftest (c2) FAILED: with diagonal H fresh@256 " \
                       f"differs from frozen@256 on {n_bad}/{R*C} weights"
    print("selftest (c2): diag-H fresh@g=256 == frozen@g=256 BITWISE  PASS")

    # (c3) real (correlated) Hessian: the mask-timing branch must engage
    RUN_STATE["g"] = 256
    RUN_STATE["fresh_masks"] = True
    lin_f3, br_f3 = _build_braq(W0, X, device)
    refit_fasterquant(br_f3, blocksize=128, percdamp=0.01, partition=3,
                      orders=(1, 1, 1))
    neq = int((lin_f3.weight.data.view(torch.int16)
               != W_frozen256.view(torch.int16)).sum().item())
    print(f"selftest (c3): fresh@g=256 differs from frozen@g=256 on "
          f"{neq}/{R*C} weights (expected > 0)")
    assert neq > 0
    RUN_STATE["fresh_masks"] = False

    # ------------------------------------------------------------------
    # (d) K2.6 E1/E3 column mmode: masks column-wise; colmem container
    #     round-trip bitwise; independent colmem decode; corruption gates
    # ------------------------------------------------------------------
    import tempfile
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref"))
    import ref_w2a4

    RUN_STATE["mmode"] = "column"
    for g_cfg in (128, 256):
        RUN_STATE["g"] = g_cfg
        lin_e, br_e = _build_braq(W0, X, device)
        _run_captured(br_e)
        m1, m2, m3 = cap_masks["m"]
        for mm in (m1, m2, m3):
            assert torch.equal(mm, mm[0:1].expand_as(mm)), \
                "selftest (d): captured mask not column-wise"
        Wq = lin_e.weight.data
        tensors, stats = derive_dpk(Wq, m1, m2, m3, B=128, g=g_cfg,
                                    mmode="column")
        assert set(tensors.keys()) == {"b0", "b1", "colmem", "s", "cb"}, \
            sorted(tensors.keys())
        meta = container_meta(stats["R"], stats["C"], stats["C_orig"],
                              stats["NG"], "selftest-col", g=g_cfg,
                              mmode="column")
        W2 = dpk_unpack.unpack(tensors, meta)
        assert torch.equal(
            W2[:, :C].contiguous().view(torch.int16).to(device),
            Wq.contiguous().view(torch.int16)), \
            f"selftest (d) g={g_cfg}: column DPK round-trip NOT bitwise"

        # independent colmem decode (raw shifts, no dpk_unpack code)
        codes_exp = torch.where(
            m3[0], torch.full_like(m3[0], 2, dtype=torch.int64),
            m2[0].to(torch.int64)).cpu()
        wrds = tensors["colmem"].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        dec = torch.stack([(wrds >> (2 * i)) & 3 for i in range(16)],
                          dim=1).reshape(-1)[:C]
        assert torch.equal(dec, codes_exp), \
            "selftest (d): independent colmem decode mismatch"

        # file round-trip through load_container (key/meta validation path)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "selftest-col.dpk.safetensors")
            save_file(tensors, p, metadata={"meta": json.dumps(meta)})
            t2, meta2 = dpk_unpack.load_container(p)
            W3 = dpk_unpack.unpack(t2, meta2)
            assert torch.equal(W3.view(torch.int16),
                               W2.cpu().view(torch.int16)), \
                "selftest (d): file round-trip NOT bitwise"

        # corruption gate 1: flip bulk<->tail on one non-salient column
        # (field position < 15 so the `3 << ...` below stays in int32 range)
        cand = ((codes_exp == 0)
                & (torch.arange(codes_exp.numel()) % 16 < 15)).nonzero()
        j0 = int(cand[0].item())
        tc = {k: v.clone() for k, v in tensors.items()}
        wv = tc["colmem"].view(torch.int32)
        wv[j0 // 16] ^= (1 << (2 * (j0 % 16)))
        Wc = dpk_unpack.unpack(tc, meta)
        neq = int((Wc.view(torch.int16)
                   != W2.cpu().view(torch.int16)).sum().item())
        assert neq > 0, "selftest (d): colmem corruption undetected"

        # corruption gate 2: invalid code 3 must be rejected
        tc2 = {k: v.clone() for k, v in tensors.items()}
        wv2 = tc2["colmem"].view(torch.int32)
        wv2[j0 // 16] |= (3 << (2 * (j0 % 16)))
        try:
            dpk_unpack.unpack(tc2, meta)
            raise AssertionError("selftest (d): code-3 colmem not rejected")
        except ValueError:
            pass

        # column-mode output must differ from the element-mode output at
        # the same g (the membership axis actually does something)
        ref_el = lin_b if g_cfg == 128 else None
        if ref_el is not None:
            neq2 = int((Wq.view(torch.int16)
                        != ref_el.weight.data.view(torch.int16)
                        ).sum().item())
            assert neq2 > 0
            print(f"selftest (d): column@g={g_cfg} differs from element@"
                  f"g={g_cfg} on {neq2}/{R*C} weights (expected > 0)")

        # (e) GEMV: bucket-vs-direct on the column container
        xw = ref_w2a4.pack_a4(ref_w2a4.make_xhat(meta, "rand", 7, "cpu"))
        tcpu = {k: v.cpu() for k, v in tensors.items()}
        ya = ref_w2a4.gemv_direct(tcpu, meta, xw, 1.0 / 64)
        yb = ref_w2a4.gemv_bucket(tcpu, meta, xw, 1.0 / 64)
        rel = ((yb - ya).abs().max()
               / ya.abs().max().clamp(min=1e-30)).item()
        assert rel < 1e-5, f"selftest (e) g={g_cfg}: bucket-vs-direct " \
                           f"norm-rel {rel:.3e} >= 1e-5"
        print(f"selftest (d/e): column@g={g_cfg} colmem round-trip BITWISE, "
              f"decode/corruption gates OK, gemv bucket-vs-direct "
              f"norm-rel={rel:.3e}  PASS")

    RUN_STATE["mmode"] = "element"

    # ------------------------------------------------------------------
    # (f) K28 importance-weighted codebook fit (DIRECTION B)
    # ------------------------------------------------------------------
    # (f1) unit: weighted primitive with UNIFORM weights == unweighted lloyd
    mask_u = torch.ones(R, C, dtype=torch.bool, device=device)
    xu = (torch.randn(R, C, generator=torch.Generator().manual_seed(7))
          * 0.05).to(torch.float32).to(device)
    rec_unw = lloyd_max_quantize(xu, mask_u, K=4, iters=20)
    rec_w, _lv = _weighted_lloyd_max_quantize(
        xu, mask_u, torch.ones(C, device=device), K=4, iters=20)
    max_abs = (rec_unw.float() - rec_w.float()).abs().max().item()
    assert max_abs < 1e-4, (
        f"selftest (f1): uniform-weight != unweighted (max |Δ| {max_abs:.2e})")
    print(f"selftest (f1): weighted(uniform) == unweighted lloyd "
          f"(max |Δ| {max_abs:.2e})  PASS")

    # (f2) full-layer gptq-weighted refit@g=256: differs from unweighted, yet
    # round-trips through derive_dpk BITWISE at the SAME bpw (storage structure
    # unchanged -> real bpw identical; only PPL can move). This is the honesty
    # guarantee for DIRECTION (B): the weighted variant buys PPL, never bpw.
    RUN_STATE["cb_weight"] = "gptq"
    RUN_STATE["g"] = 256
    lin_w, br_w = _build_braq(W0, X, device)
    _run_captured(br_w)
    m1w, m2w, m3w = cap_masks["m"]
    Wq_w = lin_w.weight.data
    neq = int((Wq_w.view(torch.int16) != W_frozen256.view(torch.int16)
               ).sum().item())
    assert neq > 0, "selftest (f2): gptq-weighted@g=256 == unweighted@g=256"
    tensors_w, stats_w = derive_dpk(Wq_w, m1w, m2w, m3w, B=128, g=256)
    meta_w = container_meta(stats_w["R"], stats_w["C"], stats_w["C_orig"],
                            stats_w["NG"], "selftest-w", g=256)
    W2w = dpk_unpack.unpack(tensors_w, meta_w)
    assert torch.equal(
        W2w[:, :C].contiguous().view(torch.int16).to(device),
        Wq_w.contiguous().view(torch.int16)), \
        "selftest (f2): weighted DPK round-trip NOT bitwise"
    print(f"selftest (f2): gptq-weighted@g=256 differs from unweighted on "
          f"{neq}/{R * C}, DPK round-trip BITWISE, "
          f"bpw={stats_w['packed_bpw']:.4f}  PASS")
    RUN_STATE["cb_weight"] = "none"
    RUN_STATE["g"] = None

    # ------------------------------------------------------------------
    # (g) K28 intra-block GPTQ feedback (DIRECTION B)
    # ------------------------------------------------------------------
    # (g1) diagonal H => Hinv diagonal => intra-block feedback term Hinv1[i,i+1:]
    # is 0 => per-column re-snap on the un-fed-back column == the frozen
    # pre-quantize snap => intra-block-gptq@g=128 == pre-quantize@g=128 bitwise.
    RUN_STATE["g"] = 128
    ib_out = {}
    for ib in (False, True):
        RUN_STATE["intra_block"] = ib
        lin_g, br_g = _build_braq(W0, X, device)
        br_g.H = torch.eye(C, device=device) * 3.0     # diagonal H
        refit_fasterquant(br_g, blocksize=128, percdamp=0.01, partition=3,
                          orders=(1, 1, 1))
        ib_out[ib] = lin_g.weight.data.detach().clone()
    n_bad = int((ib_out[True].view(torch.int16)
                 != ib_out[False].view(torch.int16)).sum().item())
    assert n_bad == 0, (f"selftest (g1): diag-H intra-block@g=128 != "
                        f"pre-quantize@g=128 on {n_bad}/{R * C}")
    print("selftest (g1): diag-H intra-block-gptq@g=128 == pre-quantize "
          "BITWISE  PASS")

    # (g2) real (correlated) H: intra-block feedback engages -> differs from the
    # original pre-quantize output, yet round-trips through derive_dpk BITWISE
    # at the same bpw (final values are still the frozen group levels).
    RUN_STATE["g"] = 128
    RUN_STATE["intra_block"] = True
    lin_g2, br_g2 = _build_braq(W0, X, device)
    _run_captured(br_g2)
    m1g, m2g, m3g = cap_masks["m"]
    Wq_g2 = lin_g2.weight.data
    neq = int((Wq_g2.view(torch.int16)
               != lin_a.weight.data.view(torch.int16)).sum().item())
    assert neq > 0, "selftest (g2): intra-block@g=128 == pre-quantize (real H)"
    tensors_g, stats_g = derive_dpk(Wq_g2, m1g, m2g, m3g, B=128, g=128)
    meta_g = container_meta(stats_g["R"], stats_g["C"], stats_g["C_orig"],
                            stats_g["NG"], "selftest-ib", g=128)
    W2g = dpk_unpack.unpack(tensors_g, meta_g)
    assert torch.equal(
        W2g[:, :C].contiguous().view(torch.int16).to(device),
        Wq_g2.contiguous().view(torch.int16)), \
        "selftest (g2): intra-block DPK round-trip NOT bitwise"
    print(f"selftest (g2): intra-block-gptq@g=128 differs from pre-quantize on "
          f"{neq}/{R * C}, DPK round-trip BITWISE, "
          f"bpw={stats_g['packed_bpw']:.4f}  PASS")
    RUN_STATE["intra_block"] = False
    RUN_STATE["g"] = None

    # ------------------------------------------------------------------
    # (h) K30 --bulk-k-map per-sublayer bulk-K override
    # ------------------------------------------------------------------
    # Reference outputs at GLOBAL bulk K = 2 and K = 4 (no map).
    fake_name = "model.layers.0.self_attn.q_proj"
    href = {}
    RUN_STATE["g"] = 256
    for bk in (2, 4):
        RUN_STATE["partition_k"] = (bk, 4, 4)
        lin_h, br_h = _build_braq(W0, X, device)
        lin_h.global_name = MODEL_NAME + fake_name
        refit_fasterquant(br_h, blocksize=128, percdamp=0.01, partition=3,
                          orders=(1, 1, 1))
        href[bk] = lin_h.weight.data.detach().clone()
    assert int((href[2].view(torch.int16)
                != href[4].view(torch.int16)).sum().item()) > 0, \
        "selftest (h): bulk-K=2 output == bulk-K=4 output (K plumbing dead)"
    # (h1) map hits this layer: global K=2 + map{layer: 4} == global K=4.
    # (h2) map misses this layer: global K=2 + map{other: 4} == global K=2.
    for kmap, want, tag_h in (
            ({fake_name: 4}, 4, "hit"),
            ({"model.layers.99.mlp.gate_proj": 4}, 2, "miss")):
        RUN_STATE["partition_k"] = (2, 4, 4)
        RUN_STATE["bulk_k_map"] = kmap
        lin_h2, br_h2 = _build_braq(W0, X, device)
        lin_h2.global_name = MODEL_NAME + fake_name
        refit_fasterquant(br_h2, blocksize=128, percdamp=0.01, partition=3,
                          orders=(1, 1, 1))
        RUN_STATE["bulk_k_map"] = None
        n_bad = int((lin_h2.weight.data.view(torch.int16)
                     != href[want].view(torch.int16)).sum().item())
        assert n_bad == 0, (
            f"selftest (h/{tag_h}): map output != global bulk-K={want} "
            f"reference on {n_bad}/{R * C} weights")
        print(f"selftest (h/{tag_h}): bulk-k-map {tag_h} == global "
              f"bulk-K={want} BITWISE  PASS")
    RUN_STATE["partition_k"] = (4, 4, 4)
    RUN_STATE["g"] = None

    # ------------------------------------------------------------------
    # (i) K32 --rd-split K-aware RD bulk/tail split
    # ------------------------------------------------------------------
    # (i1) λ=0 at (bulk_k=2, hdiag, g=256): plumbing alive (output differs
    # from the rd=None reference at the same config) + DPK round-trip bitwise.
    RUN_STATE["g"] = 256
    RUN_STATE["partition_k"] = (2, 4, 4)
    RUN_STATE["cb_weight"] = "hdiag"
    rd_out, rd_minfos = {}, {}
    for lam in (None, 0.0, 1e12):
        RUN_STATE["rd_split"] = lam
        lin_i, br_i = _build_braq(W0, X, device)
        _run_captured(br_i)
        rd_out[lam] = (lin_i.weight.data.detach().clone(),
                       [m.clone() for m in cap_masks["m"]])
        rd_minfos[lam] = cap_masks["minfos"]
    RUN_STATE["rd_split"] = None
    neq = int((rd_out[0.0][0].view(torch.int16)
               != rd_out[None][0].view(torch.int16)).sum().item())
    assert neq > 0, "selftest (i1): rd-split λ=0 output == rd=None output " \
                    "(plumbing dead)"
    Wq_i, m_i = rd_out[0.0]
    tensors_i, stats_i = derive_dpk(Wq_i, m_i[0], m_i[1], m_i[2], B=128,
                                    g=256)
    meta_i = container_meta(stats_i["R"], stats_i["C"], stats_i["C_orig"],
                            stats_i["NG"], "selftest-rd", g=256)
    W2i = dpk_unpack.unpack(tensors_i, meta_i)
    assert torch.equal(
        W2i[:, :C].contiguous().view(torch.int16).to(device),
        Wq_i.contiguous().view(torch.int16)), \
        "selftest (i1): rd-split DPK round-trip NOT bitwise"
    print(f"selftest (i1): rd-split λ=0 differs from rd=None on "
          f"{neq}/{R * C}, DPK round-trip BITWISE, "
          f"bpw={stats_i['packed_bpw']:.4f}  PASS")

    # (i2) λ→∞ minimizes rate: every block picks the LARGEST bulk fraction
    # candidate; mean realized bulk frac strictly above the λ=0 one.
    fr = {lam: [m["rd_frac"] for m in rd_minfos[lam] if m and "rd_frac" in m]
          for lam in (0.0, 1e12)}
    assert fr[0.0] and fr[1e12], "selftest (i2): no rd_frac info captured"
    mean0 = sum(fr[0.0]) / len(fr[0.0])
    meaninf = sum(fr[1e12]) / len(fr[1e12])
    beta_inf = [m["rd_beta"] for m in rd_minfos[1e12]
                if m and "rd_beta" in m]
    cand_max = [max(m["rd_cands"]) for m in rd_minfos[1e12]
                if m and "rd_cands" in m]
    assert all(b == cm for b, cm in zip(beta_inf, cand_max)), \
        f"selftest (i2): λ=1e12 did not pick max-β everywhere: {beta_inf}"
    assert meaninf > mean0, (
        f"selftest (i2): mean bulk frac λ=1e12 ({meaninf:.4f}) not > "
        f"λ=0 ({mean0:.4f})")
    print(f"selftest (i2): mean bulk frac λ=0 {mean0:.4f} < λ=1e12 "
          f"{meaninf:.4f} (max-β everywhere)  PASS")

    # (i3) cb_weight none => uniform-weight search path also runs clean.
    RUN_STATE["cb_weight"] = "none"
    RUN_STATE["rd_split"] = 0.0
    lin_i3, br_i3 = _build_braq(W0, X, device)
    _run_captured(br_i3)
    n_hit = sum(1 for m in cap_masks["minfos"] if m and "rd_beta" in m)
    assert n_hit > 0, "selftest (i3): uniform-weight rd search never engaged"
    print(f"selftest (i3): uniform-weight rd search engaged on "
          f"{n_hit} blocks  PASS")
    RUN_STATE["rd_split"] = None
    RUN_STATE["partition_k"] = (4, 4, 4)
    RUN_STATE["g"] = None

    print("SELFTEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None,
                    help="HF model name (default Qwen/Qwen3-0.6B — the "
                         "pre-flag behavior). In --restore-dpk mode the "
                         "default is the dump manifest's recorded model; an "
                         "explicit --model that contradicts the manifest is "
                         "a hard error.")
    ap.add_argument("--g", default=None,
                    help="codebook group size: 128/256/512/1024/global")
    ap.add_argument("--gate-dir", default=None,
                    help="dir with reference <layer>.wq.safetensors for "
                         "bitwise gate (G0, use with --g 128)")
    ap.add_argument("--dump-dir", default=None,
                    help="emit DPK containers + wq ground truth here")
    ap.add_argument("--stats-only", action="store_true",
                    help="run derive_dpk per layer for measured packed bpw + "
                         "invariant validation, without writing dump files")
    ap.add_argument("--mmode", choices=("element", "column"),
                    default="element",
                    help="membership mode: element (m plane, original DOML "
                         "element-wise bulk/tail) or column (K2.6 E1 "
                         "column-wise bulk/tail + colmem stream)")
    ap.add_argument("--fresh-masks", action="store_true",
                    help="K2.6 E2: recompute snap/storage masks at block "
                         "time on the feedback-updated W (original DOML "
                         "timing); codebook fit stays at group start")
    ap.add_argument("--tag", default=None,
                    help="run tag for manifest/log naming (default g<g>)")
    # K27 quality probes — all OFF by default (default path byte-identical).
    ap.add_argument("--codebook-dtype",
                    choices=("bf16", "float8_e4m3fn", "float8_e5m2"),
                    default="bf16",
                    help="K27 probe 1: round codebook levels through this "
                         "dtype (default bf16 = no rounding)")
    ap.add_argument("--two-pass", choices=("none", "global"), default="none",
                    help="K27 probe 2: 'global' refits one codebook per "
                         "(row,partition) on the final W and re-snaps "
                         "(default none)")
    ap.add_argument("--merge-tail", action="store_true",
                    help="K27 probe 3: merge bulk+tail into one non-salient "
                         "partition, no bulk/tail split (default off)")
    ap.add_argument("--bulk-k", type=int, default=4,
                    help="K27 probe 4: #levels for the BULK partition (idx0). "
                         "Default 4; e.g. 2 => 1-bit codes for bulk weights")
    ap.add_argument("--tail-k", type=int, default=4,
                    help="K27 probe 4: #levels for the TAIL partition (idx1). Default 4")
    ap.add_argument("--salient-k", type=int, default=4,
                    help="K27 probe 4: #levels for the SALIENT partition (idx2). Default 4")
    ap.add_argument("--cb-weight", choices=("none", "hdiag", "gptq"),
                    default="none",
                    help="K28 (DIRECTION B): importance-weight the codebook "
                         "Lloyd-Max fit. none=unweighted (default, byte-"
                         "identical); hdiag=weight columns by damped H "
                         "diagonal; gptq=weight by 1/Hinv[j,j]^2 (the exact "
                         "GPTQ per-column loss coefficient)")
    ap.add_argument("--cb-weight-pow", type=float, default=1.0,
                    help="K28 (DIRECTION B): exponent p applied to the "
                         "codebook weight (w^p). p<1 tempers, p>1 sharpens. "
                         "Default 1.0")
    ap.add_argument("--intra-block-gptq", action="store_true",
                    help="K28 (DIRECTION B): add proper intra-block GPTQ error "
                         "feedback (column-by-column re-snap to the frozen "
                         "group codebook + residual propagation within the "
                         "block). Default off = original inter-block-only path")
    ap.add_argument("--refit-iters", type=int, default=1,
                    help="K31 (DIRECTION B'): number of OUTER joint "
                         "codebook<->GPTQ iterations per group. 1 = current "
                         "path (byte-identical). M>1 (requires "
                         "--intra-block-gptq): after the intra-block GPTQ "
                         "sweep captures per-weight assignments, re-center the "
                         "levels on the group-start values under those "
                         "assignments and re-run the sweep; final iteration's "
                         "result + inter-block feedback are kept. Storage/bpw "
                         "unchanged.")
    ap.add_argument("--bulk-k-map", default=None,
                    help="K30 mixed-K: path to a JSON dict {sublayer_name -> "
                         "K} (manifest names, e.g. 'model.layers.26.self_attn"
                         ".q_proj'). Listed sublayers get that BULK-partition "
                         "Lloyd K instead of --bulk-k; unlisted sublayers use "
                         "--bulk-k. Tail/salient K unchanged. Default None = "
                         "no override (byte-identical).")
    ap.add_argument("--bulk-frac", type=float, default=None,
                    help="K30: per-128-block, re-split the non-salient "
                         "(bulk∪tail) elements by |W| quantile so ≈β of them "
                         "(smallest |W|) become the NEW bulk and the rest the "
                         "NEW tail; salient mask kept exactly. Element mmode "
                         "only. Default None = original DOML split (untouched). "
                         "Composes with --bulk-k 2 (new bulk gets K=2).")
    ap.add_argument("--rd-split", type=float, default=None,
                    help="K32: λ for the K-aware RATE-aware per-block "
                         "bulk/tail split. Replaces the native non-salient "
                         "border with an explicit sweep over |W|-quantile "
                         "candidates β∈{0.40..0.85 step .05}, scored by "
                         "cost=D+λ·R where D is the col_w-weighted SSE of a "
                         "K_bulk-Lloyd fit on the candidate bulk + K_tail-"
                         "Lloyd fit on the candidate tail (per-row-per-block "
                         "proxy) and R = n_bulk·bits(K_bulk) + "
                         "n_tail·bits(K_tail) + n_ns·H2(bulk frac) bits. "
                         "λ=0 = distortion-optimal. Default None = original "
                         "split (byte-identical). Mutually exclusive with "
                         "--bulk-frac; element mmode only.")
    ap.add_argument("--rd-iters", type=int, default=8,
                    help="K32: Lloyd iterations for the --rd-split SEARCH "
                         "proxy fits (final codebook fit is unchanged at 20). "
                         "Default 8.")
    ap.add_argument("--restore-dpk", default=None,
                    help="K33 restore-eval mode: instead of quantizing, load "
                         "each sublayer's <layer>.wq.safetensors from this "
                         "dump dir into the model and let the standard run.py "
                         "eval run on the restored weights. Writes NO dumps; "
                         "all other quantizer flags are ignored.")
    ap.add_argument("--eval-extra-ppl", action="store_true",
                    help="restore mode only: pass --eval_extra_ppl through to "
                         "run.py (adds c4 + ptb PPL rows to the standard "
                         "wikitext2 eval). Default off.")
    ap.add_argument("--eval-arc", action="store_true",
                    help="restore mode only: pass --eval_arc through to "
                         "run.py (adds ARC-Easy + ARC-Challenge accuracy rows "
                         "after the standard wikitext2 eval). Default off.")
    ap.add_argument("--eval-mmlu", action="store_true",
                    help="restore mode only: pass --eval_mmlu through to "
                         "run.py. Default off.")
    ap.add_argument("--eval-hellaswag", action="store_true",
                    help="restore mode only: pass --eval_hellaswag through "
                         "to run.py. Default off.")
    args = ap.parse_args()
    if args.restore_dpk:
        if not args.run:
            ap.error("--restore-dpk requires --run")
        main_restore(args)
    elif args.selftest:
        main_selftest()
    elif args.run:
        if args.g is None:
            ap.error("--run requires --g")
        if args.g != "global" and int(args.g) % 128 != 0:
            ap.error("--g must be a multiple of 128 or 'global'")
        main_run(args)
    else:
        ap.error("choose --run or --selftest")
