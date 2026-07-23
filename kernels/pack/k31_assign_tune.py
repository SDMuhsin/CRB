"""K31 STAGE 2 — assignment relaxation (learned re-rounding) on DOML dumps.

Builds on kernels/pack/k31_block_tune.py (stage 1, levels-only): same
progressive block-reconstruction setup (quant-input -> FP-target block MSE,
run.py-standard wikitext2 nsamples=128 seed=0 calibration), but now the
per-weight LEVEL ASSIGNMENTS (the b0/b1 code planes) are learned too, at ZERO
bit cost:

  * membership plane `m`, salient bitmap `s`, meta JSON: byte-identical
    (partition of every weight is FROZEN);
  * code planes b0/b1: same raw bit-plane layout, values re-learned. Each
    weight may only select among the REAL levels of its own
    (row, group, partition) codebook (n_real from the ORIGINAL pre-btuned
    dump, where derive_dpk guarantees sorted-distinct levels; the -btuned dump
    may contain accidentally collapsed adjacent slots so its own adjacent-
    change count under-estimates n_real). Bulk (K=2) weights can therefore
    only commit to codes {0,1} => b1 stays 0 on bulk, tail/salient stay
    <= 2 bits: the k29 honest-bpw accounting is unchanged by construction
    (asserted per layer at write time: per-partition max code and max
    distinct-cb-slot count must be IDENTICAL to the source);
  * codebook payload `cb`: jointly tunable (same fp8-e4m3 STE + delta
    parameterization as stage 1, smaller lr), pads re-tied to the last REAL
    slot (n_real from the ORIGINAL dump).

Relaxation: per weight a 4-way softmax over its candidate slots (invalid /
pad slots masked to -1e4). For bulk only 2 slots are valid, so the softmax
degenerates to the sigmoid the K=2 case calls for — one uniform code path.
Two forward modes, chosen by a block-0 probe:
  soft    — W = sum_k p_k * lev_k, annealed entropy hardening
            (lambda(t) = reg_frac * mse_before * ramp(t)^2, ramp active over
            the last ~third of steps) so probabilities commit;
  hardste — forward uses the argmax one-hot, backward straight-through to
            the softmax (init loss == stage-1 state exactly, no soft->hard
            gap at all).
After optimization: HARD commit (argmax over valid slots), then a short
levels-only re-tune (stage 1's TunedQuantLinear, committed assignments) to
absorb the soft->hard gap; finally pads re-tied and the block frozen.

Outputs (default <src minus -btuned> + '-atuned'): per sublayer the source
.dpk with ONLY b0/b1 (repacked planes) and cb replaced — m/s/meta
byte-identical, verified — plus the matching new wq (bitwise equal to the
container unpack, asserted). manifest.json = source manifest + a
`k31_assign_tune` record.

Usage:
  block-0 probe (no files written):
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/k31_assign_tune.py \
          --src downloads/doml_dumps/qwen3-0.6b/k31-rdsplit-lam7e-5-g256-btuned \
          --max-blocks 1 --no-write
  full run:
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/k31_assign_tune.py \
          --src downloads/doml_dumps/qwen3-0.6b/k31-rdsplit-lam7e-5-g256-btuned
  byte-compare only:
      python kernels/pack/k31_assign_tune.py --src <btuned> --out <atuned> \
          --compare-only
"""

import argparse
import copy
import glob
import json
import math
import os
import sys
import time

REPO = "/workspace/BiLLM2"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import dpk_unpack  # noqa: E402
import k31_block_tune as kbt  # noqa: E402  (stage-1 machinery, reused)

FP8 = kbt.FP8
SUBLAYER_NAMES = kbt.SUBLAYER_NAMES
N_BLOCKS = kbt.N_BLOCKS
EXPECTED_SUBLAYERS = kbt.EXPECTED_SUBLAYERS
NEG = -1.0e4          # additive mask for invalid slots (fp32-safe, no NaN)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_nreal_orig(orig_dir, layer_name):
    """n_real [R, NG, 3] from the ORIGINAL dump's cb (sorted-distinct levels
    + last-level pads by derive_dpk => 1 + #adjacent-changes is exact)."""
    fp = os.path.join(orig_dir, f"{layer_name}.dpk.safetensors")
    with safe_open(fp, framework="pt", device="cpu") as f:
        cb = f.get_tensor("cb")
    cbb = cb.to(torch.bfloat16).view(torch.int16)
    chg = (cbb[..., 1:] != cbb[..., :-1]).to(torch.int64).sum(-1)
    return 1 + chg


def pack_plane(bits: torch.Tensor) -> torch.Tensor:
    """bool [R, C] -> uint32 [R, C/32] (LSB-first, doc 02 §2a). Round-trip
    asserted against dpk_unpack.expand_plane."""
    b = bits.to(torch.bool).cpu().numpy()
    packed = np.packbits(b, axis=-1, bitorder="little")      # [R, C/8] uint8
    words = np.ascontiguousarray(packed).view(np.uint32)     # [R, C/32]
    t = torch.from_numpy(words.view(np.int32).copy()).view(torch.uint32)
    back = dpk_unpack.expand_plane(t, bits.shape[1])
    if not torch.equal(back, bits.to(torch.bool).cpu()):
        raise RuntimeError("pack_plane round-trip failed")
    return t


def bpw_invariants(tensors, meta, C_orig):
    """Per-partition (max code used + 1, max distinct cb slots) over REAL
    columns — the two data-driven quantities k29_honest_bpw derives the
    per-layer code width and codebook slot count from."""
    C = meta["C"]
    real = (torch.arange(C) < C_orig).unsqueeze(0)
    b0 = dpk_unpack.expand_plane(tensors["b0"], C)
    b1 = dpk_unpack.expand_plane(tensors["b1"], C)
    code = b0.to(torch.int64) + 2 * b1.to(torch.int64)
    part = dpk_unpack.part_matrix(tensors, meta)
    cb = tensors["cb"].to(torch.bfloat16).view(torch.int16)
    distinct = 1 + (cb[..., 1:] != cb[..., :-1]).to(torch.int64).sum(-1)
    out = []
    for p in range(3):
        pm = (part == p) & real
        kc = (int(code[pm].max().item()) + 1) if bool(pm.any()) else 1
        kb = int(distinct[..., p].max().item())
        out.append((kc, kb))
    return out


# ---------------------------------------------------------------------------
# relaxed-assignment sublayer
# ---------------------------------------------------------------------------
ZETA, GAMMA_S = 1.1, -0.1        # AdaRound rectified-sigmoid stretch


class AssignLinear(nn.Module):
    """nn.Linear replacement with learnable assignments + levels.

    levels: master = lev0 + scale*delta, fp8-e4m3 STE (stage-1 semantics).
    assignments, three modes:
      soft/hardste — logits [R, C_orig, 4] over the 4 slots of each weight's
        (row, group, partition) codebook; slots >= n_real (from the ORIGINAL
        dump) are masked. soft = mixture forward; hardste = argmax forward
        with straight-through backward.
      pair — AdaRound-style: each weight chooses between its CURRENT slot
        and its nearest-by-value alternative REAL slot via a rectified
        sigmoid h = clamp(sigmoid(v)*(zeta-gamma_s)+gamma_s, 0, 1), which
        saturates EXACTLY to 0/1 at finite v (zero commit gap for saturated
        weights). Weights with a single real slot have no alternative and
        stay fixed.
    """

    def __init__(self, sub: kbt.SublayerCodebook, n_real_src, bias,
                 gamma, mode, h0=0.15):
        super().__init__()
        assert mode in ("soft", "hardste", "pair")
        self.mode = mode
        self.hard = (mode == "hardste")   # eval/forward hardness switch
        R, C = sub.R, sub.C_orig
        self.R, self.C = R, C
        dev = sub.idx.device

        code0 = sub.idx % 4                               # [R, C] int64
        base = sub.idx - code0
        part = (sub.idx // 4) % 3
        gidx = sub.idx // 12
        nr = n_real_src.to(dev)                           # [R, NG, 3]
        n_valid = nr[torch.arange(R, device=dev).unsqueeze(1), gidx, part]
        if not bool((code0 < n_valid).all()):
            bad = int((code0 >= n_valid).sum().item())
            raise RuntimeError(
                f"{sub.layer_name}: {bad} weights have code >= n_real(orig) "
                f"— original-dump n_real is wrong for this container")

        self.register_buffer("lev0", sub.lev0)
        self.register_buffer("scale", sub.scale)
        self.register_buffer("base", base)
        self.register_buffer("code0", code0.to(torch.uint8))
        self.register_buffer("valid",
                             torch.arange(4, device=dev).view(1, 1, 4)
                             < n_valid.unsqueeze(-1))     # [R, C, 4] bool
        self.register_buffer("arange4",
                             torch.arange(4, device=dev, dtype=torch.int64))
        # entropy normalization: only weights with >= 2 valid slots count
        act = n_valid >= 2
        inv_log = torch.zeros(R, C, dtype=torch.float32, device=dev)
        inv_log[act] = 1.0 / torch.log(n_valid[act].to(torch.float32))
        self.register_buffer("inv_log_nvalid", inv_log)
        self.n_active = int(act.sum().item())

        if mode == "pair":
            # nearest-by-value alternative REAL slot per weight
            lev4 = sub.lev0.gather(
                1, (base.unsqueeze(-1) + torch.arange(
                    4, device=dev)).reshape(R, -1)).reshape(R, C, 4)
            cur = lev4.gather(2, code0.unsqueeze(-1))       # [R, C, 1]
            diff = (lev4 - cur).abs()
            diff = diff.masked_fill(~self.valid, float("inf"))
            diff.scatter_(2, code0.unsqueeze(-1), float("inf"))
            alt = diff.argmin(dim=-1)                       # [R, C]
            has_alt = torch.isfinite(diff.gather(
                2, alt.unsqueeze(-1)).squeeze(-1))
            self.register_buffer("alt", alt.to(torch.uint8))
            self.register_buffer("has_alt", has_alt)
            self.n_alt = int(has_alt.sum().item())
            v0 = math.log((h0 - GAMMA_S) / (ZETA - h0))     # sigmoid inverse
            self.v = nn.Parameter(torch.full((R, C), v0, device=dev))
            del lev4, cur, diff
        else:
            logits = torch.zeros(R, C, 4, dtype=torch.float32, device=dev)
            logits.scatter_(2, code0.unsqueeze(-1), gamma)
            self.logits = nn.Parameter(logits)
        self.delta = nn.Parameter(torch.zeros_like(sub.lev0))
        self.bias = bias                                   # frozen

    # ---- levels (stage-1 semantics) ----
    def master(self):
        return self.lev0 + self.scale * self.delta

    def levels_ste(self):
        m = self.master()
        p = m.to(FP8).to(torch.float32)
        return m + (p - m).detach()

    # ---- assignments ----
    def masked_logits(self):
        return self.logits.masked_fill(~self.valid, NEG)

    def probs(self):
        return F.softmax(self.masked_logits(), dim=-1)

    def h_pair(self):
        h = torch.clamp(torch.sigmoid(self.v) * (ZETA - GAMMA_S) + GAMMA_S,
                        0.0, 1.0)
        return h * self.has_alt                            # [R, C]

    def reg_pair(self, beta):
        r = 1.0 - (2.0 * self.h_pair() - 1.0).abs().pow(beta)
        return (r * self.has_alt).sum() / max(self.n_alt, 1)

    def frac_unsaturated(self, tol=0.05):
        with torch.no_grad():
            h = self.h_pair()
            mid = (h > tol) & (h < 1.0 - tol) & self.has_alt
            return int(mid.sum().item())

    def committed_codes(self):
        if self.mode == "pair":
            use_alt = (self.h_pair() > 0.5) & self.has_alt
            return torch.where(use_alt, self.alt.to(torch.int64),
                               self.code0.to(torch.int64))
        return self.masked_logits().argmax(dim=-1)         # [R, C] int64

    def entropy_norm_mean(self):
        # log_softmax keeps logp FINITE (~NEG) on masked slots where p
        # underflows to exactly 0 — xlogy(p, p) there has an infinite
        # d/dp = log(p)+1 and NaNs the backward.
        logp = F.log_softmax(self.masked_logits(), dim=-1)
        H = -(logp.exp() * logp).sum(-1)                   # [R, C] nats
        return (H * self.inv_log_nvalid).sum() / max(self.n_active, 1)

    def _lev4(self):
        lev = self.levels_ste()                            # [R, NG*12]
        idx4 = (self.base.unsqueeze(-1) + self.arange4).reshape(self.R, -1)
        return lev.gather(1, idx4).reshape(self.R, self.C, 4)

    def weight_bf16(self):
        if self.mode == "pair":
            lev = self.levels_ste()
            levA = lev.gather(1, self.base + self.code0.to(torch.int64))
            levB = lev.gather(1, self.base + self.alt.to(torch.int64))
            if self.hard:
                use_alt = ((self.h_pair() > 0.5)
                           & self.has_alt).to(levA.dtype).detach()
                W = levA + use_alt * (levB - levA)
            else:
                W = levA + self.h_pair() * (levB - levA)
            return W.to(torch.bfloat16)
        lev4 = self._lev4()
        if self.hard:
            k = self.committed_codes()
            if self.training and self.mode == "hardste":
                p = self.probs()
                y = F.one_hot(k, 4).to(p.dtype)
                p_st = y + p - p.detach()                  # ST estimator
                W = (p_st * lev4).sum(-1)
            else:
                W = lev4.gather(2, k.unsqueeze(-1)).squeeze(-1)
        else:
            W = (self.probs() * lev4).sum(-1)
        return W.to(torch.bfloat16)

    def forward(self, x):
        return F.linear(x, self.weight_bf16(), self.bias)


def set_hard(qlin, hard: bool):
    for ql in qlin.values():
        ql.hard = hard or ql.mode == "hardste"


class _Shim:                    # minimal `sub` stand-in for TunedQuantLinear
    pass


# ---------------------------------------------------------------------------
# per-block stage-2 tuning
# ---------------------------------------------------------------------------
def lr_factor(step, total, floor=0.1):
    return floor + (1.0 - floor) * 0.5 * (
        1.0 + math.cos(math.pi * step / max(total, 1)))


def tune_block_assign(bi, block_fp, subs, nreals, x_q, target, kw, args,
                      device, log):
    """Returns (final frozen block, {name: cb fp8 [R,NG,3,4] cpu},
    {name: committed codes uint8 [R,C_orig] cpu}, stats)."""
    t0 = time.time()
    block_q = copy.deepcopy(block_fp).to(device)
    for p in block_q.parameters():
        p.requires_grad_(False)

    qlin = {}
    for name in SUBLAYER_NAMES:
        parent = block_q
        parts = name.split(".")
        for p_ in parts[:-1]:
            parent = getattr(parent, p_)
        lin = getattr(parent, parts[-1])
        assert isinstance(lin, nn.Linear), (name, type(lin))
        sub = subs[name]
        assert tuple(lin.weight.shape) == (sub.R, sub.C_orig), name
        ql = AssignLinear(sub, nreals[name], lin.bias, args.gamma,
                          args.mode, h0=args.h0).to(device)
        setattr(parent, parts[-1], ql)
        qlin[name] = ql

    # baseline = hard forward at init == exact stage-1 (-btuned) state
    set_hard(qlin, True)
    mse_before = kbt.stream_mse(block_q, x_q, target, kw)
    set_hard(qlin, args.mode == "hardste")
    for ql in qlin.values():
        ql.train()

    assign_params = [ql.v if args.mode == "pair" else ql.logits
                     for ql in qlin.values()]
    delta_params = [ql.delta for ql in qlin.values()]
    opt = torch.optim.Adam([
        {"params": assign_params, "lr": args.lr},
        {"params": delta_params, "lr": args.lr_lev},
    ])
    all_params = assign_params + delta_params

    gen = torch.Generator().manual_seed(20_000 + bi)
    N = x_q.shape[0]
    T = args.steps
    t_reg = int(args.reg_start * T)
    t_warm = int(args.warmup_frac * T)
    t_drift = int(args.drift_start * T)
    t_ph1 = int(args.two_phase_frac * T)

    # ---- G4 (TesseraQ graft): progressive confidence freezing -----------
    # pra_stages S > 0 (pair mode only): the T steps are split into S equal
    # stages; at each stage boundary the most-CONFIDENT (|h-0.5| largest)
    # still-unfrozen cells are HARD-COMMITTED (v saturated to ±V_SAT so
    # h ∈ {0,1} exactly) until the cumulative frozen fraction reaches s/S;
    # a gradient hook zeroes their updates. This replaces the decoupled
    # outward drift (disabled when S > 0) — it is the DOML-container analog
    # of TesseraQ's 20-threshold progressive adaptive rounding.
    pra_S = int(getattr(args, "pra_stages", 0) or 0)
    frozen = {}
    if pra_S > 0 and args.mode == "pair":
        t_drift = T + 1                       # drift OFF under PRA
        V_SAT = 20.0
        for name, ql in qlin.items():
            frozen[name] = torch.zeros_like(ql.v, dtype=torch.bool)

            def _mk_hook(_name):
                def _hook(g):
                    return g.masked_fill(frozen[_name], 0.0)
                return _hook

            ql.v.register_hook(_mk_hook(name))
        stage_bounds = [int(T * (s + 1) / pra_S) for s in range(pra_S)]

        def _freeze_to_fraction(frac):
            with torch.no_grad():
                for name, ql in qlin.items():
                    fr = frozen[name]
                    elig = ql.has_alt
                    n_elig = int(elig.sum().item())
                    if n_elig == 0:
                        continue
                    target_n = int(round(frac * n_elig))
                    cur_n = int((fr & elig).sum().item())
                    n_new = target_n - cur_n
                    if n_new <= 0:
                        continue
                    h = ql.h_pair()
                    conf = (h - 0.5).abs().masked_fill(~elig, -1.0)
                    conf = conf.masked_fill(fr, -1.0)   # already frozen
                    flat = conf.flatten()
                    idx = torch.topk(flat, n_new).indices
                    newly = torch.zeros_like(flat, dtype=torch.bool)
                    newly[idx] = True
                    newly = newly.view_as(fr)
                    ql.v.data = torch.where(
                        newly,
                        torch.where(h > 0.5,
                                    torch.full_like(ql.v, V_SAT),
                                    torch.full_like(ql.v, -V_SAT)),
                        ql.v.data)
                    frozen[name] = fr | newly

    loss_first = loss_last = None
    for step in range(T):
        if pra_S > 0 and args.mode == "pair" and step in stage_bounds:
            s_idx = stage_bounds.index(step) + 1
            _freeze_to_fraction(s_idx / pra_S)
            n_fr = sum(int(f.sum().item()) for f in frozen.values())
            print(f"  b{bi:02d} PRA stage {s_idx}/{pra_S}: frozen={n_fr}",
                  flush=True)
        f = lr_factor(step, T)
        opt.param_groups[0]["lr"] = args.lr * f
        opt.param_groups[1]["lr"] = (
            0.0 if step < t_ph1 else args.lr_lev * f)
        sel = torch.randperm(N, generator=gen)[:args.batch]
        out = kbt._block_forward(block_q, x_q[sel], kw)
        mse = (out.float() - target[sel].float()).pow(2).mean()
        loss = mse
        if args.mode == "soft" and args.reg_frac > 0 and step >= t_reg:
            ramp = ((step - t_reg) / max(T - t_reg, 1)) ** 2
            lam = args.reg_frac * max(mse_before, 1e-12) * ramp
            ent = sum(ql.entropy_norm_mean() for ql in qlin.values()) \
                / len(qlin)
            loss = loss + lam * ent
        elif args.mode == "pair" and args.reg_frac > 0 and step >= t_warm:
            # AdaRound: constant weight, beta annealed high -> low
            prog = (step - t_warm) / max(T - 1 - t_warm, 1)
            beta = args.beta_lo + 0.5 * (args.beta_hi - args.beta_lo) * (
                1.0 + math.cos(math.pi * prog))
            lam = args.reg_frac * max(mse_before, 1e-12)
            reg = sum(ql.reg_pair(beta) for ql in qlin.values()) / len(qlin)
            loss = loss + lam * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        opt.step()
        if args.mode == "pair" and args.drift_max > 0 and step >= t_drift:
            # decoupled (proximal) hardening: push each v OUTWARD from the
            # h=0.5 boundary (v=0) by an annealed step. Adam's per-parameter
            # normalization makes gradient-based penalties scale-fragile
            # (probes: mean-f_reg at lam x3 changed nothing); this drift is
            # scale-free — a weight stays unflipped/unsaturated only while
            # the block MSE actively defends it at full Adam step size, and
            # the schedule integral (~drift_max*(T-t_drift)/3 >> 2.4 logits)
            # guarantees h saturates exactly to {0,1} by the end.
            ramp = (step - t_drift) / max(T - 1 - t_drift, 1)
            d = args.drift_max * ramp * ramp
            with torch.no_grad():
                for ql in qlin.values():
                    ql.v.add_(torch.sign(ql.v) * d)
        lv = mse.item()
        if step == 0:
            loss_first = lv
        loss_last = lv
        if not (lv == lv):
            raise RuntimeError(f"block {bi}: NaN loss at step {step}")
        if args.log_every and (step % args.log_every == 0 or step == T - 1):
            print(f"  b{bi:02d} step {step:4d} mse {lv:.6e} "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    for ql in qlin.values():
        ql.eval()

    # post-training soft MSE (soft mode only), then hard commit
    mse_soft = None
    if args.mode in ("soft", "pair"):
        set_hard(qlin, False)
        mse_soft = kbt.stream_mse(block_q, x_q, target, kw)
    set_hard(qlin, True)
    mse_hard = kbt.stream_mse(block_q, x_q, target, kw)

    # commit stats + commitment measure at end
    n_flip, n_w, n_unsat = 0, 0, 0
    ent_end = 0.0
    codes_new = {}
    with torch.no_grad():
        for name in SUBLAYER_NAMES:
            ql = qlin[name]
            k = ql.committed_codes()
            n_flip += int((k != ql.code0.to(torch.int64)).sum().item())
            n_w += k.numel()
            if args.mode == "pair":
                n_unsat += ql.frac_unsaturated()
            else:
                ent_end += float(ql.entropy_norm_mean().item())
            codes_new[name] = k.to(torch.uint8).cpu()
    ent_end /= len(qlin)
    flip_rate = n_flip / n_w

    # ---- short levels-only re-tune with the COMMITTED assignments ----
    ql2map = {}
    if args.retune_steps > 0:
        params2 = []
        for name in SUBLAYER_NAMES:
            ql = qlin[name]
            sub = subs[name]
            with torch.no_grad():
                lev1 = ql.master().to(FP8).to(torch.float32)
                idx_new = ql.base + codes_new[name].to(
                    ql.base.device).to(torch.int64)
            shim = _Shim()
            shim.lev0, shim.scale, shim.idx = lev1, sub.scale, idx_new
            ql2 = kbt.TunedQuantLinear(shim, ql.bias).to(device)
            parent = block_q
            parts = name.split(".")
            for p_ in parts[:-1]:
                parent = getattr(parent, p_)
            setattr(parent, parts[-1], ql2)
            ql2map[name] = ql2
            params2.append(ql2.delta)
        opt2 = torch.optim.Adam(params2, lr=args.retune_lr)
        T2 = args.retune_steps
        for step in range(T2):
            opt2.param_groups[0]["lr"] = args.retune_lr * lr_factor(step, T2)
            sel = torch.randperm(N, generator=gen)[:args.batch]
            out = kbt._block_forward(block_q, x_q[sel], kw)
            loss = (out.float() - target[sel].float()).pow(2).mean()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params2, args.grad_clip)
            opt2.step()

    # ---- finalize: fp8 masters, pad re-tie (ORIG n_real), plain Linears ----
    final_cb = {}
    with torch.no_grad():
        for name in SUBLAYER_NAMES:
            sub = subs[name]
            src_master = (ql2map[name].master() if ql2map
                          else qlin[name].master())
            cb_f = kbt.retie_pads_fp8(src_master.to(FP8),
                                      nreals[name].to(device),
                                      sub.R, sub.NG)
            final_cb[name] = cb_f.cpu()
            idx_new = (qlin[name].base
                       + codes_new[name].to(device).to(torch.int64))
            W = cb_f.to(torch.bfloat16).reshape(
                sub.R, sub.NG * 12).gather(1, idx_new)
            lin = nn.Linear(sub.C_orig, sub.R,
                            bias=qlin[name].bias is not None)
            lin.weight = nn.Parameter(W, requires_grad=False)
            if qlin[name].bias is not None:
                lin.bias = nn.Parameter(qlin[name].bias.detach(),
                                        requires_grad=False)
            lin = lin.to(device=device, dtype=torch.bfloat16)
            lin.weight.data = W          # keep exact bf16 bits
            parent = block_q
            parts = name.split(".")
            for p_ in parts[:-1]:
                parent = getattr(parent, p_)
            setattr(parent, parts[-1], lin)

    mse_final = kbt.stream_mse(block_q, x_q, target, kw)
    stats = {
        "block": bi, "mode": args.mode, "steps": args.steps, "lr": args.lr,
        "lr_lev": args.lr_lev, "batch": args.batch, "gamma": args.gamma,
        "reg_frac": args.reg_frac, "retune_steps": args.retune_steps,
        "retune_lr": args.retune_lr,
        "mse_before": mse_before, "mse_soft": mse_soft, "mse_hard": mse_hard,
        "mse_final": mse_final,
        "mse_ratio_final": (mse_final / mse_before) if mse_before > 0
        else None,
        "hard_gap": (None if mse_soft is None else mse_hard - mse_soft),
        "loss_first": loss_first, "loss_last": loss_last,
        "flip_rate": flip_rate, "n_flipped": n_flip, "n_weights": n_w,
        "entropy_end": ent_end, "n_unsaturated": n_unsat,
        "wall_s": round(time.time() - t0, 1),
    }
    print(f"K31A block {bi:2d}: MSE before {mse_before:.6e} "
          f"soft {('%.6e' % mse_soft) if mse_soft is not None else '—'} "
          f"hard {mse_hard:.6e} final {mse_final:.6e} "
          f"(x{mse_final / mse_before:.4f}) flips {n_flip}/{n_w} "
          f"({100 * flip_rate:.2f}%) ent_end {ent_end:.4f} "
          f"unsat {n_unsat} t={stats['wall_s']}s", flush=True)
    log.append(stats)
    return block_q, final_cb, codes_new, stats


# ---------------------------------------------------------------------------
# dump writing + byte-compare
# ---------------------------------------------------------------------------
def write_assign_layer(out_dir, sub: kbt.SublayerCodebook, cb_new_fp8,
                       codes_new_u8):
    """Write <name>.dpk with b0/b1 + cb replaced (m/s/meta byte-identical)
    and the matching new wq. Asserts (a) container unpack == direct gather
    bitwise, (b) k29 honest-bpw invariants identical to the source layer."""
    R, C, C_orig = sub.R, sub.C, sub.C_orig
    assert cb_new_fp8.dtype == FP8 and \
        tuple(cb_new_fp8.shape) == (R, sub.NG, 3, 4)
    assert tuple(codes_new_u8.shape) == (R, C_orig)

    b0_old = dpk_unpack.expand_plane(sub.src_tensors["b0"], C)
    b1_old = dpk_unpack.expand_plane(sub.src_tensors["b1"], C)
    code = b0_old.to(torch.int64) + 2 * b1_old.to(torch.int64)
    code[:, :C_orig] = codes_new_u8.to(torch.int64)   # pad cols: unchanged
    tensors = {
        "b0": pack_plane((code & 1).to(torch.bool)),
        "b1": pack_plane((code >> 1).to(torch.bool)),
        "m": sub.src_tensors["m"],
        "s": sub.src_tensors["s"],
        "cb": cb_new_fp8.contiguous(),
    }
    inv_old = bpw_invariants(sub.src_tensors, sub.meta, C_orig)
    inv_new = bpw_invariants(tensors, sub.meta, C_orig)
    if inv_old != inv_new:
        raise RuntimeError(
            f"{sub.layer_name}: honest-bpw invariants changed "
            f"{inv_old} -> {inv_new} (per-partition (kmax_code, kmax_cb)) — "
            f"the honest bpw would move; refusing to write")

    W_new = dpk_unpack.unpack(tensors, sub.meta)[:, :C_orig].contiguous()
    idx_new = (sub.idx.cpu() - sub.idx.cpu() % 4) \
        + codes_new_u8.to(torch.int64)
    W_dir = cb_new_fp8.to(torch.bfloat16).reshape(
        R, sub.NG * 12).gather(1, idx_new)
    if not torch.equal(W_new.view(torch.int16), W_dir.view(torch.int16)):
        raise RuntimeError(f"{sub.layer_name}: tuned container unpack != "
                           f"direct gather (code planes broken)")
    dpk_path = os.path.join(out_dir, f"{sub.layer_name}.dpk.safetensors")
    wq_path = os.path.join(out_dir, f"{sub.layer_name}.wq.safetensors")
    save_file(tensors, dpk_path, metadata={"meta": sub.meta_json})
    save_file({"wq": W_new}, wq_path, metadata={"meta": sub.meta_json})


def byte_compare_stage2(src_dir, out_dir):
    """Prove m/s/meta byte-identical source<->tuned; ONLY b0/b1/cb may
    differ. Reports code flip rate (real columns) + cb slots changed."""
    files = sorted(glob.glob(os.path.join(src_dir, "*.dpk.safetensors")))
    assert files, src_dir
    n_layers = 0
    slots_changed = slots_total = 0
    flips = weights = 0
    for fp in files:
        name = os.path.basename(fp)
        fp2 = os.path.join(out_dir, name)
        if not os.path.exists(fp2):
            raise RuntimeError(f"byte-compare: missing {fp2}")
        with safe_open(fp, framework="pt", device="cpu") as fa, \
                safe_open(fp2, framework="pt", device="cpu") as fb:
            if fa.metadata()["meta"] != fb.metadata()["meta"]:
                raise RuntimeError(f"{name}: meta JSON differs")
            if set(fa.keys()) != set(fb.keys()):
                raise RuntimeError(f"{name}: container keys differ")
            meta = json.loads(fa.metadata()["meta"])
            C, C_orig = meta["C"], meta["C_orig"]
            for k in ("m", "s"):
                ta, tb = fa.get_tensor(k), fb.get_tensor(k)
                if ta.dtype != tb.dtype or ta.shape != tb.shape or \
                        not torch.equal(ta.view(torch.int32),
                                        tb.view(torch.int32)):
                    raise RuntimeError(
                        f"{name}: tensor '{k}' NOT byte-identical — stage 2 "
                        f"must never touch membership/salient bitmaps")
            ca = fa.get_tensor("cb").view(torch.int8)
            cbt = fb.get_tensor("cb").view(torch.int8)
            if ca.shape != cbt.shape:
                raise RuntimeError(f"{name}: cb shape changed")
            slots_changed += int((ca != cbt).sum().item())
            slots_total += ca.numel()
            code_a = (dpk_unpack.expand_plane(fa.get_tensor("b0"), C).long()
                      + 2 * dpk_unpack.expand_plane(fa.get_tensor("b1"),
                                                    C).long())[:, :C_orig]
            code_b = (dpk_unpack.expand_plane(fb.get_tensor("b0"), C).long()
                      + 2 * dpk_unpack.expand_plane(fb.get_tensor("b1"),
                                                    C).long())[:, :C_orig]
            flips += int((code_a != code_b).sum().item())
            weights += code_a.numel()
        n_layers += 1
    summary = {"n_layers": n_layers, "cb_slots_changed": slots_changed,
               "cb_slots_total": slots_total, "code_flips": flips,
               "n_weights": weights, "flip_rate": flips / weights}
    print(f"K31A byte-compare: {n_layers} sublayers — m/s/meta all "
          f"byte-identical; cb slots changed {slots_changed}/{slots_total} "
          f"({100.0 * slots_changed / slots_total:.2f}%); code flips "
          f"{flips}/{weights} ({100.0 * flips / weights:.3f}%)", flush=True)
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="source dump dir (the stage-1 -btuned dump)")
    ap.add_argument("--orig", default=None,
                    help="ORIGINAL (pre-btuned) dump dir for n_real "
                         "(default: manifest k31_block_tune.src_dir)")
    ap.add_argument("--out", default=None,
                    help="tuned dump dir (default: <src minus -btuned>"
                         "-atuned)")
    ap.add_argument("--model", default=None,
                    help="HF model name; default = the src manifest's "
                         "recorded model (falling back to Qwen/Qwen3-0.6B). "
                         "A --model that contradicts the manifest is a hard "
                         "error.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", choices=("soft", "hardste", "pair"),
                    default="soft")
    ap.add_argument("--h0", type=float, default=0.1,
                    help="pair mode: initial rectified-sigmoid value "
                         "(toward-alternative fraction)")
    ap.add_argument("--beta-hi", type=float, default=18.0)
    ap.add_argument("--beta-lo", type=float, default=2.0)
    ap.add_argument("--warmup-frac", type=float, default=0.2,
                    help="pair mode: fraction of steps before the AdaRound "
                         "f_reg switches on")
    ap.add_argument("--drift-max", type=float, default=0.05,
                    help="pair mode: decoupled hardening drift on v "
                         "(logits/step at ramp end; 0 disables)")
    ap.add_argument("--drift-start", type=float, default=0.5,
                    help="pair mode: fraction of steps before the drift "
                         "ramp starts")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--nsamples", type=int, default=128,
                    help="calibration samples (G4 graft: TesseraQ uses 512; "
                         "default 128 = historical behavior)")
    ap.add_argument("--pra-stages", type=int, default=0,
                    help="G4 graft (pair mode): >0 enables TesseraQ-style "
                         "progressive confidence freezing over this many "
                         "equal stages (disables the outward drift). "
                         "0 = off (byte-identical historical path).")
    ap.add_argument("--lr", type=float, default=3e-2,
                    help="Adam lr on assignment logits")
    ap.add_argument("--lr-lev", type=float, default=1e-3,
                    help="Adam lr on level deltas (stage-1 lr was 1e-2; "
                         "0 = assignments only)")
    ap.add_argument("--two-phase-frac", type=float, default=0.0,
                    help=">0: freeze levels for the first FRAC of steps")
    ap.add_argument("--gamma", type=float, default=4.0,
                    help="init logit advantage of the current assignment")
    ap.add_argument("--reg-frac", type=float, default=0.02,
                    help="soft mode: max entropy-penalty weight as a "
                         "fraction of the block's initial MSE")
    ap.add_argument("--reg-start", type=float, default=0.65,
                    help="soft mode: fraction of steps after which the "
                         "hardening ramp starts")
    ap.add_argument("--retune-steps", type=int, default=80,
                    help="post-commit levels-only re-tune steps")
    ap.add_argument("--retune-lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stream-chunk", type=int, default=None,
                    help="sequences per no-grad stream forward (VRAM knob; "
                         "bit-neutral; default = k31_block_tune's 8)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="default = all blocks of the resolved model")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src)
    assert os.path.isdir(src_dir), src_dir
    if args.out:
        out_dir = os.path.abspath(args.out)
    elif src_dir.endswith("-btuned"):
        out_dir = src_dir[:-len("-btuned")] + "-atuned"
    else:
        out_dir = src_dir + "-atuned"
    assert out_dir != src_dir

    if args.compare_only:
        byte_compare_stage2(src_dir, out_dir)
        return

    if args.orig:
        orig_dir = os.path.abspath(args.orig)
    else:
        with open(os.path.join(src_dir, "manifest.json")) as f:
            man = json.load(f)
        orig_dir = man.get("k31_block_tune", {}).get("src_dir")
        assert orig_dir and os.path.isdir(orig_dir), (
            "cannot infer --orig from manifest; pass it explicitly")
    print(f"K31A: src={src_dir}\nK31A: orig(n_real)={orig_dir}\n"
          f"K31A: out={out_dir}", flush=True)

    kbt.MODEL_NAME = kbt.resolve_model(src_dir, args.model)
    # 2026-07-20: refresh the import-time copies of the layer geometry from
    # the model config (4B has 36 blocks; the old 28/196 was a coincidence)
    global N_BLOCKS, EXPECTED_SUBLAYERS
    kbt.set_layer_geometry(kbt.MODEL_NAME)
    N_BLOCKS = kbt.N_BLOCKS
    EXPECTED_SUBLAYERS = kbt.EXPECTED_SUBLAYERS
    if args.stream_chunk:
        kbt.STREAM_CHUNK = args.stream_chunk
    print(f"K31A: model = {kbt.MODEL_NAME} (blocks={N_BLOCKS}, "
          f"sublayers={EXPECTED_SUBLAYERS}, "
          f"stream_chunk={kbt.STREAM_CHUNK})", flush=True)

    device = args.device
    t_start = time.time()
    torch.manual_seed(0)

    print("K31A: loading model + calibration (standard run.py path)...",
          flush=True)
    model, dataloader = kbt.load_model_and_calib(device,
                                                 nsamples=args.nsamples)
    inps, layer_kwargs = kbt.capture_block0_inputs(model, dataloader, device)
    layers = model.model.layers
    assert len(layers) == N_BLOCKS
    print(f"K31A: captured {inps.shape} block-0 inputs", flush=True)
    # Park the whole model on CPU after capture: with --nsamples 512 at 4B
    # the resident model (8 GB) + sample buffers otherwise OOM a 46 GB card
    # during backward (observed 2026-07-23). The block loop moves each block
    # to the device on demand and parks it back afterwards; captured inps /
    # layer_kwargs tensors are separate device tensors and stay put.
    model.cpu()
    torch.cuda.empty_cache()

    x_fp = inps
    x_q = inps.clone()
    tune_log = []
    all_final_cb = {}
    all_codes = {}
    all_subs = {}

    n_blocks = min(args.max_blocks or N_BLOCKS, N_BLOCKS)
    for bi in range(n_blocks):
        block_fp = layers[bi].to(device)
        subs, nreals = {}, {}
        for name in SUBLAYER_NAMES:
            lname = f"model.layers.{bi}.{name}"
            sub = kbt.SublayerCodebook(src_dir, lname, device=device)
            lin_w = dict(block_fp.named_modules())[name].weight
            assert tuple(lin_w.shape) == (sub.R, sub.C_orig), name
            subs[name] = sub
            nreals[name] = load_nreal_orig(orig_dir, lname)

        target = kbt.forward_stream(block_fp, x_fp, layer_kwargs)
        block_q, final_cb, codes_new, _stats = tune_block_assign(
            bi, block_fp, subs, nreals, x_q, target, layer_kwargs, args,
            device, tune_log)
        x_q = kbt.forward_stream(block_q, x_q, layer_kwargs)
        x_fp = target
        for name in SUBLAYER_NAMES:
            lname = f"model.layers.{bi}.{name}"
            all_final_cb[lname] = final_cb[name]
            all_codes[lname] = codes_new[name]
            sub = subs[name]
            # park GPU tensors on CPU; write path only needs cpu tensors
            sub.idx = sub.idx.cpu()
            sub.lev0 = sub.lev0.cpu()
            sub.scale = sub.scale.cpu()
            all_subs[lname] = sub

        del block_q, target, subs
        layers[bi] = layers[bi].cpu()
        torch.cuda.empty_cache()

    # chunked: full fp32 materialization of two [N,2048,H] tensors is 2x10.7
    # GiB at nsamples=512 on 4B and OOMs (observed 2026-07-23)
    _se = _ref = 0.0
    for _st in range(0, x_q.shape[0], 8):
        _a = x_q[_st:_st + 8].float()
        _b = x_fp[_st:_st + 8].float()
        _se += (_a - _b).pow(2).sum().item()
        _ref += _b.pow(2).sum().item()
    final_mse = _se / x_fp.numel()
    ref = _ref / x_fp.numel()
    print(f"K31A: final-stream MSE after block {n_blocks - 1}: "
          f"{final_mse:.6e} (rel {final_mse / ref:.4e})", flush=True)
    fr = [b["flip_rate"] for b in tune_log]
    mr = [b["mse_ratio_final"] for b in tune_log]
    print(f"K31A: flip rate min={min(fr):.4f} median="
          f"{sorted(fr)[len(fr) // 2]:.4f} max={max(fr):.4f}; "
          f"MSE ratio min={min(mr):.4f} median="
          f"{sorted(mr)[len(mr) // 2]:.4f} max={max(mr):.4f}", flush=True)

    if args.no_write:
        print(f"K31A: --no-write set; tuned {len(all_final_cb)} sublayers, "
              f"nothing written. wall={time.time() - t_start:.0f}s")
        print(json.dumps(tune_log, indent=1))
        return

    if n_blocks != N_BLOCKS:
        raise SystemExit("K31A FATAL: refusing to write a dump from a "
                         "partial run (use --no-write for smoke tests)")
    assert len(all_final_cb) == EXPECTED_SUBLAYERS

    os.makedirs(out_dir, exist_ok=True)
    for lname in all_final_cb:
        write_assign_layer(out_dir, all_subs[lname], all_final_cb[lname],
                           all_codes[lname])
    print(f"K31A: wrote {len(all_final_cb)} tuned sublayers -> {out_dir}",
          flush=True)

    cmp_summary = byte_compare_stage2(src_dir, out_dir)

    with open(os.path.join(src_dir, "manifest.json")) as f:
        manifest = json.load(f)
    manifest["k31_assign_tune"] = {
        "src_dir": src_dir, "orig_dir": orig_dir,
        "model": kbt.MODEL_NAME,
        "mode": args.mode, "steps": args.steps, "lr": args.lr,
        "lr_lev": args.lr_lev, "two_phase_frac": args.two_phase_frac,
        "gamma": args.gamma, "reg_frac": args.reg_frac,
        "reg_start": args.reg_start, "h0": args.h0,
        "beta_hi": args.beta_hi, "beta_lo": args.beta_lo,
        "warmup_frac": args.warmup_frac,
        "drift_max": args.drift_max, "drift_start": args.drift_start,
        "retune_steps": args.retune_steps,
        "retune_lr": args.retune_lr, "batch": args.batch,
        "grad_clip": args.grad_clip,
        "nsamples": args.nsamples, "pra_stages": args.pra_stages,
        "parameterization": (
            "pair: AdaRound rectified-sigmoid (zeta=1.1, gamma_s=-0.1) "
            "between current slot and nearest-by-value REAL alternative "
            "(n_real from orig dump), f_reg=(1-|2h-1|^beta) annealed"
            if args.mode == "pair" else
            "per-weight 4-way masked softmax over REAL slots (n_real from "
            "orig dump)")
        + " + stage-1 delta/fp8-STE levels; hard commit + levels-only "
          "re-tune",
        "calib": (f"wikitext2 nsamples={args.nsamples} seed=0 seqlen=2048 "
                  f"(run.py path)"),
        "byte_compare": cmp_summary,
        "final_stream_mse": final_mse,
        "blocks": tune_log,
        "wall_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"K31A: DONE. tuned dump = {out_dir}  "
          f"wall={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
