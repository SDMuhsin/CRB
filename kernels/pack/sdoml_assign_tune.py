"""SDOML STAGE 2 — assignment relaxation (learned re-rounding) on SDOML dumps.

Port of kernels/pack/k31_assign_tune.py (the DOML stage-2) to the SDOML BASE
container (sdoml_dump.py).  Builds on kernels/pack/sdoml_block_tune.py (stage 1,
levels-only): same progressive block-reconstruction setup (quant-input ->
FP-target block MSE, run.py-standard wikitext2 nsamples=128 seed=0 seqlen=2048
calibration, two activation streams), but now the per-KEPT-weight CODE
ASSIGNMENT (which of the block's K=4 codebook levels each kept weight uses) is
LEARNED too, jointly with a small level re-tune, at ZERO bit cost.

Why zero bit cost (SDOML is STRUCTURALLY simpler than DOML here):
  the honest bpw (sdoml_honest_bpw.py) counts, per layer,
      mask bitmap  R*C bits (byte-identical, FROZEN),
      code stream  n_kept * ceil(log2 K) bits  (PADDED to K: 2 bits/kept),
      codebook     K * cb_bits/level * R * NG  (PADDED to K),
  NONE of which depends on WHICH of the K levels a weight picks nor on how many
  distinct levels a block uses.  So with the keep-mask (hence n_kept), K, NG, R,
  C, block_widths and cb_dtype all held identical to the source, the honest bpw
  is unchanged BY CONSTRUCTION (2.2500).  There is therefore no n_real / valid-
  slot machinery (unlike DOML's variable per-partition width): all K=4 slots are
  legal assignment candidates for every kept weight.

FROZEN: the keep mask (pruned positions stay EXACTLY 0 — they are never an
assignment candidate); NG/K/R/C/block_widths; the mask_packed bytes.
Relaxation: per KEPT weight a K=4-way softmax over its (row, 128-block)
codebook's 4 levels.  Two forward modes chosen by a block-0 probe:
  soft    — W = sum_k p_k * lev_k, annealed entropy hardening
            (lambda(t) = reg_frac * mse_before * ramp(t)^2 over the last
            ~third of steps) so the probabilities commit;
  hardste — forward uses the argmax one-hot, backward straight-through to the
            softmax (init loss == stage-1 state exactly, no soft->hard gap).
An optional `pair` mode (AdaRound rectified-sigmoid between the current slot and
its nearest-by-value alternative REAL slot — K31's winning recipe) is also
ported.  Codebook `cb` is jointly tunable via the SAME fp8-e4m3 STE + relative-
delta parameterization as stage 1 (smaller lr).  After optimization: HARD commit
(argmax), a short levels-only re-tune (stage-1's TunedQuantLinear with the
committed assignments), then the block is frozen.

Write-out (default <src minus -btuned> + '-atuned'): per layer the NEW `cb`
(fp8-e4m3) and NEW `wq` where wq[kept] = fp8_cb[committed_code] (bf16),
wq[pruned] = 0, DERIVED from the SAME fp8-projected+canonicalized levels stored
in `cb` so reassemble_bitwise is bitwise-consistent by construction;
`mask_packed` byte-identical to source.  Reuses sdoml_block_tune.write_tuned_layer
(with sub.idx repointed to the committed assignment) so all 3 container gates
run: (a) mask bytes identical, (b) reassemble bitwise, (c) NG/K/R/C/block_widths
identical (=> honest bpw structurally unchanged = still 2.25).

Usage:
  synthetic roundtrip selftest (no GPU / no real dump):
      python kernels/pack/sdoml_assign_tune.py --selftest-roundtrip
  block-0 probe (writes nothing):
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/sdoml_assign_tune.py \
          --src downloads/doml_dumps/qwen3-0.6b/sdoml-s50-btuned \
          --max-blocks 1 --no-write [--mode soft|hardste|pair]
  full run:
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/sdoml_assign_tune.py \
          --src downloads/doml_dumps/qwen3-0.6b/sdoml-s50-btuned
  re-run the 3 container gates on an existing atuned dir:
      python kernels/pack/sdoml_assign_tune.py --src <btuned> --out <atuned> \
          --verify-only
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

import sdoml_dump  # noqa: E402
import sdoml_honest_bpw  # noqa: E402
import sdoml_block_tune as sbt  # noqa: E402  (stage-1 machinery, reused)

FP8 = sbt.FP8
MODEL_NAME = sbt.MODEL_NAME
SUBLAYER_NAMES = sbt.SUBLAYER_NAMES
N_BLOCKS = sbt.N_BLOCKS
EXPECTED_SUBLAYERS = sbt.EXPECTED_SUBLAYERS
ZETA, GAMMA_S = 1.1, -0.1        # AdaRound rectified-sigmoid stretch (pair mode)


# ---------------------------------------------------------------------------
# fp8-tolerant SDPK loader + differentiable sublayer view
#   (mirror of sdoml_block_tune.SublayerCodebook, but load_layer there hard-
#    requires bf16 cb; the -btuned source we consume stores fp8-e4m3 cb, so we
#    replicate the load with an fp8/bf16-tolerant reader.  sdoml_dump.py and
#    sdoml_block_tune.py stay byte-identical.)
# ---------------------------------------------------------------------------
def load_sdpk(path, device="cpu"):
    """Load an SDPK container; accepts a bf16 OR an fp8-e4m3 codebook.
    Returns (wq bf16 [R,C], mask bool [R,C], cb [R,NG,K], meta)."""
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        exp = {"wq", "mask_packed", "cb"}
        if keys != exp:
            raise ValueError(f"{path}: keys {sorted(keys)} != {sorted(exp)}")
        md = f.metadata()
        if md is None or "meta" not in md:
            raise ValueError(f"{path}: missing JSON meta blob")
        meta = json.loads(md["meta"])
        wq = f.get_tensor("wq").to(device)
        mask_packed = f.get_tensor("mask_packed")
        cb = f.get_tensor("cb").to(device)
    R, C = meta["R"], meta["C"]
    if wq.dtype != torch.bfloat16 or tuple(wq.shape) != (R, C):
        raise ValueError(f"{path}: wq is {wq.dtype}{tuple(wq.shape)}")
    if cb.dtype not in (torch.bfloat16, FP8):
        raise ValueError(f"{path}: cb dtype {cb.dtype} not in (bf16, fp8-e4m3)")
    if cb.shape[0] != R or cb.shape[1] != meta["NG"] or cb.shape[2] != meta["K"]:
        raise ValueError(f"{path}: cb shape {tuple(cb.shape)} vs meta")
    mask = sdoml_dump.unpack_mask(mask_packed, R, C).to(device)
    return wq, mask, cb, meta


class SublayerCodebook:
    """Loads <name>.sdpk (fp8 or bf16 cb); exposes the same interface as
    sdoml_block_tune.SublayerCodebook so it drops into sbt.TunedQuantLinear and
    sbt.write_tuned_layer unchanged:
      lev0   fp32 [R, NG*K]  master init (exact source levels)
      idx    int64 [R, C]    flat slot index = block*K + code  (code = argmin)
      mask   bool  [R, C]    keep-mask
      scale  fp32 [R, NG*K]  relative-step scale (|lev0| floored)
    Hard-asserts on load: where(mask, gather(lev0->bf16, idx), 0) == wq BITWISE.
    """

    def __init__(self, dump_dir, layer_name, device="cpu"):
        self.layer_name = layer_name
        path = os.path.join(dump_dir, f"{layer_name}.sdpk.safetensors")
        wq, mask, cb, meta = load_sdpk(path, device="cpu")
        R, C, NG, K = meta["R"], meta["C"], meta["NG"], meta["K"]
        self.meta = meta
        self.block_widths = list(meta["block_widths"])
        self.R, self.C, self.NG, self.K = R, C, NG, K
        # codebook granularity (columns per codebook), DEFAULT 128. idx/base
        # below are derived from block_widths (already group-general); this read
        # honors an explicit `groupsize` meta key (g256 etc.) and guards a
        # malformed container.
        self.groupsize = int(meta.get("groupsize", meta.get("blocksize", 128)))
        if self.block_widths and max(self.block_widths) > self.groupsize:
            raise RuntimeError(
                f"{layer_name}: block_widths max {max(self.block_widths)} > "
                f"groupsize {self.groupsize}")
        with safe_open(path, framework="pt", device="cpu") as f:
            self.src_mask_packed = f.get_tensor("mask_packed")
            self.src_meta_json = f.metadata()["meta"]

        # FROZEN code plane via independent argmin decode (asserts recon==wq
        # bitwise and wq[~mask]==0 exactly).
        code_plane, _recon = sdoml_honest_bpw.decode_codes(
            wq, mask, cb.to(torch.bfloat16), self.block_widths)      # [R,C]

        gidx = torch.zeros(C, dtype=torch.int64)
        off = 0
        for b, w_b in enumerate(self.block_widths):
            gidx[off:off + w_b] = b
            off += w_b
        idx = (gidx.unsqueeze(0) * K + code_plane).contiguous()      # [R,C]

        lev0 = cb.to(torch.float32).reshape(R, NG * K).contiguous()

        Wb = lev0.to(torch.bfloat16).gather(1, idx)
        Wb = torch.where(mask, Wb, torch.zeros_like(Wb))
        eq = Wb.view(torch.int16) == wq.contiguous().view(torch.int16)
        if not bool(eq.all()):
            bad = int((~eq).sum().item())
            raise RuntimeError(
                f"{layer_name}: (levels, idx, mask) roundtrip NOT bitwise vs "
                f"wq ({bad}/{R * C} mismatches)")

        self.wq = wq
        self.lev0 = lev0.to(device)
        self.idx = idx.to(device)
        self.mask = mask.to(device)

        a = lev0.abs()
        ref = torch.zeros(R, NG * K, dtype=torch.bool)
        ref.scatter_(1, idx, True)
        self.referenced = ref
        nz = a[ref & (a > 0)]
        floor = (0.05 * nz.median().item()) if nz.numel() else 1e-4
        self.scale = a.clamp(min=floor).to(device)
        self.n_ref_slots = int(ref.sum().item())


# ---------------------------------------------------------------------------
# relaxed-assignment sublayer
# ---------------------------------------------------------------------------
class AssignLinear(nn.Module):
    """nn.Linear replacement with learnable code assignments + levels for one
    SDOML sublayer.  Pruned positions (mask=0) are forced to weight 0 and are
    never assignment candidates.

    levels: master = lev0 + scale*delta, fp8-e4m3 STE (stage-1 semantics).
    assignments, modes:
      soft/hardste — logits [R, C, K] over each KEPT weight's K codebook slots
        (all K are legal; honest bpw is padded-K so no slot is masked). soft =
        mixture forward; hardste = argmax forward, straight-through backward.
      pair — each KEPT weight chooses between its CURRENT slot and its nearest-
        by-value alternative slot via a rectified sigmoid
        h = clamp(sigmoid(v)*(zeta-gamma_s)+gamma_s, 0, 1) (saturates EXACTLY to
        0/1 at finite v).
    """

    def __init__(self, sub: SublayerCodebook, bias, gamma, mode, h0=0.1):
        super().__init__()
        assert mode in ("soft", "hardste", "pair")
        self.mode = mode
        self.hard = (mode == "hardste")
        R, C, NG, K = sub.R, sub.C, sub.NG, sub.K
        self.R, self.C, self.NG, self.K = R, C, NG, K
        dev = sub.idx.device

        code0 = sub.idx % K                                  # [R, C] int64
        base = sub.idx - code0                               # = gidx*K
        self.register_buffer("lev0", sub.lev0)
        self.register_buffer("scale", sub.scale)
        self.register_buffer("base", base)
        self.register_buffer("code0", code0.to(torch.uint8))
        self.register_buffer("keep", sub.mask)               # bool [R, C]
        self.register_buffer("mask_f", sub.mask.to(torch.float32))
        self.register_buffer("arangeK",
                             torch.arange(K, device=dev, dtype=torch.int64))
        self.n_kept = int(sub.mask.sum().item())
        self.inv_logK = 1.0 / math.log(K)

        if mode == "pair":
            lev4 = sub.lev0.gather(
                1, (base.unsqueeze(-1) + self.arangeK).reshape(R, -1)
            ).reshape(R, C, K)
            cur = lev4.gather(2, code0.unsqueeze(-1))        # [R, C, 1]
            diff = (lev4 - cur).abs()
            diff.scatter_(2, code0.unsqueeze(-1), float("inf"))   # exclude self
            alt = diff.argmin(dim=-1)                        # [R, C]
            # nearest alternative is always finite for K>=2; restrict the
            # relaxation to KEPT weights (pruned stay 0, never flipped).
            has_alt = torch.isfinite(
                diff.gather(2, alt.unsqueeze(-1)).squeeze(-1)) & sub.mask
            self.register_buffer("alt", alt.to(torch.uint8))
            self.register_buffer("has_alt", has_alt)
            self.n_alt = int(has_alt.sum().item())
            v0 = math.log((h0 - GAMMA_S) / (ZETA - h0))
            self.v = nn.Parameter(torch.full((R, C), v0, device=dev))
            del lev4, cur, diff
        else:
            logits = torch.zeros(R, C, K, dtype=torch.float32, device=dev)
            logits.scatter_(2, code0.unsqueeze(-1), gamma)
            self.logits = nn.Parameter(logits)
        self.delta = nn.Parameter(torch.zeros_like(sub.lev0))
        self.bias = bias                                     # frozen

    # ---- levels (stage-1 semantics) ----
    def master(self):
        return self.lev0 + self.scale * self.delta

    def levels_ste(self):
        m = self.master()
        p = m.to(FP8).to(torch.float32)
        return m + (p - m).detach()

    # ---- assignments ----
    def probs(self):
        return F.softmax(self.logits, dim=-1)               # all K slots legal

    def h_pair(self):
        h = torch.clamp(torch.sigmoid(self.v) * (ZETA - GAMMA_S) + GAMMA_S,
                        0.0, 1.0)
        return h * self.has_alt

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
        return self.logits.argmax(dim=-1)                    # [R, C] int64

    def entropy_norm_mean(self):
        logp = F.log_softmax(self.logits, dim=-1)
        H = -(logp.exp() * logp).sum(-1)                     # [R, C] nats
        H = H * self.keep                                    # kept weights only
        return H.sum() * self.inv_logK / max(self.n_kept, 1)

    def _levK(self):
        lev = self.levels_ste()                              # [R, NG*K]
        idxK = (self.base.unsqueeze(-1) + self.arangeK).reshape(self.R, -1)
        return lev.gather(1, idxK).reshape(self.R, self.C, self.K)

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
            W = W * self.mask_f
            return W.to(torch.bfloat16)
        levK = self._levK()
        if self.hard:
            k = self.committed_codes()
            if self.training and self.mode == "hardste":
                p = self.probs()
                y = F.one_hot(k, self.K).to(p.dtype)
                p_st = y + p - p.detach()                    # ST estimator
                W = (p_st * levK).sum(-1)
            else:
                W = levK.gather(2, k.unsqueeze(-1)).squeeze(-1)
        else:
            W = (self.probs() * levK).sum(-1)
        W = W * self.mask_f                                  # pruned -> 0
        return W.to(torch.bfloat16)

    def forward(self, x):
        return F.linear(x, self.weight_bf16(), self.bias)


def set_hard(qlin, hard: bool):
    for ql in qlin.values():
        ql.hard = hard or ql.mode == "hardste"


class _Shim:                    # minimal `sub` stand-in for sbt.TunedQuantLinear
    pass


# ---------------------------------------------------------------------------
# per-block stage-2 tuning
# ---------------------------------------------------------------------------
def lr_factor(step, total, floor=0.1):
    return floor + (1.0 - floor) * 0.5 * (
        1.0 + math.cos(math.pi * step / max(total, 1)))


def tune_block_assign(bi, block_fp, subs, x_q, target, kw, args, device, log):
    """Returns (final frozen block, {name: cb fp8 [R,NG,K] cpu},
    {name: committed codes uint8 [R,C] cpu}, stats)."""
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
        assert tuple(lin.weight.shape) == (sub.R, sub.C), name
        ql = AssignLinear(sub, lin.bias, args.gamma, args.mode,
                          h0=args.h0).to(device)
        setattr(parent, parts[-1], ql)
        qlin[name] = ql

    # baseline = hard forward at init == exact stage-1 (-btuned) state
    set_hard(qlin, True)
    mse_before = sbt.stream_mse(block_q, x_q, target, kw)
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

    gen = torch.Generator().manual_seed(30_000 + bi)
    N = x_q.shape[0]
    T = args.steps
    t_reg = int(args.reg_start * T)
    t_warm = int(args.warmup_frac * T)
    t_drift = int(args.drift_start * T)
    t_ph1 = int(args.two_phase_frac * T)
    loss_first = loss_last = None
    for step in range(T):
        f = lr_factor(step, T)
        opt.param_groups[0]["lr"] = args.lr * f
        opt.param_groups[1]["lr"] = (
            0.0 if step < t_ph1 else args.lr_lev * f)
        sel = torch.randperm(N, generator=gen)[:args.batch]
        out = sbt._block_forward(block_q, x_q[sel], kw)
        mse = (out.float() - target[sel].float()).pow(2).mean()
        loss = mse
        if args.mode == "soft" and args.reg_frac > 0 and step >= t_reg:
            ramp = ((step - t_reg) / max(T - t_reg, 1)) ** 2
            lam = args.reg_frac * max(mse_before, 1e-12) * ramp
            ent = sum(ql.entropy_norm_mean() for ql in qlin.values()) \
                / len(qlin)
            loss = loss + lam * ent
        elif args.mode == "pair" and args.reg_frac > 0 and step >= t_warm:
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

    mse_soft = None
    if args.mode in ("soft", "pair"):
        set_hard(qlin, False)
        mse_soft = sbt.stream_mse(block_q, x_q, target, kw)
    set_hard(qlin, True)
    mse_hard = sbt.stream_mse(block_q, x_q, target, kw)

    # commit assignments (argmax / rectified-sigmoid) + stats
    n_flip, n_w, n_unsat = 0, 0, 0
    ent_end = 0.0
    codes_new = {}
    with torch.no_grad():
        for name in SUBLAYER_NAMES:
            ql = qlin[name]
            k = ql.committed_codes()
            # only KEPT positions can flip (pruned weight is 0 either way)
            flipped = (k != ql.code0.to(torch.int64)) & ql.keep
            n_flip += int(flipped.sum().item())
            n_w += int(ql.keep.sum().item())
            if args.mode == "pair":
                n_unsat += ql.frac_unsaturated()
            else:
                ent_end += float(ql.entropy_norm_mean().item())
            codes_new[name] = k.to(torch.uint8).cpu()
    ent_end /= len(qlin)
    flip_rate = n_flip / max(n_w, 1)

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
            shim.lev0, shim.scale = lev1, sub.scale
            shim.idx, shim.mask = idx_new, sub.mask
            ql2 = sbt.TunedQuantLinear(shim, ql.bias).to(device)
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
            out = sbt._block_forward(block_q, x_q[sel], kw)
            loss = (out.float() - target[sel].float()).pow(2).mean()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params2, args.grad_clip)
            opt2.step()

    # ---- finalize: fp8 masters (canon signed-zero), install plain Linears ----
    final_cb = {}
    with torch.no_grad():
        for name in SUBLAYER_NAMES:
            sub = subs[name]
            src_master = (ql2map[name].master() if ql2map
                          else qlin[name].master())
            cb_f = sbt.canon_fp8_zero(src_master.to(FP8)).reshape(
                sub.R, sub.NG, sub.K).contiguous()           # fp8 [R,NG,K]
            final_cb[name] = cb_f.cpu()
            idx_new = (qlin[name].base
                       + codes_new[name].to(device).to(torch.int64))
            W = (cb_f.to(torch.bfloat16).reshape(sub.R, sub.NG * sub.K)
                 .gather(1, idx_new))
            W = torch.where(sub.mask, W, torch.zeros_like(W))
            lin = nn.Linear(sub.C, sub.R, bias=qlin[name].bias is not None)
            lin = lin.to(device=device, dtype=torch.bfloat16)
            lin.weight.data = W          # keep exact bf16 bits
            if qlin[name].bias is not None:
                lin.bias = nn.Parameter(qlin[name].bias.detach(),
                                        requires_grad=False)
            parent = block_q
            parts = name.split(".")
            for p_ in parts[:-1]:
                parent = getattr(parent, p_)
            setattr(parent, parts[-1], lin)

    mse_final = sbt.stream_mse(block_q, x_q, target, kw)
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
        "flip_rate": flip_rate, "n_flipped": n_flip, "n_kept": n_w,
        "entropy_end": ent_end, "n_unsaturated": n_unsat,
        "wall_s": round(time.time() - t0, 1),
    }
    print(f"SDATUNE block {bi:2d}: MSE before {mse_before:.6e} "
          f"soft {('%.6e' % mse_soft) if mse_soft is not None else '-'} "
          f"hard {mse_hard:.6e} final {mse_final:.6e} "
          f"(x{mse_final / mse_before:.4f}) flips {n_flip}/{n_w} "
          f"({100 * flip_rate:.2f}%) ent_end {ent_end:.4f} "
          f"unsat {n_unsat} t={stats['wall_s']}s", flush=True)
    log.append(stats)
    return block_q, final_cb, codes_new, stats


# ---------------------------------------------------------------------------
# synthetic no-GPU roundtrip selftest
# ---------------------------------------------------------------------------
def main_selftest_roundtrip():
    import tempfile

    gen = torch.Generator().manual_seed(20260708)
    R, C, K = 8, 384, 4
    bw = [128, 128, 128]
    NG = len(bw)
    cb = torch.randn(R, NG, K, generator=gen).float().sort(dim=-1).values
    cb = cb.to(torch.bfloat16)
    cb[1, 0, 0] = cb[1, 0, 1]                  # duplicate level (row 1 blk 0)
    mask = torch.rand(R, C, generator=gen) < 0.5
    mask[0, :] = False                         # fully-pruned row
    mask[R - 1, :] = True                      # fully-kept row
    codes = torch.randint(0, K, (R, C), generator=gen)
    wq = torch.zeros(R, C, dtype=torch.bfloat16)
    off = 0
    for b, w_b in enumerate(bw):
        cval = torch.gather(cb[:, b, :], 1, codes[:, off:off + w_b])
        wq[:, off:off + w_b] = torch.where(
            mask[:, off:off + w_b], cval, torch.zeros_like(cval))
        off += w_b

    with tempfile.TemporaryDirectory() as base, \
            tempfile.TemporaryDirectory() as btuned, \
            tempfile.TemporaryDirectory() as atuned:
        # 1) a bf16 BASE container
        sdoml_dump.save_layer(base, "selftest.layer", wq, mask, cb, bw,
                              sparsity=0.5)
        # 2) emulate stage-1 (-btuned): fp8 cb, codes UNCHANGED -> fp8 src
        sub0 = SublayerCodebook(base, "selftest.layer", device="cpu")
        cb_bt = sbt.canon_fp8_zero(sub0.lev0.to(FP8)).reshape(
            R, NG, K).contiguous()
        sbt.write_tuned_layer(btuned, sub0, cb_bt)             # gates a/b/c
        print("selftest: emulated fp8 -btuned source written (gates a/b/c OK)",
              flush=True)

        # 3) STAGE 2 on the fp8 source: change BOTH codes and levels.
        sub = SublayerCodebook(btuned, "selftest.layer", device="cpu")
        assert sub.meta["cb_dtype"] == "float8_e4m3fn"
        ql = AssignLinear(sub, None, gamma=4.0, mode="soft")
        with torch.no_grad():                                  # random reassign
            ql.logits.copy_(torch.randn(R, C, K, generator=gen))
        codes_new = ql.committed_codes().to(torch.uint8)       # [R, C]
        master = sub.lev0 + 0.05 * torch.randn(sub.lev0.shape, generator=gen)
        cb_new = sbt.canon_fp8_zero(master.to(FP8)).reshape(
            R, NG, K).contiguous()
        base_flat = sub.idx - (sub.idx % K)
        idx_new = base_flat + codes_new.to(torch.int64)
        n_flip = int(((codes_new.to(torch.int64) != (sub.idx % K)) & mask)
                     .sum().item())
        assert n_flip > 0, "selftest reassigned nothing — not exercising codes"
        sub.idx = idx_new                                      # repoint codes
        sbt.write_tuned_layer(atuned, sub, cb_new)             # gates a/b/c
        print(f"selftest: stage-2 write flipped {n_flip} kept codes; "
              f"gates a/b/c OK", flush=True)

        # 4) independent re-verify of the written dir (all 3 gates again)
        sbt.verify_tuned_dir(btuned, atuned)

        # 5) HONEST-BPW invariant: btuned (fp8) vs atuned (fp8) must be
        #    STRUCTURALLY identical -> same honest bpw, regardless of the
        #    changed code assignment.
        r_bt = sdoml_honest_bpw.measure_layer(
            os.path.join(btuned, "selftest.layer.sdpk.safetensors"))
        r_at = sdoml_honest_bpw.measure_layer(
            os.path.join(atuned, "selftest.layer.sdpk.safetensors"))
        for key in ("R", "C", "K", "NG", "n_weights", "n_kept",
                    "mask_raw_bits", "code_bits", "cb_bits_paddedK",
                    "cb_dtype", "cb_bits_per_level"):
            if r_bt[key] != r_at[key]:
                raise AssertionError(
                    f"honest-bpw component {key} changed {r_bt[key]} -> "
                    f"{r_at[key]} (stage-2 must be bpw-neutral)")
        print("selftest: honest-bpw components byte-for-byte invariant "
              "btuned<->atuned (mask/code/cb bits identical)", flush=True)

        # 6) independent argmin decode on the atuned container == its wq
        with safe_open(os.path.join(atuned,
                       "selftest.layer.sdpk.safetensors"),
                       framework="pt", device="cpu") as f:
            meta_o = json.loads(f.metadata()["meta"])
            wq_o, cb_o = f.get_tensor("wq"), f.get_tensor("cb")
            pk = f.get_tensor("mask_packed")
        assert cb_o.dtype == FP8 and meta_o["cb_dtype"] == "float8_e4m3fn"
        assert torch.equal(pk, sub.src_mask_packed), "mask bytes drifted"
        mask_o = sdoml_dump.unpack_mask(pk, R, C)
        sdoml_honest_bpw.decode_codes(
            wq_o, mask_o, cb_o.to(torch.bfloat16), meta_o["block_widths"])
        print("selftest: atuned container passes an INDEPENDENT argmin-decode "
              "round-trip", flush=True)

        # 7) negative control (gate a): a single mask-bit flip must be rejected.
        saved = sub.src_mask_packed
        bad = saved.clone()
        bad[0] = bad[0].item() ^ 1
        sub.src_mask_packed = bad
        try:
            sbt.write_tuned_layer(atuned, sub, cb_new)
            raise AssertionError("mask-flip did NOT break gate (a)")
        except RuntimeError as e:
            assert "mask_packed" in str(e)
            print("selftest: mask-flip negative control correctly raised "
                  "gate (a)", flush=True)
        finally:
            sub.src_mask_packed = saved

    print("SELFTEST-ROUNDTRIP PASS")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="source dump dir (the stage-1 -btuned dump)")
    ap.add_argument("--out", default=None,
                    help="tuned dump dir (default <src minus -btuned>-atuned)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", choices=("soft", "hardste", "pair"),
                    default="soft")
    ap.add_argument("--h0", type=float, default=0.1,
                    help="pair mode: initial rectified-sigmoid value")
    ap.add_argument("--beta-hi", type=float, default=18.0)
    ap.add_argument("--beta-lo", type=float, default=2.0)
    ap.add_argument("--warmup-frac", type=float, default=0.2)
    ap.add_argument("--drift-max", type=float, default=0.05)
    ap.add_argument("--drift-start", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-2,
                    help="Adam lr on assignment logits (pair: on v)")
    ap.add_argument("--lr-lev", type=float, default=1e-3,
                    help="Adam lr on level deltas (0 = assignments only)")
    ap.add_argument("--two-phase-frac", type=float, default=0.0,
                    help=">0: freeze levels for the first FRAC of steps")
    ap.add_argument("--gamma", type=float, default=4.0,
                    help="init logit advantage of the current assignment")
    ap.add_argument("--reg-frac", type=float, default=0.02)
    ap.add_argument("--reg-start", type=float, default=0.65)
    ap.add_argument("--retune-steps", type=int, default=80,
                    help="post-commit levels-only re-tune steps")
    ap.add_argument("--retune-lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stream-chunk", type=int, default=None,
                    help="sequences per no-grad stream forward (VRAM knob; "
                         "bit-neutral; default = sdoml_block_tune's 8)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="default = all blocks of the resolved model")
    ap.add_argument("--model", default=None,
                    help="HF model name; default = the src dump manifest's "
                         "model. A --model that contradicts the manifest is "
                         "a hard error.")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--selftest-roundtrip", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.selftest_roundtrip:
        main_selftest_roundtrip()
        return

    if not args.src:
        ap.error("--src is required (except for --selftest-roundtrip)")
    src_dir = os.path.abspath(args.src)
    assert os.path.isdir(src_dir), src_dir
    if args.out:
        out_dir = os.path.abspath(args.out)
    elif src_dir.endswith("-btuned"):
        out_dir = src_dir[:-len("-btuned")] + "-atuned"
    else:
        out_dir = src_dir + "-atuned"
    assert out_dir != src_dir

    if args.verify_only:
        sbt.verify_tuned_dir(src_dir, out_dir)
        return

    # 2026-07-20: resolve model from the src manifest + refresh the
    # import-time copies of the layer geometry (4B has 36 blocks; the old
    # 28/196 was a 0.6B/1.7B coincidence)
    global MODEL_NAME, N_BLOCKS, EXPECTED_SUBLAYERS
    sbt.MODEL_NAME = sbt.resolve_model(src_dir, args.model)
    sbt.set_layer_geometry(sbt.MODEL_NAME)
    MODEL_NAME = sbt.MODEL_NAME
    N_BLOCKS = sbt.N_BLOCKS
    EXPECTED_SUBLAYERS = sbt.EXPECTED_SUBLAYERS
    if args.stream_chunk:
        sbt.STREAM_CHUNK = args.stream_chunk

    print(f"SDATUNE: model = {MODEL_NAME} (blocks={N_BLOCKS}, "
          f"sublayers={EXPECTED_SUBLAYERS}, "
          f"stream_chunk={sbt.STREAM_CHUNK})", flush=True)
    print(f"SDATUNE: src={src_dir}\nSDATUNE: out={out_dir}\n"
          f"SDATUNE: mode={args.mode} steps={args.steps} lr={args.lr} "
          f"lr_lev={args.lr_lev} retune={args.retune_steps}", flush=True)

    device = args.device
    t_start = time.time()
    torch.manual_seed(0)

    print("SDATUNE: loading model + calibration (standard run.py path)...",
          flush=True)
    model, dataloader = sbt.load_model_and_calib(device)
    inps, layer_kwargs = sbt.capture_block0_inputs(model, dataloader, device)
    layers = model.model.layers
    assert len(layers) == N_BLOCKS
    print(f"SDATUNE: captured {tuple(inps.shape)} block-0 inputs", flush=True)

    x_fp = inps
    x_q = inps.clone()
    tune_log = []
    all_final_cb = {}
    all_codes = {}
    all_subs = {}

    n_blocks = min(args.max_blocks or N_BLOCKS, N_BLOCKS)
    for bi in range(n_blocks):
        block_fp = layers[bi].to(device)
        subs = {}
        for name in SUBLAYER_NAMES:
            sub = SublayerCodebook(src_dir, f"model.layers.{bi}.{name}",
                                   device=device)
            lin_w = dict(block_fp.named_modules())[name].weight
            assert tuple(lin_w.shape) == (sub.R, sub.C), name
            subs[name] = sub

        target = sbt.forward_stream(block_fp, x_fp, layer_kwargs)
        block_q, final_cb, codes_new, _stats = tune_block_assign(
            bi, block_fp, subs, x_q, target, layer_kwargs, args, device,
            tune_log)
        x_q = sbt.forward_stream(block_q, x_q, layer_kwargs)
        x_fp = target
        for name in SUBLAYER_NAMES:
            lname = f"model.layers.{bi}.{name}"
            sub = subs[name]
            # repoint sub.idx to the committed assignment for the write path
            base = sub.idx - (sub.idx % sub.K)
            idx_new = base + codes_new[name].to(sub.idx.device).to(torch.int64)
            sub.idx = idx_new.cpu()
            sub.lev0 = sub.lev0.cpu()
            sub.scale = sub.scale.cpu()
            sub.mask = sub.mask.cpu()
            all_final_cb[lname] = final_cb[name]
            all_codes[lname] = codes_new[name]
            all_subs[lname] = sub

        del block_q, target, subs
        layers[bi] = layers[bi].cpu()
        torch.cuda.empty_cache()

    final_mse = (x_q.float() - x_fp.float()).pow(2).mean().item()
    ref = x_fp.float().pow(2).mean().item()
    print(f"SDATUNE: final-stream MSE after block {n_blocks - 1}: "
          f"{final_mse:.6e} (rel {final_mse / ref:.4e})", flush=True)
    fr = [b["flip_rate"] for b in tune_log]
    mr = [b["mse_ratio_final"] for b in tune_log]
    print(f"SDATUNE: flip rate min={min(fr):.4f} median="
          f"{sorted(fr)[len(fr) // 2]:.4f} max={max(fr):.4f}; "
          f"MSE ratio min={min(mr):.4f} median="
          f"{sorted(mr)[len(mr) // 2]:.4f} max={max(mr):.4f}", flush=True)

    if args.no_write:
        print(f"SDATUNE: --no-write set; tuned {len(all_final_cb)} sublayers, "
              f"nothing written. wall={time.time() - t_start:.0f}s")
        print(json.dumps(tune_log, indent=1))
        return

    if n_blocks != N_BLOCKS:
        raise SystemExit("SDATUNE FATAL: refusing to write a dump from a "
                         "partial run (use --no-write for smoke tests)")
    assert len(all_final_cb) == EXPECTED_SUBLAYERS

    os.makedirs(out_dir, exist_ok=True)
    for lname in all_final_cb:
        # sub.idx already repointed to the committed assignment above; reuse the
        # stage-1 writer (runs all 3 container gates a/b/c).
        sbt.write_tuned_layer(out_dir, all_subs[lname], all_final_cb[lname])
    print(f"SDATUNE: wrote {len(all_final_cb)} tuned sublayers -> {out_dir}",
          flush=True)

    cmp_summary = sbt.verify_tuned_dir(src_dir, out_dir)

    manifest = {}
    src_manifest = os.path.join(src_dir, "manifest.json")
    if os.path.exists(src_manifest):
        with open(src_manifest) as f:
            manifest = json.load(f)
    manifest["sdoml_assign_tune"] = {
        "src_dir": src_dir, "mode": args.mode, "steps": args.steps,
        "lr": args.lr, "lr_lev": args.lr_lev,
        "two_phase_frac": args.two_phase_frac, "gamma": args.gamma,
        "reg_frac": args.reg_frac, "reg_start": args.reg_start, "h0": args.h0,
        "beta_hi": args.beta_hi, "beta_lo": args.beta_lo,
        "warmup_frac": args.warmup_frac, "drift_max": args.drift_max,
        "drift_start": args.drift_start, "retune_steps": args.retune_steps,
        "retune_lr": args.retune_lr, "batch": args.batch,
        "grad_clip": args.grad_clip,
        "parameterization": (
            "pair: AdaRound rectified-sigmoid (zeta=1.1, gamma_s=-0.1) between "
            "current slot and nearest-by-value alternative slot, "
            "f_reg=(1-|2h-1|^beta) annealed"
            if args.mode == "pair" else
            "per-KEPT-weight K-way softmax over the block's K levels")
        + " + stage-1 delta/fp8-STE levels; hard commit + levels-only re-tune. "
          "honest bpw padded-K => bpw-neutral by construction",
        "calib": "wikitext2 nsamples=128 seed=0 seqlen=2048 (run.py path)",
        "cb_dtype_out": "float8_e4m3fn",
        "container_gates": cmp_summary,
        "final_stream_mse": final_mse,
        "blocks": tune_log,
        "wall_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"SDATUNE: DONE. tuned dump = {out_dir}  "
          f"wall={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
