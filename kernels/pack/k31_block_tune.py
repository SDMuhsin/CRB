"""K31 block-reconstruction LEVELS tuner (BRECQ/TesseraQ family, DOML dumps).

OUTPUT-AWARE quality recovery at ZERO bit cost: tune ONLY the fp8-e4m3
codebook level VALUES of an existing DOML dump. Everything else — per-weight
level assignments (b0/b1 code planes), bulk/tail membership plane (m), the
salient column bitmap (s), the group structure and all metadata — stays
byte-identical to the source dump, so the honest bpw is unchanged by
construction; only PPL can move.

Algorithm (progressive block-wise reconstruction):
  * Load Qwen/Qwen3-0.6B (eager attn, safetensors) + the standard run.py
    calibration set (get_loaders wikitext2, nsamples=128, seed=0, seqlen=2048).
  * Two activation streams through the 28 decoder blocks:
      x_fp — activations of the pristine FP (bf16) model;
      x_q  — activations through the finished tuned-quantized blocks so far.
  * For block i: target = block_fp(x_fp_i) (== x_fp_{i+1}); tune the 7
    sublayers' levels to minimize MSE(block_q(x_q_i), target); then freeze the
    block at the fp8-projected levels and propagate x_q_{i+1} = block_q(x_q_i).
  * Differentiable reconstruction per sublayer: a flat fp32 master `levels`
    tensor [R, NG*12] and a per-weight index map [R, C_orig]
    (idx = (col//g * 3 + part) * 4 + code); W_hat = levels.gather(1, idx).
  * fp8 STE: the forward uses master.to(float8_e4m3fn) cast back (so the loss
    sees the REAL fp8-projected levels — e4m3 has only 3 mantissa bits and a
    plain-fp32 tune + final projection can erase sub-resolution gains); the
    gradient flows straight-through to the fp32 master. Every e4m3 value is
    exactly representable in bf16, so the bf16 cast used by the block matmul
    adds no further rounding.
  * Parameterization (LR justification): master = lev0 + scale * delta with
    scale = max(|lev0|, floor) per level (floor = 0.05 * median nonzero
    |lev0| of the sublayer, so exactly-zero levels can still move). Adam is
    scale-free (per-parameter step ≈ lr), so tuning `delta` with lr=LR makes
    the effective step ≈ LR *relative to the level's own magnitude* — the
    same budget for a 0.005 bulk level and a 0.5 salient level. e4m3 spacing
    is 2^-3 ≈ 12.5% relative, so a few hundred steps at lr ~1e-3..3e-3 can
    cross several fp8 quanta while cosine decay stops late-run drift.
  * FROZEN: RMSNorms (incl. q_norm/k_norm), embeddings, and every
    non-codebook parameter. Only the 7 sublayers' `delta` tensors train.
  * Pad-slot re-tie at write-out: derive_dpk pads each (row, group,
    partition)'s unused level slots by repeating the last REAL level; slots
    are re-tied to the TUNED last real level so k29_honest_bpw's
    distinct-slot accounting (and hence the honest bpw) cannot increase.

Outputs (default <src>-btuned/): per sublayer the source .dpk with ONLY the
`cb` tensor replaced (b0/b1/m/s and the meta JSON byte-identical — verified
by an explicit byte-compare pass) + a matching new .wq (bf16, asserted equal
to dpk_unpack.unpack of the new container, bitwise). manifest.json = source
manifest + a `k31_block_tune` record with the tuning config and per-block
MSE trajectories.

Usage:
  roundtrip selftest (block 0's 7 sublayers, no GPU tuning):
      python kernels/pack/k31_block_tune.py --src <dump> --selftest-roundtrip
  block-0 smoke test (no files written):
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/k31_block_tune.py \
          --src downloads/doml_dumps/qwen3-0.6b/k30-rdsplit-lam3e-5-g256 \
          --max-blocks 1 --no-write --steps 200
  full run (writes <src>-btuned + byte-compare report):
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/k31_block_tune.py \
          --src downloads/doml_dumps/qwen3-0.6b/k30-rdsplit-lam3e-5-g256
  byte-compare only (no tuning):
      python kernels/pack/k31_block_tune.py --src <src> --out <tuned> \
          --compare-only
"""

import argparse
import copy
import json
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
from safetensors.torch import save_file  # noqa: E402

import dpk_unpack  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MODEL_NAME = DEFAULT_MODEL          # set by resolve_model() in main (H17-A)
EXPECTED_SUBLAYERS = 196


def manifest_model(dump_dir):
    """Model recorded in <dump_dir>/manifest.json: explicit 'model' field,
    argv[0] (always the model name) as legacy fallback; None if absent."""
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


def resolve_model(src_dir, cli_model):
    """H17-A single mechanism for the tune stages: the src dump's manifest
    names the model; --model may CONFIRM it, but a mismatch is a hard error
    (models can never be mixed silently)."""
    man_model = manifest_model(src_dir)
    if cli_model and man_model and cli_model != man_model:
        raise SystemExit(f"--model {cli_model} != src manifest model "
                         f"{man_model} — refusing to mix models")
    return cli_model or man_model or DEFAULT_MODEL


def set_layer_geometry(model_name):
    """2026-07-20: derive N_BLOCKS/EXPECTED_SUBLAYERS from the model config
    (7 linears per decoder block). 0.6B and 1.7B both have 28 blocks, so the
    old module constants were a coincidence that 4B (36 blocks) breaks."""
    from transformers import AutoConfig
    global N_BLOCKS, EXPECTED_SUBLAYERS
    cache_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")
    N_BLOCKS = AutoConfig.from_pretrained(
        model_name, cache_dir=cache_dir).num_hidden_layers
    EXPECTED_SUBLAYERS = N_BLOCKS * len(SUBLAYER_NAMES)
N_BLOCKS = 28
SUBLAYER_NAMES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
FP8 = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Container -> differentiable (levels, index-map) view of one sublayer
# ---------------------------------------------------------------------------
class SublayerCodebook:
    """Loads <name>.dpk + <name>.wq; exposes:
      lev0   fp32 [R, NG*12]  master init (exact fp8 values)
      idx    int64 [R, C_orig] per-weight flat slot index (gather dim=1)
      n_real int64 [R, NG, 3]  real (non-pad) level slots per partition
      referenced bool [R, NG*12] slots hit by >=1 real weight
    Hard-asserts on load: gather(lev0->bf16, idx) == wq BITWISE.
    """

    def __init__(self, dump_dir, layer_name, device="cpu"):
        self.layer_name = layer_name
        dpk_path = os.path.join(dump_dir, f"{layer_name}.dpk.safetensors")
        wq_path = os.path.join(dump_dir, f"{layer_name}.wq.safetensors")
        tensors, meta = dpk_unpack.load_container(dpk_path, "cpu")
        if meta["mmode"] != "element" or meta["cbdtype"] != "float8_e4m3fn":
            raise SystemExit(
                f"{dpk_path}: tuner supports element-mmode fp8-e4m3 "
                f"containers only (got mmode={meta['mmode']}, "
                f"cbdtype={meta['cbdtype']})")
        self.meta = meta
        with safe_open(dpk_path, framework="pt", device="cpu") as f:
            self.meta_json = f.metadata()["meta"]      # byte-exact meta string
        self.src_tensors = tensors                      # cpu; b0/b1/m/s reused
        R, C, C_orig, g, NG = (meta["R"], meta["C"], meta["C_orig"],
                               meta["g"], meta["NG"])
        self.R, self.C, self.C_orig, self.g, self.NG = R, C, C_orig, g, NG

        b0 = dpk_unpack.expand_plane(tensors["b0"], C)          # [R, C] bool
        b1 = dpk_unpack.expand_plane(tensors["b1"], C)
        code = b0.to(torch.int64) + 2 * b1.to(torch.int64)      # [R, C] 0..3
        part = dpk_unpack.part_matrix(tensors, meta)            # [R, C] 0..2
        gidx = (torch.arange(C, dtype=torch.int64) // g).unsqueeze(0)
        slot = (gidx * 3 + part) * 4 + code                     # [R, C]
        idx = slot[:, :C_orig].contiguous()

        cb_fp8 = tensors["cb"]                                  # [R, NG, 3, 4]
        lev0 = cb_fp8.to(torch.float32).reshape(R, NG * 12).contiguous()

        # real (non-pad) slot count per (row, group, partition): derive_dpk
        # stores sorted DISTINCT levels then repeats the last one into pads,
        # so n_real = 1 + #(adjacent bit-pattern changes) exactly.
        cbb = cb_fp8.to(torch.bfloat16).view(torch.int16)       # [R, NG, 3, 4]
        chg = (cbb[..., 1:] != cbb[..., :-1]).to(torch.int64).sum(-1)
        self.n_real = 1 + chg                                   # [R, NG, 3]

        ref = torch.zeros(R, NG * 12, dtype=torch.bool)
        ref.scatter_(1, idx, True)
        self.referenced = ref

        with safe_open(wq_path, framework="pt", device="cpu") as f:
            wq = f.get_tensor("wq")
        assert wq.dtype == torch.bfloat16 and tuple(wq.shape) == (R, C_orig)
        # roundtrip gate: my (levels, idx) view must reproduce wq BITWISE.
        # fp8->fp32->bf16 is exact for every e4m3 value (3 mantissa bits).
        W = lev0.to(torch.bfloat16).gather(1, idx)
        eq = W.view(torch.int16) == wq.contiguous().view(torch.int16)
        if not bool(eq.all()):
            bad = int((~eq).sum().item())
            raise RuntimeError(
                f"{layer_name}: (levels, idx) roundtrip NOT bitwise vs wq "
                f"({bad}/{R * C_orig} mismatches)")

        self.lev0 = lev0.to(device)
        self.idx = idx.to(device)
        self.wq = wq

        # relative-step scale (see module docstring): |lev0| floored so
        # exactly-zero levels remain tunable.
        a = lev0.abs()
        nz = a[self.referenced & (a > 0)]
        floor = (0.05 * nz.median().item()) if nz.numel() else 1e-4
        self.scale = a.clamp(min=floor).to(device)
        self.n_ref_slots = int(ref.sum().item())


class TunedQuantLinear(nn.Module):
    """Drop-in nn.Linear replacement: W = fp8_STE(lev0 + scale*delta)[idx].
    Only `delta` is trainable; forward matmul runs in bf16 like the model."""

    def __init__(self, sub: SublayerCodebook, bias):
        super().__init__()
        self.register_buffer("lev0", sub.lev0)
        self.register_buffer("scale", sub.scale)
        self.register_buffer("idx", sub.idx)
        self.delta = nn.Parameter(torch.zeros_like(sub.lev0))
        self.bias = bias           # frozen (None for Qwen3 projections)

    def master(self):
        return self.lev0 + self.scale * self.delta

    def levels_ste(self):
        m = self.master()
        p = m.to(FP8).to(torch.float32)
        return m + (p - m).detach()          # forward: fp8 grid; grad: straight

    def weight_bf16(self):
        return self.levels_ste().gather(1, self.idx).to(torch.bfloat16)

    def forward(self, x):
        return F.linear(x, self.weight_bf16(), self.bias)


# ---------------------------------------------------------------------------
# Model + calibration (identical to run.py's standard path)
# ---------------------------------------------------------------------------
def load_model_and_calib(device, nsamples=128):
    os.chdir(REPO)
    import run as run_mod                     # argparse is __main__-guarded
    from datautils import get_loaders
    model = run_mod.get_model(MODEL_NAME)
    model.eval()
    assert model.seqlen == 2048, model.seqlen
    dataloader, _ = get_loaders("wikitext2", nsamples=nsamples, seed=0,
                                model=MODEL_NAME, seqlen=model.seqlen)
    assert len(dataloader) == nsamples
    return model, dataloader


@torch.no_grad()
def capture_block0_inputs(model, dataloader, device):
    """run.py Catcher pattern: first-block hidden states + layer kwargs."""
    model.config.use_cache = False
    model = model.to(device)
    layers = model.model.layers
    nsamples, seqlen, hidden = (len(dataloader), model.seqlen,
                                model.config.hidden_size)
    inps = torch.zeros(nsamples, seqlen, hidden,
                       dtype=next(model.parameters()).dtype, device=device)
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    assert cache["i"] == nsamples, cache["i"]
    kw = dict(cache["layer_kwargs"])
    kw.pop("past_key_values", None)
    kw["use_cache"] = False

    def _detach(v):
        if isinstance(v, torch.Tensor):
            return v.detach()
        if isinstance(v, tuple):
            return tuple(_detach(x) for x in v)
        if isinstance(v, list):
            return [_detach(x) for x in v]
        return v

    kw = {k: _detach(v) for k, v in kw.items()}
    return inps, kw


def _batched_kwargs(kw, bsz):
    """Expand batch-1 attention masks for a batched block forward
    (transformers-5.x quirk cribbed from src/run_tesseraq.py)."""
    out = dict(kw)
    am = out.get("attention_mask", None)
    if isinstance(am, torch.Tensor) and am.dim() >= 1 and am.shape[0] == 1 \
            and bsz > 1:
        out["attention_mask"] = am.expand(bsz, *am.shape[1:])
    return out


def _block_forward(block, x, kw):
    out = block(x, **_batched_kwargs(kw, x.shape[0]))
    # transformers 5.x returns a bare Tensor, NOT a tuple — never index [0].
    return out[0] if isinstance(out, tuple) else out


STREAM_CHUNK = 8  # sequences per no-grad stream forward; --stream-chunk
                  # (4B eager-attn fp32 softmax is 512 MB/seq — chunk 8 = 4 GiB
                  # spikes that OOM a co-tenanted A40; 2026-07-20)


@torch.no_grad()
def forward_stream(block, x, kw, chunk=None):
    """[N, S, H] -> [N, S, H] through one block, batch-chunked."""
    chunk = chunk or STREAM_CHUNK
    out = torch.empty_like(x)
    for st in range(0, x.shape[0], chunk):
        ed = min(st + chunk, x.shape[0])
        out[st:ed] = _block_forward(block, x[st:ed], kw)
    return out


@torch.no_grad()
def stream_mse(block, x, target, kw, chunk=None):
    chunk = chunk or STREAM_CHUNK
    tot, n = 0.0, 0
    for st in range(0, x.shape[0], chunk):
        ed = min(st + chunk, x.shape[0])
        o = _block_forward(block, x[st:ed], kw).float()
        tot += (o - target[st:ed].float()).pow(2).sum().item()
        n += o.numel()
    return tot / n


# ---------------------------------------------------------------------------
# Per-block tuning
# ---------------------------------------------------------------------------
def retie_pads_fp8(cb_fp8_flat, n_real, R, NG):
    """[R, NG*12] fp8 -> [R, NG, 3, 4] fp8 with pad slots (slot k >= n_real)
    re-tied to the last REAL slot's tuned value (derive_dpk pad convention,
    keeps k29_honest_bpw's distinct-slot count from increasing)."""
    f = cb_fp8_flat.view(torch.int8).reshape(R, NG, 3, 4).clone()
    for k in range(1, 4):
        pad = n_real <= k                       # [R, NG, 3]
        f[..., k] = torch.where(pad, f[..., k - 1], f[..., k])
    return f.view(FP8).contiguous()


def tune_block(bi, block_fp, subs, x_q, target, kw, args, device, log):
    """Tune block bi's 7 sublayers' levels; returns (final block with plain
    bf16 Linears, per-sublayer final fp8 cb [R,NG,3,4], stats dict)."""
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
        ql = TunedQuantLinear(sub, lin.bias).to(device)
        setattr(parent, parts[-1], ql)
        qlin[name] = ql

    params = [ql.delta for ql in qlin.values()]
    n_params = sum(p.numel() for p in params)
    n_ref = sum(subs[n].n_ref_slots for n in SUBLAYER_NAMES)

    mse0 = stream_mse(block_q, x_q, target, kw)   # delta=0 => source weights

    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    gen = torch.Generator().manual_seed(10_000 + bi)
    N = x_q.shape[0]
    loss_first = loss_last = None
    for step in range(args.steps):
        sel = torch.randperm(N, generator=gen)[:args.batch]
        out = _block_forward(block_q, x_q[sel], kw)
        loss = (out.float() - target[sel].float()).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        sched.step()
        lv = loss.item()
        if step == 0:
            loss_first = lv
        loss_last = lv
        if not (lv == lv):                         # NaN guard
            raise RuntimeError(f"block {bi}: NaN loss at step {step}")
        if args.log_every and (step % args.log_every == 0
                               or step == args.steps - 1):
            print(f"  b{bi:02d} step {step:4d} loss {lv:.6e} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

    # finalize: fp8-project masters, re-tie pads, install plain bf16 Linears
    final_cb = {}
    n_slots_changed = 0
    n_slots_total = 0
    with torch.no_grad():
        for name in SUBLAYER_NAMES:
            sub, ql = subs[name], qlin[name]
            cb_f = retie_pads_fp8(ql.master().to(FP8), sub.n_real.to(device),
                                  sub.R, sub.NG)
            final_cb[name] = cb_f.cpu()
            src_bits = sub.src_tensors["cb"].view(torch.int8).to(device)
            n_slots_changed += int(
                (cb_f.view(torch.int8) != src_bits).sum().item())
            n_slots_total += src_bits.numel()
            W = cb_f.to(torch.bfloat16).reshape(sub.R, sub.NG * 12).gather(
                1, sub.idx)
            # pads are never referenced, so the re-tie cannot change W; the
            # deployed weight IS the gather of the stored fp8 container.
            lin = nn.Linear(sub.C_orig, sub.R, bias=ql.bias is not None)
            lin.weight = nn.Parameter(W, requires_grad=False)
            if ql.bias is not None:
                lin.bias = nn.Parameter(ql.bias.detach(),
                                        requires_grad=False)
            lin = lin.to(device=device, dtype=torch.bfloat16)
            lin.weight.data = W        # keep exact bf16 bits (no dtype cast)
            parent = block_q
            parts = name.split(".")
            for p_ in parts[:-1]:
                parent = getattr(parent, p_)
            setattr(parent, parts[-1], lin)

    mse1 = stream_mse(block_q, x_q, target, kw)
    stats = {
        "block": bi, "steps": args.steps, "lr": args.lr, "batch": args.batch,
        "n_delta_params": n_params, "n_referenced_slots": n_ref,
        "mse_before": mse0, "mse_after": mse1,
        "mse_ratio": (mse1 / mse0) if mse0 > 0 else None,
        "loss_first": loss_first, "loss_last": loss_last,
        "fp8_slots_changed": n_slots_changed,
        "fp8_slots_total": n_slots_total,
        "wall_s": round(time.time() - t0, 1),
    }
    print(f"K31 block {bi:2d}: MSE {mse0:.6e} -> {mse1:.6e} "
          f"(x{mse1 / mse0:.4f}) params={n_params} ref_slots={n_ref} "
          f"slots_changed={n_slots_changed}/{n_slots_total} "
          f"t={stats['wall_s']}s", flush=True)
    log.append(stats)
    return block_q, final_cb, stats


# ---------------------------------------------------------------------------
# Dump writing + byte-compare
# ---------------------------------------------------------------------------
def write_tuned_layer(out_dir, sub: SublayerCodebook, cb_new_fp8):
    """Write <name>.dpk with ONLY `cb` replaced (b0/b1/m/s + meta byte-
    identical to source) and the matching new wq. Asserts the container
    round-trips to the written wq bitwise via the reference unpacker."""
    assert cb_new_fp8.dtype == FP8 and \
        tuple(cb_new_fp8.shape) == (sub.R, sub.NG, 3, 4)
    tensors = {
        "b0": sub.src_tensors["b0"],
        "b1": sub.src_tensors["b1"],
        "m": sub.src_tensors["m"],
        "s": sub.src_tensors["s"],
        "cb": cb_new_fp8.contiguous(),
    }
    W_new = dpk_unpack.unpack(tensors, sub.meta)[:, :sub.C_orig].contiguous()
    W_dir = cb_new_fp8.to(torch.bfloat16).reshape(
        sub.R, sub.NG * 12).gather(1, sub.idx.cpu())
    if not torch.equal(W_new.view(torch.int16), W_dir.view(torch.int16)):
        raise RuntimeError(f"{sub.layer_name}: tuned container unpack != "
                           f"direct gather (index map broken)")
    dpk_path = os.path.join(out_dir, f"{sub.layer_name}.dpk.safetensors")
    wq_path = os.path.join(out_dir, f"{sub.layer_name}.wq.safetensors")
    save_file(tensors, dpk_path, metadata={"meta": sub.meta_json})
    save_file({"wq": W_new}, wq_path, metadata={"meta": sub.meta_json})
    return W_new


def byte_compare(src_dir, out_dir):
    """Prove codes/masks/bitmap tensors + meta identical source<->tuned; count
    changed cb slots. Returns summary dict; raises on any violation."""
    import glob
    files = sorted(glob.glob(os.path.join(src_dir, "*.dpk.safetensors")))
    assert files, src_dir
    n_layers = 0
    slots_changed = 0
    slots_total = 0
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
            for k in ("b0", "b1", "m", "s"):
                ta, tb = fa.get_tensor(k), fb.get_tensor(k)
                if ta.dtype != tb.dtype or ta.shape != tb.shape or \
                        not torch.equal(ta.view(torch.int32),
                                        tb.view(torch.int32)):
                    raise RuntimeError(
                        f"{name}: tensor '{k}' NOT byte-identical — the "
                        f"tuner must never touch codes/masks/bitmaps")
            ca = fa.get_tensor("cb").view(torch.int8)
            cbt = fb.get_tensor("cb").view(torch.int8)
            if ca.shape != cbt.shape:
                raise RuntimeError(f"{name}: cb shape changed")
            slots_changed += int((ca != cbt).sum().item())
            slots_total += ca.numel()
        n_layers += 1
    summary = {"n_layers": n_layers, "cb_slots_changed": slots_changed,
               "cb_slots_total": slots_total}
    print(f"K31 byte-compare: {n_layers} sublayers — b0/b1/m/s/meta all "
          f"byte-identical; cb slots changed "
          f"{slots_changed}/{slots_total} "
          f"({100.0 * slots_changed / slots_total:.2f}%)", flush=True)
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source dump dir")
    ap.add_argument("--out", default=None,
                    help="tuned dump dir (default <src>-btuned)")
    ap.add_argument("--model", default=None,
                    help="HF model name; default = the src manifest's "
                         "recorded model (falling back to Qwen/Qwen3-0.6B). "
                         "A --model that contradicts the manifest is a hard "
                         "error.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Adam lr on delta = RELATIVE per-step level movement")
    ap.add_argument("--batch", type=int, default=8,
                    help="calibration sequences per step")
    ap.add_argument("--stream-chunk", type=int, default=None,
                    help="sequences per no-grad stream forward (VRAM knob; "
                         "bit-neutral; default 8)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="tune only the first N blocks (smoke tests); "
                         "default = all blocks of the resolved model")
    ap.add_argument("--no-write", action="store_true",
                    help="tune + report MSE only; write nothing")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--selftest-roundtrip", action="store_true",
                    help="load block 0's 7 sublayers, assert the (levels,idx) "
                         "view reproduces wq bitwise, exit")
    ap.add_argument("--compare-only", action="store_true",
                    help="run the byte-compare pass src<->out and exit")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out) if args.out else src_dir + "-btuned"
    assert os.path.isdir(src_dir), src_dir
    assert out_dir != src_dir

    global MODEL_NAME, STREAM_CHUNK
    MODEL_NAME = resolve_model(src_dir, args.model)
    set_layer_geometry(MODEL_NAME)
    if args.stream_chunk:
        STREAM_CHUNK = args.stream_chunk
    print(f"K31: model = {MODEL_NAME} (blocks={N_BLOCKS}, "
          f"sublayers={EXPECTED_SUBLAYERS}, stream_chunk={STREAM_CHUNK})",
          flush=True)

    if args.compare_only:
        byte_compare(src_dir, out_dir)
        return

    if args.selftest_roundtrip:
        for name in SUBLAYER_NAMES:
            sub = SublayerCodebook(src_dir, f"model.layers.0.{name}")
            print(f"roundtrip OK  {sub.layer_name}  R={sub.R} C={sub.C_orig} "
                  f"NG={sub.NG} ref_slots={sub.n_ref_slots}/{sub.lev0.numel()}"
                  f"  n_real hist="
                  f"{torch.bincount(sub.n_real.flatten(), minlength=5).tolist()}",
                  flush=True)
        print("SELFTEST-ROUNDTRIP PASS (7/7 sublayers bitwise)")
        return

    device = args.device
    t_start = time.time()
    torch.manual_seed(0)

    print(f"K31: loading model + calibration (standard run.py path)...",
          flush=True)
    model, dataloader = load_model_and_calib(device)
    inps, layer_kwargs = capture_block0_inputs(model, dataloader, device)
    layers = model.model.layers
    assert len(layers) == N_BLOCKS
    print(f"K31: captured {inps.shape} block-0 inputs; layer_kwargs keys = "
          f"{sorted(layer_kwargs.keys())}", flush=True)

    x_fp = inps                       # FP stream (model is pristine FP/bf16)
    x_q = inps.clone()                # quantized stream
    tune_log = []
    n_tuned_sublayers = 0
    all_final_cb = {}                 # layer_name -> fp8 cb
    all_subs = {}

    n_blocks = min(args.max_blocks or N_BLOCKS, N_BLOCKS)
    for bi in range(n_blocks):
        block_fp = layers[bi].to(device)
        subs = {}
        for name in SUBLAYER_NAMES:
            sub = SublayerCodebook(src_dir, f"model.layers.{bi}.{name}",
                                   device=device)
            # container weight must match the FP block's shape
            lin_w = dict(block_fp.named_modules())[name].weight
            assert tuple(lin_w.shape) == (sub.R, sub.C_orig), name
            subs[name] = sub

        # FP target = next FP stream state (pristine block, no_grad)
        target = forward_stream(block_fp, x_fp, layer_kwargs)

        block_q, final_cb, _stats = tune_block(
            bi, block_fp, subs, x_q, target, layer_kwargs, args, device,
            tune_log)

        x_q = forward_stream(block_q, x_q, layer_kwargs)
        x_fp = target
        for name in SUBLAYER_NAMES:
            all_final_cb[f"model.layers.{bi}.{name}"] = final_cb[name]
            sub = subs[name]
            # park GPU tensors on CPU (write path only needs cpu tensors) —
            # same fix k31_assign_tune already carries; retained-idx creep is
            # ~1 GB/block at 4B and OOMed btune at block 18/36 (2026-07-20)
            sub.idx = sub.idx.cpu()
            sub.lev0 = sub.lev0.cpu()
            sub.scale = sub.scale.cpu()
            all_subs[f"model.layers.{bi}.{name}"] = sub
            n_tuned_sublayers += 1

        # free
        del block_q, target
        layers[bi] = layers[bi].cpu()
        torch.cuda.empty_cache()

    # end-of-stream quality signal (proxy only; the real gate is restore-eval)
    final_mse = (x_q.float() - x_fp.float()).pow(2).mean().item()
    ref = x_fp.float().pow(2).mean().item()
    print(f"K31: final-stream MSE after block {n_blocks - 1}: {final_mse:.6e}"
          f" (rel {final_mse / ref:.4e})", flush=True)

    if args.no_write:
        print(f"K31: --no-write set; tuned {n_tuned_sublayers} sublayers, "
              f"nothing written. wall={time.time() - t_start:.0f}s")
        print(json.dumps(tune_log, indent=1))
        return

    if n_blocks != N_BLOCKS:
        raise SystemExit("K31 FATAL: refusing to write a dump from a partial "
                         "run (use --no-write for smoke tests)")
    assert n_tuned_sublayers == EXPECTED_SUBLAYERS, n_tuned_sublayers

    os.makedirs(out_dir, exist_ok=True)
    for lname, cb_new in all_final_cb.items():
        write_tuned_layer(out_dir, all_subs[lname], cb_new)
    print(f"K31: wrote {len(all_final_cb)} tuned sublayers -> {out_dir}",
          flush=True)

    cmp_summary = byte_compare(src_dir, out_dir)

    # manifest: source manifest + tuning record
    with open(os.path.join(src_dir, "manifest.json")) as f:
        manifest = json.load(f)
    manifest["k31_block_tune"] = {
        "src_dir": src_dir,
        "model": MODEL_NAME,
        "steps": args.steps, "lr": args.lr, "batch": args.batch,
        "grad_clip": args.grad_clip,
        "optimizer": "Adam + cosine, delta-parameterized "
                     "(master = lev0 + max(|lev0|, floor)*delta), fp8-e4m3 "
                     "STE forward",
        "calib": "wikitext2 nsamples=128 seed=0 seqlen=2048 (run.py path)",
        "byte_compare": cmp_summary,
        "final_stream_mse": final_mse,
        "blocks": tune_log,
        "wall_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"K31: DONE. tuned dump = {out_dir}  "
          f"wall={time.time() - t_start:.0f}s", flush=True)
    mr = [b["mse_ratio"] for b in tune_log]
    print(f"K31: per-block MSE ratio (after/before): min={min(mr):.4f} "
          f"median={sorted(mr)[len(mr) // 2]:.4f} max={max(mr):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
