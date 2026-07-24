"""SDOML block-reconstruction LEVELS tuner (port of k31_block_tune.py).

OUTPUT-AWARE quality recovery at (better-than) ZERO bit cost: tune ONLY the
codebook level VALUES of an existing SDOML BASE dump (sdoml_dump.py), then
store the tuned levels as fp8-e4m3.  Everything else — the per-weight nearest-
centroid code assignments, the keep-mask bitmap, the group structure and all
structural metadata — stays FROZEN, so the honest bpw's code stream
(n_kept * log2 K) and mask bitmap (R*C bits) are byte-identical by construction.
Storing the codebook in fp8-e4m3 instead of bf16 additionally halves the
codebook term (K*16 -> K*8 bits per row-block), i.e. 2.50 -> 2.25 honest bpw;
only PPL and the codebook dtype move.

Algorithm (progressive block-wise reconstruction, identical to K31):
  * Load Qwen/Qwen3-0.6B (eager attn, safetensors) + the standard run.py
    calibration set (get_loaders wikitext2, nsamples=128, seed=0, seqlen=2048).
  * Two activation streams through the 28 decoder blocks:
      x_fp — activations of the pristine FP (bf16) model;
      x_q  — activations through the finished tuned-quantized blocks so far.
  * For block i: target = block_fp(x_fp_i); tune the 7 sublayers' levels to
    minimize MSE(block_q(x_q_i), target); freeze at the fp8-projected levels;
    propagate x_q_{i+1} = block_q(x_q_i).
  * Differentiable reconstruction per SDOML sublayer:
      L    fp32 [R, NG*K]  master init (exact bf16 source levels)
      idx  int64 [R, C]    FROZEN flat slot index = (col_block)*K + code,
                           code = argmin_k (wq - cb)^2 per 128-col block
      m    bool  [R, C]    FROZEN keep-mask (pruned -> weight 0)
      W_hat = m * L.gather(1, idx)     (pruned positions -> 0)
    Only `L` (via `delta`) trains; `idx` and `m` are frozen, so neither the
    code stream nor the mask bitmap can change.
  * fp8 STE (K31): the forward uses master.to(float8_e4m3fn) cast back (so the
    loss sees the REAL fp8-projected levels); the gradient flows straight-
    through to the fp32 master. Every e4m3 value is exact in bf16, so the block
    matmul's bf16 cast adds no rounding.
  * Relative-step parameterization: master = lev0 + scale*delta,
    scale = max(|lev0|, floor) per slot, floor = 0.05 * median nonzero |lev0|
    of the sublayer. Adam + cosine decay, lr default 2e-3, ~300 steps/block.
  * FROZEN: RMSNorms (incl. q_norm/k_norm), embeddings, every non-codebook
    parameter. Only the 7 sublayers' `delta` tensors train.

Outputs (default <src>-btuned/): per sublayer a new .sdpk container with:
  wq          bf16 [R, C]   recomputed from FROZEN (mask, code) + tuned fp8 cb
  mask_packed uint8         BYTE-IDENTICAL to the source (gate a)
  cb          fp8-e4m3 [R, NG, K]  tuned levels
  meta        JSON = source meta + {"cb_dtype": "float8_e4m3fn"}
Container gates (self-checked + printed on the full output):
  (a) mask_packed bytes identical to source;
  (b) reassemble_bitwise(wq_new, mask, cb_new) == wq_new bitwise;
  (c) NG, K, R, C, block_widths identical to source.

Usage:
  synthetic roundtrip selftest (no GPU, no real dump needed):
      python kernels/pack/sdoml_block_tune.py --selftest-roundtrip
  block-0 smoke (writes nothing):
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/sdoml_block_tune.py \
          --src downloads/doml_dumps/qwen3-0.6b/sdoml-s50 \
          --max-blocks 1 --no-write --steps 200
  full run (writes <src>-btuned + container-gate report):
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/sdoml_block_tune.py \
          --src downloads/doml_dumps/qwen3-0.6b/sdoml-s50
  verify-only (re-run all 3 container gates on an existing tuned dir):
      python kernels/pack/sdoml_block_tune.py --src <src> --out <tuned> \
          --verify-only
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

REPO = os.environ.get("CRB_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import sdoml_dump  # noqa: E402
import sdoml_honest_bpw  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MODEL_NAME = DEFAULT_MODEL          # set by resolve_model() in main
EXPECTED_SUBLAYERS = 196            # recomputed by set_layer_geometry()
N_BLOCKS = 28
SUBLAYER_NAMES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


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
    """Same mechanism as k31_block_tune (H17-A): the src dump's manifest
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
FP8 = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Container -> differentiable (levels, index-map) view of one SDOML sublayer
# ---------------------------------------------------------------------------
class SublayerCodebook:
    """Loads <name>.sdpk; exposes:
      lev0   fp32 [R, NG*K]  master init (exact bf16 source levels)
      idx    int64 [R, C]    frozen flat slot index = block*K + code
      mask   bool  [R, C]    frozen keep-mask
    Hard-asserts on load: where(mask, gather(lev0->bf16, idx), 0) == wq BITWISE.
    """

    def __init__(self, dump_dir, layer_name, device="cpu"):
        self.layer_name = layer_name
        path = os.path.join(dump_dir, f"{layer_name}.sdpk.safetensors")
        wq, mask, cb, meta = sdoml_dump.load_layer(path, device="cpu")
        R, C, NG, K = meta["R"], meta["C"], meta["NG"], meta["K"]
        self.meta = meta
        self.block_widths = list(meta["block_widths"])
        self.R, self.C, self.NG, self.K = R, C, NG, K
        # codebook granularity (columns per codebook), DEFAULT 128. The idx map
        # below is built from block_widths, so it is already group-general; this
        # read honors an explicit `groupsize` meta key (g256 etc.) and guards a
        # malformed container.
        self.groupsize = int(meta.get("groupsize", meta.get("blocksize", 128)))
        if self.block_widths and max(self.block_widths) > self.groupsize:
            raise RuntimeError(
                f"{layer_name}: block_widths max {max(self.block_widths)} > "
                f"groupsize {self.groupsize}")
        # keep the exact source mask-bitmap bytes + meta string for gates
        with safe_open(path, framework="pt", device="cpu") as f:
            self.src_mask_packed = f.get_tensor("mask_packed")
            self.src_meta_json = f.metadata()["meta"]
        self.src_cb_bf16 = cb.clone()                       # bf16 [R, NG, K]

        # FROZEN code plane via independent argmin decode (asserts recon==wq
        # bitwise and wq[~mask]==0 exactly).
        code_plane, _recon = sdoml_honest_bpw.decode_codes(
            wq, mask, cb, self.block_widths)                # int64 [R, C]

        # per-column block index from block_widths (robust to a short last block)
        gidx = torch.zeros(C, dtype=torch.int64)
        off = 0
        for b, w_b in enumerate(self.block_widths):
            gidx[off:off + w_b] = b
            off += w_b
        idx = (gidx.unsqueeze(0) * K + code_plane).contiguous()   # [R, C]

        lev0 = cb.to(torch.float32).reshape(R, NG * K).contiguous()

        # roundtrip gate: my (levels, idx, mask) view reproduces wq BITWISE.
        Wb = lev0.to(torch.bfloat16).gather(1, idx)
        Wb = torch.where(mask, Wb, torch.zeros_like(Wb))
        eq = Wb.view(torch.int16) == wq.contiguous().view(torch.int16)
        if not bool(eq.all()):
            bad = int((~eq).sum().item())
            raise RuntimeError(
                f"{layer_name}: (levels, idx, mask) roundtrip NOT bitwise vs "
                f"wq ({bad}/{R * C} mismatches)")

        self.wq = wq                                        # cpu bf16 [R, C]
        self.lev0 = lev0.to(device)
        self.idx = idx.to(device)
        self.mask = mask.to(device)                         # bool [R, C]

        # relative-step scale (see docstring): |lev0| floored so exactly-zero
        # levels remain tunable.
        a = lev0.abs()
        ref = torch.zeros(R, NG * K, dtype=torch.bool)
        ref.scatter_(1, idx, True)
        self.referenced = ref
        nz = a[ref & (a > 0)]
        floor = (0.05 * nz.median().item()) if nz.numel() else 1e-4
        self.scale = a.clamp(min=floor).to(device)
        self.n_ref_slots = int(ref.sum().item())


class TunedQuantLinear(nn.Module):
    """Drop-in nn.Linear: W = mask * fp8_STE(lev0 + scale*delta)[idx].
    Only `delta` trains; forward matmul runs in bf16 like the model."""

    def __init__(self, sub: SublayerCodebook, bias):
        super().__init__()
        self.register_buffer("lev0", sub.lev0)
        self.register_buffer("scale", sub.scale)
        self.register_buffer("idx", sub.idx)
        self.register_buffer("mask_f", sub.mask.float())
        self.delta = nn.Parameter(torch.zeros_like(sub.lev0))
        self.bias = bias           # frozen (None for Qwen3 projections)

    def master(self):
        return self.lev0 + self.scale * self.delta

    def levels_ste(self):
        m = self.master()
        p = m.to(FP8).to(torch.float32)
        return m + (p - m).detach()          # forward: fp8 grid; grad: straight

    def weight_bf16(self):
        L = self.levels_ste()
        W = L.gather(1, self.idx) * self.mask_f          # pruned -> 0
        return W.to(torch.bfloat16)

    def forward(self, x):
        return F.linear(x, self.weight_bf16(), self.bias)


# ---------------------------------------------------------------------------
# Model + calibration (identical to run.py's standard path — verbatim K31)
# ---------------------------------------------------------------------------
def load_model_and_calib(device):
    os.chdir(REPO)
    import run as run_mod                     # argparse is __main__-guarded
    from datautils import get_loaders
    model = run_mod.get_model(MODEL_NAME)
    model.eval()
    assert model.seqlen == 2048, model.seqlen
    dataloader, _ = get_loaders("wikitext2", nsamples=128, seed=0,
                                model=MODEL_NAME, seqlen=model.seqlen)
    assert len(dataloader) == 128
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
def canon_fp8_zero(cb_fp8):
    """Map fp8-e4m3 -0.0 (byte 0x80) -> +0.0 (byte 0x00).

    -0.0 and +0.0 are NUMERICALLY identical (deployed weight and PPL are
    unchanged) but have DISTINCT bf16 bit patterns (0x8000 vs 0x0000). They are
    the ONLY fp8->bf16 value collision with differing bits, so they are the only
    way reassemble_bitwise's argmin can hit a distance-0 tie between two
    different-bit levels and pick the wrong-signed zero. Canonicalizing removes
    the tie => stored wq and cb agree bitwise and reassemble is self-consistent.
    """
    i8 = cb_fp8.view(torch.int8).clone()
    i8[i8 == -128] = 0          # int8(-128) == byte 0x80 == fp8 -0.0
    return i8.view(FP8)


def tune_block(bi, block_fp, subs, x_q, target, kw, args, device, log):
    """Tune block bi's 7 sublayers' levels; returns (block with plain bf16
    Linears, per-sublayer final fp8 cb [R,NG,K], stats dict)."""
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

    # finalize: fp8-project masters, install plain bf16 Linears
    final_cb = {}
    n_slots_changed = 0
    n_slots_total = 0
    with torch.no_grad():
        for name in SUBLAYER_NAMES:
            sub, ql = subs[name], qlin[name]
            # canonicalize signed zero so the stored fp8 cb and the recomputed
            # bf16 wq agree bitwise (removes the only distance-0 argmin tie).
            cb_f = canon_fp8_zero(ql.master().to(FP8)).reshape(
                sub.R, sub.NG, sub.K).contiguous()          # fp8 [R, NG, K]
            final_cb[name] = cb_f.cpu()
            # how many level slots the tuner moved under the fp8 grid
            src_fp8 = sub.src_cb_bf16.to(device).to(FP8).view(torch.int8)
            n_slots_changed += int(
                (cb_f.view(torch.int8) != src_fp8).sum().item())
            n_slots_total += cb_f.numel()
            # deployed weight = the gather of the STORED fp8 container (bf16)
            W = (cb_f.to(torch.bfloat16).reshape(sub.R, sub.NG * sub.K)
                 .gather(1, sub.idx))
            W = torch.where(sub.mask, W, torch.zeros_like(W))
            lin = nn.Linear(sub.C, sub.R, bias=ql.bias is not None)
            lin = lin.to(device=device, dtype=torch.bfloat16)
            lin.weight.data = W        # keep exact bf16 bits (no dtype cast)
            if ql.bias is not None:
                lin.bias = nn.Parameter(ql.bias.detach(), requires_grad=False)
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
    print(f"SDBTUNE block {bi:2d}: MSE {mse0:.6e} -> {mse1:.6e} "
          f"(x{mse1 / mse0:.4f}) params={n_params} ref_slots={n_ref} "
          f"slots_changed={n_slots_changed}/{n_slots_total} "
          f"t={stats['wall_s']}s", flush=True)
    log.append(stats)
    return block_q, final_cb, stats


# ---------------------------------------------------------------------------
# Dump writing + container gates
# ---------------------------------------------------------------------------
def write_tuned_layer(out_dir, sub: SublayerCodebook, cb_new_fp8):
    """Write <name>.sdpk with tuned fp8 cb + recomputed wq. Runs all 3
    container gates (a) mask bytes identical, (b) reassemble bitwise,
    (c) structure identical. Returns the written wq."""
    assert cb_new_fp8.dtype == FP8 and \
        tuple(cb_new_fp8.shape) == (sub.R, sub.NG, sub.K), \
        (cb_new_fp8.dtype, cb_new_fp8.shape)
    # defensive: canonicalize signed zero (idempotent if already done upstream)
    cb_new_fp8 = canon_fp8_zero(cb_new_fp8).contiguous()
    cb_bf16 = cb_new_fp8.to(torch.bfloat16)                 # exact fp8->bf16
    mask_cpu = sub.mask.cpu()
    W = (cb_bf16.reshape(sub.R, sub.NG * sub.K)
         .gather(1, sub.idx.cpu()))
    W = torch.where(mask_cpu, W, torch.zeros_like(W)).contiguous()

    # GATE (b): container self-consistency (argmin re-decode == wq bitwise).
    sdoml_dump.reassemble_bitwise(W, mask_cpu, cb_bf16, sub.block_widths)

    # GATE (a): mask bitmap byte-identical to source.
    mask_packed = sdoml_dump.pack_mask(mask_cpu)
    if not torch.equal(mask_packed, sub.src_mask_packed):
        raise RuntimeError(
            f"{sub.layer_name}: mask_packed NOT byte-identical to source")

    # GATE (c): structure identical (R, C, NG, K, block_widths).
    meta = dict(sub.meta)
    for key, want in (("R", sub.R), ("C", sub.C), ("NG", sub.NG),
                      ("K", sub.K)):
        if int(meta[key]) != int(want):
            raise RuntimeError(f"{sub.layer_name}: meta[{key}] changed")
    if list(meta["block_widths"]) != list(sub.block_widths):
        raise RuntimeError(f"{sub.layer_name}: block_widths changed")
    meta["cb_dtype"] = "float8_e4m3fn"

    tensors = {
        "wq": W,
        "mask_packed": sub.src_mask_packed,   # byte-identical source bytes
        "cb": cb_new_fp8.contiguous(),        # tuned levels, fp8-e4m3
    }
    path = os.path.join(out_dir, f"{sub.layer_name}.sdpk.safetensors")
    save_file(tensors, path, metadata={"meta": json.dumps(meta)})
    return W


def verify_tuned_dir(src_dir, out_dir):
    """Re-open every source<->tuned pair and re-run all 3 container gates on
    the WRITTEN files; count cb slots changed. Raises on any violation."""
    files = sorted(glob.glob(os.path.join(src_dir, "*.sdpk.safetensors")))
    assert files, src_dir
    n_layers = 0
    slots_changed = 0
    slots_total = 0
    for fp in files:
        name = os.path.basename(fp)
        fp2 = os.path.join(out_dir, name)
        if not os.path.exists(fp2):
            raise RuntimeError(f"verify: missing {fp2}")
        with safe_open(fp, framework="pt", device="cpu") as fa, \
                safe_open(fp2, framework="pt", device="cpu") as fb:
            ma = json.loads(fa.metadata()["meta"])
            mb = json.loads(fb.metadata()["meta"])
            # (a) mask bitmap byte-identical
            pa, pb = fa.get_tensor("mask_packed"), fb.get_tensor("mask_packed")
            if pa.dtype != pb.dtype or pa.shape != pb.shape or \
                    not torch.equal(pa, pb):
                raise RuntimeError(
                    f"{name}: mask_packed NOT byte-identical source<->tuned")
            # (c) structure identical
            for key in ("R", "C", "NG", "K", "blocksize"):
                if int(ma[key]) != int(mb[key]):
                    raise RuntimeError(f"{name}: meta[{key}] changed")
            if list(ma["block_widths"]) != list(mb["block_widths"]):
                raise RuntimeError(f"{name}: block_widths changed")
            if mb.get("cb_dtype") != "float8_e4m3fn":
                raise RuntimeError(
                    f"{name}: tuned cb_dtype != float8_e4m3fn "
                    f"(got {mb.get('cb_dtype')})")
            R, C, NG, K = ma["R"], ma["C"], ma["NG"], ma["K"]
            cb = fb.get_tensor("cb")
            if cb.dtype != FP8 or tuple(cb.shape) != (R, NG, K):
                raise RuntimeError(
                    f"{name}: tuned cb is {cb.dtype}{tuple(cb.shape)}")
            wq = fb.get_tensor("wq")
            mask = sdoml_dump.unpack_mask(pb, R, C)
            # (b) reassemble bitwise on the written container
            sdoml_dump.reassemble_bitwise(
                wq, mask, cb.to(torch.bfloat16), mb["block_widths"])
            # cb slots changed vs source (both projected to fp8)
            ca = fa.get_tensor("cb").to(FP8).view(torch.int8)
            cbt = cb.view(torch.int8)
            slots_changed += int((ca != cbt).sum().item())
            slots_total += cbt.numel()
        n_layers += 1
    summary = {"n_layers": n_layers, "cb_slots_changed": slots_changed,
               "cb_slots_total": slots_total}
    print(f"SDBTUNE container-gates: {n_layers} sublayers — "
          f"(a) mask bytes identical, (b) reassemble bitwise, (c) structure "
          f"identical — ALL PASS; cb slots changed "
          f"{slots_changed}/{slots_total} "
          f"({100.0 * slots_changed / max(slots_total, 1):.2f}%)", flush=True)
    return summary


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

    with tempfile.TemporaryDirectory() as src, \
            tempfile.TemporaryDirectory() as out:
        sdoml_dump.save_layer(src, "selftest.layer", wq, mask, cb, bw,
                              sparsity=0.5)
        sub = SublayerCodebook(src, "selftest.layer", device="cpu")
        print(f"selftest: (levels,idx,mask) roundtrip bitwise OK  "
              f"R={sub.R} C={sub.C} NG={sub.NG} K={sub.K} "
              f"ref_slots={sub.n_ref_slots}/{sub.lev0.numel()}", flush=True)

        # simulate a tune: perturb the master levels, fp8-project, write out.
        master = sub.lev0 + 0.03 * torch.randn(sub.lev0.shape, generator=gen)
        cb_new = master.to(FP8).reshape(sub.R, sub.NG, sub.K).contiguous()
        _W_new = write_tuned_layer(out, sub, cb_new)        # runs gates a/b/c

        # independent re-verify of the written directory (all 3 gates again)
        verify_tuned_dir(src, out)

        # confirm the written container has fp8 cb + preserved structure, and
        # that its argmin decode is self-consistent (independent of write_*).
        wpath = os.path.join(out, "selftest.layer.sdpk.safetensors")
        with safe_open(wpath, framework="pt", device="cpu") as f:
            meta_o = json.loads(f.metadata()["meta"])
            wq_o = f.get_tensor("wq")
            cb_o = f.get_tensor("cb")
            pk = f.get_tensor("mask_packed")
        assert cb_o.dtype == FP8 and meta_o["cb_dtype"] == "float8_e4m3fn"
        assert torch.equal(pk, sub.src_mask_packed), "mask bytes drifted"
        mask_o = sdoml_dump.unpack_mask(pk, sub.R, sub.C)
        sdoml_honest_bpw.decode_codes(
            wq_o, mask_o, cb_o.to(torch.bfloat16), meta_o["block_widths"])
        print("selftest: fp8 codebook + recomputed wq pass an INDEPENDENT "
              "argmin-decode round-trip", flush=True)

        # negative control (gate a): a single mask-bit flip must be rejected.
        bad = sub.src_mask_packed.clone()
        bad[0] = bad[0].item() ^ 1
        saved = sub.src_mask_packed
        sub.src_mask_packed = bad
        try:
            write_tuned_layer(out, sub, cb_new)
            raise AssertionError("mask-flip did NOT break gate (a)")
        except RuntimeError as e:
            assert "mask_packed" in str(e)
            print("selftest: mask-flip negative control correctly raised "
                  "gate (a)", flush=True)
        finally:
            sub.src_mask_packed = saved

    # focused signed-zero-tie regression (the write-out bug the gate caught):
    # a block holding both fp8 +0.0 (idx0) and -0.0 (idx1) with a kept weight
    # sitting on the -0.0 level. Pre-canon reassemble must FAIL; post-canon OK.
    Rz, Cz, Kz = 2, 4, 4
    bwz = [4]
    cbz = torch.zeros(Rz, 1, Kz, dtype=torch.float32)
    cbz[0, 0] = torch.tensor([0.0, 0.0, 1.0, 2.0])
    cbz[1, 0] = torch.tensor([0.5, 1.0, 1.5, 2.0])
    cbz = cbz.to(FP8)
    i8 = cbz.view(torch.int8).clone()
    i8[0, 0, 0] = 0                          # +0.0
    i8[0, 0, 1] = -128                       # -0.0
    cbz = i8.view(FP8)
    maskz = torch.tensor([[True, False, False, False],
                          [True, True, True, True]])
    codez = torch.tensor([[1, 0, 0, 0], [0, 1, 2, 3]])   # row0 col0 -> -0.0
    cbz_bf = cbz.to(torch.bfloat16)
    wqz = torch.where(maskz, cbz_bf.reshape(Rz, Kz).gather(1, codez),
                      torch.zeros(Rz, Cz, dtype=torch.bfloat16))
    try:
        sdoml_dump.reassemble_bitwise(wqz, maskz, cbz_bf, bwz)
        raise AssertionError("signed-zero tie did NOT reproduce pre-canon")
    except RuntimeError:
        print("selftest: signed-zero tie reproduced pre-canon (reassemble "
              "correctly raised)", flush=True)
    cbz_c = canon_fp8_zero(cbz)
    cbz_c_bf = cbz_c.to(torch.bfloat16)
    wqz_c = torch.where(maskz, cbz_c_bf.reshape(Rz, Kz).gather(1, codez),
                        torch.zeros(Rz, Cz, dtype=torch.bfloat16))
    sdoml_dump.reassemble_bitwise(wqz_c, maskz, cbz_c_bf, bwz)
    print("selftest: signed-zero canonicalization -> reassemble bitwise OK",
          flush=True)
    print("SELFTEST-ROUNDTRIP PASS")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="source SDOML dump dir (*.sdpk.safetensors)")
    ap.add_argument("--out", default=None,
                    help="tuned dump dir (default <src>-btuned)")
    ap.add_argument("--model", default=None,
                    help="HF model name; default = the src dump manifest's "
                         "model. A --model that contradicts the manifest is "
                         "a hard error.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3,
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
                    help="synthetic no-GPU roundtrip + container-gate selftest")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-run all 3 container gates src<->out and exit")
    args = ap.parse_args()

    if args.selftest_roundtrip:
        main_selftest_roundtrip()
        return

    if not args.src:
        ap.error("--src is required (except for --selftest-roundtrip)")
    src_dir = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out) if args.out else src_dir + "-btuned"
    assert os.path.isdir(src_dir), src_dir
    assert out_dir != src_dir

    global MODEL_NAME, STREAM_CHUNK
    MODEL_NAME = resolve_model(src_dir, args.model)
    set_layer_geometry(MODEL_NAME)
    if args.stream_chunk:
        STREAM_CHUNK = args.stream_chunk
    print(f"SDBTUNE: model = {MODEL_NAME} (blocks={N_BLOCKS}, "
          f"sublayers={EXPECTED_SUBLAYERS}, stream_chunk={STREAM_CHUNK})",
          flush=True)

    if args.verify_only:
        verify_tuned_dir(src_dir, out_dir)
        return

    device = args.device
    t_start = time.time()
    torch.manual_seed(0)

    print("SDBTUNE: loading model + calibration (standard run.py path)...",
          flush=True)
    model, dataloader = load_model_and_calib(device)
    inps, layer_kwargs = capture_block0_inputs(model, dataloader, device)
    layers = model.model.layers
    assert len(layers) == N_BLOCKS
    print(f"SDBTUNE: captured {tuple(inps.shape)} block-0 inputs; "
          f"layer_kwargs keys = {sorted(layer_kwargs.keys())}", flush=True)

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
            lin_w = dict(block_fp.named_modules())[name].weight
            assert tuple(lin_w.shape) == (sub.R, sub.C), name
            subs[name] = sub

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
            # retained-tensor creep OOMed the 4B/1.7B btunes (2026-07-20)
            sub.idx = sub.idx.cpu()
            sub.lev0 = sub.lev0.cpu()
            sub.mask = sub.mask.cpu()
            sub.scale = sub.scale.cpu()
            all_subs[f"model.layers.{bi}.{name}"] = sub
            n_tuned_sublayers += 1

        del block_q, target
        layers[bi] = layers[bi].cpu()
        torch.cuda.empty_cache()

    final_mse = (x_q.float() - x_fp.float()).pow(2).mean().item()
    ref = x_fp.float().pow(2).mean().item()
    print(f"SDBTUNE: final-stream MSE after block {n_blocks - 1}: "
          f"{final_mse:.6e} (rel {final_mse / ref:.4e})", flush=True)

    if args.no_write:
        print(f"SDBTUNE: --no-write set; tuned {n_tuned_sublayers} sublayers, "
              f"nothing written. wall={time.time() - t_start:.0f}s")
        print(json.dumps(tune_log, indent=1))
        return

    if n_blocks != N_BLOCKS:
        raise SystemExit("SDBTUNE FATAL: refusing to write a dump from a "
                         "partial run (use --no-write for smoke tests)")
    assert n_tuned_sublayers == EXPECTED_SUBLAYERS, n_tuned_sublayers

    os.makedirs(out_dir, exist_ok=True)
    for lname, cb_new in all_final_cb.items():
        write_tuned_layer(out_dir, all_subs[lname], cb_new)
    print(f"SDBTUNE: wrote {len(all_final_cb)} tuned sublayers -> {out_dir}",
          flush=True)

    cmp_summary = verify_tuned_dir(src_dir, out_dir)

    # manifest: source manifest + tuning record
    manifest = {}
    src_manifest = os.path.join(src_dir, "manifest.json")
    if os.path.exists(src_manifest):
        with open(src_manifest) as f:
            manifest = json.load(f)
    manifest["sdoml_block_tune"] = {
        "src_dir": src_dir,
        "steps": args.steps, "lr": args.lr, "batch": args.batch,
        "grad_clip": args.grad_clip,
        "optimizer": "Adam + cosine, delta-parameterized "
                     "(master = lev0 + max(|lev0|, floor)*delta), fp8-e4m3 "
                     "STE forward",
        "calib": "wikitext2 nsamples=128 seed=0 seqlen=2048 (run.py path)",
        "cb_dtype_out": "float8_e4m3fn",
        "container_gates": cmp_summary,
        "final_stream_mse": final_mse,
        "blocks": tune_log,
        "wall_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"SDBTUNE: DONE. tuned dump = {out_dir}  "
          f"wall={time.time() - t_start:.0f}s", flush=True)
    mr = [b["mse_ratio"] for b in tune_log]
    print(f"SDBTUNE: per-block MSE ratio (after/before): min={min(mr):.4f} "
          f"median={sorted(mr)[len(mr) // 2]:.4f} max={max(mr):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
