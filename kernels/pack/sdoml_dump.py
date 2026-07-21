"""SDOML honest-bpw deliverable 1 — passthrough-dump harness + SDPK packer.

Produces one `<layer_name>.sdpk.safetensors` container per quantized sublayer
of a real SDOML base run (`run.py … sdoml`, partition=1, is_sdoml branch in
bigptq.fasterquant lines ~334-395).

The SDOML BASE container is much simpler than the DOML DPK container (no
structural partition, no membership plane, no separate scale tensor). Per
linear layer, quantization proceeds in 128-column blocks; for each
(row, 128-col block) the container is:
  - a per-weight KEEP MASK (pruned positions store weight 0 exactly),
  - one K=4 codebook (K = 2**codebook_bits) of bf16 levels for that row-block,
  - implicit CODES: for each KEPT position a nearest-centroid index in 0..K-1
    (log2 K bits); pruned positions store nothing (mask=0 => weight 0).
The final stored `layer.weight.data` is bf16, with stored weight[kept] =
bf16(codebook level) and stored weight[pruned] = 0.  Codes are DERIVABLE from
(wq, cb) by argmin, so the container stores only (wq, mask bitmap, cb) and the
honest-bpw tool re-derives + counts the code stream.

Container file `<layer_name>.sdpk.safetensors`:
  tensors:
    wq          : bf16  [R, C]              — the final quantized layer weight
    mask_packed : uint8 [ceil(R*C/8)]       — np.packbits(mask.flatten(), big)
    cb          : bf16  [R, NG, K]          — per (row, block) codebook levels
  metadata JSON "meta":
    {R, C, NG, K, blocksize, sparsity, n_kept, layer_name, model_name}

Reassembly gate (BITWISE, per layer, hard-fail): for every 128-col block, cast
that block's codebook to bf16, compute code = argmin_k (wq_block - cb_bf16)^2
over K on ALL positions, recon = where(mask, cb_bf16.gather(code), 0), and
assert recon == wq_block bit-for-bit (int16 view) for EVERY element, AND
wq_block[~mask] == 0 exactly.  Any failure RAISES (never swallowed).

G0 no-regression: the two monkeypatches ONLY do extra work on the sdoml path.
  * `binary.sdoml_quantize` is wrapped, but a NON-sdoml run NEVER calls it, so
    it is never invoked; even if invoked with capture=False the wrapper only
    calls the original and returns its result unchanged (no RNG, no mutation).
  * `bigptq.BRAGPTQ.fasterquant` wrapper sets capture only when
    `braq_quantizer.method == 'sdoml'`; otherwise it calls the original and
    returns unchanged (the blocks list it clears is empty and unused).
  bigptq re-imports `sdoml_quantize` from `binary` LOCALLY inside the branch
  at call time, so patching `binary.sdoml_quantize` needs no import ordering.

Usage:
  selftest (synthetic; duplicate levels + fully-pruned + fully-kept rows):
      python kernels/pack/sdoml_dump.py --selftest
  real dump run (evaluates wikitext2 PPL as a byproduct):
      export CUDA_VISIBLE_DEVICES=1
      python kernels/pack/sdoml_dump.py --run --dump-dir <dir>
"""

import argparse
import json
import math
import os
import sys
import time

REPO = "/workspace/BiLLM2"
DEFAULT_DUMP_DIR = os.path.join(
    REPO, "downloads", "doml_dumps", "qwen3-0.6b", "sdoml-s50")
VERIFY_DIR = os.path.join(REPO, "llmdocs", "cuda_kernel", "verify")

# Must be set before run.py / csv_utils are imported.
os.environ.setdefault(
    "BILLM_BENCH_CSV", os.path.join(VERIFY_DIR, "scratch_results.csv"))

if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from safetensors import safe_open  # noqa: E402

B_BLOCK = 128
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MODEL_NAME = DEFAULT_MODEL          # retargeted by set_model() (2026-07-20)
EXPECTED_SUBLAYERS = 196  # 7 linears x n_blocks; recomputed by set_model()


def _run_argv(model_name, sparsity):
    return [
        "run.py", model_name, "wikitext2", "sdoml",
        "--blocksize", "128", "--salient_metric", "magnitude",
        "--device", "cuda:0", "--sparsity", str(sparsity),
        "--sdoml_n_iter", "20",
    ]


# run.py argv for the sdoml dump run (default = the original 0.6B s=0.5 run).
RUN_ARGV = _run_argv(MODEL_NAME, 0.5)


def set_model(model_name, sparsity):
    """2026-07-20 (mirrors doml_group_refit's H17-A set_model): retarget the
    dump run at `model_name`/`sparsity` and derive EXPECTED_SUBLAYERS from
    the model config (7 linears per decoder block — the old 196 constant
    assumed the 0.6B/1.7B 28-block coincidence, which 4B's 36 blocks breaks).
    Defaults keep the original 0.6B s=0.5 behavior identical."""
    global MODEL_NAME, RUN_ARGV, EXPECTED_SUBLAYERS
    from transformers import AutoConfig
    cache_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")
    MODEL_NAME = model_name
    RUN_ARGV = _run_argv(model_name, sparsity)
    EXPECTED_SUBLAYERS = 7 * AutoConfig.from_pretrained(
        model_name, cache_dir=cache_dir).num_hidden_layers


# ---------------------------------------------------------------------------
# Core packer / reassembly gate for one layer
# ---------------------------------------------------------------------------
@torch.no_grad()
def reassemble_bitwise(wq: torch.Tensor, mask: torch.Tensor,
                       cb: torch.Tensor, block_widths):
    """Rebuild wq from (mask, cb) block-by-block via nearest-centroid argmin.

    wq   : bf16 [R, C]   final quantized weight
    mask : bool [R, C]   keep mask (pruned positions are weight 0)
    cb   : bf16 [R, NG, K] per (row, block) codebook
    block_widths : list of per-block column counts (sum == C)

    Returns the reconstructed bf16 [R, C].  Hard-asserts bit-exact equality
    with wq (int16 view) and that pruned positions are exactly 0.
    """
    assert wq.dtype == torch.bfloat16 and wq.dim() == 2
    R, C = wq.shape
    assert mask.shape == (R, C) and mask.dtype == torch.bool
    NG, K = cb.shape[1], cb.shape[2]
    assert cb.shape[0] == R and len(block_widths) == NG
    assert sum(block_widths) == C, (sum(block_widths), C)

    recon = torch.zeros(R, C, dtype=torch.bfloat16, device=wq.device)
    off = 0
    for b, w_b in enumerate(block_widths):
        wq_blk = wq[:, off:off + w_b]                       # bf16 [R, w_b]
        m_blk = mask[:, off:off + w_b]                      # bool [R, w_b]
        cb_bf16 = cb[:, b, :].to(torch.bfloat16)            # bf16 [R, K]
        # argmin over K in float32 (bf16->f32 is exact; kept value == a cb
        # level exactly => distance 0 at that index).
        diff = (wq_blk.float().unsqueeze(-1)
                - cb_bf16.float().unsqueeze(1))             # [R, w_b, K]
        code = (diff * diff).argmin(dim=-1)                 # [R, w_b]
        rec_kept = torch.gather(cb_bf16, 1, code)           # bf16 [R, w_b]
        rec_blk = torch.where(m_blk, rec_kept,
                              torch.zeros_like(rec_kept))
        # pruned positions must be exactly 0 in the stored weight
        pruned_vals = wq_blk[~m_blk]
        if pruned_vals.numel() and not bool(
                (pruned_vals.view(torch.int16) == 0).all()):
            bad = int((pruned_vals.view(torch.int16) != 0).sum().item())
            raise RuntimeError(
                f"block {b}: {bad} pruned positions are NOT exactly 0")
        eq = rec_blk.view(torch.int16) == wq_blk.contiguous().view(torch.int16)
        if not bool(eq.all()):
            nbad = int((~eq).sum().item())
            raise RuntimeError(
                f"block {b}: reassembly NOT bitwise equal to wq "
                f"({nbad}/{R * w_b} mismatches)")
        recon[:, off:off + w_b] = rec_blk
        off += w_b
    return recon


def pack_mask(mask: torch.Tensor):
    """bool [R, C] -> uint8 torch tensor via np.packbits (big-endian bitorder)
    over the flat [R*C] mask."""
    a = mask.detach().cpu().contiguous().numpy().reshape(-1)
    packed = np.packbits(a)                                 # uint8 [ceil(RC/8)]
    return torch.from_numpy(packed.copy())


def unpack_mask(packed: torch.Tensor, R: int, C: int):
    """uint8 torch tensor -> bool [R, C] (inverse of pack_mask)."""
    a = packed.detach().cpu().contiguous().numpy()
    bits = np.unpackbits(a, count=R * C)
    return torch.from_numpy(bits.astype(bool).reshape(R, C))


@torch.no_grad()
def save_layer(dump_dir, layer_name, wq, mask, cb, block_widths,
               sparsity, model_name=MODEL_NAME):
    """Run the bitwise reassembly gate then write the SDPK container."""
    R, C = wq.shape
    NG, K = cb.shape[1], cb.shape[2]
    # --- BITWISE reassembly gate (raises on any violation) ---------------
    reassemble_bitwise(wq, mask, cb, block_widths)
    n_kept = int(mask.sum().item())
    meta = {
        "R": int(R), "C": int(C), "NG": int(NG), "K": int(K),
        "blocksize": B_BLOCK,
        "sparsity": None if sparsity is None else float(sparsity),
        "n_kept": n_kept,
        "block_widths": [int(x) for x in block_widths],
        "layer_name": layer_name, "model_name": model_name,
    }
    tensors = {
        "wq": wq.detach().cpu().contiguous(),
        "mask_packed": pack_mask(mask),
        "cb": cb.detach().cpu().to(torch.bfloat16).contiguous(),
    }
    path = os.path.join(dump_dir, f"{layer_name}.sdpk.safetensors")
    save_file(tensors, path, metadata={"meta": json.dumps(meta)})
    return path, meta


def load_layer(path, device="cpu"):
    """Load an SDPK container. Returns (wq, mask, cb, meta)."""
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
    if cb.dtype != torch.bfloat16 or cb.shape[0] != R \
            or cb.shape[1] != meta["NG"] or cb.shape[2] != meta["K"]:
        raise ValueError(f"{path}: cb is {cb.dtype}{tuple(cb.shape)}")
    mask = unpack_mask(mask_packed, R, C).to(device)
    return wq, mask, cb, meta


# ---------------------------------------------------------------------------
# --run mode: monkey-patched full SDOML run with per-layer dumps
# ---------------------------------------------------------------------------
def main_run(dump_dir):
    os.makedirs(dump_dir, exist_ok=True)
    os.chdir(REPO)

    import runpy
    import threading

    STATE = {
        "capture": False,
        "blocks": [],          # per block: (mask [R, w], cb [R, K]) on CPU
        "n_layers": 0,
        "manifest": [],
        "t0": time.time(),
    }

    # ---- patch 1: binary.sdoml_quantize -----------------------------------
    # bigptq re-imports sdoml_quantize locally at call time, so patching the
    # attribute on the `binary` module suffices; no import-order constraint.
    import binary as _bin
    _orig_sdq = _bin.sdoml_quantize

    def _sdq_wrapper(*args, **kwargs):
        out = _orig_sdq(*args, **kwargs)
        if STATE["capture"]:
            # return_aux=True path: out = (W_q, mask, codebook, phi_trace)
            assert isinstance(out, tuple) and len(out) == 4, (
                "sdoml_quantize did not return the 4-tuple aux form; "
                "capture requires return_aux=True")
            _, mask, codebook, _ = out
            STATE["blocks"].append((
                mask.detach().bool().cpu(),
                codebook.detach().float().cpu(),
            ))
        return out

    _bin.sdoml_quantize = _sdq_wrapper

    # ---- patch 2: bigptq.BRAGPTQ.fasterquant ------------------------------
    import bigptq
    _orig_fq = bigptq.BRAGPTQ.fasterquant
    print("SDDUMP: patches installed OK", flush=True)

    @torch.no_grad()
    def _process_layer(self):
        Wq = self.layer.weight.data
        assert Wq.dtype == torch.bfloat16, Wq.dtype
        R, C = int(Wq.shape[0]), int(Wq.shape[1])

        blocks = STATE["blocks"]
        n_blocks_exp = (C + B_BLOCK - 1) // B_BLOCK
        assert len(blocks) == n_blocks_exp, \
            f"captured {len(blocks)} blocks, expected {n_blocks_exp}"
        block_widths = [int(m.shape[1]) for (m, _cb) in blocks]
        assert sum(block_widths) == C, (sum(block_widths), C)
        # all codebooks must share [R, K]
        K = int(blocks[0][1].shape[1])
        for (m, cbk) in blocks:
            assert m.shape[0] == R and cbk.shape == (R, K)

        mask_full = torch.cat([m for (m, _cb) in blocks], dim=1).to(Wq.device)
        assert mask_full.shape == (R, C)
        cb = torch.stack([cbk for (_m, cbk) in blocks], dim=1).to(Wq.device)
        assert cb.shape == (R, n_blocks_exp, K)

        gname = getattr(self.layer, "global_name", None)
        assert gname is not None and gname.startswith(MODEL_NAME), gname
        layer_name = gname[len(MODEL_NAME):]

        sparsity = getattr(self.braq_quantizer, "sparsity", None)
        path, meta = save_layer(dump_dir, layer_name, Wq, mask_full, cb,
                                block_widths, sparsity,
                                model_name=MODEL_NAME)

        rec = {"layer_name": layer_name, "R": R, "C": C, "NG": n_blocks_exp,
               "K": K, "n_kept": meta["n_kept"], "sparsity": meta["sparsity"],
               "reassembly_bitwise": True,
               "t": round(time.time() - STATE["t0"], 1)}
        STATE["manifest"].append(rec)
        STATE["n_layers"] += 1
        frac = meta["n_kept"] / (R * C)
        print(f"SDDUMP[{STATE['n_layers']:3d}] {layer_name} R={R} C={C} "
              f"NG={n_blocks_exp} K={K} kept={frac:.4f} "
              f"reasm=BITWISE-OK", flush=True)

    def _fq_wrapper(self, *args, **kwargs):
        is_sdoml = getattr(self.braq_quantizer, "method", None) == "sdoml"
        STATE["capture"] = is_sdoml
        STATE["blocks"] = []
        out = _orig_fq(self, *args, **kwargs)
        STATE["capture"] = False
        if is_sdoml:
            _process_layer(self)   # raises on any gate violation (no swallow)
            STATE["blocks"] = []
        return out

    bigptq.BRAGPTQ.fasterquant = _fq_wrapper
    print("SDDUMP: BRAGPTQ.fasterquant patched OK", flush=True)

    def _watchdog():
        time.sleep(300)
        if STATE["n_layers"] == 0 and not STATE["blocks"]:
            print("SDDUMP FATAL: no captures after 300 s — hooks dead; "
                  "aborting.", file=sys.stderr, flush=True)
            os._exit(17)

    threading.Thread(target=_watchdog, daemon=True).start()

    sys.argv = list(RUN_ARGV)
    print("SDDUMP: launching run.py:", sys.argv, flush=True)
    err = None
    try:
        runpy.run_path(os.path.join(REPO, "run.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            err = f"SystemExit({e.code})"
    except Exception as e:  # noqa: BLE001
        import traceback
        err = repr(e)
        traceback.print_exc()
    finally:
        manifest = {
            "model": MODEL_NAME,
            "argv": RUN_ARGV[1:],
            "dump_dir": dump_dir,
            "n_sublayers_dumped": STATE["n_layers"],
            "expected_sublayers": EXPECTED_SUBLAYERS,
            "error": err,
            "layers": STATE["manifest"],
        }
        with open(os.path.join(dump_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"SDDUMP: done. sublayers dumped = {STATE['n_layers']} "
              f"(expected {EXPECTED_SUBLAYERS}); error = {err}", flush=True)
    if err:
        sys.exit(1)
    if STATE["n_layers"] != EXPECTED_SUBLAYERS:
        print(f"SDDUMP FATAL: dumped {STATE['n_layers']} != "
              f"{EXPECTED_SUBLAYERS}", file=sys.stderr, flush=True)
        sys.exit(2)


# ---------------------------------------------------------------------------
# --selftest mode
# ---------------------------------------------------------------------------
def main_selftest():
    import tempfile

    gen = torch.Generator().manual_seed(1234)
    R, C, K = 8, 256, 4
    NG = C // B_BLOCK                      # 2 blocks
    block_widths = [B_BLOCK, B_BLOCK]

    # codebooks bf16 [R, NG, K], sorted ascending; inject a duplicate pair.
    cb = torch.randn(R, NG, K, generator=gen).float().sort(dim=-1).values
    cb = cb.to(torch.bfloat16)
    cb[1, 0, 0] = cb[1, 0, 1]              # duplicate level in row 1 block 0

    # masks [R, C]: row 0 fully pruned, last row fully kept, rest random ~50%.
    rnd = torch.rand(R, C, generator=gen)
    mask = rnd < 0.5
    mask[0, :] = False                     # fully-pruned row
    mask[R - 1, :] = True                  # fully-kept row

    codes = torch.randint(0, K, (R, C), generator=gen)
    # build wq: kept -> cb[r, block, code], pruned -> 0 (bf16)
    wq = torch.zeros(R, C, dtype=torch.bfloat16)
    off = 0
    for b, w_b in enumerate(block_widths):
        cval = torch.gather(cb[:, b, :], 1, codes[:, off:off + w_b])
        m = mask[:, off:off + w_b]
        wq[:, off:off + w_b] = torch.where(m, cval,
                                           torch.zeros_like(cval))
        off += w_b

    with tempfile.TemporaryDirectory() as td:
        path, meta = save_layer(td, "selftest.layer", wq, mask, cb,
                                block_widths, sparsity=0.5)
        print("selftest: save_layer + reassembly gate OK; meta =", meta)

        wq2, mask2, cb2, meta2 = load_layer(path)
        assert bool((mask2 == mask).all()), "mask round-trip mismatch"
        assert bool((cb2.view(torch.int16) == cb.view(torch.int16)).all()), \
            "cb round-trip mismatch"
        # independent bitwise reassembly from the RELOADED container
        recon = reassemble_bitwise(wq2, mask2, cb2, meta2["block_widths"])
        assert bool((recon.view(torch.int16)
                     == wq2.view(torch.int16)).all()), \
            "reloaded reassembly not bitwise"
        assert bool((wq2.view(torch.int16) == wq.view(torch.int16)).all()), \
            "wq round-trip mismatch"
        print("selftest: file round-trip bitwise OK "
              f"(R={R} C={C} NG={NG} K={K}, n_kept={meta['n_kept']})")

        # negative control: flip one cb level -> reassembly of a kept position
        # must change (unless that level is unused). Flip a used level.
        cb_bad = cb2.clone()
        old = cb_bad[R - 1, 0, int(codes[R - 1, 0].item())].view(torch.int16)
        cb_bad[R - 1, 0, int(codes[R - 1, 0].item())] = \
            (old ^ 0x40).view(torch.bfloat16)
        try:
            reassemble_bitwise(wq2, mask2, cb_bad, meta2["block_widths"])
            raise AssertionError("corrupted cb did NOT break reassembly")
        except RuntimeError:
            print("selftest: corrupted-cb negative control correctly raised")

    print("SELFTEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model name (default keeps the pre-flag 0.6B "
                         "behavior identical)")
    ap.add_argument("--sparsity", type=float, default=0.5,
                    help="sdoml sparsity for the dump run (default 0.5 = "
                         "the pre-flag behavior)")
    args = ap.parse_args()
    set_model(args.model, args.sparsity)
    if args.selftest:
        main_selftest()
    elif args.run:
        main_run(args.dump_dir)
    else:
        ap.error("choose --run or --selftest")
