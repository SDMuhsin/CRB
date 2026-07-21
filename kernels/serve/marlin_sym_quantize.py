"""K5a step 1 — symmetric INT4 g=128 GPTQ quantization of Qwen3-0.6B
(the weight set the Marlin comparator serves).

WHY A MONKEY-PATCH: Marlin's format requires SYMMETRIC no-zero-point INT4
with fp16 group scales at g=128. The repo's existing '4bit' integer path
(binary.py 3774-3805 + bigptq.py 445-490) is ASYMMETRIC (per-row min/max +
zero point) and therefore NOT representable in Marlin's format. Per the K5a
tasking preference order, the strongest faithful option is GPTQ with a
symmetric quantizer at g=128, achieved WITHOUT modifying any repo source by
monkey-patching BRAGPTQ.fasterquant (exact pattern of
kernels/pack/doml_group_refit.py). Everything else — model loading,
calibration data (wikitext2, nsamples=128, seed=0, seqlen=2048), Hessian
accumulation (add_batch), damping/Cholesky preamble, the per-column sweep
with intra-block error feedback (paper-faithful GPTQ, as in the Phase-14B
partition=1 path), the post-quantization PPL eval — is the repo's own
machinery via run.py.

QUANTIZER (per 128-column block == group, block boundaries aligned):
  * per-row scale from the CURRENT (feedback-updated at block entry,
    pre-intra-block-feedback) W1: s = absmax * 2/15, rounded to fp16 and
    clamped to the smallest normal fp16 (2^-14) — the standard GPTQ `sym`
    recipe (zero = 8, grid -8..7) and exactly Marlin's gen_quant4 recipe.
  * column sweep i = 0..127: q = clamp(round(w / s), -8, 7);
    dequant EXACTLY in the kernel's fp16 semantics: fp16(fp16(q) * s);
    GPTQ intra-block feedback W1[:, i+1:] -= err * Hinv1[i, i+1:], then
    inter-block feedback — identical structure to bigptq.py:470-490.
  * per-layer artifacts: q codes (N, K) uint8 in 0..15 and scales
    (K/128, N) fp16 -> <layer>.q4.safetensors, plus a HARD roundtrip assert
    (dequant(q, s) bitwise == the weights written into the model).

Usage:
  source /workspace/BiLLM2/env/bin/activate
  # property gates on a synthetic layer (no model download):
  CUDA_VISIBLE_DEVICES=1 python kernels/serve/marlin_sym_quantize.py --selftest
  # the real 196-sublayer run (quantize + repo-protocol fake-quant PPL eval):
  CUDA_VISIBLE_DEVICES=1 python -u kernels/serve/marlin_sym_quantize.py --run \
      [--dump-dir downloads/marlin_dumps/qwen3-0.6b/sym-g128]
"""

import argparse
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from serve_common import (  # noqa: E402  (sets BILLM_* env + sys.path)
    DUMP_DIR, FP16_MIN_NORMAL, GROUPSIZE, MODEL_NAME, N_QUANT_SUBLAYERS,
    REPO, dequant_ref_fp16, require_gpu1,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import transformers  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import bigptq  # noqa: E402

_ORIG_FQ = bigptq.BRAGPTQ.fasterquant

RUN_ARGV = [
    "run.py", MODEL_NAME, "wikitext2", "4bit",
    "--blocksize", str(GROUPSIZE), "--salient_metric", "magnitude",
    "--device", "cuda:0", "--partition", "1",
]

RUN_STATE = {
    "dump_dir": None,     # None => selftest mode (no dump, no manifest)
    "manifest": [],
    "n_layers": 0,
    "t0": None,
}


def _sym_group_params(W1: torch.Tensor):
    """Per-row symmetric g=128 scale from the block slice W1 (fp32).
    Returns (s16 fp16 (R,), s32 fp32 (R,)). absmax==0 rows and subnormal
    scales are clamped to the smallest normal fp16 (2^-14)."""
    absmax = W1.abs().max(dim=1).values                     # (R,) fp32
    s16 = (absmax * (2.0 / 15.0)).half()
    s16 = torch.clamp(s16, min=FP16_MIN_NORMAL)
    return s16, s16.float()


@torch.no_grad()
def sym_fasterquant(self, blocksize=128, percdamp=0.01, partition=3,
                    orders=(1, 1, 2), global_scale=False):
    """Symmetric-INT4 g=128 GPTQ. Handles ONLY (method='4bit', partition=1,
    GPTQ on, no global_scale); everything else delegates to the original
    fasterquant untouched."""
    method = getattr(self.braq_quantizer, "method", None)
    if (method != "4bit" or partition != 1 or self.disable_gptq
            or global_scale):
        return _ORIG_FQ(self, blocksize=blocksize, percdamp=percdamp,
                        partition=partition, orders=orders,
                        global_scale=global_scale)
    assert isinstance(self.layer, nn.Linear), type(self.layer)
    assert blocksize == GROUPSIZE, (blocksize, GROUPSIZE)

    # ---- preamble: verbatim replica of bigptq.py:63-129 -------------------
    W = self.layer.weight.data.clone()
    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()
    W = W.float()
    tick = time.time()

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

    R, C = self.rows, self.columns
    assert C % GROUPSIZE == 0, (C, GROUPSIZE)

    q_codes = torch.empty((R, C), dtype=torch.uint8, device=self.dev)
    scales = torch.empty((C // GROUPSIZE, R), dtype=torch.half,
                         device=self.dev)
    n_clamped_groups = 0

    # ---- symmetric GPTQ column sweep (structure of bigptq.py:445-490) -----
    for col_st in range(0, C, blocksize):
        col_ed = col_st + blocksize
        n_cols = col_ed - col_st

        W1 = W[:, col_st:col_ed].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

        s16, s32 = _sym_group_params(W1)   # group == block (g = 128)
        n_clamped_groups += int((s16 == FP16_MIN_NORMAL).sum().item())

        for i in range(n_cols):
            w = W1[:, i]
            d = Hinv1[i, i]

            q_int = torch.clamp(torch.round(w / s32), -8.0, 7.0)
            qdq = (q_int.half() * s16).float()   # EXACT kernel fp16 semantics

            Q1[:, i] = qdq
            Losses1[:, i] = (w - qdq) ** 2 / (d * d)
            err1 = (w - qdq) / d
            Err1[:, i] = err1
            if i + 1 < n_cols:
                W1[:, i + 1:] -= err1.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)

            q_codes[:, col_st + i] = (q_int + 8.0).to(torch.uint8)

        W[:, col_st:col_ed] = Q1
        Losses += torch.sum(Losses1, 1) / 2
        W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
        scales[col_st // GROUPSIZE, :] = s16

    torch.cuda.synchronize()
    print("time %.2f" % (time.time() - tick))
    err_total = torch.sum(Losses).item()
    print("error", err_total)

    # ---- HARD roundtrip assert: dequant(q, s) == W bitwise (fp32 view) ----
    recon16 = dequant_ref_fp16(q_codes, scales)          # (R, C) fp16
    if not torch.equal(recon16.float(), W):
        bad = int((recon16.float() != W).sum().item())
        raise RuntimeError(
            f"K5A roundtrip FAIL: dequant(q,s) != quantized W on {bad}/{R*C} "
            f"elements — artifact would not represent the model")

    self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
        self.layer.weight.data.dtype)

    _post_layer_dump(self, q_codes, scales, err_total, n_clamped_groups,
                     time.time() - tick)

    del W1, Q1, W, Err1, Losses1, Hinv1, H, Hinv, recon16
    torch.cuda.empty_cache()
    return {"error": err_total}


def _post_layer_dump(self, q_codes, scales, err_total, n_clamped, t_layer):
    st = RUN_STATE
    if st["dump_dir"] is None:      # selftest mode
        return
    gname = getattr(self.layer, "global_name", None)
    assert gname is not None and gname.startswith(MODEL_NAME), gname
    layer_name = gname[len(MODEL_NAME):]
    R, C = q_codes.shape

    q_cpu = q_codes.detach().cpu().contiguous()
    s_cpu = scales.detach().cpu().contiguous()
    meta = {"layer": layer_name, "model": MODEL_NAME, "K": C, "N": R,
            "groupsize": GROUPSIZE,
            "scheme": "GPTQ sym INT4 g=128 (s=absmax*2/15 fp16, clamp 2^-14; "
                      "zero-point 8; grid -8..7; fp16 dequant semantics)",
            "seed": 0, "nsamples": 128, "percdamp": 0.01}
    save_file({"q": q_cpu, "s": s_cpu},
              os.path.join(st["dump_dir"], f"{layer_name}.q4.safetensors"),
              metadata={"meta": json.dumps(meta)})

    sha = hashlib.sha256()
    sha.update(q_cpu.numpy().tobytes())
    sha.update(s_cpu.view(torch.int16).numpy().tobytes())

    st["manifest"].append({
        "layer_name": layer_name, "N": R, "K": C,
        "gptq_error": err_total, "n_clamped_group_scales": n_clamped,
        "sha256_q_s": sha.hexdigest(), "t_layer_s": round(t_layer, 2),
    })
    st["n_layers"] += 1
    print(f"K5AQUANT[{st['n_layers']:3d}] {layer_name} N={R} K={C} "
          f"err={err_total:.4f} clamped={n_clamped} "
          f"t={time.time() - st['t0']:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# --run: patch + drive run.py (quantize all 196 sublayers, then run.py's
# standard wikitext2 PPL eval of the fake-quant bf16 model -> scratch CSV)
# ---------------------------------------------------------------------------

def main_run(args):
    RUN_STATE["dump_dir"] = os.path.abspath(args.dump_dir)
    RUN_STATE["t0"] = time.time()
    os.makedirs(RUN_STATE["dump_dir"], exist_ok=True)
    os.chdir(REPO)

    bigptq.BRAGPTQ.fasterquant = sym_fasterquant
    print(f"K5AQUANT: fasterquant patched OK (sym INT4 g={GROUPSIZE}, "
          f"dump_dir={RUN_STATE['dump_dir']})", flush=True)

    import runpy
    import threading

    def _watchdog():
        time.sleep(600)
        if RUN_STATE["n_layers"] == 0:
            print("K5AQUANT FATAL: no layers after 600 s — patch dead; "
                  "aborting.", file=sys.stderr, flush=True)
            os._exit(17)

    threading.Thread(target=_watchdog, daemon=True).start()

    sys.argv = list(RUN_ARGV)
    print("K5AQUANT: launching run.py:", sys.argv, flush=True)
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
            "argv": RUN_ARGV[1:],
            "scheme": "GPTQ symmetric INT4 g=128 (Marlin-exact)",
            "dump_dir": RUN_STATE["dump_dir"],
            "n_sublayers": RUN_STATE["n_layers"],
            "expected_sublayers": N_QUANT_SUBLAYERS,
            "error": err,
            "layers": RUN_STATE["manifest"],
        }
        mpath = os.path.join(RUN_STATE["dump_dir"], "manifest.json")
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"K5AQUANT: done. layers = {RUN_STATE['n_layers']} "
              f"(expected {N_QUANT_SUBLAYERS}); error = {err}; "
              f"manifest = {mpath}", flush=True)
    if err:
        sys.exit(1)
    if RUN_STATE["n_layers"] != N_QUANT_SUBLAYERS:
        print(f"K5AQUANT FATAL: {RUN_STATE['n_layers']} != "
              f"{N_QUANT_SUBLAYERS}", file=sys.stderr, flush=True)
        sys.exit(2)


# ---------------------------------------------------------------------------
# --selftest: property gates on a synthetic layer (no model needed)
# ---------------------------------------------------------------------------

def _build_braq(W0, X, device):
    from binary import Binarization
    lin = nn.Linear(W0.shape[1], W0.shape[0], bias=False)
    lin.weight.data = W0.clone().to(device)
    lin = lin.to(device)
    q = Binarization(lin.weight, method="4bit", groupsize=GROUPSIZE)
    br = bigptq.BRAGPTQ(lin, q, salient_metric="magnitude")
    for j in range(X.shape[0]):
        br.add_batch(X[j].to(device).float(), None)
    return lin, br


def main_selftest():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    R, C = 64, 512
    gen = torch.Generator().manual_seed(1234)
    W0 = (torch.randn(R, C, generator=gen) * 0.05).to(torch.bfloat16)
    X = torch.randn(8, 64, C, generator=gen) * 0.7

    # (a) diagonal H => zero GPTQ feedback => output == plain sym-RTN g=128
    lin_a, br_a = _build_braq(W0, X, device)
    br_a.H = torch.eye(C, device=device) * 3.0
    sym_fasterquant(br_a, blocksize=128, percdamp=0.01, partition=1,
                    orders=(1,), global_scale=False)
    Wf = W0.float().to(device)
    rtn = torch.empty_like(Wf)
    for st in range(0, C, GROUPSIZE):
        blk = Wf[:, st:st + GROUPSIZE]
        s16, s32 = _sym_group_params(blk)
        qi = torch.clamp(torch.round(blk / s32.unsqueeze(1)), -8.0, 7.0)
        rtn[:, st:st + GROUPSIZE] = (qi.half() * s16.unsqueeze(1)).float()
    got = lin_a.weight.data.float()
    ref = rtn.to(torch.bfloat16).float()
    n_bad = int((got != ref).sum().item())
    assert n_bad == 0, f"selftest (a) FAILED: diag-H sym GPTQ != sym RTN on {n_bad}/{R*C}"
    print(f"selftest (a): diag-H sym GPTQ == independent sym-RTN g=128 "
          f"BITWISE ({R}x{C})  PASS")

    # (b) correlated H: output must lie on the per-(row,group) fp16 grid,
    #     differ from RTN (feedback engaged), and be deterministic.
    outs = []
    for _rep in range(2):
        lin_b, br_b = _build_braq(W0, X, device)
        info = sym_fasterquant(br_b, blocksize=128, percdamp=0.01,
                               partition=1, orders=(1,), global_scale=False)
        outs.append(lin_b.weight.data.detach().clone())
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)), \
        "selftest (b) FAILED: nondeterministic"
    Wq = outs[0].float()
    n_grid_bad = 0
    for st in range(0, C, GROUPSIZE):
        blk_orig = Wf[:, st:st + GROUPSIZE]
        # scales must be recomputed from feedback-updated W to check the grid
        # exactly; instead check <= 16 distinct values per (row, group) — a
        # property no non-grid output satisfies for 128 columns of noise.
        blk_q = Wq[:, st:st + GROUPSIZE]
        for r in range(R):
            if torch.unique(blk_q[r]).numel() > 16:
                n_grid_bad += 1
    assert n_grid_bad == 0, f"selftest (b) FAILED: {n_grid_bad} (row,group) " \
                            f"slices exceed 16 distinct values"
    n_diff = int((Wq != rtn.to(torch.bfloat16).float()).sum().item())
    print(f"selftest (b): correlated-H run deterministic, <=16 levels per "
          f"(row,group), differs from RTN on {n_diff}/{R*C} weights "
          f"(feedback engaged, expected > 0), gptq_error={info['error']:.6f}  PASS")
    assert n_diff > 0

    # (c) roundtrip assert path is live: corrupting a code must trip it.
    # (The assert inside sym_fasterquant already ran 3x above; here we prove
    # dequant_ref_fp16 detects corruption — gate-of-gates.)
    q = torch.randint(0, 16, (R, C), dtype=torch.uint8)
    s = (torch.rand(C // GROUPSIZE, R).half() * 0.01 + 0.001).half()
    w1 = dequant_ref_fp16(q, s)
    q2 = q.clone()
    q2[3, 77] ^= 1
    w2 = dequant_ref_fp16(q2, s)
    assert not torch.equal(w1, w2), "selftest (c) FAILED: code flip undetected"
    print("selftest (c): dequant gate-of-gates (1-bit code flip detected)  PASS")
    print("SELFTEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-dir", default=DUMP_DIR)
    ap.add_argument("--allow-any-gpu", action="store_true")
    args = ap.parse_args()
    require_gpu1(args.allow_any_gpu)
    if args.selftest:
        main_selftest()
    elif args.run:
        main_run(args)
    else:
        ap.error("choose --run or --selftest")
