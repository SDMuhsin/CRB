"""SDOML restore-eval harness — reproduce an SDOML dump's own PPL bitwise.

Mirrors the K33 `--restore-dpk` pattern in doml_group_refit.py (main_restore),
adapted to the SDOML BASE container `<layer_name>.sdpk.safetensors`
(sdoml_dump.py).  Instead of quantizing, the monkey-patched
`bigptq.BRAGPTQ.fasterquant` loads the container's `wq` tensor and sets
`self.layer.weight.data = wq` (shape/dtype/device checked), then lets the
UNTOUCHED run.py finish with its standard wikitext2 seed-0 PPL eval (plus c4/ptb
when --eval-extra-ppl is passed).  NOTHING is written.

Because `wq` is the EXACT stored quantized weight, a restored dump reproduces
its own dump run's PPL to the last digit.  For the sdoml-s50 dump this is a
HARD GATE: wt2 PPL must equal 63.9822 exactly.

G0 discipline: run.py source is untouched.  The patch only replaces the
quantization work for sdoml sublayers; the model build, calibration, dataset
and eval are run.py's standard paths.  A hard pre-eval guard aborts (os._exit)
before ANY PPL is computed unless ALL EXPECTED_SUBLAYERS were restored.

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python -u kernels/pack/sdoml_restore_eval.py \
        --dir downloads/doml_dumps/qwen3-0.6b/sdoml-s50 --device cuda:0
    # add --eval-extra-ppl for c4 + ptb PPL too
"""

import argparse
import glob
import json
import os
import sys
import time

REPO = "/workspace/BiLLM2"
VERIFY_DIR = os.path.join(REPO, "llmdocs", "cuda_kernel", "verify")

# Must be set before run.py / csv_utils are imported (redirect the bench CSV so
# a restore/eval never pollutes the main results CSV).
os.environ.setdefault(
    "BILLM_BENCH_CSV", os.path.join(VERIFY_DIR, "scratch_results.csv"))

if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import bigptq  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MODEL_NAME = DEFAULT_MODEL          # resolved from the dump manifest in main
EXPECTED_SUBLAYERS = 196            # recomputed after model resolution


def build_run_argv(device, manifest_argv=None):
    """The dump run's argv with --device substituted. Prefers the argv the
    manifest recorded (the exact run that produced the dump — 2026-07-20);
    legacy dumps without one fall back to the original 0.6B s=0.5 argv."""
    if manifest_argv:
        argv = ["run.py"] + list(manifest_argv)
        for i, a in enumerate(argv):
            if a == "--device" and i + 1 < len(argv):
                argv[i + 1] = device
        return argv
    return [
        "run.py", MODEL_NAME, "wikitext2", "sdoml",
        "--blocksize", "128", "--salient_metric", "magnitude",
        "--device", device, "--sparsity", "0.5", "--sdoml_n_iter", "20",
    ]


def main_restore(args):
    dump_dir = os.path.abspath(args.dir)
    if not os.path.isdir(dump_dir):
        raise SystemExit(f"SDRESTORE FATAL: {dump_dir} is not a directory")
    n_sdpk = len(glob.glob(os.path.join(dump_dir, "*.sdpk.safetensors")))
    print(f"SDRESTORE: dump dir {dump_dir} has {n_sdpk} sdpk sublayer files",
          flush=True)

    # 2026-07-20: resolve model from the dump manifest (--model may confirm,
    # mismatch = hard error) and derive EXPECTED_SUBLAYERS from its config
    # BEFORE the count check (4B has 36 blocks; 196 was a 0.6B coincidence).
    man = {}
    mp = os.path.join(dump_dir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            man = json.load(f)
    man_model = man.get("model") or (man.get("argv") or [None])[0]
    if args.model and man_model and args.model != man_model:
        raise SystemExit(
            f"SDRESTORE FATAL: --model {args.model} != dump manifest model "
            f"{man_model} — refusing to mix models")
    global MODEL_NAME, EXPECTED_SUBLAYERS
    MODEL_NAME = args.model or man_model or DEFAULT_MODEL
    from transformers import AutoConfig
    cache_dir = os.environ.get("BILLM_DOWNLOADS_DIR",
                               os.path.join(REPO, "downloads"))
    EXPECTED_SUBLAYERS = 7 * AutoConfig.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir).num_hidden_layers
    print(f"SDRESTORE: model = {MODEL_NAME} (manifest={man_model}, "
          f"--model={args.model}, sublayers={EXPECTED_SUBLAYERS})",
          flush=True)
    if n_sdpk != EXPECTED_SUBLAYERS:
        raise SystemExit(f"SDRESTORE FATAL: {n_sdpk} sdpk files != "
                         f"{EXPECTED_SUBLAYERS} expected sublayers")
    os.chdir(REPO)

    state = {"n": 0, "t0": time.time()}

    def restore_fasterquant(self, blocksize=128, percdamp=0.01, partition=1,
                            orders=(1,), global_scale=False):
        method = getattr(self.braq_quantizer, "method", None)
        if method != "sdoml":
            raise RuntimeError(
                f"SDRESTORE: expected an sdoml sublayer, got method={method} "
                f"— restore mode only supports the standard sdoml run")
        gname = getattr(self.layer, "global_name", None)
        assert gname is not None and gname.startswith(MODEL_NAME), gname
        layer_name = gname[len(MODEL_NAME):]
        sdpk_path = os.path.join(dump_dir, f"{layer_name}.sdpk.safetensors")
        if not os.path.exists(sdpk_path):
            raise FileNotFoundError(f"SDRESTORE: missing {sdpk_path}")
        with safe_open(sdpk_path, framework="pt", device="cpu") as f:
            if "wq" not in set(f.keys()):
                raise RuntimeError(f"{sdpk_path}: no 'wq' tensor")
            wq = f.get_tensor("wq")
        W = self.layer.weight
        if wq.dtype != W.dtype or tuple(wq.shape) != tuple(W.shape):
            raise RuntimeError(
                f"SDRESTORE: {layer_name}: dump wq is "
                f"{wq.dtype}{tuple(wq.shape)}, layer weight is "
                f"{W.dtype}{tuple(W.shape)}")
        self.layer.weight.data = wq.to(W.device)
        self.H = None                      # parity with fasterquant's del self.H
        state["n"] += 1
        print(f"SDRESTORE[{state['n']:3d}] {layer_name} "
              f"R={wq.shape[0]} C={wq.shape[1]} restored "
              f"t={time.time() - state['t0']:.1f}s", flush=True)
        return {"error": float("nan"), "restored": True}

    bigptq.BRAGPTQ.fasterquant = restore_fasterquant
    print(f"SDRESTORE: fasterquant patched to RESTORE from {dump_dir}",
          flush=True)

    # Hard pre-eval guard: PPL must never be computed on a partially restored
    # model. run.py resolves evaluate_and_log_all against sys.modules at its
    # (runpy) import time, so patching the module attribute here is picked up.
    import eval_utils
    _orig_eval = eval_utils.evaluate_and_log_all

    def _guarded_eval(*a, **kw):
        if state["n"] != EXPECTED_SUBLAYERS:
            print(f"SDRESTORE FATAL: eval reached with only {state['n']}/"
                  f"{EXPECTED_SUBLAYERS} sublayers restored — aborting before "
                  f"any PPL is computed.", file=sys.stderr, flush=True)
            os._exit(3)
        print(f"SDRESTORE: all {state['n']}/{EXPECTED_SUBLAYERS} sublayers "
              f"restored — proceeding to the standard eval.", flush=True)
        return _orig_eval(*a, **kw)

    eval_utils.evaluate_and_log_all = _guarded_eval

    import runpy
    import threading

    def _watchdog():
        time.sleep(300)
        if state["n"] == 0:
            print("SDRESTORE FATAL: no restores after 300 s — patch dead; "
                  "aborting.", file=sys.stderr, flush=True)
            os._exit(17)

    threading.Thread(target=_watchdog, daemon=True).start()

    sys.argv = build_run_argv(args.device, man.get("argv"))
    if args.eval_extra_ppl:
        sys.argv.append("--eval_extra_ppl")
    if args.eval_arc:
        sys.argv.append("--eval_arc")
    print("SDRESTORE: launching run.py:", sys.argv, flush=True)
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
    print(f"SDRESTORE: done. sublayers restored = {state['n']} "
          f"(expected {EXPECTED_SUBLAYERS}); error = {err}", flush=True)
    if err:
        sys.exit(1)
    if state["n"] != EXPECTED_SUBLAYERS:
        print(f"SDRESTORE FATAL: restored {state['n']} != "
              f"{EXPECTED_SUBLAYERS}", file=sys.stderr, flush=True)
        sys.exit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="SDOML dump dir of *.sdpk.safetensors containers")
    ap.add_argument("--model", default=None,
                    help="HF model name; default = the dump manifest's model. "
                         "A --model that contradicts the manifest is a hard "
                         "error.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-extra-ppl", action="store_true",
                    help="also evaluate c4 + ptb PPL (passes --eval_extra_ppl "
                         "through to run.py)")
    ap.add_argument("--eval-arc", action="store_true",
                    help="also evaluate ARC-Easy + ARC-Challenge accuracy "
                         "(passes --eval_arc through to run.py)")
    args = ap.parse_args()
    main_restore(args)
