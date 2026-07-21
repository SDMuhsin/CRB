"""K5a GATE G-B — Marlin-served model correctness before any perf number.

Two parts, both required:

(1) LAYER GATE (B1 gates, real activations): during a real WikiText-2
    forward of the fully Marlin-served Qwen3-0.6B, capture the ACTUAL
    inputs reaching 8 representative sublayers (all 4 attention shapes +
    gate/up + two down_proj, early AND late layers). For each, compare the
    RAW fp16 Marlin kernel output (captured before the bf16 cast back into
    the residual stream) against the fp32 reference GEMM of the exact fp16
    dequant weights (Wq_ref from the G-A-verified artifacts):
        rms_rel <= 5.1e-4   and   cos_sim >= 0.999999
    plus bitwise determinism across 2 repeated forwards, plus an
    fp16-cast-overflow check on every captured input (bf16 -> fp16 must not
    produce inf).

(2) FULL-MODEL PPL (WikiText-2, seqlen 2048, seed 0 — the protocol of every
    prior milestone): measured twice —
      * resident loop (serving path; ppl_resident in serve_common), and
      * the repo's own eval_ppl_utils.qwen_eval (offloaded protocol
        implementation) as a cross-check.
    Reported as-is. Sanity bound (bug detector, not a target): PPL must be
    < 2x FP16 reference (20.9685) — a wildly-off PPL is a bug, per tasking.

Exit 0 + k5_logs/gate_B_PASS.json only if all gates pass.

Usage:
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=1 python -u kernels/serve/verify_marlin_model.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from serve_common import (  # noqa: E402
    DUMP_DIR, EVAL_SEQLEN, GATE_A_MARKER, GATE_B_MARKER, LOG_DIR, REPO,
    build_marlin_model, dequant_ref_fp16, get_wikitext2_testenc,
    load_q4_artifact, ppl_resident, require_gpu1,
)

import torch  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "kernels", "bench"))
from bench_utils import rel_err  # noqa: E402  (import-only; B1 contract)

FP16_REF_PPL = 20.9685      # FP16 Qwen3-0.6B wikitext2 (project benchmark)
GATE_RMS = 5.1e-4           # B1 gate
GATE_COS = 0.999999         # K5a tasking gate

# 8 representative sublayers: all 4 attention shapes, gate/up, and
# down_proj (the 3072-in shape) — early, middle and late blocks.
GATE_LAYERS = [
    "model.layers.0.self_attn.q_proj",     # 1024 -> 2048
    "model.layers.0.self_attn.k_proj",     # 1024 -> 1024
    "model.layers.0.self_attn.o_proj",     # 2048 -> 1024
    "model.layers.0.mlp.up_proj",          # 1024 -> 3072
    "model.layers.0.mlp.down_proj",        # 3072 -> 1024
    "model.layers.14.mlp.gate_proj",       # 1024 -> 3072 (mid)
    "model.layers.27.self_attn.q_proj",    # late
    "model.layers.27.mlp.down_proj",       # late down_proj
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DUMP_DIR)
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--skip-qwen-eval-crosscheck", action="store_true",
                    help="skip the offloaded qwen_eval protocol cross-check "
                         "(resident-loop PPL is always computed)")
    args = ap.parse_args()
    require_gpu1(args.allow_any_gpu)

    if not os.path.exists(GATE_A_MARKER):
        raise SystemExit(f"G-A marker missing ({GATE_A_MARKER}); run "
                         "verify_marlin_pack.py first — G-B consumes its "
                         "verified artifacts.")

    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    print("G-B: building Marlin-served model ...", flush=True)
    model, n_replaced = build_marlin_model(args.dump_dir)
    model = model.to(dev)
    print(f"G-B: {n_replaced} sublayers replaced; model on {dev}", flush=True)

    test_ids = get_wikitext2_testenc()
    batch = test_ids[:, :EVAL_SEQLEN].to(dev)

    # ---------------- (1) layer gate with real activations ----------------
    mods = {}
    for lname in GATE_LAYERS:
        mod = model.get_submodule(lname)
        mod._capture = []
        mods[lname] = mod

    with torch.no_grad():
        _ = model.model(input_ids=batch, use_cache=False)
        _ = model.model(input_ids=batch, use_cache=False)   # repeat for determinism

    layer_recs = []
    all_ok = True
    for lname, mod in mods.items():
        caps = mod._capture
        mod._capture = None
        assert len(caps) == 2, (lname, len(caps))
        (x_bf16, x16, c16), (_, x16b, c16b) = caps
        det_ok = torch.equal(c16, c16b) and torch.equal(x16, x16b)
        # fp16-cast overflow check on the real input
        cast_inf = int(torch.isinf(x16).sum().item())
        src_inf = int(torch.isinf(x_bf16).sum().item())
        max_abs_in = x_bf16.float().abs().max().item()

        q, s, _meta = load_q4_artifact(lname, args.dump_dir)
        Wq_ref = dequant_ref_fp16(q, s).to(dev)              # (N, K) fp16
        ref = x16.float() @ Wq_ref.float().t()               # fp32 reference
        err = rel_err(c16, ref)
        ok = (err["rms_rel"] <= GATE_RMS and err["cos_sim"] >= GATE_COS
              and det_ok and cast_inf == src_inf == 0)
        all_ok &= ok
        rec = {"layer": lname, "K": q.shape[1], "N": q.shape[0],
               "tokens": int(x16.shape[0]),
               "max_abs_input": max_abs_in,
               "fp16_cast_inf": cast_inf, "input_inf": src_inf,
               "bitwise_deterministic": det_ok, **err,
               "gate": {"rms_rel<=": GATE_RMS, "cos_sim>=": GATE_COS},
               "pass": ok}
        layer_recs.append(rec)
        print(f"  {lname}: rms_rel={err['rms_rel']:.3e} "
              f"cos={err['cos_sim']:.8f} det={det_ok} "
              f"max|x|={max_abs_in:.1f} [{'PASS' if ok else 'FAIL'}]",
              flush=True)
        del Wq_ref, ref, caps

    torch.cuda.empty_cache()

    # ---------------- (2) full-model PPL -----------------------------------
    print("G-B: resident-loop PPL (serving path) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        ppl_res, nsamples = ppl_resident(model, test_ids, dev,
                                         progress_every=50)
    t_res = time.time() - t0
    print(f"G-B: resident PPL = {ppl_res:.6f} over {nsamples} samples "
          f"({t_res:.0f}s)", flush=True)

    ppl_qwen_eval = None
    if not args.skip_qwen_eval_crosscheck:
        print("G-B: qwen_eval protocol cross-check ...", flush=True)
        from eval_ppl_utils import qwen_eval
        from datautils import get_loaders
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            _, testenc_obj = get_loaders("wikitext2", nsamples=128, seed=0,
                                         seqlen=EVAL_SEQLEN,
                                         model="Qwen/Qwen3-0.6B")
            ppl_qwen_eval = qwen_eval(model, testenc_obj, dev, "wikitext2",
                                      save=False)
        finally:
            os.chdir(cwd)
        # qwen_eval offloads layers to CPU as it goes; move back for callers
        model = model.to(dev)

    ppl_ok = ppl_res < 2.0 * FP16_REF_PPL
    xcheck_ok = (ppl_qwen_eval is None
                 or abs(ppl_qwen_eval - ppl_res) / ppl_res < 1e-3)
    verdict = all_ok and ppl_ok and xcheck_ok

    out = {
        "gate": "G-B (Marlin-served model correctness + PPL)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dump_dir": os.path.abspath(args.dump_dir),
        "n_sublayers_served": n_replaced,
        "activation_handling": "model bf16; per-layer cast bf16->fp16 in, "
                               "fp16 Marlin out cast back to bf16",
        "layer_gate": {"gate_rms_rel": GATE_RMS, "gate_cos": GATE_COS,
                       "worst_rms_rel": max(r["rms_rel"] for r in layer_recs),
                       "worst_cos": min(r["cos_sim"] for r in layer_recs),
                       "layers": layer_recs},
        "ppl_wikitext2_resident_loop": ppl_res,
        "ppl_wikitext2_qwen_eval_protocol": ppl_qwen_eval,
        "ppl_nsamples": nsamples,
        "fp16_reference_ppl": FP16_REF_PPL,
        "ppl_sanity_bound_2x_fp16": ppl_ok,
        "ppl_crosscheck_within_0.1pct": xcheck_ok,
        "pass": verdict,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(GATE_B_MARKER if verdict
              else os.path.join(LOG_DIR, "gate_B_FAIL.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nG-B layer gate: worst rms_rel="
          f"{out['layer_gate']['worst_rms_rel']:.3e}, worst cos="
          f"{out['layer_gate']['worst_cos']:.8f}")
    print(f"G-B PPL: resident={ppl_res:.4f}  qwen_eval={ppl_qwen_eval}  "
          f"(FP16 ref {FP16_REF_PPL})")
    print(f"GATE G-B: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
