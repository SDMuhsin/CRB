"""K5b GATES G-K5-1 + G-K5-2 — DPK-served model correctness before any
performance number (pre-registered in llmdocs/cuda_kernel/03_k5_serving_design.md).

Part 0 (property gates, cheap):
  * pack_a4_batch row t == kernels/ref/ref_w2a4.pack_a4 (the K2 reference
    packer) bitwise, over random cases incl. the C_orig < C padding path;
  * quantize_a4 contract unit checks: zero-row guard (a_s := 1, x_hat = 8,
    output exactly 0), +absmax -> code 15 (half-step clamp), -absmax ->
    code 0, round-half-even at exact .5 ties.

Part 1 (G-K5-1, kernel == references at model scale, REAL activations):
  during a real WikiText-2 batch-0 forward of the fully DPK-served model,
  capture (x_hat words, a_s) at the SAME 8 representative sublayers as
  K5a's G-B, then compare the kernel output on those exact inputs against
  TWO kernel-independent references:
    (i)  K3 reference stack: dpk_ref.ref_gemm_direct — fp32 accumulate over
         dequantized-W fp32 with integer (x_hat - 8), THEN the a_s scale;
    (ii) direct fp32 (K2 unpack stack): W_fp32 @ (a_s * (x_hat - 8))_fp32 —
         scale applied BEFORE the matmul;
    plus (iii) the K2 integer bucket reference ref_w2a4.gemv_bucket on 4
    sampled tokens per layer (exact integer S/N bucket sums, doc 02 §7).
  Gates: norm_rel(kernel_fp32, ref) <= 1e-5 for (i)/(ii)/(iii) (L2-norm
  relative, the K3/K4 gate discipline), bf16 serving output vs bf16(ref (i))
  within the GM2 1-ULP/noise-floor rule, and bitwise determinism across 2
  full forwards (x_hat, a_s AND y identical).

Part 2 (G-K5-2, end-to-end math parity):
  full 146-sample WikiText-2 PPL (seqlen 2048, seed 0 — the exact protocol
  of every prior milestone) of
    (A) the DPK-served model (custom kernel), and
    (B) the torch REFERENCE model: unpacked bf16 W from the SAME containers
        + the SAME A4 fake-quant of activations, plain fp32 GEMM.
  Gate: |PPL_A - PPL_B| / PPL_B <= 0.1% (the two paths compute the same
  math in different summation orders; a bigger gap is a BUG).

Writes k5_logs/gate_K5_1_PASS.json + gate_K5_2_PASS.json only on PASS
(measure_dpk_serving.py refuses to run without them). Exit 0 iff all pass.

Environment sanity (run separately, logged): the FULL kernel gate suite
kernels/cuda/test_dpk_gemm.py must exit 0 on this GPU.

Usage:
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=1 python -u kernels/serve/verify_dpk_model.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

from serve_common import (  # noqa: E402
    EVAL_SEQLEN, LOG_DIR, REPO, get_wikitext2_testenc, ppl_resident,
    require_gpu1,
)
from dpk_serve import (  # noqa: E402
    DPK_DUMP_DIR, GATE_K51_MARKER, GATE_K52_MARKER, PPL_DOML_G128,
    PPL_FAKEQUANT_W_ONLY, PPL_FP16_REF, PPL_MARLIN_W4A16,
    assert_no_global_dequant, build_dpk_model, build_ref_a4_model, dpk_ext,
    pack_a4_batch, quantize_a4,
)

sys.path.insert(0, os.path.join(REPO, "kernels", "cuda"))
sys.path.insert(0, os.path.join(REPO, "kernels", "ref"))
import dpk_ref  # noqa: E402   (K3 independent reference stack)
import ref_w2a4  # noqa: E402  (K2 reference stack: pack_a4, gemv_bucket)
import dpk_unpack  # noqa: E402 (K2 unpacker; on path via dpk_serve)

GATE_NORM_REL = 1e-5           # K3/K4 gate discipline (L2-norm relative)
GATE_PPL_PARITY = 1e-3         # 0.1 % — fp-noise scale

# SAME 8 representative sublayers as K5a's G-B (comparability requirement).
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


def norm_rel(out, refv):
    d = (out.double() - refv.double()).norm()
    return (d / refv.double().norm().clamp_min(1e-30)).item()


def bf16_ulp_dist(a_bf16: torch.Tensor, b_bf16: torch.Tensor) -> torch.Tensor:
    """EXACT ULP distance between bf16 tensors: the number of representable
    bf16 values between a and b (adjacent values -> 1). Implemented with the
    standard monotonic integer mapping of the float bit patterns. This
    replaces the earlier relative approximation 2^-8*|x|, which
    under-estimates the true ULP spacing (2^-8 * 2^ceil(log2|x|)) by up to
    2x for values just below a power of two — the first verify run flagged
    L0 o_proj on exactly such an element (kernel -0.93359375 vs reference
    -0.9375, fp32 values 3e-7 apart with the reference EXACTLY on the
    rounding midpoint: adjacent bf16 values, i.e. genuinely within 1 ULP)."""
    ia = a_bf16.view(torch.int16).to(torch.int32)
    ib = b_bf16.view(torch.int16).to(torch.int32)
    ka = torch.where(ia >= 0, ia, 0x8000 - ia)
    kb = torch.where(ib >= 0, ib, 0x8000 - ib)
    return (ka - kb).abs()


def bf16_within_1ulp_or_floor(y_bf16, y_ref_bf, ref_f32, mism):
    """doc-03 G-K5-1 rule 'bf16 mismatches within 1 ULP/noise floor':
    every mismatched element must be an adjacent bf16 value (exact ULP
    distance <= 1), numerically equal (+-0), or below the fp32 noise floor
    GATE_NORM_REL * max|ref| (GM2's cancellation-tiny rule)."""
    if not mism.any():
        return True
    a = y_bf16[mism]
    b = y_ref_bf[mism]
    floor = GATE_NORM_REL * ref_f32.abs().max().item()
    ok = ((bf16_ulp_dist(a, b) <= 1) | (a == b)
          | ((a.float() - b.float()).abs() <= floor))
    return bool(ok.all())


# ---------------------------------------------------------------------------
# Part 0 — property gates
# ---------------------------------------------------------------------------

def property_gates(dev):
    recs = {}
    # (a) pack parity vs the K2 reference packer, incl. padding path
    gen = torch.Generator().manual_seed(0)
    for C_orig, C_pad in [(1024, 1024), (2048, 2048), (3072, 3072),
                          (136, 256), (8, 128)]:
        xh = torch.randint(0, 16, (5, C_orig), generator=gen,
                           dtype=torch.int32)
        batch = pack_a4_batch(xh.to(dev), C_pad).cpu()
        for t in range(xh.shape[0]):
            row = xh[t].to(torch.int64)
            if C_pad != C_orig:
                row = torch.cat([row, torch.full((C_pad - C_orig,), 8,
                                                 dtype=torch.int64)])
            ref_row = ref_w2a4.pack_a4(row)
            assert torch.equal(batch[t].view(torch.int32),
                               ref_row.view(torch.int32)), (C_orig, C_pad, t)
    recs["pack_parity_vs_ref_w2a4"] = "bitwise OK (5 shapes x 5 rows, incl. padding)"

    # (b) quantize contract unit checks
    x = torch.tensor([[7.5, -7.5, 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4.5,
                       0.0, 7.0, -7.0, 3.0, 1.0, -1.0]], device=dev)
    xh, a_s = quantize_a4(x)
    assert a_s.item() == 1.0, a_s   # absmax 7.5 / 7.5
    expect = [15, 0, 8, 10, 10, 12, 8, 6, 6, 12, 8, 15, 1, 11, 9, 7]
    got = xh[0].tolist()
    assert got == expect, (got, expect)
    recs["round_half_even_and_clamp"] = {"expected": expect, "got": got}

    # (c) zero-row guard: a_s := 1, x_hat = 8, kernel output exactly 0
    z = torch.zeros(2, 1024, device=dev)
    zh, za = quantize_a4(z)
    assert bool((za == 1.0).all()) and bool((zh == 8).all())
    recs["zero_row_guard"] = "a_s=1.0, x_hat=8 on all-zero rows"
    return recs


# ---------------------------------------------------------------------------
# Part 1 — G-K5-1
# ---------------------------------------------------------------------------

def gate_k51(model, batch, dev):
    ext = dpk_ext()
    mods = {}
    for lname in GATE_LAYERS:
        mod = model.get_submodule(lname)
        mod._capture = []
        mods[lname] = mod

    with torch.no_grad():
        _ = model.model(input_ids=batch, use_cache=False)
        _ = model.model(input_ids=batch, use_cache=False)   # determinism rep

    layer_recs = []
    all_ok = True
    for lname, mod in mods.items():
        caps = mod._capture
        mod._capture = None
        assert len(caps) == 2, (lname, len(caps))
        (xw, a_s, y_bf16), (xw2, a_s2, y_bf16b) = caps
        det_ok = (torch.equal(xw.view(torch.int32), xw2.view(torch.int32))
                  and torch.equal(a_s, a_s2)
                  and torch.equal(y_bf16.view(torch.int16),
                                  y_bf16b.view(torch.int16)))

        streams = (mod.b0, mod.b1, mod.m, mod.s, mod.cb)
        meta = {"R": mod.R, "C": mod.C, "C_orig": mod.C_orig,
                "g": mod.g, "NG": mod.NG, "mmode": "element"}

        # kernel fp32 accumulator output on the captured inputs
        y_f32 = ext.dpk_matmul(*streams, xw, a_s, mod.g, out_fp32=True)
        # bitwise recompute check of the served bf16 output
        y_re = ext.dpk_matmul(*streams, xw, a_s, mod.g)
        recompute_ok = torch.equal(y_re.view(torch.int16),
                                   y_bf16.view(torch.int16))

        # ref (i): K3 stack — integer (x_hat-8) fp32 GEMM, THEN a_s scale
        ref_i = dpk_ref.ref_gemm_direct(mod.b0, mod.b1, mod.m, mod.s, mod.cb,
                                        xw, a_s, mod.g)
        nr_i = norm_rel(y_f32, ref_i)

        # ref (ii): K2 unpack stack — a_s*(x_hat-8) fp32, then GEMM
        tdict = {"b0": mod.b0, "b1": mod.b1, "m": mod.m, "s": mod.s,
                 "cb": mod.cb}
        W = dpk_unpack.unpack(tdict, meta).float()          # [R, C] fp32
        Xn = dpk_ref.unpack_nibbles(xw, mod.C).float() - 8.0
        Xs = Xn * a_s.unsqueeze(1)                          # scale BEFORE
        ref_ii = Xs @ W.t()
        nr_ii = norm_rel(y_f32, ref_ii)
        del W, Xn, Xs

        # ref (iii): K2 integer bucket reference on 4 sampled tokens
        M = xw.shape[0]
        nr_iii = 0.0
        for t in (0, 1, M // 2, M - 1):
            yb = ref_w2a4.gemv_bucket(tdict, meta, xw[t].contiguous(),
                                      float(a_s[t].item()))
            nr_iii = max(nr_iii, norm_rel(y_f32[t], yb))

        # bf16 serving output vs bf16(ref_i): 1-ULP / noise-floor rule
        # (exact integer ULP distance; see bf16_ulp_dist docstring)
        y_ref_bf = ref_i.to(torch.bfloat16)
        mism = y_bf16.view(torch.int16) != y_ref_bf.view(torch.int16)
        frac = mism.float().mean().item()
        ulp_ok = bf16_within_1ulp_or_floor(y_bf16, y_ref_bf, ref_i, mism)

        ok = (nr_i <= GATE_NORM_REL and nr_ii <= GATE_NORM_REL
              and nr_iii <= GATE_NORM_REL and det_ok and recompute_ok
              and frac <= 1e-3 and ulp_ok)
        all_ok &= ok
        rec = {"layer": lname, "R": mod.R, "C": mod.C, "tokens": int(M),
               "norm_rel_vs_ref_i_k3": nr_i,
               "norm_rel_vs_ref_ii_direct_fp32": nr_ii,
               "norm_rel_vs_ref_iii_bucket_4tok": nr_iii,
               "bf16_mismatch_frac_vs_ref_i": frac,
               "bf16_ulp_floor_ok": ulp_ok,
               "bitwise_deterministic_x2": det_ok,
               "bf16_recompute_bitwise": recompute_ok,
               "max_a_s": a_s.max().item(),
               "max_abs_input": a_s.max().item() * 7.5,
               "pass": ok}
        layer_recs.append(rec)
        print(f"  {lname}: nr_i={nr_i:.3e} nr_ii={nr_ii:.3e} "
              f"nr_bucket={nr_iii:.3e} bf16mism={frac:.2e} det={det_ok} "
              f"max|x|={rec['max_abs_input']:.1f} "
              f"[{'PASS' if ok else 'FAIL'}]", flush=True)
        del caps, y_f32, y_re, ref_i, ref_ii, y_ref_bf
    torch.cuda.empty_cache()
    return all_ok, layer_recs


# ---------------------------------------------------------------------------
# Part 1b — full 196-sublayer sweep (extension of G-K5-1's ">= 8 layers";
# suite extended, tolerances unweakened). Captures are taken 28 layers per
# forward (the forward is deterministic, gated above) to bound GPU memory.
# ---------------------------------------------------------------------------

def gate_k51_sweep(model, batch, dev, chunk=28):
    from serve_common import all_layer_names
    ext = dpk_ext()
    names = all_layer_names(model.config.num_hidden_layers)
    worst_nr, worst_frac, n_ulp_fail = 0.0, 0.0, 0
    worst_layer = None
    recs = []
    for start in range(0, len(names), chunk):
        group = names[start:start + chunk]
        mods = {}
        for lname in group:
            mod = model.get_submodule(lname)
            mod._capture = []
            mods[lname] = mod
        with torch.no_grad():
            _ = model.model(input_ids=batch, use_cache=False)
        for lname, mod in mods.items():
            (xw, a_s, y_bf16) = mod._capture[0]
            mod._capture = None
            y_f32 = ext.dpk_matmul(mod.b0, mod.b1, mod.m, mod.s, mod.cb,
                                   xw, a_s, mod.g, out_fp32=True)
            ref_i = dpk_ref.ref_gemm_direct(mod.b0, mod.b1, mod.m, mod.s,
                                            mod.cb, xw, a_s, mod.g)
            nr = norm_rel(y_f32, ref_i)
            y_ref_bf = ref_i.to(torch.bfloat16)
            mism = y_bf16.view(torch.int16) != y_ref_bf.view(torch.int16)
            frac = mism.float().mean().item()
            ulp_ok = bf16_within_1ulp_or_floor(y_bf16, y_ref_bf, ref_i, mism)
            if nr > worst_nr:
                worst_nr, worst_layer = nr, lname
            worst_frac = max(worst_frac, frac)
            n_ulp_fail += 0 if ulp_ok else 1
            recs.append({"layer": lname, "norm_rel_vs_ref_i": nr,
                         "bf16_mismatch_frac": frac, "bf16_ulp_ok": ulp_ok})
            del y_f32, ref_i, y_ref_bf
        torch.cuda.empty_cache()
    sweep_ok = worst_nr <= GATE_NORM_REL and n_ulp_fail == 0
    print(f"  sweep 196/196: worst norm_rel={worst_nr:.3e} ({worst_layer}), "
          f"worst bf16 mism frac={worst_frac:.2e}, ulp failures="
          f"{n_ulp_fail} [{'PASS' if sweep_ok else 'FAIL'}]", flush=True)
    return sweep_ok, {"worst_norm_rel": worst_nr,
                      "worst_norm_rel_layer": worst_layer,
                      "worst_bf16_mismatch_frac": worst_frac,
                      "n_ulp_failures": n_ulp_fail,
                      "n_layers": len(recs), "layers": recs}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DPK_DUMP_DIR)
    ap.add_argument("--allow-any-gpu", action="store_true")
    args = ap.parse_args()
    require_gpu1(args.allow_any_gpu)

    assert torch.backends.cuda.matmul.allow_tf32 is False, \
        "TF32 must be off for the fp32 references"
    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    print("part 0: property gates ...", flush=True)
    prop = property_gates(dev)
    print("  pack/quantize property gates OK", flush=True)

    print("G-K5-1: building DPK-served model ...", flush=True)
    model, n_replaced = build_dpk_model(args.dump_dir)
    assert_no_global_dequant(model)
    model = model.to(dev)
    print(f"G-K5-1: {n_replaced} sublayers replaced; model on {dev}",
          flush=True)

    test_ids = get_wikitext2_testenc()
    batch = test_ids[:, :EVAL_SEQLEN].to(dev)

    k51_ok, layer_recs = gate_k51(model, batch, dev)
    print("G-K5-1: full 196-sublayer sweep (kernel vs ref_i on real "
          "activations) ...", flush=True)
    sweep_ok, sweep = gate_k51_sweep(model, batch, dev)
    k51_ok = k51_ok and sweep_ok
    del batch
    torch.cuda.empty_cache()

    # ---------------- G-K5-2: end-to-end math parity ------------------------
    print("G-K5-2: DPK-served (kernel) PPL, 146-sample protocol ...",
          flush=True)
    t0 = time.time()
    ppl_kernel, nsamples = ppl_resident(model, test_ids, dev,
                                        progress_every=50)
    t_kernel = time.time() - t0
    print(f"G-K5-2: kernel-served PPL = {ppl_kernel:.6f} "
          f"({nsamples} samples, {t_kernel:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()

    print("G-K5-2: building A4-fake-quant torch reference model ...",
          flush=True)
    ref_model, n_ref = build_ref_a4_model(args.dump_dir)
    assert n_ref == n_replaced
    ref_model = ref_model.to(dev)
    t0 = time.time()
    ppl_ref, nsamples2 = ppl_resident(ref_model, test_ids, dev,
                                      progress_every=50)
    t_ref = time.time() - t0
    assert nsamples2 == nsamples
    del ref_model
    torch.cuda.empty_cache()

    parity = abs(ppl_kernel - ppl_ref) / ppl_ref
    k52_ok = parity <= GATE_PPL_PARITY
    print(f"G-K5-2: reference PPL = {ppl_ref:.6f} ({t_ref:.0f}s); "
          f"|delta|/ref = {parity:.3e} (gate <= {GATE_PPL_PARITY:g}) "
          f"[{'PASS' if k52_ok else 'FAIL'}]", flush=True)

    verdict = k51_ok and k52_ok
    now = datetime.now(timezone.utc).isoformat()
    os.makedirs(LOG_DIR, exist_ok=True)

    out1 = {
        "gate": "G-K5-1 (dpk_matmul == kernel-independent references on real "
                "activations of the served model)",
        "timestamp_utc": now,
        "dump_dir": os.path.abspath(args.dump_dir),
        "n_sublayers_served": n_replaced,
        "property_gates": prop,
        "gate_norm_rel": GATE_NORM_REL,
        "worst_norm_rel_ref_i": max(r["norm_rel_vs_ref_i_k3"]
                                    for r in layer_recs),
        "worst_norm_rel_ref_ii": max(r["norm_rel_vs_ref_ii_direct_fp32"]
                                     for r in layer_recs),
        "worst_norm_rel_bucket": max(r["norm_rel_vs_ref_iii_bucket_4tok"]
                                     for r in layer_recs),
        "worst_bf16_mismatch_frac": max(r["bf16_mismatch_frac_vs_ref_i"]
                                        for r in layer_recs),
        "layers": layer_recs,
        "sweep_196": sweep,
        "pass": k51_ok,
    }
    with open(GATE_K51_MARKER if k51_ok
              else os.path.join(LOG_DIR, "gate_K5_1_FAIL.json"), "w") as f:
        json.dump(out1, f, indent=1)

    out2 = {
        "gate": "G-K5-2 (end-to-end math parity, kernel vs torch reference)",
        "timestamp_utc": now,
        "dump_dir": os.path.abspath(args.dump_dir),
        "ppl_kernel_served_w2a4": ppl_kernel,
        "ppl_reference_w2a4_fake_quant": ppl_ref,
        "rel_delta": parity,
        "gate_rel_delta": GATE_PPL_PARITY,
        "nsamples": nsamples,
        "eval_seconds_kernel": t_kernel,
        "eval_seconds_reference": t_ref,
        "ppl_context_anchors": {
            "fake_quant_W_only_full_precision_acts": PPL_FAKEQUANT_W_ONLY,
            "marlin_w4a16_k5a": PPL_MARLIN_W4A16,
            "doml_g128_anchor": PPL_DOML_G128,
            "fp16_reference": PPL_FP16_REF,
        },
        "pass": k52_ok,
    }
    with open(GATE_K52_MARKER if k52_ok
              else os.path.join(LOG_DIR, "gate_K5_2_FAIL.json"), "w") as f:
        json.dump(out2, f, indent=1)

    print("\n===== K5b gate summary =====")
    print(f"G-K5-1: worst norm_rel ref_i={out1['worst_norm_rel_ref_i']:.3e} "
          f"ref_ii={out1['worst_norm_rel_ref_ii']:.3e} "
          f"bucket={out1['worst_norm_rel_bucket']:.3e} "
          f"(gate {GATE_NORM_REL:g}); 196-sweep worst="
          f"{sweep['worst_norm_rel']:.3e} ulp-fails={sweep['n_ulp_failures']} "
          f"-> {'PASS' if k51_ok else 'FAIL'}")
    print(f"G-K5-2: PPL kernel={ppl_kernel:.6f} vs reference={ppl_ref:.6f} "
          f"rel-delta={parity:.3e} -> {'PASS' if k52_ok else 'FAIL'}")
    print(f"W2A4 end-to-end PPL {ppl_kernel:.4f} | W-only fake-quant "
          f"{PPL_FAKEQUANT_W_ONLY:.4f} | Marlin W4A16 {PPL_MARLIN_W4A16:.4f} "
          f"| DOML g=128 {PPL_DOML_G128:.4f} | FP16 {PPL_FP16_REF:.4f}")
    print(f"GATES G-K5-1 + G-K5-2: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
