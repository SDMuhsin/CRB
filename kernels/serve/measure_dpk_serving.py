"""K5b GATE G-K5-3 — MODEL-SCALE Req-1 measurement of the DPK-served model.

Protocol mirrored 1:1 from kernels/serve/measure_marlin_baseline.py (K5a),
per the pre-registered contract in llmdocs/cuda_kernel/03_k5_serving_design.md:

  a. WEIGHTS-RESIDENT: clean allocator asserted (memory_allocated()==0
     before any CUDA work), full served model moved to GPU; memory_allocated
     + max_memory_allocated + per-component split (DPK streams b0/b1/m/s/cb
     summed across the 196 layers / bf16 embed+lm_head (untied, identical to
     K5a) / norms / other), reconciled against the allocator's
     requested_bytes AND against the dpk_verify/manifest theoretical stream
     bytes (delta MUST be 0). The global-dequant ban is verified structurally
     (assert_no_global_dequant) and by this component split: there is no
     [R, C] floating-point weight tensor anywhere in the resident bytes.
  b. FORWARD PEAK: one WikiText-2 batch (batch 1, seqlen 2048), 1 warmup +
     3 measured repeats — peaks MUST be identical (determinism (d)).
  b2. FULL PPL eval loop (146 samples, protocol accounting) with its own
     peak — also yields the W2A4 end-to-end PPL headline (G-K5-4 table).
  c. nvidia-smi cross-check by per-process ROW-DIFF vs the pre-CUDA snapshot
     (container PID namespace != host; a foreign ~1330 MiB process may sit
     on physical GPU 1 — reported, not hidden).

G-K5-3 PASS iff weights-resident AND forward-peak-abs AND PPL-loop-peak are
ALL strictly below the K5a Marlin bar (read live from
k5_logs/measure_marlin_baseline.json), and the determinism assertion holds.
A FAIL is reported exactly as measured, with the component breakdown.

Requires the G-K5-1 + G-K5-2 markers (correctness before any reported
number, doc 00).

Usage:
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=1 python -u kernels/serve/measure_dpk_serving.py
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
    EVAL_SEQLEN, LOG_DIR, get_wikitext2_testenc, nvidia_smi_gpu_used,
    nvidia_smi_snapshot, ppl_resident, require_gpu1,
)
from dpk_serve import (  # noqa: E402
    DPK_DUMP_DIR, GATE_K51_MARKER, GATE_K52_MARKER, PPL_DOML_G128,
    PPL_FAKEQUANT_W_ONLY, PPL_FP16_REF, PPL_MARLIN_W4A16,
    assert_no_global_dequant, build_dpk_model, dpk_component_split,
    manifest_stream_bytes,
)
from measure_marlin_baseline import identify_my_row  # noqa: E402

import torch  # noqa: E402

OUT_JSON = os.path.join(LOG_DIR, "measure_dpk_serving.json")
K5A_JSON = os.path.join(LOG_DIR, "measure_marlin_baseline.json")
MiB = 1024.0 * 1024.0

DPK_STREAM_KEYS = ("dpk_b0", "dpk_b1", "dpk_m", "dpk_s", "dpk_cb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DPK_DUMP_DIR)
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--skip-gate-check", action="store_true",
                    help="measure even without G-K5-1/2 markers (numbers are "
                         "then NOT reportable per the accountability protocol)")
    args = ap.parse_args()
    require_gpu1(args.allow_any_gpu)

    gates = {"gate_K5_1_marker": os.path.exists(GATE_K51_MARKER),
             "gate_K5_2_marker": os.path.exists(GATE_K52_MARKER)}
    if not args.skip_gate_check and not all(gates.values()):
        raise SystemExit(f"correctness gates not passed yet: {gates} — run "
                         "verify_dpk_model.py first")

    # ---- the K5a bar (read from the ledgered measurement, not hardcoded) --
    with open(K5A_JSON) as f:
        k5a = json.load(f)
    bar = {
        "weights_resident": k5a["a_weights_resident"]["memory_allocated_bytes"],
        "forward_peak_abs":
            k5a["b_forward_peak_seqlen2048_batch1"]["peak_abs_bytes"],
        "forward_peak_delta":
            k5a["b_forward_peak_seqlen2048_batch1"]["peak_delta_bytes"],
        "ppl_loop_peak": k5a["b2_full_ppl_loop"]["peak_abs_bytes"],
        "marlin_stream_bytes":
            k5a["theoretical_stream_bytes"]["measured_marlin_B_plus_s_component"],
        "marlin_ppl": k5a["b2_full_ppl_loop"]["ppl_wikitext2_seqlen2048_seed0"],
        "marlin_fwd_ms":
            k5a["b_forward_peak_seqlen2048_batch1"]["forward_ms_median_of_3"],
        "marlin_eval_s": k5a["b2_full_ppl_loop"]["eval_seconds"],
    }

    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    # ---- theoretical stream bytes from the ledgered manifest ---------------
    theo_bytes, total_qparams, agg_bpw = manifest_stream_bytes()

    # ---- clean-GPU precondition -------------------------------------------
    smi_before, _ = nvidia_smi_snapshot()    # BEFORE CUDA context creation
    smi_gpu_before = nvidia_smi_gpu_used()
    foreign_pids = {r["pid"] for r in smi_before}
    torch.cuda.synchronize(dev)              # init context
    alloc0 = torch.cuda.memory_allocated(dev)
    assert alloc0 == 0, f"allocator not clean at start: {alloc0} B"

    print("building DPK-served model on CPU ...", flush=True)
    model, n_replaced = build_dpk_model(args.dump_dir)
    assert_no_global_dequant(model)
    test_ids = get_wikitext2_testenc()       # stays on CPU; slices moved per batch

    # ---- (a) weights-resident ----------------------------------------------
    torch.cuda.reset_peak_memory_stats(dev)
    model = model.to(dev)
    torch.cuda.synchronize(dev)
    wr_alloc = torch.cuda.memory_allocated(dev)
    wr_peak = torch.cuda.max_memory_allocated(dev)
    wr_reserved = torch.cuda.memory_reserved(dev)
    stats = torch.cuda.memory_stats(dev)
    wr_requested = stats.get("requested_bytes.all.current")
    wr_n_allocs = stats.get("allocation.all.current")
    comp = dpk_component_split(model)
    comp_sum = sum(v for k, v in comp.items() if not k.startswith("_"))
    assert_no_global_dequant(model)          # re-check on the GPU copy
    smi_a_rows, _ = nvidia_smi_snapshot()
    smi_a_mine_row, _new_a = identify_my_row(smi_a_rows, foreign_pids)
    smi_a_mine = smi_a_mine_row["used_MiB"] if smi_a_mine_row else None
    dpk_measured = sum(comp[k] for k in DPK_STREAM_KEYS)
    recon_delta = dpk_measured - theo_bytes
    print(f"(a) weights-resident: allocated={wr_alloc:,} B "
          f"({wr_alloc/MiB:.2f} MiB), requested={wr_requested:,} B "
          f"({wr_n_allocs} allocations), load peak={wr_peak:,} B, "
          f"reserved={wr_reserved:,} B, nvidia-smi(this process, by row-diff)="
          f"{smi_a_mine} MiB", flush=True)
    print(f"    components: b0={comp['dpk_b0']:,} b1={comp['dpk_b1']:,} "
          f"m={comp['dpk_m']:,} s={comp['dpk_s']:,} cb={comp['dpk_cb']:,} "
          f"embed/lm_head={comp['embed_lmhead_bf16']:,} "
          f"norms={comp['norms_bf16']:,} other={comp['other']:,} "
          f"(sum {comp_sum:,}; alloc-sum delta {wr_alloc - comp_sum:,} B)",
          flush=True)
    print(f"    DPK streams total {dpk_measured:,} B vs manifest theory "
          f"{theo_bytes:,} B (delta {recon_delta:,}; aggregate "
          f"{agg_bpw:.4f} bpw over {total_qparams:,} params)", flush=True)

    # ---- (b) forward peaks --------------------------------------------------
    batch = test_ids[:, :EVAL_SEQLEN].to(dev)
    after_batch_alloc = torch.cuda.memory_allocated(dev)

    def fwd():
        with torch.no_grad():
            return model(input_ids=batch, use_cache=False)

    # warmup (one-time cuBLAS/cuDNN workspace + autotune allocations)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    out = fwd()
    torch.cuda.synchronize(dev)
    warmup_peak_delta = torch.cuda.max_memory_allocated(dev) - after_batch_alloc
    logits_dtype = str(out.logits.dtype)
    del out
    torch.cuda.synchronize(dev)

    fwd_runs = []
    ev_ms = []
    smi_b_rows = None
    for rep in range(3):
        torch.cuda.synchronize(dev)
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t_s = torch.cuda.Event(enable_timing=True)
        t_e = torch.cuda.Event(enable_timing=True)
        t_s.record()
        out = fwd()
        t_e.record()
        torch.cuda.synchronize(dev)
        peak_abs = torch.cuda.max_memory_allocated(dev)
        smi_rows, smi_mine = None, None
        if rep == 0:
            smi_rows, _ = nvidia_smi_snapshot()   # logits still alive here
            row, _ = identify_my_row(smi_rows, foreign_pids)
            smi_mine = row["used_MiB"] if row else None
        fwd_runs.append({"baseline_alloc": base, "peak_abs": peak_abs,
                         "peak_delta": peak_abs - base,
                         "nvidia_smi_this_process_MiB": smi_mine})
        ev_ms.append(t_s.elapsed_time(t_e))
        if rep == 0:
            smi_b_rows = smi_rows
        del out
        torch.cuda.synchronize(dev)

    deltas = [r["peak_delta"] for r in fwd_runs]
    det_ok = len(set(deltas)) == 1 and len({r["peak_abs"] for r in fwd_runs}) == 1
    print(f"(b) forward peak (batch 1 x {EVAL_SEQLEN}, logits {logits_dtype}): "
          f"delta={deltas[0]:,} B ({deltas[0]/MiB:.2f} MiB), "
          f"abs={fwd_runs[0]['peak_abs']:,} B "
          f"({fwd_runs[0]['peak_abs']/MiB:.2f} MiB); 3 reps deltas={deltas} "
          f"identical={det_ok}; warmup delta={warmup_peak_delta:,} B; "
          f"median fwd {sorted(ev_ms)[1]:.1f} ms", flush=True)
    del batch
    torch.cuda.synchronize(dev)

    # ---- (b2) full PPL eval loop peak (+ the W2A4 headline PPL) ------------
    torch.cuda.synchronize(dev)
    ppl_base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    ppl, nsamples = ppl_resident(model, test_ids, dev, progress_every=100)
    torch.cuda.synchronize(dev)
    ppl_peak_abs = torch.cuda.max_memory_allocated(dev)
    ppl_time = time.time() - t0
    print(f"(b2) full PPL loop: PPL={ppl:.6f} ({nsamples} samples, "
          f"{ppl_time:.0f}s), peak abs={ppl_peak_abs:,} B "
          f"({ppl_peak_abs/MiB:.2f} MiB), delta={ppl_peak_abs - ppl_base:,} B",
          flush=True)

    # ---- Req-1 verdict vs the K5a bar --------------------------------------
    ours = {"weights_resident": wr_alloc,
            "forward_peak_abs": fwd_runs[0]["peak_abs"],
            "ppl_loop_peak": ppl_peak_abs}
    req1 = {}
    for key in ("weights_resident", "forward_peak_abs", "ppl_loop_peak"):
        b = bar[key]
        v = ours[key]
        req1[key] = {"dpk_bytes": v, "marlin_bar_bytes": b,
                     "margin_bytes": b - v,
                     "margin_pct": 100.0 * (b - v) / b,
                     "pass": v < b}
    req1_pass = all(r["pass"] for r in req1.values()) and det_ok

    out_doc = {
        "milestone": "K5b DPK W2A4 model-scale serving (G-K5-3, Req-1 vs "
                     "K5a Marlin bar)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(dev),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "model": "Qwen/Qwen3-0.6B (bf16 skeleton; 196 sublayers DPK W2A4 "
                 "refit-g256 element mmode via dpk_matmul; embeddings/"
                 "lm_head(untied, same as K5a)/norms bf16)",
        "dump_dir": os.path.abspath(args.dump_dir),
        "gates_consumed": gates,
        "n_dpk_sublayers": n_replaced,
        "a_weights_resident": {
            "memory_allocated_bytes": wr_alloc,
            "requested_bytes": wr_requested,
            "n_live_allocations": wr_n_allocs,
            "max_memory_allocated_during_load_bytes": wr_peak,
            "memory_reserved_bytes": wr_reserved,
            "component_split_bytes": {k: v for k, v in comp.items()
                                      if not k.startswith("_")},
            "component_counts": comp["_counts"],
            "lm_head_tied_to_embed": comp["_lm_head_tied_to_embed"],
            "component_sum_bytes": comp_sum,
            "components_equal_requested": comp_sum == wr_requested,
            "allocated_minus_components_bytes": wr_alloc - comp_sum,
            "allocator_rounding_note": "allocated - requested = per-block "
                                       "rounding of the caching allocator",
            "no_global_dequant_check": "assert_no_global_dequant passed on "
                                       "CPU build and on the GPU copy",
            "nvidia_smi_this_process_MiB": smi_a_mine,
            "nvidia_smi_rows": smi_a_rows,
            "pid_namespace_note": "container PID namespace != host; this "
                                  "process identified by row-diff vs the "
                                  "pre-CUDA snapshot",
        },
        "b_forward_peak_seqlen2048_batch1": {
            "logits_dtype": logits_dtype,
            "warmup_peak_delta_bytes": warmup_peak_delta,
            "runs": fwd_runs,
            "peak_delta_bytes": deltas[0],
            "peak_abs_bytes": fwd_runs[0]["peak_abs"],
            "three_runs_identical": det_ok,
            "forward_ms_median_of_3": sorted(ev_ms)[1],
            "nvidia_smi_rows_during_run0": smi_b_rows,
        },
        "b2_full_ppl_loop": {
            "ppl_wikitext2_seqlen2048_seed0": ppl,
            "nsamples": nsamples,
            "peak_abs_bytes": ppl_peak_abs,
            "peak_delta_bytes": ppl_peak_abs - ppl_base,
            "eval_seconds": ppl_time,
        },
        "theoretical_stream_bytes": {
            "sum_dpk_streams_from_manifest": theo_bytes,
            "measured_dpk_stream_component": dpk_measured,
            "delta_bytes": recon_delta,
            "total_quantized_params": total_qparams,
            "aggregate_bpw": agg_bpw,
            "marlin_stream_bytes_k5a": bar["marlin_stream_bytes"],
            "stream_ratio_dpk_over_marlin":
                dpk_measured / bar["marlin_stream_bytes"],
        },
        "req1_vs_k5a_bar": req1,
        "req1_pass": req1_pass,
        "stream_reconciliation_delta_zero": recon_delta == 0,
        "ppl_table_g_k5_4": {
            "w2a4_end_to_end_this_run": ppl,
            "w_only_fake_quant_full_precision_acts": PPL_FAKEQUANT_W_ONLY,
            "marlin_w4a16_k5a": PPL_MARLIN_W4A16,
            "doml_g128_anchor": PPL_DOML_G128,
            "fp16_reference": PPL_FP16_REF,
        },
        "wall_times_context_only": {
            "dpk_forward_ms_median_of_3": sorted(ev_ms)[1],
            "marlin_forward_ms_median_of_3_k5a": bar["marlin_fwd_ms"],
            "dpk_full_eval_seconds": ppl_time,
            "marlin_full_eval_seconds_k5a": bar["marlin_eval_s"],
        },
        "nvidia_smi_before_anything": {"per_process": smi_before,
                                       "per_gpu_used": smi_gpu_before},
        "foreign_pids_on_host": sorted(foreign_pids),
        "pass": bool(req1_pass and recon_delta == 0),
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out_doc, f, indent=1)

    print("\n===== G-K5-3: DPK-served vs K5a Marlin bar (Req-1) =====")
    print(f"{'metric':28s} {'DPK (B)':>16s} {'Marlin bar (B)':>16s} "
          f"{'margin (B)':>14s} {'margin':>8s}  verdict")
    for key, label in (("weights_resident", "weights-resident"),
                       ("forward_peak_abs", "forward peak abs (1x2048)"),
                       ("ppl_loop_peak", "full-PPL-loop peak")):
        r = req1[key]
        print(f"{label:28s} {r['dpk_bytes']:>16,} {r['marlin_bar_bytes']:>16,} "
              f"{r['margin_bytes']:>14,} {r['margin_pct']:>7.2f}%  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    print(f"stream component            {dpk_measured:>16,} "
          f"{bar['marlin_stream_bytes']:>16,} "
          f"{bar['marlin_stream_bytes'] - dpk_measured:>14,} "
          f"{100.0 * (1 - dpk_measured / bar['marlin_stream_bytes']):>7.2f}%  "
          f"(ratio {dpk_measured / bar['marlin_stream_bytes']:.4f})")
    print(f"determinism (3x forward)   : {det_ok}")
    print(f"stream reconciliation delta: {recon_delta:,} B (must be 0)")
    print(f"W2A4 PPL / Marlin PPL      : {ppl:.4f} / {bar['marlin_ppl']:.4f}")
    print(f"median fwd ms (DPK/Marlin) : {sorted(ev_ms)[1]:.1f} / "
          f"{bar['marlin_fwd_ms']:.1f}")
    print(f"eval wall s (DPK/Marlin)   : {ppl_time:.0f} / "
          f"{bar['marlin_eval_s']:.0f}")
    print(f"G-K5-3: {'PASS' if out_doc['pass'] else 'FAIL'}")
    print(f"JSON: {OUT_JSON}")
    return 0 if out_doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
