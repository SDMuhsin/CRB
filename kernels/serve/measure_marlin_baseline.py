"""K5a step 3 — MODEL-SCALE memory measurement of the Marlin INT4 baseline.

This produces the Req-1 comparator bar that the DPK-served model (K5b) must
beat: peak GPU memory of Qwen3-0.6B served end-to-end with the patched
Marlin W4A16 kernel in all 196 quantized sublayers (sym INT4 g=128,
G-A/G-B-gated artifacts).

Protocol (llmdocs/cuda_kernel/00_OBJECTIVE_AND_REQUIREMENTS.md: Req 1 is
checked with torch.cuda.max_memory_allocated + nvidia-smi cross-check):

  a. WEIGHTS-RESIDENT: with a clean allocator (assert memory_allocated()==0
     before anything touches the GPU), move the full served model to GPU;
     report memory_allocated and max_memory_allocated, plus a per-component
     split (Marlin B / Marlin s / Marlin workspace / bf16 embeddings+lm_head
     (tied) / norms / other buffers) reconciled against the allocated total.
  b. FORWARD PEAK: one WikiText-2 batch (batch 1, seqlen 2048) full-model
     forward; peak delta + absolute. Repeated 3x after 1 warmup — the three
     peaks MUST be identical (deterministic allocator). Then the FULL PPL
     eval loop (resident, protocol accounting) with its own peak — this also
     yields the served-model PPL headline number.
  c. nvidia-smi cross-check for states (a) and (b) (per-process rows; a
     foreign process may hold memory on the same physical GPU — reported,
     not hidden).
  d. Determinism: the 3 repeated forward peaks are asserted identical.

Also records the theoretical stream bytes (sum over 196 layers of Marlin
B+s buffer bytes, from the G-A ledger) and reconciles them with (a)'s
measured Marlin component.

Output: k5_logs/measure_marlin_baseline.json + stdout summary.
Exit 0 only if the run is complete, gates were consumed (G-A + G-B markers
present), and the determinism assertion holds.

Usage:
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=1 python -u kernels/serve/measure_marlin_baseline.py
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
    DUMP_DIR, EVAL_SEQLEN, GATE_A_MARKER, GATE_B_MARKER, LOG_DIR,
    MarlinServeLinear, build_marlin_model, get_wikitext2_testenc,
    nvidia_smi_gpu_used, nvidia_smi_snapshot, ppl_resident, require_gpu1,
)

import torch  # noqa: E402

OUT_JSON = os.path.join(LOG_DIR, "measure_marlin_baseline.json")
MiB = 1024.0 * 1024.0


def identify_my_row(rows_now, foreign_pids):
    """nvidia-smi reports HOST pids while this container has its own PID
    namespace, so os.getpid() never matches. This process is instead the
    (asserted unique) NEW per-process row relative to the pre-CUDA
    snapshot."""
    new = [r for r in rows_now if r["pid"] not in foreign_pids]
    if len(new) != 1:
        return None, new
    return new[0], new


def component_split(model):
    """Byte accounting of every parameter/buffer on the model, deduplicated
    by storage pointer (tied lm_head/embed counted ONCE, matching the
    allocator's view)."""
    seen = set()
    comp = {"marlin_B": 0, "marlin_s": 0, "marlin_workspace": 0,
            "embed_lmhead_bf16": 0, "norms_bf16": 0, "other": 0}
    counts = dict.fromkeys(comp, 0)
    tied = False

    def _add(key, t):
        ptr = t.untyped_storage().data_ptr()
        if ptr in seen:
            return False
        seen.add(ptr)
        comp[key] += t.nelement() * t.element_size()
        counts[key] += 1
        return True

    for name, mod in model.named_modules():
        if isinstance(mod, MarlinServeLinear):
            _add("marlin_B", mod.B)
            _add("marlin_s", mod.s)
            _add("marlin_workspace", mod.workspace)

    embed_w = model.model.embed_tokens.weight
    lm_w = model.lm_head.weight
    tied = embed_w.untyped_storage().data_ptr() == lm_w.untyped_storage().data_ptr()
    _add("embed_lmhead_bf16", embed_w)
    _add("embed_lmhead_bf16", lm_w)   # no-op if tied

    for name, p in model.named_parameters():
        key = "norms_bf16" if ("norm" in name) else "other"
        _add(key, p)
    for name, b in model.named_buffers():
        if b.device.type != "cuda":
            continue
        _add("other", b)

    comp["_counts"] = counts
    comp["_lm_head_tied_to_embed"] = tied
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DUMP_DIR)
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--skip-gate-check", action="store_true",
                    help="measure even without G-A/G-B markers (numbers are "
                         "then NOT reportable per the accountability protocol)")
    args = ap.parse_args()
    require_gpu1(args.allow_any_gpu)

    gates = {"gate_A_marker": os.path.exists(GATE_A_MARKER),
             "gate_B_marker": os.path.exists(GATE_B_MARKER)}
    if not args.skip_gate_check and not all(gates.values()):
        raise SystemExit(f"correctness gates not passed yet: {gates} — run "
                         "verify_marlin_pack.py and verify_marlin_model.py first")

    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    # ---- theoretical stream bytes from the G-A ledger ----------------------
    with open(GATE_A_MARKER) as f:
        gate_a = json.load(f)
    theoretical_stream_bytes = gate_a["total_weight_bytes_B_plus_s"]
    total_qparams = gate_a["total_quantized_params"]

    # ---- clean-GPU precondition -------------------------------------------
    smi_before, _ = nvidia_smi_snapshot()    # BEFORE CUDA context creation
    smi_gpu_before = nvidia_smi_gpu_used()
    foreign_pids = {r["pid"] for r in smi_before}
    torch.cuda.synchronize(dev)              # init context
    alloc0 = torch.cuda.memory_allocated(dev)
    assert alloc0 == 0, f"allocator not clean at start: {alloc0} B"

    print("building served model on CPU ...", flush=True)
    model, n_replaced = build_marlin_model(args.dump_dir)
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
    comp = component_split(model)
    comp_sum = sum(v for k, v in comp.items() if not k.startswith("_"))
    smi_a_rows, _ = nvidia_smi_snapshot()
    smi_a_mine_row, _new_a = identify_my_row(smi_a_rows, foreign_pids)
    smi_a_mine = smi_a_mine_row["used_MiB"] if smi_a_mine_row else None
    print(f"(a) weights-resident: allocated={wr_alloc:,} B "
          f"({wr_alloc/MiB:.2f} MiB), requested={wr_requested:,} B "
          f"({wr_n_allocs} allocations), load peak={wr_peak:,} B, "
          f"reserved={wr_reserved:,} B, nvidia-smi(this process, by row-diff)="
          f"{smi_a_mine} MiB", flush=True)
    print(f"    components: marlin B={comp['marlin_B']:,} s={comp['marlin_s']:,} "
          f"ws={comp['marlin_workspace']:,} embed/lm_head={comp['embed_lmhead_bf16']:,} "
          f"norms={comp['norms_bf16']:,} other={comp['other']:,} "
          f"(sum {comp_sum:,}; alloc-sum delta {wr_alloc - comp_sum:,} B)",
          flush=True)

    marlin_measured = comp["marlin_B"] + comp["marlin_s"]
    recon_delta = marlin_measured - theoretical_stream_bytes

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

    # ---- (b2) full PPL eval loop peak (+ the headline served PPL) ----------
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

    out_doc = {
        "milestone": "K5a Marlin INT4 model-scale baseline (Req-1 comparator bar)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(dev),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "model": "Qwen/Qwen3-0.6B (bf16 skeleton; 196 sublayers Marlin sym "
                 "INT4 g=128 W4A16; embeddings/lm_head(tied)/norms bf16)",
        "dump_dir": os.path.abspath(args.dump_dir),
        "gates_consumed": gates,
        "n_marlin_sublayers": n_replaced,
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
            "sum_marlin_B_plus_s_from_gate_A": theoretical_stream_bytes,
            "measured_marlin_B_plus_s_component": marlin_measured,
            "delta_bytes": recon_delta,
            "total_quantized_params": total_qparams,
            "aggregate_bpw": gate_a["aggregate_bpw"],
        },
        "nvidia_smi_before_anything": {"per_process": smi_before,
                                       "per_gpu_used": smi_gpu_before},
        "foreign_pids_on_host": sorted(foreign_pids),
        "pass": bool(det_ok),
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out_doc, f, indent=1)

    print("\n===== K5a Marlin baseline summary =====")
    print(f"weights-resident allocated : {wr_alloc:,} B = {wr_alloc/MiB:.2f} MiB")
    print(f"  marlin B+s               : {marlin_measured:,} B = "
          f"{marlin_measured/MiB:.2f} MiB (theory {theoretical_stream_bytes:,}, "
          f"delta {recon_delta:,})")
    print(f"forward peak (1x2048)      : abs {fwd_runs[0]['peak_abs']:,} B = "
          f"{fwd_runs[0]['peak_abs']/MiB:.2f} MiB "
          f"(delta {deltas[0]/MiB:.2f} MiB), 3x identical: {det_ok}")
    print(f"full PPL loop peak         : abs {ppl_peak_abs:,} B = "
          f"{ppl_peak_abs/MiB:.2f} MiB")
    print(f"served PPL (wikitext2)     : {ppl:.6f}")
    print(f"JSON: {OUT_JSON}")
    return 0 if det_ok else 1


if __name__ == "__main__":
    sys.exit(main())
