#!/usr/bin/env python3
"""Append ik_llama.cpp Phase-4 results to cpu_baselines/results/cpu_baseline_results.csv.

Parses (never hand-typed):
  * ikllamacpp_bench_<TAG>.json        -> one row per (quant, threads): pp512+tg128
  * ikllamacpp_bench_<TAG>_numa0.json  -> one row per quant at t=12, numa=node0-cpubind
  * ikllamacpp_bench_<TAG>_rtr.json    -> one row per quant at t=24, notes=rtr (run-time repack)
  * ikllamacpp_ppl_<TAG>.log           -> one row per quant: "Final estimate: PPL over N
                                          chunks for n_ctx=2048 = X" + bpw from the
                                          model-load line "model size = ... (X.XXX BPW)"
                                          (ik's quantize does not print a whole-file BPW)
  * GGUF file sizes from the filesystem.
Schema: timestamp,framework,commit,model,quant,threads,numa,pp512_tps,tg128_tps,
        file_size_bytes,bpw,ppl_wt2_c2048,ppl_fp16_anchor,tool,notes
Idempotence: this script APPENDS; do not run twice without de-duping.
"""
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(ROOT, "llmdocs/cpu_kernel/verify/cpu_baseline_logs")
GGUF_DIR = os.path.join(ROOT, "downloads/cpu_baselines/ik_llama.cpp")
ML_GGUF_DIR = os.path.join(ROOT, "downloads/cpu_baselines/llama.cpp")
CSV_PATH = os.path.join(ROOT, "cpu_baselines/results/cpu_baseline_results.csv")

COMMIT = subprocess.run(
    ["git", "-C", os.path.join(ROOT, "temp/ik_llama.cpp"), "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()

MODEL = "Qwen3-0.6B"
TAGS = ["F16", "Q4_K_M", "Q4_0", "IQ4_KS", "IQ3_K", "Q2_K", "IQ2_KL",
        "IQ2_K", "IQ2_XS", "IQ2_KS", "IQ2_KT", "IQ2_XXS"]
IMATRIX_TAGS = {"Q2_K", "IQ2_XS", "IQ2_XXS", "IQ3_K", "IQ2_K", "IQ2_KS",
                "IQ2_KL", "IQ2_KT"}


def gguf_path(tag):
    if tag == "F16":
        return os.path.join(ML_GGUF_DIR, "qwen3-0.6b-f16.gguf")
    return os.path.join(GGUF_DIR, f"qwen3-0.6b-{tag}.gguf")


def ppl_log(tag):
    return os.path.join(LOG_DIR, f"ikllamacpp_ppl_{tag}.log")


def parse_bpw(tag):
    """Model-load BPW print: 'model size = X MiB (Y.YYY BPW)' (tensor bytes*8/elems,
    same definition as mainline llama-quantize's whole-file BPW)."""
    text = open(ppl_log(tag), encoding="utf-8", errors="replace").read()
    m = re.findall(r"model size\s*=\s*[0-9.]+ [MG]iB \(([0-9.]+) BPW\)", text)
    if not m:
        raise SystemExit(f"no BPW in {ppl_log(tag)}")
    return float(m[-1])


def parse_ppl(tag):
    text = open(ppl_log(tag), encoding="utf-8", errors="replace").read()
    m = re.findall(r"Final estimate: PPL over \d+ chunks for n_ctx=\d+ = ([0-9.]+)", text)
    if not m:
        raise SystemExit(f"no final PPL in {ppl_log(tag)}")
    return float(m[-1])


def bench_rows(path):
    """llama-bench -o json: list of records; pair pp512/tg128 by n_threads."""
    recs = json.load(open(path))
    by_threads = {}
    for r in recs:
        t = r["n_threads"]
        d = by_threads.setdefault(t, {})
        if r["n_prompt"] > 0 and r["n_gen"] == 0:
            d["pp"] = r["avg_ts"]
        elif r["n_gen"] > 0 and r["n_prompt"] == 0:
            d["tg"] = r["avg_ts"]
    return by_threads


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    anchor_ppl = parse_ppl("F16")
    rows = []

    for tag in TAGS:
        if not os.path.exists(gguf_path(tag)) or not os.path.exists(ppl_log(tag)):
            print(f"skip {tag}: missing gguf or ppl log")
            continue
        size = os.path.getsize(gguf_path(tag))
        bpw = parse_bpw(tag)
        note_im = "imatrix(ik,wt2-train-100k)" if tag in IMATRIX_TAGS else "no-imatrix"

        # PPL row
        ppl = parse_ppl(tag)
        rows.append([now, "ik_llama.cpp", COMMIT, MODEL, tag, 24, "none", "", "",
                     size, bpw, ppl, anchor_ppl, "llama-perplexity",
                     f"wt2-raw full test; ctx2048; {note_im}"])

        # speed rows: thread sweep, no pinning
        jpath = os.path.join(LOG_DIR, f"ikllamacpp_bench_{tag}.json")
        if os.path.exists(jpath):
            for t, d in sorted(bench_rows(jpath).items()):
                rows.append([now, "ik_llama.cpp", COMMIT, MODEL, tag, t, "none",
                             round(d.get("pp", float("nan")), 2),
                             round(d.get("tg", float("nan")), 2),
                             size, bpw, "", "", "llama-bench",
                             f"p512 n128 r3; {note_im}"])

        # NUMA point t=12: cpubind node0 only (set_mempolicy EPERM in container)
        npath = os.path.join(LOG_DIR, f"ikllamacpp_bench_{tag}_numa0.json")
        if os.path.exists(npath):
            for t, d in sorted(bench_rows(npath).items()):
                rows.append([now, "ik_llama.cpp", COMMIT, MODEL, tag, t, "node0-cpubind",
                             round(d.get("pp", float("nan")), 2),
                             round(d.get("tg", float("nan")), 2),
                             size, bpw, "", "", "llama-bench",
                             f"p512 n128 r3; numactl --cpunodebind=0 (membind EPERM in container); {note_im}"])

        # run-time repack point t=24 (-rtr 1): ik-specific feature
        rpath = os.path.join(LOG_DIR, f"ikllamacpp_bench_{tag}_rtr.json")
        if os.path.exists(rpath):
            for t, d in sorted(bench_rows(rpath).items()):
                rows.append([now, "ik_llama.cpp", COMMIT, MODEL, tag, t, "none",
                             round(d.get("pp", float("nan")), 2),
                             round(d.get("tg", float("nan")), 2),
                             size, bpw, "", "", "llama-bench",
                             f"rtr; p512 n128 r3; -rtr 1 run-time repack; {note_im}"])

    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"appended {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
