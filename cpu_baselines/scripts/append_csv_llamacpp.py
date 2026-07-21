#!/usr/bin/env python3
"""Append llama.cpp Phase-2 results to cpu_baselines/results/cpu_baseline_results.csv.

Parses (never hand-typed):
  * llamacpp_bench_<TAG>.json          -> one row per (quant, threads): pp512+tg128
  * llamacpp_bench_<TAG>_numa0.json    -> one row per quant at t=12, numa=node0-pinned
  * llamacpp_ppl_<TAG>.log             -> one row per quant: Final estimate PPL
  * llamacpp_quantize_<TAG>.log        -> bpw (overall from llama-quantize print)
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
GGUF_DIR = os.path.join(ROOT, "downloads/cpu_baselines/llama.cpp")
CSV_PATH = os.path.join(ROOT, "cpu_baselines/results/cpu_baseline_results.csv")

COMMIT = subprocess.run(
    ["git", "-C", os.path.join(ROOT, "temp/llama.cpp"), "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()

MODEL = "Qwen3-0.6B"
TAGS = ["F16", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K", "IQ2_M", "IQ2_XS", "IQ2_XXS", "IQ1_S"]
IMATRIX_TAGS = {"Q3_K_M", "Q2_K", "IQ2_M", "IQ2_XS", "IQ2_XXS", "IQ1_S"}


def gguf_path(tag):
    return os.path.join(GGUF_DIR, f"qwen3-0.6b-{'f16' if tag == 'F16' else tag}.gguf")


def parse_bpw(tag):
    """Overall bpw printed by llama-quantize (main: ... quantized to X bpw
    or the 'size = ... MB -> ... MB | bpw' style summary)."""
    if tag == "F16":
        log = os.path.join(LOG_DIR, "llamacpp_convert.log")
    else:
        log = os.path.join(LOG_DIR, f"llamacpp_quantize_{tag}.log")
    text = open(log, encoding="utf-8", errors="replace").read()
    # llama-quantize prints: "quant size  =   456.11 MiB (5.09 BPW)"
    m = re.findall(r"quant size\s*=\s*[0-9.]+ MiB \(([0-9.]+) BPW\)", text)
    if not m:
        raise SystemExit(f"no bpw found in {log}")
    return float(m[-1])


def parse_ppl(tag):
    log = os.path.join(LOG_DIR, f"llamacpp_ppl_{tag}.log")
    text = open(log, encoding="utf-8", errors="replace").read()
    m = re.findall(r"Final estimate: PPL = ([0-9.]+)", text)
    if not m:
        raise SystemExit(f"no final PPL in {log}")
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
        size = os.path.getsize(gguf_path(tag))
        bpw = parse_bpw(tag)
        note_im = "imatrix(wt2-train-100k)" if tag in IMATRIX_TAGS else "no-imatrix"

        # PPL row (threads=24 used for the run, but it's a quality row)
        ppl = parse_ppl(tag)
        rows.append([now, "llama.cpp", COMMIT, MODEL, tag, 24, "none", "", "",
                     size, bpw, ppl, anchor_ppl, "llama-perplexity",
                     f"wt2-raw full test; ctx2048; {note_im}"])

        # speed rows: thread sweep, no pinning
        jpath = os.path.join(LOG_DIR, f"llamacpp_bench_{tag}.json")
        for t, d in sorted(bench_rows(jpath).items()):
            rows.append([now, "llama.cpp", COMMIT, MODEL, tag, t, "none",
                         round(d.get("pp", float("nan")), 2),
                         round(d.get("tg", float("nan")), 2),
                         size, bpw, "", "", "llama-bench",
                         f"p512 n128 r3; {note_im}"])

        # NUMA point t=12: cpubind to node0 only (set_mempolicy EPERM in this
        # container, so --membind was impossible; default first-touch policy)
        npath = os.path.join(LOG_DIR, f"llamacpp_bench_{tag}_numa0.json")
        for t, d in sorted(bench_rows(npath).items()):
            rows.append([now, "llama.cpp", COMMIT, MODEL, tag, t, "node0-cpubind",
                         round(d.get("pp", float("nan")), 2),
                         round(d.get("tg", float("nan")), 2),
                         size, bpw, "", "", "llama-bench",
                         f"p512 n128 r3; numactl --cpunodebind=0 (membind EPERM in container); {note_im}"])

    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"appended {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
