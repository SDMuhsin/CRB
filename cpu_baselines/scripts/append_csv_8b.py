#!/usr/bin/env python3
"""Append Phase-6 Qwen3-8B results (both frameworks) to
cpu_baselines/results/cpu_baseline_results.csv.

Parses (never hand-typed):
  * {llamacpp8b,ikllamacpp8b}_bench_<TAG>{,_numa0,_rtr}.json
        -> one row per (quant, threads); bpw computed as
           model_size*8/model_n_params from the bench JSON itself.
  * llamacpp8b_ppl_<TAG>.log / ikllamacpp8b_ppl_<TAG>.log
        -> one PPL row per quant ("Final estimate: PPL ..." tail).
  * bpw preference: mainline quantize-log "quant size = ... (X BPW)" or ik
    ppl-log "model size = ... (X BPW)" when available, else bench-JSON value.
  * GGUF file sizes from the filesystem.
Anchor: mainline BF16 PPL is used for BOTH frameworks' ratios (the two
harnesses' 0.6B anchors agreed to 4 sig figs: 18.4881 vs 18.4883); ik PPL
rows carry a note saying so.

IDEMPOTENT: rows whose (framework,model,quant,threads,numa,tool,notes) tuple
already exists in the CSV are skipped, so re-running after late PPL
completions only appends the new rows.
"""
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(ROOT, "llmdocs/cpu_kernel/verify/cpu_baseline_logs")
ML_GGUF_DIR = os.path.join(ROOT, "downloads/cpu_baselines/llama.cpp")
IK_GGUF_DIR = os.path.join(ROOT, "downloads/cpu_baselines/ik_llama.cpp")
CSV_PATH = os.path.join(ROOT, "cpu_baselines/results/cpu_baseline_results.csv")

MODEL = "Qwen3-8B"

def commit(repo):
    return subprocess.run(["git", "-C", os.path.join(ROOT, repo),
                           "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()

ML_COMMIT = commit("temp/llama.cpp")
IK_COMMIT = commit("temp/ik_llama.cpp")

# (framework, tag, imatrix-note)
ML_TAGS = [("BF16", "no-imatrix"), ("Q4_K_M", "no-imatrix"),
           ("Q2_K", "imatrix(wt2-train-100k)"), ("IQ2_XS", "imatrix(wt2-train-100k)")]
IK_TAGS = [("BF16", "no-imatrix"), ("Q4_K_M", "no-imatrix"),
           ("IQ2_KL", "imatrix(ik,wt2-train-100k)"), ("IQ2_KT", "imatrix(ik,wt2-train-100k)")]

IK_ANCHOR_NOTE = ("anchor=mainline-bf16 (ik own 0.6B anchor agreed 18.4883 vs "
                  "18.4881; ik 8B anchor skipped)")


def gguf_path(fw, tag):
    if tag == "BF16":
        return os.path.join(ML_GGUF_DIR, "qwen3-8b-bf16.gguf")
    d = ML_GGUF_DIR if fw == "llama.cpp" else IK_GGUF_DIR
    return os.path.join(d, f"qwen3-8b-{tag}.gguf")


def stem(fw, kind, tag):
    p = "llamacpp8b" if fw == "llama.cpp" else "ikllamacpp8b"
    return os.path.join(LOG_DIR, f"{p}_{kind}_{tag}")


def read(path):
    return open(path, encoding="utf-8", errors="replace").read()


def parse_ppl(fw, tag):
    """Return final PPL float or None if the run hasn't finished."""
    log = stem(fw, "ppl", tag) + ".log"
    if not os.path.exists(log):
        return None
    m = re.findall(r"Final estimate: PPL[^=]*= ([0-9.]+)", read(log))
    return float(m[-1]) if m else None


def parse_bpw(fw, tag):
    """Preferred textual BPW: mainline quantize log / ik ppl-log load line."""
    if fw == "llama.cpp" and tag not in ("BF16",):
        log = stem(fw, "quantize", tag) + ".log"
        if os.path.exists(log):
            m = re.findall(r"quant size\s*=\s*[0-9.]+ [MG]iB \(([0-9.]+) BPW\)", read(log))
            if m:
                return float(m[-1])
    if fw == "ik_llama.cpp":
        log = stem(fw, "ppl", tag) + ".log"
        if os.path.exists(log):
            m = re.findall(r"model size\s*=\s*[0-9.]+ [MG]iB \(([0-9.]+) BPW\)", read(log))
            if m:
                return float(m[-1])
    return None  # fall back to bench-JSON computation


def bench_points(path):
    """-> (by_threads dict, bpw_from_json) ; pp/tg paired by n_threads."""
    recs = json.load(open(path))
    by_threads, bpw = {}, None
    for r in recs:
        t = r["n_threads"]
        d = by_threads.setdefault(t, {})
        if r["n_prompt"] > 0 and r["n_gen"] == 0:
            d["pp"] = r["avg_ts"]
        elif r["n_gen"] > 0 and r["n_prompt"] == 0:
            d["tg"] = r["avg_ts"]
        if r.get("model_size") and r.get("model_n_params"):
            bpw = round(r["model_size"] * 8.0 / r["model_n_params"], 3)
    return by_threads, bpw


def existing_keys():
    keys = set()
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["framework"], row["model"], row["quant"],
                      row["threads"], row["numa"], row["tool"], row["notes"]))
    return keys


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    seen = existing_keys()
    anchor = parse_ppl("llama.cpp", "BF16")
    rows = []

    def add(fw, com, tag, threads, numa, pp, tg, size, bpw, ppl, anch, tool, notes):
        key = (fw, MODEL, tag, str(threads), numa, tool, notes)
        if key in seen:
            return
        seen.add(key)
        rows.append([now, fw, com, MODEL, tag, threads, numa, pp, tg,
                     size, bpw, ppl, anch, tool, notes])

    for fw, com, tags in (("llama.cpp", ML_COMMIT, ML_TAGS),
                          ("ik_llama.cpp", IK_COMMIT, IK_TAGS)):
        for tag, note_im in tags:
            gp = gguf_path(fw, tag)
            if not os.path.exists(gp):
                print(f"skip {fw} {tag}: gguf missing")
                continue
            size = os.path.getsize(gp)
            bpw_txt = parse_bpw(fw, tag)
            bpw_json = None

            # speed rows
            for suffix, numa, extra in (("", "none", ""),
                                        ("_numa0", "node0-cpubind",
                                         "numactl --cpunodebind=0 (membind EPERM in container); "),
                                        ("_rtr", "none", "rtr; -rtr 1 run-time repack; ")):
                jpath = stem(fw, "bench", tag) + suffix + ".json"
                if not os.path.exists(jpath) or os.path.getsize(jpath) == 0:
                    continue
                pts, bj = bench_points(jpath)
                bpw_json = bpw_json or bj
                for t, d in sorted(pts.items()):
                    extra2 = extra
                    if suffix == "_rtr":
                        extra2 = "rtr; p512 n128 r3; -rtr 1 run-time repack; " + note_im
                    else:
                        extra2 = f"{extra}p512 n128 r3; {note_im}"
                    add(fw, com, tag, t, numa,
                        round(d.get("pp", float("nan")), 2),
                        round(d.get("tg", float("nan")), 2),
                        size, bpw_txt or bpw_json or "", "", "", "llama-bench", extra2)

            # PPL row
            ppl = parse_ppl(fw, tag)
            if ppl is not None:
                note = f"wt2-raw full test; ctx2048; {note_im}"
                if fw == "ik_llama.cpp":
                    note += "; " + IK_ANCHOR_NOTE
                add(fw, com, tag, 24, "none", "", "", size,
                    bpw_txt or bpw_json or "", ppl, anchor if anchor else "",
                    "llama-perplexity", note)
            else:
                print(f"note: {fw} {tag} PPL not finished yet (no row)")

    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"appended {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
