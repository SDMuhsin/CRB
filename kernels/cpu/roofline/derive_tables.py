#!/usr/bin/env python3
"""Derive decode/prefill ceiling tables from measured roofline CSVs.

Reads results/*.csv (produced by run_all.sh) and prints the ROOFLINE.md
derived tables as Markdown. All GB/s are 1e9 bytes/s; MB are 1e6 bytes.
"""
import csv
import statistics as st
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"

def med_bw(fname, key_idx, key_val, bw_idx, extra=None):
    """median over reps of the BW column for rows matching key."""
    vals = []
    with open(RES / fname) as f:
        for row in csv.reader(f):
            if row[key_idx] == key_val and (extra is None or extra(row)):
                vals.append(float(row[bw_idx]))
    return (st.median(vals), min(vals), max(vals), len(vals)) if vals else None

def stream_bw(mode, threads):
    return med_bw("stream.csv", 1, mode, 7,
                  extra=lambda r: int(r[2]) == threads)

W_LIN = 440_401_920            # Qwen3-0.6B linear weights
HEAD = 151_936 * 1024          # head/embed params
MB = 1e6

def main():
    best24 = stream_bw("b", 24)
    best48 = stream_bw("b", 48)
    best = max(best24[0], best48[0])
    best_label = "24t node-local" if best24[0] >= best48[0] else "48t node-local"

    print(f"# derived from: best node-local read BW = {best:.1f} GB/s ({best_label})")
    print(f"#   24t node-local median {best24[0]:.1f} [{best24[1]:.1f},{best24[2]:.1f}] "
          f"48t {best48[0]:.1f} [{best48[1]:.1f},{best48[2]:.1f}]")

    lin_scen = [("2.2299 bpw (K31 pareto)", 2.2299),
                ("2.46 bpw", 2.46),
                ("3.376 bpw", 3.376)]
    head_scen = [("bf16 head (311 MB)", HEAD * 2),
                 ("fp8 head (156 MB)", HEAD * 1),
                 ("int4-class head (78 MB)", HEAD * 0.5)]
    misc_lo, misc_hi = 10 * MB, 15 * MB
    misc_mid = 12.5 * MB

    print("\n## Decode ceiling table (tg t/s = BW / bytes-per-token)\n")
    print("| linears | head | MB/token (incl 12.5 misc) | tg ceiling @ best BW | tg @ 60% eff |")
    print("|---|---|---|---|---|")
    for lname, bpw in lin_scen:
        lin_b = W_LIN * bpw / 8
        for hname, head_b in head_scen:
            tot = lin_b + head_b + misc_mid
            tg = best * 1e9 / tot
            print(f"| {lname} | {hname} | {tot/MB:.1f} | {tg:.1f} | {tg*0.6:.1f} |")

    # baseline sanity: what BW do the measured baselines imply?
    print("\n## Baseline sanity (implied effective BW = tg x bytes/token)\n")
    q5k_head = HEAD * 5.5 / 8      # q5_K = 5.5 bpw super-block format
    for name, tg, lin_bpw in [("ik IQ2_KL", 136.0, 2.69), ("Q2_K_R4", 159.6, 2.625)]:
        lin_b = W_LIN * lin_bpw / 8
        tot = lin_b + q5k_head + misc_mid
        eff_bw = tg * tot / 1e9
        print(f"{name}: {tot/MB:.1f} MB/token -> implied BW {eff_bw:.1f} GB/s "
              f"= {100*eff_bw/best:.0f}% of best measured ({best:.1f})")

    # prefill
    print("\n## Prefill ceiling\n")
    vals = []
    with open(RES / "vnni.csv") as f:
        for row in csv.reader(f):
            if row[0] == "vnni_gemm" and row[1] == "24":
                vals.append(float(row[8]))
    gmacs = st.median(vals)
    macs_512 = W_LIN * 512
    t = macs_512 / (gmacs * 1e9)
    pp = 512 / t
    print(f"measured 24t GEMM: {gmacs:.0f} GMAC/s (median of {len(vals)} reps, "
          f"min {min(vals):.0f} max {max(vals):.0f})")
    print(f"pp512 linear MACs = {macs_512/1e9:.1f} GMAC -> bound = {pp:.0f} t/s "
          f"(ik baseline 2355 t/s -> {100*2355/pp:.0f}% of this bound)")

if __name__ == "__main__":
    main()
