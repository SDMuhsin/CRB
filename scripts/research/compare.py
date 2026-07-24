#!/usr/bin/env python3
"""Compare a candidate DOML variant's eval against the DOML baseline and TesseraQ.
Usage: compare.py <model:06b|17b|l1b> <tag>
Reads /scratch/ckp908/crb/logs/rerun/<tag>_eval.csv (+ _bpw.log for honest bpw).
Prints the 7 metrics with: candidate, baseline (delta), TesseraQ (WIN/lose vs the goal).
Goal = beat TesseraQ on ALL 7 (PPL lower better; accuracy higher better)."""
import csv, sys, os, re

LOG = "/scratch/ckp908/crb/logs/rerun"
# metric key -> (csv dataset, csv metric, lower_is_better)
METRICS = [
    ("wt2", "wikitext2", "perplexity", True),
    ("c4",  "c4",        "perplexity", True),
    ("ptb", "ptb",       "perplexity", True),
    ("mmlu","mmlu",      "accuracy",   False),
    ("H",   "hellaswag", "accuracy",   False),
    ("E",   "arc-easy",  "accuracy",   False),
    ("C",   "arc-challenge","accuracy",False),
]
# DOML current baseline (this box, seed 0)
BASE = {
 "06b": dict(wt2=33.00,c4=63.73,ptb=99.88,mmlu=.2722,H=.3331,E=.3763,C=.2440,bpw=2.2309),
 "17b": dict(wt2=27.73,c4=46.37,ptb=57.39,mmlu=.2734,H=.3906,E=.4562,C=.2790,bpw=2.2012),
 "l1b": dict(wt2=25.89,c4=49.37,ptb=55.44,mmlu=.2530,H=.3427,E=.4162,C=.2398,bpw=2.1341),
}
# TesseraQ target (~2.25 bpw)
TQ = {
 "06b": dict(wt2=35.72,c4=74.51,ptb=112.6,mmlu=.274,H=.327,E=.381,C=.260),
 "17b": dict(wt2=24.66,c4=51.89,ptb=65.10,mmlu=.287,H=.409,E=.464,C=.276),
 "l1b": dict(wt2=62.11,c4=120.8,ptb=640.3,mmlu=.244,H=.318,E=.378,C=.219),
}

def load_eval(tag):
    path = f"{LOG}/{tag}_eval.csv"
    if not os.path.exists(path): return None
    vals = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, ds, mt, _ in METRICS:
                if row["dataset"] == ds and row["metric"] == mt:
                    vals[k] = float(row["value"])
    return vals

def load_bpw(tag):
    path = f"{LOG}/{tag}_bpw.log"
    if not os.path.exists(path): return None
    m = re.findall(r"HONEST bpw.*?=\s*([0-9.]+)", open(path).read())
    return float(m[-1]) if m else None

def main():
    model, tag = sys.argv[1], sys.argv[2]
    v = load_eval(tag); bpw = load_bpw(tag)
    if not v: print(f"no eval csv for {tag} yet"); return
    b, t = BASE[model], TQ[model]
    print(f"\n== {tag}  (model={model})  honest bpw={bpw}  (baseline {b['bpw']}) ==")
    print(f"{'metric':6} {'cand':>9} {'baseline':>10} {'d_base':>9} {'TesseraQ':>9} {'vs_TQ':>6}")
    wins = 0; total = 0
    for k, ds, mt, lower in METRICS:
        if k not in v: continue
        total += 1
        cand, base, tq = v[k], b[k], t[k]
        dbase = cand - base
        beat = (cand < tq) if lower else (cand > tq)
        wins += beat
        fmt = "%.2f" if lower else "%.4f"
        print(f"{k:6} {fmt%cand:>9} {fmt%base:>10} {('%+.2f'%dbase) if lower else ('%+.4f'%dbase):>9} "
              f"{fmt%tq:>9} {'WIN' if beat else 'lose':>6}")
    print(f"  -> beats TesseraQ on {wins}/{total} metrics")

if __name__ == "__main__":
    main()
