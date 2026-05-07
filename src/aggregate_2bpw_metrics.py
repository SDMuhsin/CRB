"""
Consolidate 2-bpw-band PTQ benchmark metrics for the NeurIPS 2026 paper.

Produces:
  1) Per-(method, model) aggregates:
     - Language modeling: geometric mean PPL across {wikitext2, c4, ptb}
     - Zero-shot: arithmetic mean accuracy across {mmlu, hellaswag, arc-easy, arc-challenge}
  2) Per-baseline summary across all 7 model points:
     - DOML's gmean-PPL ratio vs baseline (per model)
     - DOML's mean-accuracy delta vs baseline (per model)
     - Aggregate average ratio + max % improvement
     - Tie-or-win count (DOML beats or matches baseline within margin)
  3) Cell-level audit (3 PPL + 4 zero-shot tasks per (model, method))

Margins for tie definition:
  - PPL: |DOML - baseline| / baseline <= 0.01 (1% relative)
  - Accuracy: |DOML - baseline| <= 0.005 (matches existing paper rule)

2-bpw band methods (2.00-2.30 bpw):
  doml (2.05-2.19), tesseraq (2.25), guidedquant (2.01-2.05),
  leanquant_nu (2.01-2.05), gptq-2bit (2.00), rtn-2bit (2.00)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CSVS = {
    "Qwen3-0.6B":   "qwen3_06b_ptq_benchmark.csv",
    "Qwen3-1.7B":   "qwen3_1.7b_ptq_benchmark.csv",
    "Qwen3-4B":     "qwen3_4b_ptq_benchmark.csv",
    "Qwen3-8B":     "qwen3_8b_ptq_benchmark.csv",
    "Llama-3.2-1B": "llama3_1b_ptq_benchmark.csv",
    "Llama-3.2-3B": "llama3_3b_ptq_benchmark.csv",
    "Llama-3.1-8B": "llama3_8b_ptq_benchmark.csv",
}

PPL_TASKS = ["wikitext2", "c4", "ptb"]
ZS_TASKS  = ["mmlu", "hellaswag", "arc-easy", "arc-challenge"]

BAND_METHODS = ["doml", "tesseraq", "guidedquant", "leanquant_nu", "gptq-2bit", "rtn-2bit"]
FP16 = "fp16"

PPL_REL_MARGIN = 0.01
ACC_ABS_MARGIN = 0.005


def load_dedup(model_name, csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(
        subset=["model", "method", "dataset", "metric", "seed"], keep="last"
    )
    if model_name == "Qwen3-0.6B":
        df = df[df.method != "2bit"]  # legacy duplicate of rtn-2bit
    return df


def get_value(df, method, dataset, metric):
    sub = df[(df.method == method) & (df.dataset == dataset) & (df.metric == metric)]
    if len(sub) == 0:
        return float("nan")
    val = sub.value.iloc[-1]
    try:
        v = float(val)
    except (ValueError, TypeError):
        return float("nan")
    if isinstance(val, str) and val.startswith("FAILED"):
        return float("nan")
    return v


def gmean(xs):
    xs = [x for x in xs if math.isfinite(x) and x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def amean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


# ---------- 1) per-(method, model) aggregates ----------
rows = []
for model_name, csv_name in CSVS.items():
    df = load_dedup(model_name, RESULTS_DIR / csv_name)
    for method in BAND_METHODS + [FP16]:
        ppls = [get_value(df, method, t, "perplexity") for t in PPL_TASKS]
        accs = [get_value(df, method, t, "accuracy")   for t in ZS_TASKS]
        rows.append({
            "model": model_name,
            "method": method,
            "wt2": ppls[0], "c4": ppls[1], "ptb": ppls[2],
            "ppl_gmean": gmean(ppls),
            "mmlu": accs[0], "hellaswag": accs[1],
            "arc_e": accs[2], "arc_c": accs[3],
            "acc_amean": amean(accs),
        })

agg = pd.DataFrame(rows)

print("=" * 100)
print("AGGREGATES PER (MODEL, METHOD)  — geometric-mean PPL + arithmetic-mean accuracy")
print("=" * 100)
for model_name in CSVS:
    sub = agg[agg.model == model_name].copy()
    print(f"\n--- {model_name} ---")
    sub_view = sub[["method", "ppl_gmean", "acc_amean",
                    "wt2", "c4", "ptb",
                    "mmlu", "hellaswag", "arc_e", "arc_c"]].copy()
    print(sub_view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------- 2) DOML vs each baseline ----------
print("\n" + "=" * 100)
print("DOML vs each 2-bpw baseline  — per-model ratios + aggregate summary")
print("=" * 100)

baselines = [b for b in BAND_METHODS if b != "doml"]

summary_rows = []
for baseline in baselines:
    print(f"\n>>> DOML vs {baseline}")
    print(f"{'model':<13} {'DOML_ppl':>9} {'base_ppl':>9} {'ratio':>7} {'%impr':>7}  "
          f"{'DOML_acc':>9} {'base_acc':>9} {'delta':>9}")
    ppl_ratios, ppl_imprs = [], []
    acc_deltas = []
    ppl_tie_or_wins, ppl_total = 0, 0
    acc_tie_or_wins, acc_total = 0, 0
    for model_name in CSVS:
        d = agg[(agg.model == model_name) & (agg.method == "doml")].iloc[0]
        b = agg[(agg.model == model_name) & (agg.method == baseline)].iloc[0]
        d_ppl, b_ppl = d.ppl_gmean, b.ppl_gmean
        d_acc, b_acc = d.acc_amean, b.acc_amean

        ratio = d_ppl / b_ppl if (math.isfinite(d_ppl) and math.isfinite(b_ppl) and b_ppl > 0) else float("nan")
        impr  = (1.0 - ratio) * 100 if math.isfinite(ratio) else float("nan")
        delta = (d_acc - b_acc) if (math.isfinite(d_acc) and math.isfinite(b_acc)) else float("nan")

        if math.isfinite(ratio):
            ppl_total += 1
            # tie-or-win: DOML lower OR within margin
            if d_ppl <= b_ppl * (1.0 + PPL_REL_MARGIN):
                ppl_tie_or_wins += 1
            ppl_ratios.append(ratio)
            ppl_imprs.append(impr)
        if math.isfinite(delta):
            acc_total += 1
            if delta >= -ACC_ABS_MARGIN:
                acc_tie_or_wins += 1
            acc_deltas.append(delta)

        print(f"{model_name:<13} "
              f"{d_ppl:>9.3f} {b_ppl:>9.3f} {ratio:>7.3f} {impr:>6.2f}%  "
              f"{d_acc:>9.4f} {b_acc:>9.4f} {delta:>+9.4f}")

    avg_ratio = amean(ppl_ratios) if ppl_ratios else float("nan")
    max_impr  = max(ppl_imprs) if ppl_imprs else float("nan")
    avg_delta = amean(acc_deltas) if acc_deltas else float("nan")
    max_delta = max(acc_deltas) if acc_deltas else float("nan")

    print(f"{'AVG':<13} {'':>9} {'':>9} {avg_ratio:>7.3f} {(1-avg_ratio)*100:>6.2f}%  "
          f"{'':>9} {'':>9} {avg_delta:>+9.4f}")
    print(f"  PPL gmean — DOML ties-or-wins: {ppl_tie_or_wins}/{ppl_total}  "
          f"(avg ratio {avg_ratio:.3f} = {(1-avg_ratio)*100:+.2f}% on avg, max {max_impr:+.2f}%)")
    print(f"  Acc amean — DOML ties-or-wins: {acc_tie_or_wins}/{acc_total}  "
          f"(avg delta {avg_delta:+.4f}, max {max_delta:+.4f})")

    summary_rows.append({
        "baseline": baseline,
        "ppl_avg_ratio": avg_ratio,
        "ppl_avg_impr_pct": (1 - avg_ratio) * 100 if math.isfinite(avg_ratio) else float("nan"),
        "ppl_max_impr_pct": max_impr,
        "ppl_ties_or_wins": f"{ppl_tie_or_wins}/{ppl_total}",
        "acc_avg_delta": avg_delta,
        "acc_max_delta": max_delta,
        "acc_ties_or_wins": f"{acc_tie_or_wins}/{acc_total}",
    })

# ---------- 2b) summary table ----------
print("\n" + "=" * 100)
print("HEADLINE SUMMARY  — DOML vs each 2-bpw scalar baseline (averaged across 7 model points)")
print("=" * 100)
sumdf = pd.DataFrame(summary_rows)
print(sumdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------- 3) cell-level audit ----------
print("\n" + "=" * 100)
print("CELL-LEVEL TIES-OR-WINS  — DOML vs each baseline across all 21 PPL cells + 28 acc cells")
print("=" * 100)
print("(strict-margin ties-or-wins: PPL within 1%, accuracy within 0.005)")
for baseline in baselines:
    ppl_cells, ppl_tow = 0, 0
    acc_cells, acc_tow = 0, 0
    for model_name, csv_name in CSVS.items():
        df = load_dedup(model_name, RESULTS_DIR / csv_name)
        for t in PPL_TASKS:
            d = get_value(df, "doml", t, "perplexity")
            b = get_value(df, baseline, t, "perplexity")
            if math.isfinite(d) and math.isfinite(b) and b > 0:
                ppl_cells += 1
                if d <= b * (1.0 + PPL_REL_MARGIN):
                    ppl_tow += 1
        for t in ZS_TASKS:
            d = get_value(df, "doml", t, "accuracy")
            b = get_value(df, baseline, t, "accuracy")
            if math.isfinite(d) and math.isfinite(b):
                acc_cells += 1
                if (d - b) >= -ACC_ABS_MARGIN:
                    acc_tow += 1
    print(f"  vs {baseline:<14}  PPL: {ppl_tow}/{ppl_cells}  acc: {acc_tow}/{acc_cells}")

# ---------- 4) DOML vs FP16 (sanity / how-close-to-lossless) ----------
print("\n" + "=" * 100)
print("DOML vs FP16  — gap to lossless, per model")
print("=" * 100)
print(f"{'model':<13} {'fp16_ppl':>9} {'doml_ppl':>9} {'ratio':>6}  "
      f"{'fp16_acc':>9} {'doml_acc':>9} {'delta':>9}")
fp_ratios, fp_deltas = [], []
for model_name in CSVS:
    d = agg[(agg.model == model_name) & (agg.method == "doml")].iloc[0]
    f = agg[(agg.model == model_name) & (agg.method == "fp16")].iloc[0]
    r = d.ppl_gmean / f.ppl_gmean
    delta = d.acc_amean - f.acc_amean
    fp_ratios.append(r)
    fp_deltas.append(delta)
    print(f"{model_name:<13} {f.ppl_gmean:>9.3f} {d.ppl_gmean:>9.3f} {r:>6.3f}  "
          f"{f.acc_amean:>9.4f} {d.acc_amean:>9.4f} {delta:>+9.4f}")
print(f"{'AVG':<13} {'':>9} {'':>9} {amean(fp_ratios):>6.3f}  "
      f"{'':>9} {'':>9} {amean(fp_deltas):>+9.4f}")

# Save full aggregate table
out_path = RESULTS_DIR / "aggregate_2bpw_metrics.csv"
agg.to_csv(out_path, index=False)
print(f"\nWrote {out_path}")
