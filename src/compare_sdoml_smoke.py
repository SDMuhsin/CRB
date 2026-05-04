#!/usr/bin/env python3
"""S6 comparison harness — SDOML smoke pivot + go/no-go diagnosis.

Reads:
  - /workspace/BiLLM2/results/sdoml_smoke.csv  (4 fresh S6 runs)
  - /workspace/BiLLM2/results/qwen3_06b_ptq_benchmark.csv  (DOML reference rows)

Builds a method × (dataset, metric) pivot and prints:
  - The pivot table.
  - Headline ratios (SDOML s=0.5 vs DOML, vs magfit, vs SDOML-1pass).
  - 5-criterion pass/fail scorecard.

Pass criteria (all must be True for S6 GREEN):
  C-headline:    PPL(SDOML s=0.5, wikitext2) <= 1.10 * PPL(DOML, wikitext2)
  C-alternation: PPL(SDOML s=0.5, wikitext2) < PPL(SDOML-1pass s=0.5, wikitext2)
  C-joint:       PPL(SDOML s=0.5, wikitext2) < PPL(magfit-s50, wikitext2)
  C-no-collapse: all SDOML wikitext2 PPLs < 100
  C-no-failed:   no FAILED:* rows in sdoml_smoke.csv

Usage:
  python3 src/compare_sdoml_smoke.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/workspace/BiLLM2")
SMOKE_CSV = REPO / "results" / "sdoml_smoke.csv"
DOML_REF_CSV = REPO / "results" / "qwen3_06b_ptq_benchmark.csv"

# Tags that S6 produces (sdoml at s=0.5 / s=0.2, sdoml-1pass at s=0.5, magfit at s=0.5).
HEADLINE = "sdoml-s50"
ALT_1PASS = "sdoml-s50-1pass"
JOINT_VS = "magfit-s50"
DOML_REF = "doml"
ALL_S6_METHODS = {HEADLINE, "sdoml-s20", ALT_1PASS, JOINT_VS}


def _load_smoke():
    if not SMOKE_CSV.exists():
        sys.exit(f"FATAL: {SMOKE_CSV} does not exist — run the SDOML smoke first.")
    df = pd.read_csv(SMOKE_CSV)
    return df


def _load_doml_ref():
    if not DOML_REF_CSV.exists():
        sys.exit(f"FATAL: {DOML_REF_CSV} does not exist — DOML reference missing.")
    df = pd.read_csv(DOML_REF_CSV)
    # Standard dedupe: keep latest per (model, method, dataset, metric, seed).
    df = df.drop_duplicates(
        subset=["model", "method", "dataset", "metric", "seed"],
        keep="last",
    )
    df = df[(df["model"] == "Qwen/Qwen3-0.6B") & (df["method"] == DOML_REF)
            & (df["seed"] == 0)]
    return df


def _build_pivot(smoke_df, doml_df):
    smoke = smoke_df.copy()
    smoke = smoke.drop_duplicates(
        subset=["model", "method", "dataset", "metric", "seed"],
        keep="last",
    )
    smoke = smoke[(smoke["model"] == "Qwen/Qwen3-0.6B") & (smoke["seed"] == 0)]
    combined = pd.concat([smoke, doml_df], ignore_index=True)

    # 'value' may be numeric or a "FAILED:..." string. Coerce to float for the
    # pivot; FAILED rows become NaN and are tracked separately by the
    # no-failed pass criterion.
    combined["value_num"] = pd.to_numeric(combined["value"], errors="coerce")
    combined["dm"] = combined["dataset"] + "/" + combined["metric"]
    pivot = combined.pivot_table(
        index="method",
        columns="dm",
        values="value_num",
        aggfunc="last",
    )
    return pivot, combined


def _collect_failed(smoke_df):
    if "value" not in smoke_df.columns:
        return []
    rows = smoke_df[smoke_df["value"].astype(str).str.startswith("FAILED")]
    return rows


def main():
    smoke_df = _load_smoke()
    doml_df = _load_doml_ref()

    pivot, combined = _build_pivot(smoke_df, doml_df)

    methods_present = set(pivot.index.tolist())
    print("=" * 78)
    print("S6 SDOML smoke comparison (Qwen3-0.6B, seed=0, wikitext2 calib)")
    print("=" * 78)
    print(f"Smoke CSV:    {SMOKE_CSV}")
    print(f"DOML ref CSV: {DOML_REF_CSV}")
    print(f"Methods in smoke: {sorted(methods_present & ALL_S6_METHODS) or '(none)'}")
    print(f"DOML reference:   {'present' if DOML_REF in methods_present else 'MISSING'}")
    print()

    # Pivot table
    print("--- Pivot table (method × dataset/metric) ---")
    if pivot.empty:
        print("(no rows)")
    else:
        print(pivot.round(4).to_string())
    print()

    # Headline ratios
    def _val(method, dm):
        if method not in pivot.index or dm not in pivot.columns:
            return None
        v = pivot.loc[method, dm]
        return None if pd.isna(v) else float(v)

    h_sd = _val(HEADLINE, "wikitext2/perplexity")
    h_doml = _val(DOML_REF, "wikitext2/perplexity")
    h_alt = _val(ALT_1PASS, "wikitext2/perplexity")
    h_jnt = _val(JOINT_VS, "wikitext2/perplexity")

    print("--- Headline (wikitext2 PPL) ---")
    if h_sd is not None:
        print(f"  SDOML s=0.5         : {h_sd:.4f}")
    if h_doml is not None:
        print(f"  DOML reference      : {h_doml:.4f}")
    if h_alt is not None:
        print(f"  SDOML-1pass s=0.5   : {h_alt:.4f}")
    if h_jnt is not None:
        print(f"  magfit s=0.5        : {h_jnt:.4f}")
    print()

    print("--- Ratios ---")
    if h_sd and h_doml:
        print(f"  SDOML s=0.5 / DOML       : {h_sd/h_doml:.4f}  "
              f"(target <= 1.10)")
    if h_sd and h_alt:
        print(f"  SDOML s=0.5 / 1pass      : {h_sd/h_alt:.4f}  "
              f"(target  < 1.00)")
    if h_sd and h_jnt:
        print(f"  SDOML s=0.5 / magfit     : {h_sd/h_jnt:.4f}  "
              f"(target  < 1.00)")
    print()

    # Pass criteria
    print("--- Pass criteria (S6 GREEN if all PASS) ---")

    crit = []

    # C-headline
    if h_sd is None or h_doml is None:
        crit.append(("C-headline (PPL ≤ 1.10x DOML)", "MISSING"))
    else:
        ok = h_sd <= 1.10 * h_doml
        crit.append((f"C-headline (PPL ≤ 1.10x DOML)  "
                     f"[{h_sd:.4f} vs {1.10*h_doml:.4f}]",
                     "PASS" if ok else "FAIL"))

    # C-alternation
    if h_sd is None or h_alt is None:
        crit.append(("C-alternation (SDOML < 1pass)", "MISSING"))
    else:
        ok = h_sd < h_alt
        crit.append((f"C-alternation (SDOML < 1pass)  "
                     f"[{h_sd:.4f} vs {h_alt:.4f}]",
                     "PASS" if ok else "FAIL"))

    # C-joint
    if h_sd is None or h_jnt is None:
        crit.append(("C-joint (SDOML < magfit)", "MISSING"))
    else:
        ok = h_sd < h_jnt
        crit.append((f"C-joint (SDOML < magfit)       "
                     f"[{h_sd:.4f} vs {h_jnt:.4f}]",
                     "PASS" if ok else "FAIL"))

    # C-no-collapse: all SDOML wikitext2 PPL < 100
    sdoml_rows = combined[combined["method"].astype(str).str.startswith("sdoml-s")
                          & (combined["dataset"] == "wikitext2")
                          & (combined["metric"] == "perplexity")]
    sdoml_ppls = pd.to_numeric(sdoml_rows["value"], errors="coerce").dropna().tolist()
    if not sdoml_ppls:
        crit.append(("C-no-collapse (all SDOML PPL < 100)", "MISSING"))
    else:
        ok = all(p < 100 for p in sdoml_ppls)
        crit.append((f"C-no-collapse (all SDOML PPL < 100)  "
                     f"[max={max(sdoml_ppls):.4f}]",
                     "PASS" if ok else "FAIL"))

    # C-no-failed
    failed = _collect_failed(smoke_df)
    if len(failed) == 0:
        crit.append(("C-no-failed (no FAILED:* rows)", "PASS"))
    else:
        crit.append((f"C-no-failed (no FAILED:* rows)  "
                     f"[{len(failed)} FAILED rows]", "FAIL"))

    for name, status in crit:
        marker = "[+]" if status == "PASS" else ("[-]" if status == "FAIL" else "[?]")
        print(f"  {marker} {name:60s}  {status}")
    print()

    pass_count = sum(1 for _, s in crit if s == "PASS")
    fail_count = sum(1 for _, s in crit if s == "FAIL")
    miss_count = sum(1 for _, s in crit if s == "MISSING")

    print("=" * 78)
    print(f"S6 verdict: {pass_count} PASS, {fail_count} FAIL, "
          f"{miss_count} MISSING (5 total)")
    if fail_count == 0 and miss_count == 0:
        print("STATUS: GREEN — all criteria pass; recommend S7 sweep.")
    else:
        print("STATUS: RED — at least one criterion failed/missing; "
              "review diagnosis before escalating.")
    print("=" * 78)


if __name__ == "__main__":
    main()
