"""ARC-4B RCA — flip/margin analysis over arc_probe.py JSONs.

Compares a reference probe (FP16) against one or more quantized probes:
  * accuracy, net flips (fixed vs broken relative to FP),
  * margin analysis: FP margin (gt score − best-other score) distribution of
    BROKEN questions vs kept questions — tests whether damage is concentrated
    on low-margin questions (generic noise) or hits high-margin questions too
    (systematic direction damage),
  * score-delta stats: per-question gt-choice score shift and margin shift,
  * cross-method overlap of broken sets (do DOML and TQ break the SAME
    questions? — high overlap = question hardness; low overlap = method-
    specific damage).

Usage:
    python kernels/pack/arc4b_flip_report.py \
        --ref downloads/arc4b_rca/probe_fp16.json \
        --cand doml=downloads/arc4b_rca/probe_doml_raw.json \
        --cand tq=downloads/arc4b_rca/probe_tq.json
"""

import argparse
import json

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def margins(rec):
    """(gt_score, best_other_score, margin) for one record."""
    sc = [c["score"] for c in rec["choices"]]
    gt = rec["gt"]
    gt_s = sc[gt]
    other = max(s for j, s in enumerate(sc) if j != gt)
    return gt_s, other, gt_s - other


def pstats(v):
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return "n=0"
    return (f"n={len(v)} mean={v.mean():+.4f} med={np.median(v):+.4f} "
            f"p10={np.percentile(v, 10):+.4f} p90={np.percentile(v, 90):+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", action="append", required=True,
                    help="name=path, repeatable")
    args = ap.parse_args()

    ref = load(args.ref)
    ref_by_i = {r["i"]: r for r in ref["records"]}
    print(f"REF {args.ref}: acc={ref['meta']['accuracy']:.4f} "
          f"({ref['meta']['correct']}/{ref['meta']['total']})")

    broken_sets = {}
    for cand in args.cand:
        name, _, path = cand.partition("=")
        c = load(path)
        by_i = {r["i"]: r for r in c["records"]}
        common = sorted(set(ref_by_i) & set(by_i))
        broken, fixed, kept_ok = [], [], []
        dm_broken, dm_kept, fp_margin_broken, fp_margin_kept = [], [], [], []
        for i in common:
            r, q = ref_by_i[i], by_i[i]
            _, _, m_fp = margins(r)
            _, _, m_q = margins(q)
            if r["correct"] and not q["correct"]:
                broken.append(i)
                fp_margin_broken.append(m_fp)
                dm_broken.append(m_q - m_fp)
            elif not r["correct"] and q["correct"]:
                fixed.append(i)
            elif r["correct"]:
                kept_ok.append(i)
                fp_margin_kept.append(m_fp)
                dm_kept.append(m_q - m_fp)
        broken_sets[name] = set(broken)
        print(f"\n=== {name}: acc={c['meta']['accuracy']:.4f} "
              f"({c['meta']['correct']}/{c['meta']['total']}) ===")
        print(f"  broken (FP right -> wrong): {len(broken)}   "
              f"fixed (FP wrong -> right): {len(fixed)}   "
              f"net {len(fixed) - len(broken):+d}")
        print(f"  FP margin of BROKEN qs: {pstats(fp_margin_broken)}")
        print(f"  FP margin of KEPT   qs: {pstats(fp_margin_kept)}")
        print(f"  margin shift on BROKEN: {pstats(dm_broken)}")
        print(f"  margin shift on KEPT  : {pstats(dm_kept)}")
        # margin shift across ALL common questions
        dm_all = []
        for i in common:
            _, _, m_fp = margins(ref_by_i[i])
            _, _, m_q = margins(by_i[i])
            dm_all.append(m_q - m_fp)
        print(f"  margin shift ALL      : {pstats(dm_all)}")

    names = list(broken_sets)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sa, sb = broken_sets[names[a]], broken_sets[names[b]]
            inter = len(sa & sb)
            uni = len(sa | sb)
            print(f"\nbroken-set overlap {names[a]} vs {names[b]}: "
                  f"|A|={len(sa)} |B|={len(sb)} inter={inter} "
                  f"jaccard={inter/uni if uni else 0:.3f}")


if __name__ == "__main__":
    main()
