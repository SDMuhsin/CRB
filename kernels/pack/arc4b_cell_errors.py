"""ARC-4B RCA — matrix-cell-level quantization error analysis.

For each transformer-block Linear sublayer, computes per-cell error maps and
statistics for one or two quantized containers against their correct FP
references, weighted by measured activation statistics from two streams
(wt2 calibration vs ARC-Easy task — see arc4b_actstats.py).

References:
  * DOML/K31 dump: reference = original FP16 weight (HF snapshot); the dump's
    dpk container gives the per-cell class via dpk_unpack.part_matrix
    (0=bulk, 1=tail, 2=salient).
  * TesseraQ dump (--tq-dir): reference = the dump's own wref plane
    (post-AWQ-fold pre-clip fp32). Errors are UNFOLDED back to the original
    coordinate system via the per-column fold scale s_j recovered as the
    row-median of wref/W_fp, so both methods are compared in the same space.
    (AWQ folding: W'[:,j] = W[:,j]*s_j, x'_j = x_j/s_j — function-preserving.)

Per (sublayer, method, stream) this script reports (aggregate CSV):
  * cell counts and raw error energy by class (bulk/tail/salient for DOML;
    'all' for TesseraQ),
  * activation-weighted error energy  v_i = sum_j E_ij^2 * E[x_j^2]
    (diag-H proxy of output-error power per output channel/row),
  * per-row v_i distribution: total, max share, top-1% share, kurtosis,
  * systematic-bias power  b_i = sum_j E_ij * E[x_j]  (rectification axis),
    reported as sum b_i^2 and its ratio to sum v_i,
  * weight-relative NMSE per class.
Per sublayer it saves an npz with the per-row vectors (v_i, b_i, row class
energies) for deeper digs.

CAVEAT (by design): the diag proxy IGNORES cross-column error correlation, so
it systematically overstates GPTQ-compensated errors (DOML) relative to their
true output impact. It is an attribution instrument. Ground-truth output
errors come from the empirical block-forward measurement
(arc4b_output_errors.py); conclusions must be anchored there.

Usage:
    python kernels/pack/arc4b_cell_errors.py \
        --model Qwen/Qwen3-4B \
        --doml-dir downloads/doml_dumps/qwen3-4b/k31-rdsplit-lam3e-4-g256 \
        --tq-dir  downloads/tesseraq_dumps/qwen3-4b-w2g128 \
        --actstats downloads/arc4b_rca/actstats \
        --out downloads/arc4b_rca/cell_errors
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import dpk_unpack  # noqa: E402

CLASS_NAMES = {0: "bulk", 1: "tail", 2: "salient"}


def resolve_snapshot(model_name):
    cache = os.environ.get("BILLM_DOWNLOADS_DIR",
                           os.path.join(REPO, "downloads"))
    pat = os.path.join(cache,
                       "models--" + model_name.replace("/", "--"),
                       "snapshots", "*")
    snaps = sorted(glob.glob(pat))
    if not snaps:
        raise SystemExit(f"no HF snapshot under {pat}")
    return snaps[-1]


class FPWeights:
    """Streams FP16 weights out of the HF snapshot safetensors shards."""

    def __init__(self, snap_dir):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.key2file = {}
        idx = os.path.join(snap_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                weight_map = json.load(f)["weight_map"]
            for k, fn in weight_map.items():
                self.key2file[k] = os.path.join(snap_dir, fn)
        else:
            fn = os.path.join(snap_dir, "model.safetensors")
            with safe_open(fn, framework="pt", device="cpu") as f:
                for k in f.keys():
                    self.key2file[k] = fn

    def get(self, key):
        fn = self.key2file[key]
        with self._safe_open(fn, framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def load_actstats(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d.pop("__meta__")
    out = {}
    for name, st in d.items():
        n = st["n"].item()
        out[name] = {
            "Ex": (st["sum_x"] / n).numpy(),
            "Ex2": (st["sum_x2"] / n).numpy(),
            "max_abs": st["max_abs"].numpy(),
        }
    return out, meta


def row_stats(v):
    """Distribution stats over per-row values v (numpy 1D, >=0)."""
    tot = float(v.sum())
    if tot <= 0:
        return {"total": tot, "max_share": 0.0, "top1pct_share": 0.0,
                "kurtosis": 0.0}
    vs = np.sort(v)[::-1]
    k = max(1, int(round(0.01 * len(vs))))
    m, s = v.mean(), v.std()
    kurt = float(((v - m) ** 4).mean() / (s ** 4)) if s > 0 else 0.0
    return {"total": tot,
            "max_share": float(vs[0] / tot),
            "top1pct_share": float(vs[:k].sum() / tot),
            "kurtosis": kurt}


def analyze_one(E, W, part, Ex, Ex2, prefix):
    """E, W: (R,C) fp64 numpy; part: (R,C) int in {0,1,2} or None (-> 'all').
    Returns (flat_result_dict, per_row_dict)."""
    E2 = E * E
    v = E2 @ Ex2                    # per-row diag-proxy output-error power
    b = E @ Ex                      # per-row systematic bias
    w2v = (W * W) @ Ex2             # act-weighted signal power per row

    res = {}
    res.update({f"{prefix}_v_{k}": val for k, val in row_stats(v).items()})
    res[f"{prefix}_bias2_total"] = float((b * b).sum())
    res[f"{prefix}_bias2_over_v"] = (
        float((b * b).sum() / v.sum()) if v.sum() > 0 else 0.0)
    res[f"{prefix}_nmse_w"] = (
        float(v.sum() / w2v.sum()) if w2v.sum() > 0 else float("inf"))
    res[f"{prefix}_mse_raw"] = float(E2.mean())

    classes = ([("all", None)] if part is None else
               [(CLASS_NAMES[c], c) for c in (0, 1, 2)] + [("all", None)])
    for cname, c in classes:
        mask = np.ones_like(E, dtype=bool) if c is None else (part == c)
        cnt = int(mask.sum())
        res[f"{prefix}_{cname}_count"] = cnt
        if cnt == 0:
            continue
        e2m = E2[mask]
        res[f"{prefix}_{cname}_err_energy"] = float(e2m.sum())
        # act-weighted energy of this class only
        Ec = np.where(mask, E, 0.0)
        vc = (Ec * Ec) @ Ex2
        res[f"{prefix}_{cname}_v_total"] = float(vc.sum())
        abse = np.abs(E[mask])
        for q in (50, 90, 99):
            res[f"{prefix}_{cname}_abse_p{q}"] = float(
                np.percentile(abse, q))
        wm = np.abs(W[mask])
        res[f"{prefix}_{cname}_w_energy"] = float((wm * wm).sum())

    per_row = {f"{prefix}_v": v.astype(np.float32),
               f"{prefix}_b": b.astype(np.float32),
               f"{prefix}_w2v": w2v.astype(np.float32)}
    return res, per_row


def recover_fold_scale(wref, wfp):
    """Per-column AWQ fold scale s_j = row-median of wref/wfp; robust to
    zeros. Returns (s, consistency) where consistency = median absolute
    relative deviation of the ratio across rows (0 => perfectly rank-1)."""
    ratio = np.where(np.abs(wfp) > 1e-8, wref / np.where(
        np.abs(wfp) > 1e-8, wfp, 1.0), np.nan)
    s = np.nanmedian(ratio, axis=0)
    s = np.where(np.isfinite(s) & (np.abs(s) > 1e-8), s, 1.0)
    with np.errstate(invalid="ignore"):
        dev = np.abs(ratio / s[None, :] - 1.0)
    cons = float(np.nanmedian(dev))
    return s, cons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--doml-dir", required=True)
    ap.add_argument("--tq-dir", default=None,
                    help="TesseraQ dump with wq+wref (optional; skip if the "
                         "repro has not finished)")
    ap.add_argument("--actstats", required=True,
                    help="dir holding actstats_wt2calib.pt / actstats_arceasy.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    snap = resolve_snapshot(args.model)
    print(f"FP snapshot: {snap}", flush=True)
    fpw = FPWeights(snap)

    streams = {}
    for s in ("wt2calib", "arceasy"):
        p = os.path.join(args.actstats, f"actstats_{s}.pt")
        if os.path.exists(p):
            streams[s], _ = load_actstats(p)
            print(f"actstats stream loaded: {s}", flush=True)
    if not streams:
        raise SystemExit("no actstats files found")

    doml_wq = sorted(glob.glob(os.path.join(args.doml_dir,
                                            "*.wq.safetensors")))
    if not doml_wq:
        raise SystemExit(f"no wq files in {args.doml_dir}")

    from safetensors import safe_open

    rows = []
    for i, wq_path in enumerate(doml_wq):
        lname = os.path.basename(wq_path)[:-len(".wq.safetensors")]
        with safe_open(wq_path, framework="pt", device="cpu") as f:
            wq_doml = f.get_tensor("wq").to(torch.float64).numpy()
        W = fpw.get(lname + ".weight").to(torch.float64).numpy()
        assert W.shape == wq_doml.shape, (lname, W.shape, wq_doml.shape)

        dpk_path = wq_path.replace(".wq.", ".dpk.")
        tensors, meta = dpk_unpack.load_container(dpk_path, "cpu")
        part = dpk_unpack.part_matrix(tensors, meta).numpy()
        part = part[:, :W.shape[1]]

        res = {"sublayer": lname, "R": W.shape[0], "C": W.shape[1]}
        per_row_all = {}
        E_doml = wq_doml - W
        for sname, st in streams.items():
            stt = st[lname]
            r, pr = analyze_one(E_doml, W, part, stt["Ex"], stt["Ex2"],
                                f"doml_{sname}")
            res.update(r)
            per_row_all.update(pr)

        if args.tq_dir:
            tq_wq_p = os.path.join(args.tq_dir,
                                   f"{lname}.wq.safetensors")
            tq_wr_p = os.path.join(args.tq_dir,
                                   f"{lname}.wref.safetensors")
            if os.path.exists(tq_wq_p) and os.path.exists(tq_wr_p):
                with safe_open(tq_wq_p, framework="pt", device="cpu") as f:
                    wq_tq = f.get_tensor("wq").to(torch.float64).numpy()
                with safe_open(tq_wr_p, framework="pt", device="cpu") as f:
                    wref = f.get_tensor("wref").to(torch.float64).numpy()
                s_fold, cons = recover_fold_scale(wref, W)
                res["tq_fold_consistency"] = cons
                # unfold to original coordinates: divide columns by s_j
                E_tq = (wq_tq - wref) / s_fold[None, :]
                for sname, st in streams.items():
                    stt = st[lname]
                    r, pr = analyze_one(E_tq, W, None, stt["Ex"],
                                        stt["Ex2"], f"tq_{sname}")
                    res.update(r)
                    per_row_all.update(pr)

        np.savez_compressed(
            os.path.join(args.out, f"{lname}.rows.npz"), **per_row_all)
        rows.append(res)
        if (i + 1) % 21 == 0:
            print(f"[{i+1}/{len(doml_wq)}] {lname}", flush=True)

    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (k != "sublayer", k))
    with open(os.path.join(args.out, "cell_errors.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} sublayer rows -> "
          f"{os.path.join(args.out, 'cell_errors.csv')}", flush=True)


if __name__ == "__main__":
    main()
