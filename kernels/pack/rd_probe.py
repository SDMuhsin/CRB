"""Offline rate-distortion probe for DOML non-salient weight encoders.

Immutable objective context (K29 sub-2.5 frontier): compare, on REAL
Qwen3-0.6B weights, the rate-distortion of the DOML operating point
(bulk-K2 + tail-K4 + explicit membership plane) against an
entropy-constrained uniform scalar quantizer (ECSQ) for the NON-SALIENT
weights. This is a PURE weight-space RD study -- NO model PPL eval.

Faithful setup (matches the real method's primitives):
  * Model: Qwen/Qwen3-0.6B, attn_implementation="eager", use_safetensors=True,
    loaded via run.get_model (the harness code path).
  * Calibration: wikitext2, nsamples=128, seqlen=2048, seed=0 (run.py defaults),
    via datautils.get_loaders.
  * Gram matrix H: captured via forward hooks on the ORIGINAL (unquantized)
    model, accumulated with the EXACT scaling of BRAGPTQ.add_batch
    (H *= n/(n+t); n+=t; inp = sqrt(2/n)*inp; H += inp @ inp.t()), one
    calib sample per hook call (t=1), matching the real per-sample forward
    loop. ACCEPTED APPROXIMATION for this differential RD probe: the real
    method feeds each sublayer the activations of the *previously quantized*
    layers (sequential); here every sublayer sees unquantized activations.
    This avoids a full sequential quantization pass; the RD *difference*
    between the two encoders is insensitive to this common H approximation.
  * Preamble per sublayer: verbatim replica of refit_fasterquant --
    dead-column zeroing, damp H[diag]+=0.01*mean(diag(H)),
    H_diag_raw=diag(H).clone(), Hinv=cholesky(cholesky_inverse(cholesky(H)),
    upper=True) with the same 10-retry damping ladder.
  * hdiag column weights w_j: H_diag_raw.clamp(min=0), mean-normalized to
    mean 1 (the harness _col_weights 'hdiag' path with p=1).
  * Masks: per 128-col block, bigptq.structural_guassian_distribution(
    W_blk, Hinv_blk, "magnitude", 50, orders=(1,1,2)) -> (bulk,tail,salient).
    NOTE: structural_searching ignores its `orders` arg (always order2 salient
    / order1 bulk-tail), so orders=(1,1,2) here == the real doml run's (1,1,1)
    mask output bitwise. Non-salient = bulk | tail.
  * Group size g=256 (the K29 sub-2.5 config). One codebook per (row,group,
    partition) fit over the group's union mask.
  * DOML codebooks: the harness _weighted_lloyd_max_quantize (hdiag-weighted
    Lloyd-Max), matching K29. bulk=K2, tail=K4.

  GPTQ error feedback is IGNORED for BOTH encoders in this probe (it is an
  orthogonal error-feedback term that helps both similarly; the RD question
  is about the quantizer, not the feedback). Stated explicitly.

Stratified sample: blocks {0,9,18,27} x {self_attn.q_proj, self_attn.o_proj,
mlp.gate_proj, mlp.down_proj} = 16 sublayers.
"""

import os
import sys
import json
import math
import io
from contextlib import redirect_stdout

REPO = os.environ.get("CRB_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault(
    "BILLM_BENCH_CSV",
    os.path.join(REPO, "llmdocs/cuda_kernel/verify/scratch_results.csv"))

import torch  # noqa: E402

import run  # noqa: E402  (harness get_model code path)
from datautils import get_loaders  # noqa: E402
import bigptq  # noqa: E402  (structural_guassian_distribution)
from doml_group_refit import (  # noqa: E402
    _weighted_lloyd_max_quantize, _snap_to_levels)

DEV = "cuda:0"
MODEL = "Qwen/Qwen3-0.6B"
G = 256
BLK = 128
LAYERS = [0, 9, 18, 27]
NAMES = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"]
ALPHAS = [0.15, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0]
OUT_JSON = os.path.join(REPO, "llmdocs/cuda_kernel/verify/rd_probe_results.json")
FP8_BITS = 8.0  # codebook/scale levels stored as fp8


class HAccum:
    """Verbatim replica of BRAGPTQ.add_batch (nn.Linear path) H accumulation."""

    def __init__(self, columns, dev):
        self.H = torch.zeros((columns, columns), device=dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


def preamble(H, columns, percdamp=0.01):
    """Verbatim replica of refit_fasterquant preamble. Returns (Hinv, col_w)."""
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=H.device)
    H[diag, diag] += damp
    H_diag_raw = torch.diag(H).clone()
    for _retry in range(10):
        try:
            H_chol = torch.linalg.cholesky(H)
            break
        except torch._C._LinAlgError:
            extra_damp = 1e-3 * torch.mean(torch.diag(H))
            if extra_damp == 0:
                extra_damp = 1e-6
            H[diag, diag] += extra_damp
    else:
        H_chol = torch.diag(torch.sqrt(torch.diag(H).clamp(min=1e-8)))
    H = torch.cholesky_inverse(H_chol)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    # _col_weights 'hdiag' path (p=1): clamp>=0, mean-normalize to mean 1.
    w = H_diag_raw.to(torch.float32).clamp(min=0)
    w = w / w.mean().clamp(min=1e-30)
    return Hinv, w, dead


def assign_indices(x, mask, levels):
    """Nearest-level index for masked elements (for empirical code entropy)."""
    xe = (x * mask.float()).unsqueeze(2)
    le = levels.unsqueeze(1)
    d = (xe - le) ** 2 + (~mask).unsqueeze(2).float() * 1e30
    return d.argmin(dim=2)


def merge_counts(d, codes):
    if codes.numel() == 0:
        return
    u, c = torch.unique(codes, return_counts=True)
    for k, v in zip(u.cpu().tolist(), c.cpu().tolist()):
        d[k] = d.get(k, 0) + v


def entropy_from_counts(d):
    tot = sum(d.values())
    if tot == 0:
        return 0.0
    H = 0.0
    for v in d.values():
        p = v / tot
        H -= p * math.log2(p)
    return H


def binH(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


@torch.no_grad()
def main():
    print("[rd_probe] loading model + calib...", flush=True)
    model = run.get_model(MODEL)
    model.eval()
    model.config.use_cache = False
    dataloader, _ = get_loaders("wikitext2", nsamples=128, seed=0,
                                model=MODEL, seqlen=2048)
    model.to(DEV)

    # ---- capture H on the ORIGINAL model via forward hooks -----------------
    targets = {}
    for li in LAYERS:
        layer = model.model.layers[li]
        targets[f"{li}.self_attn.q_proj"] = layer.self_attn.q_proj
        targets[f"{li}.self_attn.o_proj"] = layer.self_attn.o_proj
        targets[f"{li}.mlp.gate_proj"] = layer.mlp.gate_proj
        targets[f"{li}.mlp.down_proj"] = layer.mlp.down_proj
    accums = {n: HAccum(m.weight.shape[1], DEV) for n, m in targets.items()}

    def mk(nm):
        def hook(_m, inp, _out):
            accums[nm].add_batch(inp[0].detach())
        return hook

    handles = [m.register_forward_hook(mk(n)) for n, m in targets.items()]
    print("[rd_probe] forwarding 128 calib samples for H...", flush=True)
    for bi, batch in enumerate(dataloader):
        model(batch[0].to(DEV))
    for h in handles:
        h.remove()
    print("[rd_probe] H captured (nsamples=%d)." % accums[
        f"{LAYERS[0]}.self_attn.q_proj"].nsamples, flush=True)

    devnull = io.StringIO()

    # ---- global accumulators ----------------------------------------------
    n_bulk_tot = n_tail_tot = n_sal_tot = 0
    rowgroup_count = 0            # sum R*NG over sublayers
    nonsal_count = 0             # count of nonsal weights (unweighted)
    D_doml_num = 0.0             # weighted MSE numerators / denoms
    nonsal_wden = 0.0
    D_bulkK2_num = 0.0
    bulk_wden = 0.0
    D_tailK4_num = 0.0
    tail_wden = 0.0
    doml_bulk_counts, doml_tail_counts = {}, {}
    per_layer_p = []             # per-sublayer tail fraction for mean H
    ecsqB_Dnum = {a: 0.0 for a in ALPHAS}
    ecsqB_Dnum_unw = {a: 0.0 for a in ALPHAS}   # unweighted (sanity 3)
    ecsqB_counts = {a: {} for a in ALPHAS}
    ecsqC_bulkDnum = {a: 0.0 for a in ALPHAS}
    ecsqC_bulk_counts = {a: {} for a in ALPHAS}

    for name, mod in targets.items():
        W = mod.weight.data.clone().float()
        R, C = W.shape
        Hinv, col_w, dead = preamble(accums[name].H, C)
        W[:, dead] = 0
        NG = C // G
        l_bulk = l_tail = l_nonsal = 0
        for gi in range(NG):
            st = gi * G
            Wg = W[:, st:st + G]
            cw = col_w[st:st + G]
            wmat = cw.view(1, -1)                       # [1,g] broadcast
            # per-128-block 3-way masks (bulk,tail,salient)
            mb, mt, ms = [], [], []
            for b in range(G // BLK):
                bs = st + b * BLK
                with redirect_stdout(devnull):
                    m1, m2, m3 = bigptq.structural_guassian_distribution(
                        W[:, bs:bs + BLK], Hinv[bs:bs + BLK, bs:bs + BLK],
                        "magnitude", 50, orders=(1, 1, 2))
                mb.append(m1); mt.append(m2); ms.append(m3)
            ub = torch.cat(mb, 1)
            ut = torch.cat(mt, 1)
            us = torch.cat(ms, 1)
            nonsal = ub | ut
            ubf, utf, nsf = ub.float(), ut.float(), nonsal.float()

            # ---- Encoder A: DOML (bulk-K2 + tail-K4, hdiag-weighted Lloyd) --
            rec_b, lev_b = _weighted_lloyd_max_quantize(Wg, ub, cw, K=2, iters=20)
            rec_t, lev_t = _weighted_lloyd_max_quantize(Wg, ut, cw, K=4, iters=20)
            Wq_doml = rec_b + rec_t
            diff2 = (Wg - Wq_doml) ** 2
            D_doml_num += float((wmat * diff2 * nsf).sum())
            nonsal_wden += float((wmat * nsf).sum())
            D_bulkK2_num += float((wmat * (Wg - rec_b) ** 2 * ubf).sum())
            bulk_wden += float((wmat * ubf).sum())
            D_tailK4_num += float((wmat * (Wg - rec_t) ** 2 * utf).sum())
            tail_wden += float((wmat * utf).sum())
            # DOML code index entropy pools
            merge_counts(doml_bulk_counts, assign_indices(Wg, ub, lev_b)[ub])
            merge_counts(doml_tail_counts, assign_indices(Wg, ut, lev_t)[ut])

            nb = int(ub.sum()); nt = int(ut.sum()); nsl = int(us.sum())
            n_bulk_tot += nb; n_tail_tot += nt; n_sal_tot += nsl
            nonsal_count += nb + nt
            rowgroup_count += R
            l_bulk += nb; l_tail += nt; l_nonsal += nb + nt

            # ---- Encoder B: ECSQ over ALL nonsal (single stream) -----------
            cnt = nonsal.sum(1).clamp(min=1)
            mean = (Wg * nsf).sum(1) / cnt
            var = ((Wg - mean[:, None]) ** 2 * nsf).sum(1) / cnt
            s = var.sqrt().clamp(min=1e-8)              # per-row scale
            for a in ALPHAS:
                delta = (a * s).clamp(min=1e-12)[:, None]
                code = torch.round(Wg / delta)
                Wq = code * delta
                d2 = (Wg - Wq) ** 2
                ecsqB_Dnum[a] += float((wmat * d2 * nsf).sum())
                ecsqB_Dnum_unw[a] += float((d2 * nsf).sum())
                merge_counts(ecsqB_counts[a], code[nonsal].long())

            # ---- Encoder C: ECSQ over BULK only (tail stays K4) ------------
            cntb = ub.sum(1).clamp(min=1)
            meanb = (Wg * ubf).sum(1) / cntb
            varb = ((Wg - meanb[:, None]) ** 2 * ubf).sum(1) / cntb
            sb = varb.sqrt().clamp(min=1e-8)
            for a in ALPHAS:
                delta = (a * sb).clamp(min=1e-12)[:, None]
                code = torch.round(Wg / delta)
                Wq = code * delta
                d2 = (Wg - Wq) ** 2
                ecsqC_bulkDnum[a] += float((wmat * d2 * ubf).sum())
                merge_counts(ecsqC_bulk_counts[a], code[ub].long())

        if l_nonsal > 0:
            per_layer_p.append(l_tail / l_nonsal)
        print("[rd_probe] %-24s R=%d C=%d NG=%d nonsal-bulk=%.3f nonsal-tail=%.3f"
              % (name, R, C, NG, l_bulk / max(1, l_nonsal),
                 l_tail / max(1, l_nonsal)), flush=True)

    del model
    torch.cuda.empty_cache()

    # ---- reduce ------------------------------------------------------------
    n_nonsal = n_bulk_tot + n_tail_tot
    frac_bulk = n_bulk_tot / n_nonsal
    frac_tail = n_tail_tot / n_nonsal
    tot_all = n_bulk_tot + n_tail_tot + n_sal_tot
    part_bulk = n_bulk_tot / tot_all
    part_tail = n_tail_tot / tot_all
    part_sal = n_sal_tot / tot_all

    D_doml = D_doml_num / nonsal_wden
    D_bulkK2 = D_bulkK2_num / bulk_wden
    D_tailK4 = D_tailK4_num / tail_wden

    doml_codes = (n_bulk_tot * 1 + n_tail_tot * 2) / n_nonsal
    p_global = frac_tail
    memb_H_global = binH(p_global)
    memb_H_mean_layer = sum(binH(p) for p in per_layer_p) / len(per_layer_p)
    doml_total = doml_codes + memb_H_global
    codebook_overhead = FP8_BITS * 6.0 * rowgroup_count / n_nonsal   # 6 levels
    scale_overhead = FP8_BITS * 1.0 * rowgroup_count / n_nonsal      # 1 fp8

    # Encoder B RD curve
    curveB = []
    for a in ALPHAS:
        rate = entropy_from_counts(ecsqB_counts[a])
        D = ecsqB_Dnum[a] / nonsal_wden
        D_unw = ecsqB_Dnum_unw[a] / nonsal_count
        curveB.append({"alpha": a, "rate": rate, "D": D, "D_unw": D_unw})

    # Encoder C RD curve (full nonsal D + rate)
    curveC = []
    for a in ALPHAS:
        Hbulk = entropy_from_counts(ecsqC_bulk_counts[a])
        D_full = (ecsqC_bulkDnum[a] + D_tailK4_num) / nonsal_wden
        D_bulk = ecsqC_bulkDnum[a] / bulk_wden
        rate = (Hbulk * n_bulk_tot + 2 * n_tail_tot) / n_nonsal + memb_H_global
        curveC.append({"alpha": a, "bulk_entropy": Hbulk, "rate": rate,
                       "D_full": D_full, "D_bulk_only": D_bulk})

    # DOML code-index entropies (sanity 2)
    Hb_doml = entropy_from_counts(doml_bulk_counts)
    Ht_doml = entropy_from_counts(doml_tail_counts)

    # ---- VERDICT V1: ECSQ-B vs DOML on codes+membership --------------------
    import numpy as np
    r = np.array([c["rate"] for c in curveB])
    d = np.array([c["D"] for c in curveB])
    order = np.argsort(r)          # rate ascending
    r_s, d_s = r[order], d[order]  # d_s descending as r increases
    # D_ecsq at rate == doml_total
    D_ecsq_at_domlrate = float(np.interp(doml_total, r_s, d_s))
    v1_D_better_pct = (D_doml - D_ecsq_at_domlrate) / D_doml * 100.0
    # rate at which D_ecsq == D_doml  (invert: xp must increase -> ascending D)
    dord = np.argsort(d_s)
    rate_at_Ddoml = float(np.interp(D_doml, d_s[dord], r_s[dord]))
    v1_bits_saved = doml_total - rate_at_Ddoml
    v1_in_range = bool((r_s.min() <= doml_total <= r_s.max()) and
                       (d_s.min() <= D_doml <= d_s.max()))

    # ---- VERDICT V2: ECSQ-bulk vs bulk-K2 at matched rate 1.0 --------------
    rc = np.array([c["bulk_entropy"] for c in curveC])
    dc = np.array([c["D_bulk_only"] for c in curveC])
    oc = np.argsort(rc)
    rc_s, dc_s = rc[oc], dc[oc]
    D_ecsqbulk_at_1bit = float(np.interp(1.0, rc_s, dc_s))
    v2_pct = (D_bulkK2 - D_ecsqbulk_at_1bit) / D_bulkK2 * 100.0
    v2_in_range = bool(rc_s.min() <= 1.0 <= rc_s.max())

    # ---- sanity checks -----------------------------------------------------
    # 1: fine step -> D->~0 and rate rises; D monotone increasing in alpha
    Ds_by_alpha = [c["D"] for c in curveB]
    rates_by_alpha = [c["rate"] for c in curveB]
    s1_fine_small = Ds_by_alpha[0] < D_doml            # finest alpha well below
    s1_mono_D = all(Ds_by_alpha[i] <= Ds_by_alpha[i + 1] + 1e-12
                    for i in range(len(Ds_by_alpha) - 1))
    s1_mono_rate = all(rates_by_alpha[i] >= rates_by_alpha[i + 1] - 1e-9
                       for i in range(len(rates_by_alpha) - 1))
    s1_pass = bool(s1_fine_small and s1_mono_D and s1_mono_rate)
    # 2: DOML bulk entropy ~1.0, tail entropy ~2.0
    s2_bulk_ok = abs(Hb_doml - 1.0) < 0.10
    s2_tail_ok = abs(Ht_doml - 2.0) < 0.15
    s2_pass = bool(s2_bulk_ok and s2_tail_ok)
    # 3: uniform col_w reproduces plain-MSE ordering (monotone in alpha)
    Dunw = [c["D_unw"] for c in curveB]
    s3_pass = bool(all(Dunw[i] <= Dunw[i + 1] + 1e-12
                       for i in range(len(Dunw) - 1)))

    results = {
        "config": {"model": MODEL, "g": G, "nsamples": 128, "seqlen": 2048,
                   "seed": 0, "layers": LAYERS, "names": NAMES,
                   "alphas": ALPHAS,
                   "note": "GPTQ error feedback IGNORED for both encoders; "
                           "H from unquantized forward (differential-RD approx)"},
        "partition_fractions": {"bulk": part_bulk, "tail": part_tail,
                                "salient": part_sal,
                                "nonsal_bulk_frac": frac_bulk,
                                "nonsal_tail_frac": frac_tail},
        "counts": {"n_bulk": n_bulk_tot, "n_tail": n_tail_tot,
                   "n_sal": n_sal_tot, "n_nonsal": n_nonsal,
                   "rowgroup_count": rowgroup_count},
        "tableA_doml": {"D_doml": D_doml, "D_bulkK2": D_bulkK2,
                        "D_tailK4": D_tailK4,
                        "codes_rate": doml_codes,
                        "membership_rate_pooled_H": memb_H_global,
                        "membership_rate_mean_layer_H": memb_H_mean_layer,
                        "total_nonsal_rate": doml_total,
                        "codebook_overhead_bpw": codebook_overhead,
                        "doml_bulk_code_entropy": Hb_doml,
                        "doml_tail_code_entropy": Ht_doml},
        "tableB_ecsq_nonsal": curveB,
        "ecsq_scale_overhead_bpw": scale_overhead,
        "tableC_ecsq_bulk_only": curveC,
        "verdicts": {
            "V1_ecsq_vs_doml": {
                "doml_total_nonsal_rate": doml_total,
                "D_doml": D_doml,
                "D_ecsq_at_doml_rate": D_ecsq_at_domlrate,
                "D_better_pct": v1_D_better_pct,
                "rate_at_equal_D": rate_at_Ddoml,
                "bits_saved_at_equal_D": v1_bits_saved,
                "in_sampled_range": v1_in_range},
            "V2_ecsq_bulk_vs_bulkK2_at_1bit": {
                "D_bulkK2": D_bulkK2,
                "D_ecsq_bulk_at_1bit": D_ecsqbulk_at_1bit,
                "D_better_pct": v2_pct,
                "in_sampled_range": v2_in_range}},
        "sanity": {
            "s1_ecsq_fine_step_monotone": {"pass": s1_pass,
                                           "D_finest": Ds_by_alpha[0],
                                           "D_doml": D_doml,
                                           "mono_D": s1_mono_D,
                                           "mono_rate": s1_mono_rate},
            "s2_doml_code_entropy": {"pass": s2_pass,
                                     "bulk_entropy": Hb_doml,
                                     "tail_entropy": Ht_doml},
            "s3_uniform_colw_monotone": {"pass": s3_pass}},
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)

    # ---- compact console report -------------------------------------------
    print("\n" + "=" * 72)
    print("PARTITION FRACTIONS (pooled over 16 sublayers):")
    print("  bulk=%.4f tail=%.4f salient=%.4f  | nonsal: bulk=%.4f tail=%.4f"
          % (part_bulk, part_tail, part_sal, frac_bulk, frac_tail))
    print("\nTABLE A  DOML operating point:")
    print("  D_doml(nonsal wMSE) = %.6e" % D_doml)
    print("  D_bulkK2 = %.6e   D_tailK4 = %.6e" % (D_bulkK2, D_tailK4))
    print("  codes rate       = %.4f bits/nonsal-wt" % doml_codes)
    print("  membership rate  = %.4f (pooled H)  %.4f (mean-layer H)"
          % (memb_H_global, memb_H_mean_layer))
    print("  TOTAL nonsal rate= %.4f bits/nonsal-wt" % doml_total)
    print("  codebook overhead= %.4f bpw (6 fp8 levels/(row,group))"
          % codebook_overhead)
    print("\nTABLE B  ECSQ-nonsalient RD curve:")
    print("   alpha   rate(order0)   D_ecsq")
    for c in curveB:
        print("   %5.2f   %8.4f      %.6e" % (c["alpha"], c["rate"], c["D"]))
    print("  (ECSQ per-(row,group) scale overhead = %.4f bpw, negligible)"
          % scale_overhead)
    print("\nTABLE C  ECSQ-bulk-only (tail stays K4):")
    print("   alpha  bulk_H   rate(full)   D_full        D_bulk_only")
    for c in curveC:
        print("   %5.2f  %6.4f  %8.4f    %.6e  %.6e"
              % (c["alpha"], c["bulk_entropy"], c["rate"],
                 c["D_full"], c["D_bulk_only"]))
    print("\nVERDICTS:")
    print("  V1: DOML total nonsal rate = %.4f bits" % doml_total)
    print("      D_doml=%.6e  D_ecsq@sameRate=%.6e  -> ECSQ D better by %.2f%%"
          % (D_doml, D_ecsq_at_domlrate, v1_D_better_pct))
    print("      ECSQ reaches D_doml at rate %.4f bits -> saves %.4f bits/wt "
          "(in_range=%s)" % (rate_at_Ddoml, v1_bits_saved, v1_in_range))
    print("  V2: bulk-only @ matched 1.0 bit: D_bulkK2=%.6e "
          "D_ecsq_bulk=%.6e -> ECSQ better by %.2f%% (in_range=%s)"
          % (D_bulkK2, D_ecsqbulk_at_1bit, v2_pct, v2_in_range))
    print("\nSANITY:")
    print("  S1 fine-step/monotone : %s (D_finest=%.3e vs D_doml=%.3e, "
          "monoD=%s monoRate=%s)"
          % ("PASS" if s1_pass else "FAIL", Ds_by_alpha[0], D_doml,
             s1_mono_D, s1_mono_rate))
    print("  S2 DOML code entropy  : %s (bulk=%.4f~1.0, tail=%.4f~2.0)"
          % ("PASS" if s2_pass else "FAIL/FLAG", Hb_doml, Ht_doml))
    print("  S3 uniform-colw order : %s" % ("PASS" if s3_pass else "FAIL"))
    print("=" * 72)
    print("[rd_probe] JSON saved -> %s" % OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
