"""ARC-4B improvement G2 — AWQ-style up_proj→down_proj activation prescale.

Mechanism-driven fix for the measured root cause (see
ARC4B_root_cause_hypothesis.md): Qwen3-4B SwiGLU intermediate channels span
~10^6× activation energy; DOML's weight-magnitude salient selection puts the
massive-activation columns of mlp.down_proj into the bulk K=2 class (measured:
top task-damage channels 89-100% bulk), and a row-shared 2-level codebook
cannot give a single column the precision its activation energy demands.

The fold (TesseraQ/AWQ element, per user hint): for each block,
    W_down[:, j] *= s_j        W_up[j, :] /= s_j
with s_j = clamp((Ex2_j / median(Ex2))^alpha, 1, smax), Ex2 from the
wt2-calibration actstats (the same measure GPTQ's Hessian uses).
EXACT function preservation: down input_j = silu(gate_j(x)) * up_j(x), so
scaling up row j scales the product linearly (verified by the built-in gate).
Zero bit cost: no new planes; both sides are dumped sublayers, so every
existing dump/restore/verify tool works unchanged on the folded model.
DOML-specific synergy: up_proj's per-ROW rescale is absorbed EXACTLY by its
per-row Lloyd codebooks (levels scale with the row), so up_proj quantization
quality is invariant; down_proj's per-COLUMN upscale moves outlier columns
out of the bulk noise floor (more relative precision + likely promotion to
salient/tail by the magnitude-based partition).

Output: an HF snapshot dir (safetensors + tokenizer + config) to be passed as
--model to doml_group_refit.py / arc_probe.py / restore tools.

Usage:
    python kernels/pack/arc4b_fold_prescale.py --model Qwen/Qwen3-4B \
        --actstats downloads/arc4b_rca/actstats/actstats_wt2calib.pt \
        --alpha 0.5 --smax 256 --out downloads/arc4b_folded/qwen3-4b-a05
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--actstats", required=True,
                    help="actstats_wt2calib.pt from arc4b_actstats.py")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smax", type=float, default=256.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.chdir(REPO)
    import run as run_mod
    from datautils import get_tokenizer

    st = torch.load(args.actstats, map_location="cpu", weights_only=False)
    meta = st.pop("__meta__")
    assert meta.get("stream", "wt2calib") == "wt2calib", meta

    print(f"Loading {args.model} ...", flush=True)
    model = run_mod.get_model(args.model)
    model.eval()

    # ---- pre-fold reference logits on two probe inputs (function gate) ----
    probe_ids = [
        torch.tensor([[151644, 872, 198, 105043, 100165, 11319]]),
        torch.arange(64).remainder(1000).unsqueeze(0) + 10,
    ]
    with torch.no_grad():
        ref_logits = [model(i).logits.float() for i in probe_ids]

    n_folded, s_summary = 0, []
    for li, block in enumerate(model.model.layers):
        name = f"model.layers.{li}.mlp.down_proj"
        if name not in st:
            raise SystemExit(f"actstats missing {name}")
        n = st[name]["n"].item()
        ex2 = (st[name]["sum_x2"] / n).to(torch.float64)
        med = ex2.median().clamp(min=1e-30)
        s = (ex2 / med).pow(args.alpha).clamp(1.0, args.smax)
        # Snap to powers of two: bf16 scaling by 2^k is EXACT (exponent
        # shift, no mantissa rounding) -> the fold is function-preserving to
        # the bit; arbitrary scales cost ~2e-3 relative rounding per weight
        # which accumulated to 1.9e-2 logit drift over 36 blocks (measured).
        s = torch.pow(2.0, torch.log2(s).round()).clamp(1.0, args.smax)
        up = block.mlp.up_proj.weight
        down = block.mlp.down_proj.weight
        assert up.shape[0] == down.shape[1] == s.numel(), \
            (up.shape, down.shape, s.numel())
        with torch.no_grad():
            down_f = down.to(torch.float64) * s.unsqueeze(0)
            up_f = up.to(torch.float64) / s.unsqueeze(1)
            down.copy_(down_f.to(down.dtype))
            up.copy_(up_f.to(up.dtype))
        n_folded += 1
        s_summary.append((li, float(s.max()), float((s > 1.0).float().mean())))

    print(f"folded {n_folded} blocks; per-block (max_s, frac>1):", flush=True)
    for li, mx, fr in s_summary[:5] + s_summary[-3:]:
        print(f"  block {li}: max_s={mx:.1f} frac_scaled={fr:.3f}")

    # ---- post-fold gate: logits must match (bf16 fold rounding only) ----
    with torch.no_grad():
        worst = 0.0
        for ids, ref in zip(probe_ids, ref_logits):
            new = model(ids).logits.float()
            scale = ref.abs().max().clamp(min=1.0)
            worst = max(worst, ((new - ref).abs().max() / scale).item())
    print(f"function gate: worst |Δlogit|/max|logit| = {worst:.3e}", flush=True)
    # power-of-two scales are exact in bf16 — demand near-bit-exactness
    if worst > 1e-6:
        raise SystemExit(f"FOLD GATE FAILED: {worst:.3e} > 1e-6 — aborting, "
                         f"nothing saved")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok = get_tokenizer(args.model)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "FOLD_MANIFEST.json"), "w") as f:
        import json
        json.dump({"base_model": args.model, "alpha": args.alpha,
                   "smax": args.smax, "actstats": os.path.abspath(args.actstats),
                   "gate_worst_rel_logit_diff": worst,
                   "n_blocks_folded": n_folded}, f, indent=2)
    print(f"saved folded snapshot -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
