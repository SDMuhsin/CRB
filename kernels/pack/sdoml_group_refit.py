"""SDOML codebook-group REFIT: coarsen an SDOML BASE dump to a larger
codebook groupsize (default 256).

Takes an existing SDOML BASE container dir (bf16 codebooks, one K=4 codebook per
`src_groupsize`-col block, e.g. sdoml-s50 with groupsize=128) and produces a NEW
BASE container dir where each row uses ONE K=4 codebook per `groupsize`-col group
(default 256 = two adjacent 128-blocks). The keep-MASK is FROZEN byte-identical;
only the codebook granularity (and hence the per-group codebook + the resulting
wq levels) changes. This halves the codebook term in the honest bpw:
  fp8 g128 cb = K*8/128 = 0.25  ->  fp8 g256 cb = K*8/256 = 0.125,
so after 2-stage output-aware tuning (sdoml_block_tune -> sdoml_assign_tune) the
honest bpw drops from 2.25 to 2.125 at s=0.5. This refit itself writes a bf16
cb (K*16/256 = 0.25 cb term => 2.25 total in the bf16 stage); the fp8 halving
happens in stage-1 block-tune.

Refit algorithm (per layer, per row, per `groupsize`-col group):
  * collect that row's KEPT weights within the group (values present in the
    source container's wq at kept positions — the container has no FP originals,
    so we re-quantize the existing quantized levels into a coarser per-group
    codebook; output-aware tuning recovers the quality lost to coarsening);
  * fit ONE K-level Lloyd-Max codebook over those kept values (init at
    per-row-group quantiles; ~20 iters; EMPTY cluster -> hold previous centroid;
    fully-pruned group -> all-zero codebook);
  * cast the K levels to bf16, SORT ascending, and reassign each kept weight to
    its nearest bf16 level via the SAME argmin that reassemble_bitwise uses, so
    the container is bitwise self-consistent by construction; pruned stay 0.

Output container `<name>.sdpk.safetensors` (same schema as sdoml_dump):
  wq          bf16 [R, C]              recomputed = mask ? cb_bf16[code] : 0
  mask_packed uint8                    BYTE-IDENTICAL to source (frozen mask)
  cb          bf16 [R, NG_new, K]      refit per-group codebooks
  meta JSON: source meta + {NG: NG_new, block_widths: [groupsize]*NG_new,
             groupsize: <groupsize>}   (blocksize kept = source GPTQ blocksize)

GATES (hard-fail, per layer):
  (a) mask_packed bytes BYTE-IDENTICAL to source (we reuse the source bytes);
  (b) sdoml_dump.reassemble_bitwise(wq_new, mask, cb_new, [groupsize]*NG) OK;
  (c) C % groupsize == 0  (Qwen3-0.6B: C in {1024,2048,3072}, all divisible by
      256 => NG_128 even => every group is two whole 128-blocks, NO lone block).
      If C % groupsize != 0 the layer is REPORTED and REFUSED (never silently
      handled) so a lone-block model is caught loudly.
  n_kept is unchanged (mask frozen) — asserted.

Usage:
  synthetic selftest (no GPU, no real dump):
      python kernels/pack/sdoml_group_refit.py --selftest
  full refit sdoml-s50 -> sdoml-s50-g256:
      CUDA_VISIBLE_DEVICES=1 python -u kernels/pack/sdoml_group_refit.py \
          --src downloads/doml_dumps/qwen3-0.6b/sdoml-s50 \
          --out downloads/doml_dumps/qwen3-0.6b/sdoml-s50-g256 \
          --groupsize 256 --device cuda:0
"""

import argparse
import glob
import json
import os
import sys
import time

REPO = os.environ.get("CRB_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import sdoml_dump  # noqa: E402  (reassemble_bitwise / pack_mask reused, unmodified)
import sdoml_honest_bpw  # noqa: E402

EXPECTED_SUBLAYERS = 196


# ---------------------------------------------------------------------------
# batched masked Lloyd-Max over one `groupsize`-col group
# ---------------------------------------------------------------------------
@torch.no_grad()
def fit_group_codebook(data, valid, K, n_iter):
    """Fit one K-level codebook per ROW over the columns of a single group.

    data  : fp32 [R, Wg]   candidate values (garbage where ~valid)
    valid : bool [R, Wg]   kept-position mask within the group
    Returns cent fp32 [R, K] (NOT sorted).  Empty clusters hold; fully-invalid
    rows return all-zero centroids.
    """
    R, Wg = data.shape
    dev = data.device
    # init: per-row quantiles over VALID values (invalid -> NaN, nanquantile)
    dmask = torch.where(valid, data, torch.full_like(data, float("nan")))
    qs = torch.tensor([(k + 0.5) / K for k in range(K)],
                      dtype=torch.float32, device=dev)
    cent = torch.nanquantile(dmask, qs, dim=1).transpose(0, 1).contiguous()
    cent = torch.nan_to_num(cent, nan=0.0)               # fully-invalid -> 0
    inf = torch.full_like(data, float("inf"))
    for _ in range(n_iter):
        d2 = (data.unsqueeze(-1) - cent.unsqueeze(1)) ** 2     # [R, Wg, K]
        d2 = torch.where(valid.unsqueeze(-1), d2, inf.unsqueeze(-1))
        code = d2.argmin(dim=-1)                                # [R, Wg]
        onehot = torch.nn.functional.one_hot(code, K).to(torch.float32)
        onehot = onehot * valid.unsqueeze(-1).to(torch.float32)
        sums = (onehot * data.unsqueeze(-1)).sum(dim=1)        # [R, K]
        cnts = onehot.sum(dim=1)                               # [R, K]
        new_cent = torch.where(cnts > 0, sums / cnts.clamp(min=1.0), cent)
        cent = new_cent
    return cent


@torch.no_grad()
def refit_layer_tensors(wq, mask, K, groupsize, n_iter, device):
    """wq bf16 [R,C], mask bool [R,C] -> (wq_new bf16 [R,C], cb_new bf16
    [R, NG_new, K], block_widths). Pure re-quantization to per-`groupsize`
    codebooks; mask frozen (pruned stay 0)."""
    R, C = wq.shape
    assert C % groupsize == 0, (C, groupsize)          # gate (c) — no lone block
    NG_new = C // groupsize
    wq = wq.to(device)
    mask = mask.to(device)
    wq_new = torch.zeros(R, C, dtype=torch.bfloat16, device=device)
    cb_new = torch.zeros(R, NG_new, K, dtype=torch.bfloat16, device=device)
    for g in range(NG_new):
        sl = slice(g * groupsize, (g + 1) * groupsize)
        wq_blk = wq[:, sl]                                     # bf16 [R, Wg]
        m_blk = mask[:, sl]                                    # bool [R, Wg]
        data = wq_blk.float()
        cent = fit_group_codebook(data, m_blk, K, n_iter)     # fp32 [R, K]
        cb_bf16 = cent.sort(dim=1).values.to(torch.bfloat16)  # [R, K] ascending
        # reassign via the SAME argmin reassemble_bitwise uses (bit-consistent)
        diff = wq_blk.float().unsqueeze(-1) - cb_bf16.float().unsqueeze(1)
        code = (diff * diff).argmin(dim=-1)                    # [R, Wg]
        rec = cb_bf16.gather(1, code)                          # bf16 [R, Wg]
        rec = torch.where(m_blk, rec, torch.zeros_like(rec))
        wq_new[:, sl] = rec
        cb_new[:, g, :] = cb_bf16
    block_widths = [groupsize] * NG_new
    return wq_new.cpu(), cb_new.cpu(), block_widths


@torch.no_grad()
def refit_one_file(src_path, out_dir, groupsize, n_iter, device):
    """Refit a single container file, run gates, write the g<groupsize> output."""
    # read raw source bytes for the mask (byte-identical reuse) + meta
    with safe_open(src_path, framework="pt", device="cpu") as f:
        src_mask_packed = f.get_tensor("mask_packed")
        src_meta = json.loads(f.metadata()["meta"])
    wq, mask, _cb_src, meta = sdoml_dump.load_layer(src_path, device="cpu")
    R, C, K = meta["R"], meta["C"], meta["K"]
    if C % groupsize != 0:
        raise RuntimeError(
            f"{os.path.basename(src_path)}: C={C} not divisible by groupsize="
            f"{groupsize} (would need a lone short block) — REFUSED")

    wq_new, cb_new, block_widths = refit_layer_tensors(
        wq, mask, K, groupsize, n_iter, device)
    NG_new = len(block_widths)

    # GATE (b): reassemble_bitwise (group-general) on the refit container.
    sdoml_dump.reassemble_bitwise(wq_new, mask.cpu(), cb_new, block_widths)

    # GATE (a): mask bytes byte-identical (reuse source bytes verbatim). Also
    # cross-check pack_mask(mask) reproduces those bytes (deterministic).
    repacked = sdoml_dump.pack_mask(mask.cpu())
    if not torch.equal(repacked, src_mask_packed):
        raise RuntimeError(
            f"{os.path.basename(src_path)}: pack_mask != source mask bytes")

    n_kept = int(mask.sum().item())
    assert n_kept == int(src_meta["n_kept"]), (n_kept, src_meta["n_kept"])

    meta_out = dict(src_meta)
    meta_out["NG"] = int(NG_new)
    meta_out["block_widths"] = [int(x) for x in block_widths]
    meta_out["groupsize"] = int(groupsize)
    # keep source blocksize (GPTQ quant blocksize) untouched
    tensors = {
        "wq": wq_new.contiguous(),
        "mask_packed": src_mask_packed,          # byte-identical source bytes
        "cb": cb_new.to(torch.bfloat16).contiguous(),
    }
    out_path = os.path.join(out_dir,
                            os.path.basename(src_path))
    save_file(tensors, out_path, metadata={"meta": json.dumps(meta_out)})
    return {"layer_name": meta["layer_name"], "R": R, "C": C, "K": K,
            "NG_src": meta["NG"], "NG_new": NG_new, "n_kept": n_kept}


def refit_dir(src_dir, out_dir, groupsize, n_iter, device):
    paths = sorted(glob.glob(os.path.join(src_dir, "*.sdpk.safetensors")))
    if not paths:
        raise SystemExit(f"no *.sdpk.safetensors in {src_dir}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    recs = []
    for i, p in enumerate(paths):
        r = refit_one_file(p, out_dir, groupsize, n_iter, device)
        recs.append(r)
        if (i + 1) % 20 == 0 or i == 0 or i == len(paths) - 1:
            print(f"REFIT[{i + 1:3d}/{len(paths)}] {r['layer_name']} "
                  f"R={r['R']} C={r['C']} NG {r['NG_src']}->{r['NG_new']} "
                  f"kept={r['n_kept']} gates(a,b,c)=OK "
                  f"t={time.time() - t0:.1f}s", flush=True)
    # manifest
    manifest = {}
    src_manifest = os.path.join(src_dir, "manifest.json")
    if os.path.exists(src_manifest):
        with open(src_manifest) as f:
            manifest = json.load(f)
    manifest["sdoml_group_refit"] = {
        "src_dir": src_dir, "out_dir": out_dir, "groupsize": groupsize,
        "n_iter": n_iter, "n_layers": len(recs),
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"REFIT: wrote {len(recs)} refit sublayers -> {out_dir} "
          f"(groupsize={groupsize})  wall={time.time() - t0:.0f}s", flush=True)
    if len(recs) != EXPECTED_SUBLAYERS:
        print(f"REFIT WARNING: {len(recs)} != expected {EXPECTED_SUBLAYERS}",
              file=sys.stderr, flush=True)
    return recs


# ---------------------------------------------------------------------------
# synthetic selftest (no GPU / no real dump)
# ---------------------------------------------------------------------------
def main_selftest():
    import tempfile

    gen = torch.Generator().manual_seed(20260708)
    R, C, K = 10, 512, 4                          # C=512 => NG128=4 (even)
    src_gs = 128
    NG_src = C // src_gs
    bw_src = [src_gs] * NG_src
    # build a plausible BASE container: per-128-block bf16 codebooks + 50% mask
    cb = torch.randn(R, NG_src, K, generator=gen).float().sort(dim=-1).values
    cb = cb.to(torch.bfloat16)
    cb[1, 0, 0] = cb[1, 0, 1]                     # duplicate level
    mask = torch.rand(R, C, generator=gen) < 0.5
    mask[0, :] = False                            # fully-pruned row
    mask[R - 1, :] = True                         # fully-kept row
    # fully-pruned 256-group in row 2 (cols 0..255) to exercise empty groups
    mask[2, 0:256] = False
    codes = torch.randint(0, K, (R, C), generator=gen)
    wq = torch.zeros(R, C, dtype=torch.bfloat16)
    off = 0
    for b, w_b in enumerate(bw_src):
        cval = torch.gather(cb[:, b, :], 1, codes[:, off:off + w_b])
        wq[:, off:off + w_b] = torch.where(
            mask[:, off:off + w_b], cval, torch.zeros_like(cval))
        off += w_b

    with tempfile.TemporaryDirectory() as src, \
            tempfile.TemporaryDirectory() as out:
        sdoml_dump.save_layer(src, "selftest.layer", wq, mask, cb, bw_src,
                              sparsity=0.5)
        src_path = os.path.join(src, "selftest.layer.sdpk.safetensors")
        rec = refit_one_file(src_path, out, groupsize=256, n_iter=20,
                             device="cpu")
        print(f"selftest: refit gates(a,b,c) OK  NG {rec['NG_src']}->"
              f"{rec['NG_new']} kept={rec['n_kept']}", flush=True)

        out_path = os.path.join(out, "selftest.layer.sdpk.safetensors")
        # reload via sdoml_dump.load_layer (bf16 cb path) — must satisfy the
        # loader the tuners use, and be group-general.
        wq2, mask2, cb2, meta2 = sdoml_dump.load_layer(out_path)
        assert meta2["groupsize"] == 256 and meta2["NG"] == C // 256
        assert list(meta2["block_widths"]) == [256] * (C // 256)
        assert torch.equal(sdoml_dump.pack_mask(mask2),
                           sdoml_dump.pack_mask(mask)), "mask drifted"
        # independent argmin decode round-trip on the refit container
        sdoml_honest_bpw.decode_codes(wq2, mask2, cb2, meta2["block_widths"])
        # honest bpw of the bf16 g256 container: cb term = K*16/256 = 0.25
        r = sdoml_honest_bpw.measure_layer(out_path)
        assert r["groupsize"] == 256 and r["NG"] == C // 256
        cb_bpw = r["cb_bits_paddedK"] / r["n_weights"]
        assert abs(cb_bpw - (K * 16.0 / 256.0)) < 1e-9, cb_bpw
        print(f"selftest: g256 bf16 cb term = {cb_bpw:.4f} "
              f"(expect {K * 16.0 / 256.0:.4f}); honest decode round-trip OK",
              flush=True)

        # gate (c) negative control: a groupsize that does not divide C.
        try:
            refit_one_file(src_path, out, groupsize=300, n_iter=5,
                           device="cpu")
            raise AssertionError("non-divisor groupsize did NOT raise gate (c)")
        except RuntimeError as e:
            assert "not divisible" in str(e)
            print("selftest: non-divisor groupsize correctly refused (gate c)",
                  flush=True)

    print("SELFTEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        main_selftest()
    elif args.src and args.out:
        refit_dir(os.path.abspath(args.src), os.path.abspath(args.out),
                  args.groupsize, args.n_iter, args.device)
    else:
        ap.error("choose --selftest OR --src <dir> --out <dir>")
