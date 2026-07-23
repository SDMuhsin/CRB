"""ARC-4B RCA — per-question ARC probe.

Runs the EXACT eval_arc.py scoring procedure (0-shot, length-normalized
completion log-likelihood, same tokenizer path, same truncation rule) but
records per-question detail needed for flip/margin analysis:

    qid, gt_idx, pred_idx, correct, and per-choice (total_ll, n_tokens, score)

Model configurations:
    --restore none          : FP16 reference
    --restore doml:<dir>    : load every <layer>.wq.safetensors over the FP
                              model (K31/DOML dump layout)
    --restore tq:<dir>      : load every <layer>.wq.safetensors AND
                              norms.safetensors (AWQ-mutated RMSNorms — a TQ
                              restore without norms is silently WRONG)

Output: JSON with meta + one record per question.

Usage:
    python kernels/pack/arc_probe.py --model Qwen/Qwen3-4B --device cuda:1 \
        --restore doml:downloads/doml_dumps/qwen3-4b/k31-rdsplit-lam3e-4-g256 \
        --out downloads/arc4b_rca/probe_doml_raw.json
"""

import argparse
import glob
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def load_model(model_name):
    os.chdir(REPO)
    import run as run_mod
    model = run_mod.get_model(model_name)
    model.eval()
    model.config.use_cache = False
    return model


def _blk_of(name):
    # 'model.layers.<i>....' -> i
    return int(name.split(".")[2])


def _fp_col_picker(fp_cols, dump_dir):
    """fp_cols: 'topk:<k>[:<seed>]' | 'rand:<k>:<seed>' — per down_proj,
    columns to KEEP FP16 (values restored after the wq copy). topk = the k
    largest-Ex2 (wt2-calib actstats) input columns = the massive-activation
    sink columns; rand = k uniform columns EXCLUDING the top-k (bystander
    control at identical size). Returns fn(lname, C) -> LongTensor cols."""
    mode, _, rest = fp_cols.partition(":")
    parts = rest.split(":")
    k = int(parts[0])
    seed = int(parts[1]) if len(parts) > 1 else 0
    st = torch.load(os.path.join(REPO, "downloads/arc4b_rca/actstats/"
                                       "actstats_wt2calib.pt"),
                    map_location="cpu", weights_only=False)

    def pick(lname, C):
        ex2 = st[lname]["sum_x2"]
        assert ex2.numel() == C, (lname, ex2.numel(), C)
        top = torch.topk(ex2, k).indices
        if mode == "topk":
            return top
        gen = torch.Generator().manual_seed(seed + hash(lname) % 10000)
        mask = torch.ones(C, dtype=torch.bool)
        mask[top] = False
        rest_idx = mask.nonzero().flatten()
        sel = rest_idx[torch.randperm(rest_idx.numel(), generator=gen)[:k]]
        return sel

    return pick


def apply_restore(model, spec, fp_blocks=None, q_blocks=None, fp_pat=None,
                  fp_cols=None):
    """spec: 'none' | 'doml:<dir>' | 'tq:<dir>'. Returns n_loaded.

    Transplant filters (block indices):
      fp_blocks — blocks to KEEP FP16 (their wq files are skipped);
      q_blocks  — if given, ONLY these blocks are quantized;
      fp_cols   — 'topk:<k>' / 'rand:<k>:<seed>': per mlp.down_proj, keep
                  these input COLUMNS FP16 (see _fp_col_picker).
    """
    if spec == "none":
        return 0
    kind, _, dump_dir = spec.partition(":")
    assert kind in ("doml", "tq") and os.path.isdir(dump_dir), spec
    fp_blocks = set(fp_blocks or ())
    q_blocks = None if q_blocks is None else set(q_blocks)
    col_pick = _fp_col_picker(fp_cols, dump_dir) if fp_cols else None

    def want(bi, name=None):
        if bi in fp_blocks:
            return False
        if name is not None and fp_pat and fp_pat in name:
            return False
        return q_blocks is None or bi in q_blocks

    from safetensors import safe_open
    sd = model.state_dict()
    n = 0
    n_cols_kept = 0
    for wq_path in sorted(glob.glob(os.path.join(dump_dir,
                                                 "*.wq.safetensors"))):
        lname = os.path.basename(wq_path)[:-len(".wq.safetensors")]
        if not want(_blk_of(lname), lname):
            continue
        key = lname + ".weight"
        with safe_open(wq_path, framework="pt", device="cpu") as f:
            wq = f.get_tensor("wq")
        tgt = sd[key]
        assert tuple(wq.shape) == tuple(tgt.shape), (key, wq.shape, tgt.shape)
        keep_cols = None
        if col_pick is not None and lname.endswith("mlp.down_proj"):
            keep_cols = col_pick(lname, tgt.shape[1])
            fp_vals = tgt[:, keep_cols].clone()   # model starts FP16
        tgt.copy_(wq.to(tgt.dtype))
        if keep_cols is not None:
            tgt[:, keep_cols] = fp_vals
            n_cols_kept += keep_cols.numel()
        n += 1
    if col_pick is not None:
        print(f"fp_cols={fp_cols}: kept {n_cols_kept} FP16 columns total",
              flush=True)
    if kind == "tq":
        norms_path = os.path.join(dump_dir, "norms.safetensors")
        if not os.path.exists(norms_path):
            raise SystemExit(
                f"TQ restore REFUSED: {norms_path} missing — AWQ folded "
                f"scales into RMSNorm weights; restoring Linears alone "
                f"produces a wrong model.")
        with safe_open(norms_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not want(_blk_of(key)):
                    continue
                tgt = sd[key]
                t = f.get_tensor(key)
                assert tuple(t.shape) == tuple(tgt.shape), key
                tgt.copy_(t.to(tgt.dtype))
                n += 1
    return n


@torch.no_grad()
def probe(model, model_name, device, split, limit=None):
    from datasets import load_dataset
    from datautils import get_tokenizer
    from eval_arc import compute_completion_ll
    cache = os.path.join(os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads"),
                         "datasets")
    tokenizer = get_tokenizer(model_name)
    ds = load_dataset("allenai/ai2_arc", split, split="test",
                      cache_dir=cache)
    records, correct, total = [], 0, 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        labels = ex["choices"]["label"]
        ak = ex["answerKey"]
        try:
            gt_idx = labels.index(ak)
        except ValueError:
            lm = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
            try:
                gt_idx = labels.index(lm.get(ak, ak))
            except ValueError:
                continue
        prompt = f"Question: {ex['question']}\nAnswer:"
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        choices = []
        for choice_text in ex["choices"]["text"]:
            full_ids = tokenizer(prompt + " " + choice_text,
                                 return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] > model.seqlen:
                overshoot = full_ids.shape[1] - model.seqlen
                full_ids = full_ids[:, overshoot:]
                adj_len = max(1, prompt_len - overshoot)
            else:
                adj_len = prompt_len
            ll, ntok = compute_completion_ll(model, full_ids, adj_len)
            choices.append({"ll": ll, "ntok": ntok,
                            "score": ll / ntok if ntok > 0 else float("-inf")})
        pred_idx = max(range(len(choices)),
                       key=lambda j: choices[j]["score"])
        ok = int(pred_idx == gt_idx)
        correct += ok
        total += 1
        records.append({"i": i, "id": ex.get("id", str(i)),
                        "gt": gt_idx, "pred": pred_idx, "correct": ok,
                        "choices": choices})
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(ds)}] acc={correct/total:.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    return records, correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--restore", default="none")
    ap.add_argument("--split", default="ARC-Easy")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fp-blocks", default=None,
                    help="comma list of block indices to KEEP FP16")
    ap.add_argument("--q-blocks", default=None,
                    help="comma list: quantize ONLY these blocks")
    ap.add_argument("--fp-sublayer-pat", default=None,
                    help="substring: sublayers matching it KEEP FP16 "
                         "(e.g. 'mlp.down_proj')")
    ap.add_argument("--fp-cols", default=None,
                    help="'topk:<k>' or 'rand:<k>:<seed>' — per down_proj, "
                         "keep k input columns FP16 (topk = by wt2-calib "
                         "Ex2 = sink columns; rand = bystander control)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fp_blocks = ([int(x) for x in args.fp_blocks.split(",")]
                 if args.fp_blocks else None)
    q_blocks = ([int(x) for x in args.q_blocks.split(",")]
                if args.q_blocks else None)
    device = torch.device(args.device)
    print(f"Loading {args.model} ...", flush=True)
    model = load_model(args.model)
    n = apply_restore(model, args.restore, fp_blocks, q_blocks,
                      args.fp_sublayer_pat, args.fp_cols)
    print(f"restore={args.restore} fp_blocks={fp_blocks} "
          f"q_blocks={q_blocks} fp_pat={args.fp_sublayer_pat} "
          f"fp_cols={args.fp_cols}: {n} tensors loaded", flush=True)
    model = model.to(device)

    records, correct, total = probe(model, args.model, device, args.split,
                                    args.limit)
    acc = correct / total if total else 0.0
    print(f"FINAL {args.split} accuracy: {acc:.6f} ({correct}/{total})",
          flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"model": args.model, "restore": args.restore,
                            "fp_blocks": fp_blocks, "q_blocks": q_blocks,
                            "fp_sublayer_pat": args.fp_sublayer_pat,
                            "fp_cols": args.fp_cols,
                            "split": args.split, "accuracy": acc,
                            "correct": correct, "total": total},
                   "records": records}, f)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
