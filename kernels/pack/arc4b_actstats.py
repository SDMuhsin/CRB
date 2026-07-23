"""ARC-4B RCA — per-sublayer input-activation statistics capture.

For every Linear sublayer in every transformer block of the FP16 model,
accumulate per-INPUT-channel statistics over a token stream:
    n        : token count (scalar)
    sum_x    : per-channel sum of activations          (C,)
    sum_x2   : per-channel sum of squared activations  (C,)
    max_abs  : per-channel max |x|                     (C,)

Two streams are supported:
  * wt2calib — the EXACT GPTQ calibration stream (wikitext2, nsamples=128,
    seed=0, seqlen=2048), i.e. what DOML's Hessian saw.
  * arceasy  — the ARC-Easy test sequences exactly as the eval harness feeds
    them (prompt + " " + choice, per choice), i.e. what the downstream task
    actually runs through the model.

The comparison of the two streams is itself evidence (calibration/task
activation shift); sum_x2/n is the diag(H)-proxy used by the cell-level error
analysis (arc4b_cell_errors.py).

Output: <out>/actstats_<stream>.pt — dict layer_name -> {n, sum_x, sum_x2,
max_abs} (fp64 CPU tensors), plus a meta entry.

Usage:
    python kernels/pack/arc4b_actstats.py --model Qwen/Qwen3-4B \
        --device cuda:1 --out downloads/arc4b_rca/actstats \
        --streams wt2calib,arceasy
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def load_model(model_name):
    os.chdir(REPO)
    import run as run_mod  # argparse is __main__-guarded
    model = run_mod.get_model(model_name)
    model.eval()
    model.config.use_cache = False
    return model


class ActStats:
    def __init__(self, name, C, device):
        self.name = name
        self.n = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_x = torch.zeros(C, dtype=torch.float64, device=device)
        self.sum_x2 = torch.zeros(C, dtype=torch.float64, device=device)
        self.max_abs = torch.zeros(C, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, x):
        # x: (..., C) activation tensor
        xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
        self.n += xf.shape[0]
        self.sum_x += xf.sum(dim=0).to(torch.float64)
        self.sum_x2 += xf.pow(2).sum(dim=0).to(torch.float64)
        self.max_abs = torch.maximum(self.max_abs,
                                     xf.abs().amax(dim=0).to(torch.float64))

    def to_cpu_dict(self):
        return {"n": self.n.cpu(), "sum_x": self.sum_x.cpu(),
                "sum_x2": self.sum_x2.cpu(), "max_abs": self.max_abs.cpu()}


def register_hooks(model):
    stats, handles = {}, []
    dev = next(model.parameters()).device
    for li, block in enumerate(model.model.layers):
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                full = f"model.layers.{li}.{name}"
                st = ActStats(full, mod.in_features, dev)
                stats[full] = st

                def hook(module, inputs, _st=st):
                    _st.update(inputs[0])

                handles.append(mod.register_forward_pre_hook(hook))
    return stats, handles


@torch.no_grad()
def stream_wt2calib(model, model_name, device, nsamples):
    from datautils import get_loaders
    dataloader, _ = get_loaders("wikitext2", nsamples=nsamples, seed=0,
                                model=model_name, seqlen=model.seqlen)
    assert len(dataloader) == nsamples, len(dataloader)
    t0 = time.time()
    for i, batch in enumerate(dataloader):
        model(batch[0].to(device))
        if (i + 1) % 16 == 0:
            print(f"  wt2calib [{i+1}/{nsamples}] t={time.time()-t0:.0f}s",
                  flush=True)
    return {"stream": "wt2calib", "nsamples": nsamples,
            "seqlen": model.seqlen, "seed": 0}


@torch.no_grad()
def stream_arceasy(model, model_name, device):
    """Feed the exact sequences the ARC-Easy eval harness scores: for each
    test question, prompt + " " + choice for every choice (eval_arc.py
    formatting, same tokenizer path)."""
    from datasets import load_dataset
    from datautils import get_tokenizer
    cache = os.path.join(os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads"),
                         "datasets")
    tokenizer = get_tokenizer(model_name)
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test",
                      cache_dir=cache)
    t0, n_seq = time.time(), 0
    for i, ex in enumerate(ds):
        prompt = f"Question: {ex['question']}\nAnswer:"
        for choice_text in ex["choices"]["text"]:
            full_ids = tokenizer(prompt + " " + choice_text,
                                 return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] > model.seqlen:
                full_ids = full_ids[:, -model.seqlen:]
            model(full_ids)
            n_seq += 1
        if (i + 1) % 250 == 0:
            print(f"  arceasy [{i+1}/{len(ds)}] seqs={n_seq} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    return {"stream": "arceasy", "n_questions": len(ds), "n_seqs": n_seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", default="wt2calib,arceasy")
    ap.add_argument("--nsamples", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    print(f"Loading {args.model} ...", flush=True)
    model = load_model(args.model).to(device)

    for stream in args.streams.split(","):
        stream = stream.strip()
        stats, handles = register_hooks(model)
        print(f"=== stream {stream}: {len(stats)} sublayers hooked ===",
              flush=True)
        if stream == "wt2calib":
            meta = stream_wt2calib(model, args.model, device, args.nsamples)
        elif stream == "arceasy":
            meta = stream_arceasy(model, args.model, device)
        else:
            raise SystemExit(f"unknown stream {stream}")
        for h in handles:
            h.remove()
        out = {name: st.to_cpu_dict() for name, st in stats.items()}
        out["__meta__"] = {**meta, "model": args.model}
        path = os.path.join(args.out, f"actstats_{stream}.pt")
        torch.save(out, path)
        n_tok = next(iter(stats.values())).n.item()
        print(f"saved {path}  (tokens per sublayer ≈ {n_tok:.0f})", flush=True)


if __name__ == "__main__":
    main()
