"""
Phase 5 Diagnostic: Does optimal scalar quantization (Lloyd-Max) beat binary
residual within GPTQ? Tests whether the representation ({sign*scale} vs optimal
4-level) is the bottleneck, or GPTQ itself is.

Uses the same GPTQ column-wise error correction but replaces BRAQ's binary
residual with Lloyd-Max 4-level quantization (optimal non-uniform scalar
quantization at 2 bits per weight).
"""

import sys, os, argparse, json, time, math, gc
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ["HF_HOME"] = "./downloads"

from datautils import get_loaders, set_seed
from modelutils import find_layers
from eval_ppl_utils import qwen_eval

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda:0"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def lloydmax_quantize_column(w, nbits=2, n_iter=20):
    """Quantize a column vector using Lloyd-Max (k-means) optimal levels.

    Args:
        w: (m,) weight column
        nbits: bits per element
        n_iter: k-means iterations

    Returns: quantized column (m,)
    """
    K = 2 ** nbits  # number of levels
    m = w.shape[0]
    if m == 0:
        return w

    # Initialize centroids from evenly-spaced percentiles
    sorted_w, _ = w.sort()
    idx = torch.linspace(0, m - 1, K + 2)[1:-1].long().clamp(0, m - 1)
    centroids = sorted_w[idx].clone()

    for _ in range(n_iter):
        # Assign each weight to nearest centroid
        dists = (w.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # (m, K)
        assignments = dists.argmin(dim=1)  # (m,)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = w[mask].mean()
            else:
                new_centroids[k] = centroids[k]

        if (new_centroids - centroids).abs().max() < 1e-7:
            break
        centroids = new_centroids

    # Final assignment
    dists = (w.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    assignments = dists.argmin(dim=1)
    return centroids[assignments]


@torch.no_grad()
def gptq_lloydmax_quantize_matrix(W, H, nbits=2, blocksize=128, percdamp=0.01):
    """GPTQ with Lloyd-Max quantization instead of BRAQ.

    W: (m, k) weight matrix
    H: (k, k) input second-moment matrix

    Returns: quantized W (m, k), info dict
    """
    m, k = W.shape
    W = W.float().clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    Losses = torch.zeros(m, device=W.device)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(k, device=W.device)
    H[diag, diag] += damp

    # Cholesky with retry
    for _ in range(10):
        try:
            H_chol = torch.linalg.cholesky(H)
            break
        except torch._C._LinAlgError:
            H[diag, diag] += 1e-3 * torch.mean(torch.diag(H))
    else:
        H_chol = torch.diag(torch.sqrt(torch.diag(H).clamp(min=1e-8)))

    Hinv = torch.cholesky_inverse(H_chol)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    for col_st in range(0, k, blocksize):
        col_ed = min(col_st + blocksize, k)

        W1 = W[:, col_st:col_ed].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

        for i in range(col_ed - col_st):
            w = W1[:, i]
            d = Hinv1[i, i]

            # Lloyd-Max quantization of this column
            q = lloydmax_quantize_column(w, nbits=nbits)

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            Err1[:, i] = err1

        W[:, col_st:col_ed] = Q1
        Losses += torch.sum(Losses1, 1) / 2
        W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

    error = torch.sum(Losses).item()
    return W, {"error": error}


def get_model(model_name):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", cache_dir="./downloads",
        attn_implementation="eager"
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    return model


@torch.no_grad()
def lloydmax_sequential(model, dataloader, dev, nbits=2):
    """Quantize all layers with GPTQ + Lloyd-Max."""
    print(f"GPTQ + Lloyd-Max quantization: {nbits} bits")

    model.config.use_cache = False
    layers = model.model.layers
    nsamples = len(dataloader)

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    for layer_idx in range(len(layers)):
        t0 = time.time()
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)

        # Collect Hessians
        hessians = {}
        nsamples_count = {}
        for name in subset:
            ncols = subset[name].weight.shape[1]
            hessians[name] = torch.zeros((ncols, ncols), device=dev, dtype=torch.float32)
            nsamples_count[name] = 0

        def make_hook(name):
            def hook_fn(_, inp, out):
                x = inp[0].data
                if len(x.shape) == 3:
                    x = x.reshape(-1, x.shape[-1])
                x = x.float()
                n = x.shape[0]
                hessians[name] *= nsamples_count[name] / (nsamples_count[name] + n)
                nsamples_count[name] += n
                x = math.sqrt(2.0 / nsamples_count[name]) * x.t()
                hessians[name] += x @ x.t()
            return hook_fn

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook(name)))
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        # Quantize each sublayer
        for name in subset:
            W = subset[name].weight.data.float()
            H = hessians[name]
            Q, info = gptq_lloydmax_quantize_matrix(W, H, nbits=nbits)
            subset[name].weight.data = Q.to(dtype)
            print(f"  L{layer_idx} {name}: error={info['error']:.4e}")

        # Propagate
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[layer_idx] = layer.cpu()
        del layer, hessians
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(f"  Layer {layer_idx} done in {time.time()-t0:.1f}s")

    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nbits", type=int, default=2)
    parser.add_argument("--nsamples", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Loading model: {MODEL_NAME}")
    model = get_model(MODEL_NAME)
    model.eval()

    print("Loading calibration data...")
    dataloader, testdata = get_loaders(
        "wikitext2", nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, model=MODEL_NAME
    )

    print(f"\nGPTQ + Lloyd-Max ({args.nbits}-bit, seed={args.seed})...")
    lloydmax_sequential(model, dataloader, DEVICE, nbits=args.nbits)

    print("\nEvaluating perplexity...")
    ppl = qwen_eval(
        model, testdata, DEVICE, "wikitext2",
        save_title=f"LloydMax_GPTQ_{args.nbits}bit_seed{args.seed}",
        save=True
    )

    os.makedirs("results", exist_ok=True)
    result = {
        "model": MODEL_NAME, "method": f"LloydMax_GPTQ_{args.nbits}bit",
        "nbits": args.nbits, "seed": args.seed, "ppl": ppl,
        "fp16_ppl": 20.97, "degradation_ratio": ppl / 20.97,
    }
    outpath = f"results/lloydmax_gptq_{args.nbits}bit_seed{args.seed}.json"
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nPPL: {ppl:.2f} (degradation: {ppl/20.97:.1f}x)")


if __name__ == "__main__":
    main()
