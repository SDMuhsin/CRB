"""
LeanQuant_nu (ICLR 2025) standalone benchmark runner.

Faithful reproduction of "LeanQuant: Accurate and Scalable Large Language Model
Quantization with Loss-error-aware Grid" (Zhang & Shrivastava, ICLR 2025,
arXiv:2407.10032). Implements Algorithm 1 of the paper:
  1. Per-row weighted k-means over exponentiated inverse Hessian diagonals
     (sample_weight[i] = diag(Hinv)[i] ** (-p), p=4 by default) to learn
     a loss-error-aware non-uniform grid G_k of size 2^b per output row.
  2. Block-wise GPTQ: for each column j in a block, quantize w_{k,j} to the
     nearest centroid in G_k, compute the scaled error e = (w - q) / Hinv[j,j],
     and propagate it to all remaining columns via the upper-Cholesky of
     Hinv. Inter-block propagation via Err @ Hinv[i1:i2, i2:].

Paper calibration (Section E.Experiment Details, reproducing Table 7's
LLaMA-2-7B 2-bit = 15.51 PPL at 2.01 bpw):
  * 128 sequences of 2048 tokens from C4
  * p = 4
  * true_sequential + act_order + percdamp = 0.1 (README's 2-bit recipe)

The quantization algorithm itself is ported from the upstream
`/tmp/leanquant_source/lean_quantizer.py` (the `LeanQuant` class and its
`fasterquant` method). The only substantive deviation is the k-means
implementation: upstream uses `sklearn.cluster.KMeans` (CPU) inside a
`multiprocessing.Pool` across rows, which has prohibitive fork+IPC overhead
after the model is resident in memory (measured: 115 s per sublayer on
Qwen3-0.6B). We instead run Lloyd's algorithm directly on GPU, fully
vectorised across all rows of a weight matrix, with the paper's uniform-
linspace initialization (Table 15 ablation — uniform beats k-means++).
Semantics are identical: same objective (weighted Euclidean with
sample_weight = diag(Hinv)^(-p)), same init, same convergence test,
same max_iter=100, tol=1e-6.

Usage:
    source env/bin/activate
    python3 -u src/run_leanquant.py Qwen/Qwen3-0.6B wikitext2 \\
        --nbits 2 --exponent 4.0 --percdamp 0.1 \\
        --true_sequential --act_order \\
        --calib_dataset redpajama --nsamples 128 --seqlen 2048 \\
        --device cuda:0 --seed 0
"""

import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datautils import get_loaders, set_seed
from eval_ppl_utils import llama_eval, opt_eval, qwen_eval
from modelutils import find_layers


# =====================================================================
# LeanQuant per-row weighted k-means (GPU port of lean_quantizer.py kmeans_fit)
# =====================================================================


@torch.no_grad()
def leanquant_row_kmeans(W, sample_weight, K, max_iter=100, tol=1e-6):
    """Per-row weighted 1D k-means with uniform-linspace init, on GPU.

    Procedurally identical to upstream `kmeans_fit` in
    `/tmp/leanquant_source/lean_quantizer.py:22-32` but vectorised across
    rows on GPU instead of sklearn+multiprocessing. Init uses uniform
    linspace between row-wise min/max (paper's "uniformly spaced grid
    initialization", Appendix A / Table 15 — outperforms k-means++ at
    low bit-width). Lloyd's algorithm, weighted by sample_weight (shared
    across all rows: sample_weight[i] = diag(Hinv)[i]^(-p), so the same
    weight profile applies to every row's clustering).

    Args:
        W: (rows, cols) float32 tensor on GPU.
        sample_weight: (cols,) float32 tensor on GPU — non-negative weights.
        K: number of clusters (2^nbits).
        max_iter: Lloyd's iteration cap (sklearn default 100).
        tol: relative inertia change convergence threshold (sklearn default 1e-4;
             we match LeanQuant upstream at 1e-6).

    Returns:
        centroids: (rows, K) float32 tensor on GPU, sorted ascending per row.
    """
    device = W.device
    rows, cols = W.shape

    # Uniform linspace init per row: K points from min(w) to max(w)
    W_min = W.min(dim=1, keepdim=True).values  # (rows, 1)
    W_max = W.max(dim=1, keepdim=True).values  # (rows, 1)
    t = torch.linspace(0.0, 1.0, K, device=device, dtype=W.dtype)  # (K,)
    centroids = W_min + (W_max - W_min) * t.unsqueeze(0)  # (rows, K)

    # Handle degenerate rows where min==max (all weights equal): jitter so
    # Lloyd's doesn't collapse to a single centroid repeated K times.
    equal_mask = (W_max - W_min).abs() < 1e-30
    if equal_mask.any():
        eps = torch.linspace(0.0, 1e-6, K, device=device, dtype=W.dtype)
        centroids = torch.where(
            equal_mask, W_min + eps.unsqueeze(0), centroids
        )

    sw = sample_weight.to(W.dtype).view(1, cols)  # (1, cols)

    prev_inertia = None
    for _ in range(max_iter):
        # Assignment: distance of each point to each centroid
        # |W[r,c] - centroids[r,k]| -> (rows, cols, K)
        dist = (W.unsqueeze(2) - centroids.unsqueeze(1)).abs()
        labels = dist.argmin(dim=2)  # (rows, cols)

        # Weighted mean per cluster per row via one-hot reduction
        one_hot = torch.nn.functional.one_hot(labels, K).to(W.dtype)  # (rows, cols, K)
        weighted = one_hot * sw.unsqueeze(2)  # (rows, cols, K)
        counts = weighted.sum(dim=1)  # (rows, K) -- sum of sample_weights in cluster
        sums = (weighted * W.unsqueeze(2)).sum(dim=1)  # (rows, K)

        new_centroids = sums / counts.clamp(min=1e-30)
        empty = counts <= 0
        new_centroids = torch.where(empty, centroids, new_centroids)

        # Weighted squared inertia for convergence test (sklearn uses
        # squared euclidean sum; match that semantically)
        assigned_dist = dist.gather(2, labels.unsqueeze(2)).squeeze(2)  # (rows, cols)
        inertia = (assigned_dist.pow(2) * sw).sum()

        if prev_inertia is not None:
            denom = prev_inertia.abs() + 1e-30
            if (prev_inertia - inertia).abs() / denom < tol:
                centroids = new_centroids
                break

        centroids = new_centroids
        prev_inertia = inertia

    # Return centroids sorted ascending per row — makes downstream argmin
    # deterministic and matches what sklearn's KMeans returns after sorting
    # (sklearn returns unordered, but LeanQuant later uses argmin over centroids
    # which is order-invariant. Sorting is just cosmetic / reproducibility).
    centroids, _ = centroids.sort(dim=1)
    return centroids


# =====================================================================
# LeanQuant per-layer quantizer (Algorithm 1)
# =====================================================================


class LeanQuantLayer:
    """Accumulates H = (2/N) X^T X via a forward hook, then runs Algorithm 1."""

    def __init__(self, layer, device):
        self.layer = layer
        self.dev = device
        self.rows = layer.weight.shape[0]
        self.columns = layer.weight.shape[1]
        self.H = torch.zeros(
            (self.columns, self.columns), device=device, dtype=torch.float32
        )
        self.nsamples = 0

    def add_batch(self, inp, out):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        nbits,
        exponent,
        percdamp,
        blocksize,
        actorder,
        groupsize=-1,
    ):
        """Return (Q, bpw, quant_time) where Q is the quantized weight tensor.

        Ported from upstream lean_quantizer.py:80-215 ('fasterquant' method)
        with the non-uniform code path (isinstance(exponent, float) branch).
        Multi-group static_groups is not implemented (not used at 2-bit with
        groupsize=-1, which is the paper's default per Table 7).
        """
        assert groupsize == -1, "Only per-row (groupsize=-1) is supported"

        W = self.layer.weight.data.clone().float().to(self.dev)
        rows, cols = W.shape
        K = 2**nbits

        tick = time.time()

        H = self.H
        self.H = None

        # Mask dead columns (zero Hessian diagonal entries)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm_H = torch.argsort(torch.diag(H), descending=True)
            perm = perm_H.to(W.device)
            W = W[:, perm]
            H = H[perm_H][:, perm_H]
            invperm = torch.argsort(perm)
        else:
            invperm = None

        # Dampening + Cholesky inverse + upper Cholesky (standard GPTQ trick)
        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(cols, device=H.device)
        H[diag_idx, diag_idx] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H  # (cols, cols), upper-triangular Cholesky of H_inv

        # Non-uniform loss-error-aware grid: per-row weighted k-means with
        # sample_weight = diag(Hinv) ** (-exponent). Paper Section 3.2.1 eq (5).
        sample_weight = torch.diagonal(Hinv) ** (-float(exponent))
        # Guard: (Hinv_diag)^{-p} blows up to inf for tiny diag entries. Clamp to
        # a large finite value so the downstream weighted mean stays defined.
        sample_weight = torch.nan_to_num(sample_weight, nan=0.0, posinf=1e30, neginf=0.0)
        centroids = leanquant_row_kmeans(W, sample_weight, K)

        # Block-wise GPTQ with nearest-centroid assignment (Algorithm 1 lines 10-20)
        Q = torch.zeros_like(W)
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                codes = torch.argmin(
                    (centroids - w[:, None]).abs(), dim=1, keepdim=True
                )
                q = torch.gather(centroids, 1, codes).flatten()
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        quant_time = time.time() - tick

        # BPW: 2 bits per weight index + 2^b * 16 bits codebook per row
        bpw = nbits + K * 16.0 / cols

        del Hinv, H, W
        torch.cuda.empty_cache()

        return Q, bpw, quant_time

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


# =====================================================================
# Model loading (mirrors src/run_lnq.py)
# =====================================================================


def get_model(model_name):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    downloads_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")

    if "opt" in model_name.lower():
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=downloads_dir,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model_name.lower():
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=downloads_dir,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model.seqlen = 2048
    elif "mistral" in model_name.lower():
        # Mistral has the same decoder layout as LLaMA (RMSNorm / rotary_emb /
        # GQA / SwiGLU) so the LeanQuant pipeline and llama_eval both apply.
        from transformers import MistralForCausalLM

        model = MistralForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=downloads_dir,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model.seqlen = 2048
    elif "qwen" in model_name.lower():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=downloads_dir,
            attn_implementation="eager",
        )
        model.seqlen = min(model.config.max_position_embeddings, 2048)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    return model


def detect_model_type(model):
    class_name = model.__class__.__name__.lower()
    if "opt" in class_name:
        return "opt"
    # Mistral shares LLaMA's decoder-only layout (embed_tokens/norm/rotary_emb +
    # q/k/v/o + up/gate/down), and llama_eval handles the PPL pass correctly.
    if "llama" in class_name or "mistral" in class_name:
        return "llama"
    if "qwen" in class_name:
        return "qwen"
    raise ValueError(f"Unknown model class: {model.__class__.__name__}")


# =====================================================================
# LeanQuant pipeline: forward + hook, then quantize each sublayer
# =====================================================================


def _true_sequential_groups(model_type):
    """Sublayer ordering for true_sequential, matching upstream llama.py:81-87.

    Qwen3 and LLaMA share attention / MLP submodule naming so the same groups
    apply to both. OPT uses a flat order.
    """
    if model_type in ("llama", "qwen"):
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    # OPT (rarely used here, but keep the path for completeness)
    return [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"],
    ]


@torch.no_grad()
def leanquant_quantize_model(model, args, calib_dataset, calib_seqlen):
    """Sequentially quantize every transformer layer with LeanQuant_nu."""
    dev = torch.device(args.device)
    nsamples = args.nsamples

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # Calibration data load (reuses the project's shared tokenizer cache + flock)
    dataloader, _ = get_loaders(
        calib_dataset,
        nsamples=nsamples,
        seed=args.seed,
        seqlen=calib_seqlen,
        model=args.model,
    )

    model.config.use_cache = False
    model_type = detect_model_type(model)

    # Move embedding + norm + (rotary) to GPU for the capture phase
    if model_type == "opt":
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif model_type in ("llama", "qwen"):
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        if hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype

    # CPU activation offload for large calib runs (1024 x 4096 would be >30 GB)
    act_bytes = nsamples * calib_seqlen * model.config.hidden_size * 2
    act_device = "cpu" if act_bytes > 8 * (1024**3) else dev
    if act_device == "cpu":
        print(
            f"Activation offload: {act_bytes / 1024**3:.1f} GB > 8 GB threshold, "
            f"using CPU"
        )
    inps = torch.zeros(
        (nsamples, calib_seqlen, model.config.hidden_size),
        dtype=dtype,
        device=act_device,
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
            inps[cache["i"]] = inp.to(act_device)
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

    # Free embedding/norm from GPU
    layers[0] = layers[0].cpu()
    if model_type == "opt":
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = (
            model.model.decoder.embed_positions.cpu()
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif model_type in ("llama", "qwen"):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    # transformers>=5 still passes DynamicCache even with use_cache=False;
    # strip it to avoid stale-state shenanigans on repeated forward passes.
    if "past_key_values" in layer_kwargs:
        layer_kwargs["past_key_values"] = None

    sequential_groups = (
        _true_sequential_groups(model_type)
        if args.true_sequential
        else None
    )

    total_quant_params = 0
    total_codebook_bits = 0
    total_index_bits = 0
    quant_start = time.time()

    print(
        f"\nQuantizing {len(layers)} layers with LeanQuant_nu "
        f"(K={2**args.nbits}, {args.nbits}-bit)..."
    )
    print(
        f"  exponent p = {args.exponent}, percdamp = {args.percdamp}, "
        f"blocksize = {args.blocksize}, act_order = {args.act_order}, "
        f"true_sequential = {args.true_sequential}, propagate = "
        f"{'off' if args.no_propagate else 'on'}"
    )

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)
        full = find_layers(layer)

        if sequential_groups is not None:
            groups = [
                [n for n in names if n in full] for names in sequential_groups
            ]
            # Drop empty groups and warn about any stray sublayers
            groups = [g for g in groups if g]
            seen = {n for g in groups for n in g}
            leftovers = [n for n in full if n not in seen]
            if leftovers:
                groups.append(leftovers)
        else:
            groups = [list(full.keys())]

        print(f"\nLayer {layer_idx}/{len(layers)-1} -- {len(full)} sublayers")

        for names in groups:
            subset = {n: full[n] for n in names}

            # Create one LeanQuantLayer per sublayer; hook H accumulation
            lq = {n: LeanQuantLayer(subset[n], dev) for n in names}

            def add_batch(name):
                def tmp(_, inp, out):
                    lq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for n in names:
                handles.append(subset[n].register_forward_hook(add_batch(n)))
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            # Quantize each sublayer in this group
            for n in names:
                print(
                    f"  {n} [{subset[n].weight.shape[0]}x{subset[n].weight.shape[1]}]"
                )
                Q, bpw, qt = lq[n].fasterquant(
                    nbits=args.nbits,
                    exponent=args.exponent,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                    actorder=args.act_order,
                    groupsize=args.groupsize,
                )
                subset[n].weight.data = Q.reshape(subset[n].weight.shape).to(dtype)
                lq[n].free()

                rows = subset[n].weight.shape[0]
                cols = subset[n].weight.shape[1]
                total_quant_params += rows * cols
                total_index_bits += rows * cols * args.nbits
                total_codebook_bits += rows * (2**args.nbits) * 16
                print(
                    f"    bpw={bpw:.4f}  time={qt:.1f}s"
                )

        # Re-forward with quantized sublayers to feed inps[next_layer]
        if not args.no_propagate:
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]

        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    quant_time = time.time() - quant_start
    bpw = (total_index_bits + total_codebook_bits) / total_quant_params

    print(f"\nLeanQuant_nu quantization complete in {quant_time:.1f}s")
    print(f"  Effective BPW: {bpw:.4f}")
    print(f"    index bits:    {total_index_bits:,} ({args.nbits} per weight)")
    print(
        f"    codebook bits: {total_codebook_bits:,} "
        f"({2**args.nbits} entries × 16b per row)"
    )
    print(f"    total params:  {total_quant_params:,}")

    model.config.use_cache = True
    return model, bpw, quant_time


# =====================================================================
# Evaluation + CSV
# =====================================================================


def evaluate_ppl(model, testenc, args):
    dev = torch.device(args.device)
    model_type = detect_model_type(model)
    model_short = args.model.split("/")[-1]
    save_title = (
        f"leanquant_nu_{args.nbits}bit_{model_short}_{args.dataset}_seed{args.seed}"
    )
    if model_type == "opt":
        return opt_eval(model, testenc, dev, args.dataset, save_title=save_title)
    if model_type == "llama":
        return llama_eval(model, testenc, dev, args.dataset, save_title=save_title)
    return qwen_eval(model, testenc, dev, args.dataset, save_title=save_title)


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LeanQuant_nu (ICLR 2025) standalone benchmark runner"
    )
    parser.add_argument("model", type=str, help="HuggingFace model name")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "c4"],
        help="Evaluation dataset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Core LeanQuant hyperparameters
    parser.add_argument(
        "--nbits", type=int, default=2, choices=[2, 3, 4],
        help="Bits per weight (K = 2^nbits levels per row)",
    )
    parser.add_argument(
        "--exponent", type=float, default=4.0,
        help="Paper parameter p in sample_weight = diag(Hinv)^(-p) (paper: 4.0)",
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.1,
        help="GPTQ dampening: 0.01 (1%% avg diag) or 0.1 (README 2-bit default)",
    )
    parser.add_argument(
        "--blocksize", type=int, default=128,
        help="GPTQ block size for column-wise error propagation",
    )
    parser.add_argument(
        "--groupsize", type=int, default=-1,
        help="-1 for per-row (paper default for Table 7)",
    )
    parser.add_argument(
        "--act_order", action="store_true",
        help="Enable activation-order heuristic (paper README recommends on)",
    )
    parser.add_argument(
        "--true_sequential", action="store_true",
        help="Quantize k+v+q, o, up+gate, down sequentially (paper README recommends on)",
    )
    parser.add_argument(
        "--no_propagate", action="store_true",
        help="Skip output propagation between layers (off by default; "
             "upstream LeanQuant propagates, matching GPTQ).",
    )

    # Calibration
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument(
        "--calib_dataset", type=str, default=None,
        choices=["wikitext2", "c4", "redpajama"],
        help="Calibration dataset (default: same as eval dataset)",
    )
    parser.add_argument(
        "--seqlen", type=int, default=None,
        help="Calibration sequence length (default: model.seqlen)",
    )

    # Downstream eval
    parser.add_argument("--eval_arc", action="store_true")
    parser.add_argument("--eval_mmlu", action="store_true")
    parser.add_argument("--eval_hellaswag", action="store_true")

    args = parser.parse_args()

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
        ),
    )
    from csv_utils import append_result as csv_append

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    model = get_model(args.model)
    model.config.use_cache = False

    calib_dataset = args.calib_dataset if args.calib_dataset else args.dataset
    calib_seqlen = args.seqlen if args.seqlen else model.seqlen

    print(f"  Eval dataset: {args.dataset} (seqlen={model.seqlen})")
    print(
        f"  Calibration: {calib_dataset} (seqlen={calib_seqlen}, "
        f"nsamples={args.nsamples})"
    )

    _, testenc = get_loaders(
        args.dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
    )

    model, bpw, quant_time = leanquant_quantize_model(
        model, args,
        calib_dataset=calib_dataset,
        calib_seqlen=calib_seqlen,
    )

    extra = {
        "nbits": args.nbits,
        "exponent": args.exponent,
        "percdamp": args.percdamp,
        "blocksize": args.blocksize,
        "groupsize": args.groupsize,
        "act_order": args.act_order,
        "true_sequential": args.true_sequential,
        "no_propagate": args.no_propagate,
        "nsamples": args.nsamples,
        "calib_dataset": calib_dataset,
        "calib_seqlen": calib_seqlen,
    }
    method = "leanquant_nu"

    def _csv(dataset, metric, value):
        csv_append(
            model=args.model,
            method=method,
            dataset=dataset,
            metric=metric,
            value=value,
            bpw=bpw,
            seed=args.seed,
            blocksize=args.blocksize,
            salient_metric="",
            extra_params=extra,
            quantization_time_s=quant_time,
        )

    print(f"\n{'='*60}")
    print(f"Evaluating perplexity on {args.dataset}...")
    print(f"{'='*60}")
    ppl = evaluate_ppl(model, testenc, args)
    _csv(args.dataset, "perplexity", ppl)

    model_short = args.model.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"RESULT: leanquant_nu {args.nbits}-bit on {model_short}")
    print(f"  {args.dataset} PPL: {ppl:.2f}")
    print(f"  Seed: {args.seed}")
    print(f"  Effective bpw: {bpw:.4f}")
    print(f"  Quantization time: {quant_time:.1f}s")
    print(
        f"  Calibration: {calib_dataset} (nsamples={args.nsamples}, "
        f"seqlen={calib_seqlen})"
    )
    print(f"  Propagation: {'off' if args.no_propagate else 'on'}")
    print(f"{'='*60}")

    dev = torch.device(args.device)

    if args.eval_mmlu:
        from eval_mmlu import eval_mmlu

        mmlu_acc = eval_mmlu(model, args.model, dev)
        _csv(args.dataset, "mmlu_acc", mmlu_acc)

    if args.eval_hellaswag:
        from eval_hellaswag import eval_hellaswag

        hellaswag_acc = eval_hellaswag(model, args.model, dev)
        _csv(args.dataset, "hellaswag_acc", hellaswag_acc)

    if args.eval_arc:
        from eval_arc import eval_arc

        arc_results = eval_arc(model, args.model, dev)
        _csv(args.dataset, "arc_easy_acc", arc_results["ARC-Easy"]["accuracy"])
        _csv(
            args.dataset,
            "arc_challenge_acc",
            arc_results["ARC-Challenge"]["accuracy"],
        )

    return ppl


if __name__ == "__main__":
    main()
