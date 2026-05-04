"""
SparseGPT (ICML 2023) standalone benchmark runner.

Faithful port of "SparseGPT: Massive Language Models Can Be Accurately Pruned
in One-Shot" (Frantar & Alistarh, ICML 2023, arXiv:2301.00774,
github.com/IST-DASLab/sparsegpt). Implements the joint pruning + (optional)
quantization algorithm:

  1. Per-sublayer Hessian H = (2/N) X^T X via forward hooks during a calibration
     pass.
  2. Dampened Cholesky inverse, then upper-Cholesky factor (the standard GPTQ
     trick) -> Hinv.
  3. Block-wise OBS sweep (blocksize=128 columns):
       a. Compute keep-mask once per block from W^2 / Hinv_diag^2, threshold
          at the desired sparsity.
       b. For each column in the block: zero out pruned positions, optionally
          quantize kept positions through an affine uniform Quantizer (per-row
          scale + zero, found pre-block from the original W), then propagate
          the column residual to all remaining columns via Hinv.
     Inter-block error propagation via Err @ Hinv[i1:i2, i2:].
  4. Optional structured N:M pruning (--prunen N --prunem M) replaces the
     block-mask with a per-M-column top-k inside the sweep.

Algorithm is ported from /tmp/sparsegpt/sparsegpt.py (the SparseGPT class and
its fasterprune method) and /tmp/sparsegpt/quant.py (the Quantizer class).
The driver mirrors /tmp/sparsegpt/llama.py:llama_sequential structurally, but
is written against the BiLLM2 harness so it shares dataset loading, offline
caching, the multi-task eval pipeline (Phase 15), and CSV output with the
rest of the project's runners.

Usage:
    source env/bin/activate

    # Pure 50% unstructured pruning at fp16 (paper Table 1 setting):
    python3 -u src/run_sparsegpt.py Qwen/Qwen3-0.6B wikitext2 \\
        --sparsity 0.5 --percdamp 0.01 --true_sequential \\
        --calib_dataset c4 --nsamples 128 --seqlen 2048 \\
        --device cuda:0 --seed 0

    # Joint 50% sparse + 4-bit (paper Table 4 setting):
    python3 -u src/run_sparsegpt.py Qwen/Qwen3-0.6B wikitext2 \\
        --sparsity 0.5 --nbits 4 --percdamp 0.01 --true_sequential \\
        --calib_dataset c4 --nsamples 128 --seqlen 2048 \\
        --device cuda:0 --seed 0

    # 2:4 structured + 2-bit (most aggressive joint setting):
    python3 -u src/run_sparsegpt.py Qwen/Qwen3-0.6B wikitext2 \\
        --prunen 2 --prunem 4 --nbits 2 --percdamp 0.01 --true_sequential \\
        --calib_dataset c4 --nsamples 128 --seqlen 2048 \\
        --device cuda:0 --seed 0
"""

import argparse
import gc
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datautils import get_loaders, set_seed
from modelutils import find_layers


# =====================================================================
# Affine uniform quantizer (port of /tmp/sparsegpt/quant.py)
# =====================================================================


def _quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    """Affine uniform quantizer with per-channel scale/zero, asymmetric mode.

    Upstream LLaMA driver (llama.py:103-106) configures with
    `bits, perchannel=True, sym=False, mse=False` — that is the configuration
    we mirror by default (see `Quantizer.configure(...)` defaults).
    """

    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits, perchannel=True, sym=False, mse=False, norm=2.4,
                  grid=100, maxshrink=0.8, grouprows=1):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.grouprows = grouprows

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.grouprows > 1:
                    x = x.reshape((x.shape[0] // self.grouprows, -1))
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = (
                    torch.round(-xmin1 / scale1) if not self.sym else self.zero
                )
                q = _quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0] if weight else (
                shape[1] if len(shape) != 3 else shape[2]
            )
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            if self.grouprows > 1:
                self.scale = self.scale.unsqueeze(1).repeat(1, self.grouprows)
                self.zero = self.zero.unsqueeze(1).repeat(1, self.grouprows)
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def ready(self):
        return torch.all(self.scale != 0)


# =====================================================================
# SparseGPT per-sublayer joint prune+quant (port of sparsegpt.py:SparseGPT)
# =====================================================================


class SparseGPT:
    """Per-sublayer joint pruning + (optional) quantization driver.

    Direct port of /tmp/sparsegpt/sparsegpt.py:SparseGPT. Hessian accumulation
    and the column-block OBS sweep are byte-equivalent to upstream; only the
    surrounding harness (model dispatch, dataset loaders, CSV) is BiLLM2-side.
    """

    def __init__(self, layer, device):
        self.layer = layer
        self.dev = device
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros(
            (self.columns, self.columns), device=device, dtype=torch.float32
        )
        self.nsamples = 0
        self.quantizer = None  # set externally if joint sparse+quant

    def add_batch(self, inp, out):
        """Streaming H = (2/N) X^T X with running-average renormalisation."""
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, prunen=0, prunem=0, blocksize=128,
                    percdamp=0.01):
        """Run the joint OBS sweep.

        Returns (Q, quant_time, total_loss, mask_zeros). `mask_zeros` is the
        count of positions explicitly pruned by the keep-mask (the storage-
        level sparsity). The post-quantize weight tensor may contain
        additional zeros if `self.quantizer` is set and the affine grid
        contains a zero codebook entry — that is a quantizer-level effect,
        not a mask-level one.
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        W = W.float()

        if self.quantizer is not None:
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)
        mask_zeros = 0  # number of positions zeroed by the mask itself

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][
                        int(tmp.numel() * sparsity)
                    ]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i:(i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + torch.topk(tmp, prunen, dim=1, largest=False)[1],
                        True,
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                if self.quantizer is not None:
                    q = _quantize(
                        q.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            mask_zeros += int(mask1.sum().item())

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        quant_time = time.time() - tick
        total_loss = float(torch.sum(Losses).item())
        del Hinv, H
        torch.cuda.empty_cache()
        return W, quant_time, total_loss, mask_zeros

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


# =====================================================================
# Model loading (mirrors src/run_leanquant.py)
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
    cls = model.__class__.__name__.lower()
    if "opt" in cls:
        return "opt"
    if "llama" in cls or "mistral" in cls:
        return "llama"
    if "qwen" in cls:
        return "qwen"
    raise ValueError(f"Unknown model class: {model.__class__.__name__}")


def _true_sequential_groups(model_type):
    if model_type in ("llama", "qwen"):
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    return [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"],
    ]


# =====================================================================
# SparseGPT pipeline: forward + hook, then prune+quant each sublayer
# =====================================================================


@torch.no_grad()
def sparsegpt_quantize_model(model, args, calib_dataset, calib_seqlen):
    """Sequentially prune+quantize every transformer layer with SparseGPT."""
    dev = torch.device(args.device)
    nsamples = args.nsamples

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    dataloader, _ = get_loaders(
        calib_dataset,
        nsamples=nsamples,
        seed=args.seed,
        seqlen=calib_seqlen,
        model=args.model,
    )

    model.config.use_cache = False
    model_type = detect_model_type(model)

    # Move embeddings + norm + (rotary) to GPU for the capture phase
    if model_type == "opt":
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = (
            model.model.decoder.embed_positions.to(dev)
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

    # CPU activation offload threshold (8 GB)
    act_bytes = nsamples * calib_seqlen * model.config.hidden_size * 2
    act_device = "cpu" if act_bytes > 8 * (1024 ** 3) else dev
    if act_device == "cpu":
        print(
            f"Activation offload: {act_bytes / 1024 ** 3:.1f} GB > 8 GB threshold, "
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
            model.model.decoder.project_out = (
                model.model.decoder.project_out.cpu()
            )
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
    if "past_key_values" in layer_kwargs:
        layer_kwargs["past_key_values"] = None

    sequential_groups = (
        _true_sequential_groups(model_type) if args.true_sequential else None
    )

    total_quant_params = 0
    total_mask_zeros = 0
    quant_start = time.time()

    nbits = args.nbits
    do_quant = nbits is not None and nbits < 16

    sparsity_str = (
        f"{args.prunen}:{args.prunem}"
        if args.prunen > 0
        else f"unstructured s={args.sparsity}"
    )
    quant_str = f"+{nbits}-bit" if do_quant else " (no quant)"
    print(
        f"\nQuantizing {len(layers)} layers with SparseGPT "
        f"[{sparsity_str}{quant_str}]..."
    )
    print(
        f"  percdamp={args.percdamp}, blocksize={args.blocksize}, "
        f"true_sequential={args.true_sequential}"
    )

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)
        full = find_layers(layer)

        if sequential_groups is not None:
            groups = [
                [n for n in names if n in full] for names in sequential_groups
            ]
            groups = [g for g in groups if g]
            seen = {n for g in groups for n in g}
            leftovers = [n for n in full if n not in seen]
            if leftovers:
                groups.append(leftovers)
        else:
            groups = [list(full.keys())]

        print(f"\nLayer {layer_idx}/{len(layers) - 1} -- {len(full)} sublayers")

        for names in groups:
            subset = {n: full[n] for n in names}

            sg = {n: SparseGPT(subset[n], dev) for n in names}
            if do_quant:
                for n in names:
                    sg[n].quantizer = Quantizer()
                    # Match upstream LLaMA driver (llama.py:103-106):
                    # perchannel=True, sym=False, mse=False.
                    sg[n].quantizer.configure(
                        nbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    sg[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for n in names:
                handles.append(subset[n].register_forward_hook(add_batch(n)))
            for j in range(nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0).to(dev), **layer_kwargs
                )[0]
            for h in handles:
                h.remove()

            for n in names:
                rows, cols = subset[n].weight.shape[0], subset[n].weight.shape[1]
                print(f"  {n} [{rows}x{cols}] pruning...")
                W_q, qt, loss, mask_zeros = sg[n].fasterprune(
                    sparsity=args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    blocksize=args.blocksize,
                    percdamp=args.percdamp,
                )
                subset[n].weight.data = W_q.reshape(
                    subset[n].weight.shape
                ).to(dtype)
                sg[n].free()
                total_quant_params += rows * cols
                total_mask_zeros += mask_zeros
                print(f"    time={qt:.1f}s  recon_loss={loss:.4e}")

        # Re-forward with pruned/quantized sublayers to feed inps[next_layer]
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0).to(dev), **layer_kwargs
            )[0]

        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        gc.collect()
        inps, outs = outs, inps

    quant_time = time.time() - quant_start
    mask_sparsity = total_mask_zeros / max(total_quant_params, 1)
    print(f"\nSparseGPT quantization complete in {quant_time:.1f}s")
    print(f"  Mask sparsity (storage-level): {mask_sparsity * 100:.2f}%")

    model.config.use_cache = True
    return model, quant_time, total_quant_params, mask_sparsity


# =====================================================================
# bpw accounting (bitmap mask + optional kept-weight quantization)
# =====================================================================


def compute_bpw(args, model):
    """Return effective bits-per-weight for the current configuration.

    Encoding assumed (matches the SDOML S1 derivation §6.1 for fair compare):
      * 1-bit-per-weight bitmap to encode the keep-mask
      * (1 - s) * nbits index storage per kept weight
      * Per-row quantizer params: 2 fp16 = 32 bits per row (scale + zero)
        — only when nbits < 16.

    For the structured 2:4 case (prunen=2, prunem=4) the bitmap can be
    replaced by a fixed 2-bit-per-4-weight pattern (the upstream paper
    convention). We still report bitmap-equivalent 1 bpw since (a) it
    upper-bounds the index storage and (b) downstream comparison against
    SDOML uses the same convention. Small-N rows shift bpw by < 0.05.
    """
    nbits = args.nbits if (args.nbits and args.nbits < 16) else 16
    if args.prunen > 0:
        s_eff = float(args.prunen) / float(args.prunem)
    else:
        s_eff = float(args.sparsity)

    # Use the model's hidden size as a representative "columns" number.
    # bpw is dominated by the bitmap + (1-s)*nbits term; the codebook tail
    # 32/cols is ~0.03 bpw on hidden=1024 and shrinks with larger models.
    cols = int(getattr(model.config, "hidden_size", 1024))

    if nbits >= 16:
        # No quantization: fp16 kept weights, bitmap-encoded mask.
        return 1.0 + (1.0 - s_eff) * 16.0
    return 1.0 + (1.0 - s_eff) * float(nbits) + 32.0 / cols


def csv_method_tag(args):
    if args.prunen > 0:
        base = f"sparsegpt-{args.prunen}:{args.prunem}"
    else:
        s_pct = int(round(float(args.sparsity) * 100))
        base = f"sparsegpt-s{s_pct}"
    if args.nbits is not None and args.nbits < 16:
        base = f"{base}-w{args.nbits}"
    return base


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SparseGPT (ICML 2023) standalone benchmark runner"
    )
    parser.add_argument("model", type=str, help="HuggingFace model name")
    parser.add_argument(
        "dataset", type=str, choices=["wikitext2", "c4"], help="Eval dataset"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Pruning hyperparameters
    parser.add_argument(
        "--sparsity", type=float, default=0.5,
        help="Target unstructured sparsity in [0, 1) (paper Table 1: 0.5)",
    )
    parser.add_argument(
        "--prunen", type=int, default=0,
        help="N for N:M structured pruning (e.g. 2 with --prunem 4 = 2:4). "
             "When > 0, --sparsity is ignored.",
    )
    parser.add_argument(
        "--prunem", type=int, default=0, help="M for N:M structured pruning"
    )

    # Quantization hyperparameters (joint sparse+quant when nbits < 16)
    parser.add_argument(
        "--nbits", type=int, default=16, choices=[2, 3, 4, 8, 16],
        help="Bits per kept weight (16 = no quantization, paper Table 1; "
             "4 = paper Table 4 joint; 2/3 also supported)",
    )

    # GPTQ hyperparameters
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Hessian dampening fraction (paper default: 0.01)",
    )
    parser.add_argument(
        "--blocksize", type=int, default=128,
        help="GPTQ column-block size for adaptive mask + error propagation",
    )
    parser.add_argument(
        "--true_sequential", action="store_true",
        help="Quantize k+v+q, o, up+gate, down sequentially "
             "(paper README convention)",
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

    # Downstream eval flags via shared helper
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
        ),
    )
    from eval_utils import add_eval_cli, resolve_eval_flags, evaluate_and_log_all
    add_eval_cli(parser)

    args = parser.parse_args()

    # Validate N:M args
    if args.prunen > 0:
        assert args.prunem > 0 and args.prunen < args.prunem, (
            f"N:M pruning requires 0 < prunen < prunem, got "
            f"{args.prunen}:{args.prunem}"
        )

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

    model, quant_time, total_quant_params, mask_sparsity = sparsegpt_quantize_model(
        model, args,
        calib_dataset=calib_dataset,
        calib_seqlen=calib_seqlen,
    )

    bpw = compute_bpw(args, model)
    method = csv_method_tag(args)

    # Post-quant zero count (mask zeros + any quantizer-codebook zeros)
    layers = (
        model.model.layers if hasattr(model.model, "layers")
        else model.model.decoder.layers
    )
    zeros = 0
    total = 0
    for l in layers:
        for m in l.modules():
            if isinstance(m, nn.Linear):
                zeros += int((m.weight.data == 0).sum().item())
                total += int(m.weight.data.numel())
    post_quant_sparsity = zeros / max(total, 1)
    print(
        f"\n  Mask sparsity (storage-level):    {mask_sparsity * 100:.2f}%"
    )
    print(
        f"  Post-quant zero fraction (incl.  affine codebook 0): "
        f"{post_quant_sparsity * 100:.2f}%"
    )

    extra = {
        "sparsity": args.sparsity,
        "prunen": args.prunen,
        "prunem": args.prunem,
        "nbits": args.nbits,
        "percdamp": args.percdamp,
        "blocksize": args.blocksize,
        "true_sequential": args.true_sequential,
        "nsamples": args.nsamples,
        "calib_dataset": calib_dataset,
        "calib_seqlen": calib_seqlen,
        "mask_sparsity": round(mask_sparsity, 4),
        "post_quant_sparsity": round(post_quant_sparsity, 4),
    }

    eval_flags = resolve_eval_flags(args, primary_dataset=args.dataset)

    model_short = args.model.split("/")[-1]
    print(f"\n{'=' * 60}")
    print(f"RESULT: {method} on {model_short}")
    print(f"  Seed: {args.seed}")
    print(f"  Effective bpw: {bpw:.4f}")
    print(f"  Quantization time: {quant_time:.1f}s")
    print(
        f"  Calibration: {calib_dataset} (nsamples={args.nsamples}, "
        f"seqlen={calib_seqlen})"
    )
    print(f"  PPL eval datasets: {eval_flags['ppl_datasets']}")
    print(f"{'=' * 60}")

    evaluate_and_log_all(
        model, args.model, torch.device(args.device),
        method=method,
        bpw=bpw, seed=args.seed, blocksize=args.blocksize,
        salient_metric="",
        extra_params=extra,
        quantization_time_s=quant_time,
        ppl_datasets=eval_flags["ppl_datasets"],
        eval_mmlu=eval_flags["eval_mmlu"],
        eval_hellaswag=eval_flags["eval_hellaswag"],
        eval_arc=eval_flags["eval_arc"],
        ppl_eval_seqlen=eval_flags["ppl_eval_seqlen"],
        save_title_prefix=f"{method}_{model_short}_seed{args.seed}",
    )


if __name__ == "__main__":
    main()
