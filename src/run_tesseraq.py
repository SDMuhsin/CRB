"""
TesseraQ: Ultra Low-Bit LLM Post-Training Quantization with Block Reconstruction
ICLR 2025 — Standalone implementation for apples-to-apples comparison with DOML

Based on: https://github.com/Intelligent-Computing-Lab-Panda/TesseraQ
Paper: https://arxiv.org/abs/2410.19103

Core algorithm:
  1. Asymmetric uniform 2-bit quantization with per-group scales (group_size=128)
  2. Auto weight clipping for improved quantization ranges
  3. Progressive adaptive rounding via block reconstruction optimization
  4. Dequantization scale optimization

NOTE: This runs TesseraQ WITHOUT AWQ initialization (load_transform=False).
The paper's best results combine TesseraQ + AWQ pre-processing. Without AWQ,
this represents TesseraQ's core adaptive rounding algorithm in isolation.

Usage:
    source env/bin/activate
    python3 -u src/run_tesseraq.py Qwen/Qwen3-0.6B wikitext2 --device cuda:0 --seed 0
    python3 -u src/run_tesseraq.py meta-llama/Llama-3.2-1B wikitext2 --device cuda:0 --seed 0
    python3 -u src/run_tesseraq.py facebook/opt-1.3b wikitext2 --device cuda:0 --seed 0
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
import torch.nn.functional as F

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datautils import get_tokenizer, set_seed
from eval_ppl_utils import llama_eval, opt_eval, qwen_eval


# =====================================================================
# Quantization Primitives (from LLMC IntegerQuantizer)
# =====================================================================

class UniformQuantizer:
    """Asymmetric uniform integer quantizer with per-group granularity.

    Matches LLMC's IntegerQuantizer: bit=2, symmetric=False,
    granularity=per_group, group_size=128, calib_algo=minmax.
    """

    def __init__(self, bit=2, group_size=128, symmetric=False):
        self.bit = bit
        self.group_size = group_size
        self.symmetric = symmetric
        if symmetric:
            self.qmin = -(2 ** (bit - 1))
            self.qmax = 2 ** (bit - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bit - 1

    def reshape_to_groups(self, tensor):
        """Reshape (out, in) → (out * in/gs, gs) for per-group quantization."""
        if self.group_size <= 0 or tensor.shape[-1] <= self.group_size:
            return tensor
        if tensor.shape[-1] % self.group_size != 0:
            raise ValueError(
                f"Dimension {tensor.shape[-1]} not divisible by "
                f"group_size {self.group_size}"
            )
        return tensor.reshape(-1, self.group_size)

    def restore_shape(self, tensor, orig_shape):
        if tensor.shape == orig_shape:
            return tensor
        return tensor.reshape(orig_shape)

    def compute_qparams(self, weight):
        """Compute scales and zero-points using minmax calibration."""
        w = self.reshape_to_groups(weight)
        if self.symmetric:
            abs_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
            scales = abs_max / self.qmax
            zeros = torch.zeros(1, device=w.device, dtype=w.dtype)
        else:
            max_val = w.amax(dim=-1, keepdim=True)
            min_val = w.amin(dim=-1, keepdim=True)
            scales = (max_val - min_val).clamp(min=1e-5) / (self.qmax - self.qmin)
            zeros = torch.round(
                self.qmin - min_val / scales
            ).clamp(self.qmin, self.qmax)
        return scales, zeros

    def fake_quant(self, weight, scales, zeros):
        """Standard round-to-nearest fake quantization."""
        orig_shape = weight.shape
        w = self.reshape_to_groups(weight)
        q = torch.clamp(torch.round(w / scales) + zeros, self.qmin, self.qmax)
        dq = (q - zeros) * scales
        return self.restore_shape(dq, orig_shape)

    def fake_quant_with_rounding(self, weight, scales, zeros, rounding,
                                  output_scale_factor=None):
        """Fake quantize with learned rounding directions.

        Args:
            weight: float32 weight tensor (plain, not tracked by autograd)
            scales: per-group scales (detached)
            zeros: per-group zeros (detached)
            rounding: continuous values in (0, 1) from RectifiedSigmoid
                      (HAS requires_grad=True during optimization)
            output_scale_factor: multiplicative correction for dequant scales
                                 (HAS requires_grad=True during optimization)
        """
        orig_shape = weight.shape
        w = self.reshape_to_groups(weight)

        # floor + learned_rounding replaces round
        q = torch.clamp(
            torch.floor(w / scales) + rounding + zeros,
            self.qmin, self.qmax
        )

        if output_scale_factor is not None:
            dq = (q - zeros) * (scales * output_scale_factor)
        else:
            dq = (q - zeros) * scales

        return self.restore_shape(dq, orig_shape)


# =====================================================================
# Rectified Sigmoid (from LLMC module_utils)
# =====================================================================

class RectifiedSigmoid(nn.Module):
    """Maps R → (0, 1) with slight overshoot for progressive rounding."""

    def __init__(self, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, x):
        return torch.clamp(
            torch.sigmoid(x) * (self.zeta - self.gamma) + self.gamma, 0, 1
        )

    def inverse(self, y):
        """Inverse: given y in (0,1), return x such that forward(x) ≈ y.
        Note: gamma=-0.1, zeta=1.1 ensure the log argument is well-behaved
        for y in [0, 1], so no extra clamping needed (matches LLMC)."""
        return -torch.log((self.zeta - self.gamma) / (y - self.gamma) - 1)


# =====================================================================
# Auto Weight Clipping (simplified from LLMC auto_clip_layer)
# =====================================================================

@torch.no_grad()
def auto_clip_layer(weight, quantizer, n_grid=20, max_shrink=0.5):
    """Find optimal per-group clipping range to minimize quantization MSE.

    NOTE: This is a weight-only simplification of LLMC's auto_clip_layer.
    The LLMC version minimizes ||X @ (W_clip_quant - W)||^2 (activation-weighted),
    which requires captured activation features and distributed reduce.
    This version minimizes ||W_clip_quant - W||^2 (weight-only), which is simpler
    but may produce slightly worse clipping ranges. For a pure method comparison
    this is acceptable since DOML also doesn't use activation-weighted preprocessing.
    """
    gs = quantizer.group_size
    if gs <= 0 or weight.shape[1] <= gs:
        gs = weight.shape[1]

    w = weight.reshape(weight.shape[0], 1, -1, gs)  # (oc, 1, n_groups, gs)
    oc_batch = 256 if weight.shape[0] % 256 == 0 else 64
    if weight.shape[0] % oc_batch != 0:
        oc_batch = weight.shape[0]  # process all at once for odd sizes

    best_max_all = []
    best_min_all = []

    for i_b in range(w.shape[0] // oc_batch):
        wb = w[i_b * oc_batch:(i_b + 1) * oc_batch]
        org_max = wb.amax(dim=-1, keepdim=True)
        org_min = wb.amin(dim=-1, keepdim=True)

        best_max = org_max.clone()
        best_min = org_min.clone()
        min_errs = torch.ones_like(org_max) * 1e9

        for i_s in range(int(max_shrink * n_grid)):
            p = 1 - i_s / n_grid
            cur_max = p * org_max
            cur_min = p * org_min

            clipped = wb.clamp(cur_min, cur_max)
            clipped_flat = clipped.reshape(-1, gs)

            scales = (cur_max - cur_min).clamp(min=1e-5).reshape(-1, 1) / (
                quantizer.qmax - quantizer.qmin
            )
            zeros = torch.round(
                quantizer.qmin - cur_min.reshape(-1, 1) / scales
            ).clamp(quantizer.qmin, quantizer.qmax)

            q = torch.clamp(
                torch.round(clipped_flat / scales) + zeros,
                quantizer.qmin, quantizer.qmax,
            )
            dq = ((q - zeros) * scales).reshape(wb.shape)
            err = (dq - wb).pow(2).sum(dim=-1, keepdim=True)

            better = err < min_errs
            if better.any():
                min_errs[better] = err[better]
                best_max[better] = cur_max.expand_as(best_max)[better]
                best_min[better] = cur_min.expand_as(best_min)[better]

        best_max_all.append(best_max)
        best_min_all.append(best_min)

    best_max = torch.cat(best_max_all, dim=0)
    best_min = torch.cat(best_min_all, dim=0)
    return best_max, best_min


@torch.no_grad()
def apply_auto_clip(block, quantizer, skip_qk=True):
    """Apply auto-clip to all linear layers in a block (weight-only version)."""
    for name, module in block.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if skip_qk and any(k in name for k in ['q_proj', 'k_proj', 'query', 'key']):
            continue
        gs = quantizer.group_size
        if module.weight.shape[1] % gs != 0:
            print(f"  Skip auto-clip for {name}: dim {module.weight.shape[1]} "
                  f"not divisible by group_size {gs}")
            continue
        best_max, best_min = auto_clip_layer(module.weight.data, quantizer)
        org_shape = module.weight.shape
        if gs <= 0 or module.weight.shape[1] <= gs:
            gs_eff = module.weight.shape[1]
        else:
            gs_eff = gs
        w = module.weight.data.reshape(org_shape[0], 1, -1, gs_eff)
        w.clamp_(best_min, best_max)
        module.weight.data = w.reshape(org_shape)
        print(f"  Auto-clip: {name}")


# =====================================================================
# Model Architecture Helpers
# =====================================================================

def detect_model_type(model):
    class_name = model.__class__.__name__.lower()
    if 'opt' in class_name:
        return 'opt'
    elif 'llama' in class_name:
        return 'llama'
    elif 'qwen' in class_name:
        return 'qwen'
    raise ValueError(f"Unknown model class: {model.__class__.__name__}")


def get_blocks_and_kwargs(model, dev, trainloader, nsamples, seqlen):
    """Capture first-block inputs using the Catcher pattern from our eval code."""
    model_type = detect_model_type(model)

    if model_type == 'opt':
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif model_type in ('llama', 'qwen'):
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=dev)
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
    for i in range(nsamples):
        batch = trainloader[i][0].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if model_type == 'opt':
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif model_type in ('llama', 'qwen'):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.cpu()

    torch.cuda.empty_cache()

    # Clean up layer_kwargs for standalone block forward:
    # 1. Remove KV cache (TesseraQ doesn't use caching)
    # 2. Detach all tensors to avoid grad graph contamination from model forward
    kw = cache["layer_kwargs"]
    if 'past_key_values' in kw:
        kw['past_key_values'] = None
    if 'use_cache' in kw:
        kw['use_cache'] = False

    def detach_kwarg(v):
        if isinstance(v, torch.Tensor):
            return v.detach()
        elif isinstance(v, tuple):
            return tuple(detach_kwarg(x) for x in v)
        elif isinstance(v, list):
            return [detach_kwarg(x) for x in v]
        return v

    kw = {k: detach_kwarg(v) for k, v in kw.items()}

    return layers, inps, kw


def get_linear_layers(block):
    """Get all nn.Linear modules in a block as {name: module} dict."""
    linears = {}
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            linears[name] = module
    return linears


def set_module_by_name(block, name, new_module):
    """Set a submodule in block by dotted name (e.g., 'self_attn.q_proj')."""
    parts = name.split('.')
    parent = block
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


# =====================================================================
# FakeQuantLinear — replaces nn.Linear during optimization
# =====================================================================

class FakeQuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear that applies fake quantization
    with learnable rounding. Used during TesseraQ optimization."""

    def __init__(self, original_linear, quantizer, scales, zeros, sigmoid,
                 optimize_scale=True):
        super().__init__()
        # Store original weight as buffer (not parameter — not optimized)
        self.register_buffer('weight', original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.bias = None

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.quantizer = quantizer
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.sigmoid = sigmoid

        # Initialize rounding from fractional parts
        w_grouped = quantizer.reshape_to_groups(self.weight)
        frac = w_grouped / scales - torch.floor(w_grouped / scales)
        # buf_rounding: the learnable parameter (requires_grad set externally)
        self.register_buffer('buf_rounding', sigmoid.inverse(frac))

        # Optional dequantization scale optimization
        self.optimize_scale = optimize_scale
        if optimize_scale:
            self.register_buffer('buf_scale_factor', torch.zeros_like(scales))

        self.rounding_opt = True  # when True, use learned rounding; when False, round-to-nearest

    def forward(self, x):
        # Quantization math in float32 for precision, output cast to input dtype
        if self.rounding_opt:
            r = self.sigmoid(self.buf_rounding)
            osf = None
            if self.optimize_scale:
                osf = 2 * self.sigmoid(self.buf_scale_factor)
            w_q = self.quantizer.fake_quant_with_rounding(
                self.weight, self.scales, self.zeros, r, osf
            )
        else:
            w_q = self.quantizer.fake_quant(self.weight, self.scales, self.zeros)
        return F.linear(x, w_q.to(x.dtype), self.bias)


# =====================================================================
# TesseraQ Block Optimizer
# =====================================================================

class TesseraQOptimizer:
    """Progressive adaptive rounding with block reconstruction.

    Hyperparameters match the official TesseraQ W2g128 config:
      lr=0.001, iterations=250, batch_size=4, 20 progressive steps,
      optimize_scale=True, scale_lr=0.001.
    """

    THRESHOLDS = [
        0.8, 0.65, 0.5, 0.43, 0.38, 0.34, 0.3, 0.27, 0.24, 0.21,
        0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005
    ]

    def __init__(self, quantizer, lr=0.001, scale_lr=0.001, iterations=250,
                 batch_size=4, optimize_scale=True):
        self.quantizer = quantizer
        self.lr = lr
        self.scale_lr = scale_lr
        self.iterations = iterations
        self.batch_size = batch_size
        self.optimize_scale = optimize_scale
        self.sigmoid = RectifiedSigmoid().cuda()

    def optimize_block(self, block, inps, layer_kwargs, dev, n_calib):
        """Run TesseraQ on one transformer block.

        Returns:
            outs: (n_calib, seqlen, hidden) quantized block outputs for next block
        """
        model_dtype = next(block.parameters()).dtype
        block = block.to(dev)
        # Keep block in model dtype (float16/bfloat16) to preserve attention mask
        # compatibility. FakeQuantLinear does quantization math in float32 internally.

        # Freeze all block parameters
        for p in block.parameters():
            p.requires_grad_(False)

        # 1. Collect FP16 target outputs (sample-by-sample to save memory)
        print("    Collecting FP16 targets...")
        fp16_outs = []
        with torch.no_grad():
            for j in range(n_calib):
                with torch.cuda.amp.autocast():
                    out = block(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]
                fp16_outs.append(out.float().cpu())
        fp16_outs = torch.cat(fp16_outs, dim=0)  # (n_calib, seqlen, hidden)

        # 2. Auto-clip weights (in float32 for precision, then back to model dtype)
        print("    Auto-clipping weights...")
        linears = get_linear_layers(block)
        for name, linear in linears.items():
            linear.weight.data = linear.weight.data.float()
        apply_auto_clip(block, self.quantizer)

        # 3. Replace nn.Linear modules with FakeQuantLinear
        print("    Replacing linear layers with FakeQuantLinear...")
        original_linears = {}
        fq_linears = {}

        for name, linear in linears.items():
            # Compute qparams in float32
            scales, zeros = self.quantizer.compute_qparams(linear.weight.data)
            fql = FakeQuantLinear(
                linear, self.quantizer, scales, zeros, self.sigmoid,
                optimize_scale=self.optimize_scale,
            )
            # Restore original linear weight to model dtype
            linear.weight.data = linear.weight.data.to(model_dtype)
            original_linears[name] = linear
            fq_linears[name] = fql
            set_module_by_name(block, name, fql)

        # 4. Evaluate loss before optimization
        with torch.no_grad():
            loss_before = self._compute_loss_batched(
                block, inps[:4].to(dev), fp16_outs[:4].to(dev), layer_kwargs
            )
            print(f"    Loss before TesseraQ: {loss_before.item():.6f}")

        # 5. Progressive adaptive rounding optimization
        all_inps = inps.to(dev)  # (n_calib, seqlen, hidden) in model dtype
        all_targets = fp16_outs.to(dev)  # float32 targets

        for t_idx, threshold in enumerate(self.THRESHOLDS):
            # Enable rounding optimization mode
            for fql in fq_linears.values():
                fql.rounding_opt = True

            # Freeze confident rounding decisions
            self._update_masks(fq_linears, threshold)

            # Collect learnable parameters
            params_r = []
            params_s = []
            for fql in fq_linears.values():
                fql.buf_rounding.requires_grad_(True)
                params_r.append(fql.buf_rounding)
                if self.optimize_scale:
                    fql.buf_scale_factor.requires_grad_(True)
                    params_s.append(fql.buf_scale_factor)

            # Setup optimizer
            opt_groups = [{'params': params_r, 'lr': self.lr}]
            if self.optimize_scale:
                opt_groups.append({
                    'params': params_s,
                    'lr': self.scale_lr,
                    'weight_decay': 1e-4,
                })
            optimizer = torch.optim.Adam(opt_groups, lr=self.lr)

            # Optimize for self.iterations steps
            # NOTE: autocast is intentionally NOT used here. The FakeQuantLinear
            # does quantization math in float32 internally and casts output to
            # input dtype. Using autocast causes graph retention issues with
            # multi-sample loss computation.
            with torch.enable_grad():
                for it in range(self.iterations):
                    indices = torch.randperm(n_calib)[:self.batch_size]

                    loss = self._compute_loss_batched(
                        block, all_inps[indices], all_targets[indices],
                        layer_kwargs,
                    )

                    if not math.isfinite(loss.item()):
                        print(f"    WARNING: NaN loss at step {t_idx}, iter {it}")
                        break

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Detach gradients
            for fql in fq_linears.values():
                fql.buf_rounding.requires_grad_(False)
                if self.optimize_scale:
                    fql.buf_scale_factor.requires_grad_(False)

            del optimizer
            progress = (1 - threshold) * 100
            print(f"    Step {t_idx+1}/{len(self.THRESHOLDS)}: "
                  f"loss={loss.item():.6f}, hard-rounded={progress:.1f}%")

        # 6. Hard-round all remaining decisions
        for fql in fq_linears.values():
            fql.buf_rounding.data = 100 * fql.buf_rounding.data.sign()

        # 7. Evaluate loss after optimization
        with torch.no_grad():
            loss_after = self._compute_loss_batched(
                block, all_inps[:4], all_targets[:4], layer_kwargs
            )
            print(f"    Loss after TesseraQ: {loss_after.item():.6f}")

        # 8. Merge rounding into weights and restore original nn.Linear modules
        with torch.no_grad():
            for name, fql in fq_linears.items():
                r = self.sigmoid(fql.buf_rounding)
                osf = None
                if self.optimize_scale:
                    osf = 2 * self.sigmoid(fql.buf_scale_factor)
                w_q = self.quantizer.fake_quant_with_rounding(
                    fql.weight, fql.scales, fql.zeros, r, osf
                )
                # Write quantized weights back to original linear
                orig_linear = original_linears[name]
                orig_linear.weight.data.copy_(w_q.to(model_dtype))
                set_module_by_name(block, name, orig_linear)

        # 9. Compute quantized outputs for next block (using merged weights)
        print("    Computing quantized outputs...")
        outs = torch.zeros_like(inps)
        with torch.no_grad():
            for j in range(n_calib):
                outs[j] = block(
                    inps[j].unsqueeze(0).to(dev), **layer_kwargs
                )[0].cpu()

        # Cleanup
        block.cpu()
        del all_inps, all_targets, fp16_outs, fq_linears, original_linears
        gc.collect()
        torch.cuda.empty_cache()

        return outs

    def _compute_loss_batched(self, block, inps, targets, layer_kwargs):
        """L2 reconstruction loss.

        Processes one sample at a time to avoid attention_mask broadcasting
        issues, then averages.
        """
        batch_size = inps.shape[0]
        losses = []
        for j in range(batch_size):
            inp_j = inps[j].unsqueeze(0).detach()
            tgt_j = targets[j].unsqueeze(0).detach()
            out = block(inp_j, **layer_kwargs)[0].float()
            loss_j = (tgt_j - out).pow(2).sum(-1).mean()
            losses.append(loss_j)
        return torch.stack(losses).mean()

    def _update_masks(self, fq_linears, quantile_threshold):
        """Freeze confident rounding decisions (progressive hardening)."""
        for fql in fq_linears.values():
            score = (self.sigmoid(fql.buf_rounding) - 0.5).abs()
            value = float(np.quantile(
                score.detach().cpu().numpy(), q=quantile_threshold
            ))
            mask_up = self.sigmoid(fql.buf_rounding.data) > (value + 0.5)
            mask_down = self.sigmoid(fql.buf_rounding.data) < (0.5 - value)
            fql.buf_rounding.data[mask_up] = float('inf')
            fql.buf_rounding.data[mask_down] = float('-inf')


# =====================================================================
# Main Pipeline
# =====================================================================

def get_model(model_name):
    """Load model using our standard infrastructure."""
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    downloads_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")

    if "opt" in model_name.lower():
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", cache_dir=downloads_dir,
            use_safetensors=True, attn_implementation="eager",
        )
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model_name.lower():
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", cache_dir=downloads_dir,
            use_safetensors=True, attn_implementation="eager",
        )
        model.seqlen = 2048
    elif "qwen" in model_name.lower():
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", cache_dir=downloads_dir,
            attn_implementation="eager",
        )
        model.seqlen = min(model.config.max_position_embeddings, 2048)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    return model


def get_calibration_data(model_name, nsamples, seed, seqlen, dataset='wikitext2'):
    """Load calibration data from specified dataset."""
    from datautils import get_loaders
    trainloader, testenc = get_loaders(dataset, nsamples=nsamples, seed=seed,
                                        seqlen=seqlen, model=model_name)
    return trainloader, testenc


def tesseraq_quantize(model, trainloader, args):
    """Main TesseraQ quantization loop: process blocks sequentially."""
    dev = torch.device(args.device)
    nsamples = len(trainloader)
    seqlen = model.seqlen

    print(f"\n{'='*60}")
    print(f"TesseraQ W{args.bit}g{args.group_size} Quantization")
    print(f"Model: {args.model}")
    print(f"Calibration: {nsamples} samples, seqlen={seqlen}")
    print(f"Iterations: {args.iterations} per step × "
          f"{len(TesseraQOptimizer.THRESHOLDS)} steps = "
          f"{args.iterations * len(TesseraQOptimizer.THRESHOLDS)} total/block")
    print(f"Batch size: {args.batch_size}")
    print(f"Scale optimization: {args.optimize_scale}")
    bpw = args.bit + (32 / args.group_size)
    print(f"Effective bpw: ~{bpw:.2f}")
    print(f"{'='*60}\n")

    # Capture first-block inputs
    print("Capturing block inputs...")
    layers, inps, layer_kwargs = get_blocks_and_kwargs(
        model, dev, trainloader, nsamples, seqlen
    )

    quantizer = UniformQuantizer(
        bit=args.bit, group_size=args.group_size, symmetric=False
    )
    optimizer = TesseraQOptimizer(
        quantizer,
        lr=args.lr,
        scale_lr=args.scale_lr,
        iterations=args.iterations,
        batch_size=args.batch_size,
        optimize_scale=args.optimize_scale,
    )

    total_start = time.time()
    for i in range(len(layers)):
        print(f"\n--- Block {i}/{len(layers)} ---")
        block_start = time.time()

        outs = optimizer.optimize_block(layers[i], inps, layer_kwargs, dev, nsamples)
        inps = outs  # propagate quantized outputs to next block

        block_time = time.time() - block_start
        print(f"    Block {i} done in {block_time:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal quantization time: {total_time:.1f}s")
    return model


def evaluate_ppl(model, testenc, args):
    """Evaluate perplexity using our standard evaluation code."""
    dev = torch.device(args.device)
    model_type = detect_model_type(model)
    model_short = args.model.split('/')[-1]
    dataset = getattr(args, 'dataset', 'wikitext2')
    save_title = f"tesseraq_W{args.bit}g{args.group_size}_{model_short}_{dataset}_seed{args.seed}"

    if model_type == 'opt':
        ppl = opt_eval(model, testenc, dev, dataset, save_title=save_title)
    elif model_type == 'llama':
        ppl = llama_eval(model, testenc, dev, dataset, save_title=save_title)
    elif model_type == 'qwen':
        ppl = qwen_eval(model, testenc, dev, dataset, save_title=save_title)
    return ppl


def main():
    parser = argparse.ArgumentParser(
        description="TesseraQ (ICLR 2025) W2 quantization benchmark"
    )
    parser.add_argument('model', type=str, help='HuggingFace model name')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'c4'])
    parser.add_argument('--bit', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Calibration samples (paper uses 512)')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scale_lr', type=float, default=0.001)
    parser.add_argument('--iterations', type=int, default=250,
                        help='Iterations per progressive step')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimize_scale', action='store_true', default=True)
    parser.add_argument('--no_optimize_scale', dest='optimize_scale',
                        action='store_false')
    parser.add_argument('--eval_arc', action='store_true')
    parser.add_argument('--eval_mmlu', action='store_true')
    parser.add_argument('--eval_hellaswag', action='store_true')
    args = parser.parse_args()

    from csv_utils import append_result as csv_append

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    model = get_model(args.model)

    print(f"Loading calibration data: {args.nsamples} samples")
    trainloader, testenc = get_calibration_data(
        args.model, args.nsamples, args.seed, model.seqlen, dataset=args.dataset
    )

    tick = time.time()
    model = tesseraq_quantize(model, trainloader, args)
    quant_time = time.time() - tick

    bpw = args.bit + (32 / args.group_size)
    extra = {"bit": args.bit, "group_size": args.group_size, "iterations": args.iterations}
    def _csv(dataset, metric, value):
        csv_append(model=args.model, method="tesseraq", dataset=dataset,
                   metric=metric, value=value, bpw=bpw, seed=args.seed,
                   blocksize=args.group_size, salient_metric="",
                   extra_params=extra, quantization_time_s=quant_time)

    print(f"\n{'='*60}")
    print(f"Evaluating perplexity on {args.dataset}...")
    print(f"{'='*60}")
    ppl = evaluate_ppl(model, testenc, args)
    _csv(args.dataset, "perplexity", ppl)

    model_short = args.model.split('/')[-1]
    print(f"\n{'='*60}")
    print(f"RESULT: TesseraQ W{args.bit}g{args.group_size} on {model_short}")
    print(f"  {args.dataset} PPL: {ppl:.2f}")
    print(f"  Seed: {args.seed}")
    print(f"  Calibration: {args.nsamples} samples")
    print(f"  Effective bpw: ~{bpw:.2f}")
    print(f"  Quantization time: {quant_time:.1f}s")
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
        _csv(args.dataset, "arc_challenge_acc", arc_results["ARC-Challenge"]["accuracy"])

    return ppl


if __name__ == '__main__':
    main()
