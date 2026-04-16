"""
SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights
Huawei, 2025 — Standalone implementation for apples-to-apples comparison with DOML

Based on: https://github.com/huawei-csl/SINQ
Paper: https://arxiv.org/abs/2509.22944

Core algorithm:
  1. Tile weight matrix into column groups (group_size columns each)
  2. For each tile: Sinkhorn normalization to find row/column scales (mu1, mu2)
     that minimize matrix imbalance (max std / min std across rows+cols)
  3. Round-to-nearest quantization on the normalized matrix
  4. Dual-scale dequantization: W_recon = (W_q - z) * scale * mu2 * mu1

NOTE: SINQ is calibration-free — no forward passes through the model needed.
This is the core SINQ method (not A-SINQ which adds AWQ calibration).

Usage:
    source env/bin/activate
    python3 -u src/run_sinq.py Qwen/Qwen3-0.6B wikitext2 --device cuda:0 --seed 0
    python3 -u src/run_sinq.py facebook/opt-1.3b wikitext2 --device cuda:0 --seed 0
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.nn as nn

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datautils import get_tokenizer, set_seed
from eval_ppl_utils import llama_eval, opt_eval, qwen_eval


# =====================================================================
# Sinkhorn Normalization (from SINQ: sinq/sinkhorn.py)
# =====================================================================

def sinkhorn_log(matrix, order=8, clip_min=1e-3, clip_max=1e3, eps=1e-6,
                 stop_on_increasing_imbalance=True):
    """
    Sinkhorn iteration that returns the mu1/mu2 corresponding to the matrix
    with the minimal imbalance encountered during the iteration.

    Returns (scaled_matrix, mu1_at_minimum, mu2_at_minimum)

    Ported directly from SINQ sinq/sinkhorn.py:sinkhorn_log
    """
    dtype = torch.float32
    m = matrix.to(dtype)
    dev = m.device
    measure = torch.std

    def imbalance(mat):
        s1, s2 = measure(mat, 1), measure(mat, 0)
        s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
        s_max = torch.maximum(s1.max(), s2.max())
        return s_max / s_min

    imb_min = torch.tensor(float('inf'), dtype=dtype, device=dev)
    gate = torch.tensor(0.0, dtype=dtype, device=dev)

    tgt_small = torch.minimum(
        m.std(1).clamp(clip_min, clip_max).min(),
        m.std(0).clamp(clip_min, clip_max).min()
    ) + eps

    log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=dev)
    log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=dev)

    # Known-good candidates for step k=0
    cur0 = m
    ib0 = imbalance(cur0)
    imb_min = torch.minimum(imb_min, ib0)
    mu1_star = log_mu1.exp().clone()
    mu2_star = log_mu2.exp().clone()

    for _ in range(order):
        cur = (m / log_mu1.exp()) / log_mu2.exp()
        ib = imbalance(cur)

        # Update the best-so-far candidates
        better = (ib <= imb_min).to(dtype)
        imb_min = torch.min(imb_min, ib)
        mu1_star = torch.where(better.bool(), log_mu1.exp(), mu1_star)
        mu2_star = torch.where(better.bool(), log_mu2.exp(), mu2_star)

        # Early-exit condition
        if stop_on_increasing_imbalance:
            rising = (ib > imb_min).to(dtype)
            gate = torch.clip(gate + rising, max=1.0)

        # Still-running samples update the dual variables
        g = 1.0 - gate

        std_r = measure(cur, 1).clamp(clip_min, clip_max)
        std_c = measure(cur, 0).clamp(clip_min, clip_max)

        sal_col = (std_c / tgt_small).clamp(0.7, 2.0).log()
        sal_row = (std_r[:, None] / tgt_small).clamp(0.7, 2.0).log()

        log_mu1 = (log_mu1 + (sal_col * g)).clip(-.3, 10.)
        log_mu2 = (log_mu2 + (sal_row * g)).clip(-.3, 10.)

    scaled = m / mu1_star / mu2_star
    return scaled, mu1_star, mu2_star


# =====================================================================
# RTN Quantization (from SINQ: sinq/dual_shift.py)
# =====================================================================

def quantize_rtn(matrix, min_max):
    """
    Asymmetric round-to-nearest quantization with per-row scales/zeros.

    Ported directly from SINQ sinq/dual_shift.py:quantize_rtn (uniform mode)
    """
    w = matrix
    orig_dtype = w.dtype
    w = w.to(torch.float32)

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = min_max[1]
    min_int = min_max[0]
    scales = (max_val - min_val).clamp(min=1e-4) / max_int
    zeros = -torch.round(min_val / scales)
    q = torch.clamp(torch.round(w / scales + zeros), min_int, max_int).to(torch.int8)
    return q.contiguous(), scales.to(orig_dtype), zeros.to(orig_dtype), orig_dtype


# =====================================================================
# Dual-Scale SINQ Quantization (from SINQ: sinq/dual_shift.py)
# =====================================================================

def quantize_dual_scale_shift(matrix, min_max, method='sinq'):
    """
    Core SINQ quantization: Sinkhorn normalization + RTN quantization.

    Returns (q, s1, s2, z) where:
      - q: integer codes
      - s1: per-row scale * mu2 (row Sinkhorn scale)
      - s2: per-column scale * mu1 (column Sinkhorn scale)
      - z: per-row zero points

    Dequantize: W_recon = (q - z) * s1 * s2

    Ported directly from SINQ sinq/dual_shift.py:quantize_dual_scale_shift
    """
    dtype = matrix.dtype
    dev = matrix.device
    matrix = matrix.float()

    # Sinkhorn normalization (16 iterations, matching SINQ defaults)
    matrix, mu1, mu2 = sinkhorn_log(matrix, 16)

    # RTN quantize the normalized matrix
    q, scales, z, _ = quantize_rtn(matrix, min_max)

    # Absorb Sinkhorn scales into quantization scales
    scales2 = torch.ones(1, matrix.shape[1]).to(dev).to(mu1.dtype) * mu1
    scales = scales * mu2

    q = q.to(dtype).to(dev)
    s1 = scales.to(dtype)
    s2 = scales2.to(dtype)
    z = z.to(dtype).to(dev)

    return q, s1.to(dev), s2.to(dev), z


# =====================================================================
# Tiled Quantization (from SINQ: sinq/dual_shift.py)
# =====================================================================

def tiled_quant_rectangle(M, min_max, block, method='sinq'):
    """
    1D tiling: split weight matrix into column tiles and quantize each.

    Ported directly from SINQ sinq/dual_shift.py:tiled_quant_rectangle
    but without vmap (process tiles sequentially for simplicity/robustness).
    """
    H, W = M.shape
    block = int(block)
    orig = block

    while block >= 16 and (W % block) != 0:
        block //= 2

    assert (W % block) == 0 and block >= 16, \
        f"block must divide W (W={W}, block={block})"
    if block != orig:
        print(f"[SINQ] Adjusted tile {orig} -> {block} for W={W}", flush=True)

    n_w = W // block

    # Process tiles sequentially (avoids vmap issues)
    Q_tiles = []
    s1_tiles = []
    s2_tiles = []
    z_tiles = []

    for i in range(n_w):
        tile = M[:, i * block:(i + 1) * block].contiguous()
        q, s1, s2, z = quantize_dual_scale_shift(tile, min_max, method=method)
        Q_tiles.append(q)
        s1_tiles.append(s1)
        s2_tiles.append(s2)
        z_tiles.append(z)

    # Combine: Q is (H, W), s1 is (H*n_w, 1), s2 is (1, W), z is (H*n_w, 1)
    Q = torch.cat(Q_tiles, dim=1)  # (H, W)
    s1 = torch.cat(s1_tiles, dim=0)  # (H*n_w, 1) — actually per-tile per-row
    s2 = torch.cat(s2_tiles, dim=1)  # (1, W)
    z = torch.cat(z_tiles, dim=0)  # (H*n_w, 1)

    return Q, s1, s2, z, n_w, block


# =====================================================================
# Full layer quantize + dequantize
# =====================================================================

@torch.no_grad()
def sinq_quantize_dequantize(weight, nbits=2, group_size=64):
    """
    Quantize a weight matrix with SINQ and immediately dequantize.

    This is the "fake quantization" path: we quantize to integer codes,
    then dequantize back to float to simulate the effect of quantization
    on the model's outputs. This is the standard approach for PTQ evaluation.

    Args:
        weight: (out_features, in_features) float tensor
        nbits: quantization bit-width (2 for our benchmark)
        group_size: tile/group size for 1D tiling

    Returns:
        Reconstructed weight tensor (same shape as input)
    """
    dev = weight.device
    dtype = weight.dtype
    H, W = weight.shape

    max_v = round(2**nbits - 1)
    min_v = 0
    min_max = [min_v, max_v]

    # Move to float32 for quantization
    W_f = weight.float()

    # Tiled quantization
    Q, s1, s2, z, n_w, block = tiled_quant_rectangle(W_f, min_max, group_size)

    # Dequantize tile-by-tile to reconstruct
    W_recon = torch.zeros_like(W_f)
    for i in range(n_w):
        q_tile = Q[:, i * block:(i + 1) * block].float()
        z_tile = z[i * H:(i + 1) * H]  # per-tile per-row zeros
        s1_tile = s1[i * H:(i + 1) * H]  # per-tile per-row scale
        s2_tile = s2[:, i * block:(i + 1) * block]  # per-tile column scale

        # Dequantize: (q - z) * s1 * s2
        W_recon[:, i * block:(i + 1) * block] = (q_tile - z_tile) * s1_tile * s2_tile

    return W_recon.to(dtype)


# =====================================================================
# Model utilities
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


def get_model(model_name):
    """Load model using our standard infrastructure."""
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    downloads_dir = "./downloads"

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


def get_linear_layers(model):
    """Find all nn.Linear layers excluding lm_head and embeddings."""
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip lm_head (output projection) — same as SINQ's _IGNORE_LINEAR
            if 'lm_head' in name:
                continue
            layers[name] = module
    return layers


# =====================================================================
# SINQ Quantization Entry Point
# =====================================================================

@torch.no_grad()
def sinq_quantize_model(model, nbits=2, group_size=64, device='cuda:0'):
    """
    Apply SINQ quantization to all linear layers in the model.

    Since SINQ is calibration-free, we simply:
    1. For each linear layer, quantize + dequantize the weight in-place
    2. No forward passes needed during quantization
    """
    dev = torch.device(device)
    linear_layers = get_linear_layers(model)
    n_layers = len(linear_layers)

    print(f"\n{'='*60}")
    print(f"SINQ W{nbits} g{group_size} Quantization (calibration-free)")
    print(f"Linear layers to quantize: {n_layers}")
    print(f"{'='*60}\n")

    total_params = 0
    total_start = time.time()

    for i, (name, layer) in enumerate(linear_layers.items()):
        w = layer.weight.data
        H, W_dim = w.shape
        n_params = H * W_dim

        # Move weight to device for quantization
        w_dev = w.to(dev)

        layer_start = time.time()
        w_quant = sinq_quantize_dequantize(w_dev, nbits=nbits, group_size=group_size)
        layer_time = time.time() - layer_start

        # Compute reconstruction error
        mse = ((w_dev.float() - w_quant.float()) ** 2).mean().item()
        rel_err = (w_dev.float() - w_quant.float()).norm() / w_dev.float().norm()

        # Write back to CPU
        layer.weight.data = w_quant.to(w.device)

        total_params += n_params

        if (i + 1) % 10 == 0 or (i + 1) == n_layers:
            print(f"  [{i+1}/{n_layers}] {name}: "
                  f"({H}x{W_dim}), MSE={mse:.6f}, relErr={rel_err:.4f}, "
                  f"{layer_time:.2f}s", flush=True)

        del w_dev, w_quant
        torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\nTotal quantization time: {total_time:.1f}s")
    print(f"Total parameters quantized: {total_params:,}")
    return model


# =====================================================================
# BPW Calculation
# =====================================================================

def compute_bpw(model, nbits, group_size):
    """
    Compute effective bits-per-weight for SINQ.

    Storage per tile of (H x group_size):
      - nbits per weight for codes
      - Per-row scale s1: H * 16 bits (float16)
      - Per-row zero z: H * 16 bits (float16)
      - Per-col scale s2: group_size * 16 bits (float16)

    Overhead per weight = (H * 32 + group_size * 16) / (H * group_size)
    """
    linear_layers = get_linear_layers(model)
    total_weights = 0
    total_bits = 0

    for name, layer in linear_layers.items():
        H, W_dim = layer.weight.shape
        n_weights = H * W_dim
        total_weights += n_weights

        # Number of tiles
        gs = group_size
        while gs >= 16 and (W_dim % gs) != 0:
            gs //= 2
        n_tiles = W_dim // gs

        # Bits for codes
        code_bits = n_weights * nbits

        # Bits for scales/zeros per tile: s1 (H x 16bit), z (H x 16bit), s2 (gs x 16bit)
        overhead_bits = n_tiles * (H * 16 + H * 16 + gs * 16)

        total_bits += code_bits + overhead_bits

    bpw = total_bits / total_weights if total_weights > 0 else 0
    return bpw


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_ppl(model, testenc, args):
    """Evaluate perplexity using our standard evaluation code."""
    dev = torch.device(args.device)
    model_type = detect_model_type(model)
    model_short = args.model.split('/')[-1]
    save_title = f"sinq_W{args.nbits}g{args.group_size}_{model_short}_{args.dataset}_seed{args.seed}"

    if model_type == 'opt':
        ppl = opt_eval(model, testenc, dev, args.dataset, save_title=save_title)
    elif model_type == 'llama':
        ppl = llama_eval(model, testenc, dev, args.dataset, save_title=save_title)
    elif model_type == 'qwen':
        ppl = qwen_eval(model, testenc, dev, args.dataset, save_title=save_title)
    return ppl


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SINQ (Huawei 2025) quantization benchmark"
    )
    parser.add_argument('model', type=str, help='HuggingFace model name')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'c4'])
    parser.add_argument('--nbits', type=int, default=2,
                        help='Quantization bit-width')
    parser.add_argument('--group_size', type=int, default=64,
                        help='Group size for 1D tiling (SINQ default=64)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
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

    # Compute bpw before quantization (uses layer shapes)
    bpw = compute_bpw(model, args.nbits, args.group_size)

    # Load test data for evaluation
    from datautils import get_loaders
    tokenizer = get_tokenizer(args.model)
    _, testenc = get_loaders(args.dataset, seed=args.seed, seqlen=model.seqlen, model=args.model)

    # Quantize
    tick = time.time()
    model = sinq_quantize_model(
        model, nbits=args.nbits, group_size=args.group_size, device=args.device
    )
    quant_time = time.time() - tick

    extra = {"nbits": args.nbits, "group_size": args.group_size}
    def _csv(dataset, metric, value):
        csv_append(model=args.model, method="sinq", dataset=dataset,
                   metric=metric, value=value, bpw=bpw, seed=args.seed,
                   blocksize=args.group_size, salient_metric="",
                   extra_params=extra, quantization_time_s=quant_time)

    # Evaluate PPL
    print(f"\n{'='*60}")
    print(f"Evaluating perplexity on {args.dataset}...")
    print(f"{'='*60}")
    ppl = evaluate_ppl(model, testenc, args)
    _csv(args.dataset, "perplexity", ppl)

    model_short = args.model.split('/')[-1]
    print(f"\n{'='*60}")
    print(f"RESULT: SINQ W{args.nbits}g{args.group_size} on {model_short}")
    print(f"  {args.dataset} PPL: {ppl:.2f}")
    print(f"  Seed: {args.seed}")
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
