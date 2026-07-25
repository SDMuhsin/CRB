"""Joint pruning + quantization baselines (JSQ, SLiM) on the CRB harness.

Standalone runner mirroring src/run_sparsegpt.py / src/run_tesseraq.py conventions:
same model loaders, same C4/wikitext2 calibration protocol, same eval suite +
CSV writer, so rows drop next to the paper's SparseGPT rows unchanged.

Methods (core math vendored from the PRISM repo, component-tested against the
upstream releases there — see src/jsq_port.py / src/slim_port.py headers):

  jsq-wo  JSQ (Guo et al., ICML 2024) weight-only: per-layer calib of clipped
          input stats -> prune by |W|*sqrt(||x||^2) + rho * SAR sensitivity ->
          SmoothQuant alpha=0.5 LN->fc migration -> per-out-channel symmetric
          absmax RTN on survivors. Weight-only for apples-to-apples with the
          weight-only SparseGPT/SDOML rows.
  jsq     Faithful JSQ WnAn (weights AND activations fake-quantized to nbits,
          q/k/v outputs included). NOT rate-comparable to weight-only rows.
  slim    SLiM-LoRA (Mozaffari et al., ICML 2025): Wanda prune (shift-zeros
          act norms) -> per-matrix MSE-optimal-cap symmetric quant -> rank-
          (0.1*min_dim) fp16 saliency-SVD adapter on the compression error.
          The adapter is REAL extra rate; bpw below counts it exactly.

Sequential per-block harness: block i's calibration inputs are the outputs of
the already-compressed blocks 0..i-1 (same error propagation as the PRISM
full-model-forward harness, without its try/except dead forwards).

bpw accounting (matches src/run_sparsegpt.py convention, mask = 1 bpw):
  jsq-wo/jsq: 1 + (1-s)*nbits + 16/d_in            (fp16 scale per out row)
  slim:       1 + (1-s)*nbits + 32/2**avg_bins ... computed EXACTLY instead:
              per-linear adapter bits 16*r*(in+out) and per-matrix scale are
              summed over all compressed linears and divided by total weights.
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datautils import set_seed
from run_tesseraq import (
    get_model,
    get_calibration_data,
    get_blocks_and_kwargs,
    get_linear_layers,
    set_module_by_name,
)
from jsq_port import (
    JSQStatWrapper,
    WnAnLinear,
    generate_ss,
    smooth_ln_fcs,
)
from slim_port import SLiMLinear, shift_zeros, slim_lora_decompose


class WandaLayerWrapper:
    """Per-column L2-norm^2 running mean of a linear's inputs (Wanda scaler_row).

    Port of the PRISM/upstream-Wanda wrapper (benchmark_suite.py:1202)."""

    def __init__(self, layer):
        self.columns = layer.weight.shape[1]
        self.scaler_row = torch.zeros(self.columns, device=layer.weight.device)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t().float()                       # (in, tokens)
        t = inp.shape[1]
        self.scaler_row *= self.nsamples / (self.nsamples + t)
        self.nsamples += t
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


QKV_NAMES = {'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'}

# [zeros, total] over all replaced linears — printed at the end as the honest
# measured sparsity (must be >= --sparsity; quant maps 0 -> 0 in both methods).
_SPARSITY_CHECK = [0, 0]


def _calibrate_block(block, subset, inps, layer_kwargs, dev, wrapper_factory):
    """Run cached inputs through this block with per-linear hooks; return wrappers."""
    wrapped = {name: wrapper_factory(lin) for name, lin in subset.items()}

    def make_hook(name):
        def hook_fn(module, hin, hout):
            wrapped[name].add_batch(hin[0].data)
        return hook_fn

    handles = [lin.register_forward_hook(make_hook(name))
               for name, lin in subset.items()]
    with torch.no_grad():
        for j in range(inps.shape[0]):
            block(inps[j].unsqueeze(0).to(dev), **layer_kwargs)
    for h in handles:
        h.remove()
    return wrapped


@torch.no_grad()
def _recompute_outs(block, inps, layer_kwargs, dev):
    outs = torch.zeros_like(inps)
    for j in range(inps.shape[0]):
        outs[j] = block(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0].squeeze(0)
    return outs


def _wanda_mask(W, scaler_row, sparsity):
    """Per-output-row unstructured mask of the smallest |W|*sqrt(scaler_row)."""
    W_metric = torch.abs(W) * torch.sqrt(scaler_row.reshape(1, -1))
    n_prune = int(W.shape[1] * sparsity)
    mask = torch.zeros_like(W_metric, dtype=torch.bool)
    if n_prune > 0:
        idx = torch.sort(W_metric, dim=1, stable=True)[1][:, :n_prune]
        mask.scatter_(1, idx, True)
    return mask


def compress_jsq(model, trainloader, args, weight_only=True):
    """JSQ per-block: calib clipped stats -> joint prune -> smooth -> RTN quant.

    Faithful to the PRISM _apply_jsq driver (itself component-equivalent to
    upstream temp/JSQ): rho=2.1, clip_h=0.01, alpha=0.5, paper-Frobenius Wanda
    term (JSQ_FROB=1 branch), smoothing in BOTH variants."""
    dev = torch.device(args.device)
    nsamples = len(trainloader)
    layers, inps, layer_kwargs = get_blocks_and_kwargs(
        model, dev, trainloader, nsamples, model.seqlen)

    for i in range(len(layers)):
        t0 = time.time()
        block = layers[i].to(dev)
        subset = {n: m for n, m in get_linear_layers(block).items()}

        wrapped = _calibrate_block(
            block, subset, inps, layer_kwargs, dev,
            lambda lin: JSQStatWrapper(lin, clip_h=args.clip_h))

        # ---- prune with the joint metric ----
        for name, linear in subset.items():
            W = linear.weight.data.float()
            scaler_row = wrapped[name].scaler_row
            ss = generate_ss(wrapped[name].mean_activation(), W)
            ntok = max(wrapped[name].nsamples, 1)
            wanda = torch.abs(W) * torch.sqrt(scaler_row.reshape(1, -1) * ntok)
            W_metric = wanda + args.rho * ss
            n_prune = int(W.shape[1] * args.sparsity)
            if n_prune > 0:
                idx = torch.sort(W_metric, dim=1, stable=True)[1][:, :n_prune]
                mask = torch.zeros_like(W_metric, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                W[mask] = 0
                linear.weight.data = W.to(linear.weight.dtype)

        # ---- SmoothQuant smoothing (both variants; helps weight-only too) ----
        if not args.no_smooth:
            if hasattr(block, 'input_layernorm') and hasattr(block, 'self_attn'):
                qkv = [block.self_attn.q_proj, block.self_attn.k_proj,
                       block.self_attn.v_proj]
                smooth_ln_fcs(block.input_layernorm, qkv,
                              wrapped['self_attn.q_proj'].act_max, args.alpha)
            if hasattr(block, 'post_attention_layernorm') and hasattr(block, 'mlp') \
                    and hasattr(block.mlp, 'gate_proj'):
                smooth_ln_fcs(block.post_attention_layernorm,
                              [block.mlp.gate_proj, block.mlp.up_proj],
                              wrapped['mlp.gate_proj'].act_max, args.alpha)

        # ---- quantize: replace linears with WnAnLinear ----
        for name, linear in subset.items():
            new_linear = WnAnLinear.from_float(
                linear, nbits=args.nbits,
                act_quant=not weight_only,
                quantize_output=(not weight_only) and (name in QKV_NAMES),
            ).to(dev)
            set_module_by_name(block, name, new_linear)
            _SPARSITY_CHECK[0] += (new_linear.weight.data == 0).sum().item()
            _SPARSITY_CHECK[1] += new_linear.weight.data.numel()
            del linear

        inps = _recompute_outs(block, inps, layer_kwargs, dev)
        layers[i] = block.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  block {i}/{len(layers)} done in {time.time()-t0:.1f}s", flush=True)
    return model


def compress_slim(model, trainloader, args):
    """SLiM-LoRA per-block: Wanda calib -> prune -> SLiM-Quant + saliency LoRA.

    Faithful to the PRISM apply_slim_quantization driver (component-bit-exact
    port of upstream temp/SLiM): shift_zero_metrics on, rank_ratio default 0.1,
    per-matrix MSE-optimal-cap symmetric quant, fp16 adapter.

    Returns (model, adapter_bits_total, weight_count_total) for exact bpw."""
    dev = torch.device(args.device)
    nsamples = len(trainloader)
    layers, inps, layer_kwargs = get_blocks_and_kwargs(
        model, dev, trainloader, nsamples, model.seqlen)

    adapter_bits = 0
    weight_count = 0
    for i in range(len(layers)):
        t0 = time.time()
        block = layers[i].to(dev)
        subset = {n: m for n, m in get_linear_layers(block).items()}

        wrapped = _calibrate_block(
            block, subset, inps, layer_kwargs, dev,
            lambda lin: WandaLayerWrapper(lin))

        for name, linear in subset.items():
            W = linear.weight.data
            scaler_row = shift_zeros(wrapped[name].scaler_row.float())
            mask = _wanda_mask(W, scaler_row, args.sparsity)
            Wc, ll, lr = slim_lora_decompose(
                W, mask, scaler_row, args.nbits, args.rank_ratio,
                quantize=True, slim_lora=True)
            bias = linear.bias.data.clone() if linear.bias is not None else None
            new_linear = SLiMLinear(
                Wc.to(W.dtype), ll.to(W.dtype), lr.to(W.dtype), bias).to(dev)
            set_module_by_name(block, name, new_linear)
            adapter_bits += 16 * (ll.numel() + lr.numel())
            weight_count += W.numel()
            _SPARSITY_CHECK[0] += (new_linear.weight.data == 0).sum().item()
            _SPARSITY_CHECK[1] += new_linear.weight.data.numel()
            del linear

        inps = _recompute_outs(block, inps, layer_kwargs, dev)
        layers[i] = block.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  block {i}/{len(layers)} done in {time.time()-t0:.1f}s", flush=True)
    return model, adapter_bits, weight_count


def main():
    parser = argparse.ArgumentParser(
        description="Joint pruning+PTQ baselines (JSQ / SLiM) on the CRB harness")
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'c4'],
                        help='primary eval dataset (house rows use wikitext2)')
    parser.add_argument('--method', type=str, required=True,
                        choices=['jsq-wo', 'jsq', 'slim'])
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--nbits', type=int, default=2, choices=[2, 3, 4, 8])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--calib_dataset', type=str, default='c4',
                        choices=['wikitext2', 'c4'],
                        help='house SparseGPT rows calibrate on c4 (128x2048)')
    # JSQ knobs (paper-faithful defaults, as in the PRISM port)
    parser.add_argument('--rho', type=float, default=2.1)
    parser.add_argument('--clip_h', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--no_smooth', action='store_true')
    # SLiM knobs
    parser.add_argument('--rank_ratio', type=float, default=0.1)
    from csv_utils import append_result as _  # noqa: F401  (import check only)
    from eval_utils import add_eval_cli, resolve_eval_flags, evaluate_and_log_all
    add_eval_cli(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    model = get_model(args.model)
    if args.seqlen and args.seqlen < model.seqlen:
        model.seqlen = args.seqlen

    print(f"Calibration: {args.calib_dataset} x {args.nsamples} @ seqlen {model.seqlen}")
    trainloader, _testenc = get_calibration_data(
        args.model, args.nsamples, args.seed, model.seqlen,
        dataset=args.calib_dataset)

    d_in = model.config.hidden_size
    spct = int(round(args.sparsity * 100))
    tick = time.time()
    if args.method in ('jsq-wo', 'jsq'):
        model = compress_jsq(model, trainloader, args,
                             weight_only=(args.method == 'jsq-wo'))
        # mask(1) + survivors at nbits + one fp16 scale per output row
        bpw = 1.0 + (1.0 - args.sparsity) * args.nbits + 16.0 / d_in
        method_tag = f"{args.method}-s{spct}-w{args.nbits}"
        extra = {"sparsity": args.sparsity, "nbits": args.nbits,
                 "rho": args.rho, "clip_h": args.clip_h, "alpha": args.alpha,
                 "smooth": not args.no_smooth,
                 "calib_dataset": args.calib_dataset,
                 "nsamples": args.nsamples, "calib_seqlen": model.seqlen}
    else:
        model, adapter_bits, weight_count = compress_slim(model, trainloader, args)
        adapter_bpw = adapter_bits / max(weight_count, 1)
        bpw = 1.0 + (1.0 - args.sparsity) * args.nbits + adapter_bpw
        method_tag = f"slim-s{spct}-w{args.nbits}"
        extra = {"sparsity": args.sparsity, "nbits": args.nbits,
                 "rank_ratio": args.rank_ratio,
                 "adapter_bpw": round(adapter_bpw, 4),
                 "calib_dataset": args.calib_dataset,
                 "nsamples": args.nsamples, "calib_seqlen": model.seqlen}
        print(f"SLiM fp16 adapter overhead: {adapter_bpw:.3f} bpw "
              f"(counted in the reported bpw)")
    quant_time = time.time() - tick

    measured_s = _SPARSITY_CHECK[0] / max(_SPARSITY_CHECK[1], 1)
    extra["measured_sparsity"] = round(measured_s, 4)
    if measured_s < args.sparsity - 0.005:
        raise RuntimeError(
            f"measured weight sparsity {measured_s:.4f} < requested "
            f"{args.sparsity} — mask not applied, refusing to log")

    eval_flags = resolve_eval_flags(args, primary_dataset=args.dataset)
    model_short = args.model.split('/')[-1]
    print(f"\n{'='*60}")
    print(f"RESULT: {method_tag} on {model_short}")
    print(f"  sparsity={args.sparsity} (measured {measured_s:.4f}) "
          f"nbits={args.nbits} seed={args.seed}")
    print(f"  bpw (mask+survivors+overhead): {bpw:.4f}")
    print(f"  compression time: {quant_time:.1f}s")
    print(f"{'='*60}")

    evaluate_and_log_all(
        model, args.model, torch.device(args.device),
        method=method_tag,
        bpw=round(bpw, 4), seed=args.seed, blocksize=0,
        salient_metric="",
        extra_params=extra,
        quantization_time_s=quant_time,
        ppl_datasets=eval_flags["ppl_datasets"],
        eval_mmlu=eval_flags["eval_mmlu"],
        eval_hellaswag=eval_flags["eval_hellaswag"],
        eval_arc=eval_flags["eval_arc"],
        ppl_eval_seqlen=eval_flags["ppl_eval_seqlen"],
        save_title_prefix=f"{method_tag}_{model_short}_seed{args.seed}",
    )


if __name__ == '__main__':
    main()
