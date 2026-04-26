"""
AQLM: Additive Quantization of Language Models (ICML 2024)
Standalone runner for apples-to-apples comparison with DOML

Based on: https://github.com/Vahe1994/AQLM
Paper: https://arxiv.org/abs/2401.06118

Core algorithm:
  1. Capture input activations for each transformer layer (calibration data)
  2. Compute Hessian (XTX = X^T @ X) for each linear sublayer
  3. Initialize codebooks via residual k-means on weight groups
  4. Alternate: Adam optimize codebooks → beam search update codes
  5. Layer-wise finetuning to minimize output MSE
  6. Replace nn.Linear with QuantizedLinear (on-the-fly dequant during inference)

AQLM uses additive vector quantization: weights are grouped into 8D vectors,
each vector is encoded by an index into a learned codebook of 2^16 entries.
With 1 codebook × 16 bits per 8 weights = 2 bits per weight.

Usage:
    source env/bin/activate
    python3 -u src/run_aqlm.py Qwen/Qwen3-0.6B wikitext2 --device cuda:0 --seed 0
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.nn as nn

# Add project root to path for our imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add AQLM source to path
AQLM_SOURCE = "/tmp/aqlm_source"
assert os.path.isdir(AQLM_SOURCE), (
    f"AQLM source not found at {AQLM_SOURCE}. "
    f"Clone it first: git clone https://github.com/Vahe1994/AQLM {AQLM_SOURCE}"
)
sys.path.insert(0, AQLM_SOURCE)

from datautils import get_tokenizer, set_seed


# =====================================================================
# Patch AQLM to support Qwen3 (Qwen3 is Llama-like but model_type="qwen3")
# =====================================================================

def _patch_aqlm_for_qwen3():
    """
    AQLM's modelutils.py has LLAMA_LIKE = ("llama", "Yi", "mistral", "mixtral", "gemma", "cohere", "qwen2")
    but Qwen3 uses model_type="qwen3". Patch it to include "qwen3".
    Also patch get_sequential_groups to handle qwen3.
    Also patch get_inps to use a Catcher that proxies __getattr__ (needed for Qwen3's attention_type).
    """
    import src.modelutils as aqlm_modelutils

    # Add qwen3 to LLAMA_LIKE
    if "qwen3" not in aqlm_modelutils.LLAMA_LIKE:
        aqlm_modelutils.LLAMA_LIKE = aqlm_modelutils.LLAMA_LIKE + ("qwen3",)
        print(f"[AQLM patch] Added 'qwen3' to LLAMA_LIKE: {aqlm_modelutils.LLAMA_LIKE}")

    # Patch get_sequential_groups to handle qwen3 explicitly
    _orig_get_sequential_groups = aqlm_modelutils.get_sequential_groups

    def _patched_get_sequential_groups(model):
        if model.config.model_type == "qwen3":
            # Qwen3 has identical layer structure to Llama
            return [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        return _orig_get_sequential_groups(model)

    aqlm_modelutils.get_sequential_groups = _patched_get_sequential_groups

    # Patch get_inps to use a Catcher that proxies __getattr__
    # (Qwen3's forward loop accesses decoder_layer.attention_type before calling forward)
    import main as aqlm_main
    from src.modelutils import FALCON_TYPES, get_layers

    @torch.no_grad()
    def _patched_get_inps(model, data, model_seqlen, devices, offload_activations):
        """Patched version of AQLM's get_inps with proper attribute proxying in Catcher."""
        print("catching layer inputs from data", flush=True)
        layers = get_layers(model)
        device = devices[0] if not offload_activations else torch.device("cpu")

        if isinstance(data, torch.Tensor) and data.shape[0] == 1:
            num_sequences, num_tokens_dropped = data.numel() // model_seqlen, data.numel() % model_seqlen
            data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
            print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")
            del num_sequences, num_tokens_dropped

        assert all(sequence.shape[1] == model_seqlen for sequence in data)

        emb = model.get_input_embeddings()
        emb_device = emb.weight.device
        if emb_device.type != "cuda":
            emb = emb.to(device)
            if model.config.model_type == "opt":
                model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
                if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                    model.model.decoder.project_in = model.model.decoder.project_in.to(device)
        device = emb.weight.device
        layer_device = next(layers[0].parameters()).device
        layers[0] = layers[0].to(device)

        dtype = next(iter(model.parameters())).dtype
        nsamples_per_device = (len(data) - 1) // len(devices) + 1
        inps = [
            torch.zeros(
                (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, model.config.hidden_size),
                dtype=dtype,
                device=devices[i] if not offload_activations else "cpu",
                pin_memory=offload_activations,
            )
            for i in range(len(devices))
        ]
        forward_arg_names = ["attention_mask", "position_ids"]
        if model.config.model_type.lower() in FALCON_TYPES:
            forward_arg_names.append("alibi")
        # Qwen3 (and newer Llama-like models) pass rotary embeddings as position_embeddings
        if model.config.model_type in ("qwen3", "qwen2"):
            forward_arg_names.append("position_embeddings")

        cache = {"i": 0, "alibi": None}

        class CatcherExit(Exception):
            pass

        class Catcher(nn.Module):
            """Catcher that proxies attribute access to wrapped module."""
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
                inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
                cache["i"] += 1
                for forward_arg_name in forward_arg_names:
                    cache[forward_arg_name] = kwargs.get(forward_arg_name)
                raise CatcherExit()

        layers[0] = Catcher(layers[0])
        saved_num_threads = torch.get_num_threads()
        torch.set_num_threads(min(16, saved_num_threads))
        for batch_inps in data:
            try:
                if isinstance(batch_inps, (list, tuple)):
                    batch_inps, *_ = batch_inps
                batch_inps = batch_inps.to(device)
                model(batch_inps, attention_mask=torch.ones_like(batch_inps))
            except CatcherExit:
                pass

        torch.set_num_threads(saved_num_threads)
        layers[0] = layers[0].module

        layers[0] = layers[0].to(layer_device)
        model.get_input_embeddings().to(emb_device)
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
        torch.cuda.empty_cache()

        forward_args = {k: cache[k] for k in forward_arg_names}
        assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps), "internal error: found empty rows in inps"
        return inps, forward_args

    aqlm_main.get_inps = _patched_get_inps

    # Patch update_outs and _compute_mse_on_batch to handle Qwen3's raw tensor output.
    # Qwen3DecoderLayer.forward returns a tensor, not a tuple like Llama/Mistral.
    # AQLM's code does `layer(...)[0]` which for a tensor indexes the batch dim (wrong),
    # and `outs, *_ = layer(...)` which unpacks over the batch dim (wrong).
    import src.finetune as aqlm_finetune
    from tqdm import trange

    @torch.no_grad()
    def _patched_update_outs(layer, inps_tensor, outs_tensor, compute_mse, **forward_args):
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        out_losses = []
        for j in trange(len(inps_tensor), desc="calc outs after quantization" if compute_mse else "calc outs before quantization", leave=False):
            result = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)
            # Handle both tuple output (Llama) and tensor output (Qwen3)
            outs_batch = result[0] if isinstance(result, tuple) else result
            if compute_mse:
                batch_size = outs_batch.shape[0]
                outs_batch_loss = (
                    (outs_batch - outs_tensor[j].to(device)).float().square().view(batch_size, -1).mean(dim=-1)
                )
                outs_batch_loss /= outs_batch.float().square().view(batch_size, -1).mean(dim=-1).clamp(min=1e-6)
                outs_batch_loss = outs_batch_loss.mean()
                out_losses.append(outs_batch_loss.item())
            outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
        return out_losses

    aqlm_main.update_outs = _patched_update_outs

    # Patch _compute_mse_on_batch to handle tensor output
    import warnings
    import torch.nn.functional as F

    _orig_compute_mse = aqlm_finetune._compute_mse_on_batch

    def _patched_compute_mse_on_batch(layer, batch_iter, **kwargs):
        inps_batch, outs_batch = next(batch_iter)
        inps_batch = inps_batch.to(dtype=torch.float32)
        outs_batch = outs_batch.to(dtype=torch.float32)

        if inps_batch.shape[0] != 1:
            for name, value in list(kwargs.items()):
                if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                    if name not in ("attention_mask", "position_ids"):
                        warnings.warn(f"Tiling an unexpected kwarg {name} over batch size")
                    repeats = [len(inps_batch)] + [1 for _ in range(value.ndim - 1)]
                    kwargs[name] = value.tile(*repeats)

        result = layer(inps_batch, **kwargs)
        # Handle both tuple output (Llama) and tensor output (Qwen3)
        outs_prediction = result[0] if isinstance(result, tuple) else result
        assert outs_prediction.shape == outs_batch.shape, (
            f"Shape mismatch: prediction {outs_prediction.shape} vs target {outs_batch.shape}"
        )
        return F.mse_loss(outs_prediction, outs_batch)

    aqlm_finetune._compute_mse_on_batch = _patched_compute_mse_on_batch


# =====================================================================
# Model loading (our infrastructure)
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


def detect_model_type(model):
    class_name = model.__class__.__name__.lower()
    if 'opt' in class_name:
        return 'opt'
    elif 'llama' in class_name:
        return 'llama'
    elif 'qwen' in class_name:
        return 'qwen'
    raise ValueError(f"Unknown model class: {model.__class__.__name__}")


# =====================================================================
# AQLM Quantization Wrapper
# =====================================================================

def aqlm_quantize_model(model, args):
    """
    Run AQLM quantization on a model using the official AQLM codebase.

    Args:
        model: a pretrained HF model
        args: namespace with AQLM hyperparameters

    Returns:
        (model, bpw): quantized model and effective bits per weight
    """
    # Patch AQLM for Qwen3 support
    _patch_aqlm_for_qwen3()

    # Disable TF32 as required by AQLM
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # Import AQLM modules (after patching)
    from main import quantize_model as aqlm_quantize
    from src.aq import QuantizedWeight

    # Create AQLM args namespace
    aqlm_args = argparse.Namespace(
        # Model
        model_path=args.model,
        model_seqlen=model.seqlen,
        dtype="auto",
        # Data
        dataset=args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        val_size=args.val_size,
        # Quantization architecture
        num_codebooks=args.num_codebooks,
        nbits_per_codebook=args.nbits_per_codebook,
        in_group_size=args.in_group_size,
        out_group_size=args.out_group_size,
        scale_nbits=args.scale_nbits,
        codebook_value_nbits=args.codebook_value_nbits,
        codebook_value_num_groups=args.codebook_value_num_groups,
        # Optimization
        lr=args.lr,
        max_epochs=args.max_epochs,
        steps_per_epoch=args.steps_per_epoch,
        beam_size=args.beam_size,
        relative_mse_tolerance=args.relative_mse_tolerance,
        init_max_iter=args.init_max_iter,
        init_max_points_per_centroid=args.init_max_points_per_centroid,
        use_faiss=False,
        print_frequency=args.print_frequency,
        # Finetuning
        finetune_max_epochs=args.finetune_max_epochs,
        finetune_early_stop=args.finetune_early_stop,
        finetune_lr=args.finetune_lr,
        finetune_batch_size=args.finetune_batch_size,
        finetune_adam_beta1=0.9,
        finetune_adam_beta2=0.95,
        finetune_keep_best=args.finetune_keep_best,
        local_batch_size=args.local_batch_size,
        # Device
        devices=[torch.device(args.device)],
        offload_activations=args.offload_activations,
        # Misc
        skip_out_loss=False,
        true_sequential=args.true_sequential,
        use_checkpointing=False,
        mix_compression=False,
        use_fast_tokenizer=False,
        trust_remote_code=False,
        # Save/Load (not used)
        load=None,
        save=None,
        resume=False,
        on_save=None,
        wandb=False,
        no_quant=False,
        attn_implementation="eager",
    )

    print(f"\n{'='*60}")
    print(f"AQLM Quantization Configuration:")
    print(f"  num_codebooks={args.num_codebooks}, nbits_per_codebook={args.nbits_per_codebook}")
    print(f"  in_group_size={args.in_group_size}, out_group_size={args.out_group_size}")
    print(f"  nsamples={args.nsamples}, val_size={args.val_size}")
    print(f"  max_epochs={args.max_epochs}, steps_per_epoch={args.steps_per_epoch}")
    print(f"  relative_mse_tolerance={args.relative_mse_tolerance}")
    print(f"  finetune_max_epochs={args.finetune_max_epochs}")
    print(f"  beam_size={args.beam_size}")
    print(f"  device={args.device}")
    print(f"{'='*60}\n")

    # Run AQLM quantization
    aqlm_quantize(model, aqlm_args)

    # Compute effective bpw from quantized layers
    total_bits = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedWeight):
            bpw_layer = module.estimate_nbits_per_parameter()
            n_params = module.out_features * module.in_features
            total_bits += bpw_layer * n_params
            total_params += n_params

    bpw = total_bits / total_params if total_params > 0 else 0
    print(f"\nEffective bits per weight: {bpw:.4f}")
    print(f"Total quantized parameters: {total_params:,}")

    return model, bpw


# =====================================================================
# Evaluation
# =====================================================================

# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AQLM (ICML 2024) quantization benchmark"
    )
    parser.add_argument('model', type=str, help='HuggingFace model name')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'c4'],
                        help='Calibration + eval dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')

    # AQLM quantization architecture (2-bit defaults: 1x16 scheme)
    parser.add_argument('--num_codebooks', type=int, default=1,
                        help='Number of codebooks per layer (1 for 2-bit)')
    parser.add_argument('--nbits_per_codebook', type=int, default=16,
                        help='Bits per codebook index (16 for 2-bit with 1 codebook)')
    parser.add_argument('--in_group_size', type=int, default=8,
                        help='Number of input features grouped together (8D vectors)')
    parser.add_argument('--out_group_size', type=int, default=1,
                        help='Number of output features grouped together')
    parser.add_argument('--scale_nbits', type=int, default=0,
                        help='Bits for group-wise scale quantization (0=row-wise only)')
    parser.add_argument('--codebook_value_nbits', type=int, default=16,
                        help='Bits for codebook value storage')
    parser.add_argument('--codebook_value_num_groups', type=int, default=1)

    # Calibration data
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration sequences')
    parser.add_argument('--val_size', type=int, default=0,
                        help='Validation sequences for early stopping')

    # Optimization
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Max beam search epochs per layer')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                        help='Adam steps per beam search epoch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for Adam')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='Beam search width')
    parser.add_argument('--relative_mse_tolerance', type=float, default=0.01,
                        help='Early stopping tolerance (0.01 = stop when <1%% improvement)')
    parser.add_argument('--init_max_iter', type=int, default=100,
                        help='K-means iterations for codebook initialization')
    parser.add_argument('--init_max_points_per_centroid', type=int, default=None,
                        help='Max points per centroid during init')
    parser.add_argument('--print_frequency', type=int, default=10)

    # Finetuning
    parser.add_argument('--finetune_max_epochs', type=int, default=5,
                        help='Max finetuning epochs per layer (0 = no finetuning)')
    parser.add_argument('--finetune_early_stop', type=int, default=3)
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--finetune_batch_size', type=int, default=1)
    parser.add_argument('--finetune_keep_best', action='store_true')
    parser.add_argument('--local_batch_size', type=int, default=None)
    parser.add_argument('--offload_activations', action='store_true',
                        help='Offload activations to CPU to save GPU memory')
    parser.add_argument('--true_sequential', action='store_true',
                        help='Process sublayers sequentially (more accurate, slower)')

    # Downstream eval — flags provided by add_eval_cli (--full_eval, --eval_extra_ppl,
    # --ppl_eval_seqlen, --eval_arc, --eval_mmlu, --eval_hellaswag).
    from csv_utils import append_result as csv_append
    from eval_utils import add_eval_cli, resolve_eval_flags, evaluate_and_log_all
    add_eval_cli(parser)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    model = get_model(args.model)

    # Quantize with AQLM
    tick = time.time()
    model, bpw = aqlm_quantize_model(model, args)
    quant_time = time.time() - tick

    scheme = f"{args.num_codebooks}x{args.nbits_per_codebook}g{args.in_group_size}"
    extra = {
        "num_codebooks": args.num_codebooks,
        "nbits_per_codebook": args.nbits_per_codebook,
        "in_group_size": args.in_group_size,
        "out_group_size": args.out_group_size,
        "nsamples": args.nsamples,
        "max_epochs": args.max_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "relative_mse_tolerance": args.relative_mse_tolerance,
        "finetune_max_epochs": args.finetune_max_epochs,
        "beam_size": args.beam_size,
        "scheme": scheme,
    }

    eval_flags = resolve_eval_flags(args, primary_dataset=args.dataset)

    model_short = args.model.split('/')[-1]
    print(f"\n{'='*60}")
    print(f"RESULT: AQLM {scheme} on {model_short}")
    print(f"  Seed: {args.seed}")
    print(f"  Effective bpw: ~{bpw:.4f}")
    print(f"  Quantization time: {quant_time:.1f}s")
    print(f"  PPL eval datasets: {eval_flags['ppl_datasets']}")
    print(f"{'='*60}")

    evaluate_and_log_all(
        model, args.model, torch.device(args.device),
        method="aqlm",
        bpw=bpw, seed=args.seed, blocksize=args.in_group_size,
        salient_metric="",
        extra_params=extra,
        quantization_time_s=quant_time,
        ppl_datasets=eval_flags["ppl_datasets"],
        eval_mmlu=eval_flags["eval_mmlu"],
        eval_hellaswag=eval_flags["eval_hellaswag"],
        eval_arc=eval_flags["eval_arc"],
        ppl_eval_seqlen=eval_flags["ppl_eval_seqlen"],
        save_title_prefix=f"aqlm_{scheme}_{model_short}_seed{args.seed}",
    )


if __name__ == '__main__':
    main()
