import math
import time
import sys

import torch
import torch.nn as nn

from bigptq import BRAGPTQ
from binary import Binarization
from modelutils import find_layers
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from csv_utils import append_result as _csv_append
from eval_utils import add_eval_cli, resolve_eval_flags, evaluate_and_log_all


downloads_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")
def get_model(model_name):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model_path = os.path.join(downloads_dir, f"DOWNLOAD_{model_name}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directories exist
    
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model = torch.load(model_path)
    else:
        print(f"Downloading and saving model: {model_name}")
        if "opt" in model_name:
            from transformers import OPTForCausalLM
            model = OPTForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, use_safetensors=True, attn_implementation="eager")
            model.seqlen = model.config.max_position_embeddings
        elif "llama" in model_name.lower() or "danube" in model_name.lower() or "falcon" in model_name.lower() or "helium" in model_name.lower():
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, use_safetensors=True, attn_implementation="eager")
            model.seqlen = 2048
        elif "qwen" in model_name.lower():
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, use_safetensors=True, attn_implementation="eager")
            model.seqlen = min(model.config.max_position_embeddings, 2048)
        elif "smollm" in model_name.lower():
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, use_safetensors=True, attn_implementation="eager")
            model.seqlen = min(model.config.max_position_embeddings, 2048)
        elif "olmo" in model_name.lower():
            # Olmo2ForCausalLM (e.g. allenai/OLMo-2-0425-1B): llama-style
            # model.model.layers access, loaded via AutoModelForCausalLM.
            # OLMo-2 checkpoints are stored fp32; load in bf16 to match every
            # other model family (doml_dump.derive_dpk asserts bf16 weights).
            import torch as _torch
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_torch.bfloat16, cache_dir=downloads_dir, use_safetensors=True, attn_implementation="eager")
            model.seqlen = min(model.config.max_position_embeddings, 2048)
        elif "pythia" in model_name.lower():
            from transformers import GPTNeoXForCausalLM
            import torch as _torch
            # Pythia fp16 + eager overflows in attention (NaN at seqlen>4). Use bf16 instead.
            model = GPTNeoXForCausalLM.from_pretrained(model_name, torch_dtype=_torch.bfloat16, cache_dir=downloads_dir, attn_implementation="eager")
            model.seqlen = model.config.max_position_embeddings
        elif "bloom" in model_name.lower():
            from transformers import BloomForCausalLM
            model = BloomForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, attn_implementation="eager")
            model.seqlen = 2048
        elif "granite" in model_name.lower():
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, use_safetensors=True, attn_implementation="eager")
            model.seqlen = min(model.config.max_position_embeddings, 2048)
        else:
            raise ValueError("Unsupported model type")
        
        #torch.save(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model

@torch.no_grad()
def _secq_capture_and_quantize(layer, subset, names, inps, layer_kwargs, nsamples):
    """SECQ helper: capture inputs for named sublayers via forward pass, then quantize."""
    gptq = {}
    for name in names:
        braq_quantizer = Binarization(
            subset[name].weight,
            method='braq',  # SECQ uses braq for underlying binarization
            groupsize=groupsize,
            corr_damp=args.corr_damp,
            lam=args.lam,
            coupling=args.coupling,
        )
        gptq[name] = BRAGPTQ(
            subset[name],
            braq_quantizer,
            salient_metric=args.salient_metric,
            disable_gptq=args.disable_gptq,
        )

    def add_batch(name):
        def tmp(_, inp, out):
            gptq[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in gptq:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    for j in range(nsamples):
        layer(inps[j].unsqueeze(0).to(next(layer.parameters()).device), **layer_kwargs)
    for h in handles:
        h.remove()

    for name in gptq:
        print(f"  SECQ phase: {name}")
        print("Quantizing ...")
        gptq[name].fasterquant(
            percdamp=args.percdamp,
            blocksize=args.blocksize,
        )
        gptq[name].free()

    del gptq
    torch.cuda.empty_cache()


'''
The function is employed to calibrate and quantize models layer by layer.
'''
@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
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
    elif "llama" in args.model.lower() or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower() or "falcon" in args.model.lower() or "helium" in args.model.lower() or "olmo" in args.model.lower():
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    elif "pythia" in args.model.lower():
        layers = model.gpt_neox.layers
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(dev)
        if hasattr(model.gpt_neox, "rotary_emb"):
            model.gpt_neox.rotary_emb = model.gpt_neox.rotary_emb.to(dev)
    elif "bloom" in args.model.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    act_bytes = args.nsamples * model.seqlen * model.config.hidden_size * 2
    act_device = 'cpu' if act_bytes > 8 * (1024**3) else dev
    if act_device == 'cpu':
        print(f"Activation offload: {act_bytes / 1024**3:.1f} GB > 8 GB threshold, using CPU")
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=act_device
    )
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module): # Cache["i"] stores index of attention mask, and Cache["attention_mask"] stores attention mask itself
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
            model(batch[0].to(dev)) # Pass first batch through the model
            # This should capture attention masks into inps
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
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
    elif "llama" in args.model.lower() or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower() or "falcon" in args.model.lower() or "helium" in args.model.lower() or "olmo" in args.model.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif "pythia" in args.model.lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.cpu()
        if hasattr(model.gpt_neox, "rotary_emb"):
            model.gpt_neox.rotary_emb = model.gpt_neox.rotary_emb.cpu()
    elif "bloom" in args.model.lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    print("Ready.")

    for i in range(len(layers)):

        layer = layers[i].to(dev)

        subset = find_layers(layer)

        if args.low_quant_method == 'secq':
            # === GLU-aware mixed precision ===
            # Give gate+up projections more binary planes (order=2 for all partitions)
            # to directly reduce the cross-term Δg⊙Δu. Cross-term is quadratic in
            # per-matrix error, so halving each error quarters the cross-term.
            # Everything else stays at standard BRAQ precision.
            # Avg bitwidth: gate/up ~2 bits, attn/down ~1.1 bits → overall ~1.4 bits.

            print(f"Layer {i} — GLU-aware mixed precision")

            gptq = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
                ) == (not args.invert):
                    continue
                braq_quantizer = Binarization(
                    subset[name].weight,
                    method='braq',
                    groupsize=groupsize,
                    corr_damp=args.corr_damp,
                    lam=args.lam,
                    coupling=args.coupling,
                )
                gptq[name] = BRAGPTQ(
                    subset[name],
                    braq_quantizer,
                    salient_metric=args.salient_metric,
                    disable_gptq=args.disable_gptq,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in gptq:
                print(i, name)
                print("Quantizing ...")
                # GLU-aware: gate/up get order=2 for ALL partitions (more bits)
                if 'gate_proj' in name or 'up_proj' in name:
                    gu_order = 3  # configurable: 2 or 3 binary planes
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        partition=args.partition,
                        orders=(gu_order, gu_order, gu_order),
                    )
                else:
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        partition=args.partition,
                    )
                gptq[name].free()

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]

            del gptq

        else:
            # === Standard: all sublayers quantized with single-pass inputs ===
            ATTN_SUBLAYERS = {'self_attn.q_proj', 'self_attn.k_proj',
                              'self_attn.v_proj', 'self_attn.o_proj'}
            gptq = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
                ) == (not args.invert):
                    continue
                # Mixed mode: ternary for attention, braq for MLP
                if args.low_quant_method == 'mixed':
                    sublayer_method = 'ternary' if name in ATTN_SUBLAYERS else 'braq'
                else:
                    sublayer_method = args.low_quant_method
                braq_quantizer = Binarization(
                    subset[name].weight,
                    method=sublayer_method,
                    groupsize=groupsize,
                    corr_damp = args.corr_damp,
                    lam = args.lam,
                    coupling = args.coupling
                )
                # Codebook size for the DOML / SDOML / magfit family.
                # K = 2**codebook_bits levels per row. Read by binary.py
                # dispatch and the SDOML branches in bigptq.fasterquant.
                # doml_binary is hardcoded to K=2 in binary.py and ignores
                # this attribute.
                if sublayer_method in ('doml', 'sdoml', 'sdoml_partition', 'magfit'):
                    braq_quantizer.codebook_K = 2 ** int(args.codebook_bits)
                # SDOML: forward sparsity + n_iter onto the quantizer so
                # bigptq.fasterquant can read them via getattr. Per derivation
                # §1.3 and S4 contract. n_iter=1 selects the 1-pass ablation
                # (no joint alternation; codebook-only contribution).
                if args.low_quant_method == 'sdoml':
                    braq_quantizer.sparsity = float(args.sparsity)
                    braq_quantizer.sdoml_n_iter = int(args.sdoml_n_iter)
                # magfit (S6 ablation): magnitude-prune-then-LMQ, no joint
                # alternation. Sparsity is read by the magfit branch in
                # bigptq.fasterquant. Cleanly separate from sdoml — uses a
                # different binary.py kernel (magfit_quantize) per S6 contract.
                if args.low_quant_method == 'magfit':
                    braq_quantizer.sparsity = float(args.sparsity)
                # sdoml_partition (S8): SDOML applied independently within each
                # of 3 DOML structural column partitions. Same per-row sparsity
                # as base SDOML; routes through the partition==3 + is_sdoml_partition
                # branch in bigptq.fasterquant.
                if args.low_quant_method == 'sdoml_partition':
                    braq_quantizer.sparsity = float(args.sparsity)
                    braq_quantizer.sdoml_n_iter = int(args.sdoml_n_iter)
                    # S9 (2026-05-03): when --sdoml_asymmetric, set per-
                    # partition sparsity vector [s, 0, 0]: bulk sparse,
                    # mid + salient dense. Routes through sdoml_partition_
                    # quantize's dense-path branch for mid + salient.
                    if getattr(args, 'sdoml_asymmetric', False):
                        braq_quantizer.per_partition_sparsity = [
                            float(args.sparsity), 0.0, 0.0,
                        ]
                gptq[name] = BRAGPTQ(
                    subset[name],
                    braq_quantizer,
                    salient_metric=args.salient_metric,
                    disable_gptq=args.disable_gptq,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in gptq:
                print(i, name)
                print("Quantizing ...")
                if args.low_quant_method in ('doml', 'doml_binary'):
                    # DOML: per-row Lloyd-Max with structural partition.
                    # `doml_binary` is always partition=3. `doml` honors
                    # --partition for the partition×K reviewer-defense
                    # ablation: partition=3 (default, legacy) gives 3
                    # codebooks per row; partition=1 collapses to a single
                    # per-row codebook (no salient/mid/bulk split).
                    if args.low_quant_method == 'doml' and int(getattr(args, 'partition', 3)) == 1:
                        _doml_partition = 1
                        _doml_orders = (1,)
                    else:
                        _doml_partition = 3
                        _doml_orders = (1, 1, 1)  # order ignored by DOML quantizer
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        partition=_doml_partition,
                        orders=_doml_orders,
                    )
                elif args.low_quant_method == 'sdoml':
                    # SDOML: single per-row codebook (no structural partition).
                    # Routes through the partition==1 + is_sdoml branch in
                    # bigptq.fasterquant (Composition Candidate I, derivation §7.1).
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        partition=1,
                        orders=(1,),  # single-partition order, unused by SDOML
                    )
                elif args.low_quant_method == 'magfit':
                    # magfit (S6 ablation): magnitude-prune-then-LMQ.
                    # Routes through the partition==1 + is_magfit branch in
                    # bigptq.fasterquant. Uses Hessian-derived col_weights for
                    # the LMQ centroid step; the prune step uses |x|*sqrt(w_i)
                    # (matches BiLLM-style salience). Same GPTQ residual sweep
                    # as SDOML so the comparison is apples-to-apples.
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        partition=1,
                        orders=(1,),  # single-partition order, unused
                    )
                elif args.low_quant_method == 'sdoml_partition':
                    # SDOML+partition (S8): DOML's 3-way structural partition,
                    # joint per-row SDOML inside each partition. Routes through
                    # the partition==3 + is_sdoml_partition branch in
                    # bigptq.fasterquant.
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        partition=3,
                        orders=(1, 1, 2),  # match DOML's structural defaults
                    )
                else:
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        partition=args.partition,
                        global_scale=args.global_scale,
                    )
                gptq[name].free()

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]

            del gptq

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
'''
    opt braq       ptb : ppl37.62 :  
    opt robq       ptb : ppl31.48 :  
    opt mestrobq   ptb : ppl17.42 :  
    opt medianbraq ptb : ppl700   :
    opt orb        ptb : ppl6000  :
    opt whor       ptb : ppl1000  :
    opt arb   arb(0.5) : ppl500   :
    opt arb   arb(0.9) : ppl33.39 :
    opt arb   arb(0.8) : ppl45 :
    opt crb            : ppl17.32 :

    llama braq     ptb : ppl97 
    llama mestrobq ptb : ppl52.6
    llama crb      ptb : pp55

    
    opt braq  wikitext : ppl41
    opt crb   wikitext : ppl12
    
    llama braq     wikitext  : pp18

    -- above measures used incorrect crb --

    opt1.3B braq ptb                        : ppl 73.81
    opt1.3B crb  ptb                        : ppl 87.83
    opt1.3B crb_stable  ptb                 : ppl 82
    opt1.3B crb_stable_v2  ptb              : ppl 81
    opt1.3B crb_stable_v3  ptb              : ppl 75
    opt1.3B crb_stable_v4  ptb              : ppl 73.28
    opt1.3B crb_stable_v4 cordamp0.2 ptb    : ppl 83 
    opt1.3B crb_stable_v5           ptb     : ppl 65.59 [!]
    opt1.3B crb_stable_v6           ptb     : ppl 63.11 [!]

    opt1.3B braq wikitext2                  : ppl 61.275
    opt1.3B crb  wikitext2                  : ppl 50.70
    opt1.3B crb_stable_v6  wikitext2        : ppl 53.13 [!]
    
    opt2.7B crb  wikitext2                  : ppl 71.49

    opt2.7B braq wikitext2                  : ppl 61.275 ?
    opt2.7B crb  wikitext2                  : ppl 44
    opt2.7B crb_stable_v6    wikitext       : ppl 67 [-]
    opt2.7B crb_stable_v7    wikitext       : ppl 47.34


    opt6.7B braq ptb                        : ppl 35 
    opt6.7B crbv6 ptb                       : ppl 35 [-]
    opt6.7B crbv7 ptb                       : ppl 34.9

    opt6.7b braq             wikitext       : ppl35.84
    opt6.7b crb_stable_v6    wikitext       : ppl36.429  [-]
    opt6.7b crb_stable_v7    wikitext       : ppl

'''


@torch.no_grad()
def sbh_sequential(model, dataloader, dev, r_attn=60, r_mlp=30):
    """Quantize model using Spectral-Binary Hybrid (SVD + binary residual)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from sbh import sbh_quantize_weight_multi, compute_bitrate

    ATTN_NAMES = {'self_attn.q_proj', 'self_attn.k_proj',
                  'self_attn.v_proj', 'self_attn.o_proj'}

    print(f"SBH: r_attn={r_attn}, r_mlp={r_mlp}")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Get layers and move embeddings to device (same as quant_sequential)
    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model.lower() or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower() or "falcon" in args.model.lower() or "helium" in args.model.lower() or "olmo" in args.model.lower():
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    elif "bloom" in args.model.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    else:
        raise ValueError(f"Unsupported model for SBH: {args.model}")

    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
    # Move embeddings back to CPU
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    elif "llama" in args.model.lower() or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower() or "falcon" in args.model.lower() or "helium" in args.model.lower() or "olmo" in args.model.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif "bloom" in args.model.lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    # Compute average bitrate
    total_bits = 0
    total_params = 0

    print("SBH quantization ready.")
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            m, k = W.shape
            r = r_attn if name in ATTN_NAMES else r_mlp
            r = min(r, min(m, k))

            W_q = sbh_quantize_weight_multi(W, rank=r, binary_order=1)
            subset[name].weight.data = W_q

            bpw = compute_bitrate(m, k, r, binary_order=1)
            total_bits += bpw * m * k
            total_params += m * k
            print(f"  Layer {i} {name}: rank={r}, shape={list(W.shape)}, bpw={bpw:.2f}")

        # Compute outputs for next layer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    avg_bpw = total_bits / total_params
    print(f"\nSBH complete. Average bitrate: {avg_bpw:.3f} bits/weight")
    model.config.use_cache = use_cache


@torch.no_grad()
def mixed_sequential(model, dataloader, dev):
    """Quantize with mixed precision: ternary for attention, braq for MLP.

    Uses the GPTQ framework for error correction with per-sublayer quantizer selection.
    """
    KV_SUBLAYERS = {'self_attn.k_proj', 'self_attn.v_proj'}
    GATE_UP_SUBLAYERS = {'mlp.gate_proj', 'mlp.up_proj'}
    ATTN_SUBLAYERS = {'self_attn.q_proj', 'self_attn.k_proj',
                      'self_attn.v_proj', 'self_attn.o_proj'}

    # Per-sublayer order configuration
    qo_order = args.attn_order
    kv_order = args.kv_order if args.kv_order is not None else args.attn_order
    mlp_orders = tuple(args.mlp_orders) if args.mlp_orders is not None else (1, 1, 2)
    gate_up_orders = tuple(args.gate_up_orders) if args.gate_up_orders is not None else mlp_orders

    if args.low_quant_method == 'mixed':
        # Attention gets braq with higher order, MLP gets standard braq
        default_method = 'braq'
        attn_method = 'braq'
        print(f"Mixed mode: QO order={qo_order}, KV order={kv_order}, MLP orders={mlp_orders}, gate/up orders={gate_up_orders}")
    else:
        # Uniform method
        default_method = args.low_quant_method
        attn_method = args.low_quant_method
        print(f"Uniform {args.low_quant_method} for all sublayers")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    elif "llama" in args.model.lower() or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower() or "falcon" in args.model.lower() or "helium" in args.model.lower() or "olmo" in args.model.lower():
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    elif "bloom" in args.model.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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

    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    elif "llama" in args.model.lower() or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower() or "falcon" in args.model.lower() or "helium" in args.model.lower() or "olmo" in args.model.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif "bloom" in args.model.lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]
    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        gptq = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.quant_only in name)) == (not args.invert):
                continue
            sublayer_method = attn_method if name in ATTN_SUBLAYERS else default_method
            braq_quantizer = Binarization(
                subset[name].weight,
                method=sublayer_method,
                groupsize=groupsize,
                corr_damp=args.corr_damp,
                lam=args.lam,
                coupling=args.coupling,
            )
            gptq[name] = BRAGPTQ(
                subset[name],
                braq_quantizer,
                salient_metric=args.salient_metric,
                disable_gptq=args.disable_gptq,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        for name in gptq:
            sublayer_method = attn_method if name in ATTN_SUBLAYERS else default_method
            # For 'mixed' mode with braq: attention gets higher order (more binary planes)
            if args.low_quant_method == 'mixed' and name in ATTN_SUBLAYERS and sublayer_method == 'braq':
                order = kv_order if name in KV_SUBLAYERS else qo_order
                sublayer_orders = (order, order, order)
                print(f"{i} {name} (braq order={order})")
                print("Quantizing ...")
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                    orders=sublayer_orders,
                )
            elif args.low_quant_method == 'mixed' and name not in ATTN_SUBLAYERS:
                sublayer_mlp_orders = gate_up_orders if name in GATE_UP_SUBLAYERS else mlp_orders
                print(f"{i} {name} ({sublayer_method} orders={sublayer_mlp_orders})")
                print("Quantizing ...")
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                    orders=sublayer_mlp_orders,
                )
            else:
                print(f"{i} {name} ({sublayer_method})")
                print("Quantizing ...")
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]

        del gptq
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print("Mixed quantization complete.")


if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=['fp16','rtn',"xnor", "sign", "no", "2bit", "3bit", "4bit", "prune", "braq",'robq','mestrobq','medianbraq','orb','whor','arb','bhor','jrb','crb','crb_norefine','crb_symdamp','crb_symdamp_norefine','crb_resrhs','crb_resrhs_norefine','crb_seqalpha','crb_seqalpha_norefine','crb_adaptive','crb_hessian','crb_native','odr','new','ahor','crbv8','crbv9','crbv10','crbog','secq','sbh','ternary','mixed','doml','doml_binary','sdoml','magfit','sdoml_partition'],
        help="quantization method; `xnor` is the method using XNOR to adapt hardware calculation; `prune` is the method used in sparseGPTQ; braq is the method used in BiLLM",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--calib_dataset", type=str, default=None,
        choices=["wikitext2", "c4", "redpajama"],
        help="Calibration dataset (default: same as eval dataset).",
    )
    parser.add_argument(
        "--seqlen", type=int, default=None,
        help="Calibration sequence length (default: model.seqlen).",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian", "actmag"],
        help="Salient-column selection metric. 'actmag' ranks columns by "
             "s_j * sum_i |W_ij| with s = AWQ activation scale a**alpha "
             "(computed from calib, NOT applied to the weights); linears "
             "outside the AWQ v1 norm-group scope (o_proj/down_proj) fall "
             "back to plain magnitude.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=3,
        choices=[1, 3],
        help="Structural partition count. 3 = DOML/BRAQ salient/non-salient split "
             "(default). 1 = single per-block mask, equivalent to paper-faithful "
             "per-row GPTQ (combine with a large --blocksize for true no-groupsize "
             "GPTQ). Ignored for doml/doml_binary (always 3).",
    )
    parser.add_argument(
        "--global_scale",
        action="store_true",
        help="For --low_quant_method 2bit/4bit only: compute per-row scale ONCE "
             "on the full weight matrix (paper Table 3 GPTQ no-groupsize). Without "
             "this, per-row scale is recomputed per GPTQ block (paper Table 7 "
             "gs=blocksize).",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="For --low_quant_method sdoml/magfit only: per-row keep fraction "
             "1-sparsity. Default 0.5 (50%% pruned). Used to derive the "
             "n_keep parameter inside sdoml_quantize per derivation §1.3.",
    )
    parser.add_argument(
        "--sdoml_n_iter",
        type=int,
        default=20,
        help="For --low_quant_method sdoml only: number of joint-alternation "
             "rounds in sdoml_quantize. Default 20 (matches DOML's Lloyd-Max "
             "iter count). Set to 1 for the SDOML-1pass ablation (no "
             "alternation; tests whether the per-row codebook alone is the "
             "win, vs the joint mask + codebook alternation).",
    )
    parser.add_argument(
        "--sdoml_asymmetric",
        action="store_true",
        help="For --low_quant_method sdoml_partition only (S9 mandate "
             "2026-05-03): apply SDOML's joint mask + Lloyd-Max ONLY to "
             "the bulk partition (mask1, ~69%% of columns). Mid (mask2) "
             "and salient (mask3) partitions stay fully dense (no pruning, "
             "K=4 Lloyd-Max as in DOML). Sparsity rate `--sparsity` then "
             "refers to the BULK keep fraction, not the global keep "
             "fraction. Addresses S8's HONEST-NEGATIVE finding that uniform "
             "pruning across partitions destroys mask3's salient columns.",
    )
    parser.add_argument(
        "--codebook_bits",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="For doml/sdoml/sdoml_partition/magfit: K = 2**codebook_bits "
             "Lloyd-Max codebook levels per row. Default 2 (K=4, 2-bit) "
             "preserves the legacy DOML operating point. Use 3 for K=8 "
             "(3-bit codebook), 4 for K=16 (4-bit codebook), or 5 for "
             "K=32 (5-bit codebook). doml_binary ignores this — it is "
             "always K=2 (1-bit).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--just_download", action="store_true"
    )
    
    parser.add_argument(
        "--corr_damp", type = float, default = 0.1
    )
    parser.add_argument(
        "--lam", type = float, default = 1e-5
    )
    parser.add_argument(
        "--coupling", type=float, default=0.5,
        help="Coupling strength for crb_native: 0=BRAQ, 1=full joint solve."
    )
    parser.add_argument(
        "--skip_ppl_save",
        action="store_true"
    )
    parser.add_argument(
        "--eval_lambada",
        action="store_true",
        help="Also evaluate on LAMBADA (last-word prediction accuracy).",
    )
    parser.add_argument(
        "--eval_mrr",
        action="store_true",
        help="Also evaluate Mean Reciprocal Rank (MRR) on PTB test set.",
    )
    parser.add_argument(
        "--eval_mrr_agnews",
        action="store_true",
        help="Also evaluate Mean Reciprocal Rank (MRR) on AG News test set.",
    )
    parser.add_argument(
        "--eval_mrr_imdb",
        action="store_true",
        help="Also evaluate Mean Reciprocal Rank (MRR) on IMDB test set.",
    )
    parser.add_argument(
        "--eval_mrr_yelp",
        action="store_true",
        help="Also evaluate Mean Reciprocal Rank (MRR) on Yelp Review Full test set.",
    )
    # --eval_mmlu / --eval_hellaswag / --eval_arc plus --full_eval and
    # --eval_extra_ppl are added below by add_eval_cli(parser).
    parser.add_argument(
        "--eval_humaneval",
        action="store_true",
        help="Also evaluate on HumanEval (code generation, pass@1).",
    )
    parser.add_argument(
        "--eval_math",
        action="store_true",
        help="Also evaluate on MATH (competition mathematics).",
    )
    parser.add_argument(
        "--attn_order", type=int, default=3,
        help="Mixed mode: binary order for attention sublayers (default 3).",
    )
    parser.add_argument(
        "--kv_order", type=int, default=None,
        help="Mixed mode: binary order for k_proj/v_proj (overrides attn_order for KV).",
    )
    parser.add_argument(
        "--mlp_orders", type=list_of_ints, default=None,
        help="Mixed mode: orders for MLP partitions, e.g. '2,2,2' (default: 1,1,2).",
    )
    parser.add_argument(
        "--gate_up_orders", type=list_of_ints, default=None,
        help="Mixed mode: orders for gate_proj/up_proj partitions (overrides mlp_orders for gate/up).",
    )
    parser.add_argument(
        "--sbh_r_attn", type=int, default=60,
        help="SBH: SVD rank for attention sublayers (q,k,v,o).",
    )
    parser.add_argument(
        "--sbh_r_mlp", type=int, default=30,
        help="SBH: SVD rank for MLP sublayers (gate,up,down).",
    )
    add_eval_cli(parser)
    args = parser.parse_args()
    groupsize = args.blocksize

    device = args.device
    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}"
    save_file = "./output/" + save_title.replace("/", "_") + ".pt"

    # Phase 16 (2026-04-26): the generic '2bit' / '3bit' / '4bit' methods
    # need disambiguation in the CSV. The sbatch jobs `rtn-2bit` (2bit +
    # --disable_gptq) and `gptq-2bit` (2bit + --partition 1 --global_scale)
    # used to both land as method="2bit", making CSV rows non-self-
    # describing. Resolve to a more specific tag based on flags. The
    # default partition=3 path (no --disable_gptq, no --global_scale)
    # is the DOML-uniform ablation per Phase-14 finding and keeps the
    # base tag.
    def _resolve_csv_method(_args):
        base = _args.low_quant_method
        if base in ("2bit", "3bit", "4bit"):
            if _args.disable_gptq:
                return f"rtn-{base}"
            if getattr(_args, "partition", 3) == 1 and getattr(_args, "global_scale", False):
                return f"gptq-{base}"
        # Codebook bit-width suffix for DOML-family methods. Default 2 (K=4)
        # produces no suffix so legacy CSV rows stay byte-identical. Other
        # values append e.g. "-3bit" / "-4bit" for self-describing tags.
        cb_bits = int(getattr(_args, "codebook_bits", 2))
        cb_suffix = "" if cb_bits == 2 else f"-{cb_bits}bit"
        if base == "doml":
            # partition×K ablation tag: --partition 1 (single-codebook DOML)
            # appends "-p1" before the codebook-bits suffix. Default
            # partition=3 leaves the legacy tag byte-identical so existing
            # CSV rows are unaffected.
            p1_suffix = "-p1" if int(getattr(_args, "partition", 3)) == 1 else ""
            return f"doml{p1_suffix}{cb_suffix}"
        if base == "sdoml":
            # Self-describing tag: e.g. sdoml-s50 for sparsity=0.5.
            # If sdoml_n_iter == 1 (S6 ablation) append '-1pass' so the row
            # is unambiguous in the comparison pivot.
            s_pct = int(round(float(getattr(_args, "sparsity", 0.5)) * 100))
            n_iter = int(getattr(_args, "sdoml_n_iter", 20))
            tag = f"sdoml-s{s_pct}"
            if n_iter == 1:
                tag = f"{tag}-1pass"
            return f"{tag}{cb_suffix}"
        if base == "magfit":
            # S6 ablation tag: e.g. magfit-s50 for sparsity=0.5.
            s_pct = int(round(float(getattr(_args, "sparsity", 0.5)) * 100))
            return f"magfit-s{s_pct}{cb_suffix}"
        if base == "sdoml_partition":
            # S8 tag: SDOML+partition with sparsity, e.g. sdoml_partition-s50.
            # S9 (2026-05-03): when --sdoml_asymmetric, append "_asym" so
            # rows are unambiguous in the comparison pivot.
            s_pct = int(round(float(getattr(_args, "sparsity", 0.5)) * 100))
            n_iter = int(getattr(_args, "sdoml_n_iter", 20))
            asym = getattr(_args, "sdoml_asymmetric", False)
            tag = f"sdoml_partition-s{s_pct}"
            if asym:
                tag = f"{tag}_asym"
            if n_iter == 1:
                tag = f"{tag}-1pass"
            return f"{tag}{cb_suffix}"
        return base

    csv_method = _resolve_csv_method(args)

    # BPW lookup for CSV output
    _bpw_map = {
        'fp16': 16, 'rtn': 1.07, '2bit': 2.0, 'braq': 1.07,
        'crbog': 1.07, 'doml': 2.09, 'doml_binary': 1.07, 'ternary': 1.58,
        # Phase 16 disambiguated tags — same numeric bpw as base
        'rtn-2bit': 2.0, 'rtn-3bit': 3.0, 'rtn-4bit': 4.0,
        'gptq-2bit': 2.0, 'gptq-3bit': 3.0, 'gptq-4bit': 4.0,
    }
    if csv_method == "doml-p1" or (csv_method.startswith("doml-p1-") and csv_method.endswith("bit")):
        # Partition×K ablation: DOML with partition=1 → single per-row
        # codebook. bpw = log2(K) + 1*K*16/N (no salient/mid/bulk split).
        # Representative N=1024 (Qwen3-0.6B hidden).
        K_val = 2 ** int(getattr(args, "codebook_bits", 2))
        N_rep = 1024
        _run_bpw = math.log2(K_val) + K_val * 16.0 / N_rep
    elif csv_method.startswith("doml-") and csv_method.endswith("bit"):
        # K-bit DOML (K = 2**codebook_bits, codebook_bits != 2). 3-way
        # structural partition (G=3) → 3 codebooks per row × K levels × 16 bit.
        #   bpw = log2(K) + G*K*16/N
        # Representative N=1024 (Qwen3-0.6B hidden); larger models give
        # slightly lower codebook overhead — the value reported here is
        # an upper-bound approximation.
        K_val = 2 ** int(getattr(args, "codebook_bits", 2))
        N_rep = 1024
        _run_bpw = math.log2(K_val) + 3 * K_val * 16.0 / N_rep
    elif csv_method.startswith("sdoml-s") or csv_method.startswith("magfit-s"):
        # Effective bpw under bitmap encoding (C4): K*16/N + 1 + (1-s)*log2(K).
        # We use a representative N (model.config.hidden_size if available
        # later, but at this point the model is not loaded; use a typical
        # Qwen3-0.6B-style hidden = 1024 — bpw is dominated by the 1 + (1-s)*log2K
        # term and the K*16/N codebook tail is ~0.06 bpw at K=4 N=1024).
        # Per derivation §6.1 the dominant components are the 1-bit bitmap
        # + (1-s)*log2(K) codebook indices. magfit (S6 ablation) has the
        # SAME bitmap+codebook layout as SDOML so the bpw formula matches —
        # only the (mask, codebook) joint optimisation differs.
        s_val = float(getattr(args, "sparsity", 0.5))
        K_val = 2 ** int(getattr(args, "codebook_bits", 2))
        N_rep = 1024
        _run_bpw = (K_val * 16.0) / N_rep + 1.0 + (1.0 - s_val) * math.log2(K_val)
    elif csv_method.startswith("sdoml_partition-s"):
        # SDOML+partition (S8 contract C4): G=3 codebooks per row, each of
        # K levels. Bitmap is per-row (still 1 bit per weight). Codebook
        # indices: (1-s)*log2(K) per kept weight. So:
        #   bpw = G*K*16/N + 1 + (1-s)*log2(K)
        # For G=3, K=4, N=1024, s=0.5: 0.1875 + 1 + 1 = 2.1875.
        #
        # S9 asymmetric (--sdoml_asymmetric, tag has "_asym"): bitmap stored
        # ONLY for the bulk partition (mask1, ~69%). Mid + salient are dense
        # so no bitmap. Indices for ALL kept positions across all 3 partitions
        # at log2(K) bits each.
        #   keep_total = frac_bulk*(1-s) + frac_mid*1 + frac_sal*1
        #   bpw = G*K*16/N + frac_bulk*1 + keep_total*log2(K)
        # For frac_bulk=0.69, frac_mid=0.26, frac_sal=0.05, K=4, N=1024, s=0.5:
        #   keep_total = 0.69*0.5 + 0.26 + 0.05 = 0.655
        #   bpw = 0.1875 + 0.69 + 0.655*2 = 2.188
        s_val = float(getattr(args, "sparsity", 0.5))
        K_val = 2 ** int(getattr(args, "codebook_bits", 2))
        N_rep = 1024
        G_val = 3
        if "_asym" in csv_method:
            # DOML's typical 3-partition split (orders=(1,1,2), up_lim=10).
            # The salient partition is column-based (~5%), mid is element-
            # based (~26%), bulk is element-based (~69%). These are nominal
            # — actual fractions are reported per-block in the smoke log.
            frac_bulk, frac_mid, frac_sal = 0.69, 0.26, 0.05
            keep_total = (
                frac_bulk * (1.0 - s_val) + frac_mid + frac_sal
            )
            _run_bpw = (
                (G_val * K_val * 16.0) / N_rep
                + frac_bulk * 1.0  # bitmap only for bulk
                + keep_total * math.log2(K_val)
            )
        else:
            _run_bpw = (
                (G_val * K_val * 16.0) / N_rep
                + 1.0
                + (1.0 - s_val) * math.log2(K_val)
            )
    else:
        _run_bpw = _bpw_map.get(csv_method, '')
    _quant_time = 0.0

    # CSV helper
    def _csv(dataset, metric, value, notes=""):
        _csv_append(
            model=args.model, method=csv_method, dataset=dataset,
            metric=metric, value=value, bpw=_run_bpw, seed=args.seed,
            blocksize=groupsize, salient_metric=args.salient_metric,
            extra_params=None, quantization_time_s=_quant_time, notes=notes,
        )

    calib_dataset = args.calib_dataset if args.calib_dataset else args.dataset
    calib_seqlen = args.seqlen if args.seqlen else None

    if args.load_quantized:
        model = get_model(save_file) # 1 : Get Model
        model.eval()
    else: # braq
        model = get_model(args.model)
        model.eval()
        tick = time.time()

        if calib_seqlen is None:
            calib_seqlen = model.seqlen

        orig_seqlen = model.seqlen
        model.seqlen = calib_seqlen

        dataloader, testloader = get_loaders(
            calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=calib_seqlen,
        )

        if(args.just_download):
            print(f"Just download flag set, exiting")
            exit()

        # AWQ / SmoothQuant-style activation-scaling transform (opt-in via
        # CRB_AWQ_ALPHA). Output-preserving reparametrization of the FP model
        # folded into the RMSNorm weights (zero extra stored tensors -> bpw
        # unchanged). Applied AFTER model load + calibration data is ready but
        # BEFORE any per-layer Hessian accumulation / quantization, so DOML (and
        # every other method) then quantizes the transformed model transparently.
        # Unset env -> byte-identical to the previous behaviour.
        if os.environ.get('CRB_AWQ_ALPHA'):
            from kernels.pack.awq_transform import apply_awq_from_calib
            _awq_alpha = float(os.environ['CRB_AWQ_ALPHA'])
            print(f"[AWQ] CRB_AWQ_ALPHA set -> applying activation scaling (alpha={_awq_alpha})")
            _awq_scales = apply_awq_from_calib(model, dataloader, _awq_alpha, device)
            # Persist the per-layer scales so restore/btune/atune can re-apply
            # the g <- g/s norm fold (the DPK dump stores only the quantized
            # linears; the modified RMSNorm weights are NOT in the dump).
            _awq_scale_out = os.environ.get('CRB_AWQ_SCALE_OUT')
            if _awq_scale_out:
                from kernels.pack.awq_transform import save_scales
                save_scales(_awq_scales, _awq_scale_out)
                print(f"[AWQ] saved per-layer scales -> {_awq_scale_out}")

        # AWQ v2 (2026-07-24, opt-in via CRB_AWQ_V2=1 IN ADDITION to
        # CRB_AWQ_ALPHA; unset -> bit-identical behavior): extend the scaling
        # to o_proj/down_proj. The inverse folds land in v_proj/up_proj OUTPUT
        # ROWS (quantized linears, same block) — NOT in any norm — so the DPK
        # dump captures everything and restore/btune/atune need NO v2 fold.
        # Runs AFTER the v1 fold above: v2's calibration pass therefore sees
        # the v1-TRANSFORMED model, matching what will be quantized (v1 is
        # output-preserving, and v1/v2 touch different axes of v_proj, so the
        # two compose). Same alpha as v1.
        if os.environ.get('CRB_AWQ_V2') == '1':
            if not os.environ.get('CRB_AWQ_ALPHA'):
                raise RuntimeError(
                    "CRB_AWQ_V2=1 requires CRB_AWQ_ALPHA to be set (v2 "
                    "extends the v1 transform and reuses its alpha)")
            from kernels.pack.awq_transform import (
                apply_awq_v2_from_calib, save_v2_scales)
            print(f"[AWQV2] CRB_AWQ_V2=1 -> extending activation scaling to "
                  f"o_proj/down_proj (alpha={_awq_alpha})", flush=True)
            _awq_v2_scales, _awq_v2_meta = apply_awq_v2_from_calib(
                model, dataloader, _awq_alpha, device)
            print(f"AWQV2: scaled o_proj/down_proj "
                  f"({_awq_v2_meta['n_groups']} groups, "
                  f"GQA n_rep={_awq_v2_meta['n_rep']})", flush=True)
            # Analysis-only artifact: restore must NEVER fold these (the v2
            # folds are already inside the dumped quantized linears).
            _awq_v2_out = os.environ.get('CRB_AWQ_V2_SCALE_OUT')
            if _awq_v2_out:
                save_v2_scales(_awq_v2_scales, _awq_v2_out)
                print(f"[AWQV2] saved per-layer v2 scales -> {_awq_v2_out}")

        # CRB_SALIENT_METRIC=actmag (2026-07-23): activation-aware SALIENT
        # SELECTION ONLY. Compute the AWQ scales s = a**alpha (same math and
        # v1 norm-group scope as the CRB_AWQ_ALPHA transform above) but do NOT
        # modify any weight or norm; instead stash s on each covered linear
        # (plain `_crb_actmag_s` attribute) so the salient-column search ranks
        # columns by s_j * sum_i |W_ij| (see utils/structure.py "actmag").
        # o_proj/down_proj have no stash and keep the plain magnitude ranking.
        # Guard: only --salient_metric actmag enters this block, so the
        # default magnitude/hessian paths are bit-identical to before.
        if args.salient_metric == 'actmag':
            from kernels.pack.awq_transform import (
                collect_scales_from_calib, attach_selection_scales)
            _actmag_alpha = float(os.environ.get('CRB_ACTMAG_ALPHA', '0.5'))
            print(f"[ACTMAG] salient_metric=actmag -> computing AWQ scales "
                  f"(alpha={_actmag_alpha}) WITHOUT applying them", flush=True)
            _actmag_scales = collect_scales_from_calib(
                model, dataloader, alpha=_actmag_alpha, device=device)
            _n_cov, _n_fb = attach_selection_scales(model, _actmag_scales)
            print(f"ACTMAG: using activation-scaled magnitude saliency "
                  f"({_n_cov} linears covered, {_n_fb} fallback)", flush=True)

        if args.low_quant_method == "fp16":
            print("FP16 mode: skipping quantization")
        elif args.low_quant_method == "sbh":
            sbh_sequential(model, dataloader, device,
                           r_attn=args.sbh_r_attn, r_mlp=args.sbh_r_mlp)
        elif args.low_quant_method in ("ternary", "mixed"):
            mixed_sequential(model, dataloader, device)
        else:
            quant_sequential(model, dataloader, device)
            _quant_time = time.time() - tick
            print("quantization time:", _quant_time, "s")

        model.seqlen = orig_seqlen


    '''
    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)
    '''


    # ------------------------------------------------------------------
    # Standard eval suite — PPL on (args.dataset [+ extras]) + downstream
    # benchmarks (MMLU / HellaSwag / ARC). Each result is one CSV row.
    # ------------------------------------------------------------------
    eval_flags = resolve_eval_flags(args, primary_dataset=args.dataset)
    evaluate_and_log_all(
        model, args.model, device,
        method=csv_method,
        bpw=_run_bpw, seed=args.seed, blocksize=groupsize,
        salient_metric=args.salient_metric,
        extra_params={"corr_damp": args.corr_damp, "lam": args.lam,
                      "coupling": args.coupling},
        quantization_time_s=_quant_time,
        ppl_datasets=eval_flags["ppl_datasets"],
        eval_mmlu=eval_flags["eval_mmlu"],
        eval_hellaswag=eval_flags["eval_hellaswag"],
        eval_arc=eval_flags["eval_arc"],
        ppl_eval_seqlen=eval_flags["ppl_eval_seqlen"],
        save_title_prefix=save_title.replace("/", "_"),
    )

    # ------------------------------------------------------------------
    # OPT-only legacy benchmarks (no CSV writes; left in for backcompat)
    # ------------------------------------------------------------------
    if args.eval_lambada and "opt" in args.model:
        from eval_lambada import opt_eval_lambada
        lambada_title = f"{save_title}_LAMBADA"
        opt_eval_lambada(model, args.model, device, save_title=lambada_title)

    if args.eval_mrr and "opt" in args.model:
        from eval_mrr import opt_eval_mrr
        opt_eval_mrr(model, args.model, device,
                     save_title=f"{save_title}_MRR")

    if args.eval_mrr_agnews and "opt" in args.model:
        from eval_mrr_agnews import opt_eval_mrr_agnews
        opt_eval_mrr_agnews(model, args.model, device,
                            save_title=f"{save_title}_MRR_AGNEWS")

    if args.eval_mrr_imdb and "opt" in args.model:
        from eval_mrr_imdb import opt_eval_mrr_imdb
        opt_eval_mrr_imdb(model, args.model, device,
                          save_title=f"{save_title}_MRR_IMDB")

    if args.eval_mrr_yelp and "opt" in args.model:
        from eval_mrr_yelp import opt_eval_mrr_yelp
        opt_eval_mrr_yelp(model, args.model, device,
                          save_title=f"{save_title}_MRR_YELP")

    if args.eval_humaneval:
        from eval_humaneval import eval_humaneval
        eval_humaneval(model, args.model, device,
                       save_title=f"{save_title}_HUMANEVAL")

    if args.eval_math:
        from eval_math import eval_math
        eval_math(model, args.model, device,
                  save_title=f"{save_title}_MATH")
