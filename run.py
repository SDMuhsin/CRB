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


downloads_dir = "./downloads"
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
        elif "llama" in model_name or "danube" in model_name.lower():
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, use_safetensors=True, attn_implementation="eager")
            model.seqlen = 2048
        elif "qwen" in model_name.lower():
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, attn_implementation="eager")
            model.seqlen = min(model.config.max_position_embeddings, 2048)
        elif "smollm" in model_name.lower():
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, attn_implementation="eager")
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
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir=downloads_dir, attn_implementation="eager")
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
    elif "llama" in args.model or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower():
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
    elif "llama" in args.model or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower():
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
                        orders=(gu_order, gu_order, gu_order),
                    )
                else:
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
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
                    # DOML: use structural partition (same as BRAQ),
                    # but with Lloyd-Max quantizer per partition
                    info = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        partition=3,
                        orders=(1,1,1),  # order ignored by DOML quantizer
                    )
                else:
                    info = gptq[name].fasterquant(
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
    elif "llama" in args.model or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower():
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
    elif "llama" in args.model or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower():
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
    elif "llama" in args.model or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower():
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
    elif "llama" in args.model or "danube" in args.model.lower() or "qwen" in args.model.lower() or "smollm" in args.model.lower() or "granite" in args.model.lower():
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
        choices=['fp16','rtn',"xnor", "sign", "no", "2bit", "4bit", "prune", "braq",'robq','mestrobq','medianbraq','orb','whor','arb','bhor','jrb','crb','crb_norefine','crb_symdamp','crb_symdamp_norefine','crb_resrhs','crb_resrhs_norefine','crb_seqalpha','crb_seqalpha_norefine','crb_adaptive','crb_hessian','crb_native','odr','new','ahor','crbv8','crbv9','crbv10','crbog','secq','sbh','ternary','mixed','doml','doml_binary'],
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
        choices=["magnitude", "hessian"],
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
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="Also evaluate on MMLU (5-shot multiple choice across 57 subjects).",
    )
    parser.add_argument(
        "--eval_hellaswag",
        action="store_true",
        help="Also evaluate on HellaSwag (commonsense sentence completion).",
    )
    parser.add_argument(
        "--eval_arc",
        action="store_true",
        help="Also evaluate on ARC (AI2 Reasoning Challenge, Easy + Challenge).",
    )
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
    args = parser.parse_args()
    groupsize = args.blocksize

    device = args.device
    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}"
    save_file = "./output/" + save_title.replace("/", "_") + ".pt"

    # BPW lookup for CSV output
    _bpw_map = {
        'fp16': 16, 'rtn': 1.07, '2bit': 2.0, 'braq': 1.07,
        'crbog': 1.07, 'doml': 2.09, 'doml_binary': 1.07, 'ternary': 1.58,
    }
    _run_bpw = _bpw_map.get(args.low_quant_method, '')
    _quant_time = 0.0

    # CSV helper
    def _csv(dataset, metric, value, notes=""):
        _csv_append(
            model=args.model, method=args.low_quant_method, dataset=dataset,
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


    for dataset in [args.dataset]:#["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval
            
            ppl = opt_eval(model, testloader, device, dataset, args.log_wandb, save_title, save = not args.skip_ppl_save )
            _csv(dataset, "perplexity", ppl)

            ''' FOR ABLATION STUDY '''
            # Define the path to the JSON file
            results_path = "./output/ablation_results.json"

            # Ensure the results directory exists
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            # Load existing results or initialize an empty dict if file doesn't exist or is empty
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results = {}

            # Create the key and update the results with the new perplexity value
            key = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_{args.corr_damp}_{args.lam}"
            results[key] = ppl

            # Save the updated results back to the JSON file
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)


        elif "llama" in args.model or "danube" in args.model.lower():
            from eval_ppl_utils import llama_eval

            ppl = llama_eval(model, testloader, device, dataset, args.log_wandb, save_title, save = not args.skip_ppl_save)
            _csv(dataset, "perplexity", ppl)

            # Save to ablation results
            results_path = "./output/ablation_results.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results = {}
            key = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_{args.corr_damp}_{args.lam}"
            results[key] = ppl
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

        elif "granite" in args.model.lower():
            from eval_ppl_utils import granite_eval

            ppl = granite_eval(model, testloader, device, dataset, args.log_wandb, save_title, save = not args.skip_ppl_save)
            _csv(dataset, "perplexity", ppl)

            # Save to ablation results
            results_path = "./output/ablation_results.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results = {}
            key = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_{args.corr_damp}_{args.lam}"
            results[key] = ppl
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

        elif "qwen" in args.model.lower() or "smollm" in args.model.lower():
            from eval_ppl_utils import qwen_eval

            ppl = qwen_eval(model, testloader, device, dataset, args.log_wandb, save_title, save = not args.skip_ppl_save)
            _csv(dataset, "perplexity", ppl)

            # Save to ablation results
            results_path = "./output/ablation_results.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results = {}
            key = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_{args.corr_damp}_{args.lam}"
            results[key] = ppl
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

        elif "pythia" in args.model.lower():
            from eval_ppl_utils import pythia_eval

            ppl = pythia_eval(model, testloader, device, dataset, args.log_wandb, save_title, save = not args.skip_ppl_save)
            _csv(dataset, "perplexity", ppl)

            # Save to ablation results
            results_path = "./output/ablation_results.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results = {}
            key = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_{args.corr_damp}_{args.lam}"
            results[key] = ppl
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

        elif "bloom" in args.model.lower():
            from eval_ppl_utils import bloom_eval

            ppl = bloom_eval(model, testloader, device, dataset, args.log_wandb, save_title, save = not args.skip_ppl_save)
            _csv(dataset, "perplexity", ppl)

            # Save to ablation results
            results_path = "./output/ablation_results.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results = {}
            key = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_{args.corr_damp}_{args.lam}"
            results[key] = ppl
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

    if args.eval_lambada and "opt" in args.model:
        from eval_lambada import opt_eval_lambada
        lambada_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_LAMBADA"
        opt_eval_lambada(model, args.model, device, save_title=lambada_title)

    if args.eval_mrr and "opt" in args.model:
        from eval_mrr import opt_eval_mrr
        mrr_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_MRR"
        opt_eval_mrr(model, args.model, device, save_title=mrr_title)

    if args.eval_mrr_agnews and "opt" in args.model:
        from eval_mrr_agnews import opt_eval_mrr_agnews
        agnews_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_MRR_AGNEWS"
        opt_eval_mrr_agnews(model, args.model, device, save_title=agnews_title)

    if args.eval_mrr_imdb and "opt" in args.model:
        from eval_mrr_imdb import opt_eval_mrr_imdb
        imdb_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_MRR_IMDB"
        opt_eval_mrr_imdb(model, args.model, device, save_title=imdb_title)

    if args.eval_mrr_yelp and "opt" in args.model:
        from eval_mrr_yelp import opt_eval_mrr_yelp
        yelp_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_MRR_YELP"
        opt_eval_mrr_yelp(model, args.model, device, save_title=yelp_title)

    # Model-agnostic benchmarks (work with OPT, LLaMA, Qwen, etc.)
    if args.eval_mmlu:
        from eval_mmlu import eval_mmlu
        mmlu_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_MMLU"
        mmlu_acc = eval_mmlu(model, args.model, device, save_title=mmlu_title)
        _csv(args.dataset, "mmlu_acc", mmlu_acc)

    if args.eval_hellaswag:
        from eval_hellaswag import eval_hellaswag
        hellaswag_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_HELLASWAG"
        hellaswag_acc = eval_hellaswag(model, args.model, device, save_title=hellaswag_title)
        _csv(args.dataset, "hellaswag_acc", hellaswag_acc)

    if args.eval_arc:
        from eval_arc import eval_arc
        arc_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_ARC"
        arc_results = eval_arc(model, args.model, device, save_title=arc_title)
        _csv(args.dataset, "arc_easy_acc", arc_results["ARC-Easy"]["accuracy"])
        _csv(args.dataset, "arc_challenge_acc", arc_results["ARC-Challenge"]["accuracy"])

    if args.eval_humaneval:
        from eval_humaneval import eval_humaneval
        humaneval_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_HUMANEVAL"
        eval_humaneval(model, args.model, device, save_title=humaneval_title)

    if args.eval_math:
        from eval_math import eval_math
        math_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_MATH"
        eval_math(model, args.model, device, save_title=math_title)
