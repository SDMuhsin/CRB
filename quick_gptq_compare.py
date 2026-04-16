#!/usr/bin/env python3
"""Minimal memory GPTQ error comparison: CRB vs BRAQ on Qwen3-0.6B layer 0."""
import torch, torch.nn as nn, gc, sys, os
os.environ["TRANSFORMERS_CACHE"] = "./downloads"
os.environ["HF_HOME"] = "./downloads"
sys.path.insert(0, ".")

import binary
from binary import Binarization
from bigptq import BRAGPTQ
from modelutils import find_layers
from transformers import AutoModelForCausalLM
from datautils import get_loaders

device = 'cuda:0'
nsamples = 32  # Reduced for memory — sufficient for diagnostic comparison

# Load model to CPU explicitly
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float16,
    attn_implementation='eager', cache_dir='./downloads')
model.eval()
model.seqlen = min(model.config.max_position_embeddings, 2048)
model.config.use_cache = False

dataloader, _ = get_loaders('wikitext2', nsamples=nsamples, seed=0,
    model='Qwen/Qwen3-0.6B', seqlen=model.seqlen)

# Catcher phase - move only necessary parts to GPU
model.model.embed_tokens.to(device)
model.model.norm.to(device)
model.model.rotary_emb.to(device)
layers = model.model.layers
layers[0] = layers[0].to(device)

inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size),
    dtype=torch.float16, device=device)
cache = {'i': 0, 'layer_kwargs': {}}

class Catcher(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.module = m
    def __getattr__(self, name):
        if name == 'module': return super().__getattr__(name)
        try: return super().__getattr__(name)
        except AttributeError: return getattr(self.module, name)
    def forward(self, inp, **kw):
        inps[cache['i']] = inp; cache['i'] += 1; cache['layer_kwargs'] = kw
        raise ValueError

layers[0] = Catcher(layers[0])
for batch in dataloader:
    try: model(batch[0].to(device))
    except ValueError: pass
layers[0] = layers[0].module

# Move embeddings back to CPU
layers[0] = layers[0].cpu()
model.model.embed_tokens.cpu()
model.model.norm.cpu()
model.model.rotary_emb.cpu()
torch.cuda.empty_cache()

outs = torch.zeros_like(inps)
layer_kwargs = cache['layer_kwargs']

# Now process layer 0
layer = layers[0].to(device)
subset = find_layers(layer)

# Gather Hessian for all sublayers simultaneously
gptq_data = {}
for name in sorted(subset.keys()):
    if not isinstance(subset[name], nn.Linear): continue
    q = Binarization(subset[name].weight, method='braq')
    gptq_data[name] = BRAGPTQ(subset[name], q, salient_metric='magnitude')

handles = []
for name in gptq_data:
    def make_hook(n):
        def hook(_, inp, out):
            gptq_data[n].add_batch(inp[0].data, out.data)
        return hook
    handles.append(subset[name].register_forward_hook(make_hook(name)))

for j in range(nsamples):
    outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
for h in handles:
    h.remove()

# Compare methods on each sublayer
print(f"{'sublayer':>20s} {'method':>5s} {'GPTQ_error':>12s} {'raw_MSE':>12s}")
print("-" * 55)

for name in sorted(gptq_data.keys()):
    orig = subset[name].weight.data.clone()
    H = gptq_data[name].H.clone()
    ns = gptq_data[name].nsamples
    gptq_data[name].free()

    for method in ['braq', 'crb']:
        subset[name].weight.data = orig.clone()
        q = Binarization(subset[name].weight, method=method)
        g = BRAGPTQ(subset[name], q, salient_metric='magnitude')
        g.H = H.clone()
        g.nsamples = ns
        info = g.fasterquant(blocksize=128)
        mse = ((orig.float() - subset[name].weight.data.float())**2).mean().item()
        print(f"{name:>20s} {method:>5s} {info['error']:>12.2f} {mse:>12.6e}")
        g.free()

    subset[name].weight.data = orig
    del H
    torch.cuda.empty_cache()

print("\nDone.")
