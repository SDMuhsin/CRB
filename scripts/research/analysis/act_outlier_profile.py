"""Falcon3 root-cause lead #1: activation-outlier profile from calib data alone.

For each model: a_j = mean_tokens |RMSNorm output_j| per norm group per layer
(same hook machinery as kernels/pack/awq_transform.py). Outlier ratio =
max(a) / geomean(a) per group. Anchors: Qwen3-1.7B (~60x, AWQ big win),
Llama-3.2-1B (~6-12x, AWQ hurt). Target: Falcon3-1B.
"""
import sys, math
sys.path.insert(0, '/workspace/CRB')
import torch
from kernels.pack.awq_transform import collect_awq_scales
from run import get_model
from datautils import get_loaders

DEV = 'cuda:0'
MODELS = [
    ('Qwen/Qwen3-1.7B', 'qwen3-1.7b'),
    ('meta-llama/Llama-3.2-1B', 'llama3.2-1b'),
    ('tiiuae/Falcon3-1B-Base', 'falcon3-1b'),
]
NSAMPLES = 32

def summarize(name, scales):
    # alpha=1.0 -> s = a / geomean(a); ratio = s.max()
    print(f"\n==== {name} ====")
    print(f"{'grp':<6}{'layers':<8}{'median_ratio':<14}{'mean_ratio':<12}"
          f"{'max_ratio':<11}{'p90_ratio':<11}")
    allr = {}
    for key, tag in (('input_layernorm', 'qkv'), ('post_attention_layernorm', 'mlp')):
        ratios = torch.tensor([d[key].max().item() for d in scales])
        allr[tag] = ratios
        p90 = ratios.quantile(0.9).item()
        print(f"{tag:<6}{len(ratios):<8}{ratios.median().item():<14.2f}"
              f"{ratios.mean().item():<12.2f}{ratios.max().item():<11.2f}{p90:<11.2f}")
    # per-layer detail line (compact)
    for key, tag in (('input_layernorm', 'qkv'), ('post_attention_layernorm', 'mlp')):
        vals = ' '.join(f"{d[key].max().item():.0f}" for d in scales)
        print(f"  {tag} per-layer: {vals}")
    return allr

for hf, short in MODELS:
    print(f"\nloading {hf} ...", flush=True)
    model = get_model(hf)
    model = model.to(DEV).eval()
    calib, _ = get_loaders('wikitext2', nsamples=NSAMPLES, seed=0,
                           model=hf, seqlen=model.seqlen)
    with torch.no_grad():
        scales = collect_awq_scales(model, calib, alpha=1.0, device=DEV)
    summarize(short, scales)
    del model
    torch.cuda.empty_cache()
print("\nDONE")
