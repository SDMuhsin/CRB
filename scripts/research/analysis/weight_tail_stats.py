"""Falcon3 root-cause lead #2: per-group weight-tail statistics, CPU only.

For every decoder-layer linear, group columns by 128 (TQ-matched) and compute:
  - kurtosis (Fisher, per group -> median over groups)
  - max|w| / rms per group -> median
  - fraction of weights beyond 3*std of their group -> mean
  - clip-gain: MSE(4-level min-max quant, best shrink c in grid) /
               MSE(4-level min-max quant, c=1.0)   -> median over groups
    (low ratio = clipping helps a lot = TQ auto-clip advantage)
  - best shrink c distribution (median)
"""
import sys
sys.path.insert(0, '/workspace/CRB')
import torch

MODELS = [
    ('Qwen/Qwen3-1.7B', 'qwen3-1.7b'),
    ('meta-llama/Llama-3.2-1B', 'llama3.2-1b'),
    ('tiiuae/Falcon3-1B-Base', 'falcon3-1b'),
    ('allenai/OLMo-2-0425-1B', 'olmo2-1b'),
    ('HuggingFaceTB/SmolLM2-1.7B', 'smollm2-1.7b'),
]
G = 128
GRID = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
LIN_NAMES = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
             'gate_proj', 'up_proj', 'down_proj')


def group_view(w):
    out, inn = w.shape
    n = (inn // G) * G
    return w[:, :n].reshape(out, inn // G, G).reshape(-1, G)


def quant_mse(g, c):
    """4-level asymmetric min-max quant MSE with shrink factor c per group."""
    mn = g.min(dim=1, keepdim=True).values * c
    mx = g.max(dim=1, keepdim=True).values * c
    scale = (mx - mn).clamp_min(1e-12) / 3.0
    q = ((g.clamp(mn, mx) - mn) / scale).round().clamp(0, 3)
    deq = q * scale + mn
    return (deq - g).pow(2).mean(dim=1)


def analyze(name, short):
    from run import get_model
    print(f"\nloading {name} ...", flush=True)
    model = get_model(name)
    layers = model.model.layers
    kurts, maxrms, frac3, clipgain, bestc = [], [], [], [], []
    for layer in layers:
        for attr in LIN_NAMES:
            mod = layer.self_attn if attr in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else layer.mlp
            lin = getattr(mod, attr, None)
            if lin is None:
                continue
            g = group_view(lin.weight.data.to(torch.float32))
            mu = g.mean(dim=1, keepdim=True)
            sd = g.std(dim=1, keepdim=True).clamp_min(1e-12)
            z = (g - mu) / sd
            kurts.append((z.pow(4).mean(dim=1) - 3.0))
            rms = g.pow(2).mean(dim=1).sqrt().clamp_min(1e-12)
            maxrms.append(g.abs().max(dim=1).values / rms)
            frac3.append((z.abs() > 3).float().mean(dim=1))
            mses = torch.stack([quant_mse(g, c) for c in GRID], dim=0)  # (C, ngroups)
            best_idx = mses.argmin(dim=0)
            gain = mses.gather(0, best_idx.unsqueeze(0)).squeeze(0) / mses[0].clamp_min(1e-20)
            clipgain.append(gain)
            bestc.append(torch.tensor([GRID[i] for i in best_idx.tolist()]))
    kurts = torch.cat(kurts); maxrms = torch.cat(maxrms)
    frac3 = torch.cat(frac3); clipgain = torch.cat(clipgain); bestc = torch.cat(bestc)
    print(f"==== {short} ====")
    print(f"  groups                 : {kurts.numel()}")
    print(f"  kurtosis  med/p90      : {kurts.median().item():.3f} / {kurts.quantile(0.9).item():.3f}")
    print(f"  max|w|/rms med/p90     : {maxrms.median().item():.3f} / {maxrms.quantile(0.9).item():.3f}")
    print(f"  frac |z|>3 mean        : {frac3.mean().item():.5f}")
    print(f"  clip-gain  med/p10     : {clipgain.median().item():.4f} / {clipgain.quantile(0.1).item():.4f}")
    print(f"  best shrink c med      : {bestc.median().item():.3f}")
    print(f"  frac groups best c<1.0 : {(bestc < 0.999).float().mean().item():.4f}", flush=True)
    del model


for hf, short in MODELS:
    analyze(hf, short)
print("\nDONE")
