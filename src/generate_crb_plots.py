"""
Generate plots showing weight matrices at each stage of the CRB pipeline.
Uses real weights from BLOOM-1.7B layer 0 self_attention.query_key_value.

GROUND TRUTH (from end-to-end audit):
- A block (oc × 128) is partitioned into 3 groups via structural_guassian_distribution
- The SALIENT group has ~18% of elements, and it's COLUMN-BASED: when a column is salient,
  ALL rows in that column are salient. (~23 out of 128 columns)
- Low and Mid groups share the remaining ~105 columns (element-wise within those columns)
- Salient columns get order=2 CRB: two binary expansions α₁B₁ + α₂B₂
  - B₁ and B₂ are BOTH defined on the SAME salient columns
  - They differ in sign patterns (~54% disagree), not in position
- Non-salient elements get order=1: simple sign × mean_abs
- Final: Q = q_low*mask_low + q_mid*mask_mid + q_sal*mask_sal (covers all elements)
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "./llmdocs/crb_plots"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading BLOOM-1.7B weights...")
from transformers import BloomForCausalLM
model = BloomForCausalLM.from_pretrained(
    "bigscience/bloom-1b7", torch_dtype="auto",
    cache_dir="./downloads", attn_implementation="eager"
)
W_full = model.transformer.h[0].self_attention.query_key_value.weight.data.float()
del model; torch.cuda.empty_cache()

# Use 128-wide block, 32 rows for visualization
blocksize = 128
NR = 32
W = W_full[:NR, :blocksize].clone()

# Structural partition (on 32×128 — same as what we visualize)
print("Computing structural partition...")
sys.path.insert(0, '.')
from utils.structure import structural_guassian_distribution
H_dummy = torch.eye(blocksize)
mask_low, mask_mid, mask_sal = structural_guassian_distribution(W, H_dummy, 'magnitude', 50)

print(f"  Low:     {mask_low.float().mean():.3f}")
print(f"  Mid:     {mask_mid.float().mean():.3f}")
print(f"  Salient: {mask_sal.float().mean():.3f}")
sal_cols = (mask_sal.sum(dim=0) == NR)
print(f"  Salient columns: {sal_cols.sum().item()} / {blocksize}")

# Run CRB on all three groups
from binary import coupled_residual_binarization_stable_v7 as crb
q_low = crb(W, mask_low, order=1)
q_mid = crb(W, mask_mid, order=1)
q_sal = crb(W, mask_sal, order=2)
Q_combined = q_low * mask_low.float() + q_mid * mask_mid.float() + q_sal * mask_sal.float()

# Manually trace CRB order=2 for intermediates
mask_f = mask_sal.float()
d = mask_f.sum(dim=1); d_safe = torch.clamp(d, min=1.0)
new_matrix = W.clone() * mask_f
row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
centered = (new_matrix - row_mean[:, None]) * mask_f
B1 = torch.sign(centered) * mask_f
alpha1_init = (centered.abs() * mask_f).sum(dim=1) / d_safe
r = (centered - alpha1_init[:, None] * B1) * mask_f
r_mean = (r * mask_f).sum(dim=1) / d_safe
r_centered = (r - r_mean[:, None]) * mask_f
B2 = torch.sign(r_centered) * mask_f

lam = 1e-5; corr_damp = 0.1
def solve_alphas(B1, B2, centered, d):
    c12 = (B1 * B2 * mask_f).sum(dim=1)
    c12 = torch.where(c12 > 0, c12*(1-corr_damp), c12)
    c1w = (centered * B1).sum(dim=1); c2w = (centered * B2).sum(dim=1)
    A = d + lam; denom = A*A - c12*c12
    sd = torch.where(denom.abs()>1e-12, denom, torch.ones_like(denom))
    a1 = torch.clamp((A*c1w - c12*c2w)/sd, min=0.0)
    a2 = torch.clamp((A*c2w - c12*c1w)/sd, min=0.0)
    return a1, a2

a1, a2 = solve_alphas(B1, B2, centered, d)
t5 = (centered - a1[:,None]*B1)*mask_f; t5m = (t5*mask_f).sum(1)/d_safe
B2r = torch.sign((t5-t5m[:,None])*mask_f)*mask_f
a1, a2 = solve_alphas(B1, B2r, centered, d)
t6 = (centered - a2[:,None]*B2r)*mask_f; t6m = (t6*mask_f).sum(1)/d_safe
B1r = torch.sign((t6-t6m[:,None])*mask_f)*mask_f
a1f, a2f = solve_alphas(B1r, B2r, centered, d)
exp1 = (a1f[:,None] * B1r) * mask_f
exp2 = (a2f[:,None] * B2r) * mask_f

# Sort columns so salient columns are grouped together for clearer visualization
sal_col_idx = torch.where(sal_cols)[0]
nonsal_col_idx = torch.where(~sal_cols)[0]
col_order = torch.cat([nonsal_col_idx, sal_col_idx])

def reorder_cols(x):
    return x[:, col_order]

# ---------- Plotting ----------
CMAP = 'coolwarm'
ELEV = 28; AZIM = -55

def plot_3d(data, filename, zlim=None):
    fig = plt.figure(figsize=(2.8, 2.2), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    Z = data.numpy() if isinstance(data, torch.Tensor) else data
    rows, cols = Z.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    ax.plot_surface(X, Y, Z, cmap=CMAP, edgecolor='none', alpha=0.92,
                    antialiased=True, rcount=rows, ccount=min(cols, 80))
    if zlim: ax.set_zlim(zlim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.view_init(elev=ELEV, azim=AZIM)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(OUT_DIR, filename), bbox_inches='tight', facecolor='white', pad_inches=0.01)
    plt.close()
    print(f"  Saved {filename}")

def plot_binary_heatmap(data, mask, filename, color_neg='#3B82F6', color_pos='#EF4444'):
    """2D heatmap for masked binary. Gray=non-salient, Blue=-1, Red=+1."""
    fig, ax = plt.subplots(figsize=(2.8, 1.4), dpi=200)
    Z = data.numpy() if isinstance(data, torch.Tensor) else data
    M = mask.numpy() if isinstance(mask, torch.Tensor) else mask

    # Background: light gray everywhere
    bg = np.full(Z.shape + (4,), [0.94, 0.95, 0.96, 1.0])
    ax.imshow(bg, interpolation='nearest', aspect='auto')

    # Overlay colored cells only where mask is active
    rgba = np.zeros(Z.shape + (4,))
    # Parse hex colors
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16)/255 for i in (0,2,4))
    cn = hex_to_rgb(color_neg)
    cp = hex_to_rgb(color_pos)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if M[i,j] > 0.5:
                if Z[i,j] < 0:
                    rgba[i,j] = [cn[0], cn[1], cn[2], 1.0]
                else:
                    rgba[i,j] = [cp[0], cp[1], cp[2], 1.0]

    ax.imshow(rgba, interpolation='nearest', aspect='auto')

    # Vertical line to show salient/non-salient boundary
    n_nonsal = (~sal_cols).sum().item()
    ax.axvline(x=n_nonsal - 0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.5)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(os.path.join(OUT_DIR, filename), bbox_inches='tight', facecolor='white', pad_inches=0.01)
    plt.close()
    print(f"  Saved {filename}")

def plot_partition_map(mask_low, mask_mid, mask_sal, filename):
    """Show the 3-way partition as a colored heatmap."""
    fig, ax = plt.subplots(figsize=(2.8, 1.4), dpi=200)
    rows, cols = mask_low.shape
    # Create RGB image: low=light blue, mid=light green, salient=orange/red
    img = np.zeros((rows, cols, 3))
    ml = mask_low.numpy(); mm = mask_mid.numpy(); ms = mask_sal.numpy()
    img[ml > 0.5] = [0.75, 0.85, 1.0]    # light blue for low
    img[mm > 0.5] = [0.75, 1.0, 0.85]    # light green for mid
    img[ms > 0.5] = [1.0, 0.6, 0.4]      # orange for salient
    ax.imshow(img, interpolation='nearest', aspect='auto')

    # Vertical line at salient boundary
    n_nonsal = (~sal_cols).sum().item()
    ax.axvline(x=n_nonsal - 0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.5)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(os.path.join(OUT_DIR, filename), bbox_inches='tight', facecolor='white', pad_inches=0.01)
    plt.close()
    print(f"  Saved {filename}")

print("\nGenerating plots (columns sorted: non-salient | salient)...")

# Reorder all matrices so salient columns are on the right
W_sorted = reorder_cols(W)
centered_sorted = reorder_cols(centered)
B1r_sorted = reorder_cols(B1r)
B2r_sorted = reorder_cols(B2r)
exp1_sorted = reorder_cols(exp1)
exp2_sorted = reorder_cols(exp2)
Q_sorted = reorder_cols(Q_combined)
mask_sal_sorted = reorder_cols(mask_sal)
mask_low_sorted = reorder_cols(mask_low)
mask_mid_sorted = reorder_cols(mask_mid)

wmax = W.abs().max().item() * 1.1
wlim = (-wmax, wmax)

# 1. Original FP16 weight block
plot_3d(W_sorted, "01_fp16_weights.png", zlim=wlim)

# 2. Partition map — shows which elements are low/mid/salient
plot_partition_map(mask_low_sorted, mask_mid_sorted, mask_sal_sorted, "02_partition.png")

# 3. Centered salient weights (non-salient zeroed out)
plot_3d(centered_sorted, "02_centered.png")

# 4. B1 heatmap (salient positions only)
plot_binary_heatmap(B1r_sorted, mask_sal_sorted, "03_B1_sign.png")

# 5. α₁B₁ (sparse — non-salient columns are zero)
plot_3d(exp1_sorted, "04_alpha1_B1.png", zlim=wlim)

# 6. B2 heatmap (same salient positions, different signs)
plot_binary_heatmap(B2r_sorted, mask_sal_sorted, "06_B2_sign.png",
                    color_neg='#2563EB', color_pos='#D97706')

# 7. α₂B₂ (sparse, same positions as α₁B₁)
plot_3d(exp2_sorted, "07_alpha2_B2.png")

# 8. Combined quantized (all three groups merged — full dense matrix)
plot_3d(Q_sorted, "08_quantized.png", zlim=wlim)

# 9. Error
error_sorted = W_sorted - Q_sorted
plot_3d(error_sorted, "09_error.png")

# 5b. Residual (for diagram: centered - α₁B₁)
res_sorted = reorder_cols((centered - alpha1_init[:,None]*B1)*mask_f)
plot_3d(res_sorted, "05_residual.png")

print(f"\nDone! B1/B2 both active on {sal_cols.sum().item()} salient columns (right side of plots)")
print(f"B1 unique (non-zero): {torch.unique(B1r[B1r!=0]).tolist()}")
print(f"B2 unique (non-zero): {torch.unique(B2r[B2r!=0]).tolist()}")
print(f"Sign agreement: {((B1r==B2r)&(mask_f>0)).sum().item()}/{(mask_f>0).sum().item()}")
