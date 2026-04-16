"""Quick test: verify crb_seqalpha_norefine produces same output as BRAQ."""
import torch
import sys
sys.path.insert(0, '.')
from binary import high_order_residual, coupled_residual_binarization_seqalpha

torch.manual_seed(42)

# Create a realistic weight matrix and mask
oc, ic = 256, 128
x = torch.randn(oc, ic, dtype=torch.float32).cuda() * 0.01
mask = torch.zeros(oc, ic, dtype=torch.bool).cuda()
# Make ~18% of columns salient
mask[:, :23] = True

# Run BRAQ
braq_out = high_order_residual(x, mask, order=2)

# Run crb_seqalpha_norefine
seqalpha_out = coupled_residual_binarization_seqalpha(x, mask, order=2, skip_refinement=True)

# Compare
diff = (braq_out - seqalpha_out).abs()
max_diff = diff.max().item()
mean_diff = diff[mask].mean().item()
print(f"Max absolute diff: {max_diff:.2e}")
print(f"Mean absolute diff (masked only): {mean_diff:.2e}")
print(f"BRAQ output range: [{braq_out[mask].min():.6f}, {braq_out[mask].max():.6f}]")
print(f"SeqAlpha output range: [{seqalpha_out[mask].min():.6f}, {seqalpha_out[mask].max():.6f}]")

# Check a few specific elements
print(f"\nSample comparison (first 5 masked elements of row 0):")
for i in range(5):
    b = braq_out[0, i].item()
    s = seqalpha_out[0, i].item()
    print(f"  col {i}: BRAQ={b:.8f}, SeqAlpha={s:.8f}, diff={abs(b-s):.2e}")

if max_diff < 1e-5:
    print("\nRESULT: BRAQ-equivalent (max diff < 1e-5)")
else:
    print(f"\nRESULT: NOT BRAQ-equivalent (max diff = {max_diff:.2e})")
