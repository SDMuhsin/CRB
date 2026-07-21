import glob, os, sys, math
import numpy as np, torch
sys.path.insert(0,"/workspace/BiLLM2/kernels/pack")
import dpk_unpack
d="/workspace/BiLLM2/downloads/doml_dumps/qwen3-0.6b/k28-hdiag-ib-g256"
files=sorted(glob.glob(os.path.join(d,"*.dpk.safetensors")))
# accumulate joint (partition, code) counts and per-block code-count structure
cnt=np.zeros((3,4),dtype=np.int64)
for fp in files:
    t,meta=dpk_unpack.load_container(fp,"cpu")
    R,C,C_orig=meta["R"],meta["C"],meta["C_orig"]
    b0=dpk_unpack.expand_plane(t["b0"],C); b1=dpk_unpack.expand_plane(t["b1"],C)
    code=(b0.long()+2*b1.long()).numpy()
    part=dpk_unpack.part_matrix(t,meta).numpy()
    real=(np.arange(C)<C_orig)[None,:].repeat(R,0)
    for p in range(3):
        m=(part==p)&real
        c=code[m]
        bc=np.bincount(c,minlength=4)
        cnt[p]+=bc
names=["bulk","tail","salient"]
print("Per-partition marginal code distribution + entropy (k28 bulk-K4 g256 dump):")
tot_bits_uniform=0; tot_bits_entropy=0; tot_w=0
for p in range(3):
    n=cnt[p].sum(); pr=cnt[p]/n
    H=-sum(x*math.log2(x) for x in pr if x>0)
    print(f"  {names[p]:8s} n={n:>12d} frac={n/cnt.sum():.4f}  dist={np.round(pr,4).tolist()}  H(code|part={p})={H:.4f} bits")
    tot_bits_uniform+=n*2; tot_bits_entropy+=n*H; tot_w+=n
print(f"\nAGGREGATE H(code|part) = {tot_bits_entropy/tot_w:.4f} bits/weight (K27 reported 1.953)")
print(f"code bpw: uniform 2-bit = {tot_bits_uniform/tot_w:.4f}  vs entropy-floor = {tot_bits_entropy/tot_w:.4f}")
print(f"=> lossless code-entropy saving ceiling = {2.0 - tot_bits_entropy/tot_w:.4f} bpw (marginal, per-partition)")
