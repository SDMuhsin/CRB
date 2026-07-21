#!/usr/bin/env python3
"""Parse a P2c paired log (run_checkpoint.sh / run_paired_sweeps.sh output):
per config, median-of-round-medians [min,max]; within-round ratios
(doml2 / competitor) median [min,max]. Numbers only from the log."""
import re, sys
from statistics import median

log = open(sys.argv[1]).read()

rounds = re.split(r"^=== ROUND (\d+) hog purge", log, flags=re.M)
# rounds = [pre, '1', text1, '2', text2, ...]
per_round = {}  # round -> {(bench,type,dout,din,nth): median_ns}
rx = re.compile(
    r"SUMMARY (bench_ik|doml2_gemv|doml_gemv)\s+type=(\S+)\s+dout=(\d+)\s+din=(\d+)\s+ny=1\s+nth=(\d+)\s+nbuf=\S+\s+iters=\d+\s+median=([\d.]+) ns/call\s+\[[\d.,]+\]\s+weightBW=([\d.]+) GB/s")
per_round_bw = {}
for i in range(1, len(rounds), 2):
    r = int(rounds[i])
    d = {}
    dbw = {}
    for m in rx.finditer(rounds[i + 1]):
        bench, ty, dout, din, nth, med, bw = m.groups()
        d[(bench, ty, int(dout), int(din), int(nth))] = float(med)
        dbw[(bench, ty, int(dout), int(din), int(nth))] = float(bw)
    per_round[r] = d
    per_round_bw[r] = dbw

configs = sorted({k for d in per_round.values() for k in d})
print(f"rounds found: {sorted(per_round)}")
print(f"{'bench':10s} {'type':8s} {'shape':10s} {'nth':>3s}  median_of_round_medians [min,max] us   wGB/s")
for k in configs:
    vals = [per_round[r][k] for r in sorted(per_round) if k in per_round[r]]
    if not vals: continue
    bws = [per_round_bw[r][k] for r in sorted(per_round) if k in per_round_bw[r]]
    b, ty, dout, din, nth = k
    print(f"{b:10s} {ty:8s} {dout}x{din:<5d} {nth:>3d}  {median(vals)/1e3:8.1f} [{min(vals)/1e3:.1f},{max(vals)/1e3:.1f}]  (n={len(vals)})  {median(bws):6.1f}")

# model-weighted linears bound per token (Qwen3-0.6B: per layer q + k + v + o
# + gate + up + down; x28 layers), from median-of-round-medians
def model_bound(bench, ty, nth):
    shapes = [(2048,1024,1),(1024,1024,2),(1024,2048,1),(3072,1024,2),(1024,3072,1)]
    tot = 0.0
    for dout, din, cnt in shapes:
        k = (bench, ty, dout, din, nth)
        vals = [per_round[r][k] for r in sorted(per_round) if k in per_round[r]]
        if not vals: return None
        tot += cnt * median(vals)
    return tot * 28 / 1e3
for bench, ty in (("doml2_gemv","i8"),("bench_ik","iq2_kl"),("bench_ik","q2_k_r4")):
    for nth in (1,24):
        us = model_bound(bench, ty, nth)
        if us is not None:
            print(f"model-weighted linears bound {bench}/{ty} nth={nth}: {us:.0f} us/tok = {1e6/us:.1f} tok/s")

print("\nwithin-round ratios doml2/type (median [min,max] over rounds):")
ours = [k for k in configs if k[0] == "doml2_gemv"]
for ko in ours:
    _, oty, dout, din, nth = ko
    for cty in ("iq2_kl", "q2_k_r4"):
        kc = ("bench_ik", cty, dout, din, nth)
        rat = [per_round[r][ko] / per_round[r][kc]
               for r in sorted(per_round) if ko in per_round[r] and kc in per_round[r]]
        if not rat: continue
        print(f"  {oty:6s} {dout}x{din:<5d} nth={nth:>2d} vs {cty:8s}: "
              f"{median(rat):.3f} [{min(rat):.3f},{max(rat):.3f}]  (n={len(rat)})")
