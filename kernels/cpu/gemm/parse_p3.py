#!/usr/bin/env python3
"""Parse a P3 paired log (run_p3_checkpoint.sh / run_p3_paired.sh output):
per config, median-of-round-medians [min,max]; within-round ratios
(doml3 / competitor) median [min,max]; GMAC/s; model-weighted pp512-linears
bound. Numbers only from the log."""
import re, sys
from statistics import median

log = open(sys.argv[1]).read()

rounds = re.split(r"^=== ROUND (\d+) hog purge", log, flags=re.M)
per_round = {}   # round -> {(bench,type,dout,din,nth): median_ns}
per_round_gm = {}
rx = re.compile(
    r"SUMMARY (bench_ik|doml3_gemm)\s+type=(\S+)\s+dout=(\d+)\s+din=(\d+)\s+"
    r"ny=512\s+nth=(\d+)\s+nbuf=\S+\s+iters=\d+\s+median=([\d.]+) ns/call\s+"
    r"\[[\d.,]+\]\s+weightBW=[\d.]+ GB/s\s+([\d.]+) GMAC/s")
rx_split = re.compile(
    r"SPLIT doml3_gemm dout=(\d+) din=(\d+) ny=512 nth=(\d+) medians: "
    r"quant=([\d.]+) ns/call convert=([\d.]+) ns/call gemm=([\d.]+) ns/call")
per_round_split = {}
for i in range(1, len(rounds), 2):
    r = int(rounds[i])
    d, dg, ds = {}, {}, {}
    for m in rx.finditer(rounds[i + 1]):
        bench, ty, dout, din, nth, med, gm = m.groups()
        d[(bench, ty, int(dout), int(din), int(nth))] = float(med)
        dg[(bench, ty, int(dout), int(din), int(nth))] = float(gm)
    for m in rx_split.finditer(rounds[i + 1]):
        dout, din, nth, q, c, g = m.groups()
        ds[(int(dout), int(din), int(nth))] = (float(q), float(c), float(g))
    per_round[r] = d
    per_round_gm[r] = dg
    per_round_split[r] = ds

configs = sorted({k for d in per_round.values() for k in d})
print(f"rounds found: {sorted(per_round)}")
print(f"{'bench':10s} {'type':10s} {'shape':10s} {'nth':>3s}  "
      f"median_of_round_medians [min,max] us   GMAC/s")
for k in configs:
    vals = [per_round[r][k] for r in sorted(per_round) if k in per_round[r]]
    if not vals:
        continue
    gms = [per_round_gm[r][k] for r in sorted(per_round) if k in per_round_gm[r]]
    b, ty, dout, din, nth = k
    print(f"{b:10s} {ty:10s} {dout}x{din:<5d} {nth:>3d}  "
          f"{median(vals)/1e3:8.1f} [{min(vals)/1e3:.1f},{max(vals)/1e3:.1f}]"
          f"  (n={len(vals)})  {median(gms):7.0f}")

# model-weighted pp512 linears bound (Qwen3-0.6B: q + 2*(k/v) + o + 2*(gate/up)
# + down per layer, x28 layers; 512 tokens per call)
def model_bound(bench, ty, nth):
    shapes = [(2048, 1024, 1), (1024, 1024, 2), (1024, 2048, 1),
              (3072, 1024, 2), (1024, 3072, 1)]
    tot = 0.0
    for dout, din, cnt in shapes:
        k = (bench, ty, dout, din, nth)
        vals = [per_round[r][k] for r in sorted(per_round) if k in per_round[r]]
        if not vals:
            return None
        tot += cnt * median(vals)
    return tot * 28 / 1e9  # seconds per 512-token prefill (linears only)
print()
for bench, ty in (("doml3_gemm", "i8p_mk1"), ("doml3_gemm", "i8p_mk0"),
                  ("bench_ik", "iq2_kl"), ("bench_ik", "q8_k_r16")):
    for nth in (24, 48):
        s = model_bound(bench, ty, nth)
        if s is not None:
            print(f"pp512-linears bound {bench}/{ty} nth={nth}: "
                  f"{s*1e3:.2f} ms/call-set = {512/s:.0f} tok/s")

print("\nwithin-round ratios doml3/type (median [min,max] over rounds):")
ours = [k for k in configs if k[0] == "doml3_gemm"]
for ko in ours:
    _, oty, dout, din, nth = ko
    for cty in ("iq2_kl", "q8_k_r16"):
        kc = ("bench_ik", cty, dout, din, nth)
        rat = [per_round[r][ko] / per_round[r][kc]
               for r in sorted(per_round) if ko in per_round[r] and kc in per_round[r]]
        if not rat:
            continue
        print(f"  {oty:8s} {dout}x{din:<5d} nth={nth:>2d} vs {cty:9s}: "
              f"{median(rat):.3f} [{min(rat):.3f},{max(rat):.3f}]  (n={len(rat)})")

splits = sorted({k for d in per_round_split.values() for k in d})
if splits:
    print("\nphase split medians over rounds (us/call):")
    for k in splits:
        qs = [per_round_split[r][k][0] for r in sorted(per_round_split)
              if k in per_round_split[r]]
        cs = [per_round_split[r][k][1] for r in sorted(per_round_split)
              if k in per_round_split[r]]
        gs = [per_round_split[r][k][2] for r in sorted(per_round_split)
              if k in per_round_split[r]]
        print(f"  {k[0]}x{k[1]} nth={k[2]}: quant={median(qs)/1e3:.1f} "
              f"convert={median(cs)/1e3:.1f} gemm={median(gs)/1e3:.1f} (n={len(qs)})")
