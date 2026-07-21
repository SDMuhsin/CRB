import re, sys, statistics as st
from collections import defaultdict

LOG = "llmdocs/cpu_kernel_rnd/verify/p2b/combined_paired_20260716_004058.log"
rounds = defaultdict(dict)  # (kind,type,dout,din,ny,nth) -> {round: median_ns}
cur_round = None
for line in open(LOG):
    m = re.match(r"=== ROUND (\d+) (IK|GEMV) start", line)
    if m: cur_round = int(m.group(1)); continue
    if line.startswith("SUMMARY"):
        f = dict(re.findall(r"(\w+)=([\w.\-]+)", line))
        kind = "ik" if "bench_ik" in line else "gemv"
        key = (kind, f["type"], int(f["dout"]), int(f["din"]), int(f["ny"]), int(f["nth"]))
        rounds[key][cur_round] = float(f["median"])

SHAPES = [(2048,1024),(1024,1024),(1024,2048),(3072,1024),(1024,3072)]
NR = 6

def stats(key):
    vals = [rounds[key][r] for r in range(1, NR+1) if r in rounds[key]]
    return st.median(vals), min(vals), max(vals), len(vals)

print("=== per-config median-of-round-medians [min,max] ns, ny=1 (GEMV) ===")
for nth in (1, 24):
    print(f"\n--- {nth} thread(s) ---")
    hdr = ["shape"] + ["iq2_kl","q2_k_r4"] + ["fp","i8","fp_mf","i8_mf"]
    print("  " + " | ".join(f"{h:>22}" for h in hdr))
    for (do,di) in SHAPES:
        row = [f"{do}x{di}"]
        for t in ("iq2_kl","q2_k_r4"):
            med,lo,hi,n = stats(("ik",t,do,di,1,nth))
            row.append(f"{med/1e3:8.1f} [{lo/1e3:.1f},{hi/1e3:.1f}]")
        for t in ("fp","i8","fp_mf","i8_mf"):
            med,lo,hi,n = stats(("gemv",t,do,di,1,nth))
            row.append(f"{med/1e3:8.1f} [{lo/1e3:.1f},{hi/1e3:.1f}]")
        print("  " + " | ".join(f"{c:>22}" for c in row))

print("\n=== WITHIN-ROUND paired ratios ours/ik (<1 = we win): median [min,max] over rounds ===")
for nth in (1, 24):
    print(f"\n--- {nth} thread(s) ---")
    for (do,di) in SHAPES:
        out = [f"{do}x{di}"]
        for t in ("fp","i8","fp_mf","i8_mf"):
            for ikt in ("iq2_kl","q2_k_r4"):
                rr = []
                for r in range(1, NR+1):
                    a = rounds[("gemv",t,do,di,1,nth)].get(r)
                    b = rounds[("ik",ikt,do,di,1,nth)].get(r)
                    if a and b: rr.append(a/b)
                out.append(f"{t}/{ikt[:6]}={st.median(rr):.3f}[{min(rr):.3f},{max(rr):.3f}]")
        print("  " + "  ".join(out))

print("\n=== model-weighted per-token linears estimate (28 layers, q+k+v+o+gate+up+down), 24t ===")
# per token: q 2048x1024, k 1024x1024, v 1024x1024, o 1024x2048, gate 3072x1024, up 3072x1024, down 1024x3072
COUNTS = {(2048,1024):1,(1024,1024):2,(1024,2048):1,(3072,1024):2,(1024,3072):1}
for t in ("iq2_kl","q2_k_r4"):
    tot = sum(c*stats(("ik",t,do,di,1,24))[0] for (do,di),c in COUNTS.items())*28
    print(f"  ik {t:8}: {tot/1e3:8.1f} us/token linears -> {1e6/tot*1e3:6.1f} tok/s bound")
for t in ("fp","i8","fp_mf","i8_mf"):
    tot = sum(c*stats(("gemv",t,do,di,1,24))[0] for (do,di),c in COUNTS.items())*28
    print(f"  gemv {t:6}: {tot/1e3:8.1f} us/token linears -> {1e6/tot*1e3:6.1f} tok/s bound")
