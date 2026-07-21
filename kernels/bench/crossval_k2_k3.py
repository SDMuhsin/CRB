"""PI cross-validation: K2's real Qwen3-0.6B DPK dumps through K3's CUDA kernel.

Two independently-implemented stacks of doc 02 (K2: packer/unpacker/reference;
K3: CUDA kernel + its own reference) meet here for the first time. Checks per layer:
  X1 (format agreement): K2.unpack(container) == K3.dequant_weights(container) == wq  (bf16 bitwise)
  X2 (kernel on real weights): K3 CUDA dpk_gemv fp32 vs K2 gemv_bucket fp32  (norm-rel < 1e-5)
  X3: bf16 outputs of kernel vs bf16(K2 fp32)  (mismatch fraction reported; expect ~0)
  X4: K3 CUDA vs K3's own ref on the same real tensors (norm-rel < 1e-5)
Run: source env/bin/activate && CUDA_VISIBLE_DEVICES=0 python kernels/bench/crossval_k2_k3.py
"""
import sys, json, torch

sys.path.insert(0, "kernels/pack"); sys.path.insert(0, "kernels/ref"); sys.path.insert(0, "kernels/cuda")
from dpk_unpack import load_container, unpack as k2_unpack
from ref_w2a4 import gemv_bucket as k2_gemv_bucket, pack_a4 as k2_pack_a4
import dpk_ref as k3_ref
from build import build_dpk
from safetensors.torch import load_file

DUMP = "downloads/doml_dumps/qwen3-0.6b/sa-g128"
LAYERS = [f"model.layers.{i}.{s}" for i, s in [
    (0, "self_attn.q_proj"), (0, "mlp.down_proj"), (5, "self_attn.k_proj"),
    (9, "self_attn.o_proj"), (13, "mlp.gate_proj"), (13, "self_attn.v_proj"),
    (18, "mlp.up_proj"), (21, "self_attn.q_proj"), (24, "mlp.down_proj"),
    (27, "self_attn.o_proj"), (27, "mlp.up_proj"), (27, "mlp.gate_proj"),
]]

def u32(t):  # safetensors uint32 -> torch int32 view for the CUDA ext if needed
    return t if t.dtype == torch.int32 else t.view(torch.int32)

def norm_rel(b, a):
    d = (b.double() - a.double()).norm(); n = a.double().norm().clamp_min(1e-30)
    return (d / n).item()

def main():
    dev = "cuda:0"
    ext = build_dpk()
    results, worst = [], {"x2": 0.0, "x3": 0.0, "x4": 0.0}
    fail = False
    for name in LAYERS:
        tens, meta = load_container(f"{DUMP}/{name}.dpk.safetensors", device=dev)
        wq = load_file(f"{DUMP}/{name}.wq.safetensors")["wq"].to(dev)
        g = int(meta["g"]); C = int(meta["C"]); R = int(meta["R"])
        b0, b1, m, s, cb = (tens[k].to(dev) for k in ("b0", "b1", "m", "s", "cb"))

        # X1: three-way bitwise format agreement (K3 dequant returns fp32 by
        # design; it must round-trip to the bf16 wq exactly since levels are bf16)
        w_k2 = k2_unpack(tens, meta).to(dev)
        w_k3 = k3_ref.dequant_weights(b0, b1, m, s, cb, g)
        x1 = torch.equal(w_k2.to(torch.bfloat16), wq) and \
             torch.equal(w_k3.to(torch.bfloat16), wq)

        # activations: shared LSB-first packed words from K2's packer
        gen = torch.Generator(device="cpu").manual_seed(hash(name) & 0x7FFFFFFF)
        xhat_cpu = torch.randint(0, 16, (C,), generator=gen, dtype=torch.int64)
        xw = k2_pack_a4(xhat_cpu).to(dev)
        a_s = 0.01234567

        y_k2 = k2_gemv_bucket(tens, meta, xw.cpu(), a_s).to(dev).float()
        yk_f32 = ext.dpk_gemv(u32(b0), u32(b1), u32(m), u32(s), cb, u32(xw), a_s, g, out_fp32=True)
        yk_bf16 = ext.dpk_gemv(u32(b0), u32(b1), u32(m), u32(s), cb, u32(xw), a_s, g)
        y_k3r = k3_ref.ref_gemv_bucket(b0, b1, m, s, cb, xw, a_s, g)

        x2 = norm_rel(yk_f32, y_k2)
        x3 = (yk_bf16 != y_k2.bfloat16()).float().mean().item()
        x4 = norm_rel(yk_f32, y_k3r)
        ok = x1 and x2 < 1e-5 and x4 < 1e-5 and x3 < 5e-4
        fail |= not ok
        for k, v in (("x2", x2), ("x3", x3), ("x4", x4)): worst[k] = max(worst[k], v)
        results.append(dict(layer=name, R=R, C=C, g=g, x1_bitwise=bool(x1),
                            x2_kernel_vs_k2=x2, x3_bf16_mismatch=x3, x4_kernel_vs_k3ref=x4,
                            verdict="PASS" if ok else "FAIL"))
        print(f"{name:44s} R={R:5d} C={C:5d} X1={'OK' if x1 else 'FAIL'} "
              f"X2={x2:.3e} X3={x3:.2e} X4={x4:.3e} [{results[-1]['verdict']}]")

    json.dump(dict(results=results, worst=worst), open("kernels/bench/results_crossval_k2_k3.json", "w"), indent=1)
    print(f"\nworst: {worst}")
    print("CROSS-VALIDATION:", "FAIL" if fail else "PASS (12/12 layers)")
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()
