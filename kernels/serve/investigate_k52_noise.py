"""K5b — G-K5-2 gap investigation ("if the gap exceeds 0.1%, it is a BUG,
find it").

Measured facts going in: kernel-served W2A4 PPL 218121.72 vs torch-reference
W2A4 PPL 218401.89 -> rel delta 1.283e-3, ABOVE the pre-registered 1e-3
parity gate. Both paths agree the model COLLAPSES under per-token A4
(the pre-registered doc-03 risk #1), and G-K5-1 shows the kernel matches
kernel-independent references to norm-rel <= ~1e-7 on the real captured
activations of every sublayer. This script determines whether the residual
end-to-end gap is a kernel bug or the inherent noise floor of the criterion
itself at collapse-scale PPL:

  E1 (summation-order noise floor, NO custom kernel anywhere):
     PPL of the G-K5-2 torch reference model (RefA4Linear, single fp32
     GEMM) vs an equally-valid variant that computes each linear as TWO
     half-K fp32 GEMMs summed (RefA4LinearSplitK). The two are the same
     math in different summation orders — exactly the relationship between
     the kernel and the reference. If |delta|/PPL is of the same order as
     1.283e-3, the 0.1% criterion is un-meetable by ANY implementation
     pair at this PPL scale, i.e. the gap is measured to BE fp noise
     (chaotically amplified), not a kernel bug.

  E2 (mechanism trace): one batch through the kernel-served and reference
     models, capturing the hidden state after every decoder layer: hidden
     norm-rel per layer + fraction of A4 nibbles that FLIP when quantizing
     the (normed) hidden state of the two paths. Quantization is
     discontinuous: an fp difference of ~1e-7 at a rounding boundary flips
     a nibble = a macroscopic downstream difference; 28 stacked quantizers
     amplify implementation-level fp noise chaotically. E2 quantifies that
     growth curve.

  E3 (collapse decomposition / quantizer bug-exclusion): PPL of the
     UNQUANTIZED-weight bf16 model with the SAME per-token A4 fake-quant on
     the same 196 sublayer inputs (W16A4). If A4 alone already collapses
     the model, the W2A4 collapse is driven by the activation scheme itself
     (the pre-registered Req-3 plainest-form risk), independent of the DOML
     weights. DIAGNOSTIC ONLY: not a serving path, not tuning — no scheme
     parameter is changed anywhere.

  E4 (per-sample NLL): first 20 samples' NLLs for kernel / ref / ref-splitK
     models — shows whether the PPL gap is diffuse noise (consistent with
     chaos) or localized to specific samples (suspicious).

Output: k5_logs/investigate_k52_noise.json + stdout. Exit 0 when complete
(this is an investigation, not a gate; its verdict feeds the K5b report).

Usage:
  source /workspace/BiLLM2/env/bin/activate
  CUDA_VISIBLE_DEVICES=1 python -u kernels/serve/investigate_k52_noise.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from serve_common import (  # noqa: E402
    EVAL_SEQLEN, LOG_DIR, all_layer_names, get_wikitext2_testenc,
    load_qwen_bf16, ppl_resident, require_gpu1,
)
from dpk_serve import (  # noqa: E402
    DPK_DUMP_DIR, RefA4Linear, _get_parent, build_dpk_model,
    build_ref_a4_model, quantize_a4,
)

OUT_JSON = os.path.join(LOG_DIR, "investigate_k52_noise.json")


class RefA4LinearSplitK(nn.Module):
    """Same math as RefA4Linear, different summation order: the fp32 GEMM
    is computed as two half-K GEMMs and summed. Both are exact-fp32-GEMM
    references; their disagreement is pure summation-order noise."""

    def __init__(self, W_bf16: torch.Tensor):
        super().__init__()
        self.register_buffer("W", W_bf16.contiguous())
        self.out_features, self.in_features = W_bf16.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        xh, a_s = quantize_a4(x2)                      # SAME quantizer
        xq = (xh.float() - 8.0) * a_s.unsqueeze(1)
        Wf = self.W.float()
        h = self.in_features // 2
        y = xq[:, :h] @ Wf[:, :h].t() + xq[:, h:] @ Wf[:, h:].t()
        return y.to(torch.bfloat16).to(x.dtype).reshape(*shp[:-1],
                                                        self.out_features)


class A4OnlyLinear(nn.Module):
    """E3: original bf16 weights, SAME per-token A4 fake-quant on the input
    (W16A4). Diagnostic decomposition only."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        assert lin.bias is None
        self.register_buffer("W", lin.weight.data.clone())
        self.out_features, self.in_features = self.W.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        xh, a_s = quantize_a4(x2)
        xq = ((xh.float() - 8.0) * a_s.unsqueeze(1)).to(torch.bfloat16)
        y = xq @ self.W.t()                            # bf16 GEMM, bf16 W
        return y.to(x.dtype).reshape(*shp[:-1], self.out_features)


def swap_to_splitk(model):
    n = 0
    for lname in all_layer_names(model.config.num_hidden_layers):
        parent, leaf = _get_parent(model, lname)
        old = getattr(parent, leaf)
        assert isinstance(old, RefA4Linear), (lname, type(old))
        setattr(parent, leaf, RefA4LinearSplitK(old.W))
        n += 1
    return n


def build_w16a4_model():
    model = load_qwen_bf16()
    n = 0
    for lname in all_layer_names(model.config.num_hidden_layers):
        parent, leaf = _get_parent(model, lname)
        orig = getattr(parent, leaf)
        assert isinstance(orig, nn.Linear), (lname, type(orig))
        setattr(parent, leaf, A4OnlyLinear(orig))
        n += 1
    assert n == 196
    return model


@torch.no_grad()
def per_sample_nll(model, test_ids, dev, n, seqlen=EVAL_SEQLEN):
    loss_fct = nn.CrossEntropyLoss()
    out = []
    for i in range(n):
        batch = test_ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state
        lm_logits = model.lm_head(hidden)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        batch[:, 1:].reshape(-1))
        out.append(loss.float().item())
    return out


@torch.no_grad()
def trace_hidden_states(model, batch):
    """Hidden state AFTER each decoder layer (bare-Tensor return handled per
    the transformers-5.x block-return feedback note)."""
    hs = []
    hooks = []
    for layer in model.model.layers:
        def _hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hs.append(t.detach().float().cpu())
        hooks.append(layer.register_forward_hook(_hook))
    _ = model.model(input_ids=batch, use_cache=False)
    for h in hooks:
        h.remove()
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--nll-samples", type=int, default=20)
    args = ap.parse_args()
    require_gpu1(args.allow_any_gpu)
    assert torch.backends.cuda.matmul.allow_tf32 is False

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    test_ids = get_wikitext2_testenc()
    doc = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "dump_dir": os.path.abspath(DPK_DUMP_DIR)}

    # ---- E2 first (needs kernel model + ref model simultaneously-ish) -----
    print("E2: divergence trace, kernel-served vs reference (batch 0) ...",
          flush=True)
    batch = test_ids[:, :EVAL_SEQLEN].to(dev)

    kmodel, _ = build_dpk_model()
    kmodel = kmodel.to(dev)
    hs_k = trace_hidden_states(kmodel, batch)
    nll_kernel = per_sample_nll(kmodel, test_ids, dev, args.nll_samples)
    del kmodel
    torch.cuda.empty_cache()

    rmodel, _ = build_ref_a4_model()
    rmodel = rmodel.to(dev)
    hs_r = trace_hidden_states(rmodel, batch)
    nll_ref = per_sample_nll(rmodel, test_ids, dev, args.nll_samples)

    # per-layer divergence + nibble flips of the quantized NORMED hidden
    # (the input the next layer's q_proj actually quantizes)
    trace = []
    for L in range(len(hs_k)):
        hk, hr = hs_k[L].to(dev), hs_r[L].to(dev)
        nr = ((hk - hr).norm() / hr.norm().clamp_min(1e-30)).item()
        flips = None
        if L + 1 < len(hs_k):
            norm = rmodel.model.layers[L + 1].input_layernorm
            xk, _ = quantize_a4(norm(hk.to(torch.bfloat16))[0])
            xr, _ = quantize_a4(norm(hr.to(torch.bfloat16))[0])
            flips = (xk != xr).float().mean().item()
        trace.append({"after_layer": L, "hidden_norm_rel": nr,
                      "next_qproj_nibble_flip_frac": flips})
        if L % 4 == 0 or L == len(hs_k) - 1:
            print(f"  after layer {L:2d}: hidden norm-rel {nr:.3e}"
                  + (f", nibble flips {flips:.3e}" if flips is not None
                     else ""), flush=True)
    doc["E2_divergence_trace"] = trace
    del hs_k, hs_r
    torch.cuda.empty_cache()

    # ---- E1: reference vs split-K reference (no kernel anywhere) ----------
    print("E1: reference-model PPL (single fp32 GEMM) ...", flush=True)
    t0 = time.time()
    ppl_ref, ns = ppl_resident(rmodel, test_ids, dev, progress_every=50)
    t_ref = time.time() - t0
    print(f"  PPL_ref = {ppl_ref:.6f} ({t_ref:.0f}s)", flush=True)

    n_swapped = swap_to_splitk(rmodel)
    assert n_swapped == 196
    rmodel = rmodel.to(dev)
    nll_split = per_sample_nll(rmodel, test_ids, dev, args.nll_samples)
    t0 = time.time()
    ppl_split, _ = ppl_resident(rmodel, test_ids, dev, progress_every=50)
    t_split = time.time() - t0
    del rmodel
    torch.cuda.empty_cache()

    e1_delta = abs(ppl_split - ppl_ref) / ppl_ref
    kernel_ref_delta = 0.0012828271504346095      # measured in G-K5-2
    print(f"E1: PPL_ref={ppl_ref:.6f} PPL_ref_splitK={ppl_split:.6f} "
          f"|delta|/ref = {e1_delta:.3e}  (kernel-vs-ref gap was "
          f"{kernel_ref_delta:.3e}, pre-registered gate 1e-3)", flush=True)
    doc["E1_summation_order_floor"] = {
        "ppl_reference": ppl_ref,
        "ppl_reference_splitK": ppl_split,
        "rel_delta_ref_vs_refsplitK": e1_delta,
        "rel_delta_kernel_vs_ref_from_gate": kernel_ref_delta,
        "nsamples": ns,
        "note": "both PPLs are kernel-free fp32-GEMM references differing "
                "only in summation order; their gap is the measured noise "
                "floor of the G-K5-2 criterion at collapse-scale PPL",
    }

    # ---- E3: W16A4 decomposition -------------------------------------------
    print("E3: W16A4 (unquantized bf16 weights + same A4 fake-quant) ...",
          flush=True)
    amodel = build_w16a4_model().to(dev)
    t0 = time.time()
    ppl_w16a4, _ = ppl_resident(amodel, test_ids, dev, progress_every=50)
    t_a4 = time.time() - t0
    del amodel
    torch.cuda.empty_cache()
    print(f"  PPL_W16A4 = {ppl_w16a4:.6f} ({t_a4:.0f}s)", flush=True)
    doc["E3_w16a4_decomposition"] = {
        "ppl_w16a4": ppl_w16a4,
        "ppl_fp16_reference": 20.9685,
        "ppl_w2_fake_quant_full_precision_acts": 38.152381896972656,
        "ppl_w2a4_kernel_served": 218121.71875,
        "note": "diagnostic only (bug-exclusion + decomposition); no scheme "
                "parameter changed, no tuning",
    }

    # ---- E4: per-sample NLLs ------------------------------------------------
    doc["E4_per_sample_nll"] = {
        "n": args.nll_samples,
        "kernel": nll_kernel, "reference": nll_ref,
        "reference_splitK": nll_split,
        "max_abs_nll_delta_kernel_vs_ref":
            max(abs(a - b) for a, b in zip(nll_kernel, nll_ref)),
        "max_abs_nll_delta_ref_vs_splitK":
            max(abs(a - b) for a, b in zip(nll_split, nll_ref)),
    }
    print(f"E4: max per-sample |dNLL| kernel-vs-ref = "
          f"{doc['E4_per_sample_nll']['max_abs_nll_delta_kernel_vs_ref']:.4e}, "
          f"ref-vs-splitK = "
          f"{doc['E4_per_sample_nll']['max_abs_nll_delta_ref_vs_splitK']:.4e}",
          flush=True)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"JSON: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
