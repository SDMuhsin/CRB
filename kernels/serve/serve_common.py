"""K5a shared library — Marlin INT4 model-scale comparator for Qwen3-0.6B.

This module is the single home of:
  * artifact paths + layer naming for the sym-g128 INT4 dump,
  * the EXACT fp16 dequant semantics of the Marlin kernel
    (effective weight = fp16( (q - 8) * s ) with q in 0..15, s fp16),
  * a bit-exact UNPACKER for Marlin's packed B / permuted s buffers
    (inverse of marlin.Layer.pack; used by gate G-A),
  * MarlinServeLinear — the serving module that replaces nn.Linear
    (bf16 activations in -> fp16 -> patched Marlin kernel -> bf16 out),
  * the model builder that swaps the 196 quantized sublayers,
  * the resident-model WikiText-2 PPL loop replicating eval_ppl_utils.
    qwen_eval's loss accounting term-for-term.

Marlin build: kernels/third_party/marlin @1f25790 + marlin_racefix.patch
(NEVER use an unpatched build; see baselines/B1_int4_baseline.md).

No repo source file is modified by anything in kernels/serve/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import torch
import torch.nn as nn

REPO = "/workspace/BiLLM2"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# datautils / run.py resolve downloads relatively; pin them absolute.
os.environ.setdefault("BILLM_DOWNLOADS_DIR", os.path.join(REPO, "downloads"))
os.environ.setdefault(
    "BILLM_BENCH_CSV",
    os.path.join(REPO, "llmdocs", "cuda_kernel", "verify", "scratch_results.csv"),
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
GROUPSIZE = 128
SEED = 0
EVAL_SEQLEN = 2048
N_QUANT_SUBLAYERS = 196          # 7 linears x 28 decoder layers
SUBLAYER_NAMES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)

DUMP_DIR = os.path.join(REPO, "downloads", "marlin_dumps", "qwen3-0.6b", "sym-g128")
LOG_DIR = os.path.join(REPO, "llmdocs", "cuda_kernel", "verify", "k5_logs")
GATE_A_MARKER = os.path.join(LOG_DIR, "gate_A_PASS.json")
GATE_B_MARKER = os.path.join(LOG_DIR, "gate_B_PASS.json")

# Smallest positive normal fp16. Group scales are clamped here so that
# (q-8)*s and Marlin's own pack() round-trip (round(w/s)) are exact —
# subnormal scales would break both. Fires only for groups whose absmax
# < 15/2 * 2^-14 ~ 4.6e-4 (negligible weights); documented deviation.
FP16_MIN_NORMAL = 2.0 ** -14


def require_gpu1(allow_any: bool = False):
    """K5a territory rule: GPU 1 only (CUDA_VISIBLE_DEVICES=1)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not allow_any and cvd != "1":
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={cvd!r}; K5a scripts must run with "
            "CUDA_VISIBLE_DEVICES=1 (pass --allow-any-gpu to override).")


# ---------------------------------------------------------------------------
# Layer naming (identical convention to the DOML dumps: global_name minus
# the model prefix, e.g. "model.layers.0.self_attn.q_proj")
# ---------------------------------------------------------------------------

def all_layer_names(n_layers: int = 28):
    return [f"model.layers.{i}.{s}" for i in range(n_layers)
            for s in SUBLAYER_NAMES]


def q4_path(layer_name: str, dump_dir: str = DUMP_DIR) -> str:
    return os.path.join(dump_dir, f"{layer_name}.q4.safetensors")


def marlin_path(layer_name: str, dump_dir: str = DUMP_DIR) -> str:
    return os.path.join(dump_dir, f"{layer_name}.marlin.safetensors")


# ---------------------------------------------------------------------------
# Exact fp16 dequant semantics (what the Marlin kernel computes in registers:
# int4 -> exact (q-8) in half, then half*half multiply by the group scale)
# ---------------------------------------------------------------------------

def dequant_ref_fp16(q_nk: torch.Tensor, s_kg_n: torch.Tensor) -> torch.Tensor:
    """q_nk: (N, K) uint8 codes 0..15; s_kg_n: (K/128, N) fp16 group scales.
    Returns (N, K) fp16 = fp16((q-8) * s) — bit-exact kernel semantics."""
    N, K = q_nk.shape
    assert s_kg_n.shape == (K // GROUPSIZE, N), (q_nk.shape, s_kg_n.shape)
    assert s_kg_n.dtype == torch.half
    # expand scales to (N, K): element (n, k) uses s[k // 128, n]
    s_nk = s_kg_n.t().repeat_interleave(GROUPSIZE, dim=1)   # (N, K) fp16
    return (q_nk.to(torch.float32) - 8.0).half() * s_nk     # fp16 mul, exact semantics


# ---------------------------------------------------------------------------
# Marlin pack inverse (gate G-A). Forward reference: marlin/__init__.py
# Layer.pack(). All permutations are imported from the installed marlin
# package so the inverse can never drift from the packer.
# ---------------------------------------------------------------------------

def _perms():
    import marlin as _m
    return _m._perm, _m._scale_perm, _m._scale_perm_single


def unpack_marlin_B(B: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Invert Layer.pack's weight packing. B: (k/16, n*2) int32 (Marlin
    format). Returns (k, n) int16 codes in 0..15."""
    perm, _, _ = _perms()
    assert B.shape == (k // 16, n * 2) and B.dtype == torch.int32, (B.shape, B.dtype)
    res = torch.empty((k // 16, n * 16), dtype=torch.int32)
    Bi = B.to(torch.int32)
    for i in range(8):
        res[:, i::8] = (Bi >> (4 * i)) & 0xF
    # invert res = tmp[:, perm]
    inv = torch.argsort(perm)
    tmp = res.reshape(-1, perm.numel())[:, inv].reshape(k // 16, n * 16)
    # invert tile reshape: w.reshape(k/16,16,n/16,16).permute(0,2,1,3).reshape(k/16, n*16)
    w = tmp.reshape(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n)
    return w.to(torch.int16)


def unpack_marlin_s(s_packed: torch.Tensor, k: int, n: int,
                    groupsize: int = GROUPSIZE) -> torch.Tensor:
    """Invert Layer.pack's scale permutation for grouped mode
    (groupsize != k). s_packed: (k/groupsize, n) fp16 as stored in
    Layer.s. Returns (k/groupsize, n) fp16 in natural order."""
    _, scale_perm, _ = _perms()
    assert groupsize != k, "single-group (-1) scales use scale_perm_single"
    assert s_packed.shape == (k // groupsize, n)
    sp = torch.tensor(scale_perm)
    inv = torch.argsort(sp)
    flat = s_packed.reshape(-1, len(scale_perm))[:, inv]
    return flat.reshape(k // groupsize, n).contiguous()


# ---------------------------------------------------------------------------
# Serving module
# ---------------------------------------------------------------------------

class MarlinServeLinear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear served by the patched
    Marlin W4A16 kernel. The surrounding model stays bf16; this module casts
    activations bf16 -> fp16 at entry (exact for in-range values), runs the
    fp16xINT4 tensor-core kernel (fp32 accumulate), and casts the fp16
    output back to bf16.

    Buffers (identical layout to marlin.Layer):
      B         (k/16, n*2) int32   packed int4 weights
      s         (k/128, n)  fp16    group scales, Marlin-permuted
      workspace (n/128*16,) int32   zeroed lock scratch (persistent=False)
    """

    def __init__(self, k: int, n: int, B: torch.Tensor, s: torch.Tensor):
        super().__init__()
        assert k % 128 == 0 and n % 256 == 0, (k, n)
        assert B.shape == (k // 16, n * 2) and B.dtype == torch.int32
        assert s.shape == (k // GROUPSIZE, n) and s.dtype == torch.half
        self.in_features = k
        self.out_features = n
        self.register_buffer("B", B.contiguous())
        self.register_buffer("s", s.contiguous())
        self.register_buffer(
            "workspace", torch.zeros(n // 128 * 16, dtype=torch.int),
            persistent=False)
        # G-B capture hook: when set to a list, forward appends
        # (input_as_received, fp16_input, raw_fp16_output).
        self._capture = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import marlin
        in_dtype = x.dtype
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        x16 = x2.to(torch.half)
        C = torch.empty((x2.shape[0], self.out_features),
                        dtype=torch.half, device=x.device)
        marlin.mul(x16, self.B, C, self.s, self.workspace)
        if self._capture is not None:
            self._capture.append((x.detach(), x16.detach(), C.detach()))
        return C.to(in_dtype).reshape(*shp[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, g={GROUPSIZE}"


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def load_qwen_bf16():
    """Load Qwen3-0.6B exactly as run.py does (bf16 auto dtype, eager
    attention, safetensors), on CPU. model.seqlen = 2048."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto",
        cache_dir=os.environ["BILLM_DOWNLOADS_DIR"],
        use_safetensors=True, attn_implementation="eager")
    model.seqlen = min(model.config.max_position_embeddings, EVAL_SEQLEN)
    model.eval()
    model.config.use_cache = False
    return model


def _get_parent(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]


def load_marlin_artifact(layer_name: str, dump_dir: str = DUMP_DIR):
    from safetensors import safe_open
    path = marlin_path(layer_name, dump_dir)
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["meta"])
        B = f.get_tensor("B")
        s = f.get_tensor("s")
    return B, s, meta


def load_q4_artifact(layer_name: str, dump_dir: str = DUMP_DIR):
    from safetensors import safe_open
    path = q4_path(layer_name, dump_dir)
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["meta"])
        q = f.get_tensor("q")          # (N, K) uint8, 0..15
        s = f.get_tensor("s")          # (K/128, N) fp16, natural order
    return q, s, meta


def build_marlin_model(dump_dir: str = DUMP_DIR):
    """Load the bf16 Qwen3-0.6B skeleton and replace all 196 quantized
    sublayers with MarlinServeLinear built from the G-A-verified packed
    artifacts. Embeddings / lm_head (tied) / norms stay bf16, same split
    as the DOML runs. Returns (model_on_cpu, n_replaced)."""
    model = load_qwen_bf16()
    n = 0
    for lname in all_layer_names(model.config.num_hidden_layers):
        parent, leaf = _get_parent(model, lname)
        orig = getattr(parent, leaf)
        assert isinstance(orig, nn.Linear), (lname, type(orig))
        assert orig.bias is None, f"{lname} has a bias; serving module is bias-free"
        B, s, meta = load_marlin_artifact(lname, dump_dir)
        k, nfeat = meta["K"], meta["N"]
        assert (orig.in_features, orig.out_features) == (k, nfeat), lname
        setattr(parent, leaf, MarlinServeLinear(k, nfeat, B, s))
        n += 1
    assert n == N_QUANT_SUBLAYERS, n
    return model, n


# ---------------------------------------------------------------------------
# WikiText-2 PPL — resident-model loop, loss accounting replicated
# term-for-term from eval_ppl_utils.qwen_eval (bf16 logits into
# nn.CrossEntropyLoss, nll = loss.float() * seqlen, ppl = exp(sum/total)).
# ---------------------------------------------------------------------------

def get_wikitext2_testenc():
    from datautils import get_loaders
    cwd = os.getcwd()
    os.chdir(REPO)   # datautils caches relative to ./downloads
    try:
        _, testenc = get_loaders("wikitext2", nsamples=128, seed=SEED,
                                 seqlen=EVAL_SEQLEN, model=MODEL_NAME)
    finally:
        os.chdir(cwd)
    return testenc.input_ids


@torch.no_grad()
def ppl_resident(model, test_ids: torch.Tensor, dev, seqlen: int = EVAL_SEQLEN,
                 progress_every: int = 0):
    """Full-precision-protocol PPL with the WHOLE model resident on `dev`.
    hidden states via model.model(...) (includes the final norm), logits via
    model.lm_head — the exact op sequence of qwen_eval, without its
    layer-offloading."""
    nsamples = test_ids.numel() // seqlen
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    for i in range(nsamples):
        batch = test_ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state
        lm_logits = model.lm_head(hidden)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * seqlen)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ppl_resident: {i+1}/{nsamples}", flush=True)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item(), nsamples


# ---------------------------------------------------------------------------
# nvidia-smi cross-check helpers
# ---------------------------------------------------------------------------

def nvidia_smi_snapshot():
    """Returns (per-process rows on all GPUs, this pid's used MiB or None).

    CAVEAT (measured on this host): the container has its own PID namespace
    while nvidia-smi reports HOST pids, so `os.getpid()` never matches and
    the second element is None here. Callers must identify this process's
    row by DIFFING the row set against a snapshot taken before any CUDA
    work (the new row on the target GPU is ours) — see
    measure_marlin_baseline.identify_my_row.
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
         "--format=csv,noheader,nounits"], text=True)
    rows = []
    mine = None
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid_s, mem_s, uuid = [x.strip() for x in line.split(",")]
        rows.append({"pid": int(pid_s), "used_MiB": int(mem_s), "gpu_uuid": uuid})
        if int(pid_s) == os.getpid():
            mine = int(mem_s)
    return rows, mine


def nvidia_smi_gpu_used():
    """Per-GPU total used memory: list of {index, uuid, used_MiB}."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in out.strip().splitlines():
        idx, uuid, used = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "uuid": uuid, "used_MiB": int(used)})
    return rows
