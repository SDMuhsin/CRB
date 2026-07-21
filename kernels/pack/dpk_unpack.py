"""K2 deliverable 2a — reference DPK unpacker (doc 02 §3 invariant, exact).

Implements, in vectorized torch, the normative dequantization invariant of
llmdocs/cuda_kernel/02_storage_format_design.md §3:

element mmode:
    part(i,j) = s[j] ? 2 : (m[i][j] ? 1 : 0)   # 0=bulk P1, 1=tail P2, 2=sal P3
    code(i,j) = b0[i][j] + 2*b1[i][j]          # 0..3
    W[i,j]    = cb[i][j // g][part(i,j)][code(i,j)]

column mmode (K2.6): the m plane is absent; part(j) comes from the 2-bit
per-column `colmem` stream (uint32[C/16], LSB-first pairs: field j%16 of
word j//16; 0=bulk, 1=tail, 2=salient), broadcast over rows. The s bitmap is
still stored and MUST agree with (colmem == 2) — validated on load.

Bit i of plane word w covers column 32*w + i (LSB-first, doc 02 §2a).
Returns the FULL padded [R, C] bf16 matrix; callers truncate to C_orig
(padded columns decode to cb[.., last group, 0, 0] by the §3 padding trick —
neutral in the §7 fold, excluded from round-trip comparison).

Usage as a library:
    tensors, meta = load_container(path, device)
    W = unpack(tensors, meta)          # bf16 [R, C]
"""

import json

import torch
from safetensors import safe_open

CONTAINER_KEYS = {"b0", "b1", "m", "s", "cb"}            # element mmode
CONTAINER_KEYS_COLUMN = {"b0", "b1", "colmem", "s", "cb"}  # column mmode

# Codebook storage dtypes (K27). fp8 cb is LOSSLESS: values were fp8-snapped at
# quantization time, so fp8 -> bf16 on load is bit-exact (verified in derive_dpk
# and by the dpk_verify round-trip gate).
CBDTYPES = {
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def load_container(path, device="cpu"):
    """Load a .dpk.safetensors container. Returns (tensors, meta dict)."""
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        md = f.metadata()
        if md is None or "meta" not in md:
            raise ValueError(f"{path}: missing JSON meta blob")
        meta = json.loads(md["meta"])
        exp_keys = (CONTAINER_KEYS_COLUMN if meta.get("mmode") == "column"
                    else CONTAINER_KEYS)
        if keys != exp_keys:
            raise ValueError(
                f"{path}: container keys {sorted(keys)} != "
                f"{sorted(exp_keys)} (doc 02 §2a violated, "
                f"mmode={meta.get('mmode')})")
        for k in keys:
            tensors[k] = f.get_tensor(k).to(device)
    _validate(tensors, meta, path)
    return tensors, meta


def _validate(t, meta, path="<mem>"):
    R, C, NG, g, B = meta["R"], meta["C"], meta["NG"], meta["g"], meta["B"]
    if C % B != 0 or g % B != 0 or NG != -(-C // g):
        raise ValueError(f"{path}: inconsistent C/B/g/NG in meta")
    if meta["mmode"] not in ("element", "column"):
        raise ValueError(f"{path}: unsupported mmode {meta['mmode']}")
    if meta["cbdtype"] not in CBDTYPES:
        raise ValueError(f"{path}: unsupported cbdtype {meta['cbdtype']}")
    exp = {
        "b0": (torch.uint32, (R, C // 32)),
        "b1": (torch.uint32, (R, C // 32)),
        "s": (torch.uint32, (C // 32,)),
        "cb": (CBDTYPES[meta["cbdtype"]], (R, NG, 3, 4)),
    }
    if meta["mmode"] == "column":
        if C % 16 != 0:
            raise ValueError(f"{path}: C={C} not a multiple of 16 (colmem)")
        exp["colmem"] = (torch.uint32, (C // 16,))
    else:
        exp["m"] = (torch.uint32, (R, C // 32))
    for k, (dt, sh) in exp.items():
        if t[k].dtype != dt or tuple(t[k].shape) != sh:
            raise ValueError(
                f"{path}: {k} is {t[k].dtype}{tuple(t[k].shape)}, "
                f"expected {dt}{sh}")


def expand_plane(words: torch.Tensor, C: int) -> torch.Tensor:
    """uint32 [..., C/32] -> bool [..., C]; bit i of word w = column 32w+i."""
    w = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sh = torch.arange(32, device=w.device, dtype=torch.int64)
    bits = (w.unsqueeze(-1) >> sh) & 1
    return bits.reshape(*w.shape[:-1], C).to(torch.bool)


def expand_colmem(words: torch.Tensor, C: int) -> torch.Tensor:
    """uint32 [C/16] -> int64 [C] of 2-bit codes; field j%16 of word j//16
    occupies bits 2*(j%16)..2*(j%16)+1 (LSB-first pairs, doc 02 §2)."""
    w = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sh = torch.arange(16, device=w.device, dtype=torch.int64) * 2
    codes = (w.unsqueeze(-1) >> sh) & 3
    return codes.reshape(-1)[:C]


def part_matrix(tensors, meta) -> torch.Tensor:
    """[R, C] int64 partition index per doc 02 §3.

    element mmode: part(i,j) = s[j] ? 2 : (m[i][j] ? 1 : 0)
    column mmode:  part(j) from colmem, broadcast over rows; colmem is
    validated (no code 3; salient codes must agree with the s bitmap).
    """
    R, C = meta["R"], meta["C"]
    if meta["mmode"] == "column":
        codes = expand_colmem(tensors["colmem"], C)            # [C] 0..3
        if not bool((codes <= 2).all()):
            raise ValueError("colmem contains invalid code 3")
        s = expand_plane(tensors["s"].unsqueeze(0), C)[0]      # [C] bool
        if not bool(((codes == 2) == s).all()):
            raise ValueError("colmem salient codes disagree with s bitmap")
        return codes.unsqueeze(0).expand(R, C)
    m = expand_plane(tensors["m"], C)
    s = expand_plane(tensors["s"].unsqueeze(0), C)[0]
    dev = m.device
    return torch.where(s.unsqueeze(0).expand(R, C),
                       torch.full((R, C), 2, dtype=torch.int64, device=dev),
                       m.to(torch.int64))


@torch.no_grad()
def unpack(tensors, meta) -> torch.Tensor:
    """DPK container -> bf16 [R, C] weight matrix (§3 invariant, bit-exact)."""
    R, C, g, NG = meta["R"], meta["C"], meta["g"], meta["NG"]
    dev = tensors["cb"].device

    b0 = expand_plane(tensors["b0"], C)                    # [R, C] bool
    b1 = expand_plane(tensors["b1"], C)

    code = b0.to(torch.int64) + 2 * b1.to(torch.int64)    # [R, C] 0..3
    part = part_matrix(tensors, meta)                      # [R, C] 0..2

    gidx = (torch.arange(C, device=dev, dtype=torch.int64) // g).unsqueeze(0)
    flat = (gidx * 3 + part) * 4 + code                    # [R, C]
    # fp8 cb -> bf16 before gather (fp8 gather is unimplemented; the cast is
    # bit-exact, so bf16 storage is unchanged and fp8 is lossless).
    cb = tensors["cb"].to(torch.bfloat16)
    W = cb.reshape(R, NG * 12).gather(1, flat)             # bf16 [R, C]
    return W


def unpack_file(path, device="cpu"):
    tensors, meta = load_container(path, device)
    return unpack(tensors, meta), meta
