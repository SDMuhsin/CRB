#!/usr/bin/env python
"""DPKA v1 exporter: K31 DPK dump -> single mmap-able artifact + JSON manifest.

Reads the 196 <name>.dpk.safetensors containers of a DOML K31 dump and writes
ONE artifact file with per-tensor sections:

  b0  : full low-code bit plane, raw (R*C/8 bytes, LSB-first per spec §2.1)
  b1  : high-code bits of NON-BULK elements only (bulk b1==0, verified
        invariant); packed LSB-first in ascending column order, each row's
        segment byte-aligned; rows concatenated.
  m   : membership bits of NON-SALIENT columns only (salient m==0, verified
        invariant), row-major scan, entropy-coded with the static binary rANS
        coder of rans.py (per-tensor f1 stored in the TOC).
  s   : salient column bitmap, raw (C/8 bytes).
  cb  : 10 real fp8-e4m3fn codebook slots per (row,group):
        [bulk0, bulk1, tail0..3, sal0..3]. The two dropped bulk pad slots are
        byte-identical to bulk1 (asserted; spec §2.3 pad-replication contract).

File layout (little-endian throughout):
  0x00  char[8]  magic  "DPKART01"
  0x08  u32      version (=1)
  0x0C  u32      n_tensors
  0x10  u64      total_weights (sum of R*C_orig)
  0x18  u64      file_size (whole file, bytes)
  0x20  u64      toc_off (=64)
  0x28  pad to 64
  0x40  TOC: n_tensors x 256-byte TensorRec:
        0x00 char[128] name (NUL-padded)
        0x80 u32 R, 0x84 u32 C, 0x88 u32 C_orig, 0x8C u32 g,
        0x90 u32 NG, 0x94 u32 n_sal,
        0x98 u64 n_nonbulk, 0xA0 u64 n_m_bits,
        0xA8 u32 f1 (rANS freq of tail=1 out of 2^15), 0xAC u32 reserved(=0)
        0xB0 u64 off[5]  (absolute file offsets: b0, b1, m, s, cb)
        0xD8 u64 size[5] (payload sizes in bytes,  same order)
        (every payload 64-byte aligned)
  payloads...

Edge cases REJECTED at export (absent in this dump, per brief):
  pad columns (C != C_orig), short last group (C % g != 0),
  mmode != "element" (colmem), cbdtype != float8_e4m3fn, NaN fp8 in cb.

G-ROUNDTRIP (Python side) is enforced inline: every tensor's m stream is
decoded back with the pure-Python decoder and compared before being written.
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rans import PROB_BITS, M, encode_bits, decode_bits, pick_f1

MAGIC = b"DPKART01"
VERSION = 1
TOC_OFF = 64
REC_SIZE = 256
ALIGN = 64
PLANES = ("b0", "b1", "m", "s", "cb")

SUBLAYERS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

DTYPE_ITEMSIZE = {"U32": 4, "F8_E4M3": 1, "BF16": 2}


def read_safetensors(path):
    with open(path, "rb") as f:
        blob = f.read()
    (hlen,) = struct.unpack("<Q", blob[:8])
    hdr = json.loads(blob[8:8 + hlen].decode("utf-8"))
    data = blob[8 + hlen:]
    meta = json.loads(hdr.pop("__metadata__")["meta"])
    out = {}
    for name, d in hdr.items():
        a, b = d["data_offsets"]
        n = int(np.prod(d["shape"])) if d["shape"] else 1
        assert b - a == n * DTYPE_ITEMSIZE[d["dtype"]], (name, d)
        out[name] = (d["dtype"], tuple(d["shape"]), data[a:b])
    return out, meta


def expand_bits_lsb_first(words_u32, C):
    u8 = words_u32.astype("<u4").view(np.uint8)
    bits = np.unpackbits(u8, axis=-1, bitorder="little")
    return bits.reshape(*words_u32.shape[:-1], C)


def process_tensor(dump_dir, name):
    """Returns (stats dict, payloads dict plane->bytes)."""
    tensors, meta = read_safetensors(
        os.path.join(dump_dir, f"{name}.dpk.safetensors"))
    R, C, C_orig, g, NG = (meta["R"], meta["C"], meta["C_orig"],
                           meta["g"], meta["NG"])

    # ---- edge-case rejection (brief: assert, do not implement speculatively)
    assert meta["mmode"] == "element", \
        f"{name}: mmode={meta['mmode']!r} (colmem) not supported by DPKA v1"
    assert meta["cbdtype"] == "float8_e4m3fn", \
        f"{name}: cbdtype={meta['cbdtype']!r} not supported by DPKA v1"
    assert C == C_orig, f"{name}: pad columns (C={C} != C_orig={C_orig}) rejected"
    assert C % g == 0, f"{name}: short last group (C={C} % g={g}) rejected"
    assert C % 32 == 0 and NG == C // g, (name, C, g, NG)

    def u32t(key, shape):
        dt, sh, raw = tensors[key]
        assert dt == "U32" and sh == shape, (name, key, dt, sh)
        return np.frombuffer(raw, dtype="<u4").reshape(shape)

    b0w = u32t("b0", (R, C // 32))
    b1w = u32t("b1", (R, C // 32))
    mw = u32t("m", (R, C // 32))
    sw = u32t("s", (C // 32,))
    dt, sh, cb_raw = tensors["cb"]
    assert dt == "F8_E4M3" and sh == (R, NG, 3, 4), (name, dt, sh)
    cb = np.frombuffer(cb_raw, dtype=np.uint8).reshape(R, NG, 3, 4)

    # ---- invariant asserts
    assert not np.any((cb & 0x7F) == 0x7F), f"{name}: NaN fp8 byte in cb"
    assert np.array_equal(cb[:, :, 0, 2], cb[:, :, 0, 1]) and \
        np.array_equal(cb[:, :, 0, 3], cb[:, :, 0, 1]), \
        f"{name}: bulk cb pad-replication violated (cannot drop to 10 slots)"

    b1e = expand_bits_lsb_first(b1w, C)          # [R, C] 0/1
    me = expand_bits_lsb_first(mw, C)
    se = expand_bits_lsb_first(sw[None, :], C)[0]  # [C]
    assert not np.any(me & se[None, :]), f"{name}: m AND s != 0"

    nonsal = se == 0
    n_sal = int(C - nonsal.sum())
    nb_mask = (me | se[None, :]).astype(bool)     # non-bulk = tail | salient
    assert not np.any(b1e[~nb_mask]), f"{name}: b1 != 0 on a bulk element"
    n_nonbulk = int(nb_mask.sum())

    # ---- b1 plane: per-row byte-aligned non-bulk bits
    b1_parts = []
    for r in range(R):
        b1_parts.append(np.packbits(b1e[r][nb_mask[r]], bitorder="little").tobytes())
    b1_payload = b"".join(b1_parts)

    # ---- m plane: non-salient bits, row-major, rANS coded
    m_ns = me[:, nonsal]                          # [R, C-n_sal]
    n_m_bits = int(m_ns.size)
    n_tail = int(m_ns.sum())
    f1 = pick_f1(n_tail, n_m_bits)
    bits_list = m_ns.ravel().tolist()
    t0 = time.time()
    m_payload = encode_bits(bits_list, f1)
    t_enc = time.time() - t0
    t0 = time.time()
    back = decode_bits(m_payload, n_m_bits, f1)
    t_dec = time.time() - t0
    assert np.array_equal(np.frombuffer(bytes(back), dtype=np.uint8),
                          m_ns.ravel()), f"{name}: rANS round-trip FAILED"

    # ---- cb plane: 10 real slots [bulk0,bulk1,tail0..3,sal0..3]
    cb10 = np.concatenate(
        [cb[:, :, 0, :2], cb[:, :, 1, :], cb[:, :, 2, :]], axis=-1)
    assert cb10.shape == (R, NG, 10)
    cb_payload = cb10.tobytes()

    p = n_tail / n_m_bits
    ent = 0.0
    if 0.0 < p < 1.0:
        ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    payloads = {
        "b0": tensors["b0"][2],                   # raw bytes, layout-identical
        "b1": b1_payload,
        "m": bytes(m_payload),
        "s": tensors["s"][2],
        "cb": cb_payload,
    }
    stats = {
        "name": name, "R": R, "C": C, "C_orig": C_orig, "g": g, "NG": NG,
        "n_sal_cols": n_sal, "n_nonbulk": n_nonbulk,
        "n_m_bits": n_m_bits, "n_m_tail": n_tail, "f1": f1,
        "m_p_tail": p, "m_iid_entropy_bits_per_bit": float(ent),
        "m_coded_bits_per_bit": 8.0 * len(m_payload) / max(n_m_bits, 1),
        "rans_encode_s": round(t_enc, 3), "rans_decode_s": round(t_dec, 3),
        "roundtrip_py": True,
        "plane_bytes": {k: len(v) for k, v in payloads.items()},
    }
    return stats, payloads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="downloads/doml_dumps/qwen3-0.6b/"
                                      "k31-rdsplit-lam7e-5-g256-atuned")
    ap.add_argument("--out", default="downloads/cpu_kernel_rnd/"
                                     "qwen3-0.6b-k31.dpka")
    ap.add_argument("--manifest", default=None,
                    help="default: <out>.manifest.json")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: export only first N tensors")
    args = ap.parse_args()
    manifest_path = args.manifest or args.out + ".manifest.json"

    names = [f"model.layers.{i}.{sub}"
             for i in range(args.layers) for sub in SUBLAYERS]
    if args.limit:
        names = names[:args.limit]
    for n in names:
        p = os.path.join(args.dump, f"{n}.dpk.safetensors")
        assert os.path.exists(p), f"missing container: {p}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_tensors = len(names)
    toc_end = TOC_OFF + n_tensors * REC_SIZE

    recs = []
    all_stats = []
    total_weights = 0
    align_pad = 0
    t_start = time.time()
    with open(args.out, "wb") as f:
        f.write(b"\0" * toc_end)                  # reserve header + TOC
        pos = toc_end
        for i, name in enumerate(names):
            stats, payloads = process_tensor(args.dump, name)
            offs = {}
            for plane in PLANES:
                pad = (-pos) % ALIGN
                if pad:
                    f.write(b"\0" * pad)
                    pos += pad
                    align_pad += pad
                offs[plane] = (pos, len(payloads[plane]))
                f.write(payloads[plane])
                pos += len(payloads[plane])
            stats["plane_offsets"] = {k: offs[k][0] for k in PLANES}
            total_weights += stats["R"] * stats["C_orig"]
            recs.append((stats, offs))
            all_stats.append(stats)
            w = stats["R"] * stats["C_orig"]
            tbytes = sum(len(v) for v in payloads.values())
            print(f"[{i+1:3d}/{n_tensors}] {name:48s} "
                  f"{8.0*tbytes/w:6.4f} bpw  "
                  f"m={stats['m_coded_bits_per_bit']:.4f}b/b "
                  f"(H={stats['m_iid_entropy_bits_per_bit']:.4f}) "
                  f"enc={stats['rans_encode_s']:.1f}s", flush=True)
        file_size = pos

        # header
        f.seek(0)
        hdr = MAGIC + struct.pack("<IIQQQ", VERSION, n_tensors,
                                  total_weights, file_size, TOC_OFF)
        assert len(hdr) == 0x28
        f.write(hdr + b"\0" * (TOC_OFF - len(hdr)))
        # TOC
        for stats, offs in recs:
            nm = stats["name"].encode()
            assert len(nm) < 128
            rec = nm + b"\0" * (128 - len(nm))
            rec += struct.pack("<6I", stats["R"], stats["C"], stats["C_orig"],
                               stats["g"], stats["NG"], stats["n_sal_cols"])
            rec += struct.pack("<QQ", stats["n_nonbulk"], stats["n_m_bits"])
            rec += struct.pack("<II", stats["f1"], 0)
            rec += struct.pack("<5Q", *(offs[p][0] for p in PLANES))
            rec += struct.pack("<5Q", *(offs[p][1] for p in PLANES))
            assert len(rec) == REC_SIZE
            f.write(rec)

    # ---- manifest
    tot_plane = {k: sum(s["plane_bytes"][k] for s in all_stats) for k in PLANES}
    header_toc = toc_end
    manifest = {
        "format": "DPKA", "version": VERSION,
        "created_unix": int(time.time()),
        "dump_dir": args.dump, "artifact": args.out,
        "prob_bits": PROB_BITS,
        "n_tensors": n_tensors, "total_weights": total_weights,
        "file_size_bytes": file_size,
        "header_toc_bytes": header_toc, "align_pad_bytes": align_pad,
        "plane_total_bytes": tot_plane,
        "plane_total_bpw": {k: 8.0 * v / total_weights
                            for k, v in tot_plane.items()},
        "overhead_bpw": 8.0 * (header_toc + align_pad) / total_weights,
        "artifact_bpw": 8.0 * file_size / total_weights,
        "roundtrip_py_all": all(s["roundtrip_py"] for s in all_stats),
        "tensors": all_stats,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"\nwrote {args.out}: {file_size} bytes, {n_tensors} tensors, "
          f"{total_weights} weights")
    print(f"ARTIFACT bpw = {8.0*file_size/total_weights:.4f}")
    for k in PLANES:
        print(f"  {k:3s} {tot_plane[k]:12d} B  "
              f"{8.0*tot_plane[k]/total_weights:.4f} bpw")
    print(f"  hdr+toc+pad {header_toc+align_pad:8d} B  "
          f"{8.0*(header_toc+align_pad)/total_weights:.4f} bpw")
    print(f"G-ROUNDTRIP(py, inline): "
          f"{'PASS' if manifest['roundtrip_py_all'] else 'FAIL'} "
          f"({n_tensors}/{n_tensors} tensors)")
    print(f"total wall {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
