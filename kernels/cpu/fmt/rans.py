"""Static-probability binary rANS coder (byte-wise renormalization).

This is the DPKA v1 artifact m-plane coder. Pure Python + stdlib; the C
decoder in kernels/cpu/fmt/dpka_load.c implements the identical decode loop.

Specification (normative — precise enough to reimplement):
  - Alphabet: bits {0, 1}. bit=1 means "tail" membership.
  - Probability model: STATIC per stream. f1 = frequency of bit 1 out of
    M = 2^PROB_BITS (PROB_BITS = 15); f0 = M - f1; 1 <= f1 <= M-1.
    Cumulative frequencies: cum0 = 0, cum1 = f0.
  - State: unsigned 32-bit x, lower bound L = 2^23. This is the classic
    "rans_byte" construction (Duda's rANS, ryg's byte-wise variant):
    invariant L <= x < L*256 between symbols (encoder side, pre-renorm).
  - ENCODE processes symbols in REVERSE order, starting from x = L:
      for each bit b (last symbol first):
        f = fb, c = cumb
        while x >= (f << 16): emit byte (x & 0xFF); x >>= 8
        x = (x // f) << PROB_BITS | ... precisely: ((x // f) << 15) + (x % f) + c
      Renorm bound f << 16 == ((L >> PROB_BITS) << 8) * f.
    Final stream = [x as 4 little-endian bytes] + reverse(emitted bytes).
  - DECODE reads the stream forward: x = LE32(stream[0:4]); pos = 4;
      for each of n bits:
        slot = x & (M-1)
        bit = (slot >= f0)
        x = f_bit * (x >> 15) + slot - cum_bit
        while x < L: x = (x << 8) | stream[pos]; pos += 1
    Termination invariant (asserted): after n symbols x == L and
    pos == len(stream). This is a strong integrity check: the decoder
    provably re-arrives at the encoder's initial state.
  - Empty stream (n = 0) is 4 bytes: LE32(L).

Rate: n*H(f1/M) + O(1); the O(1) is the 4-byte flush + probability
quantization loss (~1e-5 bits/bit at PROB_BITS=15).
"""

PROB_BITS = 15
M = 1 << PROB_BITS
RANS_L = 1 << 23


def pick_f1(n_ones: int, n: int) -> int:
    """Deterministic per-stream frequency: round(n_ones/n * M), clamped to [1, M-1]."""
    assert n > 0
    f1 = (n_ones * M + n // 2) // n
    return max(1, min(M - 1, f1))


def encode_bits(bits, f1: int) -> bytes:
    """bits: sequence of 0/1 ints (list is fastest). Returns the coded stream."""
    assert 1 <= f1 <= M - 1
    f0 = M - f1
    xmax0 = f0 << 16
    xmax1 = f1 << 16
    x = RANS_L
    out = bytearray()
    ap = out.append
    for b in reversed(bits):
        if b:
            f = f1
            c = f0
            xmax = xmax1
        else:
            f = f0
            c = 0
            xmax = xmax0
        while x >= xmax:
            ap(x & 0xFF)
            x >>= 8
        q, r = divmod(x, f)
        x = (q << PROB_BITS) + r + c
    out.reverse()
    return x.to_bytes(4, "little") + bytes(out)


def decode_bits(stream: bytes, n: int, f1: int) -> bytearray:
    """Inverse of encode_bits. Returns bytearray of n 0/1 values.

    Asserts the termination invariant (final state == RANS_L, all bytes
    consumed) — any corruption or wrong (n, f1) trips it.
    """
    assert 1 <= f1 <= M - 1
    f0 = M - f1
    mask = M - 1
    x = int.from_bytes(stream[:4], "little")
    pos = 4
    slen = len(stream)
    out = bytearray(n)
    for i in range(n):
        slot = x & mask
        if slot >= f0:
            out[i] = 1
            x = f1 * (x >> PROB_BITS) + slot - f0
        else:
            x = f0 * (x >> PROB_BITS) + slot
        while x < RANS_L:
            x = (x << 8) | stream[pos]
            pos += 1
    assert x == RANS_L and pos == slen, \
        f"rANS termination invariant violated: x={x} pos={pos} len={slen}"
    return out


if __name__ == "__main__":
    # smoke self-test
    import random
    rng = random.Random(0)
    for p in (0.001, 0.1981, 0.5, 0.93):
        for n in (0, 1, 7, 1000, 65537):
            bits = [1 if rng.random() < p else 0 for _ in range(n)]
            f1 = pick_f1(sum(bits), n) if n else 1
            s = encode_bits(bits, f1)
            back = decode_bits(s, n, f1)
            assert list(back) == bits, (p, n)
            print(f"p={p} n={n} coded={len(s)}B ok")
    print("rans.py self-test PASS")
