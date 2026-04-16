"""
Spectral-Binary Hybrid (SBH) quantization.

Decomposes each weight matrix W into:
  W_q = U_r @ diag(S_r) @ V_r^T + mu + alpha * sign(R - mu)

where:
  - U_r, S_r, V_r are the top-r singular triplets (stored at fp16)
  - R = W - U_r @ diag(S_r) @ V_r^T is the residual
  - mu, alpha are per-row mean and scale of R (stored at fp16)
  - sign(R - mu) is the binary sign matrix (1 bit per weight)

Bitrate per weight: b = 1 + 16*r*(m+k)/(m*k) + 32/k
"""

import torch


@torch.no_grad()
def sbh_quantize_weight(W, rank):
    """Quantize a weight matrix using Spectral-Binary Hybrid.

    Args:
        W: [out_features, in_features] weight tensor
        rank: truncation rank for SVD (0 = pure binary, no SVD)

    Returns:
        W_q: quantized weight (dense tensor, same shape as W)
    """
    orig_dtype = W.dtype
    W_f = W.float()
    m, k = W_f.shape

    if rank > 0:
        # Truncated SVD
        U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
        r = min(rank, min(m, k))
        W_svd = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]

        # Residual
        R = W_f - W_svd
    else:
        W_svd = torch.zeros_like(W_f)
        R = W_f

    # Binary approximation of residual (per-row)
    mu = R.mean(dim=1, keepdim=True)
    R_centered = R - mu
    alpha = R_centered.abs().mean(dim=1, keepdim=True)
    B = torch.sign(R_centered)
    # Handle exact zeros in sign (rare but possible)
    B[B == 0] = 1.0

    R_binary = mu + alpha * B
    W_q = W_svd + R_binary

    return W_q.to(orig_dtype)


@torch.no_grad()
def sbh_quantize_weight_multi(W, rank, binary_order=2):
    """SBH with multi-pass binary residual (like BRAQ's residual expansion).

    Args:
        W: [out_features, in_features] weight tensor
        rank: truncation rank for SVD
        binary_order: number of binary residual passes (1=basic, 2=like BRAQ)

    Returns:
        W_q: quantized weight (dense tensor, same shape as W)
    """
    orig_dtype = W.dtype
    W_f = W.float()
    m, k = W_f.shape

    if rank > 0:
        U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
        r = min(rank, min(m, k))
        W_svd = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
        R = W_f - W_svd
    else:
        W_svd = torch.zeros_like(W_f)
        R = W_f

    # Multi-pass binary residual (same as high_order_residual logic)
    sum_binary = torch.zeros_like(R)
    current = R.clone()
    for _ in range(binary_order):
        residual = current - sum_binary
        mu = residual.mean(dim=1, keepdim=True)
        centered = residual - mu
        alpha = centered.abs().mean(dim=1, keepdim=True)
        B = torch.sign(centered)
        B[B == 0] = 1.0
        sum_binary = sum_binary + mu + alpha * B

    W_q = W_svd + sum_binary
    return W_q.to(orig_dtype)


def compute_bitrate(m, k, rank, binary_order=1):
    """Compute bits per weight for SBH at given rank.

    Breakdown:
      - SVD factors: U_r [m,r] + S_r [r] + V_r [k,r] at 16 bits
      - Binary: binary_order bits per weight + per-row mu,alpha per order at 16 bits
    """
    svd_bits = 16 * rank * (m + k + 1)
    binary_bits = binary_order * m * k  # sign bits
    scale_bits = binary_order * m * 32  # mu + alpha per row per order
    total_bits = svd_bits + binary_bits + scale_bits
    total_weights = m * k
    return total_bits / total_weights


def allocate_ranks_fixed(sublayer_shapes, r_attn, r_mlp):
    """Fixed rank allocation: r_attn for attention, r_mlp for MLP.

    Args:
        sublayer_shapes: dict of {name: (m, k)}
        r_attn: rank for attention sublayers
        r_mlp: rank for MLP sublayers

    Returns:
        ranks: dict of {name: rank}
        avg_bitrate: average bits per weight
    """
    attn_names = {'self_attn.q_proj', 'self_attn.k_proj',
                  'self_attn.v_proj', 'self_attn.o_proj'}

    ranks = {}
    total_bits = 0
    total_params = 0

    for name, (m, k) in sublayer_shapes.items():
        r = r_attn if name in attn_names else r_mlp
        r = min(r, min(m, k))
        ranks[name] = r
        total_bits += compute_bitrate(m, k, r) * m * k
        total_params += m * k

    avg_bitrate = total_bits / total_params
    return ranks, avg_bitrate


def allocate_ranks_greedy(sublayer_info, bit_budget=2.0, binary_order=1):
    """Greedy rank allocation to minimize total error under bit budget.

    Args:
        sublayer_info: list of dicts with keys: name, m, k, singular_values
        bit_budget: target average bits per weight
        binary_order: number of binary passes

    Returns:
        ranks: dict of {name: rank}
        avg_bitrate: achieved average bitrate
    """
    total_params = sum(s['m'] * s['k'] for s in sublayer_info)
    ranks = {s['name']: 0 for s in sublayer_info}

    while True:
        # Current bitrate
        total_bits = sum(
            compute_bitrate(s['m'], s['k'], ranks[s['name']], binary_order) * s['m'] * s['k']
            for s in sublayer_info
        )
        avg = total_bits / total_params
        if avg >= bit_budget:
            break

        # Find best sublayer to add one rank
        best_name = None
        best_ratio = -1
        for s in sublayer_info:
            r = ranks[s['name']]
            if r >= len(s['singular_values']) or r >= min(s['m'], s['k']):
                continue
            # Error reduction: sigma_{r+1}^2
            error_red = s['singular_values'][r].item() ** 2
            # Bit cost for one more rank
            bit_cost = 16 * (s['m'] + s['k'] + 1) / total_params
            ratio = error_red / bit_cost if bit_cost > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = s['name']

        if best_name is None:
            break
        ranks[best_name] += 1

    total_bits = sum(
        compute_bitrate(s['m'], s['k'], ranks[s['name']], binary_order) * s['m'] * s['k']
        for s in sublayer_info
    )
    avg_bitrate = total_bits / total_params
    return ranks, avg_bitrate
