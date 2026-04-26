"""
LNQ: Layerwise Non-Uniform Quantization (from GuidedQuant, ICML 2025)
Standalone runner for apples-to-apples comparison with DOML

Full pipeline modes:
  --full_pipeline: Runs all 3 stages of GuidedQuant:
    Stage 1: Fisher-weighted k-means initialization (SqueezeLLM)
    Stage 2: Saliency-weighted Hessian computation
    Stage 3: LNQ alternating optimization

  --full_pipeline --is_nosal: Fisher init + standard Hessian (fair to DOML)

  Without --full_pipeline: Core-only LNQ (simple k-means + standard Hessian)

Usage:
    source env/bin/activate
    # Full GuidedQuant pipeline:
    python3 -u src/run_lnq.py Qwen/Qwen3-0.6B wikitext2 --full_pipeline --device cuda:0
    # Full pipeline without saliency (fair comparison to DOML):
    python3 -u src/run_lnq.py Qwen/Qwen3-0.6B wikitext2 --full_pipeline --is_nosal --device cuda:0
    # Core-only (backward-compatible):
    python3 -u src/run_lnq.py Qwen/Qwen3-0.6B wikitext2 --device cuda:0
"""

import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datautils import get_loaders, get_tokenizer, set_seed
from modelutils import find_layers

try:
    import numba
    import flash1dkmeans
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =====================================================================
# Stage 1: Fisher-Weighted K-Means Initialization (SqueezeLLM)
# =====================================================================

if HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _fisher_kmeans_matrix(W, F, K, max_iter):
        """
        Fisher-weighted 1D k-means for all rows of a weight matrix in parallel.
        Uses flash1dkmeans for the core weighted clustering.

        Args:
            W: (output_dim, input_dim) float32 numpy array
            F: (output_dim, input_dim) float32 numpy array (squared gradients)
            K: int number of clusters
            max_iter: int max Lloyd iterations

        Returns:
            labels: (output_dim, input_dim) int8 numpy array
            centroids: (output_dim, K) float32 numpy array
        """
        output_dim, input_dim = W.shape
        all_labels = np.empty((output_dim, input_dim), dtype=np.int8)
        all_centroids = np.empty((output_dim, K), dtype=np.float32)

        for r in numba.prange(output_dim):
            w = W[r, :].copy()
            f = F[r, :].copy()

            for i in range(input_dim):
                if w[i] == 0.0:
                    f[i] = 0.0

            order = np.argsort(w)
            sw = np.empty(input_dim, dtype=np.float32)
            sf = np.empty(input_dim, dtype=np.float64)
            for i in range(input_dim):
                sw[i] = w[order[i]]
                sf[i] = np.float64(f[order[i]])

            sf_cum = np.cumsum(sf)
            total_f = sf_cum[input_dim - 1]

            if total_f == 0.0:
                sf_cum = np.arange(1, input_dim + 1, dtype=np.float64)
                sw_f64 = sw.astype(np.float64)
                swf_cum = np.cumsum(sw_f64)
                swf2_cum = np.cumsum(sw_f64 * sw_f64)
            else:
                swf = np.empty(input_dim, dtype=np.float64)
                for i in range(input_dim):
                    swf[i] = np.float64(sw[i]) * sf[i]
                swf_cum = np.cumsum(swf)
                swf2 = np.empty(input_dim, dtype=np.float64)
                for i in range(input_dim):
                    swf2[i] = swf[i] * np.float64(sw[i])
                swf2_cum = np.cumsum(swf2)

            if K > 2:
                centroids, borders = flash1dkmeans.numba_kmeans_1d_k_cluster(
                    sorted_X=sw,
                    n_clusters=K,
                    max_iter=max_iter,
                    weights_prefix_sum=sf_cum,
                    weighted_X_prefix_sum=swf_cum,
                    weighted_X_squared_prefix_sum=swf2_cum,
                    start_idx=0,
                    stop_idx=input_dim,
                )
            else:
                centroids, borders = flash1dkmeans.numba_kmeans_1d_two_cluster(
                    sorted_X=sw,
                    weights_prefix_sum=sf_cum,
                    weighted_X_prefix_sum=swf_cum,
                    start_idx=0,
                    stop_idx=input_dim,
                )

            for k in range(K):
                all_centroids[r, k] = np.float32(centroids[k])

            labels_sorted = np.empty(input_dim, dtype=np.int8)
            for k in range(K):
                for i in range(borders[k], borders[k + 1]):
                    labels_sorted[i] = np.int8(k)

            for i in range(input_dim):
                all_labels[r, order[i]] = labels_sorted[i]

        return all_labels, all_centroids


def fisher_kmeans_init(W, fisher_grad, K=4):
    """
    Fisher-weighted 1D k-means initialization for quantization codebook.

    Args:
        W: (output_dim, input_dim) torch tensor on any device
        fisher_grad: (output_dim, input_dim) torch tensor — squared weight gradients
        K: number of clusters (default 4 for 2-bit)

    Returns:
        labels: (output_dim, input_dim) int8 torch tensor
        centroids: (output_dim, K) float32 torch tensor
    """
    if not HAS_NUMBA:
        raise RuntimeError(
            "numba and flash1dkmeans required for Fisher k-means. "
            "Install: pip install numba==0.60.0 flash1dkmeans==0.1.4"
        )

    W_np = W.cpu().float().numpy()
    F_np = fisher_grad.cpu().float().numpy()
    labels_np, centroids_np = _fisher_kmeans_matrix(W_np, F_np, K, 50)
    return torch.from_numpy(labels_np), torch.from_numpy(centroids_np)


# =====================================================================
# Stage 2: Fisher Gradient + Saliency Computation
# =====================================================================

def compute_fisher_data(model, dataloader, dev, model_type, num_groups=4,
                        compute_sal=True):
    """
    Full-model forward+backward to compute Fisher diagonal and saliency.

    Runs the full model on each calibration sample with gradients enabled.
    A square_grad_hook on each weight replaces grad with grad^2 during backward,
    so weight.grad accumulates the Fisher diagonal (sum of squared gradients).
    If compute_sal=True, forward hooks on sublayer outputs capture per-sample
    saliency (squared output gradients grouped by channel).

    Args:
        model: HuggingFace causal LM model (moved to GPU temporarily)
        dataloader: calibration data from get_loaders (list of (input_ids,) tuples)
        dev: torch.device for computation
        model_type: 'opt', 'llama', or 'qwen'
        num_groups: channel groups for saliency (default 4)
        compute_sal: whether to compute saliency (False for is_nosal)

    Returns:
        weight_grads: dict {(layer_idx, name): tensor(out_dim, in_dim)} on CPU
        saliency: dict {(layer_idx, name): tensor(N, seq_len, num_groups)} on CPU,
                  or None if compute_sal=False
    """
    if model_type == 'opt':
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    model.to(dev)
    # Use train mode so gradient_checkpointing actually activates
    # (HF only checkpoints when model.training=True). Safe because
    # Qwen3/LLaMA use RMSNorm (not BatchNorm) and zero dropout.
    model.train()
    model.gradient_checkpointing_enable()

    sublayer_registry = {}
    for layer_idx, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            sublayer_registry[(layer_idx, name)] = module

    # Hook 1: square weight gradients during backward (Fisher diagonal)
    def square_grad_hook(grad):
        return grad.pow(2)

    weight_hooks = []
    for (layer_idx, name), module in sublayer_registry.items():
        h = module.weight.register_hook(square_grad_hook)
        weight_hooks.append(h)

    # Hook 2: capture saliency from sublayer output gradients
    saliency_data = {}
    saliency_hooks = []

    if compute_sal:
        for (layer_idx, name), module in sublayer_registry.items():
            saliency_data[(layer_idx, name)] = []

            def _make_hook(li, n, ng):
                def fwd_hook(mod, inp, out):
                    out.retain_grad()
                    def grad_hook(grad):
                        bsz, seq_len, hidden_dim = grad.shape
                        group_size = hidden_dim // ng
                        grad_sq = (grad.float() * 1e3).pow(2)
                        grad_sq = grad_sq.view(bsz, seq_len, ng, group_size)
                        grad_sq = grad_sq.mean(dim=-1)
                        saliency_data[(li, n)].append(grad_sq.bfloat16().cpu())
                    out.register_hook(grad_hook)
                return fwd_hook

            h = module.register_forward_hook(_make_hook(layer_idx, name, num_groups))
            saliency_hooks.append(h)

    print(f"  Computing Fisher data ({len(dataloader)} samples, "
          f"saliency={'on' if compute_sal else 'off'})...")
    t0 = time.time()

    for i, batch in enumerate(dataloader):
        input_ids = batch[0].to(dev)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        if (i + 1) % 32 == 0:
            print(f"    sample {i+1}/{len(dataloader)}")

    print(f"  Fisher data computed in {time.time()-t0:.1f}s")

    for h in weight_hooks:
        h.remove()
    for h in saliency_hooks:
        h.remove()

    weight_grads = {}
    for (layer_idx, name), module in sublayer_registry.items():
        if module.weight.grad is not None:
            weight_grads[(layer_idx, name)] = module.weight.grad.detach().cpu().half()
        else:
            out_dim, in_dim = module.weight.shape
            weight_grads[(layer_idx, name)] = torch.zeros(out_dim, in_dim)

    saliency = None
    if compute_sal:
        saliency = {}
        for key, chunks in saliency_data.items():
            if chunks:
                saliency[key] = torch.cat(chunks, dim=0)
            else:
                saliency[key] = None
        del saliency_data

    model.gradient_checkpointing_disable()
    model.eval()
    model.zero_grad(set_to_none=True)
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    return weight_grads, saliency


# =====================================================================
# Saliency-Weighted Hessian Hook (Stage 2, layer-by-layer)
# =====================================================================

class SaliencyHessianHook:
    """
    Forward hook that accumulates H = X^T diag(S) X for a linear sublayer,
    where S is pre-computed per-sample saliency.

    Result shape: (input_dim, input_dim, num_groups).
    Call get_hessian() to retrieve as (num_groups, input_dim, input_dim).
    """

    def __init__(self, layer_module, saliency_tensor, device):
        """
        Args:
            layer_module: nn.Linear sublayer
            saliency_tensor: (N, seq_len, num_groups) float tensor on CPU
            device: torch.device for accumulation
        """
        self.columns = layer_module.weight.shape[1]
        self.num_groups = saliency_tensor.shape[-1]
        self.XTX = torch.zeros(
            self.columns, self.columns, self.num_groups,
            dtype=torch.float32, device=device,
        )
        self.saliency = saliency_tensor.float()
        self.sample_idx = 0

    def __call__(self, module, inp, out):
        x = inp[0].data
        if x.dim() == 2:
            x = x.unsqueeze(0)
        bsz = x.shape[0]

        sal = self.saliency[self.sample_idx : self.sample_idx + bsz].to(
            self.XTX.device
        )
        self.sample_idx += bsz

        x_flat = x.reshape(-1, x.shape[-1]).float()
        sal_flat = sal.reshape(-1, sal.shape[-1]).float()

        sal_x = torch.einsum("nj,ng->njg", x_flat, sal_flat)
        block = torch.einsum("ni,njg->ijg", x_flat, sal_x)
        self.XTX += block

    def get_hessian(self):
        """Return Hessian as (num_groups, input_dim, input_dim) on CPU."""
        return self.XTX.permute(2, 0, 1).cpu()


# =====================================================================
# LNQ Core Algorithm
# Ported from GuidedQuant's any_precision/quantization/layerwise_quantize.py
# =====================================================================

@torch.no_grad()
def row_kmeans_1d(W, K=4, max_iter=20):
    """
    Row-wise 1D k-means to initialize LNQ codebook (simple, unweighted).

    For each output row of W, cluster the input_dim weight values into K centroids.

    Args:
        W: weight matrix (output_dim, input_dim) on GPU, float32
        K: number of clusters (4 for 2-bit)
        max_iter: Lloyd iterations

    Returns:
        labels: (output_dim, input_dim) int8
        centroids: (output_dim, K) float32
    """
    device = W.device
    output_dim, input_dim = W.shape

    W_sorted = W.sort(dim=1).values
    chunk_size = input_dim // K
    centroids = torch.zeros(output_dim, K, device=device, dtype=W.dtype)
    for k in range(K):
        start = k * chunk_size
        end = (k + 1) * chunk_size if k < K - 1 else input_dim
        centroids[:, k] = W_sorted[:, start:end].mean(dim=1)

    for _ in range(max_iter):
        dists = (W.unsqueeze(2) - centroids.unsqueeze(1)).abs()
        labels = dists.argmin(dim=2)

        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = (labels == k).float()
            count = mask.sum(dim=1).clamp(min=1)
            new_centroids[:, k] = (W * mask).sum(dim=1) / count
        centroids = new_centroids

    return labels.to(torch.int8), centroids


@torch.no_grad()
def objective_function(W, H, labels, C):
    """
    Hessian-weighted quantization error: mean over groups of ||W_hat - W||_H^2

    Args:
        W: (output_dim, input_dim) on GPU
        H: (num_groups, input_dim, input_dim) on GPU
        labels: (output_dim, input_dim) int8
        C: (output_dim, K) float32

    Returns:
        scalar loss
    """
    device = W.device
    labels_dev = labels.to(device)
    C_dev = C.to(device)
    W_hat = torch.gather(
        C_dev.unsqueeze(1).expand(-1, labels_dev.shape[1], -1),
        dim=2,
        index=labels_dev.unsqueeze(-1).long(),
    ).squeeze(-1)
    delta_w = W_hat - W

    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups
    delta_w = delta_w.reshape(num_groups, group_size, delta_w.shape[-1])
    objective_value = torch.einsum("nij,njk,nik->i", delta_w, H, delta_w)
    return objective_value.mean()


def update_P(W, H, labels, C, cd_cycles, cd_block_size=128):
    """
    Update assignments via coordinate descent with Hessian error feedback.

    Args:
        W: (output_dim, input_dim) original weights, on GPU
        H: (num_groups, input_dim, input_dim) Hessian, on GPU
        labels: (output_dim, input_dim) current assignments
        C: (output_dim, K) current codebook
        cd_cycles: number of coordinate descent sweeps
        cd_block_size: columns processed per block for error propagation

    Returns:
        updated labels (output_dim, input_dim) int8
    """
    device = W.device
    C = C.to(device)
    assignments_prev = labels.to(device).long()
    b, d = assignments_prev.shape
    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups

    assignments = assignments_prev.clone()

    W_hat = torch.gather(
        C.unsqueeze(1).expand(-1, d, -1), dim=2, index=assignments.unsqueeze(-1)
    ).squeeze(-1)

    W_grp = W.reshape(num_groups, group_size, d)
    C_grp = C.reshape(num_groups, group_size, C.shape[-1])
    W_hat_grp = W_hat.reshape(num_groups, group_size, d)
    H_grp = H.clone()
    B_grp = torch.zeros_like(W_grp)

    for i in range(num_groups):
        H_grp_diag = H_grp[
            i, torch.arange(d, device=device), torch.arange(d, device=device)
        ]
        H_grp_diag = H_grp_diag.reshape(1, 1, -1)
        H_grp[i, :, :] = H_grp[i, :, :] / H_grp_diag

    for k in range(cd_cycles):
        B_grp = torch.bmm(W_hat_grp - W_grp, torch.tril(H_grp, diagonal=-1))

        for start_idx in range(0, d, cd_block_size):
            end_idx = min(start_idx + cd_block_size, d)

            for update_idx in range(start_idx, end_idx):
                index = torch.arange(update_idx, update_idx + 1, device=device)
                sol = W_grp[:, :, index] - B_grp[:, :, index]

                sol_dist = torch.abs(sol - C_grp)
                argmin_dist = sol_dist.min(dim=-1).indices

                assignments[:, index] = argmin_dist.reshape(-1, 1)
                W_hat_grp[:, :, index] = torch.gather(
                    C_grp, dim=-1, index=argmin_dist.unsqueeze(-1)
                )

                if update_idx < end_idx - 1:
                    B_grp[:, :, update_idx + 1 : end_idx] += torch.bmm(
                        W_hat_grp[:, :, index] - W_grp[:, :, index],
                        H_grp[:, index, update_idx + 1 : end_idx],
                    )

            B_grp[:, :, end_idx:] += torch.bmm(
                W_hat_grp[:, :, start_idx:end_idx] - W_grp[:, :, start_idx:end_idx],
                H_grp[:, start_idx:end_idx, end_idx:],
            )

    num_changed = (assignments_prev != assignments).sum().item()
    pct_changed = num_changed / assignments_prev.numel() * 100
    print(f"    assignments changed: {pct_changed:.1f}%")

    return assignments.to(torch.int8)


def update_C(W, H, labels, C, sub_channel_size=64):
    """
    Update codebook via Hessian-weighted least-squares.

    Args:
        W: (output_dim, input_dim) on GPU
        H: (num_groups, input_dim, input_dim) on GPU
        labels: (output_dim, input_dim) int8
        C: (output_dim, K) float32
        sub_channel_size: batch size for output dimension processing

    Returns:
        updated C (output_dim, K) float32, on CPU
    """
    device = W.device
    channel_size = W.shape[0]
    input_size = H.shape[1]
    sub_input_size = 2**16

    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups
    n_cluster = C.shape[-1]

    sub_channel_size = min(sub_channel_size, group_size)
    while group_size % sub_channel_size != 0 and sub_channel_size > 1:
        sub_channel_size -= 1

    L = torch.empty_like(H)
    for i in range(num_groups):
        L[i] = torch.linalg.cholesky(H[i])
    reduced_X = L.transpose(-2, -1)

    C_hat_list = []
    for st_idx in range(0, channel_size, sub_channel_size):
        group_idx = st_idx // group_size
        reduced_X_blk = reduced_X[group_idx]

        end_idx = min(st_idx + sub_channel_size, channel_size)

        A_batch_list, b_batch_list = [], []
        labels_batch = labels[st_idx:end_idx].to(device).long()
        for st_idx_inp in range(0, input_size, sub_input_size):
            end_idx_inp = min(st_idx_inp + sub_input_size, input_size)
            X_batch = reduced_X_blk[st_idx_inp:end_idx_inp].to(device)
            P_batch = F.one_hot(labels_batch, num_classes=n_cluster).float()
            A_batch_tmp = torch.einsum("bj,ijc->ibc", X_batch, P_batch)
            b_batch_tmp = torch.einsum(
                "bj,ij->ib", X_batch, W[st_idx:end_idx]
            ).unsqueeze(-1)
            A_batch_list.append(A_batch_tmp)
            b_batch_list.append(b_batch_tmp)

        A_batch = torch.cat(A_batch_list, dim=1)
        b_batch = torch.cat(b_batch_list, dim=1)

        lambda_reg = 1e-7
        batch_size_local = A_batch.shape[0]
        dtype, dev_local = A_batch.dtype, A_batch.device
        sqrt_lambda = torch.sqrt(
            torch.tensor(lambda_reg, dtype=dtype, device=dev_local)
        )
        I_reg = (
            sqrt_lambda
            * torch.eye(n_cluster, dtype=dtype, device=dev_local)
            .unsqueeze(0)
            .expand(batch_size_local, -1, -1)
        )

        A_batch = torch.cat([A_batch.transpose(1, 2), I_reg], dim=2).transpose(1, 2)
        zeros = torch.zeros(
            (batch_size_local, n_cluster, 1), dtype=dtype, device=dev_local
        )
        b_batch = torch.cat([b_batch, zeros], dim=1)

        C_hat_batch = torch.linalg.lstsq(A_batch, b_batch).solution
        if torch.isnan(C_hat_batch).any():
            print(f"  WARNING: NaN in codebook update for rows {st_idx}:{end_idx}")
            C_hat_batch = C[st_idx:end_idx].to(device).unsqueeze(-1)
        C_hat_batch = C_hat_batch.squeeze(-1)
        C_hat_list.append(C_hat_batch)

    return torch.cat(C_hat_list, dim=0).cpu()


def train_least_squares(
    W_np, init_labels, init_centroids, H_raw, num_iterations=3, cd_cycles=4
):
    """
    LNQ alternating optimization: iterate between assignment update (update_P)
    and codebook update (update_C) to minimize Hessian-weighted reconstruction error.

    Args:
        W_np: (output_dim, input_dim) float32 tensor, original weights
        init_labels: (output_dim, input_dim) int8 tensor, initial assignments
        init_centroids: (output_dim, K) float32 tensor, initial codebook
        H_raw: Hessian tensor — either (input_dim, input_dim) for single-group
               or (num_groups, input_dim, input_dim) for multi-group
        num_iterations: alternating optimization iterations (default 3)
        cd_cycles: coordinate descent cycles per update_P call (default 4)

    Returns:
        (labels, centroids): optimized assignments and codebook
    """
    device = torch.device("cuda")

    labels = init_labels.clone().to("cpu")
    C = init_centroids.clone().to("cpu")
    W = W_np.to(device).float()

    if H_raw.dim() == 2:
        H = H_raw.unsqueeze(0).to(device).float()
    else:
        H = H_raw.to(device).float()

    num_groups = H.shape[0]
    d = H.shape[1]
    diag_idx = torch.arange(d, device=device)

    for g in range(num_groups):
        avg_diag = torch.mean(torch.diag(H[g]))
        if avg_diag == 0:
            H[g] = torch.eye(d, device=device, dtype=torch.float32) * 1e-8
            print(f"    WARNING: H[{g}] all-zero diagonal, using identity")
            continue
        damp, prev_damp = 1e-5, 0.0
        while True:
            try:
                torch.linalg.cholesky(H[g])
                if prev_damp > 0:
                    print(f"    H[{g}] dampened with factor={prev_damp:.2e}")
                break
            except Exception:
                H[g, diag_idx, diag_idx] += (damp - prev_damp) * avg_diag
                prev_damp = damp
                damp *= 10
                if damp > 1e0:
                    print(
                        f"    WARNING: H[{g}] dampening failed, using diagonal approx"
                    )
                    H[g] = torch.diag(torch.diag(H[g]).clamp(min=1e-8))
                    break

    best_obj = objective_function(W, H, labels, C).item()
    best_labels = labels.clone()
    best_C = C.clone()
    print(f"    initial objective: {best_obj:.6f}")

    for iteration in range(num_iterations):
        t0 = time.time()

        if iteration > 0:
            labels = update_P(W, H, labels, C, cd_cycles=cd_cycles)

        obj_after_P = objective_function(W, H, labels, C).item()
        print(f"    iter {iteration+1} P-update obj: {obj_after_P:.4f}")

        C = update_C(W, H, labels, C)

        obj_after_C = objective_function(W, H, labels, C).item()

        if obj_after_C < best_obj:
            best_obj = obj_after_C
            best_labels = labels.clone()
            best_C = C.clone()
            print(
                f"    iter {iteration+1} C-update obj: {obj_after_C:.4f} (improved)"
            )
        else:
            print(
                f"    iter {iteration+1} C-update obj: {obj_after_C:.4f} (no improve, stopping)"
            )
            labels = best_labels
            C = best_C
            break

        print(f"    iter {iteration+1} took {time.time()-t0:.1f}s")

    return best_labels, best_C


# =====================================================================
# Model loading
# =====================================================================


def get_model(model_name):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    downloads_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")

    if "opt" in model_name.lower():
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=downloads_dir,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model_name.lower():
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=downloads_dir,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model.seqlen = 2048
    elif "qwen" in model_name.lower():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=downloads_dir,
            attn_implementation="eager",
        )
        model.seqlen = min(model.config.max_position_embeddings, 2048)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    return model


def detect_model_type(model):
    class_name = model.__class__.__name__.lower()
    if "opt" in class_name:
        return "opt"
    elif "llama" in class_name:
        return "llama"
    elif "qwen" in class_name:
        return "qwen"
    raise ValueError(f"Unknown model class: {model.__class__.__name__}")


# =====================================================================
# LNQ Quantization Pipeline
# =====================================================================


@torch.no_grad()
def lnq_quantize_model(model, args, weight_grads=None, saliency=None,
                        calib_dataset=None, calib_seqlen=None):
    """
    Layer-by-layer LNQ quantization pipeline.

    Supports three modes:
    1. Core-only (weight_grads=None): simple k-means init + standard Hessian
    2. Full pipeline, no saliency (weight_grads set, saliency=None):
       Fisher k-means init + standard Hessian
    3. Full pipeline with saliency (both set):
       Fisher k-means init + Fisher Hessian

    Args:
        model: HuggingFace model
        args: namespace with LNQ hyperparameters
        weight_grads: dict {(layer_idx, name): tensor} or None
        saliency: dict {(layer_idx, name): tensor} or None
        calib_dataset: calibration dataset name (default: args.dataset)
        calib_seqlen: calibration sequence length (default: model.seqlen)

    Returns:
        (model, bpw, quant_time): quantized model, effective bits per weight, time
    """
    dev = torch.device(args.device)
    nsamples = args.nsamples
    K = 2**args.nbits
    use_fisher_init = weight_grads is not None
    use_fisher_hessian = saliency is not None
    num_groups = args.num_groups if use_fisher_hessian else 1

    calib_dataset = calib_dataset or args.dataset
    calib_seqlen = calib_seqlen or model.seqlen

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    dataloader, _ = get_loaders(
        calib_dataset,
        nsamples=nsamples,
        seed=args.seed,
        seqlen=calib_seqlen,
        model=args.model,
    )

    model.config.use_cache = False

    model_type = detect_model_type(model)
    if model_type == "opt":
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif model_type in ("llama", "qwen"):
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        if hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    act_bytes = nsamples * calib_seqlen * model.config.hidden_size * 2
    act_device = 'cpu' if act_bytes > 8 * (1024**3) else dev
    if act_device == 'cpu':
        print(f"Activation offload: {act_bytes / 1024**3:.1f} GB > 8 GB threshold, using CPU")
    inps = torch.zeros(
        (nsamples, calib_seqlen, model.config.hidden_size), dtype=dtype, device=act_device
    )
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to(act_device)
            cache["i"] += 1
            cache["layer_kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if model_type == "opt":
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif model_type in ("llama", "qwen"):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["layer_kwargs"]

    if "past_key_values" in layer_kwargs:
        layer_kwargs["past_key_values"] = None

    total_quant_params = 0
    total_codebook_bits = 0
    total_index_bits = 0
    quant_start = time.time()

    mode_str = "core-only"
    if use_fisher_init and use_fisher_hessian:
        mode_str = f"full pipeline (num_groups={num_groups})"
    elif use_fisher_init:
        mode_str = "Fisher init + standard Hessian (nosal)"

    print(f"\nQuantizing {len(layers)} layers with LNQ (K={K}, {args.nbits}-bit)...")
    print(f"  mode: {mode_str}")
    print(f"  num_iterations={args.num_iterations}, cd_cycles={args.cd_cycles}")

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)

        print(f"\nLayer {layer_idx}/{len(layers)-1} -- {len(subset)} sublayers")

        # Accumulate Hessian for each sublayer via forward hooks
        hessians = {}

        if use_fisher_hessian:
            for name in subset:
                sal_key = (layer_idx, name)
                sal_tensor = saliency.get(sal_key)
                if sal_tensor is None:
                    print(
                        f"  WARNING: no saliency for {name}, falling back to standard H"
                    )
                    hessians[name] = _StandardHessianHook(subset[name], dev)
                else:
                    hessians[name] = SaliencyHessianHook(subset[name], sal_tensor, dev)
        else:
            for name in subset:
                hessians[name] = _StandardHessianHook(subset[name], dev)

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(hessians[name]))
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        # Extract all Hessians to CPU and free GPU buffers before quantization
        hessian_cpu = {}
        for name in subset:
            if use_fisher_hessian and isinstance(hessians[name], SaliencyHessianHook):
                hessian_cpu[name] = hessians[name].get_hessian()
                hessians[name].XTX = None
                hessians[name].saliency = None
            else:
                hessian_cpu[name] = hessians[name].H.clone().cpu()
                hessians[name].H = None
        del hessians
        if use_fisher_hessian and saliency is not None:
            for name in subset:
                sal_key = (layer_idx, name)
                if sal_key in saliency:
                    del saliency[sal_key]
        torch.cuda.empty_cache()
        gc.collect()

        # Quantize each sublayer with LNQ
        for name in subset:
            print(
                f"  {name} [{subset[name].weight.shape[0]}x{subset[name].weight.shape[1]}]"
            )

            W = subset[name].weight.data.clone().float().to(dev)
            H = hessian_cpu[name]

            # Dead column handling
            if H.dim() == 2:
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                W[:, dead] = 0
            else:
                dead_mask = torch.ones(H.shape[1], dtype=torch.bool)
                for g in range(H.shape[0]):
                    dead_mask = dead_mask & (torch.diag(H[g].cpu()) == 0)
                if dead_mask.any():
                    for g in range(H.shape[0]):
                        H[g, dead_mask, dead_mask] = 1
                    W[:, dead_mask.to(dev)] = 0

            output_dim, input_dim = W.shape

            total_quant_params += output_dim * input_dim
            total_codebook_bits += output_dim * K * 16
            total_index_bits += output_dim * input_dim * args.nbits

            # Initialize codebook
            t0 = time.time()
            if use_fisher_init:
                fg = weight_grads.get((layer_idx, name))
                if fg is not None:
                    labels, centroids = fisher_kmeans_init(W, fg, K=K)
                    print(f"    Fisher k-means init: {time.time()-t0:.1f}s")
                else:
                    labels, centroids = row_kmeans_1d(W, K=K, max_iter=args.kmeans_init_iters)
                    print(f"    k-means init (no Fisher grad): {time.time()-t0:.1f}s")
            else:
                labels, centroids = row_kmeans_1d(
                    W, K=K, max_iter=args.kmeans_init_iters
                )
                print(f"    k-means init: {time.time()-t0:.1f}s")

            # Run LNQ alternating optimization
            t0 = time.time()
            labels, centroids = train_least_squares(
                W,
                labels,
                centroids,
                H,
                num_iterations=args.num_iterations,
                cd_cycles=args.cd_cycles,
            )
            print(f"    LNQ optimize: {time.time()-t0:.1f}s")

            # Reconstruct quantized weights and write back
            labels_dev = labels.to(dev).long()
            C_dev = centroids.to(dev)
            W_hat = torch.gather(
                C_dev.unsqueeze(1).expand(-1, input_dim, -1),
                dim=2,
                index=labels_dev.unsqueeze(-1),
            ).squeeze(-1)
            subset[name].weight.data = W_hat.to(dtype)

            del W, H, labels, centroids, labels_dev, C_dev, W_hat
            torch.cuda.empty_cache()

        del hessian_cpu

        if not args.no_propagate:
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_kwargs)[0]

        layers[layer_idx] = layer.cpu()
        del layer
        for name in subset:
            if weight_grads is not None:
                weight_grads.pop((layer_idx, name), None)
            if saliency is not None:
                saliency.pop((layer_idx, name), None)
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    quant_time = time.time() - quant_start

    bpw = (total_index_bits + total_codebook_bits) / total_quant_params
    print(f"\nLNQ quantization complete in {quant_time:.1f}s")
    print(f"  Effective BPW: {bpw:.4f}")
    print(f"    index bits: {total_index_bits:,} ({args.nbits} per weight)")
    print(f"    codebook bits: {total_codebook_bits:,} ({K} entries x 16b per row)")
    print(f"    total params: {total_quant_params:,}")

    model.config.use_cache = True
    return model, bpw, quant_time


class _StandardHessianHook:
    """Standard Hessian hook: H = (2/N) X^T X. Returns (input_dim, input_dim)."""

    def __init__(self, layer_module, device):
        self.columns = layer_module.weight.shape[1]
        self.H = torch.zeros(
            (self.columns, self.columns), device=device, dtype=torch.float32
        )
        self.nsamples = 0

    def __call__(self, module, inp, out):
        x = inp[0].data
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        tmp = x.shape[0]
        if isinstance(module, nn.Linear):
            if len(x.shape) == 3:
                x = x.reshape((-1, x.shape[-1]))
            x = x.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        x = math.sqrt(2 / self.nsamples) * x.float()
        self.H += x.matmul(x.t())


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LNQ (ICML 2025) quantization benchmark"
    )
    parser.add_argument("model", type=str, help="HuggingFace model name")
    parser.add_argument(
        "dataset", type=str, choices=["wikitext2", "c4"], help="Evaluation dataset"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    # LNQ quantization parameters
    parser.add_argument(
        "--nbits", type=int, default=2, help="Bits per weight (2 for K=4 codebook)"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=3,
        help="Alternating optimization iterations (paper default: 3)",
    )
    parser.add_argument(
        "--cd_cycles",
        type=int,
        default=4,
        help="Coordinate descent cycles per update_P (paper default: 4)",
    )
    parser.add_argument(
        "--kmeans_init_iters",
        type=int,
        default=20,
        help="Lloyd iterations for simple k-means initialization",
    )

    # Calibration
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration sequences"
    )
    parser.add_argument(
        "--calib_dataset", type=str, default=None,
        choices=["wikitext2", "c4", "redpajama"],
        help="Calibration dataset (default: same as eval dataset)",
    )
    parser.add_argument(
        "--seqlen", type=int, default=None,
        help="Calibration sequence length (default: model.seqlen)",
    )
    parser.add_argument(
        "--eval_seqlen", type=int, default=None,
        help="Eval sequence length (default: model.seqlen). Paper uses 4096 for LLaMA-2.",
    )

    # Full GuidedQuant pipeline flags
    parser.add_argument(
        "--full_pipeline",
        action="store_true",
        help="Run full GuidedQuant pipeline (Fisher init + Fisher/standard Hessian)",
    )
    parser.add_argument(
        "--is_nosal",
        action="store_true",
        help="Use standard Hessian instead of Fisher-weighted (requires --full_pipeline)",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=4,
        help="Number of channel groups for saliency Hessian (default: 4)",
    )
    parser.add_argument(
        "--no_propagate",
        action="store_true",
        help="Don't propagate quantized outputs between layers (match GuidedQuant behavior)",
    )

    # Downstream eval — flags added by add_eval_cli (defines --full_eval,
    # --eval_extra_ppl, --ppl_eval_seqlen, --eval_arc, --eval_mmlu, --eval_hellaswag).
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
        ),
    )
    from csv_utils import append_result as csv_append
    from eval_utils import add_eval_cli, resolve_eval_flags, evaluate_and_log_all
    add_eval_cli(parser)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    model = get_model(args.model)
    model.config.use_cache = False

    if args.eval_seqlen is not None:
        model.seqlen = args.eval_seqlen

    calib_dataset = args.calib_dataset if args.calib_dataset else args.dataset
    calib_seqlen = args.seqlen if args.seqlen else model.seqlen

    print(f"  Eval dataset: {args.dataset} (seqlen={model.seqlen})")
    print(f"  Calibration dataset: {calib_dataset} (seqlen={calib_seqlen}, nsamples={args.nsamples})")

    # Determine method name for CSV
    if args.full_pipeline:
        if args.is_nosal:
            method = "guidedquant_nosal"
        else:
            method = "guidedquant"
    else:
        method = "lnq"

    # Phase 0: Compute Fisher data if full pipeline
    weight_grads, saliency_data = None, None
    if args.full_pipeline:
        if not HAS_NUMBA:
            print("ERROR: --full_pipeline requires numba and flash1dkmeans.")
            print("Install: pip install numba==0.60.0 flash1dkmeans==0.1.4")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Phase 0: Computing Fisher data for full GuidedQuant pipeline")
        print(f"{'='*60}")

        model_type = detect_model_type(model)

        dataloader, _ = get_loaders(
            calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=calib_seqlen,
            model=args.model,
        )

        weight_grads, saliency_data = compute_fisher_data(
            model,
            dataloader,
            torch.device(args.device),
            model_type,
            num_groups=args.num_groups,
            compute_sal=not args.is_nosal,
        )

        total_grad_norm = sum(v.norm().item() for v in weight_grads.values())
        print(f"  Total Fisher gradient norm: {total_grad_norm:.2f}")
        if saliency_data is not None:
            n_sal = sum(1 for v in saliency_data.values() if v is not None)
            print(f"  Saliency tensors computed: {n_sal}")

    # Quantize
    model, bpw, quant_time = lnq_quantize_model(
        model, args, weight_grads=weight_grads, saliency=saliency_data,
        calib_dataset=calib_dataset, calib_seqlen=calib_seqlen,
    )

    extra = {
        "nbits": args.nbits,
        "num_iterations": args.num_iterations,
        "cd_cycles": args.cd_cycles,
        "kmeans_init_iters": args.kmeans_init_iters,
        "nsamples": args.nsamples,
        "calib_dataset": calib_dataset,
        "calib_seqlen": calib_seqlen,
        "full_pipeline": args.full_pipeline,
        "is_nosal": args.is_nosal,
        "num_groups": args.num_groups,
        "no_propagate": args.no_propagate,
    }

    eval_flags = resolve_eval_flags(args, primary_dataset=args.dataset)

    model_short = args.model.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"RESULT: {method} {args.nbits}-bit on {model_short}")
    print(f"  Seed: {args.seed}")
    print(f"  Effective bpw: {bpw:.4f}")
    print(f"  Quantization time: {quant_time:.1f}s")
    print(f"  Calibration: {calib_dataset} (nsamples={args.nsamples}, seqlen={calib_seqlen})")
    print(f"  Propagation: {'off' if args.no_propagate else 'on'}")
    print(f"  PPL eval datasets: {eval_flags['ppl_datasets']}")
    print(f"{'='*60}")

    evaluate_and_log_all(
        model, args.model, torch.device(args.device),
        method=method,
        bpw=bpw, seed=args.seed, blocksize="",
        salient_metric="",
        extra_params=extra,
        quantization_time_s=quant_time,
        ppl_datasets=eval_flags["ppl_datasets"],
        eval_mmlu=eval_flags["eval_mmlu"],
        eval_hellaswag=eval_flags["eval_hellaswag"],
        eval_arc=eval_flags["eval_arc"],
        ppl_eval_seqlen=eval_flags["ppl_eval_seqlen"],
        save_title_prefix=f"lnq_{args.nbits}bit_{model_short}_seed{args.seed}",
    )


if __name__ == "__main__":
    main()
