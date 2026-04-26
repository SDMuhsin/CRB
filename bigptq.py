import math
import time
from exceptiongroup import catch
import torch
import torch.nn as nn
import transformers
from utils.structure import structural_guassian_distribution

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

'''
BRAGPTQ is the meaning of GPTQ used Binary Residual Approximation in paper to realize 1-bit quantization
BRAGPTQ uses structural mask to distinguish outliers and other data, and takes advantage of part of GPTQ to lower error
'''
class BRAGPTQ:
    def __init__(
        self, layer, braq_quantizer,salient_metric, disable_gptq=False
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # breakpoint()

    def fasterquant(self,
                    blocksize=128,
                    percdamp=0.01,
                    partition=3,
                    orders=(1,1,2),
                    global_scale=False,
                    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        # Paper-faithful GPTQ (Frantar et al. 2023, Table 3 no-groupsize):
        # per-row min/max computed ONCE on the full (pre-feedback) weight
        # matrix, then held constant across all GPTQ blocks. Forwarded into
        # Binarization via the `global_scale`/`global_zero` attributes.
        # Only activated when `global_scale=True` and the underlying
        # quantizer supports it (currently the '2bit'/'4bit' path in
        # binary.py).
        if global_scale and getattr(self.braq_quantizer, 'method', None) in ('2bit', '3bit', '4bit'):
            bits = int(self.braq_quantizer.method[0])
            dev = W.device
            maxq_g = torch.tensor(2 ** bits - 1, device=dev)
            xg = W.flatten(1)
            tmp0 = torch.zeros(xg.shape[0], device=dev)
            xmin_g = torch.minimum(xg.min(1)[0], tmp0)
            xmax_g = torch.maximum(xg.max(1)[0], tmp0)
            degen = (xmin_g == 0) & (xmax_g == 0)
            xmin_g[degen] = -1
            xmax_g[degen] = +1
            scale_g = (xmax_g - xmin_g) / maxq_g
            zero_g = torch.round(-xmin_g / scale_g)
            shape_bc = [-1] + [1] * (W.dim() - 1)
            self.braq_quantizer.global_scale = scale_g.reshape(shape_bc)
            self.braq_quantizer.global_zero = zero_g.reshape(shape_bc)
        else:
            # Ensure stale attributes from a previous call are cleared.
            if hasattr(self.braq_quantizer, 'global_scale'):
                self.braq_quantizer.global_scale = None
            if hasattr(self.braq_quantizer, 'global_zero'):
                self.braq_quantizer.global_zero = None

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # Save raw Hessian diagonal for adaptive binarization methods
        H_diag_raw = torch.diag(H).clone()
        # Robust Cholesky: increase damping if needed
        for _retry in range(10):
            try:
                H_chol = torch.linalg.cholesky(H)
                break
            except torch._C._LinAlgError:
                extra_damp = 1e-3 * torch.mean(torch.diag(H))
                if extra_damp == 0:
                    extra_damp = 1e-6
                H[diag, diag] += extra_damp
        else:
            # Last resort: use diagonal only
            H_chol = torch.diag(torch.sqrt(torch.diag(H).clamp(min=1e-8)))
        H = torch.cholesky_inverse(H_chol)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(partition, dim=0)
            if partition == 1:
                # No structural partition — single all-ones mask (for DOML, etc.)
                mask[0] = torch.ones_like(W[:, st:ed], dtype=torch.bool)
            else:
                mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50, orders=orders)
                mask[0] = mask1
                mask[1] = mask2
                mask[2] = mask3

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]

                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i], col_weights=None))

                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                # Dispatch between two quantization modes:
                #   (A) partition == 1 AND quantizer is a simple integer method
                #       (2bit/3bit/4bit): paper-faithful GPTQ — column-by-column
                #       quantize + intra-block error feedback. Scales are either
                #       the pre-computed `global_scale` (paper Table 3 no-groupsize)
                #       or per-row-per-block min/max (paper Table 7 gs=blocksize),
                #       computed ONCE before the column sweep and held fixed.
                #   (B) partition > 1 OR non-integer quantizer (DOML/BRAQ/CRB/…):
                #       legacy pre-quantize-then-read path. DOML's structural
                #       partition is defined on the original block; per-row
                #       Lloyd-Max levels cannot be computed per-column, so we
                #       keep the original behaviour here. Accepted as a
                #       DOML-family design choice; see llmdocs/trackers/
                #       baseline_faithfulness_audit.md Phase 14B.
                is_int_quant = getattr(self.braq_quantizer, 'method', None) in (
                    '2bit', '3bit', '4bit'
                )

                if partition == 1 and is_int_quant:
                    # Paper-faithful GPTQ column sweep.
                    bits = int(self.braq_quantizer.method[0])
                    dev = W1.device
                    maxq = torch.tensor(2 ** bits - 1, device=dev, dtype=W1.dtype)

                    gs = getattr(self.braq_quantizer, 'global_scale', None)
                    gz = getattr(self.braq_quantizer, 'global_zero', None)
                    if gs is not None and gz is not None:
                        # Paper Table 3: per-row scale pre-computed on full W.
                        scale = gs.to(dev).to(W1.dtype).view(-1)   # (oc,)
                        zero = gz.to(dev).to(W1.dtype).view(-1)    # (oc,)
                    else:
                        # Paper Table 7-style: per-row scale from the original
                        # (pre-feedback) block W1. Equivalent to groupsize=blocksize.
                        zero_ref = torch.zeros(W1.shape[0], device=dev, dtype=W1.dtype)
                        row_min = torch.minimum(W1.min(dim=1)[0], zero_ref)
                        row_max = torch.maximum(W1.max(dim=1)[0], zero_ref)
                        degen = (row_min == 0) & (row_max == 0)
                        row_min = torch.where(degen, torch.full_like(row_min, -1.0), row_min)
                        row_max = torch.where(degen, torch.full_like(row_max, 1.0), row_max)
                        scale = (row_max - row_min) / maxq                 # (oc,)
                        zero = torch.round(-row_min / scale)               # (oc,)

                    qmin = torch.tensor(0.0, device=dev, dtype=W1.dtype)
                    for i in range(n_cols):
                        w = W1[:, i]                                       # (oc,)
                        d = Hinv1[i, i]

                        q_int = torch.clamp(torch.round(w / scale) + zero, qmin, maxq)
                        q = (q_int - zero) * scale

                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / (d * d)
                        err1 = (w - q) / d
                        Err1[:, i] = err1

                        # Intra-block GPTQ feedback: update remaining columns
                        # in this block using the current column's error.
                        if i + 1 < n_cols:
                            W1[:, i + 1:] -= err1.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)

                    W[:, col_st:col_ed] = Q1
                    Losses += torch.sum(Losses1, 1) / 2
                    # Inter-block GPTQ feedback.
                    W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
                else:
                    # Legacy pre-quantize-then-read path for DOML / BRAQ / CRB /
                    # other partitioned or non-integer quantizers.
                    q_part_groups = []

                    # Column importance weights from Hessian inverse diagonal
                    # for adaptive methods that need them. 1/Hinv[j,j]^2 is the
                    # GPTQ per-column loss weight (higher = more important).
                    hinv_diag = torch.diag(Hinv1)
                    col_weights = 1.0 / (hinv_diag ** 2 + 1e-12)

                    for i in range(mask.shape[0]):
                        q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=orders[i], col_weights=col_weights))

                    for i in range(n_cols):
                        w = W1[:, i]
                        d = Hinv1[i, i]

                        q = torch.zeros_like(w)
                        for j in range(mask.shape[0]):
                            q += q_part_groups[j][:, i] * mask[j, :, i]

                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / d**2
                        err1 = (w - q) / d
                        Err1[:, i] = err1

                    W[:, col_st:col_ed] = Q1
                    Losses += torch.sum(Losses1, 1) / 2
                    W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        if partition > 1:
            del mask1, mask2, mask3
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()


class COMMENTED_BRAGPTQ:
    def __init__(self, layer, braq_quantizer, salient_metric, disable_gptq=False):
        """
        Wraps a layer (e.g., nn.Linear or Conv1D) to gather statistics, 
        compute Hessian approximations, and apply GPTQ-based binarization/quantization
        with 'residual binarization' (braq) or other methods.

        Args:
            layer: The PyTorch layer whose weights we want to quantize.
            braq_quantizer: A Binarization or quantization object (e.g., braq/cabr) 
                           that implements the per-group 1-bit or 2-bit expansions.
            salient_metric: String, e.g. "magnitude" or "hessian", 
                            indicating how to determine salience of weights.
            disable_gptq: If True, GPTQ and advanced error correction are skipped 
                          in favor of simple rounding/binarization (e.g., RTN).
        """
        self.layer = layer
        self.dev = self.layer.weight.device

        # Clone the layer's weights in a standard 2D shape for analysis:
        #   - For nn.Conv2d, flatten spatial dims into columns.
        #   - For transformers.Conv1D, transpose so shape is (out_features, in_features).
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]     # Number of output channels / output dim
        self.columns = W.shape[1] # Number of input channels / input dim

        # Hessian matrix used for salience-based partitioning / error correction 
        # (approx. second-order statistics). H is columns x columns.
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        # The quantizer that performs binarization or multi-bit approximation 
        # (e.g. braq for residual binarization).
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric

        # If True, skip GPTQ error feedback and do simple quantization (RTN).
        self.disable_gptq = disable_gptq

    def add_batch(self, inp, out, blocksize=1024):
        """
        Accumulates second-order statistics (Hessian approximation) 
        from a mini-batch of layer inputs (inp) and outputs (out).

        The standard GPTQ approach uses:
            H = 2 * X X^T, 
        or some variant, to approximate the Hessian or salience metrics. 
        We store partial sums in self.H.

        Args:
            inp: The input activation to this layer of shape [batch_size, in_features]
                 or [batch_size, seq_len, in_features].
            out: The output activation (unused directly in this snippet, 
                 but can be stored for advanced solutions).
            blocksize: (Unused here) a chunking parameter for potentially large data.
        """
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        # If necessary, reshape input to 3D => 2D, or handle (batch_size, in_features) vs. (1, ...).
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        # For nn.Linear or Conv1D, we flatten the sequences (if any) 
        # and transpose to (in_features, total_batch_points).
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # Weighted update of the Hessian approximation H, so the new batch is 
        # blended with previous batches proportionally to sample size.
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        # Multiply input by sqrt(2 / nsamples) => standard GPTQ Hessian approximation approach 
        # (some variants scale differently, but the idea is to keep track of covariances).
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        # Update the Hessian approximation: H += (inp)(inp^T).
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, partition=3, orders=(1,1,2)):
        """
        Main GPTQ-based quantization routine:
          1) Use the Hessian matrix self.H for salience-based partitioning 
             (structural_gaussian_distribution).
          2) Apply binarization/quantization in blocks of 'blocksize' columns.
          3) If GPTQ is enabled (disable_gptq=False), apply error feedback 
             (model the post-quantization error, update future columns accordingly).
          4) If disable_gptq=True, do a simpler method (e.g. RTN).

        Args:
            blocksize: Number of columns to process at a time. 
                       Helps with memory efficiency.
            percdamp:  Proportion of damping added to the Hessian diagonal 
                       to improve stability in the Cholesky decomposition.
            partition: The number of partitions (e.g., 3) for masks, 
                       indicating different sets of weights (salient vs. non-salient splits).
            orders:    Tuple specifying the binarization "order" passes for each partition.
                       e.g. (1,1,2) might mean partition0 is 1-pass binarization,
                                  partition1 is 1-pass binarization,
                                  partition2 is 2-pass (residual) binarization.

        Returns:
            A dict containing error stats (e.g. "error": total quantization error).
        """
        # -- 1) Copy weights in a 2D float representation
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        # Use the previously accumulated Hessian H
        H = self.H
        del self.H  # We'll no longer need 'self.H' after we invert it.

        # Fix any zero-diagonal entries to avoid singularities
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        # Set weight columns to zero if the corresponding Hessian diag was zero
        W[:, dead] = 0

        # Track error per row
        Losses = torch.zeros(self.rows, device=self.dev)

        # -- 2) Damping: we add percdamp * mean(diag(H)) to each diagonal to stabilize invertibility
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        # -- 3) Perform a Cholesky-based inverse:
        #    1) cholesky(H) => lower triangular (or upper).
        #    2) cholesky_inverse => get H^{-1}.
        #    3) another cholesky => get an upper triangular that is effectively sqrt(H^{-1}).
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H  # We'll use this triangular form for GPTQ error feedback.

        # -- 4) Process the weight matrix in column blocks
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            # Create partition masks for (salient vs. non-salient) or 
            # (sparse, concentrated, etc.) from the theoretical approach.
            # 'structural_guassian_distribution' presumably returns 3 masks 
            # based on distribution shape or Hessian-based salience:
            #   mask1 = salient region
            #   mask2 = non-salient sparse
            #   mask3 = non-salient concentrated
            mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0)
            mask = mask.repeat_interleave(partition, dim=0)

            mask1, mask2, mask3 = structural_guassian_distribution(
                W[:, st:ed], 
                H[st:ed, st:ed], 
                self.salient_metric, 
                50
            )
            mask[0] = mask1
            mask[1] = mask2
            mask[2] = mask3

            # The quantizer's group size must be a multiple of blocksize 
            # (or vice versa) for consistent grouping.
            assert self.braq_quantizer.groupsize % blocksize == 0

            # -- 5) Two cases: 
            #    A) disable_gptq = True => skip advanced GPTQ error feedback
            #    B) disable_gptq = False => use GPTQ error correction 
            #       to refine the final weights and reduce residual.

            if self.disable_gptq:
                # RTN or simple binarization for each partition
                w = W[:, col_st:col_ed]

                # Each partition i gets quantized with order[i] 
                # (some might be single pass, some might be 2-pass residual).
                q_part_groups = []
                for i in range(mask.shape[0]):
                    # braq_quantizer.quantize => typically calls 'high_order_residual' 
                    # or 'coupled_residual_binarization' internally, depending on 'method'.
                    q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i]))

                # Combine each partition's quantized result into a single block
                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]

                # Write back to W
                W[:, col_st:col_ed] = q

            else:
                # -- 5.B) GPTQ with error feedback
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)

                # Local portion of the triangular Hinv 
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                # Quantize each partition in the block with the appropriate pass order
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(
                        W1, mask[i], order=orders[i]
                    ))

                # GPTQ Correction: for each column i, compute the error 
                # and update subsequent columns accordingly.
                for i in range(n_cols):
                    w = W1[:, i]  # Original weights in column i
                    d = Hinv1[i, i]

                    # Summation of the partition expansions for this column
                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]

                    Q1[:, i] = q

                    # Compute local loss => (w - q)^2 / d^2
                    Losses1[:, i] = (w - q) ** 2 / d**2

                    # Error feedback: err1 = (w - q) / d
                    # Subtract this error from the next columns using the triangular Hinv
                    err1 = (w - q) / d
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                # Propagate error to future columns:
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                # DEBUG block can track the running error if needed

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        # -- 6) Reshape and store the final quantized weights back into self.layer
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        # Cleanup to free memory
        del mask, mask1, mask2, mask3
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()

        return {"error": torch.sum(Losses).item()}

    def free(self):
        """
        Frees up cached activations or Hessian if needed for memory efficiency.
        """
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

