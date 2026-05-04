from numpy import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


index = 0
@torch.no_grad()
def part_mean(tensor, op='-'):
    non_zero = tensor*(tensor!=0)

    mean_val = non_zero.mean(-1).view(-1, 1)

    return mean_val

@torch.no_grad()
def high_order_residual(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() # Keep a copy of the original weight matrix x
    new_matrix = new_matrix * mask # Pick out only salient columsn
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'))) # Use only valid positions of residual (invalids are marked with nan)

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1) #  Row wise mean of residual
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None] # Subtracts mean from each row (only valid rows) : Centers all elements at 0
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1) # Gets averagei absolute value, row wise = estimate of alpha
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        
        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None] # Rescale (alpha * B)
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask # Add X = ... + Bk * alpha_k
    
    return sum_order

@torch.no_grad()
def ternary_residual(x, mask, order=2):
    """Ternary residual approximation: {-alpha, 0, +alpha} per row.

    Like high_order_residual but uses ternary ({-1, 0, +1}) instead of binary ({-1, +1}).
    The zero level naturally handles near-zero weights, giving better approximation
    for weight distributions with significant mass near zero.

    Each pass: center, compute threshold, assign {-1, 0, +1}, scale by alpha.
    Threshold = 0.5 * mean(|centered|), values below → 0.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    for od in range(order):
        residual = new_matrix - sum_order
        masked_x = torch.where(mask, residual, torch.tensor(float('nan')))

        # Per-row mean (centering)
        row_mean = torch.nanmean(masked_x, dim=1)
        row_mean = torch.where(torch.isnan(row_mean), torch.zeros_like(row_mean), row_mean)
        centered = masked_x - row_mean[:, None]

        # Per-row scale
        alpha = torch.nanmean(torch.abs(centered), dim=1)
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)

        # Ternary: threshold at 0.5 * alpha, values below → 0
        threshold = 0.5 * alpha[:, None]
        # Use nan-safe comparisons: nan > x = False, nan < x = False
        # So nan positions correctly stay at 0
        ternary = torch.zeros_like(centered)
        ternary[centered > threshold] = 1.0
        ternary[centered < -threshold] = -1.0

        # Recompute alpha as mean of |values| where ternary != 0 (better scale)
        # Replace NaN with 0 before computing abs to avoid NaN propagation
        centered_safe = torch.where(torch.isnan(centered), torch.zeros_like(centered), centered)
        abs_vals = torch.abs(centered_safe) * (ternary != 0).float()
        n_nonzero = (ternary != 0).float().sum(dim=1, keepdim=True).clamp(min=1)
        alpha_refined = abs_vals.sum(dim=1, keepdim=True) / n_nonzero

        ternary = ternary * alpha_refined + row_mean[:, None]
        sum_order = sum_order + ternary * mask

    return sum_order


@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

@torch.no_grad()
def robust_high_order_residual(x, mask, order=2, clamp_factor=2.5):
    """
    A robust variant of residual binarization that clamps outliers at each iteration.
    x: the weight matrix (oc x ic)
    mask: boolean tensor indicating where to apply binarization
    order: number of residual binarization steps
    clamp_factor: multiple of std-dev for clamping outliers
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    for _ in range(order):
        # Compute the residual for this iteration
        residual = new_matrix - sum_order

        # Only consider masked elements
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=residual.device))

        # Compute row-wise mean (ignoring NaNs)
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)

        # Center the residual around the mean
        centered_x = masked_x_tensor - mean_tensor_all[:, None]

        # Handle NaNs explicitly for standard deviation
        # Count valid (non-NaN) elements
        valid_counts = torch.sum(~torch.isnan(centered_x), dim=1).float()
        valid_counts = torch.where(valid_counts > 0, valid_counts, torch.tensor(1.0, device=residual.device))  # Avoid divide-by-zero

        # Compute squared deviations
        squared_deviations = torch.where(
            ~torch.isnan(centered_x), centered_x**2, torch.zeros_like(centered_x)
        )
        variance = torch.sum(squared_deviations, dim=1) / valid_counts
        std_tensor_all = torch.sqrt(variance)

        # Clamp outliers: anything beyond ±(clamp_factor * std)
        clamped_x_tensor = torch.clamp(
            centered_x,
            min=-clamp_factor * std_tensor_all[:, None],
            max=clamp_factor * std_tensor_all[:, None],
        )

        # Compute scale as mean(abs(.)) after clamping
        scale_tensor_all = torch.nanmean(torch.abs(clamped_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        # Binarize + scale + shift
        binary = torch.sign(clamped_x_tensor) * scale_tensor_all[:, None]
        binary = binary + mean_tensor_all[:, None]

        # Accumulate into sum_order
        sum_order += binary * mask

    return sum_order


@torch.no_grad()
def mest_robust_residual_binarization(x, mask, order=2, kappa=1.0):
    """
    Robust residual binarization with M-estimation style outlier handling.

    Parameters:
    - x (torch.Tensor): The weight matrix (rows = out_channels, cols = in_channels).
    - mask (torch.Tensor): Boolean mask indicating which entries to binarize.
    - order (int): Number of residual expansions.
    - kappa (float): Robustness parameter controlling outlier down-weighting.

    Returns:
    - torch.Tensor: The binarized approximation of `x`.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    for od in range(order):
        # Residual at the current iteration
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        # Compute robust weights: f(r) = 1 / (1 + (r / kappa)^2)
        weight_all = 1.0 / (1.0 + (masked_x_tensor / kappa) ** 2)
        weight_all = torch.where(torch.isnan(weight_all), torch.zeros_like(weight_all), weight_all)

        # Weighted mean computation
        valid_tensor = torch.logical_not(torch.isnan(masked_x_tensor))
        wsum = torch.nansum(weight_all * valid_tensor, dim=1, keepdim=True) + 1e-8
        weighted_vals = torch.where(valid_tensor, masked_x_tensor * weight_all, torch.zeros_like(masked_x_tensor))
        mean_tensor_all = torch.nansum(weighted_vals, dim=1, keepdim=True) / wsum

        # Subtract the robust mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all

        # Compute robust scale
        abs_diff = torch.abs(masked_x_tensor) * weight_all
        scale_tensor_all = torch.nansum(abs_diff, dim=1, keepdim=True) / wsum

        # Replace NaNs with zeros
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        # Binarization step: sign(r') * scale + mean
        binary = torch.sign(masked_x_tensor)
        binary = binary * scale_tensor_all
        binary = binary + mean_tensor_all

        # Accumulate the result
        sum_order = sum_order + torch.where(mask, binary, torch.zeros_like(binary))

    return sum_order

@torch.no_grad()
def median_high_order_residual(x, mask, order=2):
    """
    Proposed robust residual binarization (medianbraq).
    Uses median-based offset and scale (median absolute deviation).
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # Median-based offset
        # nanmedian returns (values, indices), so we take the .values.
        median_tensor_all = torch.nanmedian(masked_x_tensor, dim=1).values
        median_tensor_all = torch.where(torch.isnan(median_tensor_all),
                                        torch.zeros_like(median_tensor_all),
                                        median_tensor_all)

        # Subtract offset
        masked_x_tensor -= median_tensor_all[:, None]

        # Scale = median of absolute values (robust to outliers)
        abs_masked = torch.abs(masked_x_tensor)
        scale_tensor_all = torch.nanmedian(abs_masked, dim=1).values
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all),
                                       torch.zeros_like(scale_tensor_all),
                                       scale_tensor_all)

        # Binarize
        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += median_tensor_all[:, None]
        sum_order = sum_order + binary*mask
    
    return sum_order

@torch.no_grad()
def orthogonal_residual(x, mask, order=2):
    """
    Orthogonal Residual Binarization (ORB)

    This patched version handles fully masked-out rows
    by replicating 'high_order_residual' logic—any row
    with no unmasked elements becomes zero instead of NaN.
    """

    sum_order = torch.zeros_like(x)
    expansions = []
    
    for od in range(order):
        # Residual to approximate
        residual = x - sum_order

        # Mark unmasked elements; others = NaN
        masked_residual = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))
        
        # Row-wise mean (ignoring NaNs)
        mean_val = torch.nanmean(masked_residual, dim=1, keepdim=True)
        # If the entire row is NaN => force that mean to 0
        mean_val = torch.where(torch.isnan(mean_val),
                               torch.zeros_like(mean_val),
                               mean_val)
        
        # Center the residual around the mean
        centered = masked_residual - mean_val
        
        # Convert all masked-out elements to 0 (not NaN)
        centered = torch.where(mask, centered, torch.zeros_like(centered))

        # Orthogonal projection against previous expansions
        for exp in expansions:
            dot_num = (centered * exp).mean(dim=1, keepdim=True)
            dot_den = (exp * exp).mean(dim=1, keepdim=True) + 1e-12
            proj = dot_num * exp / dot_den
            centered = centered - proj
        
        # Row-wise scaling
        scale_val = torch.nanmean(torch.abs(centered), dim=1, keepdim=True)
        # If row is all zero => NaN => set to 0
        scale_val = torch.where(torch.isnan(scale_val),
                                torch.zeros_like(scale_val),
                                scale_val)
        
        # Sign + scale + shift
        binary = torch.sign(centered) * scale_val + mean_val
        
        # Update expansions & sum
        expansions.append(binary)
        sum_order = sum_order + binary * mask

    return sum_order

@torch.no_grad()
def weighted_high_order_residual(x, mask, order=2):
    """
    Weighted Residual Binarization (WHOR):
    Iteratively approximates 'x' with a sum of binary expansions, 
    weighting errors by their magnitude so that large residuals 
    get reduced more aggressively.
    
    The final bit cost is the same as standard residual binarization:
    we do 'order' expansions, each storing one mean + one scale + sign-bits.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask  # only operate within the valid region

    for od in range(order):
        # 1) Compute residual
        residual = new_matrix - sum_order

        # 2) Mask out invalid positions
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        # 3) Define weights = abs(residual). 
        #    This emphasizes elements with large residuals.
        w = torch.abs(masked_x_tensor)

        # 4) Weighted mean calculation: 
        #    mean = (sum_i w_i * r_i) / (sum_i w_i)
        numerator = torch.nansum(w * masked_x_tensor, dim=1, keepdim=True)
        denominator = torch.nansum(w, dim=1, keepdim=True) + 1e-8
        mean_tensor_all = numerator / denominator

        # 5) Subtract the mean from residual
        masked_x_tensor = masked_x_tensor - mean_tensor_all

        # 6) Weighted scale:
        #    scale = (sum_i w_i * |r_i - mean|) / (sum_i w_i)
        scale_numerator = torch.nansum(w * torch.abs(masked_x_tensor), dim=1, keepdim=True)
        scale_tensor_all = scale_numerator / denominator

        # 7) Form the binary expansion: sign + scale + mean
        binary = torch.sign(masked_x_tensor) * scale_tensor_all
        binary = binary + mean_tensor_all

        # 8) Accumulate into sum_order
        sum_order = sum_order + binary * mask

    return sum_order

@torch.no_grad()
def attenuated_residual(x, mask, order=2, gamma=0.5):
    """
    Attenuated Residual Binarization (ARB)
    - Similar to `high_order_residual` (braq)
    - Each iteration's binary correction is damped by a factor gamma.
    - Retains 1-bit expansions and same memory overhead as braq.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    for od in range(order):
        residual = new_matrix - sum_order
        # Mask out the elements
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        # Compute row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)

        # Center
        masked_x_tensor -= mean_tensor_all[:, None]

        # Compute row-wise average absolute deviation for scale
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        # Sign + scale + shift
        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]

        # Instead of subtracting full `binary`, we only subtract gamma * binary
        # and accumulate the partial correction
        sum_order = sum_order + gamma * binary * mask

    return sum_order
@torch.no_grad()
def balanced_high_order_residual(x, mask, order=2):
    """
    Balanced Residual Binarization in multiple passes.
    Enforces ~0 net sum in each pass by balancing +1/−1.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask  # only keep valid weights

    for od in range(order):
        residual = new_matrix - sum_order
        # masked view: ignore positions where mask=0
        masked_residual = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        # Compute row-wise scale (same as braq's L2-optimal alpha)
        alpha = torch.nanmean(torch.abs(masked_residual), dim=1)
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)  # safe fallback

        # Balanced sign step per row
        B = torch.zeros_like(masked_residual)

        for i in range(B.shape[0]):
            row = masked_residual[i]
            valid_mask = ~torch.isnan(row)
            valid_vals = row[valid_mask]

            if valid_vals.numel() == 0:
                continue

            sorted_vals, sorted_idx = torch.sort(valid_vals)
            n_valid = len(sorted_vals)
            half = n_valid // 2  # integer division

            # Ensure `torch.arange()` is created on the same device as `row`
            row_indices = torch.arange(len(row), device=row.device)[valid_mask][sorted_idx]

            B[i, row_indices[:half]] = -1
            B[i, row_indices[half:]] = 1

        # Convert B from masked_residual shape to a real tensor (NaN->0 for masked positions)
        B = torch.where(mask, B, torch.zeros_like(B))

        # Weighted update = alpha_i * B
        alpha = alpha.view(-1, 1)
        sum_order = sum_order + alpha * B

    return sum_order
@torch.no_grad()
def joint_residual_binarization(x, mask, iters=3):
    """
    Jointly refines two binary expansions (B1, B2) with scales (alpha1, alpha2)
    by coordinate-descent style updates, seeking to minimize || x - alpha1 B1 - alpha2 B2 ||^2
    without adding new storage. Returns final sum_order = alpha1*B1 + alpha2*B2.
    """
    # 1) Initialize with standard first-pass binarization
    x_local = x.clone() * mask
    # B1, alpha1
    mean1 = torch.nanmean(torch.where(mask, x_local, torch.tensor(float('nan'))), dim=1)
    mean1 = torch.where(torch.isnan(mean1), torch.zeros_like(mean1), mean1)
    x_shifted = x_local - mean1[:, None]
    alpha1 = torch.nanmean(torch.abs(x_shifted), dim=1)
    alpha1 = torch.where(torch.isnan(alpha1), torch.zeros_like(alpha1), alpha1)
    B1 = torch.sign(x_shifted)

    # 2) Initialize B2, alpha2 from the residual
    R = x_local - (B1 * alpha1[:, None])
    mean2 = torch.nanmean(torch.where(mask, R, torch.tensor(float('nan'))), dim=1)
    mean2 = torch.where(torch.isnan(mean2), torch.zeros_like(mean2), mean2)
    R_shifted = R - mean2[:, None]
    alpha2 = torch.nanmean(torch.abs(R_shifted), dim=1)
    alpha2 = torch.where(torch.isnan(alpha2), torch.zeros_like(alpha2), alpha2)
    B2 = torch.sign(R_shifted)

    # 3) Iterative refinement
    #    Re-fit B1 once B2 is known, then re-fit B2, etc.
    for _ in range(iters):
        # Recompute residual ignoring B2
        R1 = x_local - (B2 * alpha2[:, None])
        # Fit B1, alpha1 again
        mean1 = torch.nanmean(torch.where(mask, R1, torch.tensor(float('nan'))), dim=1)
        mean1 = torch.where(torch.isnan(mean1), torch.zeros_like(mean1), mean1)
        R1_shifted = R1 - mean1[:, None]
        alpha1 = torch.nanmean(torch.abs(R1_shifted), dim=1)
        alpha1 = torch.where(torch.isnan(alpha1), torch.zeros_like(alpha1), alpha1)
        B1 = torch.sign(R1_shifted)

        # Now re-fit B2 from the new B1
        R2 = x_local - (B1 * alpha1[:, None])
        mean2 = torch.nanmean(torch.where(mask, R2, torch.tensor(float('nan'))), dim=1)
        mean2 = torch.where(torch.isnan(mean2), torch.zeros_like(mean2), mean2)
        R2_shifted = R2 - mean2[:, None]
        alpha2 = torch.nanmean(torch.abs(R2_shifted), dim=1)
        alpha2 = torch.where(torch.isnan(alpha2), torch.zeros_like(alpha2), alpha2)
        B2 = torch.sign(R2_shifted)

    # Final combination
    sum_order = (B1 * alpha1[:, None] + mean1[:, None]) \
              + (B2 * alpha2[:, None] + mean2[:, None])
    sum_order = sum_order * mask
    return sum_order

@torch.no_grad()
def D_coupled_residual_binarization(x, mask, order=2):

    """
    Performs a two-binary-expansion approximation (like braq) but
    co-optimizes the scale factors alpha_1, alpha_2 in closed form.

    x:     (oc, ic) weight matrix
    mask:  boolean mask with same shape as x
    order: number of expansions (we only implement 2 expansions here)

    Returns: sum_order -> final approximate binarized matrix
    """
    # We will do this row by row. 
    # For each row, we:
    #   1) Subtract mean.
    #   2) Get B1, alpha_1 from sign/average of magnitude.
    #   3) Form residual, get B2, alpha_2 similarly.
    #   4) Solve for alpha_1, alpha_2 simultaneously in closed form.
    #   5) Re-add the mean.
    
    # Make a clone of x that we will modify
    new_matrix = x.clone()
    new_matrix = new_matrix * mask  # only consider valid entries

    # We'll accumulate the final approximation in sum_order
    sum_order = torch.zeros_like(new_matrix)

    oc, ic = new_matrix.shape

    # Row-wise processing
    for row_idx in range(oc):
        # Extract row and mask
        row = new_matrix[row_idx, :]
        row_mask = mask[row_idx, :]
        
        # If nothing is masked-in, skip
        if not torch.any(row_mask):
            continue
        
        # Grab just the valid elements for the masked row
        row_vals = row[row_mask]
        
        # 1) Subtract mean
        row_mean = row_vals.mean()
        centered = row_vals - row_mean
        
        # 2) First pass: B1, alpha_1
        B1 = torch.sign(centered)
        alpha_1 = centered.abs().mean()
        
        # 3) Residual, second pass B2, alpha_2
        r = centered - alpha_1 * B1
        B2 = torch.sign(r)
        alpha_2 = r.abs().mean()
        
        if order >= 2:
            # 4) Solve for alpha_1, alpha_2 in closed form
            #    We define:
            #       d = # valid elements
            #       c12 = sum(B1 * B2)
            #       c1w = sum(centered * B1)
            #       c2w = sum(centered * B2)
            d = float(row_vals.numel())
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()
            c2w = torch.sum(centered * B2).item()
            
            det = d*d - c12*c12
            if abs(det) > 1e-12:
                alpha_1_new = ( c1w*d - c2w*c12 ) / det
                alpha_2_new = ( c2w*d - c1w*c12 ) / det
                # alpha should be non-negative, so we clamp to >= 0
                alpha_1 = max(alpha_1_new, 0.0)
                alpha_2 = max(alpha_2_new, 0.0)

        # 5) Reconstruct final row approximation
        approx = row_mean + alpha_1 * B1 + alpha_2 * B2
        
        # Place this approximation back into sum_order at masked positions
        out_row = sum_order[row_idx, :]
        out_row[row_mask] = approx

    return sum_order

index = 0

@torch.no_grad()
def coupled_residual_binarization(x, mask, order=2):
    """
    A unified binarization function that:
      - For order == 1: Performs a single-pass binarization (original simple approach).
      - For order >= 2: Performs a coupled two-expansion binarization (new approach),
        which jointly solves for the two scale factors.
    """
    global index
    index += 1

    # We'll always create sum_order and clone x
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    # ---------------------------
    # Case 1: order == 1
    # ---------------------------
    if order == 1:
        # Exactly the old single‐pass binarization
        residual = new_matrix - sum_order
        # Keep only valid positions
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=residual.device))

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row-wise mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary = sign(masked_x_tensor)
        binary = torch.sign(masked_x_tensor)
        # Multiply by alpha
        binary *= scale_tensor_all[:, None]
        # Then add back the row mean
        binary += mean_tensor_all[:, None]

        # Add to sum_order for final approximation
        sum_order = sum_order + binary * mask

        return sum_order

    # ---------------------------
    # Case 2: order == 2
    # ---------------------------
    else:
        """
        Coupled two-expansion binarization:
          w ~ alpha1 * B1 + alpha2 * B2 + row_mean
        with alpha1, alpha2 solved jointly.
        """

        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                # If mask is all false in this row, skip
                continue

            row_vals = new_matrix[row_idx, row_mask]

            # 1) Subtract row mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # 2) First expansion: B1, alpha1
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # 3) Second expansion: B2, alpha2 w.r.t. residual
            r = centered - alpha1 * B1
            B2 = torch.sign(r)
            alpha2 = r.abs().mean()

            # 4) Solve alpha1, alpha2 in closed form simultaneously
            #    Minimizing || centered - alpha1 B1 - alpha2 B2 ||^2
            d = float(row_vals.numel())
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()
            c2w = torch.sum(centered * B2).item()

            det = d * d - c12 * c12
            if abs(det) > 1e-12:
                # Solve the 2x2 linear system for alpha1, alpha2
                alpha1_new = ( c1w * d - c2w * c12 ) / det
                alpha2_new = ( c2w * d - c1w * c12 ) / det
                # Constrain to be non-negative
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)

            # 5) Final approximation for that row
            approx_row = row_mean + alpha1 * B1 + alpha2 * B2
            
            # Put it back into sum_order
            sum_order[row_idx, row_mask] = approx_row

        return sum_order

@torch.no_grad()
def coupled_residual_binarization_stable(x, mask, order=2, lam=1e-5):
    """
    A unified binarization function with Tikhonov stabilization for the 2-expansion case.

    Args:
      x (tensor): weight matrix, shape (oc, ic)
      mask (tensor, bool): True where weights are to be binarized, False otherwise
      order (int): 
         - 1 => single-pass binarization: w ~ alpha * sign(w - mean)
         - >=2 => two-expansion binarization with Tikhonov-stabilized
                  closed-form for alpha1, alpha2
      lam (float): Tikhonov regularization strength (rho).
                   By default 1e-5 is used; 
                   you may tune this if alphas are still 0 or negative too frequently.

    Returns:
      sum_order (tensor): the binarized approximation of x
    """
    # We'll always create sum_order and clone x
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    # A small global index if needed
    # (optional, matching your existing pattern)
    global index
    index += 1

    # ---------------------------
    # Case 1: order == 1
    # ---------------------------
    if order == 1:
        # Exactly the old single-pass binarization
        residual = new_matrix - sum_order
        # Keep only valid positions
        masked_x_tensor = torch.where(
            mask, 
            residual, 
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row-wise mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary = sign(masked_x_tensor)
        binary = torch.sign(masked_x_tensor)
        # Multiply by alpha
        binary *= scale_tensor_all[:, None]
        # Then add back the row mean
        binary += mean_tensor_all[:, None]

        # Add to sum_order for final approximation
        sum_order = sum_order + binary * mask

        return sum_order

    # ---------------------------
    # Case 2: order >= 2
    # ---------------------------
    else:
        """
        Coupled two-expansion binarization:
          w ~ alpha1 * B1 + alpha2 * B2 + row_mean
        with alpha1, alpha2 solved in closed form 
        plus Tikhonov (ridge) stabilization term:
          + lam * (alpha1^2 + alpha2^2).
        """
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                # If mask is all false in this row, skip
                continue

            row_vals = new_matrix[row_idx, row_mask]

            # (1) Subtract row mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # (2) First expansion: B1, alpha1
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # (3) Second expansion: B2, alpha2 w.r.t. residual
            r = centered - alpha1 * B1
            B2 = torch.sign(r)
            alpha2 = r.abs().mean()

            # (4) Solve alpha1, alpha2 in closed form with Tikhonov
            d = float(row_vals.numel())
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()
            c2w = torch.sum(centered * B2).item()

            # The system is:
            #   [ (d + lam)   -c12     ] [ alpha1 ] = [ c1w ]
            #   [   -c12    (d + lam) ] [ alpha2 ]   [ c2w ]
            #
            # Denominator:
            denom = (d + lam) * (d + lam) - c12 * c12

            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom

                # Constrain to be non-negative
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)
#            print(f"[ROW = {row_idx}] : alpha_1 : ", alpha1)
#            print(f"[ROW = {row_idx}] : alpha_2 : ", alpha2)

            # (5) Final approximation for that row
            approx_row = row_mean + alpha1 * B1 + alpha2 * B2
            sum_order[row_idx, row_mask] = approx_row

        return sum_order
@torch.no_grad()
def coupled_residual_binarization_stable_v2(
    x, 
    mask, 
    order=2, 
    lam=1e-5, 
    max_iters=3
):
    """
    A second version of the stabilized coupled binarization approach, 
    now with an iterative re-fitting step for the two-expansion case.

    Args:
      x (tensor): Weight matrix of shape (oc, ic).
      mask (tensor of bool): True where we binarize, False otherwise.
      order (int):
         - 1 => Single-pass binarization: w ~ alpha * sign(w - mean).
         - >= 2 => Two-expansion binarization with Tikhonov-stabilized 
                   alpha1, alpha2, plus iterative re-fitting of B2.
      lam (float): Tikhonov regularization term.
      max_iters (int): Number of small coordinate-descent iterations 
                       in the two-expansion step.

    Returns:
      sum_order (tensor): Binarized approximation of x.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    # ------------- CASE 1: order == 1 -------------
    if order == 1:
        # Single-pass binarization, exactly as before
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(
            mask, residual, 
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Center each row
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binarize
        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        # Add back the row mean
        binary += mean_tensor_all[:, None]

        sum_order += binary * mask
        return sum_order

    # ------------- CASE 2: order >= 2 -------------
    else:
        """
        Coupled 2-expansion binarization with Tikhonov regularization 
        + iterative B2 refitting.
        """
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                # skip if this row is all unmasked
                continue

            row_vals = new_matrix[row_idx, row_mask]

            # 1) subtract mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # 2) First expansion (B1, alpha1)
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # We'll run a small coordinate-descent loop
            # around (alpha1, alpha2, B2):
            alpha2 = 0.0
            B2 = torch.sign(centered)  # dummy init
            d = float(row_vals.numel())

            for _ in range(max_iters):
                # (a) Recompute residual after current alpha1,B1
                r = centered - alpha1 * B1 * 1.0  # copy
                # (b) Re-fit B2
                B2 = torch.sign(r)

                # (c) approximate alpha2 from the residual magnitude
                alpha2_guess = r.abs().mean()

                # (d) Solve alpha1, alpha2 in closed form with Tikhonov:
                c12 = torch.sum(B1 * B2).item()
                c1w = torch.sum(centered * B1).item()
                c2w = torch.sum(centered * B2).item()

                # Tikhonov system:
                #   [ (d + lam)  -c12      ] [ alpha1 ] = [ c1w ]
                #   [ -c12      (d + lam) ] [ alpha2 ]   [ c2w ]
                denom = (d + lam) * (d + lam) - (c12 * c12)

                if abs(denom) > 1e-12:
                    alpha1_new = ((d + lam)*c1w - c12 * c2w) / denom
                    alpha2_new = ((d + lam)*c2w - c12 * c1w) / denom
                    # clamp to nonnegative
                    alpha1 = max(alpha1_new, 0.0)
                    alpha2 = max(alpha2_new, 0.0)
                else:
                    # fallback
                    alpha1 = max(c1w / (d + lam), 0.0)
                    alpha2 = max(alpha2_guess, 0.0)

                # Optionally: if alpha2 is extremely small,
                # we might break early. But let's just keep
                # the iteration going to see if we can "revive"
                # alpha2 in the next pass. No early break.

            # done iteration

            # 3) final approximation for that row
            approx_row = row_mean + alpha1 * B1 + alpha2 * B2
            sum_order[row_idx, row_mask] = approx_row

        return sum_order
@torch.no_grad()
def coupled_residual_binarization_stable_v3(x, mask, order=2, lam=1e-5):
    """
    A single-pass, closed-form binarization with re-centering and Tikhonov stability.

    When order == 1:
      -> single alpha * sign( (w - row_mean) ).
    When order >= 2:
      -> alpha1 * B1 + alpha2 * B2 + row_mean, 
         with Tikhonov-stabilized closed-form for alpha1, alpha2
         AND a re-centering step for the second residual pass.

    Args:
      x (Tensor): the weight matrix of shape (oc, ic)
      mask (Tensor bool): which entries to binarize (True => use weight)
      order (int): 
         1 => single expansion, 
         >=2 => two expansions with stabilized coupling
      lam (float): Tikhonov (ridge) regularization parameter.

    Returns:
      sum_order (Tensor): approximate binarized reconstruction, same shape as x.
    """
    # We'll accumulate final approximation here
    sum_order = torch.zeros_like(x)
    # Copy & mask out invalid positions
    new_matrix = x.clone() * mask

    # optional: track usage count
    global index
    index += 1

    if order == 1:
        # ----------------------
        # Single-pass binarization
        # ----------------------
        residual = new_matrix  # or new_matrix - sum_order, but sum_order is 0
        masked_x_tensor = torch.where(
            mask, 
            residual, 
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row-wise mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary sign
        binary = torch.sign(masked_x_tensor)
        # Multiply by scale
        binary *= scale_tensor_all[:, None]
        # Add row mean
        binary += mean_tensor_all[:, None]

        # Done
        sum_order = sum_order + binary * mask
        return sum_order

    else:
        # ----------------------
        # Two-expansion binarization with:
        #  (1) re-centering each pass 
        #  (2) Tikhonov for alpha1, alpha2
        # ----------------------
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                # skip if no valid positions in this row
                continue

            row_vals = new_matrix[row_idx, row_mask]
            d = float(row_vals.numel())

            # (1) Row mean
            row_mean = row_vals.mean()  # not masked_x_tensor => same effect
            centered = row_vals - row_mean

            # (2) B1, alpha1 from the centered row
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # (3) Residual r
            r = centered - alpha1 * B1

            # (3a) Re-center r => reduce correlation
            r_mean = r.mean()
            r_centered = r - r_mean

            # B2, alpha2 from the re-centered residual
            B2 = torch.sign(r_centered)
            alpha2 = r_centered.abs().mean()

            # (4) Solve alpha1, alpha2 in one shot with Tikhonov
            #     Minimizing ||(w-mean) - alpha1 B1 - alpha2 B2||^2 + lam (alpha1^2 + alpha2^2).
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()  # <(w-mean), B1>
            c2w = torch.sum(centered * B2).item()  # <(w-mean), B2>

            # The system is:
            #   [d + lam  , -c12      ] [alpha1] = [ c1w ]
            #   [-c12     , d + lam   ] [alpha2]   [ c2w ]
            #
            denom = (d + lam) * (d + lam) - (c12 ** 2)
            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom

                # clamp to non-negative
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)
            else:
                # fallback if denom is ~0
                # keep the naive alpha1, alpha2 from above
                pass

            # (5) Final row reconstruction
            # w_approx = row_mean + alpha1*B1 + alpha2*B2
            row_approx = row_mean + alpha1 * B1 + alpha2 * B2

            # place into sum_order
            sum_order[row_idx, row_mask] = row_approx

        return sum_order

import torch

@torch.no_grad()
def coupled_residual_binarization_stable_v4(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    A single-pass, closed-form 2-expansion binarization with:
      - Re-centering of the row (for B1)
      - Re-centering of residual (for B2)
      - Tikhonov (ridge) regularization for stability
      - Correlation damping to avoid alpha2 => 0 if B1,B2 are strongly correlated

    When order == 1:
      -> single expansion:  w ~ alpha * sign( (w-row_mean) )
    When order >= 2:
      -> w ~ row_mean + alpha1 * B1 + alpha2 * B2
         with ridge-stabilized solution for alpha1, alpha2
         plus correlation damping on c12 if c12>0.

    Args:
      x (Tensor): (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x, True => valid entries
      order (int): 
         - 1 => single expansion
         - >=2 => two expansions w/ correlation damping
      lam (float): Tikhonov/ridge strength
      corr_damp (float): factor in [0,1], how much to scale down positive c12

    Returns:
      sum_order (Tensor): approximate binarized reconstruction
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    if order == 1:
        # ---------------------------
        # Single-pass binarization
        # ---------------------------
        residual = new_matrix
        # Only valid positions
        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = avg abs value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary sign
        binary = torch.sign(masked_x_tensor)
        # Scale
        binary *= scale_tensor_all[:, None]
        # Add row mean
        binary += mean_tensor_all[:, None]

        sum_order = sum_order + binary * mask
        return sum_order

    else:
        # ---------------------------
        # Two-expansion binarization
        # with re-centering + Tikhonov + correlation damping
        # ---------------------------
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                continue

            row_vals = new_matrix[row_idx, row_mask]
            d = float(row_vals.numel())

            # 1) Row mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # 2) B1, alpha1 from centered
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # 3) Residual r
            r = centered - alpha1 * B1

            # 3a) Re-center the residual
            r_mean = r.mean()
            r_centered = r - r_mean

            # B2, alpha2 from r_centered
            B2 = torch.sign(r_centered)
            alpha2 = r_centered.abs().mean()

            # 4) Tikhonov-stabilized closed-form for alpha1, alpha2
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()  # <(w-mean), B1>
            c2w = torch.sum(centered * B2).item()  # <(w-mean), B2>

            # Correlation damping if c12>0
            if c12 > 0:
                c12 = torch.where(c12 > 0, c12 * (1.0 - corr_damp), c12)

            # Solve system:
            #   [d + lam, -c12   ] [alpha1] = [c1w]
            #   [-c12,   d + lam ] [alpha2]   [c2w]
            denom = (d + lam) * (d + lam) - (c12**2)
            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
                # clamp non-negative
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)
            else:
                # fallback if near-singular
                pass

            # 5) Final row approximation
            row_approx = row_mean + alpha1 * B1 + alpha2 * B2
            sum_order[row_idx, row_mask] = row_approx

        return sum_order

@torch.no_grad()
def coupled_residual_binarization_stable_v5(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    A single-pass or two-pass binarization with various stabilizations.
    When order == 1: single expansion  w ~ row_mean + alpha * sign(w - row_mean)
                     now with Tikhonov regularization for alpha.
    When order >= 2: two expansions   w ~ row_mean + alpha1 B1 + alpha2 B2
                     with ridge-stabilized solution + correlation damping.
    ...
    """

    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    if order == 1:
        # ---------------------------
        # Single-pass binarization
        # with Tikhonov for alpha
        # ---------------------------
        residual = new_matrix

        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row mean
        centered_x = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale WITHOUT Tikhonov would be:
        #   scale_tensor_all = torch.nanmean(torch.abs(centered_x), dim=1)
        # Instead we do Tikhonov ridge:
        #   alpha = (||centered_x||_1) / (d + lam)
        # where d is the count of valid entries.
        abs_centered = torch.abs(centered_x)
        # number of valid entries in each row:
        valid_counts = torch.sum(~torch.isnan(centered_x), dim=1).float()
        l1_sums = torch.nan_to_num(abs_centered, 0.0).sum(dim=1)  # sum of absolute

        # Tikhonov scale
        # NOTE: we clamp at 1e-12 to avoid division by zero if no valid entries
        denom = valid_counts + lam
        denom = torch.where(denom < 1e-12, torch.tensor(1e-12, device=denom.device), denom)
        scale_tensor_all = l1_sums / denom

        # Binary sign
        binary = torch.sign(centered_x)

        # Multiply by alpha
        binary *= scale_tensor_all[:, None]

        # Add row mean
        binary += mean_tensor_all[:, None]

        # Final
        sum_order = sum_order + torch.where(torch.isnan(masked_x_tensor),
                                            torch.zeros_like(binary),
                                            binary)
        return sum_order

    else:
        # ---------------------------
        # Two-expansion binarization
        # with re-centering + Tikhonov + correlation damping
        # ---------------------------
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                continue

            row_vals = new_matrix[row_idx, row_mask]
            d = float(row_vals.numel())
            if d < 1e-12:
                continue

            # 1) Row mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # 2) B1, alpha1 from centered
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # 3) Residual r
            r = centered - alpha1 * B1
            # 3a) Re-center the residual
            r_mean = r.mean()
            r_centered = r - r_mean

            # B2, alpha2 from r_centered
            B2 = torch.sign(r_centered)
            alpha2 = r_centered.abs().mean()

            # 4) Tikhonov-stabilized closed-form for alpha1, alpha2
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()  # <(w-mean), B1>
            c2w = torch.sum(centered * B2).item()  # <(w-mean), B2>

            # Correlation damping if c12>0
            if c12 > 0:
                c12 = torch.where(c12 > 0, c12 * (1.0 - corr_damp), c12)

            # Solve system:
            #   [d + lam,   -c12     ] [alpha1] = [c1w]
            #   [  -c12,    d + lam ] [alpha2]   [c2w]
            denom = (d + lam) * (d + lam) - (c12**2)
            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)

            # 5) Final row approximation
            row_approx = row_mean + alpha1 * B1 + alpha2 * B2
            sum_order[row_idx, row_mask] = row_approx

        return sum_order
@torch.no_grad()
def coupled_residual_binarization_stable_v6(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    A single-pass (order==1) or two-expansion (order>=2) binarization with:
      - Row mean centering
      - Residual re-centering
      - Tikhonov (ridge) regularization
      - Correlation damping
      - *Sign refinement step* (new in v5) for B2 in two-expansion mode

    When order == 1:
      -> Single expansion: w ~ alpha * sign( (w - row_mean) )

    When order >= 2:
      -> w ~ row_mean + alpha1 * B1 + alpha2 * B2
         *with one sign-refinement pass* for B2 after solving alpha1, alpha2.

    Args:
      x (Tensor):         (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x; True => valid entries
      order (int):        1 => single expansion, >=2 => two expansions
      lam (float):        Tikhonov/ridge strength
      corr_damp (float):  factor in [0,1], how much to scale down c12 if c12>0

    Returns:
      sum_order (Tensor): approximate binarized reconstruction, same shape as x
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    # ---------------------------
    # Case 1: single expansion
    # ---------------------------
    if order == 1:
        residual = new_matrix
        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary sign
        binary = torch.sign(masked_x_tensor)

        # Scale
        binary *= scale_tensor_all[:, None]

        # Add row mean
        binary += mean_tensor_all[:, None]

        sum_order = sum_order + binary * mask
        return sum_order

    # ---------------------------
    # Case 2: two expansions
    # with sign refinement (v5)
    # ---------------------------
    oc, ic = new_matrix.shape

    for row_idx in range(oc):
        row_mask = mask[row_idx, :]
        if not torch.any(row_mask):
            continue

        row_vals = new_matrix[row_idx, row_mask]
        d = float(row_vals.numel())

        # 1) Row mean
        row_mean = row_vals.mean()
        centered = row_vals - row_mean

        # 2) B1, alpha1 from centered
        B1 = torch.sign(centered)
        alpha1 = centered.abs().mean()

        # 3) Residual r
        r = centered - alpha1 * B1

        # 3a) Re-center the residual
        r_mean = r.mean()
        r_centered = r - r_mean

        # B2, alpha2 from r_centered
        B2 = torch.sign(r_centered)
        alpha2 = r_centered.abs().mean()

        # 4) Tikhonov-stabilized closed-form for alpha1, alpha2
        #    with correlation damping if c12>0
        def solve_alphas(B1, B2, c1w, c2w, d, lam, corr_damp):
            c12 = (B1 * B2).sum().item()
            if c12 > 0:
                c12 *= (1.0 - corr_damp)
            # System:
            #   [d + lam, -c12   ] [alpha1] = [c1w]
            #   [-c12,   d + lam ] [alpha2]   [c2w]
            denom = (d + lam) * (d + lam) - c12 * c12
            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
                return max(alpha1_new, 0.0), max(alpha2_new, 0.0)
            else:
                # fallback if near-singular
                return 0.0, 0.0

        c1w = (centered * B1).sum().item()   # <(w-mean), B1>
        c2w = (centered * B2).sum().item()   # <(w-mean), B2>
        alpha1, alpha2 = solve_alphas(B1, B2, c1w, c2w, d, lam, corr_damp)

        # -------------------------
        # (NEW in v5) Sign-Refinement Step:
        # After alpha1, alpha2 are updated, recompute B2 from the
        # *actual* final residual (w-mean - alpha1*B1). Then re-solve.
        # -------------------------
        refined_residual = centered - alpha1 * B1
        # We do not re-add r_mean here; we've effectively folded that in
        # (since alpha1, alpha2 had already accounted for it).
        B2 = torch.sign(refined_residual)
        c2w_refined = (centered * B2).sum().item()  # updated <(w-mean), B2>
        alpha1, alpha2 = solve_alphas(B1, B2, c1w, c2w_refined, d, lam, corr_damp)

        # 5) Final row approximation
        row_approx = row_mean + alpha1 * B1 + alpha2 * B2
        sum_order[row_idx, row_mask] = row_approx

    return sum_order

@torch.no_grad()
def coupled_residual_binarization_stable_v7(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1,
    skip_refinement=False,
    symmetric_damp=False
):
    """
    Vectorized CRB v7: two-expansion binarization with Tikhonov regularization,
    correlation damping, and two-way sign refinement.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    mask_f = mask.float()
    d = mask_f.sum(dim=1)  # (oc,) count of valid elements per row
    d_safe = torch.clamp(d, min=1.0)

    # ---------------------------
    # Case 1: single expansion
    # ---------------------------
    if order == 1:
        row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
        centered = (new_matrix - row_mean[:, None]) * mask_f
        B1 = torch.sign(centered) * mask_f
        alpha1 = (centered.abs() * mask_f).sum(dim=1) / d_safe
        sum_order = (row_mean[:, None] + alpha1[:, None] * B1) * mask_f
        return sum_order

    # ---------------------------
    # Case 2: two expansions with two-way sign refinement (v7)
    # ---------------------------

    def solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp):
        c12 = (B1 * B2 * mask_f).sum(dim=1)
        if symmetric_damp:
            c12 = c12 * (1.0 - corr_damp)
        else:
            c12 = torch.where(c12 > 0, c12 * (1.0 - corr_damp), c12)
        A = d + lam
        denom = A * A - c12 * c12
        safe = denom.abs() > 1e-12
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        a1 = torch.clamp((A * c1w - c12 * c2w) / safe_denom, min=0.0)
        a2 = torch.clamp((A * c2w - c12 * c1w) / safe_denom, min=0.0)
        a1 = torch.where(safe, a1, torch.zeros_like(a1))
        a2 = torch.where(safe, a2, torch.zeros_like(a2))
        return a1, a2

    # Step 1: Row mean and centering
    row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe  # (oc,)
    centered = (new_matrix - row_mean[:, None]) * mask_f   # (oc, ic)

    # Step 2: B1 = sign(centered), initial alpha1
    B1 = torch.sign(centered) * mask_f
    alpha1_init = (centered.abs() * mask_f).sum(dim=1) / d_safe

    # Step 3: Residual -> B2
    r = (centered - alpha1_init[:, None] * B1) * mask_f
    r_mean = (r * mask_f).sum(dim=1) / d_safe
    r_centered = (r - r_mean[:, None]) * mask_f
    B2 = torch.sign(r_centered) * mask_f

    # Step 4: Solve alpha1, alpha2 jointly
    c1w = (centered * B1).sum(dim=1)
    c2w = (centered * B2).sum(dim=1)
    alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

    if not skip_refinement:
        # Step 5: Refine B2 with centering, re-solve
        temp5 = (centered - alpha1[:, None] * B1) * mask_f
        temp5_mean = (temp5 * mask_f).sum(dim=1) / d_safe
        B2 = torch.sign((temp5 - temp5_mean[:, None]) * mask_f) * mask_f
        c2w = (centered * B2).sum(dim=1)
        alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

        # Step 6 (v7): Refine B1 with centering, re-solve
        temp6 = (centered - alpha2[:, None] * B2) * mask_f
        temp6_mean = (temp6 * mask_f).sum(dim=1) / d_safe
        B1 = torch.sign((temp6 - temp6_mean[:, None]) * mask_f) * mask_f
        c1w = (centered * B1).sum(dim=1)
        alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

    # Step 7: Final reconstruction with residual mean correction
    approx = (alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f
    residual_final = (centered - approx) * mask_f
    mu_correction = (residual_final * mask_f).sum(dim=1) / d_safe
    sum_order = (row_mean[:, None] + mu_correction[:, None] + alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f

    return sum_order

@torch.no_grad()
def coupled_residual_binarization_seqalpha(
    x,
    mask,
    order=2,
    skip_refinement=False
):
    """
    BRAQ-equivalent binarization using BRAQ's exact code path (nanmean, torch.where)
    for bit-exact float16 equivalence. With optional sign refinement.
    Without refinement, this produces BIT-EXACT same output as high_order_residual.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    # Use BRAQ's exact code path for base decomposition
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))

        mean_val = torch.nanmean(masked_x, dim=1)
        mean_val = torch.where(torch.isnan(mean_val), torch.zeros_like(mean_val), mean_val)
        masked_x -= mean_val[:, None]
        scale_val = torch.nanmean(torch.abs(masked_x), dim=1)
        scale_val = torch.where(torch.isnan(scale_val), torch.zeros_like(scale_val), scale_val)

        binary = torch.sign(masked_x)
        binary *= scale_val[:, None]
        binary += mean_val[:, None]
        sum_order = sum_order + binary * mask

    if skip_refinement or order < 2:
        return sum_order

    # --- Sign refinement on top of BRAQ base ---
    # Extract B1, alpha1, mean1 from first expansion
    # and B2, alpha2, mean2 from second expansion
    # by re-deriving from sum_order

    # Re-derive first expansion components
    masked_w = torch.where(mask, new_matrix, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
    mean1 = torch.nanmean(masked_w, dim=1)
    mean1 = torch.where(torch.isnan(mean1), torch.zeros_like(mean1), mean1)
    centered1 = masked_w - mean1[:, None]
    alpha1 = torch.nanmean(torch.abs(centered1), dim=1)
    alpha1 = torch.where(torch.isnan(alpha1), torch.zeros_like(alpha1), alpha1)
    B1 = torch.sign(centered1)  # NaN → 0

    # First expansion reconstruction
    exp1 = torch.zeros_like(x)
    exp1_binary = B1 * alpha1[:, None] + mean1[:, None]
    exp1 = exp1_binary * mask

    # Re-derive second expansion components
    residual1 = new_matrix - exp1
    masked_r = torch.where(mask, residual1, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
    mean2 = torch.nanmean(masked_r, dim=1)
    mean2 = torch.where(torch.isnan(mean2), torch.zeros_like(mean2), mean2)
    centered2 = masked_r - mean2[:, None]
    alpha2 = torch.nanmean(torch.abs(centered2), dim=1)
    alpha2 = torch.where(torch.isnan(alpha2), torch.zeros_like(alpha2), alpha2)
    B2 = torch.sign(centered2)

    # Step 5: Refine B2 using current alpha1
    # Recompute residual and re-sign
    temp5 = torch.where(mask, residual1, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
    temp5_mean = torch.nanmean(temp5, dim=1)
    temp5_mean = torch.where(torch.isnan(temp5_mean), torch.zeros_like(temp5_mean), temp5_mean)
    temp5_centered = temp5 - temp5_mean[:, None]
    B2 = torch.sign(temp5_centered)

    # Recompute alpha2 for new B2
    alpha2 = torch.nanmean(torch.abs(temp5_centered), dim=1)
    alpha2 = torch.where(torch.isnan(alpha2), torch.zeros_like(alpha2), alpha2)
    mean2 = temp5_mean

    # Step 6: Refine B1 using current alpha2
    exp2 = (B2 * alpha2[:, None] + mean2[:, None]) * mask
    residual_for_b1 = new_matrix - exp2
    temp6 = torch.where(mask, residual_for_b1, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
    temp6_mean = torch.nanmean(temp6, dim=1)
    temp6_mean = torch.where(torch.isnan(temp6_mean), torch.zeros_like(temp6_mean), temp6_mean)
    temp6_centered = temp6 - temp6_mean[:, None]
    B1 = torch.sign(temp6_centered)

    # Recompute alpha1 for new B1
    alpha1 = torch.nanmean(torch.abs(temp6_centered), dim=1)
    alpha1 = torch.where(torch.isnan(alpha1), torch.zeros_like(alpha1), alpha1)
    mean1 = temp6_mean

    # Recompute second expansion with refined B1
    exp1 = (B1 * alpha1[:, None] + mean1[:, None]) * mask
    residual_final = new_matrix - exp1
    masked_rf = torch.where(mask, residual_final, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
    mean2 = torch.nanmean(masked_rf, dim=1)
    mean2 = torch.where(torch.isnan(mean2), torch.zeros_like(mean2), mean2)
    centered_rf = masked_rf - mean2[:, None]
    alpha2 = torch.nanmean(torch.abs(centered_rf), dim=1)
    alpha2 = torch.where(torch.isnan(alpha2), torch.zeros_like(alpha2), alpha2)
    B2 = torch.sign(centered_rf)

    # Final output: BRAQ-style accumulation
    sum_order = exp1 + (B2 * alpha2[:, None] + mean2[:, None]) * mask

    return sum_order

@torch.no_grad()
def coupled_residual_binarization_resrhs(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1,
    skip_refinement=False
):
    """
    CRB with residual-projected RHS: c2w is computed from the residual
    (centered - alpha1*B1) instead of centered, fixing the alpha2 deflation
    that occurs because sum(centered*B2) bakes in alpha1*c12.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    mask_f = mask.float()
    d = mask_f.sum(dim=1)
    d_safe = torch.clamp(d, min=1.0)

    if order == 1:
        row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
        centered = (new_matrix - row_mean[:, None]) * mask_f
        B1 = torch.sign(centered) * mask_f
        alpha1 = (centered.abs() * mask_f).sum(dim=1) / d_safe
        sum_order = (row_mean[:, None] + alpha1[:, None] * B1) * mask_f
        return sum_order

    def solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp):
        c12 = (B1 * B2 * mask_f).sum(dim=1)
        c12 = torch.where(c12 > 0, c12 * (1.0 - corr_damp), c12)
        A = d + lam
        denom = A * A - c12 * c12
        safe = denom.abs() > 1e-12
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        a1 = torch.clamp((A * c1w - c12 * c2w) / safe_denom, min=0.0)
        a2 = torch.clamp((A * c2w - c12 * c1w) / safe_denom, min=0.0)
        a1 = torch.where(safe, a1, torch.zeros_like(a1))
        a2 = torch.where(safe, a2, torch.zeros_like(a2))
        return a1, a2

    # Step 1: Row mean and centering
    row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
    centered = (new_matrix - row_mean[:, None]) * mask_f

    # Step 2: B1 = sign(centered), initial alpha1
    B1 = torch.sign(centered) * mask_f
    alpha1_init = (centered.abs() * mask_f).sum(dim=1) / d_safe

    # Step 3: Residual -> B2
    r = (centered - alpha1_init[:, None] * B1) * mask_f
    r_mean = (r * mask_f).sum(dim=1) / d_safe
    r_centered = (r - r_mean[:, None]) * mask_f
    B2 = torch.sign(r_centered) * mask_f

    # Step 4: Solve alpha1, alpha2 jointly
    # KEY CHANGE: c2w uses residual projection instead of centered
    c1w = (centered * B1).sum(dim=1)
    c2w = (r * B2).sum(dim=1)  # residual-projected RHS
    alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

    if not skip_refinement:
        # Step 5: Refine B2 with centering, re-solve
        temp5 = (centered - alpha1[:, None] * B1) * mask_f
        temp5_mean = (temp5 * mask_f).sum(dim=1) / d_safe
        B2 = torch.sign((temp5 - temp5_mean[:, None]) * mask_f) * mask_f
        # Use residual-projected c2w after refinement too
        c2w = (temp5 * B2).sum(dim=1)
        alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

        # Step 6: Refine B1 with centering, re-solve
        temp6 = (centered - alpha2[:, None] * B2) * mask_f
        temp6_mean = (temp6 * mask_f).sum(dim=1) / d_safe
        B1 = torch.sign((temp6 - temp6_mean[:, None]) * mask_f) * mask_f
        c1w = (centered * B1).sum(dim=1)
        # Recompute c2w with residual from new B1
        r_new = (centered - alpha1[:, None] * B1) * mask_f
        c2w = (r_new * B2).sum(dim=1)
        alpha1, alpha2 = solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp)

    # Step 7: Final reconstruction with residual mean correction
    approx = (alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f
    residual_final = (centered - approx) * mask_f
    mu_correction = (residual_final * mask_f).sum(dim=1) / d_safe
    sum_order = (row_mean[:, None] + mu_correction[:, None] + alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f

    return sum_order

@torch.no_grad()
def coupled_residual_binarization_adaptive(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1,
    col_weights=None
):
    """
    Adaptive CRB: runs both refined and unrefined paths for order=2,
    selects per-row based on Hessian-weighted error from GPTQ.

    The sign refinement (Steps 5-6) reduces Frobenius error but can increase
    prediction-relevant error on some models. By comparing Hessian-weighted
    error per row, we keep refinement only where it actually helps the
    GPTQ objective.

    col_weights: (ic,) tensor of column importance from GPTQ Hessian inverse.
                 Typically 1/(Hinv_diag^2). If None, uses uniform weights
                 (equivalent to standard CRB with refinement).
    """
    # For order=1, refinement doesn't apply
    if order == 1:
        return coupled_residual_binarization_stable_v7(
            x, mask, order=1, lam=lam, corr_damp=corr_damp
        )

    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    mask_f = mask.float()
    d = mask_f.sum(dim=1)
    d_safe = torch.clamp(d, min=1.0)

    def solve_alphas_vec(B1, B2, c1w, c2w, d, lam, corr_damp):
        c12 = (B1 * B2 * mask_f).sum(dim=1)
        c12 = torch.where(c12 > 0, c12 * (1.0 - corr_damp), c12)
        A = d + lam
        denom = A * A - c12 * c12
        safe = denom.abs() > 1e-12
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        a1 = torch.clamp((A * c1w - c12 * c2w) / safe_denom, min=0.0)
        a2 = torch.clamp((A * c2w - c12 * c1w) / safe_denom, min=0.0)
        a1 = torch.where(safe, a1, torch.zeros_like(a1))
        a2 = torch.where(safe, a2, torch.zeros_like(a2))
        return a1, a2

    def make_reconstruction(row_mean, B1, B2, alpha1, alpha2, centered, mask_f, d_safe):
        approx = (alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f
        residual_final = (centered - approx) * mask_f
        mu_corr = (residual_final * mask_f).sum(dim=1) / d_safe
        return (row_mean[:, None] + mu_corr[:, None] + alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f

    # ===== SHARED: Steps 1-4 (identical for both paths) =====

    # Step 1: Row mean and centering
    row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
    centered = (new_matrix - row_mean[:, None]) * mask_f

    # Step 2: B1 = sign(centered)
    B1_init = torch.sign(centered) * mask_f
    alpha1_raw = (centered.abs() * mask_f).sum(dim=1) / d_safe

    # Step 3: Residual -> B2
    r = (centered - alpha1_raw[:, None] * B1_init) * mask_f
    r_mean = (r * mask_f).sum(dim=1) / d_safe
    r_centered = (r - r_mean[:, None]) * mask_f
    B2_init = torch.sign(r_centered) * mask_f

    # Step 4: Joint alpha solve
    c1w_init = (centered * B1_init).sum(dim=1)
    c2w_init = (centered * B2_init).sum(dim=1)
    alpha1_s4, alpha2_s4 = solve_alphas_vec(B1_init, B2_init, c1w_init, c2w_init, d, lam, corr_damp)

    # ===== PATH A: No refinement (Steps 1-4 + 7 only) =====
    Q_noref = make_reconstruction(row_mean, B1_init, B2_init, alpha1_s4, alpha2_s4, centered, mask_f, d_safe)

    # ===== PATH B: With refinement (Steps 5-6 + 7) =====
    # Step 5: Refine B2
    temp5 = (centered - alpha1_s4[:, None] * B1_init) * mask_f
    temp5_mean = (temp5 * mask_f).sum(dim=1) / d_safe
    B2_ref = torch.sign((temp5 - temp5_mean[:, None]) * mask_f) * mask_f
    c2w_ref = (centered * B2_ref).sum(dim=1)
    alpha1_s5, alpha2_s5 = solve_alphas_vec(B1_init, B2_ref, c1w_init, c2w_ref, d, lam, corr_damp)

    # Step 6: Refine B1
    temp6 = (centered - alpha2_s5[:, None] * B2_ref) * mask_f
    temp6_mean = (temp6 * mask_f).sum(dim=1) / d_safe
    B1_ref = torch.sign((temp6 - temp6_mean[:, None]) * mask_f) * mask_f
    c1w_ref = (centered * B1_ref).sum(dim=1)
    alpha1_s6, alpha2_s6 = solve_alphas_vec(B1_ref, B2_ref, c1w_ref, c2w_ref, d, lam, corr_damp)

    Q_ref = make_reconstruction(row_mean, B1_ref, B2_ref, alpha1_s6, alpha2_s6, centered, mask_f, d_safe)

    # ===== PER-BLOCK SELECTION based on clipped Hessian-weighted error =====
    # Use per-BLOCK (not per-row) selection to avoid mixing refined/unrefined
    # rows within the same block, which creates bad GPTQ compensation patterns.
    if col_weights is not None:
        cw_median = col_weights.median().clamp(min=1e-10)
        cw_clipped = torch.clamp(col_weights, max=cw_median * 100.0)
        cw = cw_clipped[None, :] * mask_f
    else:
        cw = mask_f

    total_err_noref = ((new_matrix - Q_noref) ** 2 * cw).sum()
    total_err_ref = ((new_matrix - Q_ref) ** 2 * cw).sum()

    # Use refinement only if it provides substantial Hessian-weighted improvement.
    margin = 0.001  # 0.1% improvement threshold for per-block
    if total_err_ref < total_err_noref * (1.0 - margin):
        return Q_ref
    else:
        return Q_noref

@torch.no_grad()
def coupled_residual_binarization_hessian(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1,
    col_weights=None
):
    """
    CRB with Hessian-weighted alpha solve (no sign refinement).

    Instead of minimizing Frobenius error ||W - α₁B₁ - α₂B₂||²,
    minimizes Hessian-weighted error Σ_j h_j*(W_j - α₁B₁_j - α₂B₂_j)²
    where h_j is the column importance from GPTQ Hessian.

    The signs (B₁, B₂) are the same as BRAQ/crb_norefine. Only the scale
    factors (α₁, α₂) differ — they are optimized for the prediction-relevant
    GPTQ objective rather than raw Frobenius error.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    mask_f = mask.float()
    d = mask_f.sum(dim=1)
    d_safe = torch.clamp(d, min=1.0)

    if order == 1:
        row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
        centered = (new_matrix - row_mean[:, None]) * mask_f
        B1 = torch.sign(centered) * mask_f
        if col_weights is not None:
            cw = col_weights[None, :] * mask_f
            d_h = cw.sum(dim=1).clamp(min=1e-12)
            alpha1 = (centered.abs() * cw).sum(dim=1) / d_h
        else:
            alpha1 = (centered.abs() * mask_f).sum(dim=1) / d_safe
        sum_order = (row_mean[:, None] + alpha1[:, None] * B1) * mask_f
        return sum_order

    # --- Two expansions with Hessian-weighted alpha solve ---

    # Compute Hessian-weighted mask for alpha solve
    if col_weights is not None:
        cw = col_weights[None, :] * mask_f  # (1, ic) * (oc, ic) → (oc, ic)
        d_h = cw.sum(dim=1).clamp(min=1e-12)  # weighted count per row
    else:
        cw = mask_f
        d_h = d

    def solve_alphas_hessian(B1, B2, centered, cw, d_h, lam, corr_damp):
        c12_h = (B1 * B2 * cw).sum(dim=1)
        c12_h = torch.where(c12_h > 0, c12_h * (1.0 - corr_damp), c12_h)
        c1w_h = (centered * B1 * cw).sum(dim=1)
        c2w_h = (centered * B2 * cw).sum(dim=1)
        A = d_h + lam
        denom = A * A - c12_h * c12_h
        safe = denom.abs() > 1e-12
        safe_denom = torch.where(safe, denom, torch.ones_like(denom))
        a1 = torch.clamp((A * c1w_h - c12_h * c2w_h) / safe_denom, min=0.0)
        a2 = torch.clamp((A * c2w_h - c12_h * c1w_h) / safe_denom, min=0.0)
        a1 = torch.where(safe, a1, torch.zeros_like(a1))
        a2 = torch.where(safe, a2, torch.zeros_like(a2))
        return a1, a2

    # Step 1: Row mean and centering
    row_mean = (new_matrix * mask_f).sum(dim=1) / d_safe
    centered = (new_matrix - row_mean[:, None]) * mask_f

    # Step 2: B1 = sign(centered) — same as BRAQ/CRB
    B1 = torch.sign(centered) * mask_f
    alpha1_init = (centered.abs() * mask_f).sum(dim=1) / d_safe

    # Step 3: Residual -> B2 — same as BRAQ/CRB
    r = (centered - alpha1_init[:, None] * B1) * mask_f
    r_mean = (r * mask_f).sum(dim=1) / d_safe
    r_centered = (r - r_mean[:, None]) * mask_f
    B2 = torch.sign(r_centered) * mask_f

    # Step 4: Hessian-weighted joint alpha solve
    alpha1, alpha2 = solve_alphas_hessian(B1, B2, centered, cw, d_h, lam, corr_damp)

    # Step 5: Final reconstruction with residual mean correction
    approx = (alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f
    residual_final = (centered - approx) * mask_f
    mu_correction = (residual_final * mask_f).sum(dim=1) / d_safe
    sum_order = (row_mean[:, None] + mu_correction[:, None] + alpha1[:, None] * B1 + alpha2[:, None] * B2) * mask_f

    return sum_order

@torch.no_grad()
def coupled_residual_binarization_stable_v8(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    'v8' extends the two-expansion approach by also re-optimizing the row offset
    mu in the coordinate-descent loop. This yields a more accurate final
    approximation without adding any new bits or parameters (the offset is
    the same single row-mean float we already had).

    We keep:
      - Tikhonov (ridge) stabilization
      - Correlation damping
      - Two-way sign refinement for B1, B2
      - New: offset (mu) re-solved in closed-form.

    Args:
      x (Tensor):         (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x, True => valid entries
      order (int):        1 => single expansion, >=2 => two expansions
      lam (float):        Tikhonov/ridge strength
      corr_damp (float):  factor in [0,1], how much to scale down positive c12

    Returns:
      sum_order (Tensor): approximate binarized reconstruction
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    # ---------------------------
    # Single expansion (unchanged)
    # ---------------------------
    if order == 1:
        residual = new_matrix
        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )
        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary * mask
        return sum_order

    # --------------------------------------
    # Two expansions + offset refinement (v8)
    # --------------------------------------
    oc, ic = new_matrix.shape

    def solve_alphas(B1, B2, w_centered, d, lam, corr_damp):
        # correlation
        c12 = (B1 * B2).sum().item()
        # if c12 > 0, damp it
        if c12 > 0:
            c12 *= (1.0 - corr_damp)

        # <(w - mu), B1>, <(w - mu), B2>
        c1w = (w_centered * B1).sum().item()
        c2w = (w_centered * B2).sum().item()

        # Solve system:
        #   [d + lam, -c12   ] [alpha1] = [c1w]
        #   [-c12,   d + lam ] [alpha2]   [c2w]
        denom = (d + lam) * (d + lam) - c12 * c12
        if abs(denom) > 1e-12:
            alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
            alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
            alpha1_new = max(alpha1_new, 0.0)
            alpha2_new = max(alpha2_new, 0.0)
            return alpha1_new, alpha2_new
        else:
            return 0.0, 0.0

    for row_idx in range(oc):
        row_mask = mask[row_idx, :]
        if not torch.any(row_mask):
            continue

        row_vals = new_matrix[row_idx, row_mask]
        d = float(row_vals.numel())

        # 1) Initial offset = row mean
        mu = row_vals.mean()
        w_centered = row_vals - mu

        # 2) B1, alpha1 from w_centered
        B1 = torch.sign(w_centered)
        alpha1 = w_centered.abs().mean()

        # 3) B2 from residual (re-centered)
        r1 = w_centered - alpha1 * B1
        r1_mean = r1.mean()
        B2 = torch.sign(r1 - r1_mean)
        alpha2 = (r1 - r1_mean).abs().mean()

        # 4) Solve alpha1, alpha2 (initial)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)

        # 5) Sign refinement for B2
        B2 = torch.sign(w_centered - alpha1 * B1)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)

        # 6) Sign refinement for B1 (two-way)
        B1 = torch.sign(w_centered - alpha2 * B2)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)

        # 7) (NEW in v8) Refine offset mu, then re-solve alpha + sign
        #    a) mu = mean( w_i - alpha1 B_{1i} - alpha2 B_{2i} )
        w_res = row_vals - alpha1 * B1 - alpha2 * B2
        new_mu = w_res.mean()
        # b) Re-center for alpha solves
        w_centered = row_vals - new_mu

        # Re-solve alpha1, alpha2 with updated mu
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)
        
        # c) Optional final sign refinements 
        B2 = torch.sign(w_centered - alpha1 * B1)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)
        B1 = torch.sign(w_centered - alpha2 * B2)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)

        # 8) Final reconstruction for that row
        row_approx = new_mu + alpha1 * B1 + alpha2 * B2
        sum_order[row_idx, row_mask] = row_approx

    return sum_order

@torch.no_grad()
def adaptive_high_order_residual(x, mask, order=2):
    """
    Adaptive High Order Residual Binarization.
    
    This function approximates the input tensor x (after applying mask)
    as a sum of binary components over the specified number of orders.
    
    For each order and for each channel, it computes two candidate scale factors:
    
      - Candidate 1: Uses the mean absolute deviation of the channel’s residual 
                     (as in braq), i.e. α₁ = nanmean(|r - m|).
      
      - Candidate 2: Uses the variance-based estimator, i.e. 
                     α₂ = sqrt(nanmean((r - m)²)) * sqrt(2/π),
                     which is optimal if the residual is Gaussian.
                     
    For each channel the candidate that yields the lower reconstruction error 
    is chosen adaptively. This approach minimizes quantization error under 
    non-ideal residual distributions while introducing no extra bits (the only 
    stored per-channel parameter remains the scale factor).
    
    When order = 1, the weights are represented as W ≈ α * B,
    and when order = 2, W ≈ α₁ * B₁ + α₂ * B₂, as required.
    
    Parameters:
      x (torch.Tensor): The weight tensor.
      mask (torch.Tensor): A binary mask of the same shape as x.
      order (int): The number of residual passes (default 2).
    
    Returns:
      torch.Tensor: The binarized approximation of x.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    global index
    index += 1
    # Prepare a NaN tensor for masked-out positions (ensuring device/dtype consistency)
    nan_tensor = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    
    for od in range(order):
        # Compute the current residual
        residual = new_matrix - sum_order
        # Apply the mask: invalid positions become NaN
        masked_x_tensor = torch.where(mask, residual, nan_tensor)
        
        # Compute channel-wise mean (serves as bias compensation)
        channel_mean = torch.nanmean(masked_x_tensor, dim=1)
        channel_mean = torch.where(torch.isnan(channel_mean), torch.zeros_like(channel_mean), channel_mean)
        
        # Center the residual by subtracting the channel mean
        centered = masked_x_tensor - channel_mean[:, None]
        
        # Candidate 1: Scale via mean absolute deviation (as in braq)
        candidate_scale1 = torch.nanmean(torch.abs(centered), dim=1)
        candidate_scale1 = torch.where(torch.isnan(candidate_scale1), torch.zeros_like(candidate_scale1), candidate_scale1)
        
        # Candidate 2: Scale via variance; for a Gaussian, E[|x|] = std * sqrt(2/π)
        candidate_std = torch.sqrt(torch.nanmean(centered**2, dim=1))
        candidate_scale2 = candidate_std * math.sqrt(2/math.pi)
        candidate_scale2 = torch.where(torch.isnan(candidate_scale2), torch.zeros_like(candidate_scale2), candidate_scale2)
        
        # Both candidates use the same binary pattern (the sign of the centered residual)
        binary = torch.sign(centered)
        
        # Reconstruct the candidate approximations
        rec1 = channel_mean[:, None] + candidate_scale1[:, None] * binary
        rec2 = channel_mean[:, None] + candidate_scale2[:, None] * binary
        
        # Compute per-channel reconstruction errors for both candidates
        error1 = torch.nanmean((masked_x_tensor - rec1)**2, dim=1)
        error2 = torch.nanmean((masked_x_tensor - rec2)**2, dim=1)
        
        # Select the candidate with lower error for each channel
        choose_candidate1 = (error1 <= error2)
        final_scale = torch.where(choose_candidate1, candidate_scale1, candidate_scale2)
        
        # Compute the final binary component for this iteration
        final_component = channel_mean[:, None] + final_scale[:, None] * binary
        
        # Update the accumulated representation (note: multiplication by mask preserves original sparsity)
        sum_order = sum_order + final_component * mask
        
    return sum_order

@torch.no_grad()
def adaptive_high_order_residual_v2(x, mask, order=2):
    """
    Adaptive High Order Residual Binarization using candidate 2 only.

    This function approximates the input tensor x (after applying mask)
    as a sum of binary components over the specified number of orders.
    Instead of adaptively choosing between two candidates, it always uses
    candidate 2, which computes the scale factor based on the variance:
      candidate_scale2 = sqrt(nanmean((r - m)**2)) * sqrt(2/π)
    where r is the residual and m is the channel-wise mean.

    Parameters:
      x (torch.Tensor): The weight tensor.
      mask (torch.Tensor): A binary mask of the same shape as x.
      order (int): The number of residual passes (default 2).

    Returns:
      torch.Tensor: The binarized approximation of x.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    global index
    index += 1
    # Create a NaN tensor for masked-out positions (ensuring device/dtype consistency)
    nan_tensor = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)

    for od in range(order):
        # Compute the current residual and apply the mask
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, nan_tensor)

        # Compute channel-wise mean for bias compensation
        channel_mean = torch.nanmean(masked_x_tensor, dim=1)
        channel_mean = torch.where(torch.isnan(channel_mean), torch.zeros_like(channel_mean), channel_mean)

        # Center the residual by subtracting the channel mean
        centered = masked_x_tensor - channel_mean[:, None]

        # Candidate 2: Scale via variance-based estimator (optimal if residual is Gaussian)
        candidate_std = torch.sqrt(torch.nanmean(centered**2, dim=1))
        candidate_scale2 = candidate_std * math.sqrt(2 / math.pi)
        candidate_scale2 = torch.where(torch.isnan(candidate_scale2), torch.zeros_like(candidate_scale2), candidate_scale2)

        # Use the sign of the centered residual as the binary component
        binary = torch.sign(centered)

        # Compute the final component using candidate 2's scale factor
        final_component = channel_mean[:, None] + candidate_scale2[:, None] * binary

        # Update the accumulated representation (multiplication by mask preserves sparsity)
        sum_order = sum_order + final_component * mask

    return sum_order

@torch.no_grad()
def hybrid_coupled_coordinate_residual(x, mask, order=2, lam=1e-5, corr_damp=0.1):
    """
    Hybrid binarization: runs both the stable coupled two-expansion method and braq,
    then selects the one with the lower quantization error.
    
    For order == 1 (non-salient regions), both methods reduce to a single expansion,
    while for order >= 2 (salient regions), the stable method applies re-centering,
    Tikhonov stabilization, and correlation damping.
    
    Args:
      x (Tensor): (oc, ic) weight matrix.
      mask (Bool Tensor): same shape as x; True indicates valid entries.
      order (int): 
         - 1 => single expansion: w ~ alpha * sign(w - row_mean)
         - >=2 => two expansions: w ~ row_mean + alpha1 * B1 + alpha2 * B2.
      lam (float): Tikhonov (ridge) regularization strength for stability.
      corr_damp (float): Factor in [0,1] to damp positive correlations (c12).
      
    Returns:
      Tensor: The approximate binarized reconstruction chosen from the method
              with the lower quantization error (squared L2 norm over valid entries).
    """
    # Compute approximation using braq (original high_order_residual method)
    approx_braq = high_order_residual(x, mask, order=order)
    
    # Compute approximation using the stable coupled residual binarization v4
    approx_cabr = coupled_residual_binarization_stable_v7(x, mask, order=order, lam=lam, corr_damp=corr_damp)
    
    # Compute the squared quantization error only over valid (masked) entries.
    error_braq = torch.sum(((x - approx_braq) * mask) ** 2)
    error_cabr = torch.sum(((x - approx_cabr) * mask) ** 2)
    
    # Choose the method with the lower error.
    if error_braq <= error_cabr:
        return approx_braq
    else:
        return approx_cabr


@torch.no_grad()
def bit_flip_pass(w, mask, order=2):
    """
    Implements an order-aware bit-flipping binarization technique.
    
    w: (oc, ic) block of weights.
    mask: Boolean mask of valid entries.
    order: 1 for single-pass, 2 for residual-based refinement.
    
    Returns: The binarized weight matrix with optimized bit flips.
    Complexity: O(N).
    """
    # Order = 1 (direct binarization)
    active_w = w[mask]
    if active_w.numel() == 0:
        return torch.zeros_like(w)

    alpha_1 = active_w.abs().mean()
    B_1 = torch.sign(w) * mask
    R_1 = w - alpha_1 * B_1

    # Single-pass bit flipping for order=1
    for row_idx in range(w.shape[0]):
        row_mask = mask[row_idx]
        row_r = R_1[row_idx]
        row_b = B_1[row_idx]

        active_indices = torch.where(row_mask)[0]
        for col_idx in active_indices:
            if row_b[col_idx] > 0 and row_r[col_idx] < -alpha_1:
                row_b[col_idx] = -1.0
                row_r[col_idx] += 2.0 * alpha_1
            elif row_b[col_idx] < 0 and row_r[col_idx] > alpha_1:
                row_b[col_idx] = 1.0
                row_r[col_idx] -= 2.0 * alpha_1

        B_1[row_idx] = row_b
        R_1[row_idx] = row_r

    # If order = 1, return the refined first binarization
    if order == 1:
        return alpha_1 * B_1

    # Order = 2 (residual binarization)
    R_2 = w - alpha_1 * B_1  # First-order residual
    active_r = R_2[mask]
    alpha_2 = active_r.abs().mean() if active_r.numel() > 0 else 0.0

    B_2 = torch.sign(R_2) * mask
    R_2 -= alpha_2 * B_2  # New residual

    # Single-pass bit flipping for order=2
    for row_idx in range(w.shape[0]):
        row_mask = mask[row_idx]
        row_r = R_2[row_idx]
        row_b = B_2[row_idx]

        active_indices = torch.where(row_mask)[0]
        for col_idx in active_indices:
            if row_b[col_idx] > 0 and row_r[col_idx] < -alpha_2:
                row_b[col_idx] = -1.0
                row_r[col_idx] += 2.0 * alpha_2
            elif row_b[col_idx] < 0 and row_r[col_idx] > alpha_2:
                row_b[col_idx] = 1.0
                row_r[col_idx] -= 2.0 * alpha_2

        B_2[row_idx] = row_b
        R_2[row_idx] = row_r

    # Final binarized weight: sum of two binarized components
    return alpha_1 * B_1 + alpha_2 * B_2
@torch.no_grad()
def coupled_residual_binarization_stable_v9(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    'v9' - A minimal single-pass approach that extends braq by a single
    closed-form coupling of alpha1, alpha2 after determining B1,B2.

    - If order=1: w ~ alpha * sign(w)
    - If order>=2: w ~ alpha1 * B1 + alpha2 * B2
      where B1=sign(w), B2=sign(r) with r=(w - alpha1^0*B1),
      then solve alpha1, alpha2 jointly in closed form with Tikhonov (lam)
      and optional correlation damping.

    No iterative refinement or row-mean shifting. This is simpler yet
    typically outperforms braq because alpha1 is re-optimized after
    seeing B2, rather than locked to alpha1^0.

    Args:
      x (Tensor):         (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x
      order (int):        1 => single expansion, >=2 => two expansions
      lam   (float):      Tikhonov ridge for alpha1^2+alpha2^2
      corr_damp(float):   factor in [0,1]; scale down correlation if c12>0

    Returns:
      sum_order(Tensor):  same shape as x; final approximation
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    if order == 1:
        # Single expansion: w ~ alpha * sign(w)
        # This is the same standard approach.
        masked_x = torch.where(mask, new_matrix, torch.tensor(float('nan'), device=x.device))
        scale = torch.nanmean(torch.abs(masked_x), dim=1)  # row-wise
        scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale)
        sign_mat = torch.sign(masked_x)
        sign_mat *= scale[:, None]
        sum_order = sign_mat.where(mask, torch.zeros_like(sign_mat))
        return sum_order

    # Two expansions
    oc, ic = new_matrix.shape

    for row_idx in range(oc):
        row_mask = mask[row_idx, :]
        if not torch.any(row_mask):
            continue

        w_row = new_matrix[row_idx, row_mask]
        d = float(w_row.numel())

        # 1) B1 = sign(w), alpha1^0 = mean(|w|)
        B1 = torch.sign(w_row)
        alpha1_0 = w_row.abs().mean()

        # 2) Residual => B2 = sign(r)
        R = w_row - alpha1_0 * B1
        B2 = torch.sign(R)
        alpha2_0 = R.abs().mean()  # just for reference, not final

        # 3) Solve alpha1, alpha2 in one shot with Tikhonov + correlation damping
        c11 = d + lam  # effectively <B1,B1> + lam
        c22 = d + lam  # effectively <B2,B2> + lam
        c12 = (B1 * B2).sum().item()
        if c12 > 0:
            c12 *= (1.0 - corr_damp)
        c1w = (B1 * w_row).sum().item()
        c2w = (B2 * w_row).sum().item()

        # 2x2 system
        denom = c11 * c22 - c12 * c12
        if abs(denom) < 1e-12:
            # fallback
            alpha1, alpha2 = alpha1_0, alpha2_0
        else:
            alpha1 = ( c1w*c22 - c12*c2w ) / denom
            alpha2 = ( c2w*c11 - c12*c1w ) / denom
            alpha1 = max(alpha1, 0.0)
            alpha2 = max(alpha2, 0.0)

        # 4) Reconstruct final row
        #    w_approx = alpha1*B1 + alpha2*B2
        w_approx = alpha1 * B1 + alpha2 * B2
        sum_order[row_idx, row_mask] = w_approx

    return sum_order
@torch.no_grad()
def coupled_residual_binarization_stable_v10(
    x,
    mask,
    order=2,
    eps=1e-12
):
    """
    A streamlined binarization method with zero offsets and a
    single-pass approach for stability.

    If order=1:
      - w ~ alpha * sign(w), using average magnitude for alpha.

    If order>=2:
      - 1) B1=sign(w), alpha1=mean(|w|)
      - 2) r=w-alpha1*B1, B2=sign(r), alpha2=mean(|r|)
      - 3) 'Scale Sharing': let alphaTotal=mean(|w|) for that row,
         then rescale alpha1' and alpha2' so alpha1'+alpha2'=alphaTotal.

    This ensures neither alpha gets excessively large or vanishingly small,
    while remaining extremely simple (one pass, no offset, no iteration).

    Args:
      x (Tensor):         shape (oc, ic)
      mask (Bool Tensor): shape (oc, ic), True => valid entries
      order (int):        1 => single expansion, >=2 => two expansions
      eps (float):        small constant to avoid divisions by zero

    Returns:
      sum_order (Tensor): same shape as x, binarized approximation
    """
    sum_order = torch.zeros_like(x)
    # We'll operate on a masked copy
    new_matrix = x.clone() * mask

    # We'll do row-by-row. Each row we consider only the "True" positions in mask.
    oc, ic = new_matrix.shape

    if order == 1:
        # Single expansion
        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                continue

            row_vals = new_matrix[row_idx, row_mask]  # all valid positions
            # alpha = average magnitude
            alpha = row_vals.abs().mean()
            # sign
            B = torch.sign(row_vals)
            # final reconstruction
            row_approx = alpha * B
            sum_order[row_idx, row_mask] = row_approx

    else:
        # Two expansions + scale sharing
        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                continue

            row_vals = new_matrix[row_idx, row_mask]
            d = float(row_vals.numel())

            # Step A: B1, alpha1
            B1 = torch.sign(row_vals)
            alpha1 = row_vals.abs().mean()

            # Step B: Residual -> B2, alpha2
            r = row_vals - alpha1 * B1
            B2 = torch.sign(r)
            alpha2 = r.abs().mean()

            # Step C: scale sharing
            # total scale = average magnitude of original row
            alpha_total = row_vals.abs().mean()

            alpha_sum = alpha1 + alpha2
            if alpha_sum < eps:
                # edge case: if alpha1+alpha2 is basically 0, just do alpha_total for alpha1
                alpha1_prime = alpha_total
                alpha2_prime = 0.0
            else:
                alpha1_prime = alpha_total * (alpha1 / alpha_sum)
                alpha2_prime = alpha_total * (alpha2 / alpha_sum)

            # final reconstruction
            row_approx = alpha1_prime * B1 + alpha2_prime * B2
            sum_order[row_idx, row_mask] = row_approx

    return sum_order

@torch.no_grad()
def coupled_residual_binarization_native(x, mask, order=2, coupling=0.5):
    """
    CRB-Native: Float16-native coupled residual binarization with damped
    joint 2×2 alpha solve.

    Addresses three known CRB failure modes:
    1. Float32 promotion: CRB uses mask.float() → float32, causing 11% PPL
       cascade on Qwen3. We use mask.to(x.dtype) and BRAQ's nanmean/torch.where
       code path for native-dtype computation throughout.
    2. Mean structure mismatch: CRB uses single mean + δ correction. We use
       BRAQ's two-step mean accumulation (mean₁ + mean₂).
    3. Alpha inflation: CRB's joint solve inflates α₁ by ~10%, α₂ by ~3%.
       The coupling parameter damps this: α = α_braq + coupling·(α_joint - α_braq).

    Args:
        x: Weight matrix (oc, ic), typically float16.
        mask: Boolean mask for valid columns in this partition.
        order: 1 for non-salient (single expansion), 2 for salient (two expansions).
        coupling: Float in [0, 1]. 0=BRAQ, 1=full joint solve. Default 0.5.

    Returns:
        Quantized approximation, same shape as x.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    nan_val = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)

    # === Order 1: single expansion, identical to BRAQ ===
    if order == 1:
        masked_x = torch.where(mask, new_matrix, nan_val)
        mean_val = torch.nanmean(masked_x, dim=1)
        mean_val = torch.where(torch.isnan(mean_val), torch.zeros_like(mean_val), mean_val)
        masked_x -= mean_val[:, None]
        scale_val = torch.nanmean(torch.abs(masked_x), dim=1)
        scale_val = torch.where(torch.isnan(scale_val), torch.zeros_like(scale_val), scale_val)
        binary = torch.sign(masked_x)
        binary *= scale_val[:, None]
        binary += mean_val[:, None]
        return binary * mask

    # ===== First expansion: BRAQ-exact code path =====
    masked_x = torch.where(mask, new_matrix, nan_val)
    mean1 = torch.nanmean(masked_x, dim=1)
    mean1 = torch.where(torch.isnan(mean1), torch.zeros_like(mean1), mean1)
    masked_x -= mean1[:, None]  # centered, NaN at invalid
    alpha1_braq = torch.nanmean(torch.abs(masked_x), dim=1)
    alpha1_braq = torch.where(torch.isnan(alpha1_braq), torch.zeros_like(alpha1_braq), alpha1_braq)
    B1 = torch.sign(masked_x)  # 0 at NaN positions

    exp1_braq = (B1 * alpha1_braq[:, None] + mean1[:, None]) * mask

    # ===== Second expansion: BRAQ-exact code path =====
    residual = new_matrix - exp1_braq
    masked_r = torch.where(mask, residual, nan_val)
    mean2 = torch.nanmean(masked_r, dim=1)
    mean2 = torch.where(torch.isnan(mean2), torch.zeros_like(mean2), mean2)
    masked_r -= mean2[:, None]  # centered residual, NaN at invalid
    alpha2_braq = torch.nanmean(torch.abs(masked_r), dim=1)
    alpha2_braq = torch.where(torch.isnan(alpha2_braq), torch.zeros_like(alpha2_braq), alpha2_braq)
    B2 = torch.sign(masked_r)

    # If coupling=0, return BRAQ-exact output
    if coupling == 0.0:
        exp2_braq = (B2 * alpha2_braq[:, None] + mean2[:, None]) * mask
        return exp1_braq + exp2_braq

    # ===== Joint 2×2 alpha solve in float32 (precision-critical) =====
    # The solve uses float32 for accurate dot products and division.
    # Signs (B1, B2) and means (mean1, mean2) stay in native dtype (from BRAQ).
    mask_f = mask.float()  # float32 for solve precision
    d = mask_f.sum(dim=1)
    d_safe = torch.clamp(d, min=1.0)

    # Centered weights W̄ = W - mean₁, with 0 at invalid positions
    # Promote to float32 for accurate accumulation
    centered = (new_matrix - mean1[:, None]) * mask_f  # float32

    # Cross-correlation and projections (float32 accumulation)
    c12 = (B1 * B2 * mask_f).sum(dim=1)        # Σ B₁·B₂
    c1w = (centered * B1).sum(dim=1)             # Σ W̄·B₁ = Σ|W̄| ≈ d·α₁_braq
    c2w = (centered * B2).sum(dim=1)             # Σ W̄·B₂

    # Solve: [d, c12; c12, d] · [α₁; α₂] = [c1w; c2w]
    A = d_safe
    denom = A * A - c12 * c12
    # Use a conservative epsilon — small denominators produce unstable alphas
    safe = denom.abs() > 1e-4 * A
    safe_denom = torch.where(safe, denom, torch.ones_like(denom))

    alpha1_joint = torch.clamp((A * c1w - c12 * c2w) / safe_denom, min=0.0)
    alpha2_joint = torch.clamp((A * c2w - c12 * c1w) / safe_denom, min=0.0)
    # Clamp to prevent extreme inflation (max 3x BRAQ)
    alpha1_joint = torch.clamp(alpha1_joint, max=alpha1_braq.float() * 3.0)
    alpha2_joint = torch.clamp(alpha2_joint, max=alpha2_braq.float() * 3.0)
    alpha1_joint = torch.where(safe, alpha1_joint, alpha1_braq.float())
    alpha2_joint = torch.where(safe, alpha2_joint, alpha2_braq.float())
    # Cast back to native dtype
    alpha1_joint = alpha1_joint.to(x.dtype)
    alpha2_joint = alpha2_joint.to(x.dtype)

    # ===== Damped coupling: blend between BRAQ and joint =====
    # Ensure result stays in native dtype (coupling scalar promotes to float32;
    # we must cast back to avoid dtype mismatch in GPTQ cascade)
    alpha1 = (alpha1_braq + coupling * (alpha1_joint - alpha1_braq)).to(x.dtype)
    alpha2 = (alpha2_braq + coupling * (alpha2_joint - alpha2_braq)).to(x.dtype)

    # ===== Reconstruct with BRAQ's two-step mean structure =====
    # Keep mean1 and mean2 from BRAQ (not recomputed) to preserve
    # cascade-friendly error patterns
    exp1 = (B1 * alpha1[:, None] + mean1[:, None]) * mask
    exp2 = (B2 * alpha2[:, None] + mean2[:, None]) * mask
    return exp1 + exp2


@torch.no_grad()
def sdoml_quantize(
    W,
    col_weights,
    sparsity,
    K=4,
    n_iter=20,
    init="quantile",
    return_aux=False,
    strict_mask_change=True,
):
    """Per-row joint sparse + Lloyd-Max quantization (SDOML).

    Faithful realisation of `llmdocs/sdoml/derivation.md` §4. The per-row
    Lagrangian
        Phi = sum_i w_i * (x_i - m_i * a_i)^2
    is minimised jointly over the keep-mask m, the per-row codebook C of K
    levels, and the per-weight assignment a, subject to `sum m_i >= n_keep`
    where `n_keep = int((1-sparsity) * N)`.

    The alternation has two steps per round (derivation §4.1, §4.2):

        Step A (centroid update). For each row r and each Voronoi cell
            V_k = { i : m_i = 1, nearest centroid is c_k },
        update
            c_k = sum_{i in V_k} w_i x_i / sum_{i in V_k} w_i.
        If V_k is empty the centroid is held at its previous value (the
        standard Lloyd-Max convention).

        Step B (mask + assignment update). For each row,
            mu_i = w_i * (x_i^2 - min_c (x_i - c)^2)        [keep-margin]
        is the Hessian-weighted distortion saving from quantising vs pruning.
        The size-n_keep subset with the largest mu_i is kept (sort-based
        threshold per derivation §3.3, equivalent to but faster than the
        Lagrange binary search). Each kept weight is assigned to its nearest
        centroid in the updated codebook.

    Args:
        W:           [R, N] float tensor — row-block of weights.
        col_weights: [N] float tensor — Hessian-derived per-column importance
                     w_i (DOML's `1 / (Hinv_diag^2 + eps)`); strictly positive.
        sparsity:    float in [0, 1) — fraction pruned per row.
        K:           codebook size for kept weights (default 4 = 2 bits).
        n_iter:      number of alternation rounds (default 20, matching DOML).
        init:        codebook initialisation. "quantile" places K centroids at
                     uniform quantiles of each row's kept-by-magnitude weights
                     (matches DOML's Gaussian-quantile init when K=4); "lloyd"
                     uses the Gaussian Lloyd-Max levels scaled by row mean/std.
        return_aux:  if True, also return (mask, codebook, phi_trace).

    Returns:
        W_q ([R, N] float tensor) — dequantised reconstruction with pruned
        positions exactly 0. If `return_aux=True`, additionally returns
            mask     ([R, N] bool tensor),
            codebook ([R, K] float tensor; sorted ascending),
            phi_trace ([n_iter+1] float tensor; Phi at init then after each
                       round; weakly non-increasing).

    Drift guards:
        - AssertionError if Phi ever increases iteration-to-iteration beyond
          a small float-noise tolerance (per derivation §5).
        - AssertionError if the final mask exactly matches the init mask
          (alternation was a no-op — bug).
        - AssertionError if any centroid value is NaN or Inf at any iteration.
    """
    # --- Setup: promote to float32 for the kernel (DOML convention) -------
    orig_dtype = W.dtype
    device = W.device
    W = W.to(torch.float32)
    col_weights = col_weights.to(torch.float32).to(device)

    R, N = W.shape
    assert col_weights.shape == (N,), \
        f"col_weights must be [N={N}], got {tuple(col_weights.shape)}"
    assert 0.0 <= sparsity < 1.0, f"sparsity must be in [0, 1), got {sparsity}"
    assert K >= 2, f"K must be >= 2, got {K}"
    assert (col_weights > 0).all(), "col_weights must be strictly positive"

    n_keep = int((1.0 - sparsity) * N)
    n_keep = max(1, min(N, n_keep))  # at least 1, at most N

    # Per-column weight broadcast to [R, N] for vectorised use
    cw_row = col_weights.unsqueeze(0).expand(R, N)  # [R, N]

    # --- Init mask: warm-start with magnitude * sqrt(w_i) ranking ---------
    # Per derivation §4.5: magnitude warm-start is allowed; Step B then
    # overrides it. The init does NOT determine the final mask.
    sqrt_w = col_weights.sqrt().unsqueeze(0)            # [1, N]
    init_score = (W.abs() * sqrt_w)                      # [R, N]
    # Pick the n_keep largest per row.
    _, init_keep_idx = init_score.topk(n_keep, dim=1, largest=True)  # [R, n_keep]
    init_mask = torch.zeros(R, N, dtype=torch.bool, device=device)
    init_mask.scatter_(1, init_keep_idx, True)
    mask = init_mask.clone()                             # [R, N]

    # --- Init codebook ----------------------------------------------------
    # "quantile" init: place K centroids at uniform quantiles of each row's
    # kept weights. We use the kept-by-magnitude initial set (the warm-start)
    # to compute the quantiles. Quantile points are the (k+1)/(K+1) levels
    # for k=0..K-1, which spreads centroids evenly through the kept mass.
    if init == "quantile":
        # Sort kept weights per row.
        # We have init_keep_idx [R, n_keep]; gather their values and sort.
        kept_vals = W.gather(1, init_keep_idx)           # [R, n_keep]
        kept_vals_sorted, _ = kept_vals.sort(dim=1)      # [R, n_keep]
        # Uniform quantile positions in [0, n_keep-1].
        q_positions = torch.linspace(0.0, 1.0, K + 2, device=device)[1:-1]
        # [K] in (0, 1)
        q_idx = (q_positions * (n_keep - 1)).long().clamp(0, n_keep - 1)  # [K]
        codebook = kept_vals_sorted[:, q_idx]            # [R, K]
    elif init == "lloyd":
        # DOML's Gaussian-quantile init: scale by per-row mean/std on kept set.
        kept_count = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        row_mean = (W * mask.float()).sum(dim=1, keepdim=True) / kept_count
        diff = (W - row_mean) * mask.float()
        row_var = (diff * diff).sum(dim=1, keepdim=True) / kept_count
        row_std = row_var.sqrt().clamp(min=1e-8)
        if K == 4:
            init_pos = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104],
                                     device=device, dtype=torch.float32)
        elif K == 2:
            init_pos = torch.tensor([-0.7979, 0.7979],
                                     device=device, dtype=torch.float32)
        elif K == 3:
            init_pos = torch.tensor([-1.2247, 0.0, 1.2247],
                                     device=device, dtype=torch.float32)
        else:
            init_pos = torch.linspace(-1.5, 1.5, K, device=device,
                                       dtype=torch.float32)
        codebook = row_mean + row_std * init_pos.unsqueeze(0)  # [R, K]
    else:
        raise ValueError(f"unknown init mode: {init!r}")

    # Sort initial codebook ascending so degenerate identical levels collapse
    # to deterministic order.
    codebook, _ = codebook.sort(dim=1)

    # Helper: assign each weight to nearest centroid + recover assigned value.
    def _assign_and_dist():
        # x_e: [R, N, 1], c_e: [R, 1, K]
        x_e = W.unsqueeze(2)
        c_e = codebook.unsqueeze(1)
        d = (x_e - c_e) ** 2                              # [R, N, K]
        a_idx = d.argmin(dim=2)                           # [R, N]
        d_min = d.gather(2, a_idx.unsqueeze(2)).squeeze(2)  # [R, N]
        a_val = codebook.gather(1, a_idx)                 # [R, N]
        return a_idx, a_val, d_min

    # Helper: full Phi over (mask, codebook, assignment).
    def _phi(mask_b, a_val):
        # Phi = sum_i w_i * (x_i - m_i * a_i)^2
        recon = mask_b.float() * a_val
        err = W - recon
        per_w = cw_row * (err * err)
        return per_w.sum().item()

    # Initial assignment + Phi
    a_idx, a_val, _ = _assign_and_dist()
    # For init Phi, kept = mask, assign = nearest in codebook
    phi_trace = [_phi(mask, a_val)]

    eps_increase = 1e-5  # absolute tolerance on Phi increases (float noise)

    for it in range(n_iter):
        # ---------------- Step A: centroid update ------------------------
        # New centroid c_k = sum_{i in V_k} w_i x_i / sum_{i in V_k} w_i.
        # Vectorise across rows and centroids using one-hot per centroid.
        kept = mask                                       # [R, N] bool
        cw_xw = cw_row * W                                 # [R, N]
        new_codebook = codebook.clone()
        for k in range(K):
            cell = kept & (a_idx == k)                    # [R, N] bool
            cell_f = cell.float()
            num = (cw_xw * cell_f).sum(dim=1)             # [R]
            den = (cw_row * cell_f).sum(dim=1)            # [R]
            # Empty cell -> keep previous centroid value (no division by 0).
            has_mass = den > 0
            new_val = torch.where(has_mass, num / den.clamp(min=1e-30),
                                  codebook[:, k])
            new_codebook[:, k] = new_val
        codebook = new_codebook

        # NaN/Inf guard
        assert torch.isfinite(codebook).all(), \
            f"SDOML: non-finite centroid at iter {it+1}: " \
            f"{codebook[~torch.isfinite(codebook)][:8]}"

        # ---------------- Step B: mask + assignment update ---------------
        # Re-assign every weight to its nearest centroid in the new codebook
        # (all N weights, not just kept ones — keep-margin needs nearest dist).
        a_idx, a_val, d_min = _assign_and_dist()

        # Keep-margin mu_i = w_i * (x_i^2 - min_c (x_i - c)^2)
        mu = cw_row * (W * W - d_min)                     # [R, N]

        # Sort-based threshold: keep top n_keep per row by mu.
        _, keep_idx = mu.topk(n_keep, dim=1, largest=True)  # [R, n_keep]
        new_mask = torch.zeros(R, N, dtype=torch.bool, device=device)
        new_mask.scatter_(1, keep_idx, True)
        mask = new_mask

        # ---------------- Phi monotone-decrease check --------------------
        phi_curr = _phi(mask, a_val)
        prev_phi = phi_trace[-1]
        delta = phi_curr - prev_phi
        # Allow tiny float noise — relative or absolute.
        tol = max(eps_increase, 1e-6 * abs(prev_phi))
        assert delta <= tol, (
            f"SDOML: Phi increased at iter {it+1}: prev={prev_phi:.6e} "
            f"curr={phi_curr:.6e} delta={delta:.3e} (tol={tol:.3e}). "
            f"Joint alternation must be monotonically descending per "
            f"derivation §5."
        )
        phi_trace.append(phi_curr)

    # --- Final guards ----------------------------------------------------
    # Mask must have changed from init (else alternation is degenerate).
    # `strict_mask_change=False` callers (e.g. small per-partition groups in
    # sdoml_partition_quantize) may legitimately have init==final because the
    # magnitude warm-start already coincides with the keep-margin optimum on
    # tiny groups (~tens of weights per row).
    if strict_mask_change:
        assert not torch.equal(mask, init_mask), (
            "SDOML: final mask exactly matches init mask. The alternation "
            "produced no update — likely a bug (init too strong, or Step B "
            "degenerate). Per derivation §4.5 the warm-start should be "
            "overridden by Step B."
        )
    assert torch.isfinite(codebook).all(), "SDOML: non-finite centroids at exit"

    # Sort codebook ascending for deterministic output.
    codebook_sorted, sort_perm = codebook.sort(dim=1)
    # Re-assign under sorted codebook so a_val matches.
    codebook = codebook_sorted
    a_idx, a_val, _ = _assign_and_dist()

    # Reconstruct W_q: pruned -> 0, kept -> assigned centroid.
    W_q = mask.float() * a_val
    W_q = W_q.to(orig_dtype)

    if return_aux:
        phi_tensor = torch.tensor(phi_trace, dtype=torch.float32)
        return W_q, mask, codebook, phi_tensor
    return W_q


@torch.no_grad()
def sdoml_partition_quantize(
    W,
    col_weights,
    partition_masks,
    sparsity,
    K=4,
    n_iter=20,
    init="quantile",
    return_aux=False,
    per_partition_sparsity=None,
):
    """SDOML applied independently within each of G structural column partitions.

    Combines DOML's structural partition (`utils.structure.structural_guassian_distribution`)
    with SDOML's joint mask + Lloyd-Max alternation (`sdoml_quantize`).

    Per S8 contract:
      - The column partition assignment (which columns go to which group) is a
        **pre-processing step** done ONCE upfront via `structural_guassian_distribution`
        in `bigptq.fasterquant`. The partition function takes the original block
        statistics and returns G boolean masks over the columns.
      - Within EACH partition g ∈ {0..G-1}, the mask m_g and codebook C_g are
        **jointly** optimised via `sdoml_quantize` over only that partition's
        columns. Different partitions get different (mask, codebook) pairs.
      - The per-partition `sparsity` parameter is the SAME `s` for every group
        (so total keep rate is also (1-s) — verifiable by direct counting).
      - `col_weights` is restricted to the partition's columns: the per-row
        SDOML inside partition g sees `col_weights_g = col_weights[partition_g_cols]`.

    Per S9 asymmetric extension (mandate 2026-05-03):
      - When `per_partition_sparsity` is provided as a list of G floats, EACH
        partition gets its own keep rate. Partitions with `s_g == 0` run the
        DENSE path: `lloyd_max_quantize` (no mask), reusing DOML's per-partition
        Lloyd-Max kernel — no bitmap stored, all positions kept. Partitions
        with `s_g > 0` run `sdoml_quantize` (joint mask + Lloyd-Max).
      - This addresses S8's HONEST-NEGATIVE finding: uniform pruning across
        partitions destroys mask3's salient columns (the very weights DOML
        was designed to preserve). Asymmetric `[s, 0, 0]` (bulk sparse,
        mid+salient dense) preserves DOML's structural protection while
        adding SDOML's joint optimisation to the bulk partition.

    Args:
        W:               [R, N] float tensor — row-block of weights.
        col_weights:     [N] float tensor — Hessian-derived per-column importance
                         (`1 / (Hinv_diag^2 + eps)`); strictly positive.
        partition_masks: [G, R, N] bool tensor — per-element partition assignment
                         from `structural_guassian_distribution`. Each element
                         (r, i) belongs to exactly one partition g (sums along
                         dim=0 give all-True). NOTE: DOML's partition masks are
                         per-element [R, N] bool, not per-column [G, N]. We
                         derive a per-row partition column index by checking
                         which g has True for each (r, i).
        sparsity:        float in [0, 1) — uniform per-partition keep fraction
                         (1-s). Used when `per_partition_sparsity` is None
                         (S8 backward-compat) or for partitions where the
                         per-partition value is None.
        K:               codebook size per partition (default 4 = 2 bits).
        n_iter:          inner SDOML alternation rounds per partition.
        init:            codebook init mode forwarded to `sdoml_quantize`.
        return_aux:      if True, return (mask_full, codebooks, phi_traces).
        per_partition_sparsity: optional list of G floats in [0, 1). If None,
                         defaults to [sparsity] * G (S8 symmetric path).
                         When `s_g == 0`, partition g uses the DENSE path
                         (lloyd_max_quantize, no mask, no bitmap) — S9
                         asymmetric variant per mandate 2026-05-03.

    Returns:
        W_q ([R, N] float tensor) — dequantised reconstruction with pruned
        positions exactly 0. If `return_aux=True`, additionally returns:
            mask_full     ([R, N] bool tensor) — assembled per-row keep mask.
            codebooks     (list of length G of [R_g, K] tensors, one per group;
                           per-group oc count differs because partition_masks is
                           per-element, so per-row column counts vary slightly).
            phi_traces    (list of [n_iter+1] float tensors, one per group).

    Drift guards:
        - AssertionError if the assembled mask's per-row keep rate deviates from
          (1-s) by more than 1% (rate-honesty C4).
        - AssertionError if any partition has zero columns assigned to a row
          (would crash `sdoml_quantize`'s topk; instead we skip that row's
          slot for that partition — but this should not happen with DOML's
          orders=(1,1,2)).
        - Pruned positions in W_q are exactly 0 (mask leak check upstream).
    """
    orig_dtype = W.dtype
    device = W.device
    W_f = W.to(torch.float32)
    col_weights = col_weights.to(torch.float32).to(device)

    R, N = W_f.shape
    G = partition_masks.shape[0]
    assert partition_masks.shape == (G, R, N), (
        f"partition_masks must be [G, R, N]=[{G}, {R}, {N}]; "
        f"got {tuple(partition_masks.shape)}"
    )
    assert col_weights.shape == (N,), (
        f"col_weights must be [N={N}], got {tuple(col_weights.shape)}"
    )
    # Verify that partitions are disjoint and cover everything (C1 of DOML's
    # structural_guassian_distribution). We allow tiny float-noise tolerance
    # by counting bool sums.
    coverage = partition_masks.long().sum(dim=0)            # [R, N]
    assert (coverage == 1).all(), (
        f"sdoml_partition_quantize: partitions must be disjoint and cover all"
        f" elements (each (r, i) belongs to exactly one g). "
        f"coverage stats: min={coverage.min().item()} max={coverage.max().item()}"
    )

    # ---- S9 mandate: per-partition sparsity vector --------------------------
    # When None, default to uniform [sparsity, ..., sparsity] for S8 backward-
    # compat. When a partition g has s_g == 0, that partition uses the DENSE
    # path (lloyd_max_quantize, no mask, no bitmap).
    if per_partition_sparsity is None:
        per_partition_sparsity = [sparsity] * G
    else:
        assert len(per_partition_sparsity) == G, (
            f"per_partition_sparsity must have length G={G}; got "
            f"{len(per_partition_sparsity)}"
        )
        for g, s_g in enumerate(per_partition_sparsity):
            assert 0.0 <= s_g < 1.0, (
                f"per_partition_sparsity[{g}]={s_g} not in [0, 1)"
            )

    # ---- Convert per-element partition masks to per-row column index lists ----
    # DOML's structural_guassian_distribution returns per-element masks because
    # mask3 is column-based (top-up_lim columns by sum-of-magnitudes) and
    # mask1/mask2 split the rest by per-element threshold. So per-row, the
    # partition assignment can vary across rows (mask1 vs mask2 boundary is
    # per-element). We process per-row per-group.
    #
    # For efficiency we batch rows that share the same per-row column
    # assignment. In practice mask3 is the same column set for all rows
    # (set per-block, not per-row), so partition 3 has identical columns
    # across rows. mask1/mask2 may differ per-row.
    #
    # Strategy: for each group g, find the per-row count of True positions.
    # If all rows have the same count for group g, we can vectorise as
    # W_g = W[partition_masks[g]].view(R, n_g) directly. Otherwise we iterate.

    W_q = torch.zeros_like(W_f)
    mask_full = torch.zeros(R, N, dtype=torch.bool, device=device)

    aux_codebooks = []
    aux_phi_traces = []

    for g in range(G):
        m_g = partition_masks[g]                            # [R, N] bool
        s_g = float(per_partition_sparsity[g])

        # ---- S9 dense partition path: s_g == 0, no mask, no SDOML ----------
        # Reuse DOML's per-partition Lloyd-Max (the existing kernel). We
        # additionally extract the per-row codebook from the post-converged
        # `levels` so the column sweep in bigptq can do nearest-centroid
        # snapping per row. lloyd_max_quantize itself does not return levels,
        # so we re-run its inner loop in a thin inlined form.
        if s_g == 0.0:
            # Run Lloyd-Max on this partition's columns ONLY (rest of W is
            # masked to 0 in the call). lloyd_max_quantize takes (x, mask, K,
            # iters) and returns x_q with zeros outside the mask. The kept
            # mask for this partition is m_g itself.
            #
            # We need codebook [R, K] for the bigptq column sweep. Inline
            # the lloyd_max_quantize loop so we capture levels.
            rows, cols = W_f.shape
            mk = m_g  # [R, N] bool
            mk_f = mk.float()
            mask_count = mk_f.sum(dim=1, keepdim=True).clamp(min=1)
            row_mean = (W_f * mk_f).sum(dim=1, keepdim=True) / mask_count
            row_var = ((W_f - row_mean * mk_f) ** 2 * mk_f).sum(
                dim=1, keepdim=True
            ) / mask_count
            row_std = row_var.sqrt().clamp(min=1e-8)

            if K == 4:
                init_pos = torch.tensor(
                    [-1.5104, -0.4528, 0.4528, 1.5104],
                    device=device, dtype=W_f.dtype,
                )
            elif K == 3:
                init_pos = torch.tensor(
                    [-1.2247, 0.0, 1.2247], device=device, dtype=W_f.dtype,
                )
            elif K == 2:
                init_pos = torch.tensor(
                    [-0.7979, 0.7979], device=device, dtype=W_f.dtype,
                )
            else:
                init_pos = torch.linspace(
                    -1.5, 1.5, K, device=device, dtype=W_f.dtype,
                )
            levels = row_mean + row_std * init_pos.unsqueeze(0)  # [R, K]

            masked_x = W_f * mk_f
            for _ in range(n_iter):
                x_e = masked_x.unsqueeze(2)             # [R, N, 1]
                lv_e = levels.unsqueeze(1)              # [R, 1, K]
                dists = (x_e - lv_e) ** 2
                dists = dists + (~mk).unsqueeze(2).float() * 1e30
                assignments = dists.argmin(dim=2)       # [R, N]
                new_levels = torch.zeros_like(levels)
                for k in range(K):
                    k_mask = (assignments == k) & mk
                    k_count = k_mask.float().sum(dim=1).clamp(min=1)
                    k_sum = (masked_x * k_mask.float()).sum(dim=1)
                    new_levels[:, k] = k_sum / k_count
                if torch.allclose(new_levels, levels, atol=1e-6):
                    levels = new_levels
                    break
                levels = new_levels

            # Final: sort levels per row, gather quantized values.
            levels, _ = levels.sort(dim=1)              # [R, K] sorted
            x_e = masked_x.unsqueeze(2)
            lv_e = levels.unsqueeze(1)
            dists = (x_e - lv_e) ** 2
            dists = dists + (~mk).unsqueeze(2).float() * 1e30
            assignments = dists.argmin(dim=2)
            W_q_g = levels.gather(1, assignments) * mk_f  # [R, N]

            W_q = W_q + W_q_g
            mask_full = mask_full | mk     # ALL elements in m_g are kept

            # Aux codebook: store as (sub_idx_all, codebook). We use the
            # same shape (sub_idx_tensor, cb_sub_tensor) as the SDOML
            # same-count branch so the bigptq column-sweep code reads it
            # uniformly. sub_idx covers all R rows.
            sub_idx_all = torch.arange(R, device=device, dtype=torch.long)
            aux_codebooks.append([(sub_idx_all, levels)])
            aux_phi_traces.append(None)  # dense path has no phi trajectory
            continue

        # Per-row column count for this group.
        per_row_count = m_g.long().sum(dim=1)               # [R]
        if per_row_count.numel() == 0:
            continue
        cnt_min = per_row_count.min().item()
        cnt_max = per_row_count.max().item()

        if cnt_min == 0:
            # Degenerate: at least one row has no columns in this partition.
            # Skip that row in this group (its weights are 0 here, so they
            # contribute nothing to W_q). We still need to handle it without
            # crashing sdoml_quantize.
            valid_rows = per_row_count > 0
            if not valid_rows.any():
                aux_codebooks.append(None)
                aux_phi_traces.append(None)
                continue
            # Still need same-count grouping among valid rows.
            R_valid = int(valid_rows.long().sum().item())
            row_counts = per_row_count[valid_rows]
            cnt_min_v = row_counts.min().item()
            cnt_max_v = row_counts.max().item()
            same_count = (cnt_min_v == cnt_max_v)
            valid_idx = torch.nonzero(valid_rows, as_tuple=True)[0]
        else:
            same_count = (cnt_min == cnt_max)
            valid_idx = torch.arange(R, device=device)
            R_valid = R

        if same_count:
            # All (valid) rows have exactly n_g columns in this partition.
            n_g = per_row_count[valid_idx[0]].item() if R_valid > 0 else 0
            if n_g == 0:
                aux_codebooks.append(None)
                aux_phi_traces.append(None)
                continue
            # Gather per-row column indices via topk-on-mask trick: use the bool
            # mask's True positions in row order.
            # Build [R_valid, n_g] index tensor.
            m_g_v = m_g[valid_idx]                          # [R_valid, N]
            # nonzero per row, deterministic.
            # Use sort: trues sort to end. But nonzero(as_tuple=True) gives flat.
            # Instead use argsort on m_g_v.float() descending, take first n_g.
            _, idx_sort = m_g_v.float().sort(dim=1, descending=True, stable=True)
            col_idx = idx_sort[:, :n_g]                     # [R_valid, n_g]

            # Gather W and col_weights for this partition (per-row column subset).
            W_g = W_f[valid_idx].gather(1, col_idx)         # [R_valid, n_g]
            # col_weights is per-column [N]; we need it per (row, position) here
            # because different rows pick different columns. Gather it row-wise.
            cw_full = col_weights.unsqueeze(0).expand(R_valid, N)  # [R_valid, N]
            cw_g = cw_full.gather(1, col_idx)               # [R_valid, n_g]

            # CHALLENGE: sdoml_quantize takes a per-column [n_g] col_weights, not
            # per-(row, position). When per-row column subsets DIFFER, the col-
            # weights interpretation also differs per row. We work around this
            # by running sdoml_quantize per UNIQUE column-subset signature.
            # Hash each row's col_idx and group.
            #
            # Common case (mask3): all rows share the same column subset -> 1 group.
            # Common case (mask1/mask2 with magnitude metric): same column subset
            # within a block (the threshold is per-element on a per-block matrix
            # but the matrix W[:, st:ed] is the entire block so the threshold is
            # element-wise — rows can disagree). So we group by signature.
            #
            # For computational sanity, we group rows by their col_idx signature.
            # For Qwen3-0.6B q_proj this is typically a small number of unique
            # patterns when mask3 is column-based.
            sig = col_idx.cpu().numpy().tobytes()
            # Build a per-row signature hash (cheap if R_valid is small).
            # For speed, compute signatures once and use as dict keys.
            row_sigs = {}
            col_idx_cpu = col_idx.cpu().numpy()
            for ridx in range(R_valid):
                key = col_idx_cpu[ridx].tobytes()
                row_sigs.setdefault(key, []).append(ridx)

            # Per-group SDOML output container, then scatter back.
            per_group_codebooks = []
            per_group_phi = []
            scatter_W_q_g = torch.zeros_like(W_g)           # [R_valid, n_g]
            scatter_mask_g = torch.zeros(R_valid, n_g, dtype=torch.bool, device=device)
            for sig_bytes, row_list in row_sigs.items():
                sub_idx = torch.tensor(row_list, device=device, dtype=torch.long)
                W_sub = W_g[sub_idx]                        # [R_sub, n_g]
                # All rows in this signature share the same col_idx; pull one row's
                # col_idx and gather col_weights once.
                col_idx_sub = col_idx[sub_idx[0]]           # [n_g]
                cw_sub = col_weights[col_idx_sub]           # [n_g]

                W_sub_q, mask_sub, cb_sub, phi_sub = sdoml_quantize(
                    W_sub, cw_sub,
                    sparsity=s_g, K=K, n_iter=n_iter,
                    init=init, return_aux=True,
                    strict_mask_change=False,  # tiny groups may legitimately
                                                # have init==final mask
                )
                scatter_W_q_g[sub_idx] = W_sub_q.to(scatter_W_q_g.dtype)
                scatter_mask_g[sub_idx] = mask_sub
                per_group_codebooks.append((sub_idx, cb_sub))
                per_group_phi.append(phi_sub)

            # Scatter back into the full [R, N] tensor for this partition.
            # We have W_q_g in [R_valid, n_g], need to place at col_idx in row valid_idx.
            W_q_full_for_g = torch.zeros(R, N, dtype=W_f.dtype, device=device)
            mask_full_for_g = torch.zeros(R, N, dtype=torch.bool, device=device)

            W_q_full_for_g[valid_idx.unsqueeze(1).expand(R_valid, n_g),
                           col_idx] = scatter_W_q_g
            mask_full_for_g[valid_idx.unsqueeze(1).expand(R_valid, n_g),
                            col_idx] = scatter_mask_g

            # Combine into the global accumulator. Partitions are disjoint, so
            # summing into W_q is safe (each (r, i) only contributes from one g).
            W_q = W_q + W_q_full_for_g
            mask_full = mask_full | mask_full_for_g

            aux_codebooks.append(per_group_codebooks)
            aux_phi_traces.append(per_group_phi)
        else:
            # Per-row column counts differ — process per row.
            # This is the slow path; for DOML's structural partition with
            # per-block masks it should be hit rarely.
            per_row_codebooks = []
            per_row_phi = []
            for ridx_t in valid_idx:
                ridx = int(ridx_t.item())
                m_row = m_g[ridx]                           # [N] bool
                col_idx_row = torch.nonzero(m_row, as_tuple=True)[0]   # [n_g_row]
                n_g_row = col_idx_row.numel()
                if n_g_row == 0:
                    continue
                W_row = W_f[ridx, col_idx_row].unsqueeze(0)  # [1, n_g_row]
                cw_row = col_weights[col_idx_row]            # [n_g_row]
                W_row_q, mask_row, cb_row, phi_row = sdoml_quantize(
                    W_row, cw_row,
                    sparsity=s_g, K=K, n_iter=n_iter,
                    init=init, return_aux=True,
                    strict_mask_change=False,  # single-row groups may legitimately
                                                # have init==final mask
                )
                W_q[ridx, col_idx_row] = W_row_q[0].to(W_q.dtype)
                mask_full[ridx, col_idx_row] = mask_row[0]
                per_row_codebooks.append((ridx, col_idx_row, cb_row))
                per_row_phi.append(phi_row)
            aux_codebooks.append(per_row_codebooks)
            aux_phi_traces.append(per_row_phi)

    W_q = W_q.to(orig_dtype)

    # Rate-honesty check (C4): the *per-row* keep rate should match the
    # weighted sum sum_g frac_g * (1 - s_g), where frac_g is partition g's
    # column fraction. With asymmetric per-partition sparsity, the global
    # keep rate is no longer (1 - sparsity).
    if R > 0:
        per_row_keep_rate = mask_full.float().sum(dim=1) / N
        avg_keep = per_row_keep_rate.mean().item()
        # Expected keep rate from per-partition sparsities and partition shares.
        partition_shares = partition_masks.float().mean(dim=(1, 2))  # [G]
        target_keep = sum(
            partition_shares[g].item() * (1.0 - per_partition_sparsity[g])
            for g in range(G)
        )
        # Allow 5% tolerance for integer rounding inside each partition's n_keep.
        rel_err = abs(avg_keep - target_keep) / max(target_keep, 1e-6)
        assert rel_err < 0.05, (
            f"sdoml_partition_quantize: per-row keep rate {avg_keep:.4f} differs"
            f" from target {target_keep:.4f} by >{5}%. Each partition's n_keep "
            f"is int((1-s_g)*n_g), so rounding loss is bounded by G/N "
            f"(~{G}/{N} = {G/N:.4f}). per_partition_sparsity="
            f"{per_partition_sparsity}"
        )

    if return_aux:
        return W_q, mask_full, aux_codebooks, aux_phi_traces
    return W_q


@torch.no_grad()
def magfit_quantize(
    W,
    col_weights,
    sparsity,
    K=4,
    n_iter=20,
    return_aux=False,
):
    """Magnitude-prune-then-LMQ baseline (S6 ablation).

    The decoupled "fit once" baseline for SDOML. Per-row recipe:
      Step 1 (mask): keep top-(1-sparsity)*N positions by |x_i|*sqrt(w_i),
                     where w_i is the Hessian-derived per-column weight
                     (`1 / Hinv_diag^2`). This matches the BiLLM-style
                     salience metric and the SDOML warm-start (so the *only*
                     procedural difference vs SDOML is the absence of joint
                     mask + codebook alternation — derivation §7.2).
      Step 2 (codebook): Hessian-weighted Lloyd-Max on the kept positions
                     with K levels, n_iter centroid passes. Centroid update
                     is the weighted mean over each Voronoi cell:
                         c_k = sum_{i in V_k} w_i x_i / sum_{i in V_k} w_i
                     so the comparison vs SDOML isolates the joint-vs-
                     decoupled axis (NOT weighting differences).

    This is structurally separate from `sdoml_quantize` per S6 contract:
    the lit reviewer would object if `magfit` reused the SDOML kernel
    with a flag.

    Args:
        W:           [R, N] float tensor.
        col_weights: [N] float tensor — strictly positive Hessian-derived
                     per-column importance (matches SDOML signature).
        sparsity:    float in [0, 1).
        K:           codebook size for kept weights (default 4 = 2 bits).
        n_iter:      Lloyd-Max iteration count.
        return_aux:  if True, also return (mask, codebook, phi_trace) — for
                     parity with sdoml_quantize's return signature so the
                     bigptq.py dispatch can read the codebook.

    Returns:
        W_q ([R, N] same-dtype tensor) — dequantised reconstruction with
        pruned positions exactly 0. If `return_aux=True`:
            mask     ([R, N] bool tensor),
            codebook ([R, K] float tensor; sorted ascending),
            phi_trace ([n_iter+1] float tensor; Phi after each Lloyd-Max
                       round; weakly non-increasing — provided so the
                       caller can verify weighted-LMQ converged).
    """
    orig_dtype = W.dtype
    device = W.device
    W = W.to(torch.float32)
    col_weights = col_weights.to(torch.float32).to(device)

    R, N = W.shape
    assert col_weights.shape == (N,), \
        f"col_weights must be [N={N}], got {tuple(col_weights.shape)}"
    assert 0.0 <= sparsity < 1.0, f"sparsity must be in [0, 1), got {sparsity}"
    assert K >= 2, f"K must be >= 2, got {K}"
    assert (col_weights > 0).all(), "col_weights must be strictly positive"

    n_keep = int((1.0 - sparsity) * N)
    n_keep = max(1, min(N, n_keep))

    # ---- Step 1: magnitude-prune by |x|*sqrt(w_i) (FROZEN; never updated) --
    sqrt_w = col_weights.sqrt().unsqueeze(0)          # [1, N]
    score = (W.abs() * sqrt_w)                         # [R, N]
    _, keep_idx = score.topk(n_keep, dim=1, largest=True)
    mask = torch.zeros(R, N, dtype=torch.bool, device=device)
    mask.scatter_(1, keep_idx, True)

    # ---- Step 2: Hessian-weighted Lloyd-Max on survivors -------------------
    cw_row = col_weights.unsqueeze(0).expand(R, N)    # [R, N]
    cw_xw = cw_row * W                                 # [R, N]

    # Init centroids at row-quantiles of kept weights (matches SDOML init,
    # so the "joint vs decoupled" comparison is not contaminated by init).
    kept_vals = W.gather(1, keep_idx)                  # [R, n_keep]
    kept_vals_sorted, _ = kept_vals.sort(dim=1)
    q_positions = torch.linspace(0.0, 1.0, K + 2, device=device)[1:-1]
    q_idx = (q_positions * (n_keep - 1)).long().clamp(0, n_keep - 1)
    codebook = kept_vals_sorted[:, q_idx]              # [R, K]
    codebook, _ = codebook.sort(dim=1)

    def _assign_dist():
        x_e = W.unsqueeze(2)
        c_e = codebook.unsqueeze(1)
        d = (x_e - c_e) ** 2
        a_idx = d.argmin(dim=2)
        a_val = codebook.gather(1, a_idx)
        return a_idx, a_val

    def _phi(mask_b, a_val):
        recon = mask_b.float() * a_val
        err = W - recon
        return (cw_row * (err * err)).sum().item()

    a_idx, a_val = _assign_dist()
    phi_trace = [_phi(mask, a_val)]

    for it in range(n_iter):
        # Weighted-mean centroid update on kept positions only.
        new_codebook = codebook.clone()
        for k in range(K):
            cell = mask & (a_idx == k)
            cell_f = cell.float()
            num = (cw_xw * cell_f).sum(dim=1)
            den = (cw_row * cell_f).sum(dim=1)
            has_mass = den > 0
            new_val = torch.where(has_mass, num / den.clamp(min=1e-30),
                                  codebook[:, k])
            new_codebook[:, k] = new_val
        codebook = new_codebook
        a_idx, a_val = _assign_dist()
        phi_trace.append(_phi(mask, a_val))

    # Final sort + reassign
    codebook, _ = codebook.sort(dim=1)
    a_idx, a_val = _assign_dist()

    W_q = mask.float() * a_val
    W_q = W_q.to(orig_dtype)

    if return_aux:
        phi_tensor = torch.tensor(phi_trace, dtype=torch.float32)
        return W_q, mask, codebook, phi_tensor
    return W_q


@torch.no_grad()
def lloyd_max_quantize(x, mask, K=4, iters=20):
    """Distribution-Optimal Multi-Level Quantization (DOML).

    Per-row Lloyd-Max K-level quantizer. Finds K reconstruction levels that
    minimize MSE for each row's weight distribution, then rounds each weight
    to its nearest level.

    Args:
        x: Weight matrix [rows, cols]
        mask: Boolean mask [rows, cols] indicating which columns to quantize
        K: Number of quantization levels (4 = 2 bits)
        iters: Lloyd-Max iterations
    Returns:
        Quantized weight matrix (same shape as x), zero where mask is False.
    """
    rows, cols = x.shape
    result = torch.zeros_like(x)

    # Work only on masked (valid) entries per row.
    # For efficiency, operate on the full matrix with masking.
    masked_x = x * mask.float()

    # Per-row statistics for initialization
    # Use masked values only: compute mean and std per row
    mask_count = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    row_mean = (masked_x.sum(dim=1, keepdim=True)) / mask_count
    row_var = ((masked_x - row_mean * mask.float()) ** 2 * mask.float()).sum(dim=1, keepdim=True) / mask_count
    row_std = row_var.sqrt().clamp(min=1e-8)

    # Initialize K levels per row using Gaussian quantiles (good starting point)
    # For K=4, Gaussian Lloyd-Max levels are at approximately ±0.4528σ, ±1.5104σ
    if K == 4:
        init_positions = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104],
                                       device=x.device, dtype=x.dtype)
    elif K == 3:
        init_positions = torch.tensor([-1.2247, 0.0, 1.2247],
                                       device=x.device, dtype=x.dtype)
    elif K == 2:
        init_positions = torch.tensor([-0.7979, 0.7979],
                                       device=x.device, dtype=x.dtype)
    else:
        # Equal-probability quantile midpoints under unit Gaussian. Lloyd-Max
        # iterations refine these toward the per-row optimum; works for any K.
        quantiles = (torch.arange(K, device=x.device, dtype=torch.float32) + 0.5) / K
        init_positions = torch.distributions.Normal(0.0, 1.0).icdf(quantiles).to(dtype=x.dtype)

    # levels shape: [rows, K]
    levels = row_mean + row_std * init_positions.unsqueeze(0)  # [rows, K]

    # Lloyd-Max iteration: alternate between assignment and centroid update
    for _ in range(iters):
        # Assignment: for each weight, find nearest level
        # x_expanded: [rows, cols, 1], levels_expanded: [rows, 1, K]
        x_expanded = masked_x.unsqueeze(2)         # [rows, cols, 1]
        levels_expanded = levels.unsqueeze(1)       # [rows, 1, K]
        dists = (x_expanded - levels_expanded) ** 2  # [rows, cols, K]

        # For masked-out positions, set distance to inf so they don't affect centroids
        dists = dists + (~mask).unsqueeze(2).float() * 1e30

        assignments = dists.argmin(dim=2)  # [rows, cols] — index into K levels

        # Centroid update: new level = mean of assigned weights
        new_levels = torch.zeros_like(levels)
        for k in range(K):
            k_mask = (assignments == k) & mask  # [rows, cols]
            k_count = k_mask.float().sum(dim=1).clamp(min=1)  # [rows]
            k_sum = (masked_x * k_mask.float()).sum(dim=1)     # [rows]
            new_levels[:, k] = k_sum / k_count

        # Check for convergence
        if torch.allclose(new_levels, levels, atol=1e-6):
            levels = new_levels
            break
        levels = new_levels

    # Sort levels per row (Lloyd-Max may scramble order)
    levels, _ = levels.sort(dim=1)

    # Final assignment: round each weight to nearest level
    x_expanded = masked_x.unsqueeze(2)
    levels_expanded = levels.unsqueeze(1)
    dists = (x_expanded - levels_expanded) ** 2
    dists = dists + (~mask).unsqueeze(2).float() * 1e30
    assignments = dists.argmin(dim=2)  # [rows, cols]

    # Gather quantized values
    result = levels.gather(1, assignments) * mask.float()

    return result


class Binarization(nn.Module):
    def __init__(self, weight, method="2bit", groupsize=-1, corr_damp = 0.1, lam = 1e-5, coupling = 0.5):
        super().__init__()
        oc,ic=weight.shape
        if groupsize==-1:
            groupsize=ic
        self.groupsize=groupsize
        self.n_groups=math.ceil(ic/groupsize)
        self.method=method
        self.mean = 0
        # Add defaults for the (2) mest robust method
        self.kappa = 1.0  # Robustness parameter
        self.order = 2    # Number of residual expansions

        self.corr_damp = corr_damp
        self.lam = lam
        self.coupling = coupling

    def quantize(self, w, mask, order=2, groupi=0, col_weights=None):
        if self.method=="xnor":
            w_mean = self.mean[groupi]
            w = w - w_mean  # oc, ic
            w = w.sign()
            w = w * self.scale[groupi]
            w+=w_mean
        elif self.method=="braq": # The method used in paper
            w = high_order_residual(w, mask, order=order)
        elif self.method=="ternary":
            w = ternary_residual(w, mask, order=order)
        elif self.method=="jrb":  # <-- NEW PROPOSAL
            w = joint_residual_binarization(w, mask, iters=order)
        elif self.method == 'crbog':
            w = coupled_residual_binarization(w, mask, order=order)
        elif self.method=="crb":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization_stable_v7(w, mask, order=order, corr_damp = self.corr_damp, lam = self.lam)
        elif self.method=="crb_norefine":  # CRB with joint alpha but NO sign refinement
            w = coupled_residual_binarization_stable_v7(w, mask, order=order, corr_damp = self.corr_damp, lam = self.lam, skip_refinement=True)
        elif self.method=="crb_symdamp":  # CRB with symmetric correlation damping
            w = coupled_residual_binarization_stable_v7(w, mask, order=order, corr_damp = self.corr_damp, lam = self.lam, symmetric_damp=True)
        elif self.method=="crb_symdamp_norefine":  # CRB with symmetric damping, no refinement
            w = coupled_residual_binarization_stable_v7(w, mask, order=order, corr_damp = self.corr_damp, lam = self.lam, symmetric_damp=True, skip_refinement=True)
        elif self.method=="crb_resrhs":  # CRB with residual-projected RHS (fixes alpha2 deflation)
            w = coupled_residual_binarization_resrhs(w, mask, order=order, corr_damp=self.corr_damp, lam=self.lam)
        elif self.method=="crb_resrhs_norefine":  # CRB resrhs without refinement
            w = coupled_residual_binarization_resrhs(w, mask, order=order, corr_damp=self.corr_damp, lam=self.lam, skip_refinement=True)
        elif self.method=="crb_seqalpha":  # Sequential alphas (BRAQ-style) with CRB sign refinement
            w = coupled_residual_binarization_seqalpha(w, mask, order=order)
        elif self.method=="crb_seqalpha_norefine":  # Sequential alphas, no refinement (should match BRAQ)
            w = coupled_residual_binarization_seqalpha(w, mask, order=order, skip_refinement=True)
        elif self.method=="crb_adaptive":  # Hessian-guided per-row adaptive refinement
            w = coupled_residual_binarization_adaptive(w, mask, order=order, corr_damp=self.corr_damp, lam=self.lam, col_weights=col_weights)
        elif self.method=="crb_hessian":  # Hessian-weighted alpha solve, no refinement
            w = coupled_residual_binarization_hessian(w, mask, order=order, corr_damp=self.corr_damp, lam=self.lam, col_weights=col_weights)
        elif self.method=="crb_native":  # Float16-native CRB with damped joint solve
            w = coupled_residual_binarization_native(w, mask, order=order, coupling=self.coupling)
        elif self.method=="crbv8":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization_stable_v8(w, mask, order=order)
        elif self.method=="crbv9":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization_stable_v9(w, mask, order=order)
        elif self.method=="crbv10":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization_stable_v10(w, mask, order=order)

        elif self.method=="new":  # <-- NEW PROPOSAL
            #w = hybrid_coupled_coordinate_residual(w, mask, order=order)
            w = bit_flip_pass(w, mask, order=order)
        elif self.method == 'ahor':
            w = adaptive_high_order_residual_v2(w,mask,order=order)
        elif self.method=="bhor": # T
            w = balanced_high_order_residual(w, mask, order=order)
        elif self.method=="orb": # Orthogonal Residual Binarization
            w = orthogonal_residual(w, mask, order=order)
        elif self.method=="arb":
            w = attenuated_residual(w, mask, order=order, gamma=0.8)
        elif self.method=="whor": # Weighted High Order Residual
            w = weighted_high_order_residual(w, mask, order=order)
        elif self.method=="robq":  # Our robust varianti (1)
            w = robust_high_order_residual(w, mask, order=order, clamp_factor=2.5)
        elif self.method == "mestrobq":  # New robust method
            w = mest_robust_residual_binarization(w, mask, order=self.order, kappa=self.kappa)
        elif self.method == "medianbraq":  # New robust method
            w = median_high_order_residual(w, mask, order=self.order)
        elif self.method=="sign":
            w=(w>0).float()
            w*=self.scale[groupi]
        elif self.method=="doml":
            # Distribution-Optimal Multi-Level Quantization. K = 2**codebook_bits
            # levels per row via Lloyd-Max; default K=4 (2-bit) when the
            # codebook_K attribute is not set (preserves legacy DOML calls).
            K_doml = int(getattr(self, "codebook_K", 4))
            w = lloyd_max_quantize(w, mask, K=K_doml, iters=20)
        elif self.method=="doml_binary":
            # DOML at K=2: per-row Lloyd-Max optimal binary (1 bit)
            w = lloyd_max_quantize(w, mask, K=2, iters=20)
        elif self.method=="sdoml":
            # Sparse Distribution-Optimal Multi-Level Quantization (SDOML).
            # Joint per-row mask + Lloyd-Max codebook alternation.
            # Sparsity is read from self.sparsity (default 0.5); col_weights
            # is the Hessian-derived per-column importance vector wired in by
            # bigptq.fasterquant — derivation §4 mandates Hessian-weighting.
            sparsity = float(getattr(self, "sparsity", 0.5))
            K_sd = int(getattr(self, "codebook_K", 4))
            n_iter_sd = int(getattr(self, "sdoml_n_iter", 20))
            if col_weights is None:
                # Fallback when caller has not yet wired Hessian weights:
                # uniform col_weights preserves the algorithm but matches
                # unweighted Lloyd-Max (S4 will plumb real weights).
                col_w_local = torch.ones(w.shape[1], device=w.device,
                                          dtype=w.dtype)
            else:
                col_w_local = col_weights.to(w.device)
            w = sdoml_quantize(w, col_w_local, sparsity=sparsity, K=K_sd,
                               n_iter=n_iter_sd, init="quantile")
        elif self.method=="sdoml_partition":
            # SDOML applied independently within each of 3 DOML-style structural
            # column partitions (S8 contract). The actual primary execution path
            # for this method is the `is_sdoml_partition` branch in
            # `bigptq.fasterquant` — this dispatch is a defensive fallback when
            # called outside the GPTQ wrapper (e.g. ad-hoc tests). It uses an
            # uninformed per-row partition assumption (single-partition collapse)
            # so callers should prefer the bigptq path.
            sparsity = float(getattr(self, "sparsity", 0.5))
            K_sd = int(getattr(self, "codebook_K", 4))
            n_iter_sd = int(getattr(self, "sdoml_n_iter", 20))
            if col_weights is None:
                col_w_local = torch.ones(w.shape[1], device=w.device,
                                          dtype=w.dtype)
            else:
                col_w_local = col_weights.to(w.device)
            # Fallback: pretend single partition. The real partition split
            # happens upstream in bigptq.
            w = sdoml_quantize(w, col_w_local, sparsity=sparsity, K=K_sd,
                               n_iter=n_iter_sd, init="quantile")
        elif self.method=="rtn":
            # Simple round-to-nearest binary: sign * mean(|w|) per row
            scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
            w = w.sign() * scale
        elif self.method in ['2bit','3bit','4bit']:

            bits = int(self.method[0])
            dev = w.device
            maxq = torch.tensor(2 ** bits - 1, device=dev)

            # Paper Table 3 reproduction: per-row global scale (computed once
            # on the full weight matrix before the GPTQ block loop). When set
            # by the caller (bigptq.fasterquant), use it instead of recomputing
            # per-block scales. Otherwise fall back to per-row scale computed
            # from the columns given (matches paper Table 7 gs=blocksize).
            if getattr(self, 'global_scale', None) is not None and \
               getattr(self, 'global_zero', None) is not None:
                scale = self.global_scale.to(dev).to(w.dtype)
                zero = self.global_zero.to(dev).to(w.dtype)
            else:
                x = w.clone()
                shape = x.shape  # (oc, ic_block)
                x = x.flatten(1)  # (oc, ic_block)
                tmp = torch.zeros(x.shape[0], device=dev)
                xmin = torch.minimum(x.min(1)[0], tmp)
                xmax = torch.maximum(x.max(1)[0], tmp)
                degenerate = (xmin == 0) & (xmax == 0)
                xmin[degenerate] = -1
                xmax[degenerate] = +1
                scale = (xmax - xmin) / maxq
                zero = torch.round(-xmin / scale)
                shape_bc = [-1] + [1] * (len(shape) - 1)
                scale = scale.reshape(shape_bc)
                zero = zero.reshape(shape_bc)

            w = normal_quantize(w, scale, zero, maxq)

        elif self.method=="prune":
            return torch.zeros_like(w)
        return w
