import torch
from utils.autosearch import structural_searching
from utils.mask import generate_structural_mask

'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''
def structural_guassian_distribution(tmp, H=None, metric="magnitude", up_lim=30, orders=(1,1,2), col_scale=None):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    elif metric == "actmag":
        # CRB_SALIENT_METRIC=actmag (2026-07-23): activation-scaled magnitude
        # SELECTION only. Identical to "magnitude" except the per-column
        # salient RANKING inside structural_searching becomes
        # s_j * sum_i |W_ij|, where s = a**alpha (geo-mean-normalized AWQ
        # scale, see kernels/pack/awq_transform.py) is passed via `col_scale`.
        # Weights and norms are NOT modified. When col_scale is None (linears
        # outside the AWQ v1 norm-group scope, e.g. o_proj/down_proj), this
        # falls back to the plain magnitude score.
        target_weights = tmp
    else:
        raise NotImplementedError

    if metric == "actmag" and col_scale is not None:
        optimal_split, mask3 = structural_searching(
            target_weights, up_lim, orders=orders, col_scale=col_scale)
    else:
        # NOTE: keep this call byte-identical to the pre-actmag code — the
        # K2 dump path (kernels/pack/doml_dump.py:_sgd_wrapper) and other
        # monkeypatch wrappers only accept the original signature.
        optimal_split, mask3 = structural_searching(target_weights, up_lim, orders=orders)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    return mask1, mask2, mask3


def actmag_col_scale(layer, st, ed, metric):
    """CRB_SALIENT_METRIC=actmag helper: column-slice [st:ed) of the AWQ
    activation scale stashed on `layer` (a plain `_crb_actmag_s` attribute set
    by kernels/pack/awq_transform.attach_selection_scales; full in_features
    length, float32, CPU). Returns None — i.e. the caller must NOT pass a
    col_scale kwarg, keeping the legacy call byte-identical — unless
    metric == "actmag" AND this linear is covered by the AWQ v1 norm-group
    scope (q/k/v/gate/up; o_proj and down_proj have no stash and fall back to
    the plain magnitude ranking)."""
    if metric != "actmag":
        return None
    s = getattr(layer, "_crb_actmag_s", None)
    if s is None:
        return None
    return s[st:ed]
