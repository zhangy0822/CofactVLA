"""Counterfactual Covariance Reduction (CCR) for inference-time KV features."""

import torch
from torch import Tensor


def apply_ccr(
    features: Tensor,
    batch_size: int,
    pad_masks: Tensor,
    beta: float,
    top_k: int,
    eps: float,
) -> Tensor:
    """Suppress positive covariance-difference directions in paired KV features.

    features: [2 * batch_size, KV heads, tokens, head_dim], factual branch first.
    Covariances use positions valid in both branches; masked language/padding
    cannot contribute spurious counterfactual features. Projection is applied to
    valid positions of each branch. Eigensolves run in float32.
    """
    total, heads, tokens, dim = features.shape
    if total != 2 * batch_size or tuple(pad_masks.shape) != (total, tokens):
        raise ValueError("CCR requires paired factual/counterfactual batches and matching token masks")
    if not 0 <= beta <= 1 or top_k < 1 or eps <= 0:
        raise ValueError("CCR requires beta in [0,1], top_k >= 1, and eps > 0")
    if beta == 0:
        return features
    shared = pad_masks[:batch_size].bool() & pad_masks[batch_size:].bool()
    row_mask = shared[:, None, :].expand(batch_size, heads, tokens).reshape(-1)
    factual = features[:batch_size].reshape(-1, dim)[row_mask].float()
    counterfactual = features[batch_size:].reshape(-1, dim)[row_mask].float()
    if factual.shape[0] < 2:
        return features
    factual = factual - factual.mean(dim=0, keepdim=True)
    counterfactual = counterfactual - counterfactual.mean(dim=0, keepdim=True)
    delta = (counterfactual.T @ counterfactual - factual.T @ factual) / (factual.shape[0] - 1)
    delta = (delta + delta.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(delta)
    positive = eigenvectors[:, eigenvalues > eps]
    if positive.shape[1] == 0:
        return features
    basis = positive[:, -min(top_k, positive.shape[1]) :]
    full_mask = pad_masks.bool()[:, None, :].expand(total, heads, tokens).reshape(-1)
    # KV tensors are transposed and may be non-contiguous; do not use view().
    cleaned = features.reshape(-1, dim).clone()
    valid = cleaned[full_mask].float()
    cleaned[full_mask] = (valid - beta * ((valid @ basis) @ basis.T)).to(features.dtype)
    return cleaned.reshape(features.shape)
