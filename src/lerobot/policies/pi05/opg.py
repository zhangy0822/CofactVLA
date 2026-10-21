"""Orthogonal Projection Guidance (OPG), shared by training and inference."""

from torch import Tensor


def apply_opg(factual: Tensor, counterfactual: Tensor, scale: float, eps: float = 1e-8) -> Tensor:
    """Amplify the factual velocity perpendicular to the counterfactual velocity."""
    dot = (factual * counterfactual).sum(dim=-1, keepdim=True)
    norm_sq = counterfactual.square().sum(dim=-1, keepdim=True)
    projection = dot / (norm_sq + eps) * counterfactual
    return factual + scale * (factual - projection)
