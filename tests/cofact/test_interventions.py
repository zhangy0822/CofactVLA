"""Small CPU checks of projection invariants; no weights, CUDA or robot required."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
def load_operation(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"src/lerobot/policies/pi05/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


opg = load_operation("opg")
ccr = load_operation("ccr")


def test_opg_preserves_parallel_component_and_amplifies_orthogonal_component():
    factual = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    reference = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    result = opg.apply_opg(factual, reference, 2.0)
    torch.testing.assert_close(result, torch.tensor([[[3.0, 12.0]]]))
    result.square().sum().backward()
    assert torch.isfinite(factual.grad).all() and torch.isfinite(reference.grad).all()


def test_opg_zero_reference_stays_finite():
    x = torch.randn(2, 3, 7)
    torch.testing.assert_close(opg.apply_opg(x, torch.zeros_like(x), 2), 3 * x)


def test_ccr_suppresses_positive_direction_and_preserves_other_axis():
    factual = torch.tensor([[[-1.0, -2.0], [1.0, 2.0], [-1.0, 2.0], [1.0, -2.0]]])
    counterfactual = factual * torch.tensor([2.0, 1.0])
    features = torch.stack([factual, counterfactual])
    masks = torch.ones(2, 4, dtype=torch.bool)
    result = ccr.apply_ccr(features, 1, masks, 0.25, 4, 1e-5)
    torch.testing.assert_close(result[..., 0], features[..., 0] * 0.75)
    torch.testing.assert_close(result[..., 1], features[..., 1])


def test_ccr_no_positive_spectrum_is_identity():
    x = torch.randn(1, 2, 5, 4)
    features = torch.cat([x, x * 0.5])
    result = ccr.apply_ccr(features, 1, torch.ones(2, 5).bool(), 0.15, 4, 1e-5)
    torch.testing.assert_close(result, features)


def test_ccr_ignores_masked_counterfactual_tokens_and_handles_noncontiguous_input():
    torch.manual_seed(7)
    features = torch.randn(2, 5, 2, 4).transpose(1, 2)
    masks = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])
    result = ccr.apply_ccr(features, 1, masks, 0.15, 4, 1e-5)
    changed = features.clone()
    changed[1, :, 2:] = 1e6
    result_changed = ccr.apply_ccr(changed, 1, masks, 0.15, 4, 1e-5)
    torch.testing.assert_close(result[0], result_changed[0])
    torch.testing.assert_close(result[:, :, 4], features[:, :, 4])
    assert not features.is_contiguous()
    assert torch.isfinite(result).all()


def test_ccr_small_batch_and_invalid_shape():
    features = torch.randn(2, 1, 1, 4)
    masks = torch.ones(2, 1).bool()
    torch.testing.assert_close(
        ccr.apply_ccr(features, 1, masks, 0.15, 4, 1e-5), features
    )
    with pytest.raises(ValueError):
        ccr.apply_ccr(features, 2, masks, 0.15, 4, 1e-5)
