"""sinc3 keeps its digits in float32, where the closed form cancels near x = 0."""

import pytest
import torch

from nanobrag_torch.utils.physics import _sinc3_series, sinc3

X = torch.cat([torch.logspace(-4, 0.5, 2000, dtype=torch.float64), torch.linspace(3.0, 40.0, 4000, dtype=torch.float64)])


@pytest.mark.parametrize("function", [sinc3, _sinc3_series], ids=["sinc3", "series_for_mps"])
def test_float32_matches_float64(function):
    expected = sinc3(X)
    single = function(X.to(torch.float32)).to(torch.float64)
    # Against the double value at the double x, so at large x the error is that of the argument
    assert (single - expected).abs().max().item() < 5e-7
    small = X < 1.5
    assert ((single - expected) / expected)[small].abs().max().item() < 5e-7
    assert function(X.to(torch.float32)).dtype == torch.float32


def test_series_and_closed_form_meet():
    below, above = torch.tensor([1.4999999], dtype=torch.float32), torch.tensor([1.5000001], dtype=torch.float32)
    assert _sinc3_series(below).item() == pytest.approx(_sinc3_series(above).item(), abs=3e-7)


@pytest.mark.parametrize("function", [sinc3, _sinc3_series], ids=["sinc3", "series_for_mps"])
def test_float32_gradient_is_right(function):
    x = torch.tensor([0.01, 0.1, 1.0, 1.4, 1.6, 5.0], dtype=torch.float32, requires_grad=True)
    function(x).sum().backward()
    exact = x.detach().to(torch.float64).requires_grad_(True)
    sinc3(exact).sum().backward()
    assert x.grad.to(torch.float64).numpy() == pytest.approx(exact.grad.numpy(), rel=1e-4)


def test_series_at_zero():
    x = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    value = _sinc3_series(x)
    value.sum().backward()
    assert value.item() == 1.0 and x.grad.item() == 0.0
