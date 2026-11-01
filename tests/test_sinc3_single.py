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


def test_float64_value_is_the_closed_form_down_to_1e_6():
    x = torch.logspace(-5.99, 2, 4000, dtype=torch.float64)
    assert torch.equal(sinc3(x), 3.0 * (torch.sin(x) / x - torch.cos(x)) / (x * x))


@pytest.mark.parametrize("x", [0.0, 1e-300, 1e-12, 1e-7])
def test_float64_below_1e_6_is_one_not_the_cancelled_bracket(x):
    assert sinc3(torch.tensor([x], dtype=torch.float64)).item() == pytest.approx(1.0, abs=2e-15)


@pytest.mark.parametrize("x", [0.0, 1e-300, 1e-9, 1e-6, 1e-5, 1e-3, 0.0099, 0.0101, 0.1])
def test_float64_gradient_is_minus_x_over_5_through_the_cancellation(x):
    leaf = torch.tensor([x], dtype=torch.float64, requires_grad=True)
    sinc3(leaf).sum().backward()
    assert leaf.grad.item() == pytest.approx(-x / 5 + x ** 3 / 70, rel=1e-6, abs=1e-300)


def test_round_shape_has_a_gradient_at_the_centre_of_a_peak():
    from nanobrag_torch.config import BeamConfig, CrystalConfig, CrystalShape, DetectorConfig
    from nanobrag_torch.models.crystal import Crystal
    from nanobrag_torch.models.detector import Detector
    from nanobrag_torch.simulator import Simulator

    cell = torch.tensor(70.0, dtype=torch.float64, requires_grad=True)
    crystal = Crystal(CrystalConfig(cell_a=cell, cell_b=80, cell_c=90, N_cells=(3, 3, 3), default_F=100,
                                    shape=CrystalShape.ROUND), dtype=torch.float64)
    # Odd detector: the middle pixel is on the beam, where hkl = 000 and hrad_sqr is 0
    detector = Detector(DetectorConfig(distance_mm=100, pixel_size_mm=0.1, spixels=33, fpixels=33, oversample=1),
                        dtype=torch.float64)
    simulator = Simulator(crystal, detector, beam_config=BeamConfig(wavelength_A=1.0, fluence=1e24),
                          dtype=torch.float64)
    simulator.run(oversample=1).sum().backward()
    assert torch.isfinite(cell.grad) and cell.grad != 0
