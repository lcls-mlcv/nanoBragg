"""Simulator.fluence and r_e_sqr are live state, and the scale's double precision survives."""

import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import R_E_SQR, Simulator


def make(dtype=torch.float64, fluence=1e24):
    crystal = Crystal(CrystalConfig(cell_a=70, cell_b=80, cell_c=90, N_cells=(5, 5, 5), default_F=100),
                      dtype=dtype)
    detector = Detector(DetectorConfig(distance_mm=100, pixel_size_mm=0.1, spixels=16, fpixels=16),
                        dtype=dtype)
    return Simulator(crystal, detector, beam_config=BeamConfig(wavelength_A=1.0, fluence=fluence),
                     dtype=dtype)


CHANGES = {
    "assign_float": lambda s: setattr(s, "fluence", 2e24),
    "assign_tensor": lambda s: setattr(s, "fluence", torch.tensor(2e24, dtype=s.dtype)),
    "imul": lambda s: s.fluence.__imul__(2),
    "augmented": lambda s: setattr(s, "fluence", s.fluence.__imul__(2)),
    "fill": lambda s: s.fluence.fill_(2e24),
    "data_fill": lambda s: s.fluence.data.fill_(2e24),
    "data_assign": lambda s: setattr(s.fluence, "data", torch.tensor(2e24, dtype=s.dtype)),
    "r_e_sqr_assign": lambda s: setattr(s, "r_e_sqr", 2 * s.r_e_sqr),
    "r_e_sqr_float": lambda s: setattr(s, "r_e_sqr", 2 * R_E_SQR),
    "r_e_sqr_fill": lambda s: s.r_e_sqr.fill_(2 * R_E_SQR),
    "r_e_sqr_data": lambda s: s.r_e_sqr.data.fill_(2 * R_E_SQR),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("change", sorted(CHANGES))
def test_scale_state_changes_the_image(change, dtype):
    base = make(dtype).run().to(torch.float64).sum().item()
    assert base > 0
    sim = make(dtype)
    CHANGES[change](sim)
    assert sim.run().to(torch.float64).sum().item() / base == pytest.approx(2.0, rel=2e-6)


def test_setter_keeps_the_digits_float32_loses():
    """Run in float32, the scale still comes from the double value given to it."""
    fluence = 1.0000000123e24
    sim = make(torch.float32)
    sim.fluence = fluence
    assert sim._r_e_sqr_fluence().dtype == torch.float64
    assert sim._r_e_sqr_fluence().item() == R_E_SQR * fluence
    constructed = make(torch.float32, fluence=fluence)
    assert constructed._r_e_sqr_fluence().item() == R_E_SQR * fluence


def test_scale_stays_finite_where_the_product_leaves_float32():
    """r_e^2 alone is 7.9e-30; a low fluence puts the product below float32's range."""
    sim = make(torch.float32, fluence=1e-17)
    assert sim._r_e_sqr_fluence().item() == R_E_SQR * 1e-17
    assert sim.run().sum().item() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_gradient_reaches_a_leaf_made_from_the_public_tensor(dtype):
    sim = make(dtype)
    sim.fluence.requires_grad_(True)
    image = sim.run()
    image.sum().backward()
    assert sim.fluence.grad.item() == pytest.approx(image.sum().item() / 1e24, rel=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_gradient_reaches_an_assigned_leaf_and_a_filled_one(dtype):
    sim = make(dtype)
    leaf = torch.tensor(1e24, dtype=dtype, requires_grad=True)
    sim.fluence = leaf
    image = sim.run()
    image.sum().backward()
    assert leaf.grad.item() == pytest.approx(image.sum().item() / 1e24, rel=1e-5)

    leaf.grad = None
    leaf.data.fill_(2e24)
    doubled = sim.run()
    doubled.sum().backward()
    assert doubled.sum().item() / image.sum().item() == pytest.approx(2.0, rel=2e-6)
    assert leaf.grad.item() == pytest.approx(image.sum().item() / 1e24, rel=1e-5)


def test_gradient_reaches_a_config_leaf_in_double():
    leaf = torch.tensor(1e24, dtype=torch.float64, requires_grad=True)
    sim = make(torch.float32, fluence=leaf)
    image = sim.run()
    image.to(torch.float64).sum().backward()
    assert leaf.grad.item() == pytest.approx(image.to(torch.float64).sum().item() / 1e24, rel=1e-6)


def test_gradient_reaches_an_r_e_sqr_leaf():
    sim = make(torch.float64)
    sim.r_e_sqr = torch.tensor(R_E_SQR, dtype=torch.float64, requires_grad=True)
    image = sim.run()
    image.sum().backward()
    assert sim.r_e_sqr.grad.item() == pytest.approx(image.sum().item() / R_E_SQR, rel=1e-12)
