"""rot is C's single precision unless the mosaic spread is being refined."""

import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator
from nanobrag_torch.utils.c_random import mosaic_rotation_umats

SPREAD_DEG = 0.057


def image_sum(spread, dtype=torch.float64):
    crystal = Crystal(CrystalConfig(cell_a=70, cell_b=80, cell_c=90, N_cells=(5, 5, 5), default_F=100,
                                    mosaic_spread_deg=spread, mosaic_domains=5, mosaic_seed=7), dtype=dtype)
    detector = Detector(DetectorConfig(distance_mm=100, pixel_size_mm=0.1, spixels=16, fpixels=16, oversample=1),
                        dtype=dtype)
    return Simulator(crystal, detector, beam_config=BeamConfig(wavelength_A=1.0, fluence=1e24), dtype=dtype).run().sum()


def test_a_refined_spread_is_smooth_below_float32s_step():
    leaf = torch.tensor(SPREAD_DEG, dtype=torch.float64, requires_grad=True)
    image_sum(leaf).backward()

    def central_difference(h):
        high, low = (image_sum(torch.tensor(SPREAD_DEG + d, dtype=torch.float64, requires_grad=True)).item()
                     for d in (h, -h))
        return (high - low) / (2 * h)

    assert central_difference(1e-9) == pytest.approx(leaf.grad.item(), rel=0.01)
    assert central_difference(1e-11) != 0


def test_a_line_search_under_no_grad_sees_the_same_value():
    leaf = torch.tensor(SPREAD_DEG, dtype=torch.float64, requires_grad=True)
    value = image_sum(leaf).item()
    with torch.no_grad():
        assert image_sum(leaf).item() == value


def test_a_derived_spread_gives_the_same_image_under_grad_and_no_grad():
    """0.057*exp(p) requires grad only in grad mode; under no_grad nothing marks it as refined."""
    p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    value = image_sum(SPREAD_DEG * torch.exp(p)).item()
    with torch.no_grad():
        assert image_sum(SPREAD_DEG * torch.exp(p)).item() == value


def test_a_float_spread_keeps_cs_rounding():
    """The CLI and compat.cctbx give the spread as a float, in both dtypes: PARITY-RANDOM-001."""
    radians = torch.deg2rad(torch.tensor(SPREAD_DEG, dtype=torch.float64))
    rounded = mosaic_rotation_umats(radians, 5, seed=7, dtype=torch.float64)
    crystal = Crystal(CrystalConfig(mosaic_spread_deg=SPREAD_DEG, mosaic_domains=5, mosaic_seed=7), dtype=torch.float64)
    assert torch.equal(crystal._generate_mosaic_rotations(crystal.config), rounded)


def test_a_tensor_spread_keeps_the_dtype():
    """Whether or not it requires grad: a tensor is how a refined spread arrives, also under no_grad."""
    radians = torch.deg2rad(torch.tensor(SPREAD_DEG, dtype=torch.float64))
    exact = mosaic_rotation_umats(radians, 5, seed=7, dtype=torch.float64, single_precision_rot=False)
    tensor_spread = torch.tensor(SPREAD_DEG, dtype=torch.float64)
    crystal = Crystal(CrystalConfig(mosaic_spread_deg=tensor_spread, mosaic_domains=5, mosaic_seed=7), dtype=torch.float64)
    assert torch.equal(crystal._generate_mosaic_rotations(crystal.config), exact)


def test_the_two_differ_by_a_float32_rounding_at_most():
    radians = torch.deg2rad(torch.tensor(SPREAD_DEG, dtype=torch.float64))
    rounded = mosaic_rotation_umats(radians, 5, seed=7, dtype=torch.float64)
    exact = mosaic_rotation_umats(radians, 5, seed=7, dtype=torch.float64, single_precision_rot=False)
    assert not torch.equal(rounded, exact)
    assert (rounded - exact).abs().max() < 1e-9


def test_gradient_of_the_rounded_path_is_unchanged():
    radians = torch.deg2rad(torch.tensor(SPREAD_DEG, dtype=torch.float64)).requires_grad_(True)
    rounded = mosaic_rotation_umats(radians, 5, seed=7, dtype=torch.float64)
    exact_input = radians.detach().clone().requires_grad_(True)
    exact = mosaic_rotation_umats(exact_input, 5, seed=7, dtype=torch.float64, single_precision_rot=False)
    rounded.sum().backward()
    exact.sum().backward()
    assert radians.grad.item() == pytest.approx(exact_input.grad.item(), rel=1e-6)
