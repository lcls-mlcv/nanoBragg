"""
The DiffuseModel protocol and the reference models in nanobrag_torch.diffuse.

The models are checked against their closed forms, their gradients against finite
differences (gradcheck in float64), and in a Simulator against the plain-function form of
the hook that tests/test_diffuse_term.py already checks on physics.
"""
import math
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import pytest
import torch
from torch.func import functional_call

from src.nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from src.nanobrag_torch.diffuse import ConstantDiffuse, DiffuseModel, GaussianHalo
from src.nanobrag_torch.models.crystal import Crystal
from src.nanobrag_torch.models.detector import Detector
from src.nanobrag_torch.simulator import Simulator

CRYSTAL = dict(N_cells=(3, 3, 3), default_F=0.0, cell_a=50.0, cell_b=60.0, cell_c=70.0, misset_deg=(10.0, 20.0, 30.0))
DETECTOR = dict(spixels=24, fpixels=24, distance_mm=100.0, pixel_size_mm=1.0)
BEAM = dict(wavelength_A=1.0, fluence=1e24)


def simulator(diffuse, dtype=torch.float64, crystal=None):
    return Simulator(Crystal(CrystalConfig(**{**CRYSTAL, **(crystal or {})}), dtype=dtype),
                     Detector(DetectorConfig(**DETECTOR), dtype=dtype),
                     beam_config=BeamConfig(**BEAM), dtype=dtype, diffuse=diffuse)


def test_models_and_plain_functions_satisfy_the_protocol():
    assert isinstance(ConstantDiffuse(2.0), DiffuseModel)
    assert isinstance(GaussianHalo(2.0, 0.1), DiffuseModel)
    assert isinstance(lambda hkl: hkl[:, 0], DiffuseModel)
    assert not isinstance(3.0, DiffuseModel)


@pytest.mark.parametrize("make", [lambda: ConstantDiffuse(0.0), lambda: GaussianHalo(-1.0, 0.1),
                                  lambda: GaussianHalo(1.0, 0.0)])
def test_non_positive_parameters_are_refused(make):
    with pytest.raises(ValueError, match="positive"):
        make()


def test_constant_is_its_value_everywhere():
    hkl = torch.randn(10, 3, dtype=torch.float64)
    assert torch.allclose(ConstantDiffuse(7.5)(hkl), torch.full((10,), 7.5, dtype=torch.float64), rtol=1e-15)


def test_halo_follows_its_closed_form():
    halo = GaussianHalo(amplitude=4.0, sigma=0.2)
    hkl = torch.tensor([[1.0, -2.0, 3.0], [1.1, -2.0, 3.0], [0.4, 0.3, -0.2], [2.5, 0.0, 0.0]], dtype=torch.float64)
    offset2 = torch.tensor([0.0, 0.01, 0.16 + 0.09 + 0.04, 0.25], dtype=torch.float64)
    expected = 4.0 * torch.exp(-offset2 / (2 * 0.04))
    assert torch.allclose(halo(hkl), expected, rtol=1e-12)


def test_halo_gradcheck_in_hkl_and_parameters():
    halo = GaussianHalo(amplitude=3.0, sigma=0.15)
    # away from half-integers, where round() jumps
    hkl = (torch.randint(-5, 5, (6, 3)) + 0.4 * (torch.rand(6, 3) - 0.5)).to(torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(lambda h: halo(h), (hkl,))

    params = (torch.tensor(math.log(3.0), dtype=torch.float64, requires_grad=True),
              torch.tensor(math.log(0.15), dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(
        lambda a, s: functional_call(halo, {"log_amplitude": a, "log_sigma": s}, (hkl.detach(),)), params)


def test_constant_model_gives_the_image_of_a_constant_function():
    as_module = simulator(ConstantDiffuse(11.0))
    as_function = simulator(lambda hkl: torch.full((len(hkl),), 11.0, dtype=torch.float64))
    assert torch.allclose(as_module.run(oversample=1), as_function.run(oversample=1), rtol=1e-14)


def test_halo_parameters_get_the_simulator_gradient_of_finite_differences():
    halo = GaussianHalo(amplitude=50.0, sigma=0.12)
    sim = simulator(halo)
    sim.run(oversample=1)
    sim.components["diffuse"].sum().backward()
    for name in ["log_amplitude", "log_sigma"]:
        parameter = getattr(halo, name)
        h = 1e-6
        values = []
        for step in (h, -h):
            with torch.no_grad():
                parameter += step
            with torch.no_grad():
                values.append(float(simulator(halo).run(oversample=1).sum()))
            with torch.no_grad():
                parameter -= step
        fd = (values[0] - values[1]) / (2 * h)
        assert float(parameter.grad) == pytest.approx(fd, rel=1e-6), name


def test_halo_carries_the_geometry_gradient_through_hkl():
    """The halo moves with the lattice: the image's derivative in wavelength matches finite differences."""
    halo = GaussianHalo(amplitude=50.0, sigma=0.12)

    def image_sum(wavelength):
        sim = simulator(halo)
        sim.wavelength = wavelength
        sim.run(oversample=1)
        return sim.components["diffuse"].sum()
    wavelength = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    image_sum(wavelength).backward()
    h = 1e-6
    fd = (image_sum(torch.tensor(1.0 + h, dtype=torch.float64))
          - image_sum(torch.tensor(1.0 - h, dtype=torch.float64))) / (2 * h)
    assert float(wavelength.grad) != 0.0
    assert float(wavelength.grad) == pytest.approx(float(fd), rel=1e-5)


def test_models_run_in_float32_simulators():
    sim = simulator(GaussianHalo(5.0, 0.2), dtype=torch.float32)
    image = sim.run(oversample=1)
    assert image.dtype == torch.float32 and torch.isfinite(image).all() and image.max() > 0
