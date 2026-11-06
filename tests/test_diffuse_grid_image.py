"""DiffuseGrid in images (plan D3): grid vs exact, gradients through the full image, ExperimentModel groups.

Grid vs exact: the image is a positive linear functional of the scattering per cell, so the
trilinear error bound per cell, (step^2 / 8) times the sum over axes of max |d^2 I / dh_i^2|,
bounds the image error by that number times the image of a constant 1. The bounds are a theorem,
not a fitted tolerance; the measured errors (scripts/diffuse/grid_vs_exact.py) sit below them.

Gradients: CPU float64, 16 x 16, the crystal with F = 0 so that every geometry gradient comes
through the grid's interpolation. Trilinear interpolation is piecewise linear in hkl, so each
finite-difference check first asserts that no sample changes grid cell between its two points.
"""
import math
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import numpy as np
import pytest
import torch
from torch import nn
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot, SpotMetric,
)
from nanobrag_torch.diffuse import ConstantDiffuse, DiffuseGrid, GaussianHalo
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel, SerialExperiment
from nanobrag_torch.simulator import Simulator

DTYPE = torch.float64
MMM = np.array([np.diag(d) for d in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))])  # 222; Laue mmm
CELL = (30.0, 35.0, 40.0)


def _crystal(**kwargs):
    base = dict(cell_a=CELL[0], cell_b=CELL[1], cell_c=CELL[2], misset_deg=(5.0, -3.0, 2.0), N_cells=(5, 5, 5),
                default_F=0.0, shape=CrystalShape.GAUSS, spot_metric=SpotMetric.HKL)
    return CrystalConfig(**{**base, **kwargs})


def _detector(pixels, **kwargs):
    base = dict(distance_mm=60.0, pixel_size_mm=12.8 / pixels, spixels=pixels, fpixels=pixels, oversample=1,
                detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)
    return DetectorConfig(**{**base, **kwargs})


def _image(diffuse, pixels=24, oversample=1):
    sim = Simulator(Crystal(_crystal(), dtype=DTYPE), Detector(_detector(pixels), dtype=DTYPE),
                    beam_config=BeamConfig(wavelength_A=1.2, fluence=1e24), dtype=DTYPE, diffuse=diffuse)
    with torch.no_grad():
        sim.run(oversample=oversample)
    return sim.components["diffuse"]


D_MIN = 7.0  # the 12.8 mm detector at 60 mm reaches 8.1 A at 1.2 A


class Smooth(nn.Module):
    """A smooth, positive, symmetric (mmm) model: A (2 + cos(2 pi h / 3) cos(2 pi k / 4) cos(2 pi l / 5))."""

    periods = (3.0, 4.0, 5.0)

    def __init__(self, amplitude=10.0):
        super().__init__()
        self.amplitude = amplitude

    def forward(self, hkl):
        waves = [torch.cos(2 * math.pi * hkl[:, i] / p) for i, p in enumerate(self.periods)]
        return self.amplitude * (2 + waves[0] * waves[1] * waves[2])

    def curvature(self):  # sum over axes of max |d^2 I / dh_i^2|
        return self.amplitude * sum((2 * math.pi / p) ** 2 for p in self.periods)


@pytest.mark.parametrize("sigma,step", [(0.3, 1 / 4), (0.3, 1 / 8), (0.15, 1 / 8)])
def test_grid_image_is_within_the_trilinear_bound_of_the_exact_halo(sigma, step):
    halo = GaussianHalo(100.0, sigma)
    exact = _image(halo)
    grid = DiffuseGrid.from_model(halo, step=step, d_min=D_MIN, cell=CELL, point_group=MMM, refinable=False)
    error = (_image(grid) - exact).abs()
    # per axis max |d^2/dh^2| of a Gaussian is A / sigma^2; the nodes lie on the half-integer kinks
    bound = 3 * step ** 2 / 8 * 100.0 / sigma ** 2 * _image(ConstantDiffuse(1.0))
    assert exact.max() > 0 and (error <= bound).all(), float((error / bound).max())
    assert float(error.max()) > 0  # not exact: the measurement is not vacuous


def test_grid_image_of_a_smooth_model_is_within_its_bound_and_converges_as_step_squared():
    model = Smooth()
    exact = _image(model)
    errors = []
    for step in (1 / 2, 1 / 4):
        grid = DiffuseGrid.from_model(model, step=step, d_min=D_MIN, cell=CELL, point_group=MMM, refinable=False)
        error = (_image(grid) - exact).abs()
        assert (error <= step ** 2 / 8 * model.curvature() * _image(ConstantDiffuse(1.0))).all()
        errors.append(float(error.max()))
    assert 3.0 < errors[0] / errors[1] < 5.0  # O(step^2)


def test_a_grid_too_small_for_the_detector_is_zero_there_or_raises():
    halo = GaussianHalo(100.0, 0.3)
    small = DiffuseGrid.from_model(halo, step=0.25, extent=(1.0, 1.0, 1.0), refinable=False)
    image = _image(small)
    assert (image == 0).any() and (image > 0).any()
    strict = DiffuseGrid.from_model(halo, step=0.25, extent=(1.0, 1.0, 1.0), refinable=False, outside="error")
    with pytest.raises(ValueError, match="outside the grid"):
        _image(strict)


def test_oversampled_and_symmetric_images_match_p1():
    halo = GaussianHalo(100.0, 0.3)
    p1 = DiffuseGrid.from_model(halo, step=1 / 4, d_min=D_MIN, cell=CELL, refinable=False)
    asu = DiffuseGrid.from_model(halo, step=1 / 4, d_min=D_MIN, cell=CELL, point_group=MMM, refinable=False)
    torch.testing.assert_close(_image(asu, pixels=12, oversample=2), _image(p1, pixels=12, oversample=2),
                               rtol=1e-13, atol=0)


# Gradients through the full image.

P = 16
GRID = "diffuse_params.model"
GEOMETRY = [
    ("crystal_params.delta_misset_raw", 0), ("crystal_params.delta_misset_raw", 2),
    ("crystal_params.delta_log_a", None), ("crystal_params.delta_log_c", None),
    ("detector_params.delta_log_distance_mm", None), ("beam_params.delta_log_lambda_scale", None),
]


def _halo_grid(refinable=False):
    return DiffuseGrid.from_model(GaussianHalo(40.0, 0.15), step=1 / 8, d_min=6.0, cell=CELL, point_group=MMM,
                                  refinable=refinable)


def _model(grid, **kwargs):
    model = ExperimentModel(_crystal(), _detector(P, pixel_size_mm=0.8), BeamConfig(wavelength_A=1.2, fluence=1e33),
                            dtype=DTYPE, diffuse=grid, **kwargs)
    with torch.no_grad():
        model.crystal_params.delta_misset_raw.copy_(torch.tensor([0.01, -0.02, 0.015]))
    return model


def _weights():
    return torch.rand(P, P, generator=torch.Generator().manual_seed(7), dtype=DTYPE)


def _loss(model):
    return (model() * _weights()).sum()


class _Cells:
    """Records the grid cell of every sample the grid sees (a forward hook, so names do not change)."""

    def __init__(self, grid):
        self.grid, self.seen = grid, []
        grid.register_forward_hook(self._record)

    def _record(self, module, inputs, output):
        u = (inputs[0].detach() - self.grid.origin) / self.grid.step
        self.seen.append(torch.floor(u))

    def take(self):
        cells, self.seen = torch.cat(self.seen), []
        return cells


def _set(parameter, component, value):
    with torch.no_grad():
        if component is None:
            parameter.fill_(value)
        else:
            parameter[component] = value


def _central_difference(model, name, component, cells, step):
    parameter = dict(model.named_parameters())[name]
    start = float(parameter if component is None else parameter[component])
    values, seen = [], []
    for sign in (1, -1):
        _set(parameter, component, start + sign * step)
        with torch.no_grad():
            values.append(float(_loss(model)))
        seen.append(cells.take())
    _set(parameter, component, start)
    assert torch.equal(seen[0], seen[1]), "a sample crossed a grid cell: the difference straddles a kink"
    return (values[0] - values[1]) / (2 * step)


@pytest.mark.parametrize("name,component", GEOMETRY)
def test_geometry_gradient_through_the_interpolation_matches_central_differences(name, component):
    grid = _halo_grid()
    model = _model(grid)
    cells = _Cells(grid)
    model.zero_grad(set_to_none=True)
    _loss(model).backward()
    cells.take()
    parameter = dict(model.named_parameters())[name]
    grad = parameter.grad if component is None else parameter.grad[component]
    fd = _central_difference(model, name, component, cells, step=1e-7)
    assert float(grad) != 0.0
    assert float(grad) == pytest.approx(fd, rel=1e-6)


def test_value_gradient_matches_central_differences_at_the_nodes_used():
    grid = _halo_grid(refinable=True)
    model = _model(grid)
    _loss(model).backward()
    grad = grid.values.grad.clone()
    used = torch.nonzero(grad).reshape(-1)
    assert used.numel() > 10
    for index in used[torch.linspace(0, used.numel() - 1, 5).long()]:
        with torch.no_grad():
            values = []
            for sign in (1, -1):
                grid.values[index] += sign * 1.0
                values.append(float(_loss(model)))
                grid.values[index] -= sign * 1.0
        fd = (values[0] - values[1]) / 2.0  # the image is linear in the values: no truncation error at any step
        assert float(grad[index]) == pytest.approx(fd, rel=1e-8)


@pytest.mark.parametrize("name", ["crystal_params.delta_misset_raw", "crystal_params.delta_log_b",
                                  "detector_params.delta_log_distance_mm", "beam_params.delta_log_lambda_scale"])
def test_gradcheck_in_the_geometry(name):
    grid = _halo_grid()
    model = _model(grid)
    cells = _Cells(grid)
    start = dict(model.named_parameters())[name].detach().clone()
    eps = 1e-7
    with torch.no_grad():  # gradcheck's perturbations, each element by +-eps, must stay in their cells
        functional_call(model, {name: start}, ())
        reference = cells.take()
        for i in range(start.numel()):
            for sign in (1, -1):
                moved = start.clone().reshape(-1)
                moved[i] += sign * eps
                functional_call(model, {name: moved.reshape(start.shape)}, ())
                assert torch.equal(cells.take(), reference)

    def loss(value):
        return (functional_call(model, {name: value}, ()) * _weights()).sum()

    assert gradcheck(loss, (start.requires_grad_(True),), eps=eps, atol=1e-8, rtol=1e-5)


def test_gradcheck_in_the_values_along_random_directions():
    grid = _halo_grid(refinable=True)
    model = _model(grid)
    base = grid.values.detach().clone()
    directions = torch.randn(3, base.numel(), generator=torch.Generator().manual_seed(8), dtype=DTYPE)

    def loss(c):
        return (functional_call(model, {f"{GRID}.values": base + c @ directions}, ()) * _weights()).sum()

    assert gradcheck(loss, (torch.zeros(3, dtype=DTYPE, requires_grad=True),), eps=1e-3, atol=1e-8, rtol=1e-6)


# ExperimentModel and SerialExperiment.

def test_refinable_values_are_the_diffuse_group_and_frozen_ones_are_state():
    refinable = _model(_halo_grid(refinable=True))
    assert refinable.PARAMETER_GROUPS["diffuse"] == (f"{GRID}.values",)
    refinable.refine("diffuse")
    assert refinable.diffuse.values.requires_grad
    frozen = _model(_halo_grid(refinable=False))
    assert frozen.PARAMETER_GROUPS["diffuse"] == ()
    assert f"{GRID}.values" in frozen.state_dict()
    assert all(not n.startswith(f"{GRID}.") for n, _ in frozen.named_parameters())


def test_param_init_frozen_freezes_the_values_in_place():
    grid = _halo_grid(refinable=True)
    _model(grid, param_init="frozen")
    assert not grid.values.requires_grad


def test_shots_share_one_grid():
    grid = _halo_grid(refinable=True)
    missets = [(5.0, -3.0, 2.0), (20.0, 10.0, -5.0)]

    def shot(misset):
        return ExperimentModel(_crystal(misset_deg=misset), _detector(P, pixel_size_mm=0.8),
                               BeamConfig(wavelength_A=1.2, fluence=1e33), dtype=DTYPE, diffuse=grid)

    serial = SerialExperiment([shot(m) for m in missets])
    assert sum(n.endswith(".values") for n, _ in serial.named_parameters()) == 1
    images = serial()
    sum((image * _weights()).sum() for image in images).backward()
    shared = grid.values.grad.clone()
    grid.values.grad = None
    for image, misset in zip(images, missets):
        alone = shot(misset)
        reference = alone()
        torch.testing.assert_close(image, reference, rtol=1e-13, atol=0.0)
        (reference * _weights()).sum().backward()
    torch.testing.assert_close(shared, grid.values.grad, rtol=1e-12, atol=0.0)
