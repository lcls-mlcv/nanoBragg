"""D3b: a DiffuseGrid with an analytic core (plan section 2.2).

A sharp halo core (a one-phonon 1/k^2 with a k floor) needs nodes far finer than any affordable
uniform grid. The model supplies the core as a cheap analytic DiffuseModel; the grid holds the rest,
model - core, which is smooth; the grid's value is core(hkl) + the residual's interpolant.
Measured in scripts/diffuse/core_candidates.py.
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
from nanobrag_torch.diffuse import ConstantDiffuse, DiffuseGrid, DiffuseModel, LorentzianHalo
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.simulator import Simulator

DTYPE = torch.float64
MMM = np.array([np.diag(d) for d in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))])
CELL = (30.0, 35.0, 40.0)
D_MIN = 7.0


class Smooth(nn.Module):
    periods = (3.0, 4.0, 5.0)

    def __init__(self, amplitude=5.0):
        super().__init__()
        self.amplitude = amplitude

    def forward(self, hkl):
        waves = [torch.cos(2 * math.pi * hkl[:, i] / p) for i, p in enumerate(self.periods)]
        return self.amplitude * (2 + waves[0] * waves[1] * waves[2])

    def curvature(self):
        return self.amplitude * sum((2 * math.pi / p) ** 2 for p in self.periods)


class Sum(nn.Module):
    def __init__(self, *parts):
        super().__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, hkl):
        return sum(part(hkl) for part in self.parts)


def _sharp():
    return LorentzianHalo(amplitude=1000.0, kappa=0.05, metric=(1.0, 1.3, 0.8))


# 1. The reference core model.

def test_lorentzian_halo_follows_its_closed_form_and_is_a_model():
    halo = _sharp()
    assert isinstance(halo, DiffuseModel)
    hkl = torch.tensor([[1.0, 2.0, -3.0], [1.1, 2.0, -3.0], [0.0, 0.2, 0.1], [0.5, 0.0, 0.0]], dtype=DTYPE)
    d2 = torch.tensor([0.0, 0.01, 1.3 * 0.04 + 0.8 * 0.01, 0.25], dtype=DTYPE)
    torch.testing.assert_close(halo(hkl), 1000.0 * 0.0025 / (d2 + 0.0025), rtol=1e-13, atol=0)
    with pytest.raises(ValueError, match="positive"):
        LorentzianHalo(1.0, 0.0)


def test_lorentzian_halo_gradcheck():
    halo = _sharp()
    hkl = (torch.randint(-3, 3, (6, 3)) + 0.4 * (torch.rand(6, 3) - 0.5)).to(DTYPE).requires_grad_()
    assert gradcheck(lambda h: halo(h), (hkl,))
    params = (halo.log_amplitude.detach().clone().requires_grad_(), halo.log_kappa.detach().clone().requires_grad_())
    assert gradcheck(lambda a, k: functional_call(halo, {"log_amplitude": a, "log_kappa": k}, (hkl.detach(),)), params)


# 2. The cored grid.

def _cored(model=None, core=None, step=1 / 4, point_group=MMM, refinable=False):
    core = core if core is not None else _sharp()
    model = model if model is not None else Sum(core, Smooth())
    return DiffuseGrid.from_model(model, step=step, d_min=D_MIN, cell=CELL, point_group=point_group,
                                  refinable=refinable, core=core), model, core


def test_cored_grid_is_the_core_plus_the_residual_and_exact_at_the_nodes():
    grid, model, core = _cored()
    assert grid.core is core
    nodes = grid.value_hkl()[:500]
    torch.testing.assert_close(grid(nodes), model(nodes).detach(), rtol=1e-12, atol=1e-12)
    hkl = (torch.rand(300, 3, dtype=DTYPE, generator=torch.Generator().manual_seed(1)) - 0.5) * 6
    residual = DiffuseGrid.from_model(lambda h: model(h) - core(h), step=1 / 4, d_min=D_MIN, cell=CELL,
                                      point_group=MMM, refinable=False)
    torch.testing.assert_close(grid(hkl), (core(hkl) + residual(hkl)).detach(), rtol=1e-13, atol=1e-12)


def test_the_core_may_come_from_the_models_core_method():
    class WithCore(Sum):
        def core(self):
            return self.parts[0]

    model = WithCore(_sharp(), Smooth())
    grid = DiffuseGrid.from_model(model, step=1 / 4, d_min=D_MIN, cell=CELL, point_group=MMM, core=True)
    assert grid.core is model.parts[0]
    with pytest.raises(ValueError, match="core"):
        DiffuseGrid.from_model(Smooth(), step=1 / 4, d_min=D_MIN, cell=CELL, core=True)


def test_the_core_follows_the_query_and_symmetry_does_not_change_it():
    asu, model, core = _cored()
    p1, _, _ = _cored(model=model, core=core, point_group=None)
    hkl = (torch.rand(400, 3, dtype=DTYPE, generator=torch.Generator().manual_seed(2)) - 0.5) * 6
    torch.testing.assert_close(asu(hkl), p1(hkl), rtol=1e-13, atol=1e-12)
    assert asu(hkl.float()).dtype == torch.float32


# 3. Images: grid vs exact.

def _detector(pixels):
    return DetectorConfig(distance_mm=60.0, pixel_size_mm=12.8 / pixels, spixels=pixels, fpixels=pixels, oversample=1,
                          detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)


def _crystal(**kwargs):
    return CrystalConfig(**{**dict(cell_a=CELL[0], cell_b=CELL[1], cell_c=CELL[2], misset_deg=(5.0, -3.0, 2.0),
                                   N_cells=(10, 10, 10), default_F=0.0, shape=CrystalShape.GAUSS,
                                   spot_metric=SpotMetric.HKL), **kwargs})


def _image(diffuse, pixels=48):
    sim = Simulator(Crystal(_crystal(), dtype=DTYPE), Detector(_detector(pixels), dtype=DTYPE),
                    beam_config=BeamConfig(wavelength_A=1.2, fluence=1e24), dtype=DTYPE, diffuse=diffuse)
    with torch.no_grad():
        sim.run(oversample=1)
    return sim.components["diffuse"]


def test_with_the_core_the_image_error_is_the_residuals_trilinear_bound():
    """The residual is the smooth part, so the image error is under its bound; without the core the
    same grid is far off (the core is sharper than the step)."""
    smooth = Smooth()
    core = _sharp()
    model = Sum(core, smooth)
    exact = _image(model)
    cored = DiffuseGrid.from_model(model, step=1 / 4, d_min=D_MIN, cell=CELL, point_group=MMM, refinable=False,
                                   core=core)
    bare = DiffuseGrid.from_model(model, step=1 / 4, d_min=D_MIN, cell=CELL, point_group=MMM, refinable=False)
    bound = (1 / 4) ** 2 / 8 * smooth.curvature() * _image(ConstantDiffuse(1.0))
    error = (_image(cored) - exact).abs()
    assert (error <= bound).all() and float(error.max()) > 0
    assert float((_image(bare) - exact).abs().max()) > 10 * float(bound.max())


# 4. Gradients through the full image: geometry (the core is analytic, the residual piecewise linear),
#    the core's own parameters and the residual's values.

P = 16
CORE = "diffuse_params.model.core"


def _model(grid):
    model = ExperimentModel(_crystal(N_cells=(5, 5, 5)), DetectorConfig(
        distance_mm=60.0, pixel_size_mm=0.8, spixels=P, fpixels=P, oversample=1,
        detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM),
        BeamConfig(wavelength_A=1.2, fluence=1e33), dtype=DTYPE, diffuse=grid)
    with torch.no_grad():
        model.crystal_params.delta_misset_raw.copy_(torch.tensor([0.01, -0.02, 0.015]))
    return model


def _weights():
    return torch.rand(P, P, generator=torch.Generator().manual_seed(7), dtype=DTYPE)


def _loss(model):
    return (model() * _weights()).sum()


def test_core_parameters_and_values_are_the_diffuse_group():
    grid, _, _ = _cored(refinable=True)
    model = _model(grid)
    group = set(model.PARAMETER_GROUPS["diffuse"])
    assert {f"{CORE}.log_amplitude", f"{CORE}.log_kappa", "diffuse_params.model.values"} <= group


class _Cells:
    def __init__(self, grid):
        self.grid, self.seen = grid, []
        grid.register_forward_hook(lambda m, i, o: self.seen.append(torch.floor((i[0].detach() - grid.origin) / grid.step)))

    def take(self):
        cells, self.seen = torch.cat(self.seen), []
        return cells


@pytest.mark.parametrize("name,component", [
    ("crystal_params.delta_misset_raw", 0), ("crystal_params.delta_log_a", None),
    ("detector_params.delta_log_distance_mm", None), ("beam_params.delta_log_lambda_scale", None),
    (f"{CORE}.log_kappa", None), (f"{CORE}.log_amplitude", None),
])
def test_gradients_match_central_differences(name, component):
    grid, _, _ = _cored()
    model = _model(grid)
    cells = _Cells(grid)
    model.zero_grad(set_to_none=True)
    _loss(model).backward()
    cells.take()
    parameter = dict(model.named_parameters())[name]
    grad = float(parameter.grad if component is None else parameter.grad[component])
    start = float(parameter if component is None else parameter[component])
    values, seen = [], []
    for sign in (1, -1):
        with torch.no_grad():
            if component is None:
                parameter.fill_(start + sign * 1e-7)
            else:
                parameter[component] = start + sign * 1e-7
            values.append(float(_loss(model)))
        seen.append(cells.take())
    with torch.no_grad():
        if component is None:
            parameter.fill_(start)
        else:
            parameter[component] = start
    assert torch.equal(seen[0], seen[1]), "a sample crossed a grid cell"
    fd = (values[0] - values[1]) / 2e-7
    assert grad != 0.0 and grad == pytest.approx(fd, rel=1e-6)


def test_gradcheck_in_the_residual_values_and_the_core():
    grid, _, _ = _cored(refinable=True)
    model = _model(grid)
    base = grid.values.detach().clone()
    directions = torch.randn(2, base.numel(), generator=torch.Generator().manual_seed(8), dtype=DTYPE)
    kappa = grid.core.log_kappa.detach().clone()

    def loss(c, k):
        params = {"diffuse_params.model.values": base + c @ directions, f"{CORE}.log_kappa": k}
        return (functional_call(model, params, ()) * _weights()).sum()

    assert gradcheck(loss, (torch.zeros(2, dtype=DTYPE, requires_grad=True), kappa.requires_grad_()),
                     eps=1e-6, atol=1e-7, rtol=1e-5)


# 5. Audit fixes (2026-11-05): full metric, symmetry, the half-integer kink, the frozen residual,
#    outside the box, from_values, nbytes, float32.

import warnings

PG_422 = np.array([np.linalg.matrix_power(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), i) @ m
                   for i in range(4) for m in (np.eye(3, dtype=np.int64), np.diag([1, -1, -1]))])


def test_lorentzian_halo_takes_a_full_metric():
    metric = ((1.0, 0.2, 0.0), (0.2, 1.3, 0.1), (0.0, 0.1, 0.8))
    halo = LorentzianHalo(10.0, 0.05, metric=metric)
    hkl = torch.tensor([[0.1, -0.05, 0.02], [2.03, 1.0, -0.04]], dtype=DTYPE)
    d = hkl - torch.round(hkl)
    q = torch.einsum("ni,ij,nj->n", d, torch.tensor(metric, dtype=DTYPE), d)
    torch.testing.assert_close(halo(hkl), 10.0 * 0.0025 / (q + 0.0025), rtol=1e-13, atol=0)
    with pytest.raises(ValueError, match="symmetric positive definite"):
        LorentzianHalo(1.0, 0.05, metric=((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_a_core_that_breaks_the_point_group_is_refused():
    broken = LorentzianHalo(100.0, 0.05, metric=(1.0, 1.3, 1.0))  # m_h != m_k: not 4/mmm
    model = Sum(broken, Smooth())
    with pytest.raises(ValueError, match="not invariant"):
        DiffuseGrid.from_model(model, step=1 / 4, extent=(2, 2, 2), point_group=PG_422, core=broken)
    fine = LorentzianHalo(100.0, 0.05, metric=(1.0, 1.0, 1.3))
    DiffuseGrid.from_model(Sum(fine, Smooth()), step=1 / 4, extent=(2, 2, 2), point_group=PG_422, core=fine)


def test_a_core_needs_nodes_on_the_half_integer_planes():
    core = _sharp()
    with pytest.raises(ValueError, match="half-integer"):
        DiffuseGrid.from_model(Sum(core, Smooth()), step=1 / 3, extent=(1, 1, 1), core=core)


def test_from_values_refuses_a_core_and_the_constructor_takes_the_residual():
    with pytest.raises(ValueError, match="residual"):
        DiffuseGrid.from_values(torch.zeros(5, 5, 5), origin=(-1.0, -1.0, -1.0), step=0.5, core=_sharp())
    core = _sharp()
    grid = DiffuseGrid(torch.zeros(5, 5, 5, dtype=DTYPE), origin=(-1.0, -1.0, -1.0), step=0.5, core=core)
    hkl = torch.tensor([[0.1, 0.2, 0.3]], dtype=DTYPE)
    torch.testing.assert_close(grid(hkl), core(hkl).detach())


def test_outside_the_box_only_the_core_remains():
    grid, model, core = _cored()
    far = torch.tensor([[40.02, 0.0, 0.0]], dtype=DTYPE)
    torch.testing.assert_close(grid(far), core(far).detach())


class _SharedAmplitude(nn.Module):
    """core(theta) + smooth(theta): the smooth part scales with the core's amplitude, so a residual
    sampled at theta_b goes stale as the amplitude moves."""

    def __init__(self, core):
        super().__init__()
        self.core_term, self.smooth = core, Smooth()

    def forward(self, hkl):
        return self.core_term(hkl) + self.smooth(hkl) * torch.exp(self.core_term.log_amplitude - math.log(1000.0))


def test_the_residual_is_frozen_at_build_and_its_drift_is_exposed_and_warned():
    core = _sharp()
    model = _SharedAmplitude(core)
    grid = DiffuseGrid.from_model(model, step=1 / 4, d_min=D_MIN, cell=CELL, point_group=MMM, refinable=False,
                                  core=core)
    assert grid.core_drift() == 0.0
    nodes = grid.value_hkl()[:400]
    errors = []
    for delta in (0.01, 0.04):
        with torch.no_grad():
            core.log_amplitude.fill_(math.log(1000.0) + delta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stale = grid(nodes) - model(nodes)
        # the grid is core(theta) + model(theta_b) - core(theta_b): its error is smooth(theta_b) - smooth(theta)
        expected = Smooth()(nodes) * (1 - math.exp(delta))
        torch.testing.assert_close(stale.detach(), expected, rtol=1e-10, atol=1e-10)
        errors.append(float(stale.abs().max()))
        assert grid.core_drift() == pytest.approx(delta)
    assert errors[1] > 3.5 * errors[0]
    # the gradient in theta is the core's alone
    grid.zero_grad()
    hkl = nodes[:50]
    grid(hkl).sum().backward()
    reference = _sharp()
    with torch.no_grad():
        reference.log_amplitude.copy_(core.log_amplitude)
    reference(hkl).sum().backward()
    torch.testing.assert_close(core.log_amplitude.grad, reference.log_amplitude.grad, rtol=1e-12, atol=0)
    # past the tolerance (a placeholder, to be set on V2) a warning, once
    grid.core_drift_tolerance = 0.03
    with pytest.warns(UserWarning, match="drifted"):
        grid(hkl)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid(hkl)  # once only


def test_nbytes_counts_the_core():
    grid, _, core = _cored()
    held = sum(t.numel() * t.element_size() for t in list(core.parameters()) + list(core.buffers()))
    assert held > 0
    assert grid.nbytes() == grid.values.numel() * 8 + grid._keys.numel() * 8 + held


def test_float32_core_and_cored_grid_at_lysozyme_sized_indices():
    """d = hkl - round(hkl) in float32 carries about ulp(|h|) at |h| = 40, so the core's relative error
    is about ulp(40) * |d ln L / d d| <= ulp(40) sqrt(max m) / kappa, plus float32 rounding."""
    core = _sharp()
    g = torch.Generator().manual_seed(11)
    hkl = (torch.rand(5000, 3, generator=g, dtype=DTYPE) - 0.5) * 80
    exact = core(hkl).detach()
    single = core(hkl.float()).detach().double()
    ulp40 = 2 ** (math.floor(math.log2(40)) - 23)
    tolerance = 2 * ulp40 * math.sqrt(1.3) / 0.05 + 16 * torch.finfo(torch.float32).eps
    assert float(((single - exact) / exact).abs().max()) <= tolerance
