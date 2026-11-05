"""Gradients of ExperimentModel(diffuse=...) against central finite differences and gradcheck (plan section 4).

CPU float64 on a 16 x 16 image. The crystal has F = 0, so the Bragg term is zero and every
geometry gradient below comes through the diffuse term's Miller indices.
"""
import math
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import pytest
import torch
from torch.autograd import gradcheck
from torch.autograd.function import once_differentiable
from torch.func import functional_call

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot, SpotMetric,
)
from nanobrag_torch.diffuse import GaussianHalo
from nanobrag_torch.models.experiment import ExperimentModel

P = 16
HALO = "diffuse_params.model"
K_D = "diffuse_params.delta_log_scale"
GEOMETRY = [  # (parameter, component), through the diffuse term
    ("crystal_params.delta_misset_raw", 0), ("crystal_params.delta_misset_raw", 2),
    ("crystal_params.delta_log_a", None), ("crystal_params.delta_gamma_raw", None),
    ("detector_params.delta_log_distance_mm", None),
    ("detector_params.delta_beam_s_raw", None), ("detector_params.delta_beam_f_raw", None),
    ("beam_params.delta_log_lambda_scale", None),
    ("crystal_params.delta_log_N_cells", 0), ("crystal_params.delta_log_N_cells", 2),  # through the diffuse term and its shape factor
]
DIFFUSE = [(f"{HALO}.log_amplitude", None), (f"{HALO}.log_sigma", None), (K_D, None)]


def _model(dtype=torch.float64, halo=None, device=None, **crystal_kwargs):
    crystal = CrystalConfig(**{**dict(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0),
                                      N_cells=(5, 5, 5), default_F=0.0, shape=CrystalShape.GAUSS), **crystal_kwargs})
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=0.8, spixels=P, fpixels=P, oversample=1,
                              detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)
    beam = BeamConfig(wavelength_A=1.2, fluence=1e33)  # a few photons a pixel
    model = ExperimentModel(crystal, detector, beam, dtype=dtype, device=device,
                            diffuse=halo if halo is not None else GaussianHalo(40.0, 0.15, dtype=dtype).to(device or "cpu"))
    with torch.no_grad():  # off the starting point, where the parameters are all exactly zero
        model.crystal_params.delta_misset_raw.copy_(torch.tensor([0.01, -0.02, 0.015]))
        model.diffuse_params.delta_log_scale.fill_(0.1)
    return model.refine("diffuse_scale")


def _weights(dtype=torch.float64):
    return torch.rand(P, P, generator=torch.Generator().manual_seed(7), dtype=torch.float64).to(dtype)


def _loss(model):
    return (model() * _weights(model.dtype)).sum()


def _central_difference(model, name, component, step=1e-6):
    parameter = dict(model.named_parameters())[name]
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            if component is None:
                parameter += sign * step
            else:
                parameter[component] += sign * step
            values.append(float(_loss(model)))
            if component is None:
                parameter -= sign * step
            else:
                parameter[component] -= sign * step
    return (values[0] - values[1]) / (2 * step)


def _autograd(model):
    model.zero_grad(set_to_none=True)
    (model() * _weights(model.dtype).to(model.device)).sum().backward()
    return {n: p.grad for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("name,component", DIFFUSE + GEOMETRY)
def test_gradient_matches_central_differences(name, component):
    model = _model()
    grads = _autograd(model)
    grad = grads[name] if component is None else grads[name][component]
    fd = _central_difference(model, name, component)
    assert float(grad) != 0.0
    assert float(grad) == pytest.approx(fd, rel=1e-6)


# The shape factor's own derivatives, through the model (audit 2026-11-05): ROUND in N, TOPHAT in
# the cell (inside the cell, and cut by it) and GAUSS-HKL in Ndef
SHAPE_CASES = {
    "round-N": (dict(shape=CrystalShape.ROUND), [("crystal_params.delta_log_N_cells", i) for i in range(3)]),
    "tophat-cell": (dict(shape=CrystalShape.TOPHAT), [("crystal_params.delta_log_a", None),
                                                       ("crystal_params.delta_gamma_raw", None)]),
    "tophat-cell-cut": (dict(shape=CrystalShape.TOPHAT, cell_a=120.0, cell_b=125.0, cell_c=115.0),
                        [("crystal_params.delta_log_a", None), ("crystal_params.delta_log_b", None)]),
    "gauss-ndef": (dict(shape=CrystalShape.GAUSS, spot_metric=SpotMetric.HKL, N_cells_def=(0.3, -0.2, 0.1)),
                   [("crystal_params.delta_N_cells_def", i) for i in range(3)]),
}


@pytest.mark.parametrize("case", sorted(SHAPE_CASES))
def test_shape_factor_gradients_match_central_differences(case):
    crystal, names = SHAPE_CASES[case]
    model = _model(**crystal)
    grads = _autograd(model)
    for name, component in names:
        grad = grads[name] if component is None else grads[name][component]
        fd = _central_difference(model, name, component)
        assert float(grad) != 0.0
        assert float(grad) == pytest.approx(fd, rel=1e-5), (name, component)


def test_bragg_is_zero_so_the_geometry_gradient_is_the_diffuse_terms():
    model = _model()
    model()
    assert model.components["bragg"].abs().max() == 0
    assert model.components["diffuse"].min() > 0


def test_k_d_gradient_is_the_diffuse_part():
    model = _model()
    grads = _autograd(model)
    expected = (model.components["diffuse"] * _weights()).sum()
    torch.testing.assert_close(grads[K_D], expected.detach(), rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("name", [f"{HALO}.log_sigma", "crystal_params.delta_misset_raw",
                                  "detector_params.delta_log_distance_mm", "beam_params.delta_log_lambda_scale",
                                  K_D])
def test_gradcheck(name):
    model = _model()
    start = dict(model.named_parameters())[name].detach().clone().requires_grad_(True)

    def loss(value):
        return (functional_call(model, {name: value}, ()) * _weights()).sum()

    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-8, rtol=1e-5)


def test_float32_gradients_follow_float64():
    """The same model in float32 (parameters and run) gives the float64 gradients to single precision."""
    double, single = _autograd(_model()), _autograd(_model(torch.float32))
    for name, _ in DIFFUSE + GEOMETRY:
        assert single[name].dtype == torch.float32
        torch.testing.assert_close(single[name].double(), double[name], rtol=2e-3, atol=1e-3 * double[name].abs().max())


@pytest.mark.parametrize("device", [
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_float32_gradients_on_a_gpu_follow_the_cpu(device):
    """The whole model on the GPU in float32: the gradient values, not just their device, against CPU float64."""
    double = _autograd(_model())
    model = _model(torch.float32, device=torch.device(device))
    single = _autograd(model)
    for name, _ in DIFFUSE + GEOMETRY:
        assert single[name].device.type == device
        torch.testing.assert_close(single[name].cpu().double(), double[name], rtol=2e-3,
                                   atol=1e-3 * double[name].abs().max())


class _FirstOrderOnly(torch.autograd.Function):
    """Identity whose backward cannot itself be differentiated, as eryx's eigh backward."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return grad


class _FirstOrderHalo(GaussianHalo):
    def forward(self, hkl):
        return _FirstOrderOnly.apply(super().forward(hkl))


def test_backward_does_not_need_create_graph():
    """A model with a first-order-only backward refines: the gradients are those of the plain model."""
    plain = _autograd(_model())
    first_order = _model(halo=_FirstOrderHalo(40.0, 0.15))
    loss = _loss(first_order)
    grads = torch.autograd.grad(loss, first_order.trainable_parameters())  # create_graph=False
    for (name, parameter), grad in zip([(n, p) for n, p in first_order.named_parameters() if p.requires_grad], grads):
        torch.testing.assert_close(grad, plain[name], rtol=1e-13, atol=0.0)


def test_lbfgsb_refines_a_first_order_only_model():
    pytest.importorskip("scipy")
    from nanobrag_torch.utils.refinement import refine_lbfgsb

    model = _model(halo=_FirstOrderHalo(40.0, 0.15)).freeze().refine("diffuse")
    target = _model(halo=GaussianHalo(30.0, 0.12))
    with torch.no_grad():
        observed = target()
    result = refine_lbfgsb(model, lambda: ((model() - observed) ** 2).sum(), maxiter=3)
    with torch.no_grad():
        assert result.nit >= 1 and ((model() - observed) ** 2).sum() < ((_model()() - observed) ** 2).sum()
