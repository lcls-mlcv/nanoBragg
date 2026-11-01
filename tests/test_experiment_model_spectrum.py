"""Refinable spectrum in ExperimentModel: lambda offset and scale, source weights (plan C2).

A plain Simulator ignores source weights, as C does; use_source_weights and ExperimentModel
weight them as diffBragg does. tests/test_diffbragg_parity.py compares both with diffBragg.
"""
import math
from dataclasses import replace

import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.simulator import Simulator

DTYPE = torch.float64
WAVELENGTHS_A = (1.19, 1.2, 1.21)
WEIGHTS = (1.0, 3.0, 2.0)


def _crystal():
    return CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0),
                         N_cells=(5, 5, 5), default_F=100.0, shape=CrystalShape.GAUSS)


def _detector(pixels=32, oversample=1):
    return DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=pixels, fpixels=pixels,
                          detector_convention=DetectorConvention.MOSFLM,
                          detector_pivot=DetectorPivot.BEAM, oversample=oversample)


def _beam(wavelengths=None, weights=None, **kwargs):
    beam = BeamConfig(wavelength_A=1.2, fluence=1e24, **kwargs)
    if wavelengths is not None:
        n = len(wavelengths)
        beam.source_directions = torch.tensor([[-1.0, 0.0, 0.0]] * n, dtype=DTYPE)  # MOSFLM beam is +x
        beam.source_wavelengths = torch.tensor(wavelengths, dtype=DTYPE) * 1e-10
        beam.source_weights = None if weights is None else torch.tensor(weights, dtype=DTYPE)
    return beam


def _image(beam, oversample=1, **run):
    detector = Detector(_detector(oversample=oversample), dtype=DTYPE)
    return Simulator(Crystal(_crystal(), beam, dtype=DTYPE), detector, beam_config=beam, dtype=DTYPE).run(**run)


def _single(wavelength):
    return _image(replace(_beam(), wavelength_A=wavelength))


@pytest.mark.parametrize("oversample", [1, 2])
def test_use_source_weights_is_the_fluence_fraction_sum(oversample):
    expected = sum(w / sum(WEIGHTS) * _image(replace(_beam(), wavelength_A=lam), oversample=oversample)
                   for lam, w in zip(WAVELENGTHS_A, WEIGHTS))
    got = _image(_beam(WAVELENGTHS_A, WEIGHTS, use_source_weights=True), oversample=oversample)
    assert got.sum() > 0
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=0.0)


def test_weights_stay_ignored_without_the_flag_and_equal_weights_are_cs_image():
    c_image = _image(_beam(WAVELENGTHS_A, WEIGHTS))
    assert torch.equal(c_image, _image(_beam(WAVELENGTHS_A)))
    assert torch.equal(c_image, _image(_beam(WAVELENGTHS_A, (2.0, 2.0, 2.0), use_source_weights=True)))
    assert not torch.allclose(c_image, _image(_beam(WAVELENGTHS_A, WEIGHTS, use_source_weights=True)))


def test_only_the_shape_of_the_weights_counts():
    first = _image(_beam(WAVELENGTHS_A, WEIGHTS, use_source_weights=True))
    scaled = _image(_beam(WAVELENGTHS_A, tuple(7 * w for w in WEIGHTS), use_source_weights=True))
    torch.testing.assert_close(first, scaled, rtol=1e-12, atol=0.0)


def test_weights_survive_pixel_batching():
    beam = _beam(WAVELENGTHS_A, WEIGHTS, use_source_weights=True)
    torch.testing.assert_close(_image(beam, pixel_batch_size=7), _image(beam), rtol=1e-12, atol=0.0)


def _model(wavelengths=WAVELENGTHS_A, weights=WEIGHTS, pixels=32):
    return ExperimentModel(_crystal(), _detector(pixels), _beam(wavelengths, weights), dtype=DTYPE)


def test_model_weights_a_spectrum_as_diffbragg_and_trains_it():
    model = _model()
    assert model.beam_params.delta_log_source_weights.requires_grad
    assert model.beam_params.lambda_offset.requires_grad
    assert model.beam_params.delta_log_lambda_scale.requires_grad
    with torch.no_grad():
        image = model()
    torch.testing.assert_close(image, _image(_beam(WAVELENGTHS_A, WEIGHTS, use_source_weights=True)),
                               rtol=1e-10, atol=0.0)


def test_model_without_a_spectrum_reproduces_the_config_image_and_blocks_weights():
    model = ExperimentModel(_crystal(), _detector(), _beam(), dtype=DTYPE)
    with torch.no_grad():
        image = model()
    assert torch.equal(image, _single(1.2))
    assert "beam_params.delta_log_source_weights" in model.unrefinable
    with pytest.raises(ValueError, match="fewer than two sources"):
        model.refine("source_weights")
    model.freeze().refine("wavelength")
    assert [n for n, p in model.named_parameters() if p.requires_grad] == [
        "beam_params.lambda_offset", "beam_params.delta_log_lambda_scale"]


def test_lambda_offset_and_scale_reach_the_single_wavelength():
    model = ExperimentModel(_crystal(), _detector(), _beam(), dtype=DTYPE)
    with torch.no_grad():
        model()  # the steps below must reach the persistent simulator
        model.beam_params.lambda_offset.fill_(0.02)
        model.beam_params.delta_log_lambda_scale.fill_(math.log(1.01))
        image = model()
    torch.testing.assert_close(image, _single(0.02 + 1.01 * 1.2), rtol=1e-10, atol=0.0)


def test_lambda_offset_and_scale_and_weights_reach_every_source():
    model = _model()
    bumps = (0.1, -0.3, 0.2)
    with torch.no_grad():
        model()
        model.beam_params.lambda_offset.fill_(0.02)
        model.beam_params.delta_log_lambda_scale.fill_(math.log(1.01))
        model.beam_params.delta_log_source_weights.copy_(torch.tensor(bumps, dtype=DTYPE))
        image = model()
    moved = tuple(0.02 + 1.01 * lam for lam in WAVELENGTHS_A)
    weights = tuple(w * math.exp(b) for w, b in zip(WEIGHTS, bumps))
    torch.testing.assert_close(image, _image(_beam(moved, weights, use_source_weights=True)), rtol=1e-10, atol=0.0)


def test_moving_all_weights_together_does_nothing():
    model = _model()
    w = torch.rand(32, 32, dtype=DTYPE, generator=torch.Generator().manual_seed(1))
    (model() * w).sum().backward()
    assert abs(float(model.beam_params.delta_log_source_weights.grad.sum())) < 1e-12 * float(
        model.beam_params.delta_log_source_weights.grad.abs().max())


@pytest.mark.parametrize("name, start", [
    ("beam_params.lambda_offset", torch.tensor(0.002, dtype=DTYPE)),
    ("beam_params.delta_log_lambda_scale", torch.tensor(0.001, dtype=DTYPE)),
    ("beam_params.delta_log_source_weights", torch.tensor([0.1, -0.2, 0.05], dtype=DTYPE)),
])
def test_gradcheck_spectrum(name, start):
    model = _model(pixels=16)

    def loss(value):
        return functional_call(model, {name: value}, ()).sum()

    assert gradcheck(loss, (start.clone().requires_grad_(True),), eps=1e-6, atol=1e-8, rtol=1e-4)
