"""ROI simulation and the likelihood losses (plan C4).

tests/test_diffbragg_parity.py checks the gaussian target and gradient against hopper's.
"""
import math

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
from nanobrag_torch.utils.losses import gaussian_nll, poisson_nll

DTYPE = torch.float64
ROIS = [(2, 12, 3, 9), (15, 24, 10, 22), (26, 31, 1, 6)]
ABC = [(0.5, -0.25, 3.0), (-1.0, 2.0, 7.0), (0.0, 0.0, 1.0)]
PIXELS = torch.tensor([[0, 0], [5, 7], [31, 31], [12, 20], [5, 7], [31, 0]])


def _crystal(**kwargs):
    return CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(5, 5, 5),
                         default_F=100.0, shape=kwargs.pop("shape", CrystalShape.GAUSS), **kwargs)


def _detector(**kwargs):
    kwargs.setdefault("oversample", 1)
    return DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=32, fpixels=32,
                          detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM, **kwargs)


def _beam(**kwargs):
    return BeamConfig(wavelength_A=1.2, fluence=1e24, **kwargs)


def _sources(beam, n=3):
    beam.source_directions = torch.tensor([[-1.0, 0.0, 0.0]] * n, dtype=DTYPE)  # MOSFLM beam is +x
    beam.source_wavelengths = torch.linspace(1.19, 1.21, n, dtype=DTYPE) * 1e-10
    beam.source_weights = torch.tensor([1.0, 3.0, 2.0], dtype=DTYPE)
    beam.use_source_weights = True
    return beam


CASES = {
    "plain": dict(),
    "square": dict(crystal=dict(shape=CrystalShape.SQUARE)),
    "oversample": dict(detector=dict(oversample=3)),
    "oversample-omega": dict(detector=dict(oversample=2, oversample_omega=True)),
    "thick": dict(detector=dict(detector_thick_um=100.0, detector_thicksteps=3, detector_abs_um=500.0)),
    "curved": dict(detector=dict(curved_detector=True)),
    "curved-oversample": dict(detector=dict(curved_detector=True, oversample=2)),
    "roi-limits": dict(detector=dict(roi_xmin=4, roi_xmax=25, roi_ymin=2, roi_ymax=29)),
    "water": dict(beam=dict(water_size_um=10.0)),
    "sources": dict(sources=True),
    "sources-oversample": dict(sources=True, detector=dict(oversample=2)),
    "nopolar": dict(beam=dict(nopolar=True)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_pixels_are_the_full_image_at_those_pixels(case):
    spec = CASES[case]
    beam = _beam(**spec.get("beam", {}))
    if spec.get("sources"):
        _sources(beam)
    simulator = Simulator(Crystal(_crystal(**spec.get("crystal", {})), beam, dtype=DTYPE),
                          Detector(_detector(**spec.get("detector", {})), dtype=DTYPE),
                          beam_config=beam, dtype=DTYPE)
    full = simulator.run()
    assert full.abs().sum() > 0
    expected = full[tuple(PIXELS.T)]
    torch.testing.assert_close(simulator.run(pixels=PIXELS), expected, rtol=1e-13, atol=0.0)
    torch.testing.assert_close(simulator.run(pixels=PIXELS, pixel_batch_size=2), expected, rtol=1e-13, atol=0.0)


def test_pixels_refuse_debug_output():
    beam = _beam()
    simulator = Simulator(Crystal(_crystal(), beam, dtype=DTYPE), Detector(_detector(), dtype=DTYPE),
                          beam_config=beam, dtype=DTYPE, debug_config=dict(printout=True))
    with pytest.raises(ValueError, match="whole image"):
        simulator.run(pixels=PIXELS)


def _model(**kwargs):
    return ExperimentModel(_crystal(), _detector(**kwargs), _beam(), dtype=DTYPE,
                           background_rois=ROIS, tilt_abc=ABC)


def test_model_roi_is_the_full_model_at_its_pixels():
    model = _model()
    w = torch.rand(32, 32, dtype=DTYPE, generator=torch.Generator().manual_seed(1))
    full = model()
    roi = model(roi_only=True)
    assert roi.shape == (sum((x2 - x1) * (y2 - y1) for x1, x2, y1, y2 in ROIS),)
    torch.testing.assert_close(roi, model.roi_values(full), rtol=1e-13, atol=0.0)

    params = [p for p in model.parameters() if p.requires_grad]
    selected = torch.zeros(32, 32, dtype=DTYPE).index_put_(tuple(model.background_params.pixels.T),
                                                           torch.tensor(1.0, dtype=DTYPE))
    reference = torch.autograd.grad((full * w * selected).sum(), params, allow_unused=True)
    got = torch.autograd.grad((roi * model.roi_values(w)).sum(), params, allow_unused=True)
    for r, g in zip(reference, got):
        if r is None:
            assert g is None
        else:
            torch.testing.assert_close(g, r, rtol=1e-11, atol=1e-13)  # some gradients are roundoff


def test_roi_values_keep_a_mask_a_mask():
    model = _model()
    trusted = torch.ones(32, 32, dtype=torch.bool)
    trusted[5, 5] = False
    assert model.roi_values(trusted).dtype == torch.bool
    assert (~model.roi_values(trusted)).sum() == 1


def test_masked_pixels_get_no_background_either():
    model = _model(roi_xmin=0, roi_xmax=20, roi_ymin=0, roi_ymax=31)
    with torch.no_grad():
        roi = model(roi_only=True)
    assert (roi[model.background_params.pixels[:, 1] > 20] == 0).all()


def test_later_roi_keeps_its_pixels_once_in_the_list():
    rois = [(0, 8, 0, 8), (4, 12, 4, 12)]
    model = ExperimentModel(_crystal(), _detector(), _beam(), dtype=DTYPE, background_rois=rois)
    pixels = model.background_params.pixels
    assert len(pixels) == 64 + 64 - 16
    assert len(torch.unique(pixels, dim=0)) == len(pixels)


def test_gaussian_nll_is_the_normal_likelihood_and_hoppers_formula():
    model = torch.tensor([4.0, 9.0, 16.0], dtype=DTYPE, requires_grad=True)
    data = torch.tensor([5.0, 7.0, 16.0], dtype=DTYPE)
    sigma = 3.0
    got = gaussian_nll(model, data, sigma)
    scale = (model + sigma**2).sqrt()
    expected = -torch.distributions.Normal(model, scale).log_prob(data).sum()
    torch.testing.assert_close(got, expected, rtol=1e-13, atol=0.0)
    got.backward()
    V, r = (model + sigma**2).detach(), (data - model).detach()
    torch.testing.assert_close(model.grad, 0.5 / V * (1 - 2 * r - r**2 / V), rtol=1e-12, atol=0.0)


def test_gaussian_gain_trusted_and_per_pixel_sigma():
    model = torch.tensor([4.0, 9.0, 16.0], dtype=DTYPE)
    data = torch.tensor([5.0, 7.0, 16.0], dtype=DTYPE)
    trusted = torch.tensor([True, False, True])
    sigma = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    photons = gaussian_nll(model[trusted], data[trusted], sigma[trusted])
    adu = gaussian_nll(model, 2.5 * data, 2.5 * sigma, trusted=trusted, adu_per_photon=2.5)
    torch.testing.assert_close(adu, photons, rtol=1e-13, atol=0.0)


def test_gaussian_variance_floor_keeps_it_finite():
    model = torch.tensor([-10.0], dtype=DTYPE, requires_grad=True)
    loss = gaussian_nll(model, torch.tensor([1.0], dtype=DTYPE), sigma_rdout=1.0)
    assert torch.isfinite(loss)


def test_poisson_nll_is_the_poisson_likelihood():
    model = torch.tensor([4.0, 0.5, 16.0], dtype=DTYPE, requires_grad=True)
    data = torch.tensor([5.0, 0.0, 16.0], dtype=DTYPE)
    got = poisson_nll(model, data)
    expected = -torch.distributions.Poisson(model).log_prob(data).sum()
    torch.testing.assert_close(got, expected, rtol=1e-13, atol=0.0)
    got.backward()
    torch.testing.assert_close(model.grad, 1 - data / model.detach(), rtol=1e-12, atol=0.0)


def test_poisson_gain_and_trusted():
    model = torch.tensor([4.0, 9.0], dtype=DTYPE)
    data = torch.tensor([5.0, -3.0], dtype=DTYPE)
    trusted = torch.tensor([True, False])
    torch.testing.assert_close(poisson_nll(model, 2 * data, trusted=trusted, adu_per_photon=2.0),
                               poisson_nll(model[:1], data[:1]), rtol=1e-13, atol=0.0)
    with pytest.raises(ValueError, match="counts >= 0"):
        poisson_nll(model, data)


def test_gradcheck_nll_through_the_model():
    model = _model()
    name = "crystal_params.delta_misset_raw"
    data = model.roi_values(torch.full((32, 32), 5.0, dtype=DTYPE))

    def loss(value):
        return gaussian_nll(functional_call(model, {name: value}, kwargs=dict(roi_only=True)), data, 1.0)

    start = torch.tensor([0.01, -0.02, 0.015], dtype=DTYPE, requires_grad=True)
    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("nll", ["gaussian", "poisson"])
def test_refinement_on_shoeboxes_recovers_scale_and_orientation(nll):
    # 1e3 photons a pixel so that the likelihood sees the Bragg signal
    def build():
        beam = BeamConfig(wavelength_A=1.2, fluence=1e30)
        return ExperimentModel(_crystal(), _detector(), beam, dtype=DTYPE, background_rois=ROIS, tilt_abc=ABC)

    truth = build()
    with torch.no_grad():
        truth.beam_params.delta_log_spot_scale.fill_(0.3)
        truth.crystal_params.delta_misset_raw.copy_(torch.tensor([0.02, -0.03, 0.01], dtype=DTYPE))
        data = truth(roi_only=True)
    start = build()
    start.freeze().refine("scale", "misset")
    optimizer = torch.optim.LBFGS(start.trainable_parameters(), max_iter=40, tolerance_grad=1e-10,
                                  tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        pred = start(roi_only=True)
        loss = gaussian_nll(pred, data, 3.0) if nll == "gaussian" else poisson_nll(pred, data)
        loss.backward()
        return loss

    optimizer.step(closure)
    torch.testing.assert_close(start.beam_params.delta_log_spot_scale,
                               truth.beam_params.delta_log_spot_scale, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(start.crystal_params.delta_misset_raw,
                               truth.crystal_params.delta_misset_raw, rtol=1e-3, atol=1e-3)
    assert math.isfinite(float(closure()))
