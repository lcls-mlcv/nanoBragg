"""ExperimentModel(diffuse=...): the diffuse model's groups, k_d, the ROI path and shared shots (plan D2).

The term itself is checked on physics in tests/test_diffuse_term.py; these tests check that the
model carries it: its parameters are refinable groups, k_d scales only the diffuse part, the ROI
and multi-panel paths give the whole-image pixels, and SerialExperiment holds one model.
"""
import math
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import pytest
import torch

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot, SpotMetric,
)
from nanobrag_torch.diffuse import GaussianHalo, bragg_peak_factor
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel, SerialExperiment
from nanobrag_torch.simulator import Simulator

DTYPE = torch.float64
P = 24
ROIS = [(2, 12, 3, 9), (14, 22, 10, 20)]
HALO = "diffuse_params.model"
K_D = "diffuse_params.delta_log_scale"


def _crystal(**kwargs):
    base = dict(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(5, 5, 5),
                default_F=100.0, shape=CrystalShape.GAUSS)
    return CrystalConfig(**{**base, **kwargs})


def _detector(**kwargs):
    base = dict(distance_mm=60.0, pixel_size_mm=0.4, spixels=P, fpixels=P, oversample=1,
                detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)
    return DetectorConfig(**{**base, **kwargs})


def _beam():
    return BeamConfig(wavelength_A=1.2, fluence=1e24)


def _halo():
    return GaussianHalo(amplitude=40.0, sigma=0.15)


def _model(diffuse=None, crystal=None, detector=None, **kwargs):
    detector = _detector(**(detector or {})) if not isinstance(detector, list) else detector
    return ExperimentModel(_crystal(**(crystal or {})), detector, _beam(), dtype=DTYPE, diffuse=diffuse, **kwargs)


def _simulator(diffuse, crystal=None, detector=None):
    """The bare Simulator the model should be running."""
    return Simulator(Crystal(_crystal(**(crystal or {})), dtype=DTYPE), Detector(_detector(**(detector or {})), dtype=DTYPE),
                     beam_config=_beam(), dtype=DTYPE, diffuse=diffuse)


# 1. The module is a submodule, its parameters a group present only with a model.

def test_the_module_is_a_submodule_and_its_parameters_are_the_diffuse_group():
    halo = _halo()
    model = _model(halo)
    assert model.diffuse_params.model is halo
    names = dict(model.named_parameters())
    assert names[f"{HALO}.log_amplitude"] is halo.log_amplitude
    assert set(model.PARAMETER_GROUPS["diffuse"]) == {f"{HALO}.log_amplitude", f"{HALO}.log_sigma"}
    assert model.PARAMETER_GROUPS["diffuse_scale"] == (K_D,)
    assert f"{HALO}.log_sigma" in model.state_dict() and K_D in model.state_dict()


def test_without_a_model_there_is_no_group_parameter_or_state():
    model = _model()
    assert "diffuse" not in model.PARAMETER_GROUPS and "diffuse_scale" not in model.PARAMETER_GROUPS
    assert not any("diffuse" in n for n in model.state_dict())
    assert not hasattr(model, "diffuse_params")
    assert "diffuse" not in ExperimentModel.PARAMETER_GROUPS  # the class's groups are not changed by a model
    _model(_halo())
    assert "diffuse" not in ExperimentModel.PARAMETER_GROUPS
    with pytest.raises(KeyError, match="unknown parameter or group 'diffuse'"):
        model.refine("diffuse")


def test_groups_cover_every_parameter_exactly_once_with_a_model():
    model = _model(_halo())
    grouped = [n for names in model.PARAMETER_GROUPS.values() for n in names]
    assert sorted(grouped) == sorted(n for n, _ in model.named_parameters())
    assert set(model.PARAMETER_GROUPS) - set(ExperimentModel.PARAMETER_GROUPS) == set(ExperimentModel.OPTIONAL_GROUPS)


def test_a_parameter_the_module_froze_stays_frozen_and_is_unrefinable():
    halo = _halo()
    halo.log_sigma.requires_grad_(False)
    model = _model(halo)
    assert f"{HALO}.log_sigma" in model.unrefinable
    model.refine("diffuse")
    assert not halo.log_sigma.requires_grad and halo.log_amplitude.requires_grad


def test_the_halo_trains_and_freezes_with_the_model():
    halo = _halo()
    model = _model(halo)
    assert halo.log_amplitude.requires_grad and halo.log_sigma.requires_grad
    model.freeze("diffuse")
    assert not halo.log_amplitude.requires_grad
    model.refine("diffuse")
    assert halo.log_sigma.requires_grad
    frozen = _model(_halo(), param_init="frozen")
    assert frozen.trainable_parameters() == []


# 2. k_d: log, frozen by default, scales only the diffuse part.

def test_k_d_is_frozen_by_default_and_refinable():
    model = _model(_halo())
    k_d = dict(model.named_parameters())[K_D]
    assert float(k_d) == 0.0 and not k_d.requires_grad
    assert all(p is not k_d for p in model.trainable_parameters())
    model.refine("diffuse_scale")
    assert k_d.requires_grad


SHAPES = {
    "square": dict(shape=CrystalShape.SQUARE),
    "gauss-reciprocal": dict(shape=CrystalShape.GAUSS),
    "gauss-hkl": dict(shape=CrystalShape.GAUSS, spot_metric=SpotMetric.HKL),
    "tophat": dict(shape=CrystalShape.TOPHAT, spot_metric=SpotMetric.HKL),
    "round": dict(shape=CrystalShape.ROUND),
}


def _expected_factor(crystal):
    config = _crystal(**crystal)
    cell = Crystal(config, dtype=DTYPE)
    return float(bragg_peak_factor(config.shape, torch.tensor(config.N_cells, dtype=DTYPE), config.spot_metric,
                                   volume=cell.V / cell.c_reciprocal_scale ** 3))


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_diffuse_part_is_the_simulators_times_the_shape_factor(shape):
    """The tie corrected for the shape: integrated Bragg / diffuse is |F|^2 / I_d (tests/test_diffuse_shape_integral.py)."""
    halo = _halo()
    model = _model(halo, crystal=SHAPES[shape])
    with torch.no_grad():
        image = model()
        sim = _simulator(halo, crystal=SHAPES[shape])
        sim.run()
    factor = _expected_factor(SHAPES[shape])
    assert float(model.diffuse_params.shape_factor) == pytest.approx(factor, rel=1e-14)
    # (not bit for bit under GAUSS-HKL, where the model's zero Ndef takes the matrix path: Simulator._n_cells_def)
    torch.testing.assert_close(model.components["bragg"], sim.components["bragg"], rtol=1e-12, atol=0.0)
    torch.testing.assert_close(model.components["diffuse"], factor * sim.components["diffuse"], rtol=1e-14, atol=0.0)
    assert torch.equal(model.components["bragg"] + model.components["diffuse"] + model.components["background"], image)
    if shape == "square":  # factor 1: the Simulator's tie, bit for bit
        assert torch.equal(model.components["diffuse"], sim.components["diffuse"])
    else:
        assert abs(factor - 1) > 1e-3


def test_the_shape_factor_can_be_turned_off():
    halo = _halo()
    model = _model(halo, diffuse_shape_factor=False)
    with torch.no_grad():
        model()
        sim = _simulator(halo)
        sim.run()
    assert float(model.diffuse_params.shape_factor) == 1.0
    assert torch.equal(model.components["diffuse"], sim.components["diffuse"])


def test_the_shape_factor_follows_the_crystal_size_and_cell():
    model = _model(_halo())
    with torch.no_grad():
        model.crystal_params.delta_log_N_cells.fill_(math.log(6.0 / 5.0))
        model.crystal_params.delta_log_a.fill_(math.log(32.0 / 30.0))
        model()
    expected = _expected_factor(dict(N_cells=(6, 6, 6), cell_a=32.0))
    assert float(model.diffuse_params.shape_factor) == pytest.approx(expected, rel=1e-12)
    assert expected != pytest.approx(_expected_factor({}), rel=1e-3)


def test_bragg_only_is_the_bragg_part_alone():
    for kwargs, run in [({}, {}), (dict(background_rois=ROIS), dict(roi_only=True))]:
        with torch.no_grad():
            with_halo = _model(_halo(), **kwargs)
            image = with_halo(bragg_only=True, **run)
            assert with_halo.components is None
            assert torch.equal(image, _model(**kwargs)(bragg_only=True, **run))
            full = with_halo(**run)
            assert torch.equal(with_halo.components["bragg"], image) and not torch.equal(full, image)
    panels = _panels(_halo(), background_rois=ROIS, background_panels=ROI_PANELS)
    plain = _panels(None, background_rois=ROIS, background_panels=ROI_PANELS)
    with torch.no_grad():
        for run in ({}, dict(roi_only=True)):
            assert torch.equal(panels(bragg_only=True, **run), plain(bragg_only=True, **run))


def test_k_d_scales_only_the_diffuse_component():
    model = _model(_halo())
    with torch.no_grad():
        model()
        before = {k: v.clone() for k, v in model.components.items()}
        dict(model.named_parameters())[K_D].fill_(math.log(3.0))
        model()
    after = model.components
    assert torch.equal(after["bragg"], before["bragg"])
    torch.testing.assert_close(after["diffuse"], 3.0 * before["diffuse"], rtol=1e-14, atol=0.0)
    torch.testing.assert_close(after["total"], after["bragg"] + after["diffuse"] + after["background"],
                               rtol=1e-15, atol=0.0)


def test_components_are_none_without_a_model():
    model = _model()
    model()
    assert model.components is None


# 3. The ROI path and panels carry the term; components come through the model.

PANELS = [(4.8, 4.8), (-5.0, 4.8)]
ROI_PANELS = [1, 0]


def _panels(diffuse, **kwargs):
    return _model(diffuse, detector=[_detector(beam_center_f=f, beam_center_s=s) for f, s in PANELS], **kwargs)


@pytest.mark.parametrize("detector", [{}, dict(oversample=2), dict(detector_thick_um=300.0, detector_abs_um=200.0,
                                                                  detector_thicksteps=3)])
def test_roi_pixels_are_the_whole_image_pixels_part_by_part(detector):
    model = _model(_halo(), detector=detector, background_rois=ROIS, tilt_abc=[(0.5, -0.25, 3.0), (1.0, 2.0, 7.0)])
    with torch.no_grad():
        whole = model()
        whole_parts = {k: v.clone() for k, v in model.components.items()}
        values = model(roi_only=True)
    index = tuple(model.background_params.pixels.T)
    torch.testing.assert_close(values, whole[index], rtol=1e-12, atol=0.0)
    for key, part in model.components.items():
        assert part.shape == values.shape
        torch.testing.assert_close(part, whole_parts[key][index], rtol=1e-12, atol=0.0)
    assert model.components["diffuse"].abs().min() > 0


def test_roi_across_panels_are_the_whole_images_at_their_pixels():
    model = _panels(_halo(), background_rois=ROIS, background_panels=ROI_PANELS)
    with torch.no_grad():
        whole = model()
        assert whole.shape == (2, P, P) and model.components["diffuse"].shape == (2, P, P)
        whole_diffuse = model.components["diffuse"].clone()
        values = model(roi_only=True)
    background = model.background_params
    index = (background.panel, *background.pixels.T)
    torch.testing.assert_close(values, whole[index], rtol=1e-12, atol=0.0)
    torch.testing.assert_close(model.components["diffuse"], whole_diffuse[index], rtol=1e-12, atol=0.0)


def test_panels_are_the_single_panel_simulators():
    halo = _halo()
    model = _panels(halo)
    with torch.no_grad():
        model()
        factor = _expected_factor({})
        for panel, (f, s) in enumerate(PANELS):
            sim = _simulator(halo, detector=dict(beam_center_f=f, beam_center_s=s))
            sim.run()
            torch.testing.assert_close(model.components["bragg"][panel], sim.components["bragg"], rtol=1e-12, atol=0.0)
            torch.testing.assert_close(model.components["diffuse"][panel], factor * sim.components["diffuse"],
                                       rtol=1e-12, atol=0.0)


def test_roi_centroids_follow_the_bragg_part():
    rois = [(0, P, 0, P)]
    with torch.no_grad():
        plain = _model(background_rois=rois).roi_centroids()
        with_halo = _model(GaussianHalo(amplitude=1e6, sigma=0.3), background_rois=rois).roi_centroids()
    torch.testing.assert_close(torch.as_tensor(with_halo), torch.as_tensor(plain), rtol=1e-12, atol=0.0)


# 4. SerialExperiment holds one diffuse model.

def _shot(halo, misset):
    return _model(halo, crystal=dict(misset_deg=misset))


MISSETS = [(5.0, -3.0, 2.0), (20.0, 10.0, -5.0)]


def test_shots_share_one_diffuse_block():
    halo = _halo()
    serial = SerialExperiment([_shot(halo, m) for m in MISSETS])
    block = serial.shots[0].diffuse_params
    assert all(shot.diffuse_params is block for shot in serial.shots)
    names = [n for n, _ in serial.named_parameters()]
    assert sum(n.endswith("log_sigma") for n in names) == 1
    assert sum(n.endswith("delta_log_scale") for n in names) == 1
    serial.refine("diffuse_scale")
    w = torch.rand(P, P, dtype=DTYPE, generator=torch.Generator().manual_seed(2))
    images = serial()
    sum((image * w).sum() for image in images).backward()
    expected = {"log_sigma": 0.0, "delta_log_scale": 0.0}
    for image, misset in zip(images, MISSETS):
        alone = _shot(_halo(), misset)
        alone.refine("diffuse_scale")
        reference = alone()
        torch.testing.assert_close(image, reference, rtol=1e-12, atol=0.0)
        (reference * w).sum().backward()
        expected["log_sigma"] += alone.diffuse_params.model.log_sigma.grad
        expected["delta_log_scale"] += alone.diffuse_params.delta_log_scale.grad
    torch.testing.assert_close(halo.log_sigma.grad, expected["log_sigma"], rtol=1e-11, atol=0.0)
    torch.testing.assert_close(block.delta_log_scale.grad, expected["delta_log_scale"], rtol=1e-11, atol=0.0)


def test_shots_must_share_the_diffuse_model_or_all_have_none():
    with pytest.raises(ValueError, match="same diffuse model"):
        SerialExperiment([_shot(_halo(), MISSETS[0]), _shot(_halo(), MISSETS[1])])
    with pytest.raises(ValueError, match="same diffuse model"):
        SerialExperiment([_shot(_halo(), MISSETS[0]), _model(crystal=dict(misset_deg=MISSETS[1]))])
    with pytest.raises(ValueError, match="same diffuse model"):
        SerialExperiment([_model(crystal=dict(misset_deg=MISSETS[0])), _shot(_halo(), MISSETS[1])])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_a_model_on_mps_in_float32_runs_in_a_cpu_float64_model():
    class OnMps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.halo = GaussianHalo(40.0, 0.15, dtype=torch.float32).to("mps")

        def forward(self, hkl):
            # cast on the CPU, then move: the backward of a combined .to(mps, float32) would make float64 on MPS
            return self.halo(hkl.to(dtype=torch.float32).to(device="mps"))

    model = _model(OnMps())
    image = model()
    image.sum().backward()
    reference = _model(GaussianHalo(40.0, 0.15, dtype=torch.float32))
    assert image.dtype == DTYPE and model.diffuse_params.model.halo.log_sigma.grad.device.type == "mps"
    torch.testing.assert_close(image, reference(), rtol=1e-5, atol=0.0)
