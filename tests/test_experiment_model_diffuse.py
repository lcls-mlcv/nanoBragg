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
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.diffuse import GaussianHalo
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


def test_the_image_is_bragg_plus_the_simulators_diffuse_term():
    halo = _halo()
    model = _model(halo)
    image = model(bragg_only=True)
    reference = _simulator(halo).run()
    assert torch.equal(image, reference)  # k_d frozen at 1 is the absolute tie of the Simulator
    components = model.components
    assert torch.equal(components["bragg"] + components["diffuse"], image)
    assert components["diffuse"].abs().max() > 0 and components["bragg"].abs().max() > 0


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
