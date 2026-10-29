"""ExperimentModel parameters: trainable by default, chosen with refine()/freeze() (plan C2)."""
import pytest
import torch

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.simulator import Simulator

DTYPE = torch.float64


def _configs():
    crystal = CrystalConfig(cell_a=70.0, cell_b=80.0, cell_c=90.0,
                            cell_alpha=88.0, cell_beta=91.0, cell_gamma=93.0,
                            misset_deg=(5.0, -3.0, 2.0), N_cells=(4, 4, 4),
                            default_F=100.0)
    detector = DetectorConfig(distance_mm=80.0, pixel_size_mm=0.2, spixels=48, fpixels=48,
                              detector_convention=DetectorConvention.MOSFLM,
                              detector_pivot=DetectorPivot.BEAM, oversample=1)
    beam = BeamConfig(wavelength_A=1.2, fluence=1e24)
    return crystal, detector, beam


def _config_image():
    crystal, detector, beam = _configs()
    return Simulator(Crystal(crystal, beam, dtype=DTYPE), Detector(detector, dtype=DTYPE),
                     beam_config=beam, dtype=DTYPE).run()


def _grads(model):
    model.zero_grad(set_to_none=True)
    model().sum().backward()
    return {n: p.grad for n, p in model.named_parameters()}


def test_default_is_trainable_and_reproduces_the_config_image():
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    assert all(p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model(), _config_image(), rtol=1e-12, atol=0.0)
    assert all(g is not None for g in _grads(model).values())


def test_groups_cover_every_parameter_exactly_once():
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    grouped = [n for names in ExperimentModel.PARAMETER_GROUPS.values() for n in names]
    assert sorted(grouped) == sorted(n for n, _ in model.named_parameters())


def test_freeze_then_refine_one_group():
    model = ExperimentModel(*_configs(), dtype=DTYPE).freeze().refine("misset")
    assert model.trainable_parameters() == [model.crystal_params.delta_misset_raw]
    grads = _grads(model)
    assert grads["crystal_params.delta_misset_raw"].abs().sum() > 0
    assert all(g is None for n, g in grads.items() if n != "crystal_params.delta_misset_raw")


def test_frozen_init_can_be_unfrozen():
    model = ExperimentModel(*_configs(), dtype=DTYPE, param_init="frozen")
    assert model.trainable_parameters() == []
    torch.testing.assert_close(model(), _config_image(), rtol=1e-12, atol=0.0)
    model.refine("cell", "crystal_params.delta_log_a")  # group and name may overlap
    assert len(model.trainable_parameters()) == 6
    assert _grads(model)["crystal_params.delta_log_a"].abs() > 0


def test_frozen_parameter_holds_its_value_during_optimization():
    model = ExperimentModel(*_configs(), dtype=DTYPE).freeze("cell")
    cell_before = {n: p.detach().clone() for n, p in model.named_parameters()
                   if n in ExperimentModel.PARAMETER_GROUPS["cell"]}
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-2)
    for _ in range(2):
        optimizer.zero_grad()
        model().sum().backward()
        optimizer.step()
    for n, p in model.named_parameters():
        if n in cell_before:
            assert torch.equal(p, cell_before[n])
    assert model.crystal_params.delta_misset_raw.abs().sum() > 0  # the rest moved


def test_bad_names_are_rejected():
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    with pytest.raises(KeyError, match="unknown parameter or group 'cel'"):
        model.refine("cel")
    with pytest.raises(ValueError, match="param_init"):
        ExperimentModel(*_configs(), dtype=DTYPE, param_init="learn")


def test_stage_a_is_an_alias_for_trainable():
    model = ExperimentModel(*_configs(), dtype=DTYPE, param_init="stage_a")
    assert all(p.requires_grad for p in model.parameters())
