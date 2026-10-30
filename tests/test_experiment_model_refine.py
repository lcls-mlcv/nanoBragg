"""ExperimentModel parameters: trainable by default, chosen with refine()/freeze() (plan C2)."""
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


def _configs(shape=CrystalShape.GAUSS):
    # GAUSS so continuous Ncells is refinable (it is not under SQUARE).
    crystal = CrystalConfig(cell_a=70.0, cell_b=80.0, cell_c=90.0,
                            cell_alpha=88.0, cell_beta=91.0, cell_gamma=93.0,
                            misset_deg=(5.0, -3.0, 2.0), N_cells=(4, 4, 4),
                            default_F=100.0, shape=shape)
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
    # eta has nothing to refine without mosaic spread, detz_shift is for origin-defined
    # (dxtbx) detectors; every other parameter trains
    assert set(model.unrefinable) == {"crystal_params.delta_log_eta", "detector_params.delta_detz_shift_mm",
                                       "fhkl_params.delta_log_F", "beam_params.delta_log_source_weights"}
    assert all(p.requires_grad for n, p in model.named_parameters() if n not in model.unrefinable)
    torch.testing.assert_close(model(), _config_image(), rtol=1e-12, atol=0.0)
    assert all(g is not None for n, g in _grads(model).items() if n not in model.unrefinable)


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
    assert all(p.requires_grad for n, p in model.named_parameters() if n not in model.unrefinable)


def test_scale_multiplies_the_image():
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    with torch.no_grad():
        base = model()
        model.beam_params.delta_log_spot_scale.fill_(torch.log(torch.tensor(2.5, dtype=DTYPE)))
        scaled = model()
    torch.testing.assert_close(scaled, 2.5 * base, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("name", ["beam_params.delta_log_spot_scale",
                                  "crystal_params.delta_log_N_cells"])
def test_gradcheck_scale_and_ncells(name):
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    start = torch.full_like(dict(model.named_parameters())[name], 0.05).requires_grad_(True)

    def loss(value):
        return functional_call(model, {name: value}, ()).sum()

    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-8, rtol=1e-4)


def test_continuous_ncells_moves_the_image_smoothly():
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    sums = []
    with torch.no_grad():
        for n in (4.0, 4.3, 4.6):
            model.crystal_params.delta_log_N_cells.fill_(torch.log(torch.tensor(n / 4.0, dtype=DTYPE)))
            sums.append(model().sum().item())
    assert sums[0] < sums[1] < sums[2] < 2 * sums[0]


def test_ncells_frozen_and_unrefinable_under_square():
    model = ExperimentModel(*_configs(CrystalShape.SQUARE), dtype=DTYPE)
    ncells = model.crystal_params.delta_log_N_cells
    assert not ncells.requires_grad
    assert all(p.requires_grad for n, p in model.named_parameters()
               if p is not ncells and n not in model.unrefinable)
    with pytest.raises(ValueError, match="not refinable with shape=SQUARE"):
        model.refine("Ncells")
    model.freeze().refine()  # refine-all skips it instead of raising
    assert not ncells.requires_grad
    assert set(model.unrefinable) == {"crystal_params.delta_log_N_cells", "crystal_params.delta_log_eta",
                                      "detector_params.delta_detz_shift_mm",
                                       "fhkl_params.delta_log_F", "beam_params.delta_log_source_weights"}
    assert len(model.trainable_parameters()) == len(list(model.parameters())) - len(model.unrefinable)
