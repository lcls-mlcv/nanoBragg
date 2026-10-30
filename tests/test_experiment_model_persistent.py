"""ExperimentModel builds Crystal/Detector/Simulator once and updates them in place (plan C1).

The persistent path must give the same image and the same gradients as building
fresh models from the same parameter values, at every step of a parameter walk.
"""
import pytest
import torch

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


def _fresh_image(model):
    """Reference: build all three models from the current parameter values."""
    crystal_cfg = model.crystal_params.build_config()
    detector_cfg = model.detector_params.build_config()
    beam_cfg = model.beam_params.build_config()
    simulator = Simulator(Crystal(crystal_cfg, beam_cfg, dtype=DTYPE),
                          Detector(detector_cfg, dtype=DTYPE),
                          beam_config=beam_cfg, dtype=DTYPE)
    return simulator.run() + model.background_params.image()


def _set_step(model, k):
    """Move every Stage-A parameter to a step-dependent, non-zero value."""
    with torch.no_grad():
        for i, (_, p) in enumerate(sorted(model.named_parameters())):
            p.copy_(torch.full_like(p, 0.01 * (k + 1) * (1 + 0.3 * i)))


@pytest.mark.parametrize("steps", [3])
def test_persistent_forward_and_gradients_match_fresh_build(steps):
    model = ExperimentModel(*_configs(), dtype=DTYPE, param_init="stage_a")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(params) >= 10  # cell, misset, detector and fluence blocks are all live

    first_simulator = None
    for k in range(steps):
        _set_step(model, k)

        model.zero_grad()
        image = model()
        image.sum().backward()
        persistent_grads = {n: p.grad.clone() for n, p in params.items()}

        model.zero_grad()
        reference = _fresh_image(model)
        reference.sum().backward()
        fresh_grads = {n: p.grad.clone() for n, p in params.items()}

        assert image.sum() > 0
        torch.testing.assert_close(image, reference, rtol=1e-10, atol=0.0)
        for name in params:
            torch.testing.assert_close(persistent_grads[name], fresh_grads[name],
                                       rtol=1e-8, atol=1e-12, msg=f"d/d{name} at step {k}")
        assert any(g.abs().sum() > 0 for g in persistent_grads.values())

        simulator, *_ = model._build_simulator()
        if first_simulator is None:
            first_simulator = simulator
        assert simulator is first_simulator, "forward() rebuilt the Simulator"


def test_optimizer_steps_reuse_one_simulator(monkeypatch):
    built = []
    original_init = Simulator.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Simulator, "__init__", counting_init)

    model = ExperimentModel(*_configs(), dtype=DTYPE, param_init="stage_a")
    target = _fresh_image(model).detach() * 1.1
    built.clear()
    # This image is ~1e-3 photons a pixel: a step of lr=1e-3 in the plane, in photons, swamps it
    model.freeze("background")
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        loss = ((model() - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert len(built) == 1
    assert losses[-1] < losses[0]


def test_set_structure_factors_after_first_forward_reaches_the_live_crystal():
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    before = model().sum()
    grid = torch.full((9, 9, 9), 200.0, dtype=DTYPE)
    model.set_structure_factors(grid, dict(h_min=-4, h_max=4, k_min=-4, k_max=4,
                                           l_min=-4, l_max=4))
    after = model().sum()
    assert model._crystal.hkl_data is not None
    assert not torch.isclose(before, after)
