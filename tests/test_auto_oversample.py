"""Auto oversample (-1) must not make a refinement loss discontinuous (plan item B2).

nanoBragg.c picks oversample = ceil(3 * xtalsize / reciprocal_pixel_size). With the
default geometry (lambda 1 A, 100 mm, 0.1 mm pixels) and N = 5, cell_a = 200 A puts
the argument exactly on 3.0, so re-selecting it every run flips 3 -> 4 across a 1e-6
step and the summed image jumps (numerical d/da = -3.3e5 against an analytical -71).
"""
import warnings

import pytest
import torch
from torch.autograd import gradcheck

from nanobrag_torch.config import CrystalConfig, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator

DTYPE = torch.float64


def _simulator(cell_a):
    config = CrystalConfig(cell_a=cell_a, cell_b=100.0, cell_c=100.0,
                           cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0,
                           N_cells=(5, 5, 5), default_F=100.0)
    detector = Detector(DetectorConfig(spixels=64, fpixels=64,
                                       beam_center_s=3.2, beam_center_f=3.2), dtype=DTYPE)
    return Simulator(Crystal(config, dtype=DTYPE), detector, crystal_config=config, dtype=DTYPE)


def _set_cell_a(sim, value):
    # The physics reads Crystal.cell_a; the auto-oversample formula reads the config.
    sim.crystal.cell_a = value
    sim.crystal.config.cell_a = value


def _selected(capsys):
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("auto-selected")]
    return [int(l.split()[1].split("-")[0]) for l in lines]


def test_persistent_simulator_is_continuous_at_the_oversample_boundary():
    leaf = torch.tensor(200.0, dtype=DTYPE, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _simulator(leaf)

        def loss(cell_a):
            _set_cell_a(sim, cell_a)
            return sim.run().sum()

        assert gradcheck(loss, (leaf,), eps=1e-6, atol=1e-5, rtol=0.05, raise_exception=True)


def test_auto_value_frozen_and_warned_when_an_input_requires_grad(capsys):
    sim = _simulator(torch.tensor(200.0, dtype=DTYPE, requires_grad=True))
    with pytest.warns(UserWarning, match="keeping the auto-selected 3-fold"):
        sim.run()
    _set_cell_a(sim, torch.tensor(400.0, dtype=DTYPE, requires_grad=True))
    sim.run()
    assert _selected(capsys) == [3]  # selected once; the second run reuses it silently


def test_auto_value_recomputed_each_run_without_grad(capsys):
    sim = _simulator(200.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no freeze warning without differentiable inputs
        sim.run()
        _set_cell_a(sim, torch.tensor(400.0, dtype=DTYPE))
        sim.run()
    assert _selected(capsys) == [3, 6]
