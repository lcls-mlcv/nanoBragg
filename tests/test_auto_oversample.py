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

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig, DetectorConvention, DetectorPivot
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator

pytestmark = pytest.mark.usefixtures("library_log_on_stdout")

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
    with pytest.warns(UserWarning, match="keeping the auto-selected 3-fold"):
        sim = _simulator(torch.tensor(200.0, dtype=DTYPE, requires_grad=True))
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


def _leaf(value):
    return torch.tensor(value, dtype=DTYPE, requires_grad=True)


def _vector(*values):
    return tuple(_leaf(float(v)) for v in values)


def _made(crystal_args=None, detector_args=None, beam_args=None):
    crystal_args = dict(dict(cell_a=100.0, cell_b=100.0, cell_c=100.0, N_cells=(30, 30, 30), default_F=100.0),
                        **(crystal_args or {}))
    detector_args = dict(dict(distance_mm=100.0, pixel_size_mm=0.1, spixels=8, fpixels=8), **(detector_args or {}))
    crystal = Crystal(CrystalConfig(**crystal_args), dtype=DTYPE)
    detector = Detector(DetectorConfig(**detector_args), dtype=DTYPE)
    return Simulator(crystal, detector, beam_config=BeamConfig(fluence=1e24, **(beam_args or {})), dtype=DTYPE)


CUSTOM = dict(detector_convention=DetectorConvention.CUSTOM)
INPUTS = {
    "cell_a": (dict(crystal_args=dict(cell_a=_leaf(100.0))), "cell_a"),
    "cell_b": (dict(crystal_args=dict(cell_b=_leaf(100.0))), "cell_b"),
    "cell_c": (dict(crystal_args=dict(cell_c=_leaf(100.0))), "cell_c"),
    "cell_alpha": (dict(crystal_args=dict(cell_alpha=_leaf(90.0))), "cell_alpha"),
    "cell_beta": (dict(crystal_args=dict(cell_beta=_leaf(90.0))), "cell_beta"),
    "cell_gamma": (dict(crystal_args=dict(cell_gamma=_leaf(90.0))), "cell_gamma"),
    "N_cells": (dict(crystal_args=dict(N_cells=(_leaf(30.0), 30, 30))), "N_cells"),
    "misset_deg": (dict(crystal_args=dict(misset_deg=_vector(1, 2, 3))), "misset_deg"),
    "wavelength": (dict(beam_args=dict(wavelength_A=_leaf(1.0))), "wavelength"),
    "distance": (dict(detector_args=dict(distance_mm=_leaf(100.0))), "distance_mm"),
    "close_distance": (dict(detector_args=dict(close_distance_mm=_leaf(100.0))), "close_distance_mm"),
    "pixel_size": (dict(detector_args=dict(pixel_size_mm=_leaf(0.1))), "pixel_size_mm"),
    "beam_center_f": (dict(detector_args=dict(beam_center_f=_leaf(1.0))), "beam_center_f"),
    "beam_center_s": (dict(detector_args=dict(beam_center_s=_leaf(1.0))), "beam_center_s"),
    "rotx": (dict(detector_args=dict(detector_rotx_deg=_leaf(5.0))), "detector_rotx_deg"),
    "roty": (dict(detector_args=dict(detector_roty_deg=_leaf(5.0))), "detector_roty_deg"),
    "rotz": (dict(detector_args=dict(detector_rotz_deg=_leaf(5.0))), "detector_rotz_deg"),
    "twotheta": (dict(detector_args=dict(detector_twotheta_deg=_leaf(5.0))), "detector_twotheta_deg"),
    "twotheta_axis": (dict(detector_args=dict(detector_twotheta_deg=5.0, twotheta_axis=_leaf([0.0, 0.0, -1.0]))),
                      "twotheta_axis"),
    "fdet": (dict(detector_args=dict(custom_fdet_vector=_vector(0, 0, 1), custom_sdet_vector=(0.0, -1.0, 0.0),
                                     custom_odet_vector=(1.0, 0.0, 0.0), **CUSTOM)), "custom_fdet_vector"),
    "odet": (dict(detector_args=dict(custom_fdet_vector=(0.0, 0.0, 1.0), custom_sdet_vector=(0.0, -1.0, 0.0),
                                     custom_odet_vector=_vector(1, 0, 0), **CUSTOM)), "custom_odet_vector"),
    "pix0": (dict(detector_args=dict(pix0_override_m=_leaf([0.1, 0.0, 0.0]))), "pix0_override_m"),
}


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_every_input_freezes_at_construction_and_says_which(name):
    kwargs, reported = INPUTS[name]
    with pytest.warns(UserWarning, match=rf"differentiable inputs \(.*{reported}.*\): keeping the auto-selected"):
        sim = _made(**kwargs)
    assert sim._frozen_auto_oversample >= 1
    assert sim._auto_oversample() == sim._frozen_auto_oversample


def test_every_detector_geometry_field_is_an_input():
    sim = _made()
    assert set(Detector._GEOMETRY_FIELDS) <= set(sim._auto_oversample_inputs())


def test_a_later_leaf_freezes_at_the_next_run():
    sim = _made()
    assert sim._frozen_auto_oversample is None
    sim.detector.config.detector_twotheta_deg = _leaf(20.0)
    with pytest.warns(UserWarning, match="detector_twotheta_deg"):
        sim.run()
    assert sim._frozen_auto_oversample == 10


def test_no_freeze_with_an_explicit_oversample():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim = _made(detector_args=dict(detector_twotheta_deg=_leaf(5.0), oversample=2))
    assert sim._frozen_auto_oversample is None


@pytest.mark.parametrize("pivot", [DetectorPivot.BEAM, DetectorPivot.SAMPLE], ids=["beam", "sample"])
def test_moving_leaf_cannot_step_the_oversample(pivot):
    """twotheta takes the auto value 9 -> 18 by 60 degrees: frozen, it stays at the first."""
    leaf = _leaf(0.0)
    with pytest.warns(UserWarning):
        sim = _made(detector_args=dict(detector_twotheta_deg=leaf, detector_pivot=pivot))
    for degrees in (20.0, 30.0, 40.0, 60.0):
        leaf.data.fill_(degrees)
        sim.detector.get_pixel_coords()
        assert sim._auto_oversample() == 9


def test_the_loss_does_not_jump_across_the_crossing():
    """Between these two angles the free auto value steps from 9 to 10."""
    low, high = 1.0456460586283356e-06, 1.045664248522371e-06
    leaf = _leaf(low)
    with pytest.warns(UserWarning):
        sim = _made(detector_args=dict(detector_twotheta_deg=leaf, spixels=24, fpixels=24))
    a = sim.run().sum().item()
    leaf.data.fill_(high)
    b = sim.run().sum().item()
    assert abs(b - a) / a < 1e-11


def test_without_a_leaf_it_still_follows_the_geometry():
    assert [_made(detector_args=dict(detector_twotheta_deg=d))._auto_oversample() for d in (0.0, 20.0, 60.0)] == [9, 10, 18]
