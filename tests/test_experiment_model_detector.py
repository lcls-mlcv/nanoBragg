"""Detector distance and diffBragg's detz_shift in ExperimentModel (plan C2).

Distance must refine whichever config field drives the geometry; a detector
given by its origin (dxtbx panel) refines detz_shift, a lab-z origin shift.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.compat.cctbx import detector_config_from_dxtbx_panel
from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.experiment import ExperimentModel

DTYPE = torch.float64
DETZ = "detector_params.delta_detz_shift_mm"
DIST = "detector_params.delta_log_distance_mm"


def _crystal():
    return CrystalConfig(cell_a=50.0, cell_b=60.0, cell_c=70.0, cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0,
                         misset_deg=(10.0, 20.0, 30.0), N_cells=(5, 5, 5), default_F=100.0,
                         shape=CrystalShape.GAUSS)


def _beam():
    return BeamConfig(wavelength_A=1.0, fluence=1e24)


def _panel(origin=(-3.2, 3.2, -80.0), tilt_deg=0.0):
    """dxtbx-like panel, optionally tilted about (1, 1, 0) so both its axes get z components."""
    t = math.radians(tilt_deg)
    k = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
    fast, slow = rot @ np.array([1.0, 0.0, 0.0]), rot @ np.array([0.0, -1.0, 0.0])
    return SimpleNamespace(get_fast_axis=lambda: tuple(fast), get_slow_axis=lambda: tuple(slow),
                           get_origin=lambda: tuple(origin), get_pixel_size=lambda: (0.1, 0.1),
                           get_image_size=lambda: (64, 64))


def _dxtbx_model(origin=(-3.2, 3.2, -80.0), tilt_deg=0.0, **kwargs):
    det = detector_config_from_dxtbx_panel(_panel(origin, tilt_deg), (0, 0, -1), oversample=1)
    return ExperimentModel(_crystal(), det, _beam(), dtype=DTYPE, **kwargs)


def test_dxtbx_detector_refines_detz_shift_not_distance():
    model = _dxtbx_model()
    assert model.detector_params.origin_defined
    assert DIST in model.unrefinable and DETZ not in model.unrefinable
    with pytest.raises(ValueError, match="refine 'detz_shift'"):
        model.refine("distance")
    model().sum().backward()
    assert model.detector_params.delta_detz_shift_mm.grad.abs() > 0


@pytest.mark.parametrize("tilt_deg", [0.0, 12.0])
def test_detz_shift_equals_moving_the_panel_origin(tilt_deg):
    """A step of dz mm gives the image of the same panel with origin z + dz."""
    origin = np.array([-3.2, 3.2, -80.0])
    model = _dxtbx_model(tuple(origin), tilt_deg)
    model()  # build first, so the step goes through the persistent objects
    with torch.no_grad():
        model.detector_params.delta_detz_shift_mm.fill_(-1.5)
        stepped = model()
        moved = _dxtbx_model(tuple(origin + [0.0, 0.0, -1.5]), tilt_deg, param_init="frozen")()
    torch.testing.assert_close(stepped, moved, rtol=1e-9, atol=0.0)


def test_gradcheck_detz_shift():
    model = _dxtbx_model(tilt_deg=12.0)
    start = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)

    def loss(value):
        return functional_call(model, {DETZ: value}, ()).sum()

    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-8, rtol=1e-4)


def _close_distance_config(close_mm):
    return DetectorConfig(close_distance_mm=close_mm, pixel_size_mm=0.1, spixels=64, fpixels=64,
                          detector_convention=DetectorConvention.MOSFLM,
                          detector_pivot=DetectorPivot.SAMPLE, oversample=1)


def test_distance_refines_close_distance_when_it_drives():
    """With -close_distance, distance_mm is derived; refining it used to have no effect."""
    model = ExperimentModel(_crystal(), _close_distance_config(80.0), _beam(), dtype=DTYPE)
    assert DETZ in model.unrefinable and DIST not in model.unrefinable
    model().sum().backward()
    assert model.detector_params.delta_log_distance_mm.grad.abs() > 0
    with torch.no_grad():
        model.detector_params.delta_log_distance_mm.fill_(math.log(1.05))
        stepped = model()
        fresh = ExperimentModel(_crystal(), _close_distance_config(84.0), _beam(), dtype=DTYPE)()
    torch.testing.assert_close(stepped, fresh, rtol=1e-9, atol=0.0)


def test_c_convention_detector_refines_distance():
    det = DetectorConfig(distance_mm=80.0, pixel_size_mm=0.1, spixels=64, fpixels=64,
                         detector_convention=DetectorConvention.MOSFLM,
                         detector_pivot=DetectorPivot.BEAM, oversample=1)
    model = ExperimentModel(_crystal(), det, _beam(), dtype=DTYPE)
    assert not model.detector_params.origin_defined
    with pytest.raises(ValueError, match="refine 'distance' for this detector"):
        model.refine("detz_shift")
    model().sum().backward()
    assert model.detector_params.delta_log_distance_mm.grad.abs() > 0
