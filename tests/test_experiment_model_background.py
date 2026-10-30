"""Refinable background planes in ExperimentModel (plan C2).

tests/test_diffbragg_parity.py checks the planes against diffBragg's own fit on a cctbx
build. diffBragg holds the background fixed, so it has no gradient to compare; the
gradient here is checked against the plane's basis and with gradcheck.
"""
import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.experiment import BackgroundStageAParams, ExperimentModel
from nanobrag_torch.simulator import Simulator

DTYPE = torch.float64
ROIS = [(2, 7, 3, 9), (10, 14, 1, 5)]
ABC = [(0.5, -0.25, 3.0), (-1.0, 2.0, 7.0)]


def _configs(pixels=16, **detector):
    crystal = CrystalConfig(cell_a=30.0, cell_b=30.0, cell_c=30.0, misset_deg=(5.0, -3.0, 2.0),
                            N_cells=(5, 5, 5), default_F=100.0, shape=CrystalShape.GAUSS)
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=pixels, fpixels=pixels,
                              detector_convention=DetectorConvention.MOSFLM,
                              detector_pivot=DetectorPivot.BEAM, oversample=1, **detector)
    return crystal, detector, BeamConfig(wavelength_A=1.2, fluence=1e24)


def _expected_planes(pixels=16):
    expected = np.zeros((pixels, pixels))
    for (x1, x2, y1, y2), (a, b, c) in zip(ROIS, ABC):
        y, x = np.indices((y2 - y1, x2 - x1))
        expected[y1:y2, x1:x2] = a * x + b * y + c
    return torch.as_tensor(expected)


def test_default_is_one_zero_plane_that_trains_and_leaves_the_image_alone():
    crystal_cfg, detector_cfg, beam_cfg = _configs()
    model = ExperimentModel(crystal_cfg, detector_cfg, beam_cfg, dtype=DTYPE)
    assert model.background_params.delta_tilt_abc.shape == (1, 3)
    assert model.background_params.delta_tilt_abc.requires_grad
    with torch.no_grad():
        image = model()
    crystal = Crystal(crystal_cfg, beam_cfg, dtype=DTYPE)
    reference = Simulator(crystal, Detector(detector_cfg, dtype=DTYPE), beam_config=beam_cfg, dtype=DTYPE).run()
    assert torch.equal(image, reference)


def test_plane_origin_is_the_roi_corner_and_pixels_outside_get_nothing():
    planes = BackgroundStageAParams(16, 16, ROIS, ABC, dtype=DTYPE)
    torch.testing.assert_close(planes.image(), _expected_planes(), rtol=0, atol=0)


def test_later_roi_paints_over_an_earlier_one():
    planes = BackgroundStageAParams(4, 4, [(0, 3, 0, 3), (1, 4, 1, 4)], [(0, 0, 1.0), (1.0, 0, 5.0)], dtype=DTYPE)
    expected = [[1, 1, 1, 0], [1, 5, 6, 7], [1, 5, 6, 7], [0, 5, 6, 7]]
    torch.testing.assert_close(planes.image(), torch.tensor(expected, dtype=DTYPE))


def test_background_is_added_after_scale_and_respects_the_mask():
    crystal_cfg, detector_cfg, beam_cfg = _configs(roi_xmin=0, roi_xmax=11, roi_ymin=0, roi_ymax=15)
    model = ExperimentModel(crystal_cfg, detector_cfg, beam_cfg, dtype=DTYPE, background_rois=ROIS, tilt_abc=ABC)
    bare = ExperimentModel(crystal_cfg, detector_cfg, beam_cfg, dtype=DTYPE)
    with torch.no_grad():
        for m in (model, bare):
            m.beam_params.delta_log_spot_scale.fill_(1.0)
        difference = model() - bare()
    expected = _expected_planes()
    expected[:, 12:] = 0  # the second ROI reaches past roi_xmax
    torch.testing.assert_close(difference, expected, rtol=0, atol=1e-9)


def test_delta_is_the_change_across_the_region():
    planes = BackgroundStageAParams(16, 16, ROIS, ABC, dtype=DTYPE)
    with torch.no_grad():
        planes.delta_tilt_abc.copy_(torch.tensor([(5.0, 6.0, 1.0), (8.0, 4.0, 0.0)], dtype=DTYPE))
    expected = [(0.5 + 5 / 5, -0.25 + 6 / 6, 4.0), (-1.0 + 8 / 4, 2.0 + 4 / 4, 7.0)]
    torch.testing.assert_close(planes.tilt_abc(), torch.tensor(expected, dtype=DTYPE))


def test_gradient_is_the_plane_basis():
    crystal_cfg, detector_cfg, beam_cfg = _configs()
    model = ExperimentModel(crystal_cfg, detector_cfg, beam_cfg, dtype=DTYPE, background_rois=ROIS, tilt_abc=ABC)
    w = torch.rand(16, 16, dtype=DTYPE, generator=torch.Generator().manual_seed(1))
    (model() * w).sum().backward()
    expected = []
    for x1, x2, y1, y2 in ROIS:
        y, x = torch.meshgrid(torch.arange(y2 - y1), torch.arange(x2 - x1), indexing="ij")
        patch = w[y1:y2, x1:x2]
        expected.append(torch.stack([(patch * x).sum() / (x2 - x1), (patch * y).sum() / (y2 - y1), patch.sum()]))
    torch.testing.assert_close(model.background_params.delta_tilt_abc.grad, torch.stack(expected), rtol=1e-12, atol=0)


def test_gradcheck_background():
    model = ExperimentModel(*_configs(pixels=16), dtype=DTYPE, background_rois=ROIS, tilt_abc=ABC)
    name = "background_params.delta_tilt_abc"
    start = torch.linspace(-0.2, 0.2, 6, dtype=DTYPE).reshape(2, 3).requires_grad_(True)

    def loss(value):
        return functional_call(model, {name: value}, ()).sum()

    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-8, rtol=1e-4)


def test_refinement_recovers_the_planes():
    with torch.no_grad():
        data = ExperimentModel(*_configs(), dtype=DTYPE, background_rois=ROIS, tilt_abc=ABC)()
    model = ExperimentModel(*_configs(), dtype=DTYPE, background_rois=ROIS)
    model.freeze().refine("background")
    optimizer = torch.optim.LBFGS(model.trainable_parameters(), max_iter=50, tolerance_grad=1e-12,
                                  tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (model() - data).pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    torch.testing.assert_close(model.background_params.tilt_abc(), torch.tensor(ABC, dtype=DTYPE), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("roi", [(0, 17, 0, 4), (5, 5, 0, 4), (-1, 3, 0, 4)])
def test_bad_rois_are_rejected(roi):
    with pytest.raises(ValueError, match="empty or outside"):
        BackgroundStageAParams(16, 16, [roi], dtype=DTYPE)


def test_tilt_abc_must_match_the_rois():
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        BackgroundStageAParams(16, 16, ROIS, [(1.0, 2.0, 3.0)], dtype=DTYPE)
