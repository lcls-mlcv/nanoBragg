"""Several panels and several shots in ExperimentModel (plan C5).

tests/test_diffbragg_parity.py compares shoeboxes on a multi-panel dxtbx detector with
diffBragg's pan_fast_slow simulation.
"""
import numpy as np
import pytest
import torch

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.experiment import ExperimentModel, SerialExperiment
from nanobrag_torch.utils.losses import gaussian_nll
from nanobrag_torch.utils.structure_factors import StructureFactors

DTYPE = torch.float64
P = 32


def _crystal(misset=(5.0, -3.0, 2.0)):
    return CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=misset, N_cells=(5, 5, 5),
                         default_F=100.0, shape=CrystalShape.GAUSS)


def _panel(beam_f=6.4, beam_s=6.4):
    return DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=P, fpixels=P, oversample=1,
                          beam_center_f=beam_f, beam_center_s=beam_s,
                          detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)


def _beam():
    return BeamConfig(wavelength_A=1.2, fluence=1e24)


PANELS = [(6.4, 6.4), (-7.0, 6.4), (6.4, -7.0)]
ROIS = [(2, 12, 3, 9), (15, 24, 10, 22), (1, 6, 1, 6), (10, 20, 10, 20)]
ROI_PANELS = [2, 0, 0, 1]
ABC = [(0.5, -0.25, 3.0), (-1.0, 2.0, 7.0), (0.0, 0.0, 1.0), (0.1, 0.1, 0.0)]


def _multi(**kwargs):
    return ExperimentModel(_crystal(), [_panel(*p) for p in PANELS], _beam(), dtype=DTYPE, **kwargs)


def test_panels_are_the_single_panel_images():
    model = _multi()
    with torch.no_grad():
        images = model()
    assert images.shape == (3, P, P)
    for image, panel in zip(images, PANELS):
        with torch.no_grad():
            alone = ExperimentModel(_crystal(), _panel(*panel), _beam(), dtype=DTYPE)()
        torch.testing.assert_close(image, alone, rtol=1e-12, atol=0.0)
    assert not torch.allclose(images[0], images[1])


def test_panels_share_one_set_of_offsets():
    model = _multi()
    alone = ExperimentModel(_crystal(), _panel(), _beam(), dtype=DTYPE)
    assert len(list(model.parameters())) == len(list(alone.parameters()))  # one background plane is per panel


def test_a_step_moves_every_panel():
    model = _multi()
    with torch.no_grad():
        model()
        model.detector_params.delta_log_distance_mm.fill_(0.05)
        model.beam_params.delta_log_spot_scale.fill_(0.2)
        moved = model()
    expected = _multi()
    with torch.no_grad():
        expected.detector_params.delta_log_distance_mm.fill_(0.05)
        expected.beam_params.delta_log_spot_scale.fill_(0.2)
        torch.testing.assert_close(moved, expected(), rtol=1e-12, atol=0.0)


def test_shared_offset_collects_the_gradient_of_every_panel():
    model = _multi()
    w = torch.rand(3, P, P, dtype=DTYPE, generator=torch.Generator().manual_seed(1))
    (model() * w).sum().backward()
    expected = torch.zeros((), dtype=DTYPE)
    for i, panel in enumerate(PANELS):
        alone = ExperimentModel(_crystal(), _panel(*panel), _beam(), dtype=DTYPE)
        (alone() * w[i]).sum().backward()
        expected = expected + alone.detector_params.delta_log_distance_mm.grad
    torch.testing.assert_close(model.detector_params.delta_log_distance_mm.grad, expected, rtol=1e-11, atol=0.0)


def test_rois_across_panels_are_the_full_images_at_their_pixels():
    model = _multi(background_rois=ROIS, tilt_abc=ABC, background_panels=ROI_PANELS)
    background = model.background_params
    assert background.panel.tolist() == sorted(background.panel.tolist())
    with torch.no_grad():
        full, roi = model(), model(roi_only=True)
    assert roi.shape == (sum((x2 - x1) * (y2 - y1) for x1, x2, y1, y2 in ROIS),)
    torch.testing.assert_close(roi, model.roi_values(full), rtol=1e-13, atol=0.0)
    # the plane is on the panel its ROI names
    x1, x2, y1, y2 = ROIS[0]
    y, x = np.indices((y2 - y1, x2 - x1))
    planes = background.image()
    torch.testing.assert_close(planes[2, y1:y2, x1:x2], torch.as_tensor(0.5 * x - 0.25 * y + 3.0), rtol=1e-13, atol=0)
    assert planes[1, y1:y2, x1:x2].abs().sum() == 0


def test_rois_of_all_panels_take_one_kernel_call_and_no_dense_coordinates():
    model = _multi(background_rois=ROIS, tilt_abc=ABC, background_panels=ROI_PANELS)
    with torch.no_grad():
        model(roi_only=True)
        assert len(model._simulators) == 1
        assert all(d._pixel_coords_cache is None for d in model._detectors[1:])
        calls = []
        run = model._simulator._pixel_intensity
        model._simulator._pixel_intensity = lambda *a, **k: calls.append(1) or run(*a, **k)
        model(roi_only=True)
    assert len(calls) == 1


@pytest.mark.parametrize("case", [
    dict(oversample=2),
    dict(oversample=2, oversample_omega=True),
    dict(detector_thick_um=100.0, detector_thicksteps=3, detector_abs_um=500.0),
    dict(roi_xmin=4, roi_xmax=20, roi_ymin=2, roi_ymax=29),
])
def test_rois_across_panels_on_every_kernel_path(case):
    panels = [_panel(*p) for p in PANELS]
    for panel in panels:
        for key, value in case.items():
            setattr(panel, key, value)
    model = ExperimentModel(_crystal(), panels, _beam(), dtype=DTYPE, background_rois=ROIS, tilt_abc=ABC,
                            background_panels=ROI_PANELS)
    with torch.no_grad():
        full = model()
        assert full.abs().sum() > 0
        torch.testing.assert_close(model(roi_only=True), model.roi_values(full), rtol=1e-12, atol=1e-16)
        torch.testing.assert_close(model(roi_only=True, pixel_batch_size=7), model.roi_values(full),
                                   rtol=1e-12, atol=1e-16)


def test_each_pixel_uses_its_own_panels_distance():
    panels = [_panel(*p) for p in PANELS]
    panels[1].distance_mm = 90.0
    model = ExperimentModel(_crystal(), panels, _beam(), dtype=DTYPE, background_rois=ROIS, tilt_abc=ABC,
                            background_panels=ROI_PANELS)
    with torch.no_grad():
        full = model()
        torch.testing.assert_close(model(roi_only=True), model.roi_values(full), rtol=1e-12, atol=1e-16)


def test_rois_across_panels_gradient_matches_whole_images():
    rois_model = _multi(background_rois=ROIS, tilt_abc=ABC, background_panels=ROI_PANELS)
    selected = torch.zeros(3, P, P, dtype=DTYPE)
    background = rois_model.background_params
    selected[(background.panel, *background.pixels.T)] = 1.0
    w = torch.rand(3, P, P, dtype=DTYPE, generator=torch.Generator().manual_seed(4))
    params = [p for p in rois_model.parameters() if p.requires_grad]
    expected = torch.autograd.grad((rois_model() * w * selected).sum(), params, allow_unused=True)
    got = torch.autograd.grad((rois_model(roi_only=True) * rois_model.roi_values(w)).sum(), params, allow_unused=True)
    moved = 0
    for g, e in zip(got, expected):
        assert (g is None) == (e is None)
        if g is not None:
            torch.testing.assert_close(g, e, rtol=1e-10, atol=1e-13)
            moved += bool(e.abs().sum() > 0)
    assert moved >= 8


def test_rois_across_panels_follow_a_geometry_step():
    model = _multi(background_rois=ROIS, tilt_abc=ABC, background_panels=ROI_PANELS)
    model.freeze("distance", "detector_rotation", "twotheta", "beam_center")
    with torch.no_grad():
        before = model(roi_only=True)
        kept = model._panel_pixels
        model(roi_only=True)
        assert model._panel_pixels is kept
        model.detector_params.delta_log_distance_mm.fill_(0.1)
        moved = model(roi_only=True)
        assert not torch.allclose(before, moved)
        torch.testing.assert_close(moved, model.roi_values(model()), rtol=1e-12, atol=1e-16)


def test_bad_panels_are_rejected():
    with pytest.raises(ValueError, match="one of the 3 panels"):
        _multi(background_rois=ROIS, background_panels=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="same size"):
        small = DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=16, fpixels=16)
        ExperimentModel(_crystal(), [_panel(), small], _beam(), dtype=DTYPE)


def test_frozen_detector_keeps_its_pixel_coordinates_until_an_edit():
    model = _multi()
    model.freeze("distance", "detector_rotation", "twotheta", "beam_center")
    with torch.no_grad():
        model()
        coords = model._simulator._cached_pixel_coords_meters
        model()
        assert model._simulator._cached_pixel_coords_meters is coords
        model.detector_params.delta_log_distance_mm.fill_(0.1)
        moved = model()
        assert model._simulator._cached_pixel_coords_meters is not coords
        fresh = _multi()
        fresh.detector_params.delta_log_distance_mm.requires_grad_(False).fill_(0.1)
        torch.testing.assert_close(moved, fresh(), rtol=1e-12, atol=0.0)


def test_backward_twice_without_a_step():
    model = _multi()
    for _ in range(2):
        model.zero_grad()
        model().sum().backward()
    assert model.detector_params.delta_log_distance_mm.grad != 0


def _sf():
    hkl = np.array([(h, k, l) for h in range(-2, 3) for k in range(-2, 3) for l in range(-2, 3)  # noqa: E741
                    if (h, k, l) > (0, 0, 0)])
    return StructureFactors(hkl, np.random.default_rng(0).uniform(50, 500, len(hkl)))


MISSETS = [(5.0, -3.0, 2.0), (20.0, 10.0, -5.0), (-30.0, 7.0, 15.0)]


def _shot(sf, misset, **kwargs):
    return ExperimentModel(_crystal(misset), _panel(), _beam(), dtype=DTYPE, structure_factors=sf, **kwargs)


def test_each_shot_is_its_own_model_and_fhkl_is_shared():
    sf = _sf()
    serial = SerialExperiment([_shot(sf, m) for m in MISSETS])
    F = serial.shots[0].fhkl_params.delta_log_F
    assert all(shot.fhkl_params.delta_log_F is F for shot in serial.shots)
    names = [n for n, _ in serial.named_parameters()]
    assert sum(n.endswith("delta_log_F") for n in names) == 1
    assert sum(n.endswith("delta_misset_raw") for n in names) == 3
    w = torch.rand(P, P, dtype=DTYPE, generator=torch.Generator().manual_seed(2))
    images = serial()
    sum((image * w).sum() for image in images).backward()
    expected_F = torch.zeros_like(F)
    for image, shot, misset in zip(images, serial.shots, MISSETS):
        alone = _shot(sf, misset)
        reference = alone()
        torch.testing.assert_close(image, reference, rtol=1e-12, atol=0.0)
        (reference * w).sum().backward()
        expected_F = expected_F + alone.fhkl_params.delta_log_F.grad
        torch.testing.assert_close(shot.crystal_params.delta_misset_raw.grad,
                                   alone.crystal_params.delta_misset_raw.grad, rtol=1e-11, atol=1e-13)
    assert (expected_F != 0).sum() > 5
    torch.testing.assert_close(F.grad, expected_F, rtol=1e-11, atol=1e-13)


def test_a_fhkl_step_reaches_every_shot_and_a_minibatch_runs_only_those():
    sf = _sf()
    serial = SerialExperiment([_shot(sf, m) for m in MISSETS])
    with torch.no_grad():
        before = serial()
        serial.shots[0].fhkl_params.delta_log_F.fill_(0.5)
        after = serial([2, 0])
        fresh = _shot(sf, MISSETS[2])
        fresh.fhkl_params.delta_log_F.fill_(0.5)
        expected = fresh()
    assert len(after) == 2
    assert not torch.allclose(after[0], before[2])
    torch.testing.assert_close(after[0], expected, rtol=1e-12, atol=0.0)


def test_shots_read_one_fhkl_grid():
    sf = _sf()
    serial = SerialExperiment([_shot(sf, m) for m in MISSETS])
    serial()
    assert len({id(shot._crystal.hkl_data) for shot in serial.shots}) == 1


def test_group_control_reaches_all_shots():
    sf = _sf()
    serial = SerialExperiment([_shot(sf, m) for m in MISSETS])
    serial.freeze().refine("Fhkl", "scale")
    assert len(serial.trainable_parameters()) == 1 + 3


def test_shared_detector_exists_once():
    sf = _sf()
    serial = SerialExperiment([_shot(sf, m) for m in MISSETS], share_detector=True)
    w = torch.rand(P, P, dtype=DTYPE, generator=torch.Generator().manual_seed(2))
    images = serial()
    detectors = {id(shot._simulator.detector) for shot in serial.shots}
    assert len(detectors) == 1
    coords = {shot._simulator._cached_pixel_coords_meters.data_ptr() for shot in serial.shots}
    assert len(coords) == 1
    sum((image * w).sum() for image in images).backward()
    expected = torch.zeros((), dtype=DTYPE)
    for image, misset in zip(images, MISSETS):
        alone = _shot(sf, misset)
        reference = alone()
        torch.testing.assert_close(image, reference, rtol=1e-12, atol=0.0)
        (reference * w).sum().backward()
        expected = expected + alone.detector_params.delta_log_distance_mm.grad
    torch.testing.assert_close(serial.shots[0].detector_params.delta_log_distance_mm.grad, expected,
                               rtol=1e-11, atol=0.0)


def test_shared_detector_survives_optimizer_steps():
    sf = _sf()
    serial = SerialExperiment([_shot(sf, m) for m in MISSETS], share_detector=True)
    serial.freeze().refine("distance", "scale")
    optimizer = torch.optim.Adam(serial.trainable_parameters(), lr=1e-3)
    target = [image.detach() * 1.05 for image in serial()]
    for _ in range(3):
        optimizer.zero_grad()
        sum(((image - t) ** 2).sum() for image, t in zip(serial(), target)).backward()
        optimizer.step()
    optimizer.zero_grad()
    sum(image.sum() for image in serial()).backward()  # and again with nothing moved in between
    sum(image.sum() for image in serial()).backward()


def test_shots_must_share_the_same_structure_factors_and_be_fresh():
    with pytest.raises(ValueError, match="same StructureFactors"):
        SerialExperiment([_shot(_sf(), MISSETS[0]), _shot(_sf(), MISSETS[1])])
    sf = _sf()
    ran = _shot(sf, MISSETS[1])
    with torch.no_grad():
        ran()
    with pytest.raises(ValueError, match="have not run"):
        SerialExperiment([_shot(sf, MISSETS[0]), ran])


def test_refinement_recovers_shared_amplitudes_from_several_shots():
    sf = _sf()
    truth = SerialExperiment([_shot(sf, m) for m in MISSETS])
    bump = torch.as_tensor(np.random.default_rng(1).uniform(-0.3, 0.3, len(sf)))
    with torch.no_grad():
        truth.shots[0].fhkl_params.delta_log_F.copy_(bump)
        for shot, scale in zip(truth.shots, (0.1, -0.2, 0.3)):
            shot.beam_params.delta_log_spot_scale.fill_(scale)
        data = truth()
    serial = SerialExperiment([_shot(sf, m) for m in MISSETS])
    serial.freeze().refine("Fhkl", "scale")
    optimizer = torch.optim.LBFGS(serial.trainable_parameters(), max_iter=60, tolerance_grad=1e-14,
                                  tolerance_change=1e-18, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = sum(((model - d) ** 2).sum() for model, d in zip(serial(), data)) * 1e6
        loss.backward()
        return loss

    first = float(closure())
    optimizer.step(closure)
    assert float(closure()) < 1e-3 * first
