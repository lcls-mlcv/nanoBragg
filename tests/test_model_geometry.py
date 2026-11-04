"""What an ExperimentModel simulates, read back out for export: geometry(), bragg_only, roi_centroids(), and
the checks of experiment_from_model / reflections_from_model that need no cctbx."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nanobrag_torch.compat.cctbx import experiment_model_from_dxtbx

DT = torch.float64
ROIS = [(2, 9, 3, 10), (6, 13, 5, 12), (0, 4, 12, 16)]  # the second overlaps the first


def _rotation(axis, deg):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.deg2rad(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


U = _rotation((0.3, -0.5, 0.8), 37.0)
CELL = (30.0, 35.0, 40.0, 90.0, 90.0, 90.0)


def dxtbx_objects(real_space=None, origin=(-1.6, 1.9, -60.0), fast=(1.0, 0.0, 0.0), slow=(0.0, -1.0, 0.0),
                  wavelength=1.2):
    """Duck-typed dxtbx crystal, panel and beam; the crystal from its real-space vectors (rows)."""
    if real_space is None:
        real_space = U @ np.diag(CELL[:3])  # columns a, b, c
        real_space = real_space.T
    real_space = np.asarray(real_space, dtype=float)
    A = np.linalg.inv(real_space)  # columns a*, b*, c*
    a, b, c = (np.linalg.norm(v) for v in real_space)
    crystal = SimpleNamespace(
        get_unit_cell=lambda: SimpleNamespace(parameters=lambda: (a, b, c, 90.0, 90.0, 90.0)),
        get_A=lambda: A.ravel(),
        get_space_group=lambda: SimpleNamespace(crystal_system=lambda: "Orthorhombic"))
    panel = SimpleNamespace(get_fast_axis=lambda: fast, get_slow_axis=lambda: slow, get_origin=lambda: origin,
                            get_pixel_size=lambda: (0.2, 0.2), get_image_size=lambda: (16, 16),
                            get_thickness=lambda: 0.0, get_mu=lambda: 0.0)
    beam = SimpleNamespace(get_wavelength=lambda: wavelength, get_s0=lambda: (0.0, 0.0, -1.0 / wavelength))
    return [panel], beam, crystal


def model_of(objects, **kwargs):
    detector, beam, crystal = objects
    kwargs.setdefault("rois", ROIS)
    return experiment_model_from_dxtbx(detector, beam, crystal, Ncells_abc=(6, 6, 6), default_F=100.0,
                                       fluence=1e30, dtype=DT, **kwargs)


def refined():
    m = model_of(dxtbx_objects())
    with torch.no_grad():
        m.crystal_params.delta_misset_raw.copy_(torch.tensor([0.01, -0.02, 0.015], dtype=DT))
        m.crystal_params.delta_log_a.fill_(0.01)
        m.detector_params.delta_detz_shift_mm.fill_(0.4)
        m.beam_params.lambda_offset.fill_(0.003)
    return m


def test_geometry_gives_back_what_was_imported():
    g = model_of(dxtbx_objects()).geometry()
    np.testing.assert_allclose(np.stack([g.real_space_a, g.real_space_b, g.real_space_c]), (U @ np.diag(CELL[:3])).T,
                               rtol=0, atol=1e-12)
    (panel,) = g.panels
    np.testing.assert_allclose(panel.origin_mm, (-1.6, 1.9, -60.0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(panel.fast_axis, (1.0, 0.0, 0.0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(panel.slow_axis, (0.0, -1.0, 0.0), rtol=0, atol=1e-15)
    assert g.wavelength_A == pytest.approx(1.2, rel=1e-15)
    np.testing.assert_allclose(g.beam_direction, (0.0, 0.0, -1.0), rtol=0, atol=1e-15)


def test_geometry_follows_refined_parameters():
    g = refined().geometry()
    assert np.linalg.norm(g.real_space_a) == pytest.approx(30.0 * np.exp(0.01), rel=1e-12)
    np.testing.assert_allclose(g.panels[0].origin_mm, (-1.6, 1.9, -60.0 + 0.4), rtol=0, atol=1e-12)
    assert g.wavelength_A == pytest.approx(1.203, rel=1e-14)


def test_a_model_built_from_the_refined_geometry_simulates_the_same_image():
    m = refined()
    g = m.geometry()
    panel = g.panels[0]
    again = model_of(dxtbx_objects(np.stack([g.real_space_a, g.real_space_b, g.real_space_c]), tuple(panel.origin_mm),
                                   tuple(panel.fast_axis), tuple(panel.slow_axis), g.wavelength_A))
    with torch.no_grad():
        image, image_again = m(), again()
    assert float(image.max()) > 0
    torch.testing.assert_close(image_again, image, rtol=1e-10, atol=1e-10 * float(image.max()))


def test_bragg_only_leaves_out_the_background():
    m = model_of(dxtbx_objects())
    with torch.no_grad():
        m.background_params.delta_tilt_abc.copy_(torch.arange(9, dtype=DT).reshape(3, 3) + 1)
        both, bragg = m(roi_only=True), m(roi_only=True, bragg_only=True)
        image, bragg_image = m(), m(bragg_only=True)
    torch.testing.assert_close(both - bragg, m.background_params.values(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(image - bragg_image, m.background_params.image(), rtol=1e-12, atol=1e-12)


def test_roi_centroids_are_hoppers_over_each_whole_box():
    m = refined()
    with torch.no_grad():
        image = m(bragg_only=True).numpy()
    got = m.roi_centroids()
    assert got.shape == (len(ROIS), 2)
    for (x1, x2, y1, y2), (x, y) in zip(ROIS, got):
        box = image[y1:y2, x1:x2]
        if box.sum() == 0:
            assert np.isnan(x) and np.isnan(y)
            continue
        Y, X = np.indices(box.shape)
        assert x == pytest.approx((X * box).sum() / box.sum() + x1 + 0.5, rel=1e-12)
        assert y == pytest.approx((Y * box).sum() / box.sum() + y1 + 0.5, rel=1e-12)
    assert np.isfinite(got).all(axis=1).sum() >= 1


def test_a_box_without_bragg_signal_has_no_centroid():
    m = model_of(dxtbx_objects(), rois=[(0, 2, 0, 2)])
    with torch.no_grad():
        m.beam_params.delta_log_spot_scale.fill_(-1e3)
    assert np.isnan(m.roi_centroids()).all()


def test_an_observed_centroid_without_model_signal_gets_no_centroid():
    # a tophat spot ends at a sharp edge, so a box can hold signal and zeros both
    m = model_of(dxtbx_objects(), shape="tophat", rois=[(0, 16, 0, 16)])
    m.freeze()
    with torch.no_grad():
        image = m(bragg_only=True).numpy()
    (free,) = m.roi_centroids()
    assert np.isfinite(free).all()
    peak_y, peak_x = np.unravel_index(image.argmax(), image.shape)
    empty_y, empty_x = np.argwhere(image == 0)[0]
    on = m.roi_centroids(observed=[(peak_x + 0.5, peak_y + 0.5)])[0]
    np.testing.assert_array_equal(on, free)
    assert np.isnan(m.roi_centroids(observed=[(empty_x + 0.5, empty_y + 0.5)])[0]).all()
    assert np.isnan(m.roi_centroids(observed=[(20.0, 3.0)])[0]).all()  # outside its box


def _tophat_edge():
    """A tophat model over the whole image and a lit pixel whose left neighbour is dark."""
    m = model_of(dxtbx_objects(), shape="tophat", rois=[(0, 16, 0, 16)])
    m.freeze()
    with torch.no_grad():
        image = m(bragg_only=True).numpy()
    lit = np.argwhere((image[:, 1:] > 0) & (image[:, :-1] == 0))
    y, x = lit[0]
    return m, int(y), int(x) + 1


def test_the_observed_pixel_is_hoppers_int_of_x_minus_half():
    m, y, x = _tophat_edge()
    # hopper checks pixel int(x_obs - x1 - 0.5): a centroid at x + 0.9 checks pixel x, at x + 0.4 pixel x - 1
    assert np.isfinite(m.roi_centroids(observed=[(x + 0.9, y + 0.5)])[0]).all()
    assert np.isnan(m.roi_centroids(observed=[(x + 0.4, y + 0.5)])[0]).all()
    # and int() truncates toward zero: just left of the box still checks its first column
    np.testing.assert_array_equal(m.roi_centroids(observed=[(-0.3, y + 0.5)]),
                                  m.roi_centroids(observed=[(0.5, y + 0.5)]))


class _Panel:
    def __init__(self, origin, local_origin=None):
        self.origin, self.local_origin = tuple(origin), tuple(local_origin or origin)

    def get_origin(self):
        return self.origin

    def get_local_origin(self):
        return self.local_origin

    def set_frame(self, fast, slow, origin):
        self.fast, self.slow, self.origin = fast, slow, tuple(origin)


class _Crystal:
    def set_A(self, A):
        self.A = np.asarray(A).reshape(3, 3)


class _Beam:
    def __init__(self, s0=(0.0, 0.0, -1 / 1.2)):
        self.s0, self.wavelength = s0, 1.2

    def get_s0(self):
        return self.s0

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength


def _experiment(**kwargs):
    from types import SimpleNamespace
    fields = dict(crystal=_Crystal(), detector=[_Panel((-1.6, 1.9, -60.0))], beam=_Beam(), goniometer=None, scan=None)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def test_experiment_from_model_writes_what_geometry_gives():
    from nanobrag_torch import experiment_from_model

    m = refined()
    g = m.geometry()
    experiment = _experiment()
    crystal_only = experiment_from_model(m, experiment)
    assert crystal_only.detector is experiment.detector and crystal_only.beam is experiment.beam
    new = experiment_from_model(m, experiment, detector=True, wavelength=True)
    np.testing.assert_allclose(np.linalg.inv(new.crystal.A), np.stack([g.real_space_a, g.real_space_b, g.real_space_c]),
                               rtol=1e-12)
    np.testing.assert_allclose(new.detector[0].origin, g.panels[0].origin_mm, rtol=0, atol=1e-12)
    assert new.beam.wavelength == g.wavelength_A
    assert experiment.detector[0].origin == (-1.6, 1.9, -60.0) and experiment.beam.wavelength == 1.2


@pytest.mark.parametrize("change, message", [
    (dict(detector=[_Panel((0, 0, -60.0)), _Panel((0, 0, -60.0))]), "panels"),
    (dict(beam=_Beam((1 / 1.2, 0.0, 0.0))), "dxtbx's frame"),
    (dict(goniometer=object()), "goniometer"),
])
def test_experiment_from_model_refuses_an_experiment_it_does_not_describe(change, message):
    from nanobrag_torch import experiment_from_model

    with pytest.raises(ValueError, match=message):
        experiment_from_model(refined(), _experiment(**change))


def test_experiment_from_model_refuses_a_hierarchical_detector_for_its_panels():
    from nanobrag_torch import experiment_from_model

    experiment = _experiment(detector=[_Panel((-1.6, 1.9, -60.0), local_origin=(0.0, 0.0, 0.0))])
    experiment_from_model(refined(), experiment)  # the crystal alone is fine
    with pytest.raises(ValueError, match="hierarchical"):
        experiment_from_model(refined(), experiment, detector=True)


class _Table(dict):
    def __len__(self):
        return len(self["xyzobs.px.value"])


@pytest.mark.parametrize("rows, panels, message", [
    (None, [0, 0], "rows must give"),
    ([0, 1, 5], [0, 0, 0], "rows must give"),
    ([0, 1, 2], [0, 0, 1], "panel differs"),
])
def test_reflections_from_model_refuses_rows_that_do_not_match(rows, panels, message):
    from nanobrag_torch import reflections_from_model

    table = _Table({"xyzobs.px.value": [(0.5, 0.5, 0.0)] * len(panels), "panel": panels})
    with pytest.raises(ValueError, match=message):
        reflections_from_model(refined(), table, _experiment(), rows=rows)
