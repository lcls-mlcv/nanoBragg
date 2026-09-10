"""
Tests for nanobrag_torch.compat.cctbx.

The first group uses duck-typed stand-ins for dxtbx Panel / Beam / Crystal so
the geometry and orientation ingestion can be verified without cctbx. The last
test runs only when cctbx + dxtbx are importable and compares against
simtbx.nanoBragg directly.
"""
import math
import os

import numpy as np
import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from nanobrag_torch.compat.cctbx import (  # noqa: E402
    MultiPanelSimulator,
    beam_config_from_dxtbx,
    crystal_config_from_A,
    crystal_config_from_dxtbx,
    detector_config_from_dxtbx_panel,
    isotropic_umats,
    rotate_A,
    set_mosaic_blocks,
    set_structure_factors,
    simulator_from_dxtbx,
    to_raw_pixels,
)
from nanobrag_torch.models.crystal import Crystal  # noqa: E402
from nanobrag_torch.models.detector import Detector  # noqa: E402


# --------------------------------------------------------------------------- #
# duck-typed dxtbx models
# --------------------------------------------------------------------------- #
class FakePanel:
    """Minimal dxtbx.model.Panel look-alike (mm units, dxtbx conventions)."""

    def __init__(self, origin_mm, fast, slow, pixel_mm=0.1, image_size=(64, 48), thickness_mm=0.0, mu=0.0):
        self._origin = np.asarray(origin_mm, dtype=float)
        self._fast = np.asarray(fast, dtype=float) / np.linalg.norm(fast)
        self._slow = np.asarray(slow, dtype=float) / np.linalg.norm(slow)
        self._pixel = pixel_mm
        self._image_size = tuple(image_size)  # (fast, slow) like dxtbx
        self._thickness = thickness_mm
        self._mu = mu

    def get_origin(self): return tuple(self._origin)
    def get_fast_axis(self): return tuple(self._fast)
    def get_slow_axis(self): return tuple(self._slow)
    def get_pixel_size(self): return (self._pixel, self._pixel)
    def get_image_size(self): return self._image_size
    def get_thickness(self): return self._thickness
    def get_mu(self): return self._mu
    def get_gain(self): return 1.0

    def get_pixel_lab_coord(self, xy):
        """dxtbx: lab coordinate (mm) of pixel position (fast, slow) in pixel units."""
        f, s = xy
        return self._origin + f * self._pixel * self._fast + s * self._pixel * self._slow


class FakeDetector(list):
    pass


class FakeBeam:
    def __init__(self, direction=(0, 0, 1), wavelength=1.0, pol_fraction=0.999, pol_normal=(0, 1, 0)):
        d = np.asarray(direction, dtype=float)
        self._dir = d / np.linalg.norm(d)
        self._wl = wavelength
        self._pf = pol_fraction
        self._pn = np.asarray(pol_normal, dtype=float)

    def get_wavelength(self): return self._wl
    # dxtbx: s0 = direction / wavelength, pointing from source to sample
    def get_s0(self): return tuple(self._dir / self._wl)
    def get_unit_s0(self): return tuple(self._dir)
    def get_polarization_fraction(self): return self._pf
    def get_polarization_normal(self): return tuple(self._pn)


class FakeUnitCell:
    def __init__(self, params): self._p = tuple(params)
    def parameters(self): return self._p


class FakeCrystal:
    """dxtbx.model.Crystal look-alike: A = U·B, columns are a*, b*, c* (Å⁻¹)."""

    def __init__(self, cell, U=None):
        self._cell = tuple(cell)
        a, b, c, al, be, ga = cell
        al, be, ga = (math.radians(x) for x in (al, be, ga))
        # real-space basis (PDB/orthogonalisation convention: a along x, b in xy)
        va = np.array([a, 0.0, 0.0])
        vb = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
        cx = c * math.cos(be)
        cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
        cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
        vc = np.array([cx, cy, cz])
        real = np.stack([va, vb, vc], axis=1)          # columns a, b, c
        B = np.linalg.inv(real).T                       # columns a*, b*, c*
        self._U = np.eye(3) if U is None else np.asarray(U, dtype=float)
        self._A = self._U @ B

    def get_unit_cell(self): return FakeUnitCell(self._cell)
    def get_A(self): return tuple(self._A.reshape(-1))
    def get_U(self): return tuple(self._U.reshape(-1))
    def get_real_space_vectors(self):
        real = np.linalg.inv(self._A)                   # rows a, b, c (Real^T·A = I)
        return [tuple(r) for r in real]


def rotation_matrix(axis, deg):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    x, y, z = axis; t = math.radians(deg); c, s = math.cos(t), math.sin(t); C = 1 - c
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def simple_detector(distance_mm=100.0, pixel_mm=0.1, image_size=(64, 48), beam_dir=(0, 0, 1)):
    """Like SimData.simple_detector: a single panel normal to the beam, beam through its centre."""
    nf, ns = image_size
    fast = np.array([1.0, 0.0, 0.0]); slow = np.array([0.0, -1.0, 0.0])
    origin = np.asarray(beam_dir, float) * distance_mm - fast * nf * pixel_mm / 2 - slow * ns * pixel_mm / 2
    return FakeDetector([FakePanel(origin, fast, slow, pixel_mm, image_size)])


# --------------------------------------------------------------------------- #
# detector geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tilt_deg,shift_mm", [(0.0, (0, 0, 0)), (7.0, (3.0, -2.0, 0.0)), (-12.0, (-5.0, 4.0, 1.5))])
def test_panel_pixel_positions_match_dxtbx(tilt_deg, shift_mm):
    """Torch pixel centres must equal panel.get_pixel_lab_coord(f+0.5, s+0.5) for any tilted panel."""
    R = rotation_matrix((1, 1, 0), tilt_deg)
    fast = R @ np.array([1.0, 0.0, 0.0]); slow = R @ np.array([0.0, -1.0, 0.0])
    origin = R @ np.array([-3.2, 2.4, 100.0]) + np.asarray(shift_mm)
    panel = FakePanel(origin, fast, slow, pixel_mm=0.1, image_size=(64, 48))
    beam = FakeBeam(direction=(0, 0, 1), wavelength=1.0)

    cfg = detector_config_from_dxtbx_panel(panel, beam.get_s0())
    det = Detector(cfg, dtype=torch.float64)
    coords_m = det.get_pixel_coords()  # (slow, fast, 3) metres, pixel centres

    for s, f in [(0, 0), (0, 63), (47, 0), (47, 63), (23, 31)]:
        expected_mm = panel.get_pixel_lab_coord((f + 0.5, s + 0.5))
        got_mm = coords_m[s, f].numpy() * 1000.0
        assert np.allclose(got_mm, expected_mm, atol=1e-6), (s, f, got_mm, expected_mm)

    # pix0 is the dxtbx origin (outer corner of the first pixel)
    assert np.allclose(det.pix0_vector.numpy() * 1000.0, origin, atol=1e-6)


def test_lefthanded_panel_is_flipped():
    """A panel whose fast×slow points away from the sample must still ingest (odet inverted)."""
    fast = np.array([1.0, 0.0, 0.0]); slow = np.array([0.0, 1.0, 0.0])  # fast×slow = +z, origin at +z → fine
    origin = np.array([-3.2, -2.4, 100.0])
    cfg = detector_config_from_dxtbx_panel(FakePanel(origin, fast, slow), (0, 0, 1))
    assert cfg.close_distance_mm > 0
    fast2 = np.array([1.0, 0.0, 0.0]); slow2 = np.array([0.0, -1.0, 0.0])  # fast×slow = -z
    cfg2 = detector_config_from_dxtbx_panel(FakePanel(origin, fast2, slow2), (0, 0, 1))
    assert cfg2.close_distance_mm > 0
    assert np.allclose(cfg2.custom_odet_vector, (0, 0, 1))
    det = Detector(cfg2, dtype=torch.float64)
    assert np.allclose(det.pix0_vector.numpy() * 1000.0, origin, atol=1e-6)


def test_panel_thickness_and_attenuation_ingested():
    panel = FakePanel((-3.2, 2.4, 100.0), (1, 0, 0), (0, -1, 0), thickness_mm=0.45, mu=3.9)
    cfg = detector_config_from_dxtbx_panel(panel, (0, 0, 1))
    assert cfg.detector_thick_um == pytest.approx(450.0)
    assert cfg.detector_abs_um == pytest.approx(1000.0 / 3.9)


# --------------------------------------------------------------------------- #
# crystal orientation and structure factors
# --------------------------------------------------------------------------- #
def test_crystal_reciprocal_vectors_reproduce_A():
    """The torch crystal's a*,b*,c* must be the columns of dxtbx A (Å⁻¹), for a rotated triclinic cell."""
    cell = (70.0, 80.0, 90.0, 75.0, 85.0, 95.0)
    U = rotation_matrix((0.3, -0.5, 0.8), 37.0)
    xtal = FakeCrystal(cell, U)
    cfg = crystal_config_from_dxtbx(xtal, Ncells_abc=(5, 5, 5), default_F=100.0)
    crystal = Crystal(cfg, dtype=torch.float64)
    A = np.asarray(xtal.get_A()).reshape(3, 3)
    got = np.stack([crystal.a_star.numpy(), crystal.b_star.numpy(), crystal.c_star.numpy()], axis=1)
    assert np.allclose(got, A, rtol=1e-9, atol=1e-12)
    # real-space vectors follow (metric duality) and cell lengths are preserved
    for vec, length in zip((crystal.a, crystal.b, crystal.c), cell[:3]):
        assert np.linalg.norm(vec.numpy()) == pytest.approx(length, rel=1e-9)
    real = np.asarray(xtal.get_real_space_vectors())
    assert np.allclose(np.stack([crystal.a.numpy(), crystal.b.numpy(), crystal.c.numpy()]), real, atol=1e-8)


def test_structure_factor_box_lookup():
    cell = (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
    cfg = crystal_config_from_A(cell, FakeCrystal(cell).get_A(), Ncells_abc=3, default_F=7.0)
    crystal = Crystal(cfg, dtype=torch.float64)
    idx = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 4]])
    amp = np.array([10.0, 10.0, 25.0])
    set_structure_factors(crystal, idx, amp)
    assert crystal.hkl_metadata == {"h_min": -1, "h_max": 1, "k_min": -2, "k_max": 2, "l_min": -3, "l_max": 4}
    assert crystal.hkl_data.shape == (3, 5, 8)
    F = crystal.get_structure_factor(torch.tensor([1.0, 0.0, 0.0, 5.0], dtype=torch.float64),
                                     torch.tensor([2.0, 0.0, 1.0, 5.0], dtype=torch.float64),
                                     torch.tensor([3.0, 4.0, 1.0, 5.0], dtype=torch.float64))
    assert F.tolist() == [10.0, 25.0, 7.0, 7.0]  # hit, hit, in-box gap -> default, out of box -> default


def test_set_mosaic_blocks_is_used_and_counts_domains():
    cell = (60.0, 60.0, 60.0, 90.0, 90.0, 90.0)
    cfg = crystal_config_from_A(cell, FakeCrystal(cell).get_A(), Ncells_abc=4, default_F=50.0,
                                mosaic_spread_deg=0.3, mosaic_domains=1)
    crystal = Crystal(cfg, dtype=torch.float64)
    umats = isotropic_umats(0.3, 3, seed=5)
    assert umats.shape == (6, 3, 3)
    assert np.allclose((umats @ umats.transpose(1, 2)).numpy(), np.eye(3), atol=1e-12)
    set_mosaic_blocks(crystal, umats)
    assert crystal.config.mosaic_domains == 6
    out = crystal.get_rotated_real_vectors(crystal.config)
    a = out[0][0] if isinstance(out[0], (tuple, list)) else out[0]
    assert a.shape[1] == 6
    # first block rotates a by the first umat exactly
    expected = (umats[0].numpy() @ crystal.a.numpy())
    assert np.allclose(a[0, 0].numpy(), expected, atol=1e-10)


# --------------------------------------------------------------------------- #
# beam
# --------------------------------------------------------------------------- #
def test_beam_config_from_dxtbx_and_spectrum():
    beam = FakeBeam(direction=(0, 0, 1), wavelength=1.3, pol_fraction=0.95, pol_normal=(0, 1, 0))
    cfg = beam_config_from_dxtbx(beam, spectrum=[(1.29, 0.5), (1.31, 1.5)], spot_scale=3.0)
    assert cfg.wavelength_A == pytest.approx(1.3)
    assert cfg.polarization_factor == pytest.approx(0.95)
    assert cfg.spot_scale == 3.0
    assert cfg.source_directions.shape == (2, 3)
    assert np.allclose(cfg.source_directions.numpy(), [[0, 0, -1], [0, 0, -1]])  # sample -> source
    assert np.allclose(cfg.source_wavelengths.numpy(), [1.29e-10, 1.31e-10])


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def _end_to_end_models():
    cell = (78.0, 78.0, 37.0, 90.0, 90.0, 90.0)
    U = rotation_matrix((1, 2, 3), 20.0)
    return simple_detector(150.0, 0.2, (64, 48)), FakeBeam(wavelength=1.2), FakeCrystal(cell, U)


def test_simulator_from_dxtbx_runs_and_spot_scale_is_linear():
    det, beam, xtal = _end_to_end_models()
    sim = simulator_from_dxtbx(det, beam, xtal, Ncells_abc=6, default_F=100.0, oversample=1)
    img = sim.run()
    assert img.shape == (48, 64)
    assert torch.isfinite(img).all() and float(img.max()) > 0
    sim3 = simulator_from_dxtbx(det, beam, xtal, Ncells_abc=6, default_F=100.0, oversample=1, spot_scale=3.0)
    assert torch.allclose(sim3.run(), 3.0 * img)
    raw = to_raw_pixels(img)
    assert np.asarray(raw).shape == (48, 64)


def test_orientation_gradient_flows_through_rotate_A():
    """d(image)/d(rotX) via autograd must match a central finite difference (diffBragg RotXYZ convention)."""
    det, beam, xtal = _end_to_end_models()
    A0 = np.asarray(xtal.get_A()).reshape(3, 3)
    cell = xtal.get_unit_cell().parameters()

    def image(theta):
        cfg = crystal_config_from_A(cell, rotate_A(A0, rotx_deg=theta), Ncells_abc=6, default_F=100.0)
        crystal = Crystal(cfg, dtype=torch.float64)
        detector = Detector(detector_config_from_dxtbx_panel(det[0], beam.get_s0(), oversample=1), dtype=torch.float64)
        from nanobrag_torch.simulator import Simulator
        from nanobrag_torch.config import BeamConfig
        return Simulator(crystal, detector, beam_config=BeamConfig(wavelength_A=beam.get_wavelength()),
                         dtype=torch.float64).run()

    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    img = image(theta)
    loss = img.sum()
    loss.backward()
    assert theta.grad is not None and torch.isfinite(theta.grad)
    h = 1e-3
    fd = (image(torch.tensor(h)).sum() - image(torch.tensor(-h)).sum()) / (2 * h)
    assert float(theta.grad) == pytest.approx(float(fd), rel=1e-3, abs=1e-6)


def test_multipanel_simulator_stacks_panels_and_shares_crystal():
    nf, ns, px = 32, 24, 0.2
    fast = np.array([1.0, 0, 0]); slow = np.array([0, -1.0, 0])
    base = np.array([0, 0, 120.0]) - fast * nf * px / 2 - slow * ns * px / 2
    panels = FakeDetector([
        FakePanel(base + np.array([-8.0, 0, 0]), fast, slow, px, (nf, ns)),
        FakePanel(base + np.array([8.0, 0, 0]), fast, slow, px, (nf, ns)),
    ])
    cell = (60.0, 60.0, 60.0, 90.0, 90.0, 90.0)
    ms = MultiPanelSimulator(panels, FakeBeam(wavelength=1.0), FakeCrystal(cell), Ncells_abc=5, default_F=100.0, oversample=1)
    out = ms.run()
    assert out.shape == (2, ns, nf)
    assert ms.panels[0].crystal is ms.panels[1].crystal
    # the two panels are mirror images about the beam for a cubic cell in canonical orientation
    assert torch.allclose(out[0], torch.flip(out[1], dims=[1]), rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# real cctbx comparison (skipped unless cctbx/dxtbx importable)
# --------------------------------------------------------------------------- #
def test_against_simtbx_nanoBragg():
    pytest.importorskip("dxtbx", reason="cctbx/dxtbx not installed")
    from simtbx.nanoBragg import nanoBragg, shapetype  # type: ignore
    from simtbx.nanoBragg.sim_data import SimData  # type: ignore
    from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal  # type: ignore
    from nanobrag_torch.compat.cctbx import simulator_from_sim_data

    nb = NBcrystal(init_defaults=True)
    nb.n_mos_domains = 1
    nb.mos_spread_deg = 0
    nb.xtal_shape = shapetype.Square
    nb.Ncells_abc = (7, 7, 7)
    SIM = SimData(use_default_crystal=True)
    SIM.detector = SimData.simple_detector(150, 0.1, (256, 256))
    SIM.crystal = nb
    SIM.instantiate_nanoBragg(oversample=1, verbose=0, interpolate=0, default_F=100.0)
    SIM.D.add_nanoBragg_spots()
    ref = SIM.D.raw_pixels.as_numpy_array()

    sim = simulator_from_sim_data(SIM, dtype=torch.float64)
    img = sim.run().numpy()
    assert img.shape == ref.shape
    corr = np.corrcoef(ref.ravel(), img.ravel())[0, 1]
    assert corr > 0.999, corr
    assert img.sum() == pytest.approx(ref.sum(), rel=2e-2)
