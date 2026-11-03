"""
The optional diffuse term, tested on physics, not on its own output.

With one unit cell, F_latt = 1 at every pixel, so a constant F makes Bragg a smooth image with
exactly the tail a diffuse term must have: Bragg(F) and diffuse(F²) are the same image. Peak
positions test the Miller indices, and scaling laws test how the term grows with crystal size.
"""
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import pytest
import torch

from src.nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from src.nanobrag_torch.models.crystal import Crystal
from src.nanobrag_torch.models.detector import Detector
from src.nanobrag_torch.simulator import Simulator

F0 = 37.0
CRYSTAL = dict(N_cells=(1, 1, 1), default_F=F0, cell_a=50.0, cell_b=60.0, cell_c=70.0, misset_deg=(10.0, 20.0, 30.0))
DETECTOR = dict(spixels=32, fpixels=32, distance_mm=100.0, pixel_size_mm=1.0)
BEAM = dict(wavelength_A=1.0, fluence=1e24)


def constant(value):
    return lambda hkl: torch.full((len(hkl),), value, dtype=torch.float64)


def simulator(crystal=None, detector=None, beam=None, dtype=torch.float64, **kw):
    return Simulator(Crystal(CrystalConfig(**{**CRYSTAL, **(crystal or {})}), dtype=dtype),
                     Detector(DetectorConfig(**{**DETECTOR, **(detector or {})}), dtype=dtype),
                     beam_config=BeamConfig(**{**BEAM, **(beam or {})}), dtype=dtype, **kw)


@pytest.mark.parametrize("detector,beam,run,tolerance", [
    ({}, {}, {}, 1e-12),
    ({}, dict(nopolar=True), {}, 1e-12),
    ({}, dict(polarization_factor=0.5), {}, 1e-12),
    (dict(point_pixel=True), {}, {}, 1e-12),
    (dict(detector_twotheta_deg=25.0), {}, {}, 1e-12),
    (dict(detector_rotx_deg=7.0, detector_roty_deg=-5.0), {}, {}, 1e-12),
    (dict(detector_thick_um=500.0, detector_abs_um=300.0, detector_thicksteps=40), {}, dict(oversample_thick=True), 3e-3),
])
def test_one_cell_bragg_and_diffuse_are_the_same_image(detector, beam, run, tolerance):
    bragg = simulator(detector=detector, beam=beam).run(oversample=1, **run)
    sim = simulator(crystal=dict(default_F=0.0), detector=detector, beam=beam, diffuse=constant(F0 ** 2))
    sim.run(oversample=1, **run)
    assert torch.abs(sim.components["diffuse"] / bragg - 1).max() < tolerance


def test_one_cell_multiple_sources_match_bragg_per_source():
    directions = torch.tensor([[-1.0, 0, 0], [-0.9992, 0.04, 0]], dtype=torch.float64)
    directions = directions / directions.norm(dim=1, keepdim=True)
    beam = dict(source_directions=directions, source_wavelengths=torch.tensor([1.0e-10, 1.1e-10], dtype=torch.float64),
                source_weights=torch.ones(2, dtype=torch.float64))
    bragg = simulator(beam=beam).run(oversample=1)
    sim = simulator(crystal=dict(default_F=0.0), beam=beam, diffuse=constant(F0 ** 2))
    sim.run(oversample=1)
    assert torch.abs(sim.components["diffuse"] / bragg - 1).max() < 1e-12


def test_off_is_identical_and_a_zero_term_changes_nothing():
    crystal = dict(N_cells=(4, 4, 4))
    plain = simulator(crystal=crystal, dtype=torch.float32).run(oversample=1)
    sim = simulator(crystal=crystal, dtype=torch.float32, diffuse=constant(0.0))
    with_zero = sim.run(oversample=1)
    assert torch.equal(plain, with_zero)
    assert torch.equal(sim.components["bragg"], plain) and sim.components["diffuse"].abs().max() == 0
    off = simulator(crystal=crystal, dtype=torch.float32)
    off.run(oversample=1)
    assert off.components is None


def test_components_add_up_and_keep_the_precision():
    sim = simulator(crystal=dict(N_cells=(3, 3, 3)), dtype=torch.float32, diffuse=constant(5.0))
    image = sim.run(oversample=1)
    assert image.dtype == torch.float32
    assert torch.equal(sim.components["bragg"] + sim.components["diffuse"], image)


def test_miller_indices_are_integers_at_the_bragg_peaks():
    seen = {}

    def record(hkl):
        seen["hkl"] = hkl.clone()
        return torch.zeros(len(hkl), dtype=torch.float64)

    crystal = dict(N_cells=(20, 20, 20), cell_a=30.0, cell_b=30.0, cell_c=30.0)
    detector = dict(spixels=256, fpixels=256, pixel_size_mm=0.5)
    image = simulator(crystal=crystal, detector=detector, diffuse=record).run(oversample=1)
    hkl = seen["hkl"].reshape(256, 256, 3)
    bright = torch.nonzero(image > 0.5 * image.max())
    assert len(bright) >= 1
    fractional = hkl[bright[:, 0], bright[:, 1]]
    assert (fractional - fractional.round()).abs().max() < 1.0 / 20


def test_hkl_follow_q_dot_cell_for_a_known_geometry():
    """Beam along x, unrotated orthogonal cell: h = a (s - s0)_x / lambda, by hand."""
    seen = {}

    def record(hkl):
        seen["hkl"] = hkl.clone()
        return torch.zeros(len(hkl), dtype=torch.float64)

    sim = simulator(crystal=dict(misset_deg=(0.0, 0.0, 0.0)), diffuse=record)
    sim.run(oversample=1)
    coords = sim.detector.get_pixel_coords().to(torch.float64)
    unit = coords / coords.norm(dim=-1, keepdim=True)
    q_per_A = (unit - sim.incident_beam_direction.to(torch.float64)) / 1.0
    cell = torch.tensor([50.0, 60.0, 70.0], dtype=torch.float64)
    expected = q_per_A * cell
    assert torch.allclose(seen["hkl"].reshape(32, 32, 3), expected, rtol=1e-6, atol=1e-6)


def test_diffuse_grows_with_volume_and_fluence():
    base = simulator(diffuse=constant(9.0))
    base.run(oversample=1)
    bigger = simulator(crystal=dict(N_cells=(2, 3, 5)), diffuse=constant(9.0))
    bigger.run(oversample=1)
    assert torch.allclose(bigger.components["diffuse"], 30 * base.components["diffuse"], rtol=1e-12)
    brighter = simulator(beam=dict(fluence=3e24), diffuse=constant(9.0))
    brighter.run(oversample=1)
    assert torch.allclose(brighter.components["diffuse"], 3 * base.components["diffuse"], rtol=1e-12)


def test_diffuse_falls_with_distance_as_the_solid_angle():
    near = simulator(diffuse=constant(9.0))
    near.run(oversample=1)
    far = simulator(detector=dict(distance_mm=200.0, pixel_size_mm=2.0), diffuse=constant(9.0))
    far.run(oversample=1)
    assert torch.allclose(far.components["diffuse"], near.components["diffuse"], rtol=1e-10)     # same angles


def test_orientations_average_equals_separate_rotated_runs():
    def term(hkl):
        return (hkl[:, 0] - 0.3) ** 2 + 2.0
    many = simulator(crystal=dict(phi_steps=2, osc_range_deg=2.0), diffuse=term)
    many.run(oversample=1)
    singles = []
    for start in (0.0, 1.0):
        one = simulator(crystal=dict(phi_start_deg=start, phi_steps=1, osc_range_deg=1.0), diffuse=term)
        one.run(oversample=1)
        singles.append(one.components["diffuse"])
    assert torch.allclose(many.components["diffuse"], 0.5 * (singles[0] + singles[1]), rtol=1e-10)
    first = simulator(crystal=dict(phi_steps=2, osc_range_deg=2.0), diffuse=term, diffuse_orientations="first")
    first.run(oversample=1)
    assert torch.allclose(first.components["diffuse"], singles[0], rtol=1e-10)


def test_chunks_subsets_and_masks_agree_with_the_whole_image():
    sim = simulator(diffuse=lambda hkl: 1.0 + hkl[:, 1] ** 2)
    whole = sim.run(oversample=1).clone()
    assert torch.allclose(sim.run(oversample=1, pixel_batch_size=7), whole, rtol=1e-12)
    pixels = torch.tensor([[3, 4], [20, 11], [31, 31]])
    assert torch.allclose(sim.run(oversample=1, pixels=pixels), whole[pixels[:, 0], pixels[:, 1]], rtol=1e-12)
    masked = simulator(detector=dict(roi_xmin=8, roi_xmax=20, roi_ymin=0, roi_ymax=31), diffuse=lambda hkl: 1.0 + hkl[:, 1] ** 2)
    image = masked.run(oversample=1)
    assert masked.components["diffuse"][:, :8].abs().max() == 0 and image[:, 21:].abs().max() == 0


def test_gradient_reaches_term_parameters_and_geometry():
    scale = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    sim = simulator(crystal=dict(default_F=0.0), diffuse=lambda hkl: scale * (1.0 + hkl[:, 0] ** 2))
    sim.run(oversample=1)
    total = sim.components["diffuse"].sum()
    total.backward()
    assert scale.grad == pytest.approx(float(total) / 3.0, rel=1e-10)


def test_gradient_through_wavelength_matches_finite_difference():
    def image_sum(wavelength):
        sim = simulator(crystal=dict(default_F=0.0), diffuse=lambda hkl: 1.0 + hkl[:, 0] ** 2)
        sim.wavelength = wavelength
        sim.run(oversample=1)
        return sim.components["diffuse"].sum()
    wavelength = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    image_sum(wavelength).backward()
    h = 1e-5
    fd = (image_sum(torch.tensor(1.0 + h, dtype=torch.float64)) - image_sum(torch.tensor(1.0 - h, dtype=torch.float64))) / (2 * h)
    assert float(wavelength.grad) == pytest.approx(float(fd), rel=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_runs_the_term_in_float32_and_matches_the_cpu():
    """MPS has no doubles; the term used to be evaluated in float64 there and raised TypeError."""
    detector = dict(detector_thick_um=500.0, detector_abs_um=300.0, detector_thicksteps=4)
    images = {}
    for device in ["cpu", "mps"]:
        d = torch.device(device)
        sim = Simulator(Crystal(CrystalConfig(**{**CRYSTAL, "default_F": 0.0}), device=d, dtype=torch.float32),
                        Detector(DetectorConfig(**{**DETECTOR, **detector}), device=d, dtype=torch.float32),
                        beam_config=BeamConfig(**BEAM), device=d, dtype=torch.float32,
                        diffuse=constant(F0 ** 2))
        images[device] = sim.run(oversample=1).cpu()
    assert torch.isfinite(images["mps"]).all() and images["cpu"].max() > 0
    assert torch.abs(images["mps"] / images["cpu"] - 1).max() < 1e-5


def test_unknown_orientation_mode_is_refused():
    with pytest.raises(ValueError, match="diffuse_orientations"):
        simulator(diffuse=constant(1.0), diffuse_orientations="some")
