"""Physics that holds without nanoBragg in C, for the cases where C itself is wrong.

See docs/bugs/verified_c_bugs.md. Each test states a law, not a reference image: sources
add, a rotation sweep is the mean of its steps, per-sample corrections are local, and a
grating factor cannot exceed N.
"""
import math

import numpy as np
import pytest
import torch

from nanobrag_torch.config import (BeamConfig, CrystalConfig, CrystalShape, DetectorConfig,
                                   DetectorConvention)
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator
from nanobrag_torch.utils.physics import sincg, sincg_miller

TILT = torch.tensor([-0.2, 0.1, 0.0], dtype=torch.float64)      # sample -> source, not along the beam
BEAM = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)


def unit(v):
    return v / v.norm()


def same(image):
    """Relative only: pytest.approx adds an absolute 1e-12, about the size of these images."""
    return pytest.approx(image, rel=1e-9, abs=0)


def image(dtype=torch.float64, sources=None, wavelengths=None, nopolar=False, crystal=None, detector=None):
    cc = dict(cell_a=70.0, cell_b=80.0, cell_c=90.0, cell_alpha=75.0, cell_beta=85.0, cell_gamma=95.0,
              N_cells=(5, 5, 5), default_F=100.0, misset_deg=(10.0, 20.0, 30.0))
    cc.update(crystal or {})
    dc = dict(detector_convention=DetectorConvention.MOSFLM, distance_mm=100.0, pixel_size_mm=0.1,
              spixels=48, fpixels=48, oversample=1)
    dc.update(detector or {})
    bc = dict(wavelength_A=1.0, fluence=1e15, nopolar=nopolar)
    if sources is not None:
        n = len(sources)
        bc.update(source_directions=torch.stack([unit(s) for s in sources]),
                  source_weights=torch.ones(n, dtype=torch.float64),
                  source_wavelengths=torch.full((n,), 1e-10, dtype=torch.float64)
                  if wavelengths is None else torch.as_tensor(wavelengths, dtype=torch.float64))
    cconf, dconf, bconf = CrystalConfig(**cc), DetectorConfig(**dc), BeamConfig(**bc)
    sim = Simulator(Crystal(cconf, dtype=dtype), Detector(dconf, dtype=dtype), cconf, bconf, dtype=dtype)
    return sim.run().detach().cpu().numpy().astype(float)


class TestSources:
    def test_one_explicit_source_sets_the_beam(self):
        """One source off the beam axis is a different experiment from the default beam."""
        assert np.abs(image(sources=[TILT]) - image()).max() > 0.1 * image().max()

    def test_explicit_source_on_the_beam_is_the_default(self):
        assert image(sources=[BEAM]) == same(image())

    def test_sources_add(self):
        """Two sources are the mean of the two single-source images."""
        a, b, both = image(sources=[BEAM]), image(sources=[TILT]), image(sources=[BEAM, TILT])
        assert both == same((a + b) / 2)

    def test_one_source_equals_itself_twice(self):
        assert image(sources=[TILT]) == same(image(sources=[TILT, TILT]))

    def test_single_source_polarization_uses_that_source(self):
        """Kahn factor 0: polar = (1 + cos^2 2theta)/2 with 2theta measured from the source."""
        d = dict(spixels=8, fpixels=8)
        ratio = image(sources=[TILT], detector=d) / image(sources=[TILT], nopolar=True, detector=d)
        det = Detector(DetectorConfig(detector_convention=DetectorConvention.MOSFLM, distance_mm=100.0,
                                      pixel_size_mm=0.1, spixels=8, fpixels=8), dtype=torch.float64)
        out = det.get_pixel_coords().to(torch.float64)
        out = out / out.norm(dim=-1, keepdim=True)
        cos2t = (out * -unit(TILT)).sum(-1).numpy()
        assert ratio == pytest.approx(0.5 * (1 + cos2t ** 2), rel=1e-9)

    def test_single_source_wavelength_is_used(self):
        """A lone source's own wavelength sets the pattern, as the sourcefile column would."""
        assert np.abs(image(sources=[BEAM], wavelengths=[1.1e-10]) - image()).max() > 0.1 * image().max()


class TestRotation:
    def test_sweep_from_zero_is_the_mean_of_its_steps(self):
        """C-PARITY-001: the phi = 0 step must be rendered at phi = 0."""
        steps = [image(crystal=dict(phi_start_deg=p, osc_range_deg=0.0, phi_steps=1)) for p in (0.0, 1.0, 2.0)]
        sweep = image(crystal=dict(phi_start_deg=0.0, osc_range_deg=3.0, phi_steps=3))
        assert sweep == same(np.mean(steps, axis=0))


class TestPerSampleCorrections:
    def test_oversample_omega_is_a_local_correction(self):
        """Solid angle varies slowly, so doing it per sample barely moves a pixel."""
        d = dict(oversample=2)
        ratio = image(detector=dict(d, oversample_omega=True)).sum() / image(detector=d).sum()
        assert ratio == pytest.approx(1.0, abs=2e-3)

    def test_oversample_thick_sums_capture_over_layers(self):
        """A very thin sensor captures 1 - exp(-t mu / cos), summed over layers."""
        t_um, mu_um = 1e-3, 1e-3
        d = dict(detector_thick_um=t_um, detector_abs_um=mu_um, detector_thicksteps=4, oversample=2,
                 oversample_thick=True)
        thin = image(detector=dict(oversample=2))
        thick = image(detector=d)
        det = Detector(DetectorConfig(detector_convention=DetectorConvention.MOSFLM, distance_mm=100.0,
                                      pixel_size_mm=0.1, spixels=48, fpixels=48), dtype=torch.float64)
        out = det.get_pixel_coords().to(torch.float64)
        parallax = (out[..., 0] / out.norm(dim=-1)).numpy()
        step = t_um / 3
        captured = 1 - np.exp(-4 * step / mu_um / parallax)
        assert (thick / thin) == pytest.approx(captured, rel=2e-3)


class TestGratingFactor:
    H = torch.tensor([99.999975, 100.00002, 99.999985, -99.99998, 37.0, 37.00001, 12.5, 250.000012, 0.0, 3e-6],
                     dtype=torch.float32)

    @pytest.mark.parametrize("n", [5.0, 13.0, 100.0])
    def test_single_precision_keeps_the_peak(self, n):
        exact = sincg(math.pi * self.H.double(), torch.tensor(n, dtype=torch.float64))
        got = sincg_miller(self.H, torch.tensor(n, dtype=torch.float32))
        assert got.abs().max() <= n * (1 + 1e-5)
        assert got.double() == pytest.approx(exact, rel=1e-3, abs=1e-3)

    def test_old_call_was_broken_in_single_precision(self):
        """The reason sincg_miller exists: pi*h rounds away the distance from the peak."""
        old = sincg(torch.pi * self.H, torch.tensor(5.0, dtype=torch.float32))
        assert old.abs().max() > 5.0 * 1.2

    @pytest.mark.parametrize("n", [2.0, 5.0, 6.0, 13.0])
    def test_matches_sin_ratio_everywhere(self, n):
        h = torch.linspace(-7.3, 7.3, 1001, dtype=torch.float64)
        h = h[(h - h.round()).abs() > 1e-6]
        want = torch.sin(n * math.pi * h) / torch.sin(math.pi * h)
        assert sincg_miller(h, torch.tensor(n, dtype=torch.float64)) == pytest.approx(want, rel=1e-9, abs=1e-9)

    def test_limit_at_integers_carries_the_sign(self):
        h = torch.tensor([0.0, 1.0, 2.0, -3.0], dtype=torch.float64)
        got = sincg_miller(h, torch.tensor(4.0, dtype=torch.float64))
        assert got.tolist() == pytest.approx([4.0, -4.0, 4.0, -4.0])

    def test_non_integer_n_follows_the_definition(self):
        h, n = torch.tensor([2.3, 41.7], dtype=torch.float64), 5.3
        want = torch.sin(n * math.pi * h) / torch.sin(math.pi * h)
        assert sincg_miller(h, torch.tensor(n, dtype=torch.float64)) == pytest.approx(want, rel=1e-9)

    def test_gradients_match_finite_differences(self):
        h = torch.tensor([3.21, 40.6], dtype=torch.float64, requires_grad=True)
        n = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
        gh, gn = torch.autograd.grad(sincg_miller(h, n).sum(), (h, n))
        e = 1e-6
        with torch.no_grad():
            fd_h = (sincg_miller(h + e, n) - sincg_miller(h - e, n)) / (2 * e)
            fd_n = ((sincg_miller(h, n + e) - sincg_miller(h, n - e)) / (2 * e)).sum()
        assert gh == pytest.approx(fd_h, rel=1e-6)
        assert gn.item() == pytest.approx(fd_n.item(), rel=1e-6)

    def test_square_image_agrees_between_precisions(self):
        """A pixel next to a peak at large h, where the old float32 image had |F_latt| > N."""
        c = dict(cell_a=100.0, cell_b=100.0, cell_c=100.0, cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0,
                 misset_deg=(0.0, 0.0, 0.0), shape=CrystalShape.SQUARE)
        s = [torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)]
        a = image(dtype=torch.float32, sources=s, crystal=c, nopolar=True, detector=dict(spixels=128, fpixels=128))
        b = image(dtype=torch.float64, sources=s, crystal=c, nopolar=True, detector=dict(spixels=128, fpixels=128))
        assert a == pytest.approx(b, rel=5e-3, abs=1e-3 * b.max())
