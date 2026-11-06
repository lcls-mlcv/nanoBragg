"""
D2-T1 (docs/diffuse_joint_plan.md, section 2.1): the lattice factor F_latt^2 integrated over one
Bragg peak in hkl, per crystal shape, relative to N_cells.

The diffuse term carries N_cells * I_d per unit hkl volume. The Bragg term is tied to it with the
same scale only if a Bragg peak integrates to N_cells * |F|^2 in hkl, which is what SQUARE gives
(the integral of sincg^2 over a period is N per axis). This measures the factor for each shape
through compute_physics_for_position itself, the C-parity function, and records it. Nothing in
the C path is changed by it.

How: one pixel, so one scattering vector S, and one "mosaic domain" per point of a midpoint grid
over the cell of hkl about the lattice point (3, -2, 5): domain i has rot_a = h_i S / |S|^2 (and
likewise b, c), so S . rot_a = h_i. The kernel sums F_cell^2 F_latt^2 over domains, so with
F_cell = 1 and no polarization its output over the number of grid points is the integral of
F_latt^2 over the cell. Each point is rounded to its nearest lattice point by the kernel, so the
integral is of the one peak, truncated at the cell boundary, as the image has it.

Measured (2026-11-05, CPU float64), factor = integral / N_cells:
- SQUARE: 1 to 1e-15 (exact for a grid of at least N points per axis).
- GAUSS, spot_metric=HKL: (0.315 pi)^(3/2) = 0.98444, from exp(-2 r^2 / 0.63).
- TOPHAT, spot_metric=HKL: (4/3) pi 0.63^3 = 1.04739, the volume of the r < 0.63 ball.
- ROUND: 1 in the limit (0.723601254558268^2 = pi/6, and sinc3^2 integrates to 6/pi), but the
  sinc3^2 tail falls only as 1/r^4, so the cell truncates it: 0.89722 (N = 5), 0.94960 (10),
  0.97475 (20), about 1 - 0.5/N.
- GAUSS and TOPHAT, spot_metric=RECIPROCAL (the default, bl831 nanoBragg.c): the radius is
  |dh a* + dk b* + dl c*|^2 (Na Nb Nc)^2 with a* in 1/Angstrom, so the factor is the HKL one
  times V / N_cells^2, V the volume (Angstrom^3) of C's reciprocal vectors: 2.646 for a
  30 x 35 x 40 cell of 5^3 cells, 0.23 for a lysozyme-sized cell (237,000 A^3) of 10^3 cells.
  It depends on cell and crystal size, so the tie is not N_cells there.
"""
import math
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import pytest
import torch

from nanobrag_torch.config import CrystalConfig, CrystalShape, SpotMetric
from nanobrag_torch.diffuse import bragg_peak_factor
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.simulator import compute_physics_for_position

DTYPE = torch.float64
GAUSS = (0.315 * math.pi) ** 1.5
TOPHAT = 4.0 / 3.0 * math.pi * 0.63 ** 3
ORTHORHOMBIC = (30.0, 35.0, 40.0, 90.0, 90.0, 90.0)


def peak_integral(shape, N, metric=SpotMetric.HKL, points=96, cell=ORTHORHOMBIC, chunk=1 << 17, fudge=1.0,
                  N_cells_def=None):
    """The integral of F_latt^2 over the cell of hkl about one lattice point, over N_cells; and the crystal."""
    crystal = Crystal(CrystalConfig(cell_a=cell[0], cell_b=cell[1], cell_c=cell[2], cell_alpha=cell[3],
                                    cell_beta=cell[4], cell_gamma=cell[5], N_cells=N, shape=shape,
                                    spot_metric=metric, fudge=fudge), dtype=DTYPE)
    # The reciprocal vectors as Simulator.run hands them to the kernel for the RECIPROCAL metric
    stars = [v * crystal.c_reciprocal_scale for v in (crystal.a_star, crystal.b_star, crystal.c_star)]
    pixel = torch.tensor([[0.1, 0.02, 0.01]], dtype=DTYPE)
    incident = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    wavelength = torch.tensor(1.0, dtype=DTYPE)
    S = ((pixel / pixel.norm() - incident) / (wavelength * 1e-10))[0]
    offsets = (torch.arange(points, dtype=DTYPE) + 0.5) / points - 0.5
    grid = torch.stack(torch.meshgrid(offsets, offsets, offsets, indexing="ij"), dim=-1).reshape(-1, 3)
    grid = grid + torch.tensor([3.0, -2.0, 5.0], dtype=DTYPE)
    Na, Nb, Nc = (torch.tensor(float(n), dtype=DTYPE) for n in N)
    total = torch.zeros((), dtype=DTYPE)
    for start in range(0, len(grid), chunk):
        hkl = grid[start:start + chunk]
        rot = [(hkl[:, i:i + 1] * S / (S @ S)).unsqueeze(0) for i in range(3)]
        rot_star = [v.expand(1, len(hkl), 3) for v in stars]
        intensity, _ = compute_physics_for_position(
            pixel, *rot, *rot_star, incident_beam_direction=incident, wavelength=wavelength,
            crystal_get_structure_factor=lambda h, k, l: torch.ones_like(h),  # noqa: E741
            N_cells_a=Na, N_cells_b=Nb, N_cells_c=Nc, crystal_shape=shape, spot_metric=metric,
            crystal_fudge=fudge, apply_polarization=False,
            N_cells_def=None if N_cells_def is None else torch.tensor(N_cells_def, dtype=DTYPE))
        total = total + intensity.sum()
    return float(total / points ** 3 / (N[0] * N[1] * N[2])), crystal


def test_the_kernel_sees_the_miller_indices_of_the_grid():
    """The construction: with rot_a = h S / |S|^2 the kernel computes this h (read through its diffuse hook)."""
    seen = {}

    def record(hkl):
        seen["hkl"] = hkl.clone()
        return torch.zeros(len(hkl), dtype=DTYPE)

    pixel = torch.tensor([[0.1, 0.02, 0.01]], dtype=DTYPE)
    incident = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    S = ((pixel / pixel.norm() - incident) / 1e-10)[0]
    hkl = torch.tensor([[3.25, -2.4, 5.1], [2.6, -1.5, 4.9]], dtype=DTYPE)
    rot = [(hkl[:, i:i + 1] * S / (S @ S)).unsqueeze(0) for i in range(3)]
    one = torch.tensor(1.0, dtype=DTYPE)
    compute_physics_for_position(pixel, *rot, *rot, incident_beam_direction=incident, wavelength=one,
                                 N_cells_a=one, N_cells_b=one, N_cells_c=one, apply_polarization=False,
                                 diffuse=record)
    assert torch.allclose(seen["hkl"].reshape(-1, 3), hkl, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("N", [(5, 5, 5), (3, 4, 6)])
def test_square_integrates_to_n_cells(N):
    factor, _ = peak_integral(CrystalShape.SQUARE, N, points=32)
    assert factor == pytest.approx(1.0, abs=1e-12)


def test_gauss_hkl_metric_integrates_to_its_gaussian_volume():
    factor, _ = peak_integral(CrystalShape.GAUSS, (5, 5, 5))
    assert GAUSS == pytest.approx(0.984443146, abs=1e-9)
    assert factor == pytest.approx(GAUSS, rel=1e-7)


def test_tophat_hkl_metric_integrates_to_its_ball():
    factor, _ = peak_integral(CrystalShape.TOPHAT, (3, 3, 3), points=192)
    assert TOPHAT == pytest.approx(1.047394424, abs=1e-9)
    assert factor == pytest.approx(TOPHAT, rel=5e-4)  # the grid's ball: 1.04724 at 192 points


@pytest.mark.parametrize("n,measured", [(5, 0.897221), (10, 0.949601), (20, 0.974753)])
def test_round_falls_short_of_n_cells_by_its_truncated_tail(n, measured):
    assert 0.723601254558268 ** 2 == pytest.approx(math.pi / 6, rel=1e-14)  # untruncated, the factor is 1
    factor, _ = peak_integral(CrystalShape.ROUND, (n, n, n), points=128)
    assert factor == pytest.approx(measured, rel=1e-5)  # grids of 96, 128, 160 agree to 2e-6
    assert 0.45 < (1 - factor) * n < 0.52


@pytest.mark.parametrize("shape,constant,cell,N,tolerance", [
    (CrystalShape.GAUSS, GAUSS, ORTHORHOMBIC, (5, 5, 5), 1e-4),
    (CrystalShape.GAUSS, GAUSS, (30.0, 30.0, 30.0, 90.0, 90.0, 90.0), (5, 5, 5), 1e-4),
    (CrystalShape.GAUSS, GAUSS, (30.0, 35.0, 40.0, 85.0, 95.0, 100.0), (5, 5, 5), 1e-3),
    (CrystalShape.TOPHAT, TOPHAT, (30.0, 30.0, 30.0, 90.0, 90.0, 90.0), (5, 5, 5), 5e-3),
])
def test_reciprocal_metric_factor_scales_as_volume_over_n_cells_squared(shape, constant, cell, N, tolerance):
    factor, crystal = peak_integral(shape, N, SpotMetric.RECIPROCAL, points=192 if shape is CrystalShape.TOPHAT
                                    else 96, cell=cell)
    volume = float(crystal.V) / float(crystal.c_reciprocal_scale) ** 3
    n_cells = N[0] * N[1] * N[2]
    assert factor == pytest.approx(constant * volume / n_cells ** 2, rel=tolerance)
    if cell == ORTHORHOMBIC:
        assert factor == pytest.approx(2.6459, rel=1e-4)  # not 1: the tie is not N_cells here


# The shape correction of the tie (decided 2026-11-05): ExperimentModel multiplies the diffuse term by
# bragg_peak_factor, so that integrated Bragg over diffuse is |F|^2 / I_d whatever the shape. The
# factor must be the kernel's own peak integral, measured here as above.
CUBIC_100 = (100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
TRICLINIC_BIG = (110.0, 120.0, 100.0, 80.0, 95.0, 105.0)
RECIPROCAL = SpotMetric.RECIPROCAL
# shape, N, peak_integral options, tolerance on kernel / factor. The kernel's midpoint grid is the
# limit where the cell cuts the peak (the integrand is not small at the boundary): 1e-4 there.
FACTOR_CASES = {
    "square": (CrystalShape.SQUARE, (5, 5, 5), {}, 1e-12),
    "square-uneven": (CrystalShape.SQUARE, (3, 4, 6), dict(points=32), 1e-12),
    "gauss-hkl": (CrystalShape.GAUSS, (5, 5, 5), {}, 1e-7),
    "gauss-hkl-uneven": (CrystalShape.GAUSS, (3, 4, 6), {}, 1e-6),
    "gauss-hkl-small": (CrystalShape.GAUSS, (1, 2, 2), {}, 1e-4),
    "gauss-hkl-fudge": (CrystalShape.GAUSS, (5, 5, 5), dict(fudge=1.7), 1e-7),
    "gauss-hkl-ndef": (CrystalShape.GAUSS, (6, 6, 6), dict(N_cells_def=(0.8, -0.5, 0.4)), 1e-6),
    "gauss-hkl-ndef-small": (CrystalShape.GAUSS, (2, 3, 2), dict(N_cells_def=(0.5, -0.3, 0.2)), 1e-4),
    "tophat-hkl": (CrystalShape.TOPHAT, (3, 3, 3), dict(points=192), 5e-4),
    "tophat-hkl-small": (CrystalShape.TOPHAT, (1, 1, 2), dict(points=192), 5e-3),
    "round-5": (CrystalShape.ROUND, (5, 5, 5), dict(points=128), 1e-5),
    "round-20": (CrystalShape.ROUND, (20, 20, 20), dict(points=128), 1e-5),
    "round-uneven": (CrystalShape.ROUND, (4, 6, 9), dict(points=128), 1e-5),
    "round-fractional": (CrystalShape.ROUND, (5.5, 6.2, 4.7), dict(points=128), 1e-5),
    "round-fudge": (CrystalShape.ROUND, (6, 6, 6), dict(points=128, fudge=0.6), 1e-5),
    "gauss-reciprocal": (CrystalShape.GAUSS, (5, 5, 5), dict(metric=RECIPROCAL), 1e-4),
    "gauss-reciprocal-triclinic": (CrystalShape.GAUSS, (5, 5, 5), dict(
        metric=RECIPROCAL, cell=(30.0, 35.0, 40.0, 85.0, 95.0, 100.0)), 1e-4),
    "tophat-reciprocal": (CrystalShape.TOPHAT, (5, 5, 5), dict(
        metric=RECIPROCAL, points=192, cell=(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)), 5e-3),
    # The audit's cases (2026-11-05): the cell cuts the peak, which the untruncated forms ignored
    # (kernel / untruncated was 0.910, 0.019, 0.019)
    "gauss-reciprocal-79x79x38": (CrystalShape.GAUSS, (5, 5, 5), dict(metric=RECIPROCAL, cell=(79.0, 79.0, 38.0, 90.0, 90.0, 90.0)), 1e-4),
    "gauss-reciprocal-100-N3": (CrystalShape.GAUSS, (3, 3, 3), dict(metric=RECIPROCAL, cell=CUBIC_100), 1e-4),
    "tophat-reciprocal-100-N3": (CrystalShape.TOPHAT, (3, 3, 3), dict(metric=RECIPROCAL, cell=CUBIC_100), 1e-12),
    "tophat-reciprocal-120-N5": (CrystalShape.TOPHAT, (5, 5, 5), dict(
        metric=RECIPROCAL, points=192, cell=(120.0, 120.0, 120.0, 90.0, 90.0, 90.0)), 5e-3),
    "tophat-reciprocal-triclinic-big": (CrystalShape.TOPHAT, (5, 5, 5), dict(metric=RECIPROCAL, points=192, cell=TRICLINIC_BIG), 5e-3),
    "gauss-reciprocal-triclinic-big": (CrystalShape.GAUSS, (4, 6, 5), dict(metric=RECIPROCAL, cell=TRICLINIC_BIG), 1e-4),
}


def _factor(shape, N, crystal, metric=SpotMetric.HKL, fudge=1.0, N_cells_def=None, **_):
    stars = torch.stack([crystal.a_star, crystal.b_star, crystal.c_star]) * crystal.c_reciprocal_scale
    return bragg_peak_factor(shape, torch.tensor(N, dtype=DTYPE), spot_metric=metric, fudge=fudge,
                             reciprocal_vectors=stars, N_cells_def=N_cells_def)


@pytest.mark.parametrize("case", sorted(FACTOR_CASES))
def test_the_tie_factor_is_the_kernels_peak_integral(case):
    """Integrated Bragg / (factor * diffuse) is |F|^2 / I_d for every shape: the kernel's integral over the factor is 1."""
    shape, N, options, tolerance = FACTOR_CASES[case]
    measured, crystal = peak_integral(shape, N, **options)
    factor = _factor(shape, N, crystal, **options)
    assert measured / float(factor) == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("shape", [CrystalShape.GAUSS, CrystalShape.TOPHAT])
@pytest.mark.parametrize("edge", [30.0, 60.0, 100.0, 150.0])
@pytest.mark.parametrize("n", [3, 5, 10])
def test_reciprocal_metric_factor_over_a_sweep_of_cells_and_sizes(shape, edge, n):
    """From peaks far inside the cell (fast path) to a cell inside the peak (factor = N_cells)."""
    cell = (edge, 1.1 * edge, 0.9 * edge, 90.0, 90.0, 90.0)
    points = 192 if shape is CrystalShape.TOPHAT else 96
    if shape is CrystalShape.TOPHAT and 0.63 * min(cell[:3]) / n ** 3 < 10 / points:
        pytest.skip("a ball under 10 steps of the kernel's grid in radius: its grid volume is no reference")
    measured, crystal = peak_integral(shape, (n, n, n), RECIPROCAL, points=points, cell=cell)
    factor = float(_factor(shape, (n, n, n), crystal, RECIPROCAL))
    assert factor <= n ** 3 * (1 + 1e-12)  # F_latt^2 <= N_cells^2 everywhere
    assert measured / factor == pytest.approx(1.0, abs=1e-4 if shape is CrystalShape.GAUSS else 1e-2)


def test_square_factor_is_exactly_one():
    factor = bragg_peak_factor(CrystalShape.SQUARE, torch.tensor([5.0, 7.0, 3.0], dtype=DTYPE))
    assert factor.item() == 1.0


def _numeric_gradient(fn, x, step=1e-6):
    grads = []
    for i in range(len(x)):
        e = torch.zeros_like(x)
        e[i] = step
        grads.append((float(fn(x + e)) - float(fn(x - e))) / (2 * step))
    return torch.tensor(grads, dtype=DTYPE)


CRYSTAL_STARS = torch.tensor([[1 / 30.0, 0.001, 0.0], [0.0, 1 / 35.0, 0.002], [0.0, 0.0, 1 / 40.0]], dtype=DTYPE)
BIG_STARS = CRYSTAL_STARS * 0.3  # a cell about 3x as large: the peak is cut by the cell


@pytest.mark.parametrize("shape,metric,N,stars", [
    (CrystalShape.GAUSS, RECIPROCAL, (5.0, 6.0, 7.0), CRYSTAL_STARS),  # fast path
    (CrystalShape.GAUSS, RECIPROCAL, (5.0, 6.0, 7.0), BIG_STARS),  # truncated quadrature
    (CrystalShape.TOPHAT, RECIPROCAL, (5.0, 6.0, 7.0), BIG_STARS),
    (CrystalShape.GAUSS, SpotMetric.HKL, (1.5, 2.0, 2.5), None),
    (CrystalShape.ROUND, SpotMetric.HKL, (5.0, 5.0, 5.0), None),
    (CrystalShape.ROUND, SpotMetric.HKL, (4.3, 6.1, 9.2), None),
    (CrystalShape.ROUND, SpotMetric.HKL, (20.0, 20.0, 20.0), None),
])
def test_the_factor_is_differentiable_in_size_and_cell(shape, metric, N, stars):
    N = torch.tensor(N, dtype=DTYPE, requires_grad=True)
    factor = bragg_peak_factor(shape, N, spot_metric=metric, reciprocal_vectors=stars)
    (grad,) = torch.autograd.grad(factor, N)
    fd = _numeric_gradient(lambda n: bragg_peak_factor(shape, n, spot_metric=metric, reciprocal_vectors=stars),
                           N.detach())
    assert fd.abs().max() > 0
    torch.testing.assert_close(grad, fd, rtol=1e-5, atol=1e-7 * float(factor))
    if stars is not None:
        stars = stars.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(bragg_peak_factor(shape, N.detach(), metric, reciprocal_vectors=stars), stars)
        flat = _numeric_gradient(lambda v: bragg_peak_factor(shape, N.detach(), metric, reciprocal_vectors=v.reshape(3, 3)),
                                 stars.detach().reshape(-1), step=1e-9)
        torch.testing.assert_close(grad.reshape(-1), flat, rtol=1e-4, atol=1e-6 * flat.abs().max())


def test_round_cost_is_about_linear_in_n():
    """N^3 cubature took 2.2 s at N = 100 and minutes at 500; the radial form is linear in N."""
    import time
    times = {}
    for n in (5, 20, 100, 500):
        start = time.perf_counter()
        factor = bragg_peak_factor(CrystalShape.ROUND, torch.tensor([n, n, n], dtype=DTYPE))
        times[n] = time.perf_counter() - start
        assert 1 - 0.6 / n < float(factor) < 1
    assert times[500] < 2.0, times
