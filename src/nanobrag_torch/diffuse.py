"""
Diffuse scattering models for ``Simulator(diffuse=...)``: the protocol and reference models.

This is an extension, not part of nanoBragg.c. A diffuse model maps fractional Miller
indices to scattering per unit cell. The Simulator renders it through the Bragg term's own
path with N_cells * I_cell(hkl) in place of (F_cell * F_latt)^2 at every sample (subpixel,
sensor layer, source, phi step, mosaic domain), since diffuse scattering adds incoherently
from the cells. Solid angle, polarization, capture, dmin, source weights and the division
by steps then follow the Bragg rules exactly, and the two terms share one absolute scale
(see Simulator._diffuse_physics_for_position).

torchBragg never imports a diffuse library. Physical models (for example eryx's lattice
dynamics) implement DiffuseModel themselves; the reference models here are small, closed
form and differentiable, for tests, demonstrations and recovery checks.
"""
import math
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from .config import CrystalShape, SpotMetric

__all__ = ["DiffuseModel", "ConstantDiffuse", "GaussianHalo", "DiffuseGrid", "bragg_peak_factor"]


@runtime_checkable
class DiffuseModel(Protocol):
    """
    Anything callable as ``model(hkl) -> intensity``.

    hkl: (n, 3) fractional Miller indices in the lattice of the crystal's real-space vectors,
        on the simulator's device and in its dtype; n counts every sample, source and
        orientation of the pixels being rendered (pixel_batch_size bounds it). It carries
        the autograd graph of the geometry (orientation, cell, detector, wavelength); a model
        that wants those gradients keeps it in its graph.
    returns: (n,) scattering per unit cell in electrons squared, real and finite everywhere,
        on any device and in any floating dtype (the Simulator converts it).

    An nn.Module is the usual implementation: its parameters are then ordinary torch
    parameters that an optimizer can refine.
    """

    def __call__(self, hkl: torch.Tensor) -> torch.Tensor: ...

    # Optional, not part of the runtime check: ``grid() -> DiffuseGrid``, the model's values on a
    # reciprocal grid (nanobrag_torch.diffuse_grid). DiffuseGrid.from_model samples any model.


class ConstantDiffuse(nn.Module):
    """The same scattering per cell everywhere, ``value`` electrons squared (refinable, log)."""

    def __init__(self, value: float, dtype: torch.dtype = torch.float64):
        super().__init__()
        if not value > 0:
            raise ValueError(f"value must be positive, got {value!r}")
        self.log_value = nn.Parameter(torch.tensor(math.log(value), dtype=dtype))

    def forward(self, hkl: torch.Tensor) -> torch.Tensor:
        return self.log_value.exp().to(device=hkl.device, dtype=hkl.dtype).expand(hkl.shape[0])


class GaussianHalo(nn.Module):
    """
    A Gaussian halo about every lattice point:

        I(hkl) = amplitude * exp(-|hkl - round(hkl)|^2 / (2 sigma^2))

    with the distance measured in Miller units and amplitude (electrons squared per cell) and
    sigma refinable, both as logs. round() has zero derivative, so the gradient with respect
    to hkl, and so to the geometry, is that of the offset from the nearest lattice point.
    A stand-in for thermal diffuse halos: sharp near Bragg peaks, smooth elsewhere.
    """

    def __init__(self, amplitude: float, sigma: float, dtype: torch.dtype = torch.float64):
        super().__init__()
        if not (amplitude > 0 and sigma > 0):
            raise ValueError(f"amplitude and sigma must be positive, got {amplitude!r}, {sigma!r}")
        self.log_amplitude = nn.Parameter(torch.tensor(math.log(amplitude), dtype=dtype))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma), dtype=dtype))

    def forward(self, hkl: torch.Tensor) -> torch.Tensor:
        offset = hkl - torch.round(hkl)
        sigma = self.log_sigma.exp().to(device=hkl.device, dtype=hkl.dtype)
        amplitude = self.log_amplitude.exp().to(device=hkl.device, dtype=hkl.dtype)
        return amplitude * torch.exp(-(offset * offset).sum(dim=-1) / (2 * sigma * sigma))


def bragg_peak_factor(shape: CrystalShape, N: torch.Tensor, spot_metric: SpotMetric = SpotMetric.RECIPROCAL,
                      fudge: float = 1.0, reciprocal_vectors: Optional[torch.Tensor] = None,
                      N_cells_def=None) -> torch.Tensor:
    """The integral of F_latt^2 over one Bragg peak's cell of hkl, over Na*Nb*Nc.

    The diffuse term carries N_cells * I_d per unit hkl volume, so a Bragg peak that integrates
    to factor * N_cells * |F|^2 makes integrated Bragg over diffuse factor * |F|^2 / I_d.
    ExperimentModel multiplies the diffuse term by this factor (an extension; nanoBragg.c's
    lattice factor is unchanged), which makes that ratio |F|^2 / I_d for every shape. The
    kernel assigns each point to its nearest lattice point, so the integral is over that
    point's cell, |h - h0| < 1/2, and a peak wider than the cell is cut by it. Measured against
    the kernel in tests/test_diffuse_shape_integral.py (D2-T1).

    N: (Na, Nb, Nc), a tensor that may carry a graph; the result is differentiable in N, in
        reciprocal_vectors and in N_cells_def. fudge as nanoBragg.c's -fudge.
    reciprocal_vectors: for spot_metric=RECIPROCAL, the rows a*, b*, c* (1/Angstrom) that the
        kernel measures the spot radius with (Crystal's, times c_reciprocal_scale).
    N_cells_def: diffBragg's (Nd, Ne, Nf), for GAUSS with spot_metric=HKL.

    - SQUARE: 1 exactly (sincg^2 integrates to N per axis over its period).
    - GAUSS and TOPHAT: the radius is d^T M d for the offset d = h - h0, with M = NABC^2 for the
      HKL metric (NABC = diag(N), with Ndef off the diagonal) and M = (Na Nb Nc)^2 R R^T for the
      RECIPROCAL one, which is not dimensionless: there the peak's width in h is about
      1/(N_cells |a*|), so a large cell or a small crystal makes it wider than the cell.
      Untruncated, GAUSS gives (0.315 pi / fudge)^(3/2) / sqrt(det M) and TOPHAT
      (4/3) pi (0.3969 / fudge)^(3/2) / sqrt(det M), times n_scale^2 / N_cells (n_scale is the
      kernel's peak height, det NABC with Ndef). These closed forms are used where the cell cuts
      less than 1e-10 of the Gaussian (a bound from its marginals) or none of the ball; otherwise
      the cut integral is computed: Gauss-Legendre in 3D for GAUSS (smooth, 48 nodes a side), and
      for TOPHAT the length of the ball's chord inside the cell along l, integrated over h and k
      (192 nodes a side; the edges of the ball make this good to about 1e-4).
    - ROUND: f^(-3/2) untruncated (0.723601254558268^2 = pi/6, and sinc3(pi sqrt(f) r)^2
      integrates to (6/pi) f^(-3/2)), but its 1/r^4 tail is cut at the cell, about 1 - 0.5/N.
      Computed radially, as the integral of sinc3^2 times the area of the sphere of radius r
      inside the cell (_round_peak_factor): work linear in N, differentiable in N.
    """
    N = torch.as_tensor(N)
    if not N.is_floating_point():
        N = N.to(torch.float64)
    if shape == CrystalShape.SQUARE:
        return torch.ones((), dtype=N.dtype, device=N.device)
    if shape == CrystalShape.ROUND:
        return _round_peak_factor(N, float(fudge))
    if shape not in (CrystalShape.GAUSS, CrystalShape.TOPHAT):
        raise ValueError(f"Unsupported crystal shape: {shape}")
    Na, Nb, Nc = N.unbind()
    n_cells = Na * Nb * Nc
    n_scale = n_cells
    if spot_metric is SpotMetric.RECIPROCAL:
        if reciprocal_vectors is None:
            raise ValueError("the RECIPROCAL spot metric needs the reciprocal vectors the kernel measures with")
        R = torch.as_tensor(reciprocal_vectors).to(dtype=N.dtype, device=N.device)
        M = n_cells * n_cells * (R @ R.T)
    else:
        zero = torch.zeros((), dtype=N.dtype, device=N.device)
        Nd = Ne = Nf = zero
        if N_cells_def is not None and shape == CrystalShape.GAUSS:
            Nd, Ne, Nf = (torch.as_tensor(v).to(dtype=N.dtype, device=N.device) for v in N_cells_def)
            n_scale = n_cells - Na * Ne * Ne - Nb * Nf * Nf - Nc * Nd * Nd + 2 * Nd * Ne * Nf
        NABC = torch.stack([torch.stack([Na, Nd, Nf]), torch.stack([Nd, Nb, Ne]), torch.stack([Nf, Ne, Nc])])
        M = NABC @ NABC
    if shape == CrystalShape.GAUSS:
        integral = _cut_gaussian(M, 2.0 * fudge / 0.63)
    else:
        integral = _cut_ball(M, 0.3969 / fudge)
    return n_scale * n_scale * integral / n_cells


def _gauss_legendre(n: int, like: torch.Tensor):
    """n Gauss-Legendre nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return (torch.as_tensor(0.5 * (x + 1), dtype=like.dtype, device=like.device),
            torch.as_tensor(0.5 * w, dtype=like.dtype, device=like.device))


def _cut_gaussian(M: torch.Tensor, c: float) -> torch.Tensor:
    """The integral of exp(-c d^T M d) over the cell |d_i| < 1/2."""
    marginal_variance = torch.diagonal(torch.linalg.inv(M)) / (2 * c)
    cut = torch.special.erfc(0.5 / torch.sqrt(2 * marginal_variance)).sum()
    if float(cut) < 1e-10:
        return (math.pi / c) ** 1.5 / torch.sqrt(torch.linalg.det(M))
    x, w = _gauss_legendre(48, M)
    d = torch.stack(torch.meshgrid(x - 0.5, x - 0.5, x - 0.5, indexing="ij"), dim=-1)
    weights = w[:, None, None] * w[None, :, None] * w[None, None, :]
    return (weights * torch.exp(-c * torch.einsum("...i,ij,...j->...", d, M, d))).sum()


def _cut_ball(M: torch.Tensor, radius_sqr: float) -> torch.Tensor:
    """The volume of the ellipsoid d^T M d < radius_sqr inside the cell |d_i| < 1/2."""
    extent = torch.sqrt(radius_sqr * torch.diagonal(torch.linalg.inv(M)))
    if float(extent.max()) <= 0.5:  # the whole ball is inside
        return 4.0 / 3.0 * math.pi * radius_sqr ** 1.5 / torch.sqrt(torch.linalg.det(M))
    corners = torch.tensor([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)],
                           dtype=M.dtype, device=M.device)
    if float(torch.einsum("ni,ij,nj->n", corners, M, corners).max()) < radius_sqr:  # the whole cell is inside
        return torch.ones((), dtype=M.dtype, device=M.device)
    # Along l the ball is the chord |l - l0| < w; integrate its length inside the cell over (h, k)
    panels, (x, w) = 24, _gauss_legendre(8, M)
    nodes = ((torch.arange(panels, dtype=M.dtype, device=M.device)[:, None] + x) / panels - 0.5).reshape(-1)
    weights = (w.expand(panels, -1) / panels).reshape(-1)
    h, k = torch.meshgrid(nodes, nodes, indexing="ij")
    beta = M[0, 2] * h + M[1, 2] * k
    gamma = M[0, 0] * h * h + 2 * M[0, 1] * h * k + M[1, 1] * k * k
    disc = beta * beta - M[2, 2] * (gamma - radius_sqr)
    inside = disc > 0
    half = torch.sqrt(torch.where(inside, disc, torch.ones_like(disc))) / M[2, 2]
    centre = -beta / M[2, 2]
    chord = torch.clamp(torch.minimum(centre + half, torch.full_like(half, 0.5))
                        - torch.maximum(centre - half, torch.full_like(half, -0.5)), min=0.0)
    chord = torch.where(inside, chord, torch.zeros_like(chord))
    return (weights[:, None] * weights[None, :] * chord).sum()


def _sinc3_sqr(x: torch.Tensor) -> torch.Tensor:
    small = x < 1e-2
    safe = torch.where(small, torch.ones_like(x), x)
    value = torch.where(small, 1 - x * x / 10 + x ** 4 / 280,
                        3.0 * (torch.sin(safe) / safe - torch.cos(safe)) / (safe * safe))
    return value * value


def _safe_acos(t: torch.Tensor) -> torch.Tensor:
    """acos(t) for t < 1, else 0, with a finite gradient everywhere."""
    below = t < 1
    return torch.where(below, torch.acos(torch.where(below, t, torch.zeros_like(t))), torch.zeros_like(t))


def _smooth_panels(lo: torch.Tensor, hi: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    """Gauss-Legendre on [lo, hi] through u = 3x^2 - 2x^3, whose zero slope at both ends takes the
    square-root behaviour of the integrands at their breakpoints."""
    u = 3 * x * x - 2 * x ** 3
    du = 6 * x * (1 - x)
    span = (hi - lo)[..., None]
    return lo[..., None] + span * u, span * w * du


def _round_peak_factor(N: torch.Tensor, fudge: float) -> torch.Tensor:
    """(pi/6) times the integral of sinc3(pi sqrt(fudge) |u|)^2 over the box |u_i| < N_i / 2.

    u_i = N_i (h_i - h0_i), so this is the ROUND peak's integral over its cell of hkl, over
    N_cells. As a radial integral: sinc3^2(r) times S(r), the area of the sphere of radius r inside
    the box, which needs no more nodes per period of sinc3^2 than a 1D integral does, so the work
    grows as N. S(r) = 2 r times the integral over 0 < z < min(r, c) of L(rho), the angle of the
    circle of radius rho = sqrt(r^2 - z^2) inside the rectangle |x| < a, |y| < b, in closed form.
    The node positions follow N but carry no gradient: an interior breakpoint does not change an
    exact integral, the outer limit sqrt(a^2 + b^2 + c^2) is where S vanishes, and with ties
    (a = b) a sorted breakpoint would hand its boundary term to the wrong axis. The gradient is
    then the integral of the integrand's derivative in N, which the panels, split at its kinks,
    integrate as well as the value; z < c is a true limit and keeps its gradient.
    """
    a, b, c = (0.5 * n for n in N.unbind())
    zero = torch.zeros_like(a)
    hab, hac, hbc = torch.sqrt(a * a + b * b), torch.sqrt(a * a + c * c), torch.sqrt(b * b + c * c)
    breaks = torch.sort(torch.stack([zero, a, b, c, hab, hac, hbc, torch.sqrt(a * a + b * b + c * c)])).values.detach()
    x8, w8 = _gauss_legendre(16, N)
    width = 0.5 / math.sqrt(fudge)  # half a period of sinc3^2
    r_nodes, r_weights = [], []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        count = math.ceil(float(hi - lo) / width)
        if count == 0:
            continue
        steps = torch.arange(count + 1, dtype=N.dtype, device=N.device) / count
        edges = lo + (hi - lo) * steps
        nodes, weights = _smooth_panels(edges[:-1], edges[1:], x8, w8)
        r_nodes.append(nodes.reshape(-1))
        r_weights.append(weights.reshape(-1))
    r, wr = torch.cat(r_nodes), torch.cat(r_weights)

    z_top = torch.minimum(r, c.expand_as(r))
    cuts = [torch.minimum(torch.sqrt(torch.clamp(r * r - t.detach() ** 2, min=0.0)), z_top.detach()) for t in (a, b, hab)]
    # z_top, the one limit with a gradient, stays last: sorted among equal cuts it could land on a
    # panel of zero length, whose boundary term has the wrong sign
    interior = torch.sort(torch.stack([torch.zeros_like(r), *cuts], dim=-1), dim=-1).values
    z_breaks = torch.cat([interior, z_top[:, None]], dim=-1)
    x12, w12 = _gauss_legendre(20, N)
    z, wz = _smooth_panels(z_breaks[:, :-1], z_breaks[:, 1:], x12, w12)  # (nodes, 4, 12)
    rho = torch.sqrt(torch.clamp(r[:, None, None] ** 2 - z * z, min=1e-300))
    acos_a, acos_b = _safe_acos(a / rho), _safe_acos(b / rho)
    corner = torch.clamp(acos_a - (0.5 * math.pi - acos_b), min=0.0)  # asin(b/rho) = pi/2 - acos(b/rho)
    angle = 2 * math.pi - 4 * acos_a - 4 * acos_b + 4 * corner
    area = 2 * r * (wz * angle).sum(dim=(-2, -1))
    return 0.723601254558268 ** 2 * (wr * _sinc3_sqr(math.pi * math.sqrt(fudge) * r) * area).sum()


from .diffuse_grid import DiffuseGrid  # noqa: E402  (after the models: it imports none of them)
