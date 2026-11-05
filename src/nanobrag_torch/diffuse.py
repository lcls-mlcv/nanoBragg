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
import functools
import math
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from .config import CrystalShape, SpotMetric

__all__ = ["DiffuseModel", "ConstantDiffuse", "GaussianHalo", "bragg_peak_factor"]


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
                      fudge: float = 1.0, volume: Optional[torch.Tensor] = None,
                      N_cells_def=None) -> torch.Tensor:
    """The integral of F_latt^2 over one Bragg peak's cell of hkl, over Na*Nb*Nc.

    The diffuse term carries N_cells * I_d per unit hkl volume, so a Bragg peak that integrates
    to factor * N_cells * |F|^2 makes integrated Bragg over diffuse factor * |F|^2 / I_d.
    ExperimentModel multiplies the diffuse term by this factor (an extension; nanoBragg.c's
    lattice factor is unchanged), which makes that ratio |F|^2 / I_d for every shape. Measured
    against the kernel in tests/test_diffuse_shape_integral.py (D2-T1).

    N: (Na, Nb, Nc), a tensor that may carry a graph. fudge as nanoBragg.c's -fudge.
    volume: for spot_metric=RECIPROCAL, the volume in Angstrom^3 of the reciprocal vectors the
        kernel measures the radius with (Crystal.V / Crystal.c_reciprocal_scale**3).
    N_cells_def: diffBragg's (Nd, Ne, Nf), for GAUSS with spot_metric=HKL.

    - SQUARE: 1 exactly (sincg^2 integrates to N per axis).
    - GAUSS: (0.315 pi / fudge)^(3/2), from exp(-2 r^2 fudge / 0.63); with N_cells_def, times
      |det NABC| / (Na Nb Nc). TOPHAT: (4/3) pi (0.3969 / fudge)^(3/2), its ball. Both are the
      untruncated integrals, which the cell cuts by under 1e-6 at N >= 5 (3e-4 at N = 3).
    - With spot_metric=RECIPROCAL the radius is |dh a* + dk b* + dl c*|^2 (Na Nb Nc)^2, not
      dimensionless, and the GAUSS or TOPHAT constant is multiplied by volume / (Na Nb Nc)^2.
    - ROUND: 1 untruncated (0.723601254558268^2 = pi/6, sinc3^2 integrates to 6/pi), but its
      1/r^4 tail is cut at the cell, about 1 - 0.5/N. No closed form: Gauss-Legendre quadrature of
      sinc3^2 over the cell, to 1e-6, cached by (N, fudge) and without a gradient in N.
    """
    N = torch.as_tensor(N)
    Na, Nb, Nc = N[0], N[1], N[2]
    n_cells = Na * Nb * Nc
    if shape == CrystalShape.SQUARE:
        return torch.ones((), dtype=N.dtype, device=N.device)
    if shape == CrystalShape.ROUND:
        values = [float(v) for v in N.detach().cpu()]
        return torch.tensor(_round_peak_factor(*values, float(fudge)), dtype=N.dtype, device=N.device)
    if shape == CrystalShape.GAUSS:
        constant = (0.315 * math.pi / fudge) ** 1.5
    elif shape == CrystalShape.TOPHAT:
        constant = 4.0 / 3.0 * math.pi * (0.3969 / fudge) ** 1.5
    else:
        raise ValueError(f"Unsupported crystal shape: {shape}")
    if spot_metric is SpotMetric.RECIPROCAL:
        if volume is None:
            raise ValueError("the RECIPROCAL spot metric needs the volume of the reciprocal vectors")
        return constant * volume / (n_cells * n_cells)
    if N_cells_def is not None and shape == CrystalShape.GAUSS:
        Nd, Ne, Nf = (torch.as_tensor(v, dtype=N.dtype, device=N.device) for v in N_cells_def)
        det = n_cells - Na * Ne * Ne - Nb * Nf * Nf - Nc * Nd * Nd + 2 * Nd * Ne * Nf
        return constant * det.abs() / n_cells
    return constant * torch.ones_like(n_cells)


@functools.lru_cache(maxsize=64)
def _round_peak_factor(Na: float, Nb: float, Nc: float, fudge: float) -> float:
    """(pi/6) * the integral of sinc3(pi sqrt(fudge) |u|)^2 over the box |u_i| < N_i / 2.

    u_i = N_i (h_i - h0_i), so this is the ROUND peak's integral over its cell of hkl, over N_cells.
    Panels of half a period of sinc3^2 with 6 Gauss-Legendre nodes each, on one octant.
    """
    nodes, weights = np.polynomial.legendre.leggauss(6)
    width = 0.5 / math.sqrt(max(fudge, 1.0))

    def axis(n):
        half = 0.5 * n
        edges = np.linspace(0.0, half, max(1, math.ceil(half / width)) + 1)
        mid, span = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
        return ((mid[:, None] + span[:, None] * nodes).ravel(), (span[:, None] * weights).ravel())

    (x, wx), (y, wy), (z, wz) = axis(Na), axis(Nb), axis(Nc)
    yz2 = (y[:, None] ** 2 + z[None, :] ** 2)
    wyz = wy[:, None] * wz[None, :]
    total = 0.0
    for xi, wxi in zip(x, wx):
        r = math.pi * np.sqrt(fudge * (xi * xi + yz2))
        safe = np.where(r == 0.0, 1.0, r)
        sinc3 = np.where(r == 0.0, 1.0, 3.0 * (np.sin(safe) / safe - np.cos(safe)) / (safe * safe))
        total += wxi * float((wyz * sinc3 * sinc3).sum())
    return 8.0 * 0.723601254558268 ** 2 * total
