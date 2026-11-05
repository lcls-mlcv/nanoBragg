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
from typing import Protocol, runtime_checkable

import torch
from torch import nn

__all__ = ["DiffuseModel", "ConstantDiffuse", "GaussianHalo"]


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
