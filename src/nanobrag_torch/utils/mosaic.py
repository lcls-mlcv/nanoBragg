"""diffBragg's deterministic mosaic-domain ensemble, differentiable in eta.

A port of simtbx.nanoBragg.anisotropic_mosaicity.AnisoUmats (diagonal model,
``how=1``; isotropic is the case eta_a = eta_b = eta_c). For M domains it takes
M/2 rotation axes C spread over a hemisphere (J. Magn. Reson. 138, 288 (1999),
eq. A6), pairs each with a quantile u from a seeded shuffle of {0, 2/M, 4/M, ...},
and rotates about C by ±η_eff·√2·erfinv(u), where

    η_eff = 1 / sqrt(Cᵀ · diag(1/η_a², 1/η_b², 1/η_c²) · C)

is the spread along C of an ellipsoid with semi-axes η_a, η_b, η_c on the lab
x, y, z axes. With ``crystal_axes`` S (columns: unit real-space a, b, c) the
ellipsoid is S·diag(1/η²)·S⁻¹ instead, so η_a, η_b, η_c lie along the crystal
axes: that is diffBragg's refinement path (SimData.update_umats_for_refinement,
``transform_eta=True``, S from the crystal at the start of refinement).

The rotations are a smooth function of η, so autograd gives diffBragg's
d/d(eta) (ETA_ID 19). nanoBragg.c's random mosaic blocks are a different
ensemble; this is what diffBragg refines.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import torch

DIFFBRAGG_UMAT_SEED = 8675309


@lru_cache(maxsize=16)
def _samples(n_domains: int, seed: int):
    """Hemisphere axes (n/2, 3) and shuffled quantiles (n/2,), as AnisoUmats builds them."""
    if n_domains < 2 or n_domains % 2:
        raise ValueError(f"the diffBragg mosaic ensemble needs an even number of domains, got {n_domains}")
    n = n_domains // 2
    t = [(i + 0.5) / n for i in range(n)]  # (arange(1, N+1) - 0.5)/N
    axes = []
    for ti in t:
        theta, phi = math.acos(ti), math.sqrt(math.pi * n) * math.asin(ti)
        axes.append((math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)))
    quantiles = [float(i) / n for i in range(n)]
    random.Random(seed).shuffle(quantiles)
    return axes, quantiles


def diffbragg_umats(eta_deg, n_domains: int, seed: int = DIFFBRAGG_UMAT_SEED,
                    crystal_axes=None, dtype=torch.float64, device=None) -> torch.Tensor:
    """(n_domains, 3, 3) mosaic rotations for spread ``eta_deg`` (degrees).

    ``eta_deg`` is a scalar (isotropic) or a 3-vector (η_a, η_b, η_c on lab x, y, z,
    or on ``crystal_axes`` if given); tensors requiring grad are differentiated through. Order matches AnisoUmats:
    for each axis, the +angle then the −angle rotation.
    """
    if isinstance(eta_deg, torch.Tensor):
        dtype, device = eta_deg.dtype, eta_deg.device
    eta = torch.as_tensor(eta_deg, dtype=dtype, device=device).reshape(-1)
    if eta.numel() == 1:
        eta = eta.expand(3)
    elif eta.numel() != 3:
        raise ValueError("eta_deg must be a scalar or a 3-vector")
    axes_list, quantiles = _samples(int(n_domains), int(seed))
    axes = torch.tensor(axes_list, dtype=dtype, device=device)
    u = torch.tensor(quantiles, dtype=dtype, device=device)

    if crystal_axes is None:
        quad = (axes * axes / (eta * eta)).sum(dim=1)
    else:
        S = torch.as_tensor(crystal_axes, dtype=dtype, device=device)
        tensor = S @ torch.diag(1.0 / (eta * eta)) @ torch.linalg.inv(S)
        quad = torch.einsum("ni,ij,nj->n", axes, tensor, axes).abs()
    eta_eff = 1.0 / torch.sqrt(quad)
    angle = eta_eff * (math.sqrt(2.0) * torch.special.erfinv(u) * math.pi / 180.0)
    angle = torch.stack([angle, -angle], dim=1).reshape(-1)          # (+, −) per axis
    axis = axes.repeat_interleave(2, dim=0)

    # Rodrigues: R = I + sin θ K + (1 − cos θ) K²
    x, y, z = axis.unbind(dim=1)
    zero = torch.zeros_like(x)
    K = torch.stack([torch.stack([zero, -z, y], dim=1),
                     torch.stack([z, zero, -x], dim=1),
                     torch.stack([-y, x, zero], dim=1)], dim=1)
    eye = torch.eye(3, dtype=dtype, device=device).expand_as(K)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return eye + s * K + (1 - c) * (K @ K)
