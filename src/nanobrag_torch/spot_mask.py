"""Masks around predicted Bragg spots, for diffuse and joint fits (plan section 2.2).

Near a lattice point the diffuse core and the Bragg peak trade off, so diffuse and joint fits may
leave out the pixels around each spot. The spots are predicted from the geometry alone, not from an
image: each pixel center's fractional hkl, the nearest integer hkl, and its distance in the HKL
metric, sum_i (N_i d_i)^2, in units of a GAUSS-HKL spot's sd, 0.397/N per axis. Every integer hkl
within n_sd sds of some pixel center is a spot; its center is the pixel nearest it. Each view (one
source at one phi step and mosaic domain) predicts its own spots, and the mask is their union. hkl is
taken at pixel centers only, not at the subpixels an oversampled image also samples.

The mask is a plain boolean tensor, built once from the geometry it is given and carrying no
gradient, so it stays fixed while the geometry refines. Write it into DetectorConfig.mask_array, or
take a strided pixel set from it (strided_pixels).
"""
from typing import Iterator, Tuple

import torch

SPOT_SD = 0.397
"""A GAUSS spot's sd in h under the HKL metric is SPOT_SD / N (plan section 5, V1)."""


def _views(simulator) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """For each source, phi step and mosaic domain: the fractional hkl at each pixel center, (S, F, 3), and
    |s| in 1/A, (S, F). As the kernel computes them: s = (d - s0) / lambda, with d the unit vector to the
    pixel and s0 the incident direction (minus the source direction), and h = a . s."""
    simulator._ensure_pixel_coords_current()
    pixels = simulator._cached_pixel_coords_meters.detach()
    d = pixels / pixels.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    if simulator._source_directions is not None:
        incident = -simulator._source_directions.detach().to(d)
        wavelength_A = simulator._source_wavelengths_A.detach().to(d)
    else:
        incident = simulator.incident_beam_direction.detach().to(d).reshape(1, 3)
        wavelength_A = torch.as_tensor(simulator.wavelength).detach().to(d).reshape(1)
    (rot_a, rot_b, rot_c), _ = simulator.crystal.get_rotated_real_vectors(simulator.crystal.config)
    axes = torch.stack([rot_a, rot_b, rot_c], dim=-2).detach().to(d).reshape(-1, 3, 3)    # (n_orient, 3, 3), A
    for source in range(len(incident)):
        s = (d - incident[source]) / wavelength_A[source]
        s_norm = s.norm(dim=-1)
        for axis in axes:
            yield s @ axis.T, s_norm


def predicted_spot_mask(simulator, radius_px: float, n_sd: float = 3.0) -> torch.Tensor:
    """(S, F) bool, True where a pixel is kept: False within radius_px pixels (distance < radius_px) of a
    predicted spot center. No spot is predicted where |s| > 1/dmin, the kernel's dmin rule."""
    with torch.no_grad():
        N = torch.stack([torch.as_tensor(simulator.crystal.N_cells_a), torch.as_tensor(simulator.crystal.N_cells_b),
                         torch.as_tensor(simulator.crystal.N_cells_c)]).detach()
        dmin = simulator.beam_config.dmin
        centers = None
        for hkl, s_norm in _views(simulator):
            if centers is None:
                S, F, _ = hkl.shape
                centers = torch.zeros(S * F, dtype=torch.bool, device=hkl.device)
                N = N.to(hkl)
            nearest = torch.round(hkl).reshape(-1, 3)
            dist2 = (((hkl.reshape(-1, 3) - nearest) * N) ** 2).sum(-1)
            inside = dist2 < (n_sd * SPOT_SD) ** 2
            if dmin > 0:
                inside &= s_norm.reshape(-1) <= 1.0 / dmin
            pixel = torch.nonzero(inside).flatten()
            if len(pixel):
                _, spot = torch.unique(nearest[pixel].long(), dim=0, return_inverse=True)
                d2 = dist2[pixel]
                best = torch.full((int(spot.max()) + 1,), float("inf"), dtype=d2.dtype, device=d2.device)
                best = best.scatter_reduce(0, spot, d2, reduce="amin")
                # Pixels equally near a spot (a beam on a pixel corner) tie: the lowest pixel index is its center.
                candidate = torch.where(d2 == best[spot], pixel, torch.full_like(pixel, S * F))
                first = torch.full_like(best, S * F, dtype=pixel.dtype).scatter_reduce(0, spot, candidate, reduce="amin")
                centers[first] = True
        return _dilate(centers.reshape(S, F), radius_px)


def _dilate(centers: torch.Tensor, radius_px: float) -> torch.Tensor:
    """keep = no center within distance < radius_px."""
    if radius_px <= 0 or not centers.any():
        return torch.ones_like(centers)
    reach = int(torch.ceil(torch.tensor(radius_px)).item())
    offsets = torch.arange(-reach, reach + 1, device=centers.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    disk = ((dy ** 2 + dx ** 2) < radius_px ** 2).to(torch.float32)
    hit = torch.nn.functional.conv2d(centers.to(torch.float32)[None, None], disk[None, None], padding=reach)[0, 0]
    return hit == 0


def strided_pixels(keep: torch.Tensor, stride: int) -> torch.Tensor:
    """(n, 2) long (slow, fast) of every stride-th kept pixel in row-major order, starting with the first:
    the pixels= format of Simulator.run."""
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride!r}")
    return torch.nonzero(keep)[::stride]
