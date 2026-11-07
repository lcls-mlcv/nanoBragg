"""Masks around predicted Bragg spots, for diffuse and joint fits (plan section 2.2).

Near a lattice point the diffuse core and the Bragg peak trade off, so diffuse and joint fits may
leave out the pixels around each spot. The spots are predicted from the geometry alone, not from an
image, and they may be far smaller than a pixel (lysozyme at N 20: a spot sd of 0.02 in h, a pixel
of about 0.3), so they are found on the detector surface, not at pixel centers:

- Around each pixel, hkl is linear in the in-plane offset delta (pixels, slow and fast):
  hkl(delta) = hkl_0 + J delta, with hkl_0 and J from the coordinates the kernel uses (planar,
  then curved where the detector is curved).
- For the integer hkl g nearest hkl_0, the HKL-metric distance |g - hkl(delta)|_M, with
  M = diag(N^2), is smallest at delta*. That smallest distance is the lattice point's distance from
  the detector surface (its excitation error) and delta* is where the spot falls.
- g is a spot if that distance is within n_sd GAUSS-HKL spot sds (0.397 each, the sd being 0.397/N
  per axis in h) and delta* lies within this pixel (|delta*| <= 1/2, with a margin for the
  linearization; the pixel with the smallest |delta*| wins, then the lowest index).
- Each view (one source at one phi step and mosaic domain) predicts its own spots; the mask is their
  union. No spot is predicted where |s| > 1/dmin at the pixel, the kernel's dmin rule.

The width rule is GAUSS's under the HKL metric with fudge 1 and no N_cells_def; other spot shapes
are refused rather than masked with the wrong width. Thick-sensor layers are not sampled: a spot is
placed where the beam enters the sensor's front face.

The mask is a plain boolean tensor, built once from the geometry it is given and carrying no
gradient, so it stays fixed while the geometry refines. Write it into DetectorConfig.mask_array
(ExperimentModel.set_mask_arrays), or take a strided pixel set from it (strided_pixels).
"""
from typing import Iterator, Tuple

import torch

from .config import CrystalShape, SpotMetric

SPOT_SD = 0.397
"""A GAUSS spot's sd in h under the HKL metric is SPOT_SD / N (plan section 5, V1)."""

# Each pixel considers only the lattice point nearest its center hkl. The pixel holding a spot has its center
# within half the pixel's span of the spot along the detector, and the spot within n_sd sds of the lattice point
# off it, so in each Miller index i that lattice point is the nearest integer while
# span_i / 2 + n_sd * SPOT_SD / N_i < 1/2. Past that the mask is refused rather than mis-made (the audit of
# 2026-11-06 measured 7-12% of the photons outside the mask at a span of about 1.2, N 20).

_MARGIN = 0.75
"""Largest |delta*| (pixels, either axis) at which a pixel may still claim a spot: above the 1/2 of an
exact linearization, so that a spot on a pixel edge is never left without a center."""


def _check_spot_width(crystal) -> None:
    config = crystal.config
    if (config.shape is not CrystalShape.GAUSS or getattr(config, "spot_metric", None) is not SpotMetric.HKL
            or float(config.fudge) != 1.0 or any(float(v) != 0.0 for v in config.N_cells_def)):
        raise ValueError("spot masks know the GAUSS spot width under SpotMetric.HKL with fudge 1 and no N_cells_def; "
                         f"got shape {config.shape}, spot_metric {getattr(config, 'spot_metric', None)}, "
                         f"fudge {config.fudge}, N_cells_def {config.N_cells_def}")


def _directions(simulator, offsets: torch.Tensor) -> torch.Tensor:
    """Unit vectors from the sample to each pixel moved by offsets (k, 2) in pixels (slow, fast): (k, S, F, 3)."""
    simulator._ensure_pixel_coords_current()
    detector = simulator.detector
    center = simulator._cached_pixel_coords_meters.detach()
    pixel = torch.as_tensor(detector.pixel_size).detach().to(center)
    s_axis = torch.as_tensor(detector.sdet_vec).detach().to(center)
    f_axis = torch.as_tensor(detector.fdet_vec).detach().to(center)
    moved = center + (offsets[:, 0, None, None, None] * s_axis + offsets[:, 1, None, None, None] * f_axis) * pixel
    if simulator._curved_detector:
        moved = detector.apply_curved_mapping(moved)
    return moved / moved.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def _views(simulator) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """For each source, phi step and mosaic domain: hkl at the pixel centers (S, F, 3), its Jacobian in the
    in-plane offset (S, F, 3, 2; per pixel, a central difference over the pixel's width), and |s| at the
    centers (S, F) in 1/A. As the kernel computes them: s = (d - s0) / lambda, h = a . s."""
    half = 0.5
    offsets = torch.tensor([[0.0, 0.0], [half, 0.0], [-half, 0.0], [0.0, half], [0.0, -half]])
    if simulator._source_directions is not None:
        incident = -simulator._source_directions.detach()
        wavelength_A = simulator._source_wavelengths_A.detach()
    else:
        incident = simulator.incident_beam_direction.detach().reshape(1, 3)
        wavelength_A = torch.as_tensor(simulator.wavelength).detach().reshape(1)
    d = _directions(simulator, offsets.to(simulator._cached_pixel_coords_meters))
    incident, wavelength_A = incident.to(d), wavelength_A.to(d)
    (rot_a, rot_b, rot_c), _ = simulator.crystal.get_rotated_real_vectors(simulator.crystal.config)
    axes = torch.stack([rot_a, rot_b, rot_c], dim=-2).detach().to(d).reshape(-1, 3, 3)    # (n_orient, 3, 3), A
    for source in range(len(incident)):
        s = (d - incident[source]) / wavelength_A[source]                                 # (5, S, F, 3)
        s_norm = s[0].norm(dim=-1)
        for axis in axes:
            hkl = s @ axis.T
            jacobian = torch.stack([(hkl[1] - hkl[2]) / (2 * half), (hkl[3] - hkl[4]) / (2 * half)], dim=-1)
            yield hkl[0], jacobian, s_norm


def predicted_spot_mask(simulator, radius_px: float, n_sd: float = 3.0) -> torch.Tensor:
    """(S, F) bool, True where a pixel is kept: False within radius_px pixels (distance < radius_px) of a
    predicted spot center (module docstring)."""
    _check_spot_width(simulator.crystal)
    with torch.no_grad():
        N = torch.stack([torch.as_tensor(simulator.crystal.N_cells_a), torch.as_tensor(simulator.crystal.N_cells_b),
                         torch.as_tensor(simulator.crystal.N_cells_c)]).detach()
        dmin = simulator.beam_config.dmin
        centers = None
        for hkl, jacobian, s_norm in _views(simulator):
            if centers is None:
                S, F, _ = hkl.shape
                centers = torch.zeros(S * F, dtype=torch.bool, device=hkl.device)
                N = N.to(hkl)
            hkl, jacobian = hkl.reshape(-1, 3), jacobian.reshape(-1, 3, 2)
            span = jacobian.abs().sum(-1).amax(0)                                          # (3,) per Miller index
            if bool((span / 2 + n_sd * SPOT_SD / N >= 0.5).any()):
                raise ValueError(f"a pixel spans up to {[round(float(v), 2) for v in span]} in the Miller indices, too "
                                 f"much for spots {n_sd} sds wide at N {[float(v) for v in N]} to be placed from the "
                                 "lattice point nearest each pixel center; use finer pixels or a smaller cell")
            g = torch.round(hkl)
            # In the metric's coordinates, x = N * (g - hkl_0) and A = N * J: delta* = argmin |x - A delta|.
            x = (g - hkl) * N
            A = jacobian * N[:, None]
            normal = A.transpose(-1, -2) @ A                                              # (n, 2, 2)
            delta = torch.linalg.solve(normal, (A.transpose(-1, -2) @ x[..., None]))[..., 0]
            residual = x - (A @ delta[..., None])[..., 0]
            inside = (residual ** 2).sum(-1) < (n_sd * SPOT_SD) ** 2
            reach = delta.abs().amax(-1)
            inside &= reach <= _MARGIN
            if dmin > 0:
                inside &= s_norm.reshape(-1) <= 1.0 / dmin
            pixel = torch.nonzero(inside).flatten()
            if len(pixel):
                _, spot = torch.unique(g[pixel].long(), dim=0, return_inverse=True)
                # One center per spot: the smallest |delta*|, then the lowest pixel index (both as floats, so the
                # reductions run on every backend; pixel indices stay exact in float32 below 2^24).
                score = reach[pixel]
                best = torch.full((int(spot.max()) + 1,), float("inf"), dtype=score.dtype, device=score.device)
                best = best.scatter_reduce(0, spot, score, reduce="amin")
                index = pixel.to(score.dtype)
                candidate = torch.where(score == best[spot], index, torch.full_like(index, float("inf")))
                first = torch.full_like(best, float("inf")).scatter_reduce(0, spot, candidate, reduce="amin")
                centers[first.long()] = True
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
