"""Masks around predicted Bragg spots, for diffuse and joint fits (plan section 2.2).

Near a lattice point the diffuse core and the Bragg peak trade off, so diffuse and joint fits may
leave out the pixels around each spot. The spots are predicted from the geometry alone, not from an
image: each pixel centre's fractional hkl, the nearest integer hkl, and its distance in the HKL
metric, sum_i (N_i d_i)^2, in units of a GAUSS-HKL spot's sd, 0.397/N per axis. Every integer hkl
within n_sd sds of some pixel is a spot; its centre is the pixel nearest it. Orientations (phi steps,
mosaic domains) and sources each predict their own spots, and the mask is their union.

The mask is a plain boolean tensor, built once from the geometry it is given and carrying no
gradient, so it stays fixed while the geometry refines. Write it into DetectorConfig.mask_array, or
take a strided pixel set from it (strided_pixels).
"""
import torch

SPOT_SD = 0.397
"""A GAUSS spot's sd in h under the HKL metric is SPOT_SD / N (plan section 5, V1)."""


def _pixel_hkl(simulator) -> torch.Tensor:
    """Fractional hkl at each pixel centre, (n_sources * n_phi * n_mos, S, F, 3), as the kernel computes it:
    h = a . (d - s0) / lambda with d the unit vector to the pixel and s0 the incident direction."""
    simulator._ensure_pixel_coords_current()
    pixels = simulator._cached_pixel_coords_meters.detach()
    d = pixels / pixels.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    if simulator._source_directions is not None:
        incident = -simulator._source_directions.detach().to(d)
        wavelength_A = simulator._source_wavelengths_A.detach().to(d)
    else:
        incident = simulator.incident_beam_direction.detach().to(d).reshape(1, 3)
        wavelength_A = torch.as_tensor(simulator.wavelength, dtype=d.dtype, device=d.device).detach().reshape(1)
    s = (d[None] - incident[:, None, None]) / wavelength_A.reshape(-1, 1, 1, 1)          # (n_src, S, F, 3), 1/A
    (rot_a, rot_b, rot_c), _ = simulator.crystal.get_rotated_real_vectors(simulator.crystal.config)
    axes = torch.stack([rot_a, rot_b, rot_c], dim=-2).detach().to(d).reshape(-1, 3, 3)  # (n_orient, 3, 3), A
    hkl = torch.einsum("osfj,ncj->nosfc", s, axes)
    return hkl.reshape(-1, *hkl.shape[2:]), s.norm(dim=-1)


def predicted_spot_mask(simulator, radius_px: float, n_sd: float = 3.0) -> torch.Tensor:
    """(S, F) bool, True where a pixel is kept: False within radius_px pixels (distance < radius_px) of a
    predicted spot centre. Spots beyond the simulator's dmin are not predicted."""
    with torch.no_grad():
        hkl, s_norm = _pixel_hkl(simulator)
        n_views, S, F, _ = hkl.shape
        N = torch.stack([torch.as_tensor(simulator.crystal.N_cells_a), torch.as_tensor(simulator.crystal.N_cells_b),
                         torch.as_tensor(simulator.crystal.N_cells_c)]).to(hkl).detach()
        nearest = torch.round(hkl)
        dist2 = (((hkl - nearest) * N) ** 2).sum(-1)
        inside = dist2 < (n_sd * SPOT_SD) ** 2
        dmin = getattr(simulator.beam_config, "dmin", 0.0) or 0.0
        if dmin > 0:
            n_src = s_norm.shape[0]
            inside &= (s_norm.repeat_interleave(n_views // n_src, dim=0) <= 1.0 / dmin)
        centres = torch.zeros(S * F, dtype=torch.bool, device=hkl.device)
        view, pixel = torch.nonzero(inside.reshape(n_views, -1), as_tuple=True)
        if len(view):
            keys = torch.cat([view[:, None], nearest.reshape(n_views, -1, 3)[view, pixel].long()], dim=1)
            _, spot = torch.unique(keys, dim=0, return_inverse=True)
            d2 = dist2.reshape(n_views, -1)[view, pixel]
            best = torch.full((int(spot.max()) + 1,), float("inf"), dtype=d2.dtype, device=d2.device)
            best = best.scatter_reduce(0, spot, d2, reduce="amin")
            # Pixels equally near a spot (a beam on a pixel corner) tie: the lowest pixel index is its centre.
            nearest_pixel = torch.where(d2 == best[spot], pixel, torch.full_like(pixel, S * F))
            first = torch.full_like(best, S * F, dtype=pixel.dtype).scatter_reduce(0, spot, nearest_pixel, reduce="amin")
            centres[first] = True
        return _dilate(centres.reshape(S, F), radius_px)


def _dilate(centres: torch.Tensor, radius_px: float) -> torch.Tensor:
    """keep = no centre within distance < radius_px."""
    if radius_px <= 0 or not centres.any():
        return torch.ones_like(centres)
    reach = int(torch.ceil(torch.tensor(radius_px)).item())
    offsets = torch.arange(-reach, reach + 1, device=centres.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    disk = ((dy ** 2 + dx ** 2) < radius_px ** 2).to(torch.float32)
    hit = torch.nn.functional.conv2d(centres.to(torch.float32)[None, None], disk[None, None], padding=reach)[0, 0]
    return hit == 0


def strided_pixels(keep: torch.Tensor, stride: int) -> torch.Tensor:
    """(n, 2) long (slow, fast) of every stride-th kept pixel in row-major order, starting with the first:
    the pixels= format of Simulator.run."""
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride!r}")
    return torch.nonzero(keep)[::stride]
