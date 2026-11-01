"""
Auto-selection logic for count/range/step parameters.

This module resolves missing combinations of count/range/step for divergence, dispersion,
and thickness sampling the way nanoBragg.c does, and generates the sources from them.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import torch
import math


@dataclass
class SamplingParams:
    """Resolved sampling parameters after auto-selection."""
    count: int
    range: float
    step: float


def auto_select_sampling(
    count: Optional[int] = None,
    range_val: Optional[float] = None,
    step: Optional[float] = None,
    default_range: float = 1.0,
    parameter_type: str = "angle",
    unset_range: float = -1.0,
) -> SamplingParams:
    """
    Resolve count, range and step exactly as nanoBragg.c does.

    Divergence is nanoBragg.c:1431-1529, dispersion 1532-1580, thickness 1583-1631, each with
    the sanity check at 1676-1695:

    - count <= 0:
        - range < 0, step <= 0: nothing to do: count 1, range 0, step 0
        - range < 0, step > 0: range = step, count = 2
        - range >= 0, step <= 0: step = range, count = 2 (for thickness, step = range/2)
        - range >= 0, step > 0: count = ceil(range/step)
    - count > 0:
        - range < 0, step <= 0: range = default_range, step = range/count
        - range < 0, step > 0: range = step, count = 2
        - range >= 0, step <= 0: count is at least 2, step = range/(count-1)
        - range >= 0, step > 0: all three stand as given
    - then a range, step or count that is not positive means no spread: 1, 0, 0

    Args:
        count: Number of steps (None or int)
        range_val: Total range to cover (None or float)
        step: Step size (None or float)
        default_range: Range when only a count is provided (1.0 rad for angles,
                      0.5e-6 m for thickness)
        parameter_type: "angle" or "thickness"
        unset_range: The range C starts from: -1 for divergence, 0 for dispersion and thickness

    Returns:
        SamplingParams with resolved count, range, and step values
    """
    count = count if count is not None else -1
    range_val = range_val if range_val is not None else unset_range
    step = step if step is not None else -1.0

    if count <= 0:
        if range_val < 0.0:
            if step <= 0.0:
                count, range_val, step = 1, 0.0, 0.0
            else:
                range_val, count = step, 2
        elif step <= 0.0:
            step = range_val / 2.0 if parameter_type == "thickness" else range_val
            count = 2
        else:
            count = math.ceil(range_val / step)
    elif range_val < 0.0:
        if step <= 0.0:
            range_val = default_range
            step = range_val / count
        else:
            range_val, count = step, 2
    elif step <= 0.0:
        if count <= 1:
            count = 2
        step = range_val / (count - 1)

    if range_val <= 0.0 or step <= 0.0 or count <= 0:
        return SamplingParams(count=1, range=0.0, step=0.0)
    return SamplingParams(count=count, range=range_val, step=step)


def auto_select_divergence(
    hdivsteps: Optional[int] = None,
    hdivrange: Optional[float] = None,
    hdivstep: Optional[float] = None,
    vdivsteps: Optional[int] = None,
    vdivrange: Optional[float] = None,
    vdivstep: Optional[float] = None,
) -> Tuple[SamplingParams, SamplingParams]:
    """
    Auto-select horizontal and vertical divergence parameters.

    Args:
        hdivsteps: Horizontal divergence step count
        hdivrange: Horizontal divergence range in radians
        hdivstep: Horizontal divergence step size in radians
        vdivsteps: Vertical divergence step count
        vdivrange: Vertical divergence range in radians
        vdivstep: Vertical divergence step size in radians

    Returns:
        Tuple of (horizontal_params, vertical_params)
    """
    h_params = auto_select_sampling(
        count=hdivsteps,
        range_val=hdivrange,
        step=hdivstep,
        default_range=1.0,  # 1.0 rad for angles
        parameter_type="angle"
    )

    v_params = auto_select_sampling(
        count=vdivsteps,
        range_val=vdivrange,
        step=vdivstep,
        default_range=1.0,  # 1.0 rad for angles
        parameter_type="angle"
    )

    return h_params, v_params


def auto_select_dispersion(
    dispsteps: Optional[int] = None,
    dispersion: Optional[float] = None,
    dispstep: Optional[float] = None,
) -> SamplingParams:
    """
    Auto-select spectral dispersion parameters.

    Note: dispersion is a fraction (0-1), not a range in the same sense as divergence.
    The actual wavelength range is λ0·(1 ± dispersion/2).

    Args:
        dispsteps: Number of wavelength steps
        dispersion: Fractional dispersion (0-1)
        dispstep: Step size in fractional units

    Returns:
        SamplingParams for spectral dispersion
    """
    return auto_select_sampling(
        count=dispsteps,
        range_val=dispersion,
        step=dispstep,
        default_range=1.0,  # nanoBragg.c:1563, reached only by a negative -dispersion
        parameter_type="dispersion",
        unset_range=0.0,  # so -dispsteps alone changes nothing
    )


def auto_select_thickness(
    thicksteps: Optional[int] = None,
    detector_thick: Optional[float] = None,
    thickstep: Optional[float] = None,
) -> SamplingParams:
    """
    Auto-select detector thickness sampling parameters.

    Args:
        thicksteps: Number of thickness layers
        detector_thick: Total detector thickness in meters
        thickstep: Thickness step size in meters

    Returns:
        SamplingParams for thickness sampling
    """
    return auto_select_sampling(
        count=thicksteps,
        range_val=detector_thick,
        step=thickstep,
        default_range=0.5e-6,  # nanoBragg.c:1614, reached only by a negative thickness
        parameter_type="thickness",
        unset_range=0.0,
    )


def generate_sources_from_divergence_dispersion(
    hdiv_params: SamplingParams,
    vdiv_params: SamplingParams,
    disp_params: SamplingParams,
    central_wavelength_m: float,
    source_distance_m: float = 10.0,
    beam_direction: Optional[torch.Tensor] = None,
    polarization_axis: Optional[torch.Tensor] = None,
    round_div: bool = True,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate source arrays from divergence and dispersion parameters, as nanoBragg.c:2352-2415.

    - Horizontal/vertical divergence grids with optional elliptical trimming
    - Each divergence point starts from vector −source_distance·b
    - Rotated about polarization axis by vertical angle and vertical axis by horizontal angle
    - Wavelengths λ0·(1 + dispstep·tic − dispersion/2)

    Args:
        hdiv_params: Horizontal divergence sampling parameters
        vdiv_params: Vertical divergence sampling parameters
        disp_params: Dispersion sampling parameters (range is fraction 0-1)
        central_wavelength_m: Central wavelength λ0 in meters
        source_distance_m: Distance from source to sample in meters
        beam_direction: Unit beam direction vector (default [0,0,1])
        polarization_axis: Polarization axis for vertical rotation (default [0,1,0])
        round_div: If True, apply elliptical trimming to divergence grid

    Returns:
        Tuple of:
        - directions: (N, 3) tensor of unit direction vectors
        - weights: (N,) tensor of weights (all equal per spec)
        - wavelengths: (N,) tensor of wavelengths in meters
    """
    # In double, as C has it, whatever dtype the caller wants back.
    if beam_direction is None:
        beam_direction = torch.tensor([1.0, 0.0, 0.0])  # MOSFLM default
    if polarization_axis is None:
        polarization_axis = torch.tensor([0.0, 0.0, 1.0])  # Default vertical
    beam_direction = beam_direction.to(torch.float64)
    polarization_axis = polarization_axis.to(torch.float64)

    # Divergence angles: hdiv = hdivstep*tic - hdivrange/2 (nanoBragg.c:2356-2357), also for
    # a single step, so that count 1 with a nonzero range sits off axis.
    # Use manual arithmetic to avoid torch.linspace gradient issues
    hdiv_angles = torch.arange(hdiv_params.count, dtype=torch.float64) * hdiv_params.step - hdiv_params.range / 2
    vdiv_angles = torch.arange(vdiv_params.count, dtype=torch.float64) * vdiv_params.step - vdiv_params.range / 2

    # Create 2D meshgrid of divergence angles: (H, V)
    h_grid, v_grid = torch.meshgrid(hdiv_angles, vdiv_angles, indexing='ij')
    # Flatten to 1D arrays
    h_flat = h_grid.reshape(-1)
    v_flat = v_grid.reshape(-1)

    # Round off the corners exactly as nanoBragg.c:2359-2361 does: the 1.1 and the half-step
    # correction for an even count are C's. A zero range divides to NaN, which keeps the point.
    if round_div:
        h_range, h_step = hdiv_params.range, hdiv_params.step
        v_range, v_step = vdiv_params.range, vdiv_params.step
        test = ((h_flat.double()**2 - h_step*h_step/4.0*(1 - hdiv_params.count % 2)) / h_range / h_range
                + (v_flat.double()**2 - v_step*v_step/4.0*(1 - vdiv_params.count % 2)) / v_range / v_range)
        inside_mask = ~(test*4.0 > 1.1)
        h_flat = h_flat[inside_mask]
        v_flat = v_flat[inside_mask]

    n_div = h_flat.shape[0]

    # lambda = lambda0*(1 + dispstep*tic - dispersion/2) (nanoBragg.c:2405)
    # Use manual arithmetic for gradient preservation
    wavelengths = central_wavelength_m * (
        1.0 + torch.arange(disp_params.count, dtype=torch.float64) * disp_params.step - disp_params.range / 2
    )

    n_wave = wavelengths.shape[0]

    # Broadcast divergence and wavelengths: each divergence point gets all wavelengths
    # Shape: (n_div, n_wave)
    h_angles_expanded = h_flat.unsqueeze(1).expand(n_div, n_wave).reshape(-1)  # (n_div * n_wave,)
    v_angles_expanded = v_flat.unsqueeze(1).expand(n_div, n_wave).reshape(-1)  # (n_div * n_wave,)
    wavelengths_expanded = wavelengths.unsqueeze(0).expand(n_div, n_wave).reshape(-1)  # (n_div * n_wave,)

    n_sources = h_angles_expanded.shape[0]

    # Compute vertical axis (perpendicular to beam and polarization) once
    vertical_axis = torch.linalg.cross(beam_direction, polarization_axis)
    vertical_axis = vertical_axis / torch.linalg.norm(vertical_axis)

    # Start with -beam_direction for all sources: (n_sources, 3)
    directions = -beam_direction.unsqueeze(0).expand(n_sources, 3).clone()

    # Batch apply vertical rotation (about polarization axis) using Rodrigues' formula
    # Only apply where |v_angle| > 1e-10
    v_mask = torch.abs(v_angles_expanded) > 1e-10
    if v_mask.any():
        k = polarization_axis.unsqueeze(0)  # (1, 3)
        theta = v_angles_expanded[v_mask].unsqueeze(1)  # (n_active, 1)
        dirs_to_rotate = directions[v_mask]  # (n_active, 3)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        k_dot_d = (k * dirs_to_rotate).sum(dim=1, keepdim=True)  # (n_active, 1)

        # Rodrigues: v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
        cross_term = torch.linalg.cross(k.expand_as(dirs_to_rotate), dirs_to_rotate)
        directions[v_mask] = (
            dirs_to_rotate * cos_theta +
            cross_term * sin_theta +
            k.expand_as(dirs_to_rotate) * k_dot_d * (1 - cos_theta)
        )

    # Batch apply horizontal rotation (about vertical axis)
    h_mask = torch.abs(h_angles_expanded) > 1e-10
    if h_mask.any():
        k = vertical_axis.unsqueeze(0)  # (1, 3)
        theta = h_angles_expanded[h_mask].unsqueeze(1)  # (n_active, 1)
        dirs_to_rotate = directions[h_mask]  # (n_active, 3)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        k_dot_d = (k * dirs_to_rotate).sum(dim=1, keepdim=True)  # (n_active, 1)

        cross_term = torch.linalg.cross(k.expand_as(dirs_to_rotate), dirs_to_rotate)
        directions[h_mask] = (
            dirs_to_rotate * cos_theta +
            cross_term * sin_theta +
            k.expand_as(dirs_to_rotate) * k_dot_d * (1 - cos_theta)
        )

    # Batch normalize all directions
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)

    # Create weights (all equal per spec)
    weights = torch.ones(n_sources, dtype=dtype)

    # C divides by zero steps here and writes a NaN image (-divsteps 1 rounds its one point
    # away). Keep the beam, rather than copying that.
    if n_sources == 0:
        directions = -beam_direction.unsqueeze(0)
        weights = torch.tensor([1.0], dtype=dtype)
        wavelengths_expanded = torch.tensor([central_wavelength_m], dtype=torch.float64)

    return directions.to(dtype), weights, wavelengths_expanded.to(dtype)
