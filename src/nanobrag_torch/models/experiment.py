"""
Stage-A parameterized experiment model.

This module implements thin nn.Module wrappers around the existing config-driven
Crystal, Detector, and Beam models.  It exposes optimizer-friendly raw
parameters while delegating all physics and geometry to the core components.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import torch
from torch import nn

from ..config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from ..models.crystal import Crystal
from ..models.detector import Detector
from ..utils.mosaic import diffbragg_umats
from ..utils.structure_factors import StructureFactors
from ..utils.tensor_utils import as_tensor_preserving_grad
from ..utils.unit_cell import (
    B_matrix, FREE_VARIABLES, check_cell, constrained_cell, normalize_crystal_system,
)


_PARAM_INIT_MODES = ("trainable", "stage_a", "frozen")


def _param_factory(param_init: str):
    """Every Stage-A delta is an nn.Parameter; "frozen" only turns its gradient off.

    One code path whatever the mode, so a frozen model can be unfrozen later with
    ExperimentModel.refine(). "stage_a" is the old name for "trainable".
    """
    if param_init not in _PARAM_INIT_MODES:
        raise ValueError(f"param_init must be one of {_PARAM_INIT_MODES}, got {param_init!r}")
    requires_grad = param_init != "frozen"
    return lambda x: nn.Parameter(x, requires_grad=requires_grad)


class CrystalStageAParams(nn.Module):
    """Stage-A learnable parameterization for crystal geometry.

    ``crystal_system`` constrains the cell (see utils/unit_cell.py): only its free
    variables move, tied lengths follow the free one and fixed angles stay put. The
    parameters of tied quantities stay frozen and are listed in ``unrefinable``.

    When the base config carries an orientation matrix (``mosflm_*_star``, as the
    dxtbx/cctbx path and ``-mat`` produce) Crystal ignores the cell parameters, so the
    orientation is held as U = A0·B(cell0)⁻¹ and every step rebuilds A = U·B(cell),
    B being cctbx's. That is diffBragg's unit-cell derivative.

    Mosaic spread η (``mosaic_spread_deg`` with ``mosaic_domains`` > 1) is modelled
    with diffBragg's deterministic domain ensemble (utils/mosaic.py), which is
    smooth in η, not nanoBragg.c's random blocks. ``anisotropic_mosaicity`` gives
    η_a, η_b, η_c along the crystal's a, b, c axes as they are at construction
    (diffBragg's eta_abc refinement, transform_eta=True); otherwise one η.
    """

    # cell variable -> its delta parameter
    _CELL_PARAMS = {
        "a": "delta_log_a", "b": "delta_log_b", "c": "delta_log_c",
        "alpha": "delta_alpha_raw", "beta": "delta_beta_raw", "gamma": "delta_gamma_raw",
    }

    def __init__(
        self,
        base_config: CrystalConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        param_init: str = "trainable",
        max_angle_delta_deg: float = 10.0,
        max_misset_delta_deg: float = 10.0,
        crystal_system: str = "triclinic",
        anisotropic_mosaicity: bool = False,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.dtype = dtype
        self.max_angle_delta_deg = float(max_angle_delta_deg)
        self.max_misset_delta_deg = float(max_misset_delta_deg)

        # Store an immutable copy of the base config for defaults/non-learnable fields.
        # Use dataclasses.replace to avoid sharing accidental mutable state.
        self._base_config = replace(base_config)

        # Base cell parameters as buffers (Å, degrees).
        self.register_buffer(
            "base_cell_a",
            torch.as_tensor(base_config.cell_a, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_cell_b",
            torch.as_tensor(base_config.cell_b, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_cell_c",
            torch.as_tensor(base_config.cell_c, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_cell_alpha",
            torch.as_tensor(
                base_config.cell_alpha, device=self.device, dtype=self.dtype
            ),
        )
        self.register_buffer(
            "base_cell_beta",
            torch.as_tensor(
                base_config.cell_beta, device=self.device, dtype=self.dtype
            ),
        )
        self.register_buffer(
            "base_cell_gamma",
            torch.as_tensor(
                base_config.cell_gamma, device=self.device, dtype=self.dtype
            ),
        )

        # Base misset angles (degrees).
        m0x, m0y, m0z = base_config.misset_deg
        self.register_buffer(
            "base_misset_x",
            torch.as_tensor(m0x, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_misset_y",
            torch.as_tensor(m0y, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_misset_z",
            torch.as_tensor(m0z, device=self.device, dtype=self.dtype),
        )

        param_cls = _param_factory(param_init)

        # Log-length deltas (unconstrained, start at zero).
        self.delta_log_a = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        self.delta_log_b = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        self.delta_log_c = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )

        # Bounded angle deltas (raw).
        self.delta_alpha_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        self.delta_beta_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        self.delta_gamma_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )

        # Bounded misset deltas (raw 3-vector).
        self.delta_misset_raw = param_cls(
            torch.zeros((3,), device=self.device, dtype=self.dtype)
        )

        # Mosaic block size Na, Nb, Nc: log deltas (continuous, positive).
        self.register_buffer(
            "base_N_cells",
            torch.as_tensor(
                [float(n) for n in base_config.N_cells], device=self.device, dtype=self.dtype
            ),
        )
        self.delta_log_N_cells = param_cls(
            torch.zeros((3,), device=self.device, dtype=self.dtype)
        )
        # Parameters that must not move, with the reason; refine() refuses them.
        self.unrefinable = {}

        # SQUARE's sincg(pi*h, N) = sin(N*pi*h)/sin(pi*h) only has a finite limit at
        # integer h for integer N: at N = 5.3 the image sum is 157x and the peak 4000x
        # the N = 5 values, and d(sum)/dN changes sign between 5.0 and 5.001. Continuous
        # Ncells is only meaningful for the smooth shapes (GAUSS, ROUND, TOPHAT), which
        # is also what diffBragg refines.
        self.ncells_refinable = base_config.shape != CrystalShape.SQUARE
        if not self.ncells_refinable:
            self.unrefinable["delta_log_N_cells"] = (
                "Ncells is not refinable with shape=SQUARE: sin(N*pi*h)/sin(pi*h) "
                "diverges next to every Bragg peak for non-integer N. Use "
                "CrystalShape.GAUSS (as diffBragg does), ROUND or TOPHAT."
            )

        self.crystal_system = normalize_crystal_system(crystal_system)
        base_cell = tuple(float(getattr(base_config, f"cell_{n}"))
                          for n in ("a", "b", "c", "alpha", "beta", "gamma"))
        check_cell(self.crystal_system, base_cell)
        free = FREE_VARIABLES[self.crystal_system]
        for var, pname in self._CELL_PARAMS.items():
            if var not in free:
                self.unrefinable[pname] = (
                    f"{var} is not a free variable of a {self.crystal_system} cell "
                    f"(free: {', '.join(free)})"
                )
        # Orientation held as U when the config gives an orientation matrix.
        self.oriented_by_matrix = all(
            getattr(base_config, f"mosflm_{x}_star", None) is not None for x in "abc"
        )
        if self.oriented_by_matrix:
            A0 = torch.stack([
                torch.as_tensor(getattr(base_config, f"mosflm_{x}_star"), device=self.device,
                                dtype=torch.float64).detach()
                for x in "abc"
            ], dim=1)
            B0 = B_matrix(*base_cell, device=self.device)
            U = A0 @ torch.linalg.inv(B0)
            err = (U @ U.T - torch.eye(3, dtype=U.dtype, device=self.device)).abs().max()
            if err > 1e-6:
                raise ValueError(
                    f"the cell {base_cell} does not match the orientation matrix "
                    f"(U = A·B⁻¹ is off orthogonal by {float(err):.2g})"
                )
            self.register_buffer("U", U.to(self.dtype))

        # Mosaic spread: log offsets from the config's eta (one, or three if anisotropic).
        base_eta = float(base_config.mosaic_spread_deg)
        self.mosaic_domains = int(base_config.mosaic_domains)
        self.models_mosaicity = base_eta > 0 and self.mosaic_domains > 1
        if self.models_mosaicity and self.mosaic_domains % 2:
            raise ValueError(
                f"mosaic_domains must be even for diffBragg's mosaic ensemble (± rotation pairs), "
                f"got {self.mosaic_domains}"
            )
        self.anisotropic_mosaicity = bool(anisotropic_mosaicity)
        self.register_buffer(
            "base_eta", torch.as_tensor(base_eta, device=self.device, dtype=self.dtype)
        )
        self.delta_log_eta = param_cls(
            torch.zeros((3 if self.anisotropic_mosaicity else 1,), device=self.device, dtype=self.dtype)
        )
        if self.models_mosaicity and self.anisotropic_mosaicity:
            # unit real-space a, b, c (columns) of the starting crystal
            if self.oriented_by_matrix:
                real = torch.linalg.inv(A0)  # rows are a, b, c
            else:
                start = Crystal(replace(base_config), device=self.device, dtype=torch.float64)
                real = torch.stack([start.a, start.b, start.c]).detach()
            self.register_buffer("eta_axes", (real / real.norm(dim=1, keepdim=True)).T.to(self.dtype))
        else:
            self.eta_axes = None
        if not self.models_mosaicity:
            self.unrefinable["delta_log_eta"] = (
                "eta is not refinable without mosaic spread: set mosaic_spread_deg > 0 and "
                "an even mosaic_domains > 1"
            )

        for pname in self.unrefinable:
            getattr(self, pname).requires_grad_(False)

    def eta(self) -> torch.Tensor:
        """Mosaic spread in degrees: shape (1,), or (3,) for anisotropic mosaicity."""
        return self.base_eta * torch.exp(self.delta_log_eta)

    def mosaic_umats(self) -> Optional[torch.Tensor]:
        """(mosaic_domains, 3, 3) mosaic rotations for the current eta, or None without mosaicity."""
        if not self.models_mosaicity:
            return None
        return diffbragg_umats(self.eta(), self.mosaic_domains, crystal_axes=self.eta_axes).to(self.dtype)

    def build_config(self) -> CrystalConfig:
        """Return a CrystalConfig with physical values derived from raw parameters."""
        # Free variables only: log-length deltas keep lengths positive, tanh windows
        # bound the angle deltas. Tied and fixed quantities come from the crystal system.
        max_angle = self.max_angle_delta_deg
        free = {}
        for var in FREE_VARIABLES[self.crystal_system]:
            base = getattr(self, f"base_cell_{var}")
            delta = getattr(self, self._CELL_PARAMS[var])
            if var in ("a", "b", "c"):
                free[var] = base * torch.exp(delta)
            else:
                free[var] = base + max_angle * torch.tanh(delta)
        fixed = {var: getattr(self, f"base_cell_{var}") for var in self._CELL_PARAMS}
        cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma = (
            torch.as_tensor(x, device=self.device, dtype=self.dtype)
            for x in constrained_cell(self.crystal_system, free, fixed)
        )

        # Bounded misset deltas.
        max_misset = self.max_misset_delta_deg
        delta_misset = max_misset * torch.tanh(self.delta_misset_raw)
        misset_x = self.base_misset_x + delta_misset[0]
        misset_y = self.base_misset_y + delta_misset[1]
        misset_z = self.base_misset_z + delta_misset[2]

        # Create a fresh config clone and overwrite the learnable fields.
        cfg = replace(self._base_config)
        cfg.cell_a = cell_a
        cfg.cell_b = cell_b
        cfg.cell_c = cell_c
        cfg.cell_alpha = cell_alpha
        cfg.cell_beta = cell_beta
        cfg.cell_gamma = cell_gamma
        cfg.misset_deg = (misset_x, misset_y, misset_z)
        if self.oriented_by_matrix:
            A = self.U @ B_matrix(cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma)
            cfg.mosflm_a_star, cfg.mosflm_b_star, cfg.mosflm_c_star = A[:, 0], A[:, 1], A[:, 2]
        n_cells = self.base_N_cells * torch.exp(self.delta_log_N_cells)
        cfg.N_cells = (n_cells[0], n_cells[1], n_cells[2])
        return cfg


class DetectorStageAParams(nn.Module):
    """Stage-A learnable parameterization for detector geometry.

    Distance refines whichever field drives the geometry: ``close_distance_mm``
    when it is set (SAMPLE pivot, ``-close_distance``), else ``distance_mm``.

    A detector given by its origin (dxtbx panels through compat.cctbx: CUSTOM
    convention, SAMPLE pivot, close distance) instead refines ``detz_shift``,
    diffBragg's parameter: a shift of the origin along lab z in mm. For such a
    detector both would move the same thing, so ``distance`` is listed in
    ``unrefinable`` there, and ``detz_shift`` is unrefinable everywhere else.
    """

    def __init__(
        self,
        base_config: DetectorConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        param_init: str = "trainable",
        max_tilt_delta_deg: float = 5.0,
        beam_shift_pixels: float = 5.0,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.dtype = dtype
        self.max_tilt_delta_deg = float(max_tilt_delta_deg)
        self.beam_shift_pixels = float(beam_shift_pixels)

        self._base_config = replace(base_config)

        # Base distance and angles.
        self.register_buffer(
            "base_distance_mm",
            torch.as_tensor(base_config.distance_mm, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_rotx_deg",
            torch.as_tensor(base_config.detector_rotx_deg, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_roty_deg",
            torch.as_tensor(base_config.detector_roty_deg, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_rotz_deg",
            torch.as_tensor(base_config.detector_rotz_deg, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_twotheta_deg",
            torch.as_tensor(base_config.detector_twotheta_deg, device=self.device, dtype=self.dtype),
        )

        # Base beam-center in mm (after DetectorConfig defaults).
        beam_s = base_config.beam_center_s
        beam_f = base_config.beam_center_f
        self.register_buffer(
            "base_beam_center_s_mm",
            torch.as_tensor(beam_s, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "base_beam_center_f_mm",
            torch.as_tensor(beam_f, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "pixel_size_mm",
            torch.as_tensor(base_config.pixel_size_mm, device=self.device, dtype=self.dtype),
        )

        param_cls = _param_factory(param_init)

        # Log-distance delta.
        self.delta_log_distance_mm = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        # Origin shift along lab z (mm), additive.
        self.delta_detz_shift_mm = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )

        self.unrefinable = {}
        self.close_distance_drives = base_config.close_distance_mm is not None
        if self.close_distance_drives:
            self.register_buffer(
                "base_close_distance_mm",
                torch.as_tensor(base_config.close_distance_mm, device=self.device, dtype=self.dtype),
            )
        self.origin_defined = (
            self.close_distance_drives
            and base_config.detector_convention == DetectorConvention.CUSTOM
            and base_config.detector_pivot == DetectorPivot.SAMPLE
        )
        if self.origin_defined:
            # pix0 = -beam_center_f·f - beam_center_s·s + close_distance·o, so a lab-z shift
            # of the origin moves those three by -dz·f_z, -dz·s_z and +dz·o_z.
            for name in ("fdet", "sdet", "odet"):
                vec = torch.as_tensor(getattr(base_config, f"custom_{name}_vector"),
                                      device=self.device, dtype=self.dtype)
                self.register_buffer(f"{name}_z", vec[2].clone())
            self.unrefinable["delta_log_distance_mm"] = (
                "this detector is given by its origin (dxtbx panel): refine 'detz_shift', "
                "diffBragg's origin shift along z, instead of 'distance'"
            )
        else:
            self.unrefinable["delta_detz_shift_mm"] = (
                "detz_shift moves a detector given by its origin (dxtbx panels: CUSTOM "
                "convention, SAMPLE pivot); refine 'distance' for this detector"
            )
        for pname in self.unrefinable:
            getattr(self, pname).requires_grad_(False)

        # Bounded tilt and twotheta deltas.
        self.delta_rotx_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        self.delta_roty_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        self.delta_rotz_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        self.delta_twotheta_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )

        # Beam-center deltas in pixel units (coarse centering).
        self.delta_beam_s_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )
        self.delta_beam_f_raw = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )

    def build_config(self) -> DetectorConfig:
        """Return a DetectorConfig with physical values derived from raw parameters."""
        # Distance (mm, positive).
        distance_mm = self.base_distance_mm * torch.exp(self.delta_log_distance_mm)

        # Bounded small-angle tilts.
        max_tilt = self.max_tilt_delta_deg
        rotx = self.base_rotx_deg + max_tilt * torch.tanh(self.delta_rotx_raw)
        roty = self.base_roty_deg + max_tilt * torch.tanh(self.delta_roty_raw)
        rotz = self.base_rotz_deg + max_tilt * torch.tanh(self.delta_rotz_raw)
        twotheta = self.base_twotheta_deg + max_tilt * torch.tanh(self.delta_twotheta_raw)

        # Beam-center deltas in mm via pixel-size scaling.
        max_shift_mm = self.beam_shift_pixels * self.pixel_size_mm
        beam_delta_s_mm = max_shift_mm * torch.tanh(self.delta_beam_s_raw)
        beam_delta_f_mm = max_shift_mm * torch.tanh(self.delta_beam_f_raw)

        beam_center_s = self.base_beam_center_s_mm + beam_delta_s_mm
        beam_center_f = self.base_beam_center_f_mm + beam_delta_f_mm

        cfg = replace(self._base_config)
        if self.origin_defined:
            dz = self.delta_detz_shift_mm
            cfg.close_distance_mm = self.base_close_distance_mm + dz * self.odet_z
            beam_center_f = beam_center_f - dz * self.fdet_z
            beam_center_s = beam_center_s - dz * self.sdet_z
        elif self.close_distance_drives:
            cfg.close_distance_mm = self.base_close_distance_mm * torch.exp(self.delta_log_distance_mm)
        else:
            cfg.distance_mm = distance_mm
        cfg.detector_rotx_deg = rotx
        cfg.detector_roty_deg = roty
        cfg.detector_rotz_deg = rotz
        cfg.detector_twotheta_deg = twotheta
        cfg.beam_center_s = beam_center_s
        cfg.beam_center_f = beam_center_f
        return cfg


class BeamStageAParams(nn.Module):
    """Stage-A learnable parameterization for beam fluence."""

    def __init__(
        self,
        base_config: BeamConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        param_init: str = "trainable",
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.dtype = dtype
        self._base_config = replace(base_config)

        self.register_buffer(
            "base_fluence",
            torch.as_tensor(base_config.fluence, device=self.device, dtype=self.dtype),
        )

        param_cls = _param_factory(param_init)

        self.delta_log_fluence = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )

        # Overall scale G (cctbx/diffBragg spot_scale; C has none): log delta.
        self.register_buffer(
            "base_spot_scale",
            torch.as_tensor(base_config.spot_scale, device=self.device, dtype=self.dtype),
        )
        self.delta_log_spot_scale = param_cls(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )

    def build_config(self) -> BeamConfig:
        """Return a BeamConfig with physical values derived from raw parameters."""
        fluence = self.base_fluence * torch.exp(self.delta_log_fluence)

        cfg = replace(self._base_config)
        cfg.fluence = fluence
        cfg.spot_scale = self.base_spot_scale * torch.exp(self.delta_log_spot_scale)
        return cfg


class StructureFactorStageAParams(nn.Module):
    """Stage-A learnable structure factors: one amplitude per unique reflection.

    F_i = F0_i·exp(delta_log_F_i) for the M reflections of a StructureFactors
    (utils/structure_factors.py); every symmetry mate in the P1 box takes its
    amplitude through a fixed gather, so the gradient of a mate lands on its
    unique reflection. diffBragg refines a per-ASU intensity scale s_i (I ∝ s_i·F0_i²),
    which is exp(2·delta_log_F_i) here. A reflection with F0 = 0 stays at 0.

    Without structure factors the parameter is empty and unrefinable, and the crystal
    keeps whatever F it was given (default_F, or a fixed grid via
    ExperimentModel.set_structure_factors).
    """

    def __init__(
        self,
        structure_factors=None,
        default_F: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        param_init: str = "trainable",
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.dtype = dtype
        self.structure_factors = structure_factors
        self.unrefinable = {}
        param_cls = _param_factory(param_init)

        if structure_factors is None:
            M = 0
            self.hkl_metadata = None
            self.unrefinable["delta_log_F"] = (
                "Fhkl is not refinable without structure factors: pass structure_factors= "
                "(a StructureFactors, see nanobrag_torch.compat.cctbx.asu_structure_factors)"
            )
        else:
            M = len(structure_factors)
            if structure_factors.default_F is not None:
                default_F = structure_factors.default_F
            index, self.hkl_metadata, constants = structure_factors.grid_index(float(default_F))
            self.register_buffer("grid_index", torch.as_tensor(index, device=self.device))
            self.register_buffer("constants", torch.as_tensor(constants, device=self.device, dtype=self.dtype))
        self.register_buffer(
            "base_F",
            torch.as_tensor(
                structure_factors.amplitudes if M else np.zeros(0), device=self.device, dtype=self.dtype
            ),
        )
        self.delta_log_F = param_cls(torch.zeros(M, device=self.device, dtype=self.dtype))

        for pname in self.unrefinable:
            getattr(self, pname).requires_grad_(False)

    @property
    def active(self) -> bool:
        return self.structure_factors is not None

    def amplitudes(self) -> torch.Tensor:
        """Current amplitudes of the M unique reflections."""
        return self.base_F * torch.exp(self.delta_log_F)

    def hkl_grid(self) -> torch.Tensor:
        """The dense P1 box [h-h_min][k-k_min][l-l_min] for Crystal.hkl_data."""
        return torch.cat([self.amplitudes(), self.constants])[self.grid_index]


class ExperimentModel(nn.Module):
    """
    High-level experiment composition model for Stage-A optimization.

    Owns parameter wrappers for crystal, detector, and beam, and delegates
    physics to the existing Simulator.

    Every refinable quantity is an nn.Parameter holding a zero offset from the
    config value, so a new model reproduces the config image exactly. All of them
    are trainable by default (param_init="trainable"); choose what to refine with
    refine()/freeze(), by group name (see PARAMETER_GROUPS) or full parameter name:

        model.freeze()                      # nothing trains
        model.refine("misset", "cell")      # then only orientation and cell
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-2)

    For images only, run under torch.no_grad() so no autograd graph is kept.

    ``crystal_system`` (default "triclinic", six free cell parameters) constrains
    the cell refinement; see nanobrag_torch.utils.unit_cell and ``unrefinable``.
    With mosaic spread, ``anisotropic_mosaicity`` refines eta_a, eta_b, eta_c
    instead of one eta (diffBragg's ensemble; see CrystalStageAParams).
    ``structure_factors`` (a StructureFactors) makes F refinable, one amplitude per
    unique reflection (see StructureFactorStageAParams); a fixed grid can be given
    instead with ``hkl_data``/``hkl_metadata`` or set_structure_factors().
    """

    # Group name -> parameter names, for refine()/freeze().
    PARAMETER_GROUPS = {
        "cell": (
            "crystal_params.delta_log_a", "crystal_params.delta_log_b",
            "crystal_params.delta_log_c", "crystal_params.delta_alpha_raw",
            "crystal_params.delta_beta_raw", "crystal_params.delta_gamma_raw",
        ),
        "misset": ("crystal_params.delta_misset_raw",),
        "distance": ("detector_params.delta_log_distance_mm",),
        "detz_shift": ("detector_params.delta_detz_shift_mm",),
        "detector_rotation": (
            "detector_params.delta_rotx_raw", "detector_params.delta_roty_raw",
            "detector_params.delta_rotz_raw",
        ),
        "twotheta": ("detector_params.delta_twotheta_raw",),
        "beam_center": (
            "detector_params.delta_beam_s_raw", "detector_params.delta_beam_f_raw",
        ),
        "fluence": ("beam_params.delta_log_fluence",),
        "scale": ("beam_params.delta_log_spot_scale",),
        "Ncells": ("crystal_params.delta_log_N_cells",),
        "eta": ("crystal_params.delta_log_eta",),
        "Fhkl": ("fhkl_params.delta_log_F",),
    }

    def __init__(
        self,
        crystal_config: CrystalConfig,
        detector_config: DetectorConfig,
        beam_config: BeamConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        param_init: str = "trainable",
        hkl_data: Optional[torch.Tensor] = None,
        hkl_metadata: Optional[dict] = None,
        crystal_system: str = "triclinic",
        anisotropic_mosaicity: bool = False,
        structure_factors: Optional[StructureFactors] = None,
    ) -> None:
        super().__init__()
        if structure_factors is not None and hkl_data is not None:
            raise ValueError("give structure_factors (refinable) or hkl_data (fixed), not both")
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.dtype = dtype

        # Stage-A parameter blocks.
        self.crystal_params = CrystalStageAParams(
            base_config=crystal_config,
            device=self.device,
            dtype=self.dtype,
            param_init=param_init,
            crystal_system=crystal_system,
            anisotropic_mosaicity=anisotropic_mosaicity,
        )
        self.detector_params = DetectorStageAParams(
            base_config=detector_config,
            device=self.device,
            dtype=self.dtype,
            param_init=param_init,
        )
        self.beam_params = BeamStageAParams(
            base_config=beam_config,
            device=self.device,
            dtype=self.dtype,
            param_init=param_init,
        )
        self.fhkl_params = StructureFactorStageAParams(
            structure_factors,
            default_F=crystal_config.default_F,
            device=self.device,
            dtype=self.dtype,
            param_init=param_init,
        )

        # Optional externally-provided structure factors (dense HKL grid).
        # Downstream projects (e.g. dbex) can populate these either via the
        # constructor or via set_structure_factors() below.
        self._external_hkl_data: Optional[torch.Tensor] = hkl_data
        self._external_hkl_metadata: Optional[dict] = hkl_metadata

        # Crystal, Detector and Simulator are built on the first forward() and then
        # kept: every later step writes the new parameter tensors into them instead
        # of rebuilding (and re-compiling) all three.
        self._simulator = None
        self._crystal: Optional[Crystal] = None
        self._detector: Optional[Detector] = None
        self._beam_config: Optional[BeamConfig] = None

    def _resolve(self, names) -> list:
        """Parameter names for these groups/parameters (all of them if none given)."""
        params = dict(self.named_parameters())
        if not names:
            return list(params)
        resolved = []
        for name in names:
            if name in self.PARAMETER_GROUPS:
                resolved.extend(self.PARAMETER_GROUPS[name])
            elif name in params:
                resolved.append(name)
            else:
                raise KeyError(
                    f"unknown parameter or group {name!r}; groups: "
                    f"{sorted(self.PARAMETER_GROUPS)}, parameters: {sorted(params)}"
                )
        return resolved

    @property
    def unrefinable(self) -> dict:
        """Parameter name -> why it cannot be refined (crystal shape, crystal system, detector kind,
        no structure factors)."""
        blocked = {}
        for block in ("crystal_params", "detector_params", "fhkl_params"):
            blocked.update({f"{block}.{n}": why for n, why in getattr(self, block).unrefinable.items()})
        return blocked

    def refine(self, *names: str) -> "ExperimentModel":
        """Turn gradients on for these groups/parameters (all of them if none given).

        Parameters in ``unrefinable`` stay frozen: Ncells under the SQUARE shape, and
        the cell parameters a crystal system ties or fixes. A group refines its
        refinable members; naming an unrefinable parameter, or a group with no
        refinable member, raises ValueError with the reason.
        """
        params = dict(self.named_parameters())
        blocked = self.unrefinable
        for name in names:
            members = self.PARAMETER_GROUPS.get(name, (name,))
            if all(m in blocked for m in members):
                raise ValueError(blocked[members[0]])
        for name in self._resolve(names):
            if name not in blocked:
                params[name].requires_grad_(True)
        return self

    def freeze(self, *names: str) -> "ExperimentModel":
        """Turn gradients off for these groups/parameters (all of them if none given).

        A frozen parameter keeps its current value, so freezing mid-refinement
        holds it where the optimizer left it.
        """
        params = dict(self.named_parameters())
        for name in self._resolve(names):
            params[name].requires_grad_(False)
        return self

    def trainable_parameters(self) -> list:
        """The parameters that currently require grad, for handing to an optimizer."""
        return [p for p in self.parameters() if p.requires_grad]

    def set_structure_factors(
        self, hkl_data: torch.Tensor, hkl_metadata: dict
    ) -> None:
        """
        Override structure factors with an external dense HKL grid.

        Args:
            hkl_data:
                Dense HKL grid as a tensor (e.g. shape [H, K, L]) on any
                device/dtype. The ExperimentModel will move it to its own
                device/dtype before use.
            hkl_metadata:
                Dictionary containing at least the integer bounds:
                'h_min', 'h_max', 'k_min', 'k_max', 'l_min', 'l_max'.

        This is the supported hook for downstream projects that build their
        own P1 grids and want ExperimentModel to reuse them, rather than
        loading HKL files internally via Crystal.load_hkl().
        """
        if self.fhkl_params.active:
            raise ValueError("this model refines its structure factors (structure_factors=); "
                             "a fixed grid would replace them")
        self._external_hkl_data = hkl_data
        self._external_hkl_metadata = hkl_metadata
        if self._crystal is not None:
            self._apply_structure_factors(self._crystal)

    def _apply_structure_factors(self, crystal: Crystal) -> None:
        if self.fhkl_params.active:
            crystal.hkl_data = self.fhkl_params.hkl_grid()
            crystal.hkl_metadata = dict(self.fhkl_params.hkl_metadata)
            return
        if self._external_hkl_data is None:
            return
        hkl_tensor = torch.as_tensor(self._external_hkl_data)
        crystal.hkl_data = hkl_tensor.to(device=self.device, dtype=self.dtype)
        crystal.hkl_metadata = (
            dict(self._external_hkl_metadata)
            if self._external_hkl_metadata is not None
            else None
        )

    def _apply_mosaicity(self, crystal: Crystal) -> None:
        umats = self.crystal_params.mosaic_umats()
        if umats is not None:
            crystal.mosaic_umats_override = umats
            crystal.config.mosaic_domains = int(umats.shape[0])

    # Fields the Stage-A blocks produce. Everything else in the configs is fixed at
    # construction.
    _CRYSTAL_CELL_FIELDS = (
        "cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma",
    )
    _DETECTOR_FIELDS = (
        "distance_mm", "close_distance_mm", "detector_rotx_deg", "detector_roty_deg", "detector_rotz_deg",
        "detector_twotheta_deg", "beam_center_s", "beam_center_f",
    )

    def _sync(self):
        """Return the persistent Simulator with the current parameter values in it.

        Crystal re-derives its vectors from its cell tensors and config.misset_deg on
        every access; Detector rebuilds its geometry when a config field changes
        (Detector._ensure_geometry_current, called by Simulator.run); Simulator takes
        fluence from its own tensor. So writing the new tensors in is enough.
        """
        crystal_cfg = self.crystal_params.build_config()
        detector_cfg = self.detector_params.build_config()
        beam_cfg = self.beam_params.build_config()

        if self._simulator is None:
            self._construct(crystal_cfg, detector_cfg, beam_cfg)
            return self._simulator

        crystal = self._crystal
        for name in self._CRYSTAL_CELL_FIELDS:
            value = getattr(crystal_cfg, name)
            setattr(crystal, name, torch.as_tensor(value, device=self.device, dtype=self.dtype))
            # The config copy feeds Simulator's auto-oversample choice.
            setattr(crystal.config, name, value)
        crystal.config.misset_deg = crystal_cfg.misset_deg
        if self.crystal_params.oriented_by_matrix:
            for x in "abc":
                name = f"mosflm_{x}_star"
                setattr(crystal.config, name, getattr(crystal_cfg, name))
        crystal.N_cells_a, crystal.N_cells_b, crystal.N_cells_c = (
            torch.as_tensor(n, device=self.device, dtype=self.dtype)
            for n in crystal_cfg.N_cells
        )
        # Simulator's auto-oversample choice reads the config copy.
        crystal.config.N_cells = crystal_cfg.N_cells
        self._apply_mosaicity(crystal)
        if self.fhkl_params.active:
            self._apply_structure_factors(crystal)

        for name in self._DETECTOR_FIELDS:
            setattr(self._detector.config, name, getattr(detector_cfg, name))

        self._beam_config.fluence = beam_cfg.fluence
        # Simulator.run() reads spot_scale from this config on every run.
        self._beam_config.spot_scale = beam_cfg.spot_scale
        self._simulator.fluence = as_tensor_preserving_grad(
            beam_cfg.fluence, device=self.device, dtype=self.dtype
        )
        return self._simulator

    def _construct(self, crystal_cfg, detector_cfg, beam_cfg) -> None:
        crystal = Crystal(
            config=crystal_cfg,
            beam_config=beam_cfg,
            device=self.device,
            dtype=self.dtype,
        )
        self._apply_structure_factors(crystal)
        self._apply_mosaicity(crystal)

        detector = Detector(
            config=detector_cfg,
            device=self.device,
            dtype=self.dtype,
        )

        # Import Simulator lazily here to avoid circular imports at module load
        # time (Simulator imports models.crystal / models.detector).
        from ..simulator import Simulator

        self._simulator = Simulator(
            crystal=crystal,
            detector=detector,
            beam_config=beam_cfg,
            device=self.device,
            dtype=self.dtype,
        )
        self._crystal, self._detector, self._beam_config = crystal, detector, beam_cfg

    def _build_simulator(self):
        """
        Return (simulator, crystal, detector, beam_config) holding the current
        parameter values. These are the persistent objects forward() runs, not
        fresh copies.

        Note: This is primarily intended for tests and tooling; callers should
        prefer ExperimentModel.forward() for normal use.
        """
        simulator = self._sync()
        return simulator, self._crystal, self._detector, self._beam_config

    def forward(
        self,
        pixel_batch_size: Optional[int] = None,
        oversample: Optional[int] = None,
        oversample_omega: Optional[bool] = None,
        oversample_polar: Optional[bool] = None,
        oversample_thick: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Run the diffraction simulation for the current parameter values.

        Returns:
            Float image tensor with shape (spixels, fpixels).
        """
        simulator = self._sync()

        return simulator.run(
            pixel_batch_size=pixel_batch_size,
            oversample=oversample,
            oversample_omega=oversample_omega,
            oversample_polar=oversample_polar,
            oversample_thick=oversample_thick,
        )
