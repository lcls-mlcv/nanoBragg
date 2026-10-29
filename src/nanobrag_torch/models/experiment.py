"""
Stage-A parameterized experiment model.

This module implements thin nn.Module wrappers around the existing config-driven
Crystal, Detector, and Beam models.  It exposes optimizer-friendly raw
parameters while delegating all physics and geometry to the core components.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
from torch import nn

from ..config import BeamConfig, CrystalConfig, DetectorConfig
from ..models.crystal import Crystal
from ..models.detector import Detector
from ..utils.tensor_utils import as_tensor_preserving_grad


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
    """Stage-A learnable parameterization for crystal geometry."""

    def __init__(
        self,
        base_config: CrystalConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        param_init: str = "trainable",
        max_angle_delta_deg: float = 10.0,
        max_misset_delta_deg: float = 10.0,
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

    def build_config(self) -> CrystalConfig:
        """Return a CrystalConfig with physical values derived from raw parameters."""
        # Exponentiate log-length deltas to maintain positivity.
        cell_a = self.base_cell_a * torch.exp(self.delta_log_a)
        cell_b = self.base_cell_b * torch.exp(self.delta_log_b)
        cell_c = self.base_cell_c * torch.exp(self.delta_log_c)

        # Bounded angle deltas using tanh window.
        max_angle = self.max_angle_delta_deg
        cell_alpha = self.base_cell_alpha + max_angle * torch.tanh(self.delta_alpha_raw)
        cell_beta = self.base_cell_beta + max_angle * torch.tanh(self.delta_beta_raw)
        cell_gamma = self.base_cell_gamma + max_angle * torch.tanh(self.delta_gamma_raw)

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
        return cfg


class DetectorStageAParams(nn.Module):
    """Stage-A learnable parameterization for detector geometry."""

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

    def build_config(self) -> BeamConfig:
        """Return a BeamConfig with physical values derived from raw parameters."""
        fluence = self.base_fluence * torch.exp(self.delta_log_fluence)

        cfg = replace(self._base_config)
        cfg.fluence = fluence
        return cfg


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
        "detector_rotation": (
            "detector_params.delta_rotx_raw", "detector_params.delta_roty_raw",
            "detector_params.delta_rotz_raw",
        ),
        "twotheta": ("detector_params.delta_twotheta_raw",),
        "beam_center": (
            "detector_params.delta_beam_s_raw", "detector_params.delta_beam_f_raw",
        ),
        "fluence": ("beam_params.delta_log_fluence",),
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
    ) -> None:
        super().__init__()
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
        params = dict(self.named_parameters())
        if not names:
            return list(params.values())
        resolved = []
        for name in names:
            if name in self.PARAMETER_GROUPS:
                resolved.extend(params[n] for n in self.PARAMETER_GROUPS[name])
            elif name in params:
                resolved.append(params[name])
            else:
                raise KeyError(
                    f"unknown parameter or group {name!r}; groups: "
                    f"{sorted(self.PARAMETER_GROUPS)}, parameters: {sorted(params)}"
                )
        return resolved

    def refine(self, *names: str) -> "ExperimentModel":
        """Turn gradients on for these groups/parameters (all of them if none given)."""
        for p in self._resolve(names):
            p.requires_grad_(True)
        return self

    def freeze(self, *names: str) -> "ExperimentModel":
        """Turn gradients off for these groups/parameters (all of them if none given).

        A frozen parameter keeps its current value, so freezing mid-refinement
        holds it where the optimizer left it.
        """
        for p in self._resolve(names):
            p.requires_grad_(False)
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
        self._external_hkl_data = hkl_data
        self._external_hkl_metadata = hkl_metadata
        if self._crystal is not None:
            self._apply_structure_factors(self._crystal)

    def _apply_structure_factors(self, crystal: Crystal) -> None:
        if self._external_hkl_data is None:
            return
        hkl_tensor = torch.as_tensor(self._external_hkl_data)
        crystal.hkl_data = hkl_tensor.to(device=self.device, dtype=self.dtype)
        crystal.hkl_metadata = (
            dict(self._external_hkl_metadata)
            if self._external_hkl_metadata is not None
            else None
        )

    # Fields the Stage-A blocks produce. Everything else in the configs is fixed at
    # construction.
    _CRYSTAL_CELL_FIELDS = (
        "cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma",
    )
    _DETECTOR_FIELDS = (
        "distance_mm", "detector_rotx_deg", "detector_roty_deg", "detector_rotz_deg",
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

        for name in self._DETECTOR_FIELDS:
            setattr(self._detector.config, name, getattr(detector_cfg, name))

        self._beam_config.fluence = beam_cfg.fluence
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
