"""
Detector model for nanoBragg PyTorch implementation.

This module defines the Detector class responsible for managing all detector
geometry calculations and pixel coordinate generation.
"""

from typing import Optional, Tuple

import torch

from ..config import DetectorConfig, DetectorConvention
from ..utils.units import mm_to_angstroms, degrees_to_radians
from ..utils.tensor_utils import as_tensor_preserving_grad


def _nonzero_scalar(x) -> bool:
    """Return a Python bool for 'x != 0' even if x is a 0-dim torch.Tensor."""
    if isinstance(x, torch.Tensor):
        # Use tensor operations to maintain gradient flow
        return bool((torch.abs(x) > 1e-12).item())
    return abs(float(x)) > 1e-12


class Detector:
    """
    Detector model managing geometry and pixel coordinates.

    **Authoritative Specification:** For a complete specification of this
    component's coordinate systems, conventions, and unit handling, see the
    full architectural deep dive: `docs/architecture/detector.md`.

    Responsible for:
    - Detector position and orientation (basis vectors)
    - Pixel coordinate generation and caching
    - Solid angle corrections
    """

    def __init__(
        self, config: Optional[DetectorConfig] = None, device=None, dtype=torch.float32
    ):
        """Initialize detector from configuration."""
        # Normalize device to ensure consistency
        if device is not None:
            # Create a dummy tensor on the device to get the actual device with index
            temp = torch.zeros(1, device=device)
            self.device = temp.device
        else:
            self.device = torch.device("cpu")
        self.dtype = dtype

        # Use provided config or create default
        if config is None:
            config = DetectorConfig()  # Use defaults
        self.config = config

        # NOTE: Detector geometry works in METERS, not Angstroms!
        # This is different from the physics calculations which use Angstroms
        # The C-code detector geometry calculations use meters as evidenced by
        # DETECTOR_PIX0_VECTOR outputting values like 0.1 (for 100mm distance)
        #
        # DBEX-GRADIENT-001: distance, pixel_size, and close_distance are now
        # properties that dynamically read from config to support post-creation
        # parameter updates with gradient flow. This enables DBEX-style optimization
        # where config parameters are overwritten with differentiable tensors.
        #
        # Internal backing field for close_distance when it needs to be cached
        # (e.g., after _calculate_pix0_vector updates it based on r-factor)
        self._close_distance_cached: Optional[torch.Tensor] = None

        # Copy dimension parameters
        self.spixels = config.spixels
        self.fpixels = config.fpixels

        # Convert beam center from mm to pixels
        # Note: beam center is given in mm from detector origin
        # MOSFLM convention adds +0.5 pixel offset per specs/spec-a-core.md:72
        self.beam_center_s: torch.Tensor
        self.beam_center_f: torch.Tensor

        # Import DetectorConvention for checking
        from ..config import DetectorConvention

        # Calculate pixel coordinates from mm values
        # CRITICAL [DETECTOR-CONFIG-001]: MOSFLM +0.5 pixel offset handling
        # Per specs/spec-a-core.md §72: "Fbeam = Ybeam + 0.5·pixel; Sbeam = Xbeam + 0.5·pixel"
        # Per arch.md §ADR-03: "MOSFLM: Fbeam = Ybeam + 0.5·pixel; Sbeam = Xbeam + 0.5·pixel."
        #
        # The MOSFLM +0.5 pixel offset is part of the beam-center MAPPING formula and
        # must be applied to ALL beam centers (both auto-calculated and explicitly-provided)
        # when using MOSFLM convention. This is NOT a default value adjustment - it's part
        # of how MOSFLM convention defines the relationship between Xbeam/Ybeam (user input)
        # and Fbeam/Sbeam (internal working values).
        #
        # The offset is applied here during the mm→pixel conversion to ensure it affects
        # ALL MOSFLM beam centers consistently, matching the C-code behavior.

        # BL-1: Guard against None values (should be set by __post_init__, but defend)
        if config.beam_center_s is None or config.beam_center_f is None:
            raise ValueError(
                "beam_center_s and beam_center_f must be set. "
                "DetectorConfig.__post_init__ should have set defaults."
            )

        # Convert mm to pixels
        beam_center_s_pixels = config.beam_center_s / config.pixel_size_mm
        beam_center_f_pixels = config.beam_center_f / config.pixel_size_mm

        # DETECTOR-CONFIG-001 Phase C3: Conditional MOSFLM +0.5 pixel offset
        # MOSFLM convention ALWAYS applies +0.5 pixel offset per spec-a-core.md:72
        # The offset is part of the beam-center mapping formula for MOSFLM convention:
        #   Fbeam = Ybeam + 0.5·pixel; Sbeam = Xbeam + 0.5·pixel
        if config.detector_convention == DetectorConvention.MOSFLM:
            beam_center_s_pixels = beam_center_s_pixels + 0.5
            beam_center_f_pixels = beam_center_f_pixels + 0.5

        # Convert to tensors on proper device
        if isinstance(beam_center_s_pixels, torch.Tensor):
            self.beam_center_s = beam_center_s_pixels.to(device=self.device, dtype=self.dtype)
        else:
            self.beam_center_s = torch.tensor(
                beam_center_s_pixels,
                device=self.device,
                dtype=self.dtype,
            )

        if isinstance(beam_center_f_pixels, torch.Tensor):
            self.beam_center_f = beam_center_f_pixels.to(device=self.device, dtype=self.dtype)
        else:
            self.beam_center_f = torch.tensor(
                beam_center_f_pixels,
                device=self.device,
                dtype=self.dtype,
            )

        # Initialize basis vectors
        if self._is_default_config():
            # PHASE 1 FIX: Corrected basis vectors to match C implementation exactly
            # From C code trace: DETECTOR_FAST_AXIS 0 0 1, DETECTOR_SLOW_AXIS 0 -1 0, DETECTOR_NORMAL_AXIS 1 0 0
            # This fixes the 192.67mm coordinate system error
            self.fdet_vec = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype  # Fast along positive Z
            )
            self.sdet_vec = torch.tensor(
                [0.0, -1.0, 0.0], device=self.device, dtype=self.dtype  # Slow along negative Y
            )
            self.odet_vec = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device, dtype=self.dtype   # Origin along positive X (beam)
            )
        else:
            # Calculate basis vectors dynamically in Phase 2
            self.fdet_vec, self.sdet_vec, self.odet_vec = (
                self._calculate_basis_vectors()
            )

        # Calculate and cache pix0_vector (position of first pixel)
        self._calculate_pix0_vector()

        self._pixel_coords_cache: Optional[torch.Tensor] = None
        self._geometry_version = 0
        self._cached_basis_vectors = (
            self.fdet_vec.clone(),
            self.sdet_vec.clone(),
            self.odet_vec.clone(),
        )
        self._cached_pix0_vector = self.pix0_vector.clone()

        # Initialize attributes used by pyrefly only if not already set
        # These will be set properly in _calculate_pix0_vector
        if not hasattr(self, 'distance_corrected'):
            self.distance_corrected: Optional[torch.Tensor] = None
        if not hasattr(self, 'r_factor'):
            self.r_factor: Optional[torch.Tensor] = None
        if not hasattr(self, 'pix0_vector'):
            self.pix0_vector: Optional[torch.Tensor] = None

    # =========================================================================
    # DBEX-GRADIENT-001: Dynamic properties for geometry parameters
    # =========================================================================
    # These properties read from config dynamically to support post-creation
    # parameter updates with gradient flow preservation.

    @property
    def distance(self) -> torch.Tensor:
        """
        Detector distance in meters, dynamically read from config.

        DBEX-GRADIENT-001: This property enables post-creation parameter updates
        where config.distance_mm can be overwritten with a differentiable tensor.
        The property preserves gradient flow by using as_tensor_preserving_grad.

        Returns:
            torch.Tensor: Distance in meters (config.distance_mm / 1000)
        """
        return as_tensor_preserving_grad(
            self.config.distance_mm / 1000.0, device=self.device, dtype=self.dtype
        )

    @property
    def pixel_size(self) -> torch.Tensor:
        """
        Pixel size in meters, dynamically read from config.

        DBEX-GRADIENT-001: This property enables post-creation parameter updates
        where config.pixel_size_mm can be overwritten with a differentiable tensor.

        Returns:
            torch.Tensor: Pixel size in meters (config.pixel_size_mm / 1000)
        """
        return as_tensor_preserving_grad(
            self.config.pixel_size_mm / 1000.0, device=self.device, dtype=self.dtype
        )

    @property
    def close_distance(self) -> torch.Tensor:
        """
        Close distance in meters (for obliquity calculations).

        DBEX-GRADIENT-001: This property supports both cached values (from r-factor
        calculations in _calculate_pix0_vector) and dynamic config reads.

        Returns:
            torch.Tensor: Close distance in meters
        """
        # If cached by _calculate_pix0_vector, return cached value
        if self._close_distance_cached is not None:
            return self._close_distance_cached
        # Otherwise derive from config
        if self.config.close_distance_mm is not None:
            return as_tensor_preserving_grad(
                self.config.close_distance_mm / 1000.0, device=self.device, dtype=self.dtype
            )
        # Default to distance if not specified
        return self.distance

    @close_distance.setter
    def close_distance(self, value: torch.Tensor) -> None:
        """
        Set cached close_distance value.

        This setter is used by _calculate_pix0_vector to cache the r-factor
        corrected close_distance.

        Args:
            value: Close distance tensor in meters
        """
        self._close_distance_cached = value

    def _is_default_config(self) -> bool:
        """Check if using default config (for backward compatibility)."""
        from ..config import DetectorConvention

        c = self.config
        # Check all basic parameters
        # Note: MOSFLM default beam_center is now 51.25 mm per spec-a-core.md §71
        # Formula: (detsize + pixel)/2 = (102.4 + 0.1)/2 = 51.25 mm
        # The MOSFLM +0.5 pixel mapping offset is applied during mm→pixel conversion in __init__
        basic_check = (
            c.distance_mm == 100.0
            and c.pixel_size_mm == 0.1
            and c.spixels == 1024
            and c.fpixels == 1024
            and c.beam_center_s == 51.25
            and c.beam_center_f == 51.25
        )

        # Check detector convention is default (MOSFLM)
        convention_check = c.detector_convention == DetectorConvention.MOSFLM

        # Check rotation parameters (handle both float and tensor)
        rotx_check = (
            c.detector_rotx_deg == 0
            if isinstance(c.detector_rotx_deg, (int, float))
            else torch.allclose(
                c.detector_rotx_deg, torch.tensor(0.0, dtype=c.detector_rotx_deg.dtype)
            )
        )
        roty_check = (
            c.detector_roty_deg == 0
            if isinstance(c.detector_roty_deg, (int, float))
            else torch.allclose(
                c.detector_roty_deg, torch.tensor(0.0, dtype=c.detector_roty_deg.dtype)
            )
        )
        rotz_check = (
            c.detector_rotz_deg == 0
            if isinstance(c.detector_rotz_deg, (int, float))
            else torch.allclose(
                c.detector_rotz_deg, torch.tensor(0.0, dtype=c.detector_rotz_deg.dtype)
            )
        )
        twotheta_check = (
            c.detector_twotheta_deg == 0
            if isinstance(c.detector_twotheta_deg, (int, float))
            else torch.allclose(
                c.detector_twotheta_deg,
                torch.tensor(0.0, dtype=c.detector_twotheta_deg.dtype),
            )
        )

        return bool(
            basic_check
            and convention_check
            and rotx_check
            and roty_check
            and rotz_check
            and twotheta_check
        )

    def to(self, device=None, dtype=None):
        """Move detector to specified device and/or dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        # Move basis vectors to new device/dtype
        self.fdet_vec = self.fdet_vec.to(device=self.device, dtype=self.dtype)
        self.sdet_vec = self.sdet_vec.to(device=self.device, dtype=self.dtype)
        self.odet_vec = self.odet_vec.to(device=self.device, dtype=self.dtype)

        # Move beam center tensors (handle both tensor and scalar cases)
        if isinstance(self.beam_center_s, torch.Tensor):
            self.beam_center_s = self.beam_center_s.to(device=self.device, dtype=self.dtype)
        else:
            self.beam_center_s = torch.tensor(
                self.beam_center_s,
                device=self.device,
                dtype=self.dtype,
            )

        if isinstance(self.beam_center_f, torch.Tensor):
            self.beam_center_f = self.beam_center_f.to(device=self.device, dtype=self.dtype)
        else:
            self.beam_center_f = torch.tensor(
                self.beam_center_f,
                device=self.device,
                dtype=self.dtype,
            )

        # Invalidate cache since device/dtype changed
        self.invalidate_cache()
        return self

    def invalidate_cache(self):
        """Invalidate cached pixel coordinates when geometry changes."""
        self._pixel_coords_cache = None
        self._geometry_version += 1
        # Recalculate pix0_vector when geometry changes
        self._calculate_pix0_vector()

    def _apply_mosflm_beam_convention(self):
        """
        Apply MOSFLM beam center convention with axis swap and pixel offset.
        
        MOSFLM convention requires:
        1. Axis swap: beam_center_s (slow) → F axis, beam_center_f (fast) → S axis
        2. +0.5 pixel offset for both axes
        3. Convert to meters for internal geometry calculations
        
        Returns:
            tuple: (beam_f_m, beam_s_m) in meters for use in pix0_vector calculation
        """
        from ..config import DetectorConvention
        
        if self.config.detector_convention == DetectorConvention.MOSFLM:
            # MOSFLM convention: Apply axis swap AND +0.5 pixel offset
            # beam_center_s → Fbeam (with swap and offset)
            # beam_center_f → Sbeam (with swap and offset)
            beam_s_mm = self.config.beam_center_s if hasattr(self.config, 'beam_center_s') else 51.2
            beam_f_mm = self.config.beam_center_f if hasattr(self.config, 'beam_center_f') else 51.2
            
            # Convert to pixels first, add 0.5 pixel offset, then to meters
            beam_s_pixels = beam_s_mm / self.config.pixel_size_mm  # mm to pixels
            beam_f_pixels = beam_f_mm / self.config.pixel_size_mm  # mm to pixels
            
            # MOSFLM axis mapping with 0.5 pixel offset:
            # Fbeam = beam_center_s (in pixels) + 0.5, then to meters
            # Sbeam = beam_center_f (in pixels) + 0.5, then to meters  
            adjusted_f = (beam_s_pixels + 0.5) * self.pixel_size  # S→F mapping
            adjusted_s = (beam_f_pixels + 0.5) * self.pixel_size  # F→S mapping
        else:
            # Standard convention: no axis swap, no pixel offset
            beam_s_mm = self.config.beam_center_s if hasattr(self.config, 'beam_center_s') else 51.2
            beam_f_mm = self.config.beam_center_f if hasattr(self.config, 'beam_center_f') else 51.2
            
            adjusted_f = beam_f_mm / 1000.0  # mm to meters
            adjusted_s = beam_s_mm / 1000.0  # mm to meters
        
        return adjusted_f, adjusted_s

    def get_effective_distance(self):
        """
        Apply geometric distance correction for tilted detector.
        
        When detector is tilted, effective distance = nominal_distance / cos(tilt_angle)
        where tilt_angle is between beam direction [1,0,0] and detector normal.
        
        Returns:
            torch.Tensor: Effective distance in meters
        """
        # Get the rotated detector normal vector (after all rotations)
        detector_normal = self.odet_vec  # Should already be normalized
        
        # Beam travels along positive X axis in MOSFLM convention
        beam_direction = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        
        # Calculate cosine of angle between beam and detector normal
        cos_angle = torch.dot(beam_direction, detector_normal)
        
        # Prevent division by zero for perpendicular detector
        cos_angle = torch.clamp(cos_angle, min=0.001)
        
        # Apply the distance correction formula
        if isinstance(self.distance, torch.Tensor):
            effective_distance = self.distance / cos_angle
        else:
            effective_distance = torch.tensor(self.distance, dtype=self.dtype, device=self.device) / cos_angle
        
        return effective_distance

    def _is_custom_convention(self) -> bool:
        """
        Check if C code will use CUSTOM convention instead of MOSFLM.

        Based on the C code analysis:
        - MOSFLM convention (no explicit -twotheta_axis): Uses +0.5 pixel offset
        - CUSTOM convention (explicit -twotheta_axis): No +0.5 pixel offset

        This method delegates to the DetectorConfig.should_use_custom_convention()
        method which replicates the exact C code logic.

        Returns:
            bool: True if CUSTOM convention should be used
        """
        return self.config.should_use_custom_convention()

    def _calculate_pix0_vector(self):
        """
        Calculate the position of the first pixel (0,0) in 3D space.

        This follows the C-code convention where pix0_vector represents the
        3D position of pixel (0,0), taking into account the beam center offset
        and detector positioning.

        The calculation depends on:
        1. detector_pivot mode (BEAM vs SAMPLE)
        2. detector convention (MOSFLM vs CUSTOM)
        3. r-factor distance correction (AT-GEO-003)

        CRITICAL: C code switches to CUSTOM convention when twotheta_axis is
        explicitly specified and differs from MOSFLM default [0,0,-1].
        CUSTOM convention removes the +0.5 pixel offset that MOSFLM adds.

        - BEAM pivot: pix0_vector = -Fbeam*fdet_vec - Sbeam*sdet_vec + distance*beam_vec
        - SAMPLE pivot: Calculate pix0_vector BEFORE rotations, then rotate it

        C-Code Implementation Reference (from nanoBragg.c):
        For r-factor calculation (lines 1727-1732):
        ```c
        /* first off, what is the relationship between the two "beam centers"? */
        rotate(odet_vector,vector,detector_rotx,detector_roty,detector_rotz);
        ratio = dot_product(beam_vector,vector);
        if(ratio == 0.0) { ratio = DBL_MIN; }
        if(isnan(close_distance)) close_distance = fabs(ratio*distance);
        distance = close_distance/ratio;
        ```

        For SAMPLE pivot (lines 376-385):
        ```c
        if(detector_pivot == SAMPLE){
            printf("pivoting detector around sample\n");
            /* initialize detector origin before rotating detector */
            pix0_vector[1] = -Fclose*fdet_vector[1]-Sclose*sdet_vector[1]+close_distance*odet_vector[1];
            pix0_vector[2] = -Fclose*fdet_vector[2]-Sclose*sdet_vector[2]+close_distance*odet_vector[2];
            pix0_vector[3] = -Fclose*fdet_vector[3]-Sclose*sdet_vector[3]+close_distance*odet_vector[3];

            /* now swing the detector origin around */
            rotate(pix0_vector,pix0_vector,detector_rotx,detector_roty,detector_rotz);
            rotate_axis(pix0_vector,pix0_vector,twotheta_axis,detector_twotheta);
        }
        ```
        For BEAM pivot (lines 398-403):
        ```c
        if(detector_pivot == BEAM){
            printf("pivoting detector around direct beam spot\n");
            pix0_vector[1] = -Fbeam*fdet_vector[1]-Sbeam*sdet_vector[1]+distance*beam_vector[1];
            pix0_vector[2] = -Fbeam*fdet_vector[2]-Sbeam*sdet_vector[2]+distance*beam_vector[2];
            pix0_vector[3] = -Fbeam*fdet_vector[3]-Sbeam*sdet_vector[3]+distance*beam_vector[3];
        }
        ```

        MOSFLM vs CUSTOM convention (from nanoBragg.c lines 1218-1219 vs 1236-1239):
        MOSFLM: Fbeam = Ybeam + 0.5*pixel_size, Sbeam = Xbeam + 0.5*pixel_size
        CUSTOM: Fclose = Xbeam, Sclose = Ybeam (no +0.5 offset)

        Note: Pixel coordinates are generated at pixel corners/edges (pixel 0 at position 0, etc.)
        """
        from ..config import DetectorPivot, DetectorConvention
        from ..utils.geometry import angles_to_rotation_matrix, rotate_axis
        from ..utils.units import degrees_to_radians

        # Convert pix0_override to tensor if provided (CLI-FLAGS-003 Phase F2)
        # This will be used instead of calculating pix0 from pivot formulas,
        # but we still need to calculate r_factor and derive close_distance from it
        pix0_override_tensor = None
        if self.config.pix0_override_m is not None:
            override = self.config.pix0_override_m
            if isinstance(override, (tuple, list)):
                pix0_override_tensor = torch.tensor(override, device=self.device, dtype=self.dtype)
            elif isinstance(override, torch.Tensor):
                pix0_override_tensor = override.to(device=self.device, dtype=self.dtype)
            else:
                raise TypeError(f"pix0_override_m must be tuple, list, or Tensor, got {type(override)}")

        # Calculate r-factor for distance correction (AT-GEO-003)
        c = self.config

        # Get rotation angles as tensors
        detector_rotx = degrees_to_radians(c.detector_rotx_deg)
        detector_roty = degrees_to_radians(c.detector_roty_deg)
        detector_rotz = degrees_to_radians(c.detector_rotz_deg)
        detector_twotheta = degrees_to_radians(c.detector_twotheta_deg)

        if not isinstance(detector_rotx, torch.Tensor):
            detector_rotx = torch.tensor(detector_rotx, device=self.device, dtype=self.dtype)
        elif detector_rotx.device != self.device or detector_rotx.dtype != self.dtype:
            detector_rotx = detector_rotx.to(device=self.device, dtype=self.dtype)

        if not isinstance(detector_roty, torch.Tensor):
            detector_roty = torch.tensor(detector_roty, device=self.device, dtype=self.dtype)
        elif detector_roty.device != self.device or detector_roty.dtype != self.dtype:
            detector_roty = detector_roty.to(device=self.device, dtype=self.dtype)

        if not isinstance(detector_rotz, torch.Tensor):
            detector_rotz = torch.tensor(detector_rotz, device=self.device, dtype=self.dtype)
        elif detector_rotz.device != self.device or detector_rotz.dtype != self.dtype:
            detector_rotz = detector_rotz.to(device=self.device, dtype=self.dtype)

        if not isinstance(detector_twotheta, torch.Tensor):
            detector_twotheta = torch.tensor(detector_twotheta, device=self.device, dtype=self.dtype)
        elif detector_twotheta.device != self.device or detector_twotheta.dtype != self.dtype:
            detector_twotheta = detector_twotheta.to(device=self.device, dtype=self.dtype)

        # Get beam vector using self.beam_vector property
        # This honors CUSTOM convention overrides from CLI (e.g., -beam_vector)
        beam_vector = self.beam_vector

        # Always calculate r-factor to preserve gradient flow
        # When rotations are zero, the rotation matrix will be identity
        # and r-factor will naturally be 1.0
        if True:  # Always execute to preserve gradients
            # nanoBragg.c computes ratio = beam·odet after rotx/y/z but BEFORE the twotheta
            # swing: rotate(odet_vector,vector,rotx,roty,rotz); ratio = dot_product(beam_vector,vector);
            _, _, odet_initial = self._initial_basis_vectors()
            rotation_matrix = angles_to_rotation_matrix(detector_rotx, detector_roty, detector_rotz)
            ratio = torch.dot(beam_vector, torch.matmul(rotation_matrix, odet_initial))

            # Prevent division by zero while maintaining gradient flow
            # Use torch operations to preserve gradients
            min_ratio = torch.tensor(1e-10, device=self.device, dtype=self.dtype)
            ratio = torch.where(
                torch.abs(ratio) < min_ratio,
                torch.sign(ratio) * min_ratio,
                ratio
            )
            # Handle zero case
            ratio = torch.where(
                ratio == 0,
                min_ratio,
                ratio
            )

            # Update distance based on r-factor
            # If close_distance is specified, use it; otherwise calculate from nominal distance
            if hasattr(self.config, 'close_distance_mm') and self.config.close_distance_mm is not None:
                close_distance = self.config.close_distance_mm / 1000.0  # Convert mm to meters
            else:
                close_distance = abs(ratio * self.distance)  # Default from C code

            # Update the actual distance using r-factor (implements AT-GEO-003)
            self.distance_corrected = close_distance / ratio

            # CRITICAL: Update the stored close_distance for obliquity calculations
            # This is needed when detector is rotated (e.g., with twotheta)
            self.close_distance = close_distance

        # Store r-factor for later verification
        # Ensure it's a tensor with correct dtype
        if isinstance(ratio, torch.Tensor):
            self.r_factor = ratio.to(device=self.device, dtype=self.dtype)
        else:
            self.r_factor = torch.tensor(ratio, device=self.device, dtype=self.dtype)

        if self.config.detector_pivot == DetectorPivot.BEAM:
            # BEAM pivot mode: detector rotates around the direct beam spot
            # Use exact C-code formula: pix0_vector = -Fbeam*fdet_vec - Sbeam*sdet_vec + distance*beam_vec

            # Calculate Fbeam and Sbeam from beam centers
            # For MOSFLM: Fbeam = Ybeam + 0.5*pixel_size, Sbeam = Xbeam + 0.5*pixel_size (C-code line 382)
            # CLI-FLAGS-003 Phase H5e unit correction (2025-10-24):
            # CRITICAL: self.beam_center_f and self.beam_center_s are stored as config values
            # from DetectorConfig in MILLIMETERS (see config.py:186-188, __main__.py:921-922, 930-931).
            # Must convert mm→m before use in geometry calculations.
            #
            # C-code reference (nanoBragg.c:1184-1239, specifically lines 1220-1221 for MOSFLM):
            # Fbeam_m = (Ybeam_mm + 0.5*pixel_mm) / 1000
            # Sbeam_m = (Xbeam_mm + 0.5*pixel_mm) / 1000
            #
            # Since __main__.py already applies the axis swap at lines 921-922, we only need to:
            # 1. Convert beam_center_f/s from mm to m (÷ 1000)
            # 2. Add the MOSFLM +0.5 pixel offset (in meters)
            #
            # Reports/2025-10-cli-flags/phase_h5/py_traces/2025-10-22/diff_notes.md documents
            # the 1.1mm ΔF that this fix resolves.

            # BL-1: Guard against None (should not happen after __post_init__, but pyrefly requires it)
            if self.config.beam_center_f is None or self.config.beam_center_s is None:
                raise ValueError(
                    "beam_center_f and beam_center_s must be set by DetectorConfig.__post_init__"
                )

            # CRITICAL [DETECTOR-CONFIG-001]: The +0.5 offset was already applied in __init__
            # (lines 104-106), so self.beam_center_f/s are now in pixels WITH the offset
            # We just need to convert from pixels to meters:
            Fbeam = self.beam_center_f * self.pixel_size  # pixels * (m/pixel) → meters
            Sbeam = self.beam_center_s * self.pixel_size  # pixels * (m/pixel) → meters

            # Reuse beam_vector from r-factor calculation above
            # This ensures CUSTOM convention overrides (e.g., -beam_vector) are honored
            # beam_vector is already set via self.beam_vector property

            # CLI-FLAGS-003 Phase H5b: Handle pix0_override for BEAM pivot
            # PRECEDENCE RULE (Phase H5a evidence, 2025-10-22):
            # Per reports/2025-10-cli-flags/phase_h5/c_precedence_2025-10-22.md:
            # C code IGNORES -pix0_vector_mm when custom detector vectors are present.
            # Custom vectors supersede pix0 overrides.
            #
            # pix0_override application workflow (when no custom vectors):
            # 1. Subtract beam term to get detector offset: pix0_delta = pix0_override - distance*beam
            # 2. Project onto detector axes to get Fbeam_override, Sbeam_override
            #    Fbeam = -dot(pix0_delta, fdet); Sbeam = -dot(pix0_delta, sdet)
            # 3. Update beam_center_f/s tensors for header consistency
            # 4. Apply standard BEAM formula with derived Fbeam/Sbeam
            #
            # Detection of custom vectors: check if any of custom_fdet/sdet/odet_vector are supplied
            has_custom_vectors = any([
                self.config.custom_fdet_vector is not None,
                self.config.custom_sdet_vector is not None,
                self.config.custom_odet_vector is not None
            ])

            # Apply pix0 override only when NO custom vectors are present
            if pix0_override_tensor is not None and not has_custom_vectors:
                # Ensure all tensors on same device/dtype
                beam_vector_local = beam_vector.to(device=self.device, dtype=self.dtype)
                fdet_local = self.fdet_vec.to(device=self.device, dtype=self.dtype)
                sdet_local = self.sdet_vec.to(device=self.device, dtype=self.dtype)
                pix0_override_local = pix0_override_tensor.to(device=self.device, dtype=self.dtype)

                # Compute beam term: distance_corrected * beam_vector
                beam_term = self.distance_corrected * beam_vector_local

                # Subtract beam term to get detector offset
                pix0_delta = pix0_override_local - beam_term

                # Project onto detector axes (with sign convention from C code)
                # Fbeam = -dot(pix0_delta, fdet); Sbeam = -dot(pix0_delta, sdet)
                Fbeam_override = -torch.dot(pix0_delta, fdet_local)
                Sbeam_override = -torch.dot(pix0_delta, sdet_local)

                # Override the Fbeam/Sbeam values with derived ones
                Fbeam = Fbeam_override
                Sbeam = Sbeam_override

                # Update beam_center_f/s tensors to maintain header consistency
                # Convert from meters back to pixels: beam_center = offset_m / pixel_size_m
                # CRITICAL [DETECTOR-CONFIG-001]: beam_center_f/s now STORE pixel values WITH offset already
                # So we just convert Fbeam/Sbeam (which are in meters) back to pixels
                self.beam_center_f = (Fbeam / self.pixel_size).to(device=self.device, dtype=self.dtype)
                self.beam_center_s = (Sbeam / self.pixel_size).to(device=self.device, dtype=self.dtype)

            # Use exact C-code formula WITH distance correction (AT-GEO-003)
            # BEAM pivot formula (C-code reference: nanoBragg.c lines 1833-1835):
            # pix0_vector[1] = -Fbeam*fdet_vector[1]-Sbeam*sdet_vector[1]+distance*beam_vector[1];
            # pix0_vector[2] = -Fbeam*fdet_vector[2]-Sbeam*sdet_vector[2]+distance*beam_vector[2];
            # pix0_vector[3] = -Fbeam*fdet_vector[3]-Sbeam*sdet_vector[3]+distance*beam_vector[3];

            self.pix0_vector = (
                -Fbeam * self.fdet_vec
                - Sbeam * self.sdet_vec
                + self.distance_corrected * beam_vector
            )
        else:
            # SAMPLE pivot mode: detector rotates around the sample
            # IMPORTANT: Compute pix0 BEFORE rotating, using the same formula as C:
            # pix0 = -Fclose*fdet - Sclose*sdet + close_distance*odet

            fdet_initial, sdet_initial, odet_initial = self._initial_basis_vectors()

            # Distances from pixel (0,0) center to the beam spot, measured along detector axes
            # Mapping clarification:
            # - In C code: Fbeam = Ybeam + 0.5*pixel, Sbeam = Xbeam + 0.5*pixel (MOSFLM)
            # - In c_reference_utils.py: Xbeam=beam_center_s, Ybeam=beam_center_f
            # - In PyTorch: beam_center_f is fast, beam_center_s is slow
            # For consistency with BEAM pivot mode:
            # - Fclose (fast coord) ← beam_center_f (fast param)
            # - Sclose (slow coord) ← beam_center_s (slow param)
            # NOTE: The beam centers already have the +0.5 offset from __init__ for MOSFLM!

            # CRITICAL [DETECTOR-CONFIG-001]: The +0.5 offset was already applied in __init__
            # (lines 104-106 for MOSFLM), so self.beam_center_f/s are now in pixels WITH the offset
            # We just need to convert from pixels to meters for all conventions:
            Fclose = self.beam_center_f * self.pixel_size  # pixels * (m/pixel) → meters
            Sclose = self.beam_center_s * self.pixel_size  # pixels * (m/pixel) → meters

            # Compute pix0 BEFORE rotations using close_distance if specified
            # When close_distance is provided, use it directly for SAMPLE pivot
            # CLI-FLAGS-003 Phase L3k.3c.4: CRITICAL FIX - use close_distance not distance
            # C code (nanoBragg.c:1739-1745) uses close_distance for SAMPLE pivot pix0 calc
            # close_distance is r-factor corrected (set at line 475 above)
            # Using nominal self.distance causes 2.85µm pix0_z error
            if hasattr(self.config, 'close_distance_mm') and self.config.close_distance_mm is not None:
                initial_distance = self.config.close_distance_mm / 1000.0  # Convert mm to meters
            else:
                initial_distance = self.close_distance  # Use r-factor corrected close_distance

            pix0_initial = (
                -Fclose * fdet_initial
                - Sclose * sdet_initial
                + initial_distance * odet_initial
            )

            # Now rotate pix0 with detector_rotx/roty/rotz and twotheta, same as C
            pix0_rotated = torch.matmul(rotation_matrix, pix0_initial)

            if isinstance(c.twotheta_axis, torch.Tensor):
                twotheta_axis = c.twotheta_axis.to(device=self.device, dtype=self.dtype)
            elif c.twotheta_axis is not None:
                twotheta_axis = torch.tensor(c.twotheta_axis, device=self.device, dtype=self.dtype)
            else:
                # Default to convention-specific axis (set in config.__post_init__)
                # MOSFLM uses [0, 0, -1], XDS uses [1, 0, 0], etc.
                # This should already be set in config, but fallback to MOSFLM default
                twotheta_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=self.dtype)

            # Always apply twotheta rotation to preserve gradients
            # When detector_twotheta is zero, this will be identity
            pix0_rotated = rotate_axis(pix0_rotated, twotheta_axis, detector_twotheta)

            # CLI-FLAGS-003 Phase F2: SAMPLE pivot always uses calculated pix0
            # The C code (nanoBragg.c:1739-1745) shows that pix0_override is IGNORED for SAMPLE pivot;
            # instead, Fclose/Sclose from beam centers are used in the standard formula, then rotated.
            self.pix0_vector = pix0_rotated

        # ALWAYS recalculate close_distance from final pix0_vector (C code nanoBragg.c:1846)
        # This ensures consistency between pix0 and close_distance for all pivot modes
        # CRITICAL: Keep as tensor for differentiability (Core Rule #9)
        close_dist_tensor = torch.dot(self.pix0_vector, self.odet_vec)
        self.close_distance = close_dist_tensor

        # CLI-FLAGS-003 Phase H4a: Post-rotation beam-centre recomputation
        # Port nanoBragg.c lines 1851-1860 to update Fbeam/Sbeam and distance_corrected
        #
        # C-Code Implementation Reference (from nanoBragg.c, lines 1851-1860):
        # ```c
        # /* where is the direct beam now? */
        # /* difference between beam impact vector and detector origin */
        # newvector[1] = close_distance/ratio*beam_vector[1]-pix0_vector[1];
        # newvector[2] = close_distance/ratio*beam_vector[2]-pix0_vector[2];
        # newvector[3] = close_distance/ratio*beam_vector[3]-pix0_vector[3];
        # /* extract components along detector vectors */
        # Fbeam = dot_product(fdet_vector,newvector);
        # Sbeam = dot_product(sdet_vector,newvector);
        # distance = close_distance/ratio;
        # ```
        #
        # This recomputation is crucial when custom pix0 vectors or rotations are present.
        # The beam impact point moves relative to the detector origin after rotations,
        # so Fbeam/Sbeam must be recalculated to maintain geometric consistency.

        # Compute beam impact vector minus detector origin
        # newvector = (close_distance / r_factor) * beam_vector - pix0_vector
        beam_impact_term = (close_dist_tensor / self.r_factor) * beam_vector
        newvector = beam_impact_term - self.pix0_vector

        # Extract components along detector axes to get updated Fbeam/Sbeam
        # Note: C code updates Fbeam/Sbeam variables but does NOT update Xbeam/Ybeam beam centers
        # These recomputed values are used for subsequent geometry calculations in the C code
        # but we don't need to store them as they're only intermediate values
        Fbeam_recomputed = torch.dot(self.fdet_vec, newvector)
        Sbeam_recomputed = torch.dot(self.sdet_vec, newvector)

        # Update distance_corrected from close_distance and r_factor
        # This matches C code: distance = close_distance/ratio (line 1859)
        self.distance_corrected = close_dist_tensor / self.r_factor

    def get_pixel_coords(self) -> torch.Tensor:
        """
        Get 3D coordinates of all detector pixels.

        Supports both planar and curved (spherical) detector mappings.
        For curved detector: pixels are mapped to a spherical arc by rotating
        from the beam direction by angles Sdet/distance and Fdet/distance.

        Returns:
            torch.Tensor: Pixel coordinates with shape (spixels, fpixels, 3) in meters
        """
        # Check if geometry has changed by comparing cached values
        geometry_changed = False
        if hasattr(self, "_cached_basis_vectors") and hasattr(
            self, "_cached_pix0_vector"
        ):
            # Check if basis vectors have changed
            # Move cached vectors to current device and dtype for comparison
            cached_f = self._cached_basis_vectors[0].to(device=self.device, dtype=self.dtype)
            cached_s = self._cached_basis_vectors[1].to(device=self.device, dtype=self.dtype)
            cached_o = self._cached_basis_vectors[2].to(device=self.device, dtype=self.dtype)

            if not (
                torch.allclose(self.fdet_vec, cached_f, atol=1e-15)
                and torch.allclose(
                    self.sdet_vec, cached_s, atol=1e-15
                )
                and torch.allclose(
                    self.odet_vec, cached_o, atol=1e-15
                )
            ):
                geometry_changed = True
            # Check if pix0_vector has changed
            cached_pix0 = self._cached_pix0_vector.to(device=self.device, dtype=self.dtype)
            if not torch.allclose(
                self.pix0_vector, cached_pix0, atol=1e-15
            ):
                geometry_changed = True

        if self._pixel_coords_cache is None or geometry_changed:
            if self.config.curved_detector:
                # Curved detector mapping (spherical arc)
                pixel_coords = self._compute_curved_pixel_coords()
            else:
                # Standard planar detector mapping
                pixel_coords = self._compute_planar_pixel_coords()

            self._pixel_coords_cache = pixel_coords

            # Update cached values for future comparisons
            self._cached_basis_vectors = (
                self.fdet_vec.clone(),
                self.sdet_vec.clone(),
                self.odet_vec.clone(),
            )
            self._cached_pix0_vector = self.pix0_vector.clone()
            self._geometry_version += 1

        return self._pixel_coords_cache

    def _compute_planar_pixel_coords(self) -> torch.Tensor:
        """
        Compute pixel coordinates for a standard planar detector.

        Returns:
            torch.Tensor: Pixel coordinates with shape (spixels, fpixels, 3) in meters
        """
        # Create pixel index grids - pixel centers to match C code behavior
        # The C code uses pixel centers (0.5, 1.5, 2.5, ...) not corners
        # Adding 0.5 to indices places coordinates at pixel centers
        s_indices = torch.arange(self.spixels, device=self.device, dtype=self.dtype) + 0.5
        f_indices = torch.arange(self.fpixels, device=self.device, dtype=self.dtype) + 0.5

        # Create meshgrid of indices
        s_grid, f_grid = torch.meshgrid(s_indices, f_indices, indexing="ij")

        # Calculate pixel coordinates using pix0_vector as the reference
        # pixel_coords = pix0_vector + s * pixel_size * sdet_vec + f * pixel_size * fdet_vec

        # Expand vectors for broadcasting
        pix0_expanded = self.pix0_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        sdet_expanded = self.sdet_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        fdet_expanded = self.fdet_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)

        # Calculate pixel coordinates
        pixel_coords = (
            pix0_expanded
            + s_grid.unsqueeze(-1) * self.pixel_size * sdet_expanded
            + f_grid.unsqueeze(-1) * self.pixel_size * fdet_expanded
        )

        return pixel_coords

    @staticmethod
    def _rotate_axis(v: torch.Tensor, axis: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Rodrigues rotation of ``v`` about unit ``axis`` by ``phi``, matching
        ``rotate_axis()`` in nanoBragg.c:3378-3391 term for term::

            temp = axis*(axis.v)*(1-cos(phi)) + v*cos(phi) + (axis x v)*sin(phi)

        Args:
            v: (..., 3) vectors to rotate.
            axis: (3,) unit rotation axis.
            phi: (...) rotation angle in radians, broadcastable against ``v[..., 0]``.

        Returns:
            torch.Tensor: rotated vectors, shape broadcast of ``v`` and ``phi``.
        """
        cos_phi = torch.cos(phi).unsqueeze(-1)
        sin_phi = torch.sin(phi).unsqueeze(-1)
        axis_dot_v = torch.sum(v * axis, dim=-1, keepdim=True)
        cross = torch.cross(axis.expand_as(v), v, dim=-1)
        return axis * axis_dot_v * (1.0 - cos_phi) + v * cos_phi + cross * sin_phi

    def apply_curved_mapping(self, planar_coords: torch.Tensor) -> torch.Tensor:
        """
        Map planar detector positions onto the ``-curved_det`` sphere exactly as
        nanoBragg.c:2707-2716 does inside the sensor-layer / sub-pixel loop::

            vector = distance*beam_vector;
            rotate_axis(vector,   newvector, sdet_vector, pixel_pos[2]/distance);
            rotate_axis(newvector, pixel_pos, fdet_vector, pixel_pos[3]/distance);

        Two things are easy to get wrong and are load-bearing here:

        * the rotation angles are the **lab-frame Y and Z components** of the planar
          ``pixel_pos`` (indices 1 and 2 zero-based), divided by ``distance`` -- not the
          detector-plane ``Fdet``/``Sdet`` offsets, and they therefore include the whole
          ``pix0_vector`` (beam centre) contribution;
        * both angles come from the *planar* ``pixel_pos``: C evaluates
          ``pixel_pos[3]/distance`` as an argument before the second ``rotate_axis`` call
          overwrites ``pixel_pos``;
        * the result replaces the position outright -- every pixel ends up exactly
          ``distance`` from the sample, with no ``pix0_vector`` offset left in it.

        ``distance`` is the r-factor corrected distance (``close_distance/ratio``), the
        value C holds in ``distance`` by the time the render loop runs.

        Args:
            planar_coords: (..., 3) planar positions in meters, including any sub-pixel
                and sensor-layer offsets.

        Returns:
            torch.Tensor: (..., 3) curved positions in meters.
        """
        distance = self.get_corrected_distance().to(
            device=planar_coords.device, dtype=planar_coords.dtype
        )
        beam_vector = self.beam_vector.to(
            device=planar_coords.device, dtype=planar_coords.dtype
        )
        sdet_vec = self.sdet_vec.to(device=planar_coords.device, dtype=planar_coords.dtype)
        fdet_vec = self.fdet_vec.to(device=planar_coords.device, dtype=planar_coords.dtype)

        # vector = distance * beam_vector, broadcast over every pixel/sub-pixel
        start = (distance * beam_vector).expand(planar_coords.shape)

        # "treat detector pixel coordinates as radians"
        angle_about_s = planar_coords[..., 1] / distance  # pixel_pos[2] in C (lab Y)
        angle_about_f = planar_coords[..., 2] / distance  # pixel_pos[3] in C (lab Z)

        rotated = self._rotate_axis(start, sdet_vec, angle_about_s)
        rotated = self._rotate_axis(rotated, fdet_vec, angle_about_f)
        return rotated

    def _compute_curved_pixel_coords(self) -> torch.Tensor:
        """
        Pixel-centre coordinates for a curved (``-curved_det``) detector.

        Equivalent to the planar centres pushed through :meth:`apply_curved_mapping`.
        With over-sampling or sensor layers the simulator instead maps each sub-pixel
        position, as nanoBragg.c does inside its loops.

        Returns:
            torch.Tensor: Pixel coordinates with shape (spixels, fpixels, 3) in meters
        """
        return self.apply_curved_mapping(self._compute_planar_pixel_coords())

    def get_planar_pixel_coords(self) -> torch.Tensor:
        """
        Planar pixel-centre coordinates, ignoring ``curved_detector``.

        The simulator needs these even in curved mode: nanoBragg.c builds the planar
        ``pixel_pos`` (with sub-pixel and sensor-layer offsets) first and only then
        replaces it with the curved one, so the curved mapping has to be applied after
        the offsets, not before.

        Returns:
            torch.Tensor: Pixel coordinates with shape (spixels, fpixels, 3) in meters
        """
        return self._compute_planar_pixel_coords()

    def thickness_layers(self):
        """
        Sensor layers as nanoBragg.c resolves them: ``(n_layers, layer_step_m, mu_per_m)``.

        With a thickness T and ``thicksteps`` N (nanoBragg.c:1583-1637, 1691-1695):

        * N not given: 2 layers, step T/2;
        * N given: at least 2 layers, step T/(N-1), so the layers sit at 0, step, ..., T;
        * no attenuation depth given: mu = 1/T;
        * T <= 0 or attenuation depth 0 ("-detector_abs 0/inf"): one layer, no absorption.

        Layer t starts ``t*step`` behind the front face along odet and absorbs
        exp(-t*step*mu/rho) - exp(-(t+1)*step*mu/rho), with rho = diffracted·odet.
        Returns ``(1, None, None)`` when thickness is not modelled.
        """
        c = self.config
        thick_um = c.detector_thick_um
        if thick_um is None or float(thick_um) <= 0.0:
            return 1, None, None
        if c.detector_abs_um is not None and float(c.detector_abs_um) == 0.0:
            return 1, None, None
        thick_m = torch.as_tensor(thick_um, device=self.device, dtype=self.dtype) * 1e-6
        if c.detector_thicksteps is None or c.detector_thicksteps <= 0:
            n_layers, layer_step_m = 2, thick_m / 2
        else:
            n_layers = max(int(c.detector_thicksteps), 2)
            layer_step_m = thick_m / (n_layers - 1)
        if c.detector_abs_um is None:
            mu = 1.0 / thick_m
        else:
            mu = 1.0 / (torch.as_tensor(c.detector_abs_um, device=self.device, dtype=self.dtype) * 1e-6)
        return n_layers, layer_step_m, mu

    def get_solid_angle(self, pixel_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the solid angle factor for each pixel.

        Per spec section "Pixel mapping and solid angle":
        - Default: Ω = (pixel_size^2 / |pos|^2) · (close_distance/|pos|)
        - With point_pixel: Ω = 1/|pos|^2

        Args:
            pixel_coords: Optional pre-computed pixel coordinates. If None, will compute them.
                         Shape should be (spixels, fpixels, 3) in meters.

        Returns:
            torch.Tensor: Solid angle factor for each pixel, shape (spixels, fpixels)
        """
        # Get pixel coordinates if not provided
        if pixel_coords is None:
            pixel_coords = self.get_pixel_coords()

        # Calculate distance from sample to each pixel
        R = torch.norm(pixel_coords, dim=-1)  # Shape: (spixels, fpixels)

        if self.config.point_pixel:
            # Point pixel mode: use 1/R^2 solid angle only (no obliquity)
            omega = 1.0 / (R * R)
        else:
            # Default mode: include pixel size and obliquity factor
            # Ω = (pixel_size^2 / R^2) · (close_distance/R)
            # where close_distance is the minimum distance along detector normal

            # The obliquity factor is close_distance/R
            # For the standard case, close_distance = distance * cos(angle)
            # where angle is between beam and detector normal
            omega = (self.pixel_size * self.pixel_size) / (R * R) * (self.close_distance / R)

        return omega

    @property
    def beam_vector(self) -> torch.Tensor:
        """
        Get the beam vector based on detector convention.

        For CUSTOM convention with user-supplied custom_beam_vector, use that.
        Otherwise use convention defaults.

        Returns:
            torch.Tensor: Unit beam vector representing incident beam direction (source→sample, along photon propagation)

        Note:
            C-code reference: nanoBragg.c lines 2578, 2989
                source_X[source] = -source_distance*beam_vector[1]
                incident[1] = -source_X[source] = beam_vector[1] * source_distance
            Therefore beam_vector represents the incident beam direction (photon propagation).
        """
        # CUSTOM convention with user override
        if (self.config.detector_convention == DetectorConvention.CUSTOM
            and self.config.custom_beam_vector is not None):
            # Convert tuple to tensor on correct device/dtype
            return torch.tensor(
                self.config.custom_beam_vector,
                device=self.device,
                dtype=self.dtype
            )
        # Convention defaults
        elif self.config.detector_convention in (
            DetectorConvention.MOSFLM,
            DetectorConvention.DENZO,
        ):
            # nanoBragg.c:1193 (MOSFLM) and nanoBragg.c:1208 (DENZO) both set
            # beam_vector = [1,0,0]; DENZO is MOSFLM with a different beam-centre
            # offset (Fbeam = Ybeam + 0.0*pixel instead of + 0.5*pixel).
            return torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        else:
            # XDS, DIALS, ADXV and CUSTOM (without override) conventions use beam along +Z
            return torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)

    def get_r_factor(self) -> torch.Tensor:
        """
        Get the r-factor (ratio) used for distance correction.

        r-factor = dot(beam_vector, rotated_detector_normal)

        Returns:
            torch.Tensor: r-factor value
        """
        if not hasattr(self, 'r_factor'):
            # Calculate on demand if not already computed
            self._calculate_pix0_vector()
        # Ensure r_factor is a tensor with correct dtype
        if not isinstance(self.r_factor, torch.Tensor):
            self.r_factor = torch.tensor(self.r_factor, device=self.device, dtype=self.dtype)
        else:
            # Ensure dtype consistency
            self.r_factor = self.r_factor.to(device=self.device, dtype=self.dtype)
        return self.r_factor

    def get_corrected_distance(self) -> torch.Tensor:
        """
        Get the corrected distance after r-factor calculation.

        distance = close_distance / r-factor

        Returns:
            torch.Tensor: Corrected distance in meters
        """
        if not hasattr(self, 'distance_corrected'):
            # Calculate on demand if not already computed
            self._calculate_pix0_vector()
        # Ensure distance_corrected is a tensor with correct dtype
        if not isinstance(self.distance_corrected, torch.Tensor):
            self.distance_corrected = torch.tensor(self.distance_corrected, device=self.device, dtype=self.dtype)
        else:
            # Ensure dtype consistency
            self.distance_corrected = self.distance_corrected.to(device=self.device, dtype=self.dtype)
        return self.distance_corrected

    def verify_beam_center_preservation(self, tolerance: float = 1e-6) -> Tuple[bool, dict]:
        """
        Verify that the beam center is preserved after detector rotations.

        This implements the acceptance test AT-GEO-003: After all detector transformations,
        the direct beam position should still map to the user-specified beam center.

        For BEAM pivot: R = distance_corrected * beam_vector - pix0_vector
        For SAMPLE pivot: The preservation is verified differently as per spec

        Args:
            tolerance: Tolerance for beam center comparison (in meters)

        Returns:
            tuple: (is_preserved, details_dict) where details_dict contains:
                - 'original_beam_f': Original beam center F component (meters)
                - 'original_beam_s': Original beam center S component (meters)
                - 'computed_beam_f': Computed beam center F after transformations
                - 'computed_beam_s': Computed beam center S after transformations
                - 'error_f': Difference in F component
                - 'error_s': Difference in S component
                - 'max_error': Maximum absolute error
        """
        from ..config import DetectorConvention, DetectorPivot

        # Get beam vector (honours every convention plus CUSTOM overrides)
        beam_vector = self.beam_vector

        # Calculate direct beam position after all transformations
        # For both BEAM and SAMPLE pivots, the formula is the same:
        # R = distance_corrected * beam_vector - pix0_vector
        # This works because for SAMPLE pivot, pix0_vector is already rotated
        R = self.distance_corrected * beam_vector - self.pix0_vector

        # Project onto detector basis vectors to get Fbeam and Sbeam
        Fbeam_computed = torch.dot(R, self.fdet_vec)
        Sbeam_computed = torch.dot(R, self.sdet_vec)

        # Get original beam center in meters
        # CRITICAL [DETECTOR-CONFIG-001]: beam_center_f/s now ALREADY have the +0.5 offset
        # applied in __init__ (lines 104-106 for MOSFLM), so we just convert pixels→meters
        Fbeam_original = self.beam_center_f * self.pixel_size
        Sbeam_original = self.beam_center_s * self.pixel_size

        # Calculate errors
        error_f = abs(Fbeam_computed - Fbeam_original)
        error_s = abs(Sbeam_computed - Sbeam_original)
        max_error = max(error_f, error_s)

        # Check if preserved within tolerance
        is_preserved = max_error < tolerance

        details = {
            'original_beam_f': Fbeam_original.item() if isinstance(Fbeam_original, torch.Tensor) else Fbeam_original,
            'original_beam_s': Sbeam_original.item() if isinstance(Sbeam_original, torch.Tensor) else Sbeam_original,
            'computed_beam_f': Fbeam_computed.item(),
            'computed_beam_s': Sbeam_computed.item(),
            'error_f': error_f.item() if isinstance(error_f, torch.Tensor) else error_f,
            'error_s': error_s.item() if isinstance(error_s, torch.Tensor) else error_s,
            'max_error': max_error.item() if isinstance(max_error, torch.Tensor) else max_error,
        }

        return bool(is_preserved), details

    def _initial_basis_vectors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unrotated (fdet, sdet, odet) for the configured convention, as nanoBragg.c sets them."""
        c = self.config
        from ..config import DetectorConvention

        if c.detector_convention == DetectorConvention.MOSFLM:
            # PHASE 1 FIX: Corrected MOSFLM basis vectors to match C implementation
            fdet_vec = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype  # Fast along +Z (CORRECT)
            )
            sdet_vec = torch.tensor(
                [0.0, -1.0, 0.0], device=self.device, dtype=self.dtype  # Slow along -Y (CORRECT)
            )
            odet_vec = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device, dtype=self.dtype   # Normal along +X (CORRECT)
            )
        elif c.detector_convention == DetectorConvention.XDS:
            # XDS convention: detector surface normal points away from source
            fdet_vec = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device, dtype=self.dtype
            )
            sdet_vec = torch.tensor(
                [0.0, 1.0, 0.0], device=self.device, dtype=self.dtype
            )
            odet_vec = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype
            )
        elif c.detector_convention == DetectorConvention.DIALS:
            # DIALS convention: beam [0,0,1], f=[1,0,0], s=[0,1,0], o=[0,0,1]
            # Similar to XDS but with specific beam and twotheta axis conventions
            fdet_vec = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device, dtype=self.dtype  # Fast along +X
            )
            sdet_vec = torch.tensor(
                [0.0, 1.0, 0.0], device=self.device, dtype=self.dtype  # Slow along +Y
            )
            odet_vec = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype  # Normal along +Z
            )
        elif c.detector_convention == DetectorConvention.ADXV:
            # ADXV convention per spec: beam b = [0 0 1]; f = [1 0 0]; s = [0 -1 0]; o = [0 0 1]
            fdet_vec = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device, dtype=self.dtype  # Fast along +X
            )
            sdet_vec = torch.tensor(
                [0.0, -1.0, 0.0], device=self.device, dtype=self.dtype  # Slow along -Y (like MOSFLM)
            )
            odet_vec = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype  # Normal along +Z (beam direction)
            )
        elif c.detector_convention == DetectorConvention.DENZO:
            # DENZO convention per spec: Same as MOSFLM bases (beam [1,0,0], f=[0,0,1], s=[0,-1,0], o=[1,0,0])
            # Note: Different beam center mapping but same basis vectors as MOSFLM
            fdet_vec = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype  # Fast along +Z (same as MOSFLM)
            )
            sdet_vec = torch.tensor(
                [0.0, -1.0, 0.0], device=self.device, dtype=self.dtype  # Slow along -Y (same as MOSFLM)
            )
            odet_vec = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device, dtype=self.dtype   # Normal along +X (same as MOSFLM)
            )
        elif c.detector_convention == DetectorConvention.CUSTOM:
            # CUSTOM convention uses user-provided vectors or defaults to MOSFLM
            if c.custom_fdet_vector is not None:
                fdet_vec = torch.tensor(
                    c.custom_fdet_vector, device=self.device, dtype=self.dtype
                )
            else:
                # Default to MOSFLM fast vector
                fdet_vec = torch.tensor(
                    [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype
                )

            if c.custom_sdet_vector is not None:
                sdet_vec = torch.tensor(
                    c.custom_sdet_vector, device=self.device, dtype=self.dtype
                )
            else:
                # Default to MOSFLM slow vector
                sdet_vec = torch.tensor(
                    [0.0, -1.0, 0.0], device=self.device, dtype=self.dtype
                )

            if c.custom_odet_vector is not None:
                odet_vec = torch.tensor(
                    c.custom_odet_vector, device=self.device, dtype=self.dtype
                )
            else:
                # Default to MOSFLM normal vector
                odet_vec = torch.tensor(
                    [1.0, 0.0, 0.0], device=self.device, dtype=self.dtype
                )
        else:
            raise ValueError(f"Unknown detector convention: {c.detector_convention}")

        return fdet_vec, sdet_vec, odet_vec

    def _calculate_basis_vectors(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate detector basis vectors from configuration.

        This method dynamically computes the detector's fast, slow, and
        normal basis vectors based on user-provided configuration, such as
        detector rotations (`-detector_rot*`) and the two-theta angle.

        The calculation follows this exact sequence:
        1. Initialize basis vectors according to detector convention (MOSFLM or XDS)
        2. Apply detector rotations in order: X-axis, Y-axis, Z-axis
        3. Apply two-theta rotation around the specified axis (if non-zero)

        All rotations preserve the orthonormality of the basis vectors and
        maintain differentiability when rotation angles are provided as tensors
        with requires_grad=True.

        Note: This method takes no parameters as it uses self.config and
        self.device/dtype. The returned vectors are guaranteed to be on the
        same device and have the same dtype as the detector.

        C-Code Implementation Reference (from nanoBragg.c, lines 1319-1412):
        The C code performs these calculations in a large block within main()
        after parsing arguments. The key operations to replicate are:

        ```c
            /* initialize detector origin from a beam center and distance */
            /* there are two conventions here: mosflm and XDS */
            // ... logic to handle different conventions ...

            if(detector_pivot == SAMPLE){
                printf("pivoting detector around sample\n");
                /* initialize detector origin before rotating detector */
                pix0_vector[1] = -Fclose*fdet_vector[1]-Sclose*sdet_vector[1]+close_distance*odet_vector[1];
                pix0_vector[2] = -Fclose*fdet_vector[2]-Sclose*sdet_vector[2]+close_distance*odet_vector[2];
                pix0_vector[3] = -Fclose*fdet_vector[3]-Sclose*sdet_vector[3]+close_distance*odet_vector[3];

                /* now swing the detector origin around */
                rotate(pix0_vector,pix0_vector,detector_rotx,detector_roty,detector_rotz);
                rotate_axis(pix0_vector,pix0_vector,twotheta_axis,detector_twotheta);
            }
            /* now orient the detector plane */
            rotate(fdet_vector,fdet_vector,detector_rotx,detector_roty,detector_rotz);
            rotate(sdet_vector,sdet_vector,detector_rotx,detector_roty,detector_rotz);
            rotate(odet_vector,odet_vector,detector_rotx,detector_roty,detector_rotz);

            /* also apply orientation part of twotheta swing */
            rotate_axis(fdet_vector,fdet_vector,twotheta_axis,detector_twotheta);
            rotate_axis(sdet_vector,sdet_vector,twotheta_axis,detector_twotheta);
            rotate_axis(odet_vector,odet_vector,twotheta_axis,detector_twotheta);

            /* make sure beam center is preserved */
            if(detector_pivot == BEAM){
                printf("pivoting detector around direct beam spot\n");
                pix0_vector[1] = -Fbeam*fdet_vector[1]-Sbeam*sdet_vector[1]+distance*beam_vector[1];
                pix0_vector[2] = -Fbeam*fdet_vector[2]-Sbeam*sdet_vector[2]+distance*beam_vector[2];
                pix0_vector[3] = -Fbeam*fdet_vector[3]-Sbeam*sdet_vector[3]+distance*beam_vector[3];
            }
        ```

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The calculated
            (fdet_vec, sdet_vec, odet_vec) basis vectors, each with shape (3,)
        """
        from ..utils.geometry import angles_to_rotation_matrix, rotate_axis

        # Get configuration parameters
        c = self.config

        # Convert rotation angles to radians (handling both scalar and tensor inputs)
        detector_rotx = degrees_to_radians(c.detector_rotx_deg)
        detector_roty = degrees_to_radians(c.detector_roty_deg)
        detector_rotz = degrees_to_radians(c.detector_rotz_deg)
        detector_twotheta = degrees_to_radians(c.detector_twotheta_deg)

        # Ensure all angles are tensors for consistent handling
        if not isinstance(detector_rotx, torch.Tensor):
            detector_rotx = torch.tensor(
                detector_rotx, device=self.device, dtype=self.dtype
            )
        elif detector_rotx.device != self.device or detector_rotx.dtype != self.dtype:
            detector_rotx = detector_rotx.to(device=self.device, dtype=self.dtype)

        if not isinstance(detector_roty, torch.Tensor):
            detector_roty = torch.tensor(
                detector_roty, device=self.device, dtype=self.dtype
            )
        elif detector_roty.device != self.device or detector_roty.dtype != self.dtype:
            detector_roty = detector_roty.to(device=self.device, dtype=self.dtype)

        if not isinstance(detector_rotz, torch.Tensor):
            detector_rotz = torch.tensor(
                detector_rotz, device=self.device, dtype=self.dtype
            )
        elif detector_rotz.device != self.device or detector_rotz.dtype != self.dtype:
            detector_rotz = detector_rotz.to(device=self.device, dtype=self.dtype)

        if not isinstance(detector_twotheta, torch.Tensor):
            detector_twotheta = torch.tensor(
                detector_twotheta, device=self.device, dtype=self.dtype
            )
        elif detector_twotheta.device != self.device or detector_twotheta.dtype != self.dtype:
            detector_twotheta = detector_twotheta.to(device=self.device, dtype=self.dtype)

        fdet_vec, sdet_vec, odet_vec = self._initial_basis_vectors()

        # Apply detector rotations (rotx, roty, rotz) using the C-code's rotate function logic
        # The C-code applies rotations in order: X, then Y, then Z
        rotation_matrix = angles_to_rotation_matrix(
            detector_rotx, detector_roty, detector_rotz
        )

        # Apply the rotation matrix to all three basis vectors
        fdet_vec = torch.matmul(rotation_matrix, fdet_vec)
        sdet_vec = torch.matmul(rotation_matrix, sdet_vec)
        odet_vec = torch.matmul(rotation_matrix, odet_vec)

        # Apply two-theta rotation around the specified axis
        if isinstance(c.twotheta_axis, torch.Tensor):
            twotheta_axis = c.twotheta_axis.to(device=self.device, dtype=self.dtype)
        elif c.twotheta_axis is not None:
            twotheta_axis = torch.tensor(
                c.twotheta_axis, device=self.device, dtype=self.dtype
            )
        else:
            # Default to convention-specific axis (set in config.__post_init__)
            # MOSFLM uses [0, 0, -1], XDS uses [1, 0, 0], etc.
            # This should already be set in config, but fallback to MOSFLM default
            twotheta_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=self.dtype)

        # Always apply twotheta rotation to preserve gradients
        # When detector_twotheta is zero, this will be identity
        fdet_vec = rotate_axis(fdet_vec, twotheta_axis, detector_twotheta)
        sdet_vec = rotate_axis(sdet_vec, twotheta_axis, detector_twotheta)
        odet_vec = rotate_axis(odet_vec, twotheta_axis, detector_twotheta)

        return fdet_vec, sdet_vec, odet_vec
