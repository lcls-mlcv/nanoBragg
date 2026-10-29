"""
Acceptance test for AT-FLU-001: Fluence calculation.

Per spec: Provide -flux, -exposure, -beamsize, and also -fluence; set beamsize smaller than sample_y/z.
Expectation: fluence SHALL be recomputed as flux·exposure/beamsize^2 whenever flux != 0 and exposure > 0
and beamsize ≥ 0; exposure > 0 SHALL recompute flux consistently.

Sample clipping by the beam is done by the CLI, as in nanoBragg.c:1134-1143, and is covered by
the parity case samplesize-clipped-by-beam in tests/parity_cases.yaml.
"""

import pytest
import torch

from nanobrag_torch.config import BeamConfig


class TestAT_FLU_001:
    """Test fluence calculation per AT-FLU-001."""

    def test_fluence_calculation_from_flux_exposure_beamsize(self):
        """Test fluence is recomputed from flux, exposure, and beamsize."""

        # Test case 1: fluence computed from flux, exposure, beamsize
        config = BeamConfig(
            flux=1e15,  # photons/s
            exposure=0.1,  # seconds
            beamsize_mm=0.5,  # mm
            fluence=1.0  # This should be overridden
        )

        # Expected: fluence = flux * exposure / (beamsize_m)^2
        # beamsize_m = 0.5 / 1000 = 0.0005 m
        # fluence = 1e15 * 0.1 / (0.0005)^2 = 1e14 / 0.00000025 = 4e20
        expected_fluence = 1e15 * 0.1 / (0.0005 * 0.0005)
        assert config.fluence == pytest.approx(expected_fluence)

    def test_no_fluence_calculation_when_flux_zero(self):
        """Test fluence is NOT recomputed when flux is 0."""

        initial_fluence = 1e24
        config = BeamConfig(
            flux=0.0,  # flux is 0
            exposure=1.0,
            beamsize_mm=1.0,
            fluence=initial_fluence
        )

        # Fluence should not be changed
        assert config.fluence == initial_fluence

    def test_no_fluence_calculation_when_exposure_zero(self):
        """Test fluence is NOT recomputed when exposure is 0."""

        initial_fluence = 1e24
        config = BeamConfig(
            flux=1e15,
            exposure=0.0,  # exposure is 0
            beamsize_mm=1.0,
            fluence=initial_fluence
        )

        # Fluence should not be changed
        assert config.fluence == initial_fluence

    def test_fluence_calculation_with_beamsize_zero(self):
        """Test fluence handling when beamsize is exactly 0 but flux and exposure are set."""

        initial_fluence = 1e24
        config = BeamConfig(
            flux=1e15,
            exposure=1.0,
            beamsize_mm=0.0,  # beamsize is 0
            fluence=initial_fluence
        )

        # When beamsize is 0 but flux and exposure are set, fluence should not be recalculated
        # (can't divide by 0)
        assert config.fluence == initial_fluence

    def test_flux_recomputation_from_fluence_and_exposure(self):
        """Test flux is recomputed when exposure > 0 and fluence/beamsize are set."""

        config = BeamConfig(
            flux=0.0,  # This should be recomputed
            exposure=0.5,  # > 0
            beamsize_mm=2.0,  # > 0
            fluence=1e20
        )

        # Expected: flux = fluence * beamsize^2 / exposure
        # beamsize_m = 2.0 / 1000 = 0.002 m
        # flux = 1e20 * (0.002)^2 / 0.5 = 1e20 * 0.000004 / 0.5 = 8e14
        beamsize_m = 0.002
        expected_flux = 1e20 * (beamsize_m * beamsize_m) / 0.5
        assert config.flux == pytest.approx(expected_flux)
