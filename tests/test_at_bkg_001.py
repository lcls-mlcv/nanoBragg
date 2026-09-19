"""
Test for AT-BKG-001: Water background term.

From spec:
- AT-BKG-001 Water background term
  - Setup: -water set to a finite µm; otherwise zero contributions (e.g., default_F=0);
    compute one pixel.
  - Expectation: I_bg = (F_bg^2) · r_e^2 · fluence · (water_size^3) · 1e6 · Avogadro / water_MW
    with F_bg = 2.57, Avogadro = 6.02214179e23, water_MW = 18.

nanoBragg.c seeds each pixel's accumulator with this value (I = I_bg at nanoBragg.c:2665)
rather than adding it to the finished image, so what reaches the pixel is
I_bg · r_e^2 · fluence / steps · polar · omega · capture_fraction — scaled by r_e^2·fluence a
second time and modulated by the per-pixel corrections. It is therefore not flat across the
detector: omega falls off towards the corners. Image parity against the C binary for water
runs is covered by PARITY-WATER-001 in tests/parity_cases.yaml.
"""

import torch
import pytest
from src.nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from src.nanobrag_torch.models.crystal import Crystal
from src.nanobrag_torch.models.detector import Detector
from src.nanobrag_torch.simulator import Simulator


class TestAT_BKG_001:
    """Test suite for AT-BKG-001: Water background term calculation."""

    def test_water_background_calculation(self):
        """Test that water background adds expected constant to all pixels."""
        # Create minimal configuration with water background
        crystal_config = CrystalConfig(
            phi_steps=1,
            mosaic_domains=1,
            N_cells=(1, 1, 1),
            default_F=0.0,  # Zero structure factor to isolate background
        )

        # Create detector with small grid for testing
        detector_config = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            fpixels=10,
            spixels=10,
        )

        # Configure beam with water background
        water_size_um = 10.0  # 10 micrometers
        beam_config = BeamConfig(
            wavelength_A=6.2,
            water_size_um=water_size_um,
        )

        # Create objects
        crystal = Crystal(crystal_config, device="cpu", dtype=torch.float64)
        detector = Detector(detector_config, device="cpu", dtype=torch.float64)

        # Create simulator
        simulator = Simulator(
            crystal=crystal,
            detector=detector,
            crystal_config=crystal_config,
            beam_config=beam_config,
            device="cpu",
            dtype=torch.float64,
        )

        # Run simulation
        image = simulator.run()

        # Calculate expected background value
        F_bg = 2.57
        Avogadro = 6.02214179e23  # mol^-1
        water_MW = 18.0  # g/mol
        r_e_sqr = 7.94079248018965e-30  # m^2
        fluence = simulator.fluence
        water_size_m = water_size_um * 1e-6

        expected_I_bg = (
            F_bg * F_bg
            * r_e_sqr
            * fluence
            * (water_size_m ** 3)
            * 1e6  # Unit inconsistency factor from spec
            * Avogadro
            / water_MW
        )

        # C runs I_bg through the same tail as the Bragg terms: r_e^2*fluence/steps and the
        # per-pixel solid angle (polarization is 1 along the beam for the default Kahn factor
        # only, so compute it explicitly here).
        from src.nanobrag_torch.utils.physics import polarization_factor

        steps = crystal_config.phi_steps * crystal_config.mosaic_domains  # 1 source, oversample 1
        coords = detector.get_pixel_coords()
        omega = detector.get_solid_angle(coords)
        diffracted = coords / torch.linalg.norm(coords, dim=-1, keepdim=True)
        polar = polarization_factor(
            torch.tensor(beam_config.polarization_factor, dtype=torch.float64),
            simulator.incident_beam_direction.expand_as(diffracted).reshape(-1, 3),
            diffracted.reshape(-1, 3),
            torch.tensor(beam_config.polarization_axis, dtype=torch.float64),
        ).reshape(coords.shape[:2])

        expected = expected_I_bg * r_e_sqr * fluence / steps * polar * omega
        assert torch.allclose(image, expected, rtol=1e-9), (image[0, 0].item(), expected[0, 0].item())

        # and it is not flat: the corners see a smaller solid angle than the centre
        assert image.max() > image.min() * (1 + 1e-6)

    def test_water_background_zero(self):
        """Test that zero water size produces no background."""
        # Create minimal configuration without water background
        crystal_config = CrystalConfig(
            phi_steps=1,
            mosaic_domains=1,
            N_cells=(1, 1, 1),
            default_F=0.0,  # Zero structure factor
        )

        # Create detector
        detector_config = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            fpixels=10,
            spixels=10,
        )

        # Configure beam with NO water background
        beam_config = BeamConfig(
            wavelength_A=6.2,
            water_size_um=0.0,  # No water
        )

        # Create objects
        crystal = Crystal(crystal_config, device="cpu", dtype=torch.float64)
        detector = Detector(detector_config, device="cpu", dtype=torch.float64)

        # Create simulator
        simulator = Simulator(
            crystal=crystal,
            detector=detector,
            crystal_config=crystal_config,
            beam_config=beam_config,
            device="cpu",
            dtype=torch.float64,
        )

        # Run simulation
        image = simulator.run()

        # Check that all pixels are zero (no background, no diffraction)
        assert torch.allclose(image, torch.zeros_like(image), atol=1e-20)

    def test_water_background_additive(self):
        """Test that water background adds to existing diffraction pattern."""
        # Create configuration with both diffraction and background
        crystal_config = CrystalConfig(
            phi_steps=1,
            mosaic_domains=1,
            N_cells=(1, 1, 1),
            default_F=100.0,  # Non-zero structure factor
        )

        # Create detector
        detector_config = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            fpixels=10,
            spixels=10,
        )

        # Configure beam with water background
        water_size_um = 5.0
        beam_config_with_water = BeamConfig(
            wavelength_A=6.2,
            water_size_um=water_size_um,
        )

        beam_config_no_water = BeamConfig(
            wavelength_A=6.2,
            water_size_um=0.0,
        )

        # Create objects
        crystal = Crystal(crystal_config, device="cpu", dtype=torch.float64)
        detector = Detector(detector_config, device="cpu", dtype=torch.float64)

        # Create simulators with and without water
        simulator_with_water = Simulator(
            crystal=crystal,
            detector=detector,
            crystal_config=crystal_config,
            beam_config=beam_config_with_water,
            device="cpu",
            dtype=torch.float64,
        )

        simulator_no_water = Simulator(
            crystal=crystal,
            detector=detector,
            crystal_config=crystal_config,
            beam_config=beam_config_no_water,
            device="cpu",
            dtype=torch.float64,
        )

        # Run simulations
        image_with_water = simulator_with_water.run()
        image_no_water = simulator_no_water.run()

        # The background enters before the common scaling, so the difference between the
        # two images is exactly the background-only image (same geometry, same steps).
        F_bg = 2.57
        Avogadro = 6.02214179e23
        water_MW = 18.0
        r_e_sqr = 7.94079248018965e-30
        fluence = simulator_with_water.fluence
        water_size_m = water_size_um * 1e-6

        expected_I_bg = (
            F_bg * F_bg
            * r_e_sqr
            * fluence
            * (water_size_m ** 3)
            * 1e6
            * Avogadro
            / water_MW
        )

        # The difference is the background as it reaches the image: scaled by
        # r_e^2*fluence/steps and modulated by polarization and solid angle, exactly as a
        # background-only run produces it.
        from src.nanobrag_torch.config import BeamConfig as _BeamConfig

        background_only_crystal = CrystalConfig(
            phi_steps=crystal_config.phi_steps,
            mosaic_domains=crystal_config.mosaic_domains,
            N_cells=crystal_config.N_cells,
            default_F=0.0,
        )
        background_only = Simulator(
            crystal=Crystal(background_only_crystal, device="cpu", dtype=torch.float64),
            detector=detector,
            crystal_config=background_only_crystal,
            beam_config=beam_config_with_water,
            device="cpu",
            dtype=torch.float64,
        ).run()

        difference = image_with_water - image_no_water
        assert torch.allclose(difference, background_only, rtol=1e-9)

        # it scales with water_size^3 and is well below the un-scaled I_bg, which C would
        # only produce if the background skipped the common scaling
        assert difference.mean() < expected_I_bg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])