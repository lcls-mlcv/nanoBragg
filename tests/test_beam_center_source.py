"""
DETECTOR-CONFIG-001: Beam Center Source Tracking Tests

Tests for the beam_center_source attribute. The spec (spec-a-core.md §72) applied the
MOSFLM +0.5 pixel offset only to auto-calculated defaults; nanoBragg.c:1200 applies it
to every MOSFLM beam centre, and the C-first ruling makes C the reference.
"""

import pytest
import torch
from src.nanobrag_torch.config import DetectorConfig, DetectorConvention
from src.nanobrag_torch.models.detector import Detector


class TestBeamCenterSource:
    """Test beam_center_source tracking and conditional MOSFLM offset application."""

    def test_mosflm_auto_calculated_applies_offset(self):
        """Test that MOSFLM convention applies +0.5 pixel offset for auto-calculated defaults.

        Per spec-a-core.md §72 and arch.md §ADR-03, the MOSFLM +0.5 pixel offset
        is part of the default beam center mapping formula and should be applied
        when beam centers are derived from detector size defaults.
        """
        # Setup: MOSFLM convention with auto-calculated beam centers
        config = DetectorConfig(
            detector_convention=DetectorConvention.MOSFLM,
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=256,
            fpixels=256,
            beam_center_s=12.8,  # 256 * 0.1 / 2 = 12.8mm (detector center)
            beam_center_f=12.8,
            beam_center_source="auto",  # Explicitly mark as auto-calculated
        )

        detector = Detector(config)

        # Expected: beam_center in pixels = (12.8 / 0.1) + 0.5 = 128.5 pixels
        expected_pixels = 128.5

        assert abs(detector.beam_center_s.item() - expected_pixels) < 0.01, \
            f"MOSFLM auto beam_center_s should be 128.5, got {detector.beam_center_s.item()}"
        assert abs(detector.beam_center_f.item() - expected_pixels) < 0.01, \
            f"MOSFLM auto beam_center_f should be 128.5, got {detector.beam_center_f.item()}"

    def test_mosflm_explicit_gets_offset_like_c(self):
        """MOSFLM adds +0.5 pixel to explicit beam centres too, as nanoBragg.c does.

        nanoBragg.c:1200-1201 sets Fbeam = Ybeam + 0.5*pixel_size and
        Sbeam = Xbeam + 0.5*pixel_size for every MOSFLM run, whether -Xbeam/-Ybeam
        were given or defaulted. DETECTOR-CONFIG-001 specified no offset for explicit
        values; C wins over the spec (C-first ruling), so beam_center_source does not
        change the mapping.
        """
        # Setup: MOSFLM convention with explicit beam centers
        config = DetectorConfig(
            detector_convention=DetectorConvention.MOSFLM,
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=256,
            fpixels=256,
            beam_center_s=12.8,  # User explicitly provides 12.8mm
            beam_center_f=12.8,
            beam_center_source="explicit",  # User-provided value
        )

        detector = Detector(config)

        # Expected: (12.8 / 0.1) + 0.5 = 128.5 pixels, same as the auto case
        expected_pixels = 128.5

        assert abs(detector.beam_center_s.item() - expected_pixels) < 0.01, \
            f"MOSFLM explicit beam_center_s should be 128.5, got {detector.beam_center_s.item()}"
        assert abs(detector.beam_center_f.item() - expected_pixels) < 0.01, \
            f"MOSFLM explicit beam_center_f should be 128.5, got {detector.beam_center_f.item()}"

    def test_xds_no_offset_regardless_of_source(self):
        """Test that XDS convention never applies +0.5 pixel offset, regardless of source.

        The +0.5 pixel offset is specific to MOSFLM convention. XDS and other conventions
        should use beam centers as-is for both auto-calculated and explicit values.
        """
        # Test both auto and explicit sources with XDS convention
        for source in ["auto", "explicit"]:
            config = DetectorConfig(
                detector_convention=DetectorConvention.XDS,
                distance_mm=100.0,
                pixel_size_mm=0.1,
                spixels=256,
                fpixels=256,
                beam_center_s=12.8,
                beam_center_f=12.8,
                beam_center_source=source,
            )

            detector = Detector(config)

            # Expected: beam_center in pixels = 12.8 / 0.1 = 128.0 pixels (no offset)
            expected_pixels = 128.0

            assert abs(detector.beam_center_s.item() - expected_pixels) < 0.01, \
                f"XDS beam_center_s should be 128.0 for source={source}, got {detector.beam_center_s.item()}"
            assert abs(detector.beam_center_f.item() - expected_pixels) < 0.01, \
                f"XDS beam_center_f should be 128.0 for source={source}, got {detector.beam_center_f.item()}"

    def test_explicit_matches_default_value(self):
        """An explicit centre equal to C's MOSFLM default maps like the default.

        C's default is Xbeam = (detsize + pixel)/2 (nanoBragg.c:1191), and the +0.5
        pixel is then added in both cases (nanoBragg.c:1200), so explicit and auto
        must land on the same pixel.
        """
        # Calculate what would be the MOSFLM default for this detector
        detector_size = 256
        pixel_size = 0.1
        # MOSFLM default formula: (detsize + pixel) / 2
        mosflm_default = (detector_size * pixel_size + pixel_size) / 2  # 12.85mm

        # User provides this exact value explicitly
        config = DetectorConfig(
            detector_convention=DetectorConvention.MOSFLM,
            distance_mm=100.0,
            pixel_size_mm=pixel_size,
            spixels=detector_size,
            fpixels=detector_size,
            beam_center_s=mosflm_default,
            beam_center_f=mosflm_default,
            beam_center_source="explicit",  # Source flag overrides value heuristic
        )

        detector = Detector(config)

        # Expected: 12.85 / 0.1 + 0.5 = 129.0 pixels
        expected_pixels = mosflm_default / pixel_size + 0.5

        assert abs(detector.beam_center_s.item() - expected_pixels) < 0.01, \
            f"Explicit beam_center_s should match provided value {expected_pixels}, got {detector.beam_center_s.item()}"
        assert abs(detector.beam_center_f.item() - expected_pixels) < 0.01, \
            f"Explicit beam_center_f should match provided value {expected_pixels}, got {detector.beam_center_f.item()}"

        # The source flag must not change the mapping: auto gives the same pixel
        config_auto = DetectorConfig(
            detector_convention=DetectorConvention.MOSFLM,
            distance_mm=100.0,
            pixel_size_mm=pixel_size,
            spixels=detector_size,
            fpixels=detector_size,
            beam_center_s=mosflm_default,
            beam_center_f=mosflm_default,
            beam_center_source="auto",
        )
        detector_auto = Detector(config_auto)
        assert abs(detector.beam_center_s.item() - detector_auto.beam_center_s.item()) < 1e-9
        assert abs(detector.beam_center_f.item() - detector_auto.beam_center_f.item()) < 1e-9

    def test_default_beam_center_source_is_auto(self):
        """Test that beam_center_source defaults to 'auto' for backward compatibility.

        When beam_center_source is not specified, it should default to "auto" to
        preserve existing behavior for code that doesn't set this field explicitly.
        """
        # Create config without specifying beam_center_source
        config = DetectorConfig(
            detector_convention=DetectorConvention.MOSFLM,
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=256,
            fpixels=256,
            beam_center_s=12.8,
            beam_center_f=12.8,
            # beam_center_source not specified - should default to "auto"
        )

        # Verify default value
        assert config.beam_center_source == "auto", \
            f"beam_center_source should default to 'auto', got {config.beam_center_source}"

        detector = Detector(config)

        # With default "auto", MOSFLM should apply +0.5 offset
        expected_pixels = 128.5  # (12.8 / 0.1) + 0.5

        assert abs(detector.beam_center_s.item() - expected_pixels) < 0.01, \
            f"Default auto beam_center_s should be 128.5, got {detector.beam_center_s.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
