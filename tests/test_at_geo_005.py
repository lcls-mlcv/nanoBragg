"""
Acceptance Test AT-GEO-005: Curved detector mapping

Per spec:
- AT-GEO-005 Curved detector mapping
  - Setup: Enable -curved_det; choose several off-center pixels.
  - Expectation: The curved mapping SHALL yield |pos| equal for all pixels at a
    given Fdet,Sdet angular offset (spherical arc mapping), and differ from planar
    mapping in a way consistent with the spec's small-angle rotations about s and f
    by Sdet/distance and Fdet/distance respectively.
"""

import torch
import pytest
import numpy as np
from src.nanobrag_torch.config import DetectorConfig
from src.nanobrag_torch.models.detector import Detector


class TestATGEO005CurvedDetector:
    """Tests for AT-GEO-005: Curved detector mapping."""

    def test_curved_detector_equal_distance(self):
        """Test that curved mapping yields equal |pos| for all pixels."""
        # Setup: Create detector with curved_detector enabled
        config = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=100,
            fpixels=100,
            beam_center_s=50.0,  # Center of detector
            beam_center_f=50.0,
            curved_detector=True  # Enable curved detector
        )
        detector = Detector(config)

        # Get pixel coordinates
        pixel_coords = detector.get_pixel_coords()

        # For a curved detector, ALL pixels should be at approximately the same distance
        # from the sample (that's what makes it "curved" - it's a spherical surface)
        distances = torch.norm(pixel_coords, dim=-1)

        # Check that all distances are approximately equal to the nominal distance
        expected_distance = config.distance_mm / 1000.0  # Convert to meters

        # Get the min and max distances
        min_dist = distances.min().item()
        max_dist = distances.max().item()

        # The variation should be very small (only due to numerical approximations)
        # For small angle approximations, the error should be on the order of angle^2
        # Maximum angle is approximately detector_size/distance ~ 10/100 = 0.1 rad
        # So relative error should be ~ (0.1)^2 = 0.01 or 1%
        assert np.isclose(min_dist, max_dist, rtol=0.02), \
            f"Curved detector: all pixels should be approximately equidistant. Min: {min_dist}, Max: {max_dist}"

        # The mean distance should be close to the nominal distance
        mean_dist = distances.mean().item()
        assert np.isclose(mean_dist, expected_distance, rtol=0.01), \
            f"Mean distance {mean_dist} should be close to nominal distance {expected_distance}"

    def test_curved_vs_planar_difference(self):
        """Test that curved mapping differs from planar mapping as expected."""
        # Setup: Create two detectors, one planar and one curved
        config_planar = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=100,
            fpixels=100,
            beam_center_s=50.0,
            beam_center_f=50.0,
            curved_detector=False  # Planar detector
        )

        config_curved = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=100,
            fpixels=100,
            beam_center_s=50.0,
            beam_center_f=50.0,
            curved_detector=True  # Curved detector
        )

        detector_planar = Detector(config_planar)
        detector_curved = Detector(config_curved)

        # Get pixel coordinates for both
        coords_planar = detector_planar.get_pixel_coords()
        coords_curved = detector_curved.get_pixel_coords()

        # For curved detector, all pixels should be at the same distance
        # This is the key difference from planar detector
        center_s = 50
        center_f = 50
        center_curved = coords_curved[center_s, center_f]
        dist_center_curved = torch.norm(center_curved).item()

        # Check that center pixel is approximately at the nominal distance
        expected_distance = config_curved.distance_mm / 1000.0  # meters
        assert np.isclose(dist_center_curved, expected_distance, rtol=1e-3), \
            f"Center pixel should be at distance {expected_distance}m, got {dist_center_curved}m"

        # Check off-center pixels - should differ
        # The further from center, the larger the difference
        corner_s = 10
        corner_f = 10
        corner_planar = coords_planar[corner_s, corner_f]
        corner_curved = coords_curved[corner_s, corner_f]

        difference = torch.norm(corner_planar - corner_curved).item()
        assert difference > 1e-6, \
            "Off-center pixels should differ between planar and curved detectors"

        # For curved detector, distance from sample should be constant
        dist_corner_curved = torch.norm(corner_curved).item()
        dist_center_curved = torch.norm(center_curved).item()
        assert np.allclose(dist_corner_curved, dist_center_curved, rtol=1e-3), \
            f"Curved detector: all pixels should be equidistant. Corner: {dist_corner_curved}, Center: {dist_center_curved}"

        # For planar detector, corner pixels are further from sample
        center_planar = coords_planar[center_s, center_f]  # Add missing variable definition
        dist_corner_planar = torch.norm(corner_planar).item()
        dist_center_planar = torch.norm(center_planar).item()
        assert dist_corner_planar > dist_center_planar, \
            "Planar detector: corner pixels should be further from sample than center"

    def test_matches_c_two_rotate_axis_calls(self):
        """The curved mapping must reproduce nanoBragg.c:2707-2716 exactly.

        C does, per sub-pixel::

            vector = distance*beam_vector;
            rotate_axis(vector,    newvector, sdet_vector, pixel_pos[2]/distance);
            rotate_axis(newvector, pixel_pos, fdet_vector, pixel_pos[3]/distance);

        Note the angles are the lab-frame Y and Z components of the *planar* pixel_pos
        (so they include pix0_vector / the beam centre), NOT the in-plane Sdet/Fdet
        offsets, and rotate_axis is the full Rodrigues rotation, not a first-order
        approximation. This test used to encode a first-order approximation driven by
        Sdet/Fdet and a negated beam vector; both were wrong against C.
        """
        config = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=100,
            fpixels=100,
            beam_center_s=50.0,
            beam_center_f=50.0,
            curved_detector=True
        )
        detector = Detector(config)

        coords = detector.get_pixel_coords()
        planar = detector.get_planar_pixel_coords()

        distance = detector.get_corrected_distance()
        beam = detector.beam_vector

        def rotate_axis(v, axis, phi):
            """Literal transcription of nanoBragg.c rotate_axis()."""
            c, s = torch.cos(phi), torch.sin(phi)
            dot = torch.dot(axis, v) * (1.0 - c)
            return axis * dot + v * c + torch.cross(axis, v, dim=0) * s

        # A few off-centre pixels, including one far out on both axes
        for test_s, test_f in [(60, 70), (5, 95), (99, 0), (50, 50)]:
            pixel_pos = planar[test_s, test_f]

            vector = distance * beam
            newvector = rotate_axis(vector, detector.sdet_vec, pixel_pos[1] / distance)
            expected = rotate_axis(newvector, detector.fdet_vec, pixel_pos[2] / distance)

            actual = coords[test_s, test_f]
            assert torch.allclose(actual, expected, atol=1e-12), \
                f"pixel ({test_s},{test_f}): got {actual.tolist()}, C gives {expected.tolist()}"

            # And the defining property: exactly "distance" from the sample
            assert torch.allclose(torch.norm(actual), distance, rtol=1e-12)

    def test_curved_mapping_applies_to_subpixels(self):
        """C maps each sub-pixel, so mapping an offset position != offsetting a mapped one."""
        config = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=64,
            fpixels=64,
            beam_center_s=3.2,
            beam_center_f=3.2,
            curved_detector=True,
        )
        detector = Detector(config)
        planar = detector.get_planar_pixel_coords()

        # A quarter-pixel sub-sample offset along fast
        offset = 0.25 * detector.pixel_size * detector.fdet_vec
        mapped_offset = detector.apply_curved_mapping(planar + offset)

        # Every mapped sub-pixel still sits exactly "distance" from the sample ...
        distance = detector.get_corrected_distance()
        assert torch.allclose(torch.norm(mapped_offset, dim=-1), distance.expand(64, 64), rtol=1e-6)

        # ... and differs from simply shifting the mapped pixel centres.
        naive = detector.get_pixel_coords() + offset
        assert not torch.allclose(mapped_offset, naive, atol=1e-9)

    def test_gradient_flow_curved_detector(self):
        """Test that gradients flow through curved detector mapping."""
        # Setup: Create detector with requires_grad distance
        distance_tensor = torch.tensor(100.0, requires_grad=True)

        config = DetectorConfig(
            distance_mm=distance_tensor,
            pixel_size_mm=0.1,
            spixels=50,  # Smaller for faster test
            fpixels=50,
            beam_center_s=25.0,
            beam_center_f=25.0,
            curved_detector=True
        )
        detector = Detector(config)

        # Get pixel coordinates
        coords = detector.get_pixel_coords()

        # Create a loss that depends on the pixel positions
        # For example, sum of distances from origin
        distances = torch.norm(coords, dim=-1)
        loss = distances.sum()

        # Compute gradient
        loss.backward()

        # Check that distance has a gradient
        assert distance_tensor.grad is not None, \
            "Distance parameter should have gradient"

        # The gradient should be non-zero
        assert torch.abs(distance_tensor.grad) > 1e-10, \
            "Distance gradient should be non-zero"

        # For a curved detector where all pixels are at distance from sample,
        # the gradient should be positive (increasing distance increases sum of distances)
        assert distance_tensor.grad > 0, \
            "Distance gradient should be positive for sum of distances loss"

    def test_beam_center_affects_curvature(self):
        """Test that beam center position affects the curved mapping correctly."""
        # Create two curved detectors with different beam centers
        config1 = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=100,
            fpixels=100,
            beam_center_s=30.0,  # Off-center
            beam_center_f=30.0,
            curved_detector=True
        )

        config2 = DetectorConfig(
            distance_mm=100.0,
            pixel_size_mm=0.1,
            spixels=100,
            fpixels=100,
            beam_center_s=70.0,  # Different off-center position
            beam_center_f=70.0,
            curved_detector=True
        )

        detector1 = Detector(config1)
        detector2 = Detector(config2)

        coords1 = detector1.get_pixel_coords()
        coords2 = detector2.get_pixel_coords()

        # All pixels on both detectors should be at approximately the same distance
        # (they're both on spherical surfaces with the same radius)
        dist1_all = torch.norm(coords1, dim=-1)
        dist2_all = torch.norm(coords2, dim=-1)

        mean_dist1 = dist1_all.mean().item()
        mean_dist2 = dist2_all.mean().item()

        expected_distance = config1.distance_mm / 1000.0  # Same for both

        assert np.isclose(mean_dist1, expected_distance, rtol=0.01), \
            f"Detector 1 mean distance {mean_dist1} should be close to {expected_distance}"

        assert np.isclose(mean_dist2, expected_distance, rtol=0.01), \
            f"Detector 2 mean distance {mean_dist2} should be close to {expected_distance}"

        # The beam centers affect the pix0_vector, which shifts where the detector is positioned
        # But with curved detector, all pixels remain on the spherical surface
        # The two detectors should have different pix0_vectors
        pix0_diff = torch.norm(detector1.pix0_vector - detector2.pix0_vector).item()
        assert pix0_diff > 1e-6, \
            "Different beam centers should result in different pix0_vectors"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])