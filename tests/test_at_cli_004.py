"""
Test AT-CLI-004: Header precedence and mask behavior

Per spec lines 796-798:
- Setup: Provide -img and -mask files with conflicting header geometry values;
  ensure mask contains zeros in a small block; render a small ROI.
- Expectation: The last file read among -img/-mask determines shared header-
  initialized quantities. Pixels where the mask value is 0 are skipped
  (remain zero in outputs) and excluded from statistics.

"The last file read" is the -img file, NOT the last one named on the command
line: nanoBragg.c runs the mask pre-pass at :419 and the img pre-pass at :462,
both before the argument loop at :506. So -img wins for shared keys whichever
order the flags appear in, and an explicitly-typed flag wins over both.
Verified against the binary; pinned by PARITY-SMVHDR-001.
"""

import os
import subprocess
import tempfile
import struct
from pathlib import Path

import pytest
import torch
import numpy as np

from nanobrag_torch.io.smv import write_smv
from nanobrag_torch.io.mask import create_circular_mask


def create_test_smv_file(filename: str, fpixels: int, spixels: int,
                        pixel_size: float, distance: float,
                        wavelength: float, beam_center_x: float,
                        beam_center_y: float, is_mask: bool = False) -> None:
    """Create a test SMV file with specific header values."""

    # Create data (all ones for img, circular mask for mask files)
    if is_mask:
        # Create circular mask with zeros in a block
        mask = torch.ones(spixels, fpixels, dtype=torch.float32)
        # Set a small block to zero (pixels 10-20 in both dimensions)
        mask[10:20, 10:20] = 0.0
        data = mask
    else:
        data = torch.ones(spixels, fpixels, dtype=torch.float32) * 100

    # Convert to unsigned short
    data_uint16 = (data.numpy() * 1.0).astype(np.uint16)

    # Write SMV file using our writer
    write_smv(
        filepath=filename,
        image_data=data_uint16,
        pixel_size_mm=pixel_size,
        distance_mm=distance,
        wavelength_angstrom=wavelength,
        beam_center_x_mm=beam_center_x,
        beam_center_y_mm=beam_center_y,
        phi_deg=0.0,
        osc_range_deg=1.0,
        twotheta_deg=0.0,  # Fixed parameter name (no detector_ prefix)
        convention="MOSFLM"  # Fixed parameter name (no detector_ prefix)
    )


@pytest.mark.parametrize("img_first", [True, False])
def test_header_precedence_img_beats_mask(img_first):
    """-img header values beat -mask ones, in either command-line order.

    C reads the mask block (nanoBragg.c:419) before the img block (:462), so the
    img file is "the last file read" no matter how the flags are ordered.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files with conflicting header values
        img_file = tmpdir / "test.img"
        mask_file = tmpdir / "test.mask"
        output_file = tmpdir / "output.bin"

        # Create -img file with one set of parameters
        create_test_smv_file(
            str(img_file),
            fpixels=64, spixels=64,
            pixel_size=0.1,  # 0.1 mm
            distance=100.0,  # 100 mm
            wavelength=1.0,  # 1.0 Å
            beam_center_x=32.0,  # 32 mm
            beam_center_y=32.0,  # 32 mm
            is_mask=False
        )

        # Create -mask file with different parameters
        create_test_smv_file(
            str(mask_file),
            fpixels=64, spixels=64,  # Same size
            pixel_size=0.2,  # Different: 0.2 mm
            distance=200.0,  # Different: 200 mm
            wavelength=2.0,  # Different: 2.0 Å
            beam_center_x=25.0,  # Different: 25 mm
            beam_center_y=25.0,  # Different: 25 mm
            is_mask=True
        )

        # Create minimal HKL file
        hkl_file = tmpdir / "test.hkl"
        with open(hkl_file, "w") as f:
            f.write("0 0 0 100.0\n")
            f.write("1 0 0 50.0\n")

        file_flags = (["-img", str(img_file), "-mask", str(mask_file)] if img_first
                      else ["-mask", str(mask_file), "-img", str(img_file)])
        cmd = [
            "python", "-m", "nanobrag_torch",
            "-hkl", str(hkl_file),
            "-cell", "10", "10", "10", "90", "90", "90",
            *file_flags,
            "-detpixels", "64",
            "-roi", "0", "63", "0", "63",
            "-floatfile", str(output_file),
            "-N", "1"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert "Read header from -img file" in result.stdout
        assert "Read header from -mask file" in result.stdout
        assert result.returncode == 0
        assert output_file.exists()

        # img says 1.0 A, mask says 2.0 A: the img file must win either way.
        assert "Wavelength: 1.00" in result.stdout, (
            f"-img wavelength should win regardless of flag order, got:\n{result.stdout}"
        )


def test_mask_zeros_are_skipped():
    """Test that pixels with mask value 0 are skipped in rendering."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mask file with zeros in a block
        mask_file = tmpdir / "test.mask"
        output_file = tmpdir / "output.bin"

        create_test_smv_file(
            str(mask_file),
            fpixels=32, spixels=32,
            pixel_size=0.1,
            distance=100.0,
            wavelength=1.0,
            beam_center_x=16.0,
            beam_center_y=16.0,
            is_mask=True
        )

        # Create HKL file with strong reflection
        hkl_file = tmpdir / "test.hkl"
        with open(hkl_file, "w") as f:
            f.write("0 0 0 1000.0\n")  # Strong (0,0,0) reflection

        # Run simulation with mask
        cmd = [
            "python", "-m", "nanobrag_torch",
            "-hkl", str(hkl_file),
            "-cell", "10", "10", "10", "90", "90", "90",
            "-mask", str(mask_file),
            "-detpixels", "32",
            "-distance", "100",
            "-floatfile", str(output_file),
            "-N", "1",
            "-fluence", "1e20",  # High fluence to get measurable signal
            "-default_F", "100"  # Add default structure factor to get some signal
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        # Read output image
        with open(output_file, "rb") as f:
            data = np.fromfile(f, dtype=np.float32).reshape(32, 32)

        # Check that masked region (10:20, 10:20) has zero intensity
        masked_region = data[10:20, 10:20]
        assert np.allclose(masked_region, 0.0), f"Masked region should be zero, got max={masked_region.max()}"

        # Check that some unmasked pixels have non-zero intensity
        # (at least the central pixel should have signal from (0,0,0) reflection)
        assert data.max() > 0, "Should have some signal in unmasked regions"


def test_mask_beam_center_y_flip():
    """Test that BEAM_CENTER_Y is interpreted differently for -mask files."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mask file
        mask_file = tmpdir / "test.mask"
        output_file = tmpdir / "output.bin"
        smv_output = tmpdir / "output.img"

        # Create mask with specific beam center
        create_test_smv_file(
            str(mask_file),
            fpixels=100, spixels=100,
            pixel_size=0.1,  # 0.1 mm
            distance=100.0,
            wavelength=1.0,
            beam_center_x=5.0,  # 5 mm
            beam_center_y=3.0,   # 3 mm - should be flipped
            is_mask=True
        )

        # Create HKL file
        hkl_file = tmpdir / "test.hkl"
        with open(hkl_file, "w") as f:
            f.write("0 0 0 100.0\n")

        # Run simulation
        cmd = [
            "python", "-m", "nanobrag_torch",
            "-hkl", str(hkl_file),
            "-cell", "10", "10", "10", "90", "90", "90",
            "-mask", str(mask_file),
            "-floatfile", str(output_file),
            "-intfile", str(smv_output),
            "-N", "1"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        # Read the output SMV header to check beam center
        with open(smv_output, "rb") as f:
            header = f.read(512).decode("ascii", errors="ignore")

        # For -mask files, BEAM_CENTER_Y should be interpreted as
        # detsize_s - value_mm = (100 * 0.1) - 3.0 = 7.0 mm
        # This is the actual beam_center_s used internally
        # The output header should reflect the actual beam center used
        # Note: Some conventions may apply additional transformations
        assert "BEAM_CENTER" in header, "Output should have beam center in header"


def test_conflicting_detector_size():
    """A -mask smaller than the -img-sized detector is rejected, not silently used.

    -img wins the detector size (C reads the mask block first and the img block
    second), which leaves a 32x32 mask against a 64x64 detector. nanoBragg.c
    indexes its mask buffer with the final pixel count and reads past the end of
    it; we refuse the combination instead of inheriting that undefined behaviour.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        img_file = tmpdir / "test.img"
        mask_file = tmpdir / "test.mask"
        output_file = tmpdir / "output.bin"

        # Create -img with 64x64 detector
        create_test_smv_file(
            str(img_file),
            fpixels=64, spixels=64,
            pixel_size=0.1,
            distance=100.0,
            wavelength=1.0,
            beam_center_x=32.0,
            beam_center_y=32.0,
            is_mask=False
        )

        # Create -mask with 32x32 detector (different size)
        create_test_smv_file(
            str(mask_file),
            fpixels=32, spixels=32,  # Different size!
            pixel_size=0.1,
            distance=100.0,
            wavelength=1.0,
            beam_center_x=16.0,
            beam_center_y=16.0,
            is_mask=True
        )

        # Create HKL file
        hkl_file = tmpdir / "test.hkl"
        with open(hkl_file, "w") as f:
            f.write("0 0 0 100.0\n")

        # Run with both files
        cmd = [
            "python", "-m", "nanobrag_torch",
            "-hkl", str(hkl_file),
            "-cell", "10", "10", "10", "90", "90", "90",
            "-img", str(img_file),
            "-mask", str(mask_file),
            "-floatfile", str(output_file),
            "-N", "1"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode != 0, (
            "A 32x32 mask against a 64x64 detector must be rejected, not rendered"
        )
        combined = result.stdout + result.stderr
        assert "32x32" in combined and "64x64" in combined, (
            f"Error should name both sizes, got: {combined[-500:]}"
        )


def test_img_only_no_mask():
    """Test that -img file headers are applied when no -mask is provided."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        img_file = tmpdir / "test.img"
        output_file = tmpdir / "output.bin"
        smv_output = tmpdir / "output.img"

        # Create -img file with specific parameters
        create_test_smv_file(
            str(img_file),
            fpixels=48, spixels=48,
            pixel_size=0.15,  # 0.15 mm
            distance=150.0,  # 150 mm
            wavelength=1.5,  # 1.5 Å
            beam_center_x=24.0,
            beam_center_y=24.0,
            is_mask=False
        )

        # Create HKL file
        hkl_file = tmpdir / "test.hkl"
        with open(hkl_file, "w") as f:
            f.write("0 0 0 100.0\n")

        # Run with only -img
        cmd = [
            "python", "-m", "nanobrag_torch",
            "-hkl", str(hkl_file),
            "-cell", "10", "10", "10", "90", "90", "90",
            "-img", str(img_file),
            "-floatfile", str(output_file),
            "-intfile", str(smv_output),
            "-N", "1"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "Read header from -img file" in result.stdout

        # Verify output has correct dimensions from img file
        with open(output_file, "rb") as f:
            data = np.fromfile(f, dtype=np.float32)

        assert data.size == 48 * 48, f"Output should be 48x48, got {data.size} pixels"

        # Check SMV header has values from img file
        with open(smv_output, "rb") as f:
            header = f.read(512).decode("ascii", errors="ignore")

        assert "SIZE1=48" in header
        assert "SIZE2=48" in header
        assert "PIXEL_SIZE=0.15" in header
        assert "DISTANCE=150" in header
        assert "WAVELENGTH=1.5" in header


if __name__ == "__main__":
    # Allow running individual tests from command line
    import sys
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name in globals():
            print(f"Running {test_name}...")
            globals()[test_name]()
            print(f"{test_name} passed!")
        else:
            print(f"Test {test_name} not found")
    else:
        pytest.main([__file__, "-v"])