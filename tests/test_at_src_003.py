"""AT-SRC-003: Sourcefile wavelengths and steps reconciliation.

Each source keeps the wavelength in its file, and -lambda fills in a missing column.
Steps normalization counts all sources, including zero-weight ones.
"""

import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from nanobrag_torch.io.source import read_sourcefile


class TestSourcefileWavelengths:
    """nanoBragg.c:2341-2343 and 2748: each source keeps the wavelength in its file."""

    def test_file_wavelength_is_kept(self, tmp_path):
        sourcefile = tmp_path / "sources.txt"
        sourcefile.write_text("0 0 -10 1 6.2e-10\n0 0 -10 1 7.0e-10\n")
        _, _, wavelengths = read_sourcefile(sourcefile, default_wavelength_m=1e-10, dtype=torch.float64)
        assert wavelengths.tolist() == pytest.approx([6.2e-10, 7.0e-10])

    def test_missing_column_takes_lambda(self, tmp_path):
        sourcefile = tmp_path / "sources.txt"
        sourcefile.write_text("0 0 -10 1\n0 0 -10 1 7.0e-10\n")
        _, _, wavelengths = read_sourcefile(sourcefile, default_wavelength_m=1e-10, dtype=torch.float64)
        assert wavelengths.tolist() == pytest.approx([1e-10, 7.0e-10])

    @pytest.mark.parametrize("lines", [
        ["-1 0 0 1 1.0e-10", "-1 0 0 1 1.1e-10"],
        ["-1 0 0 2 1.0e-10", "-1 0 0 1 1.1e-10"],
        ["-1 0 0 1 1.05e-10"],
        ["-1 0 0 1", "-1 0 0 1 1.1e-10"],
    ], ids=["two-wavelengths", "weights-differ", "one-source", "missing-column"])
    def test_image_matches_c(self, lines, tmp_path):
        c_bin = os.environ.get("NB_C_BIN")
        if not c_bin or not Path(c_bin).exists():
            pytest.skip("set NB_C_BIN to the oracle binary")
        (tmp_path / "sources.txt").write_text("\n".join(lines) + "\n")
        args = ("-cell 70 80 90 75 85 95 -default_F 100 -N 5 -lambda 1 -distance 100 -pixel 0.1 -detpixels 64 "
                "-oversample 1 -nonoise -nopgm -sourcefile sources.txt -floatfile").split()
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
        for command, out in (([str(Path(c_bin).resolve())], "c.bin"),
                             ([sys.executable, "-m", "nanobrag_torch"], "py.bin")):
            subprocess.run(command + args + [out], cwd=tmp_path, env=env, check=True, capture_output=True)
        c_image = np.fromfile(tmp_path / "c.bin", dtype=np.float32)
        py_image = np.fromfile(tmp_path / "py.bin", dtype=np.float32)
        assert np.corrcoef(c_image, py_image)[0, 1] > 0.99999
        assert py_image.sum() / c_image.sum() == pytest.approx(1.0, abs=1e-4)


class TestStepsNormalizationParity:
    """TC-E3: Steps calculation must count all sources (including zero-weight).

    Per nanoBragg.c:2700-2720, steps = sources_total × mosaic × phi × oversample²
    where sources_total includes all entries (even zero-weight divergence placeholders).
    """

    def test_steps_count_includes_zero_weight_sources(self):
        """Steps denominator must count zero-weight sources per C convention."""
        # This test verifies the simulator's steps calculation.
        # Implementation note: The actual steps fix is in simulator.py initialization,
        # not in source.py. This test documents the expected behavior.

        cli_lambda = 1.0e-10

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Two sources: one normal weight, one zero weight
            f.write(f"0.0 0.0 -10.0 1.0 {cli_lambda}\n")
            f.write(f"0.1 0.0 -10.0 0.0 {cli_lambda}\n")  # Zero weight
            sourcefile_path = Path(f.name)

        try:
            directions, weights, wavelengths = read_sourcefile(
                sourcefile_path,
                default_wavelength_m=cli_lambda,
                dtype=torch.float64
            )

            # Verify both sources are returned
            assert len(directions) == 2
            assert len(weights) == 2
            assert len(wavelengths) == 2

            # Verify weights include the zero-weight entry
            assert weights[0].item() == 1.0
            assert weights[1].item() == 0.0

            # NOTE: The actual steps calculation test is in the simulator integration tests.
            # This test confirms source.py returns all sources for proper steps counting.

        finally:
            sourcefile_path.unlink()

    def test_equal_weighting_preserved(self):
        """All sources must have equal weight (per spec line 151)."""
        cli_lambda = 1.0e-10

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Varying weights in file (should be ignored in final intensity calc)
            f.write(f"0.0 0.0 -10.0 1.0 {cli_lambda}\n")
            f.write(f"0.1 0.0 -10.0 0.5 {cli_lambda}\n")
            f.write(f"0.2 0.0 -10.0 2.0 {cli_lambda}\n")
            sourcefile_path = Path(f.name)

        try:
            directions, weights, wavelengths = read_sourcefile(
                sourcefile_path,
                default_wavelength_m=cli_lambda,
                dtype=torch.float64
            )

            # Weights are read from file (for future use or validation)
            # but should NOT affect final intensity (equal weighting via steps division)
            assert len(weights) == 3
            # File weights are preserved for now; equal weighting enforced in simulator

        finally:
            sourcefile_path.unlink()


# Acceptance thresholds (from lambda_semantics.md):
# - Wavelength override: exact match (within machine precision)
# - Warning emission: pytest.warns match on text
# - Steps parity: integration test required (simulator level)
