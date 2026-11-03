"""A beam in, or nearly in, the detector plane: C's ratio and distance, and so its oversample and image."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

NB_C_BIN = os.environ.get("NB_C_BIN")
CELL = "-cell 70 80 90 90 90 90 -default_F 100 -N 5 -detpixels 16 -oversample 0 -oversample_polar".split()
ALONG = "-odet_vector 0 1 0 -fdet_vector 0 0 1 -sdet_vector 1 0 0"


def test_ratio_of_exactly_zero_runs_without_c(tmp_path):
    out = tmp_path / "t.bin"
    run = subprocess.run([sys.executable, "-m", "nanobrag_torch", *CELL, *ALONG.split(), "-floatfile", str(out)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "auto-selected 1-fold oversampling" in run.stdout + run.stderr
    assert not np.fromfile(out, dtype=np.float32).any()


@pytest.mark.skipif(not (NB_C_BIN and Path(NB_C_BIN).exists()), reason="NB_C_BIN not set")
@pytest.mark.parametrize("flags", [
    ALONG, ALONG + " -pivot sample", ALONG + " -twotheta 20", ALONG + " -close_distance 80", ALONG + " -point_pixel",
    "-odet_vector 1e-12 1 0 -fdet_vector 0 0 1 -sdet_vector 1 -1e-12 0",
    "-detector_roty 90", "-detector_roty 90 -pivot sample", "-detector_rotx 90", "-twotheta 90 -pivot sample",
])
def test_oversample_and_image_are_cs(flags, tmp_path):
    argv = CELL + flags.split()
    c = subprocess.run([NB_C_BIN, *argv, "-floatfile", "c.bin", "-nopgm", "-nonoise", "-nointfile"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120, check=True)
    run = subprocess.run([sys.executable, "-m", "nanobrag_torch", *argv, "-dtype", "float64", "-floatfile", "t.bin"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    expected = re.search(r"auto-selected (\d+)-fold oversampling", c.stdout).group(0)
    assert expected in run.stdout + run.stderr
    theirs = np.fromfile(tmp_path / "c.bin", dtype=np.float32)
    ours = np.fromfile(tmp_path / "t.bin", dtype=np.float32)
    np.testing.assert_allclose(ours, theirs, rtol=2e-6, atol=1e-25)
