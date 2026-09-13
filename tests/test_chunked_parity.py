"""
pixel_batch_size must not change the image.

The row-chunked path used to run its own copy of the kernel and return before
detector absorption, water background and the pixel trace were applied, so a
three-layer sensor came out several times brighter than the unchunked run.
Both paths now share Simulator._pixel_intensity.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator


def _simulate(pixel_batch_size, water_size_um=0.0, **detector_kwargs):
    detector = DetectorConfig(spixels=40, fpixels=32, pixel_size_mm=0.2, distance_mm=80.0, **detector_kwargs)
    crystal = CrystalConfig(cell_a=60, cell_b=70, cell_c=80, cell_alpha=85, cell_beta=95, cell_gamma=100,
                            misset_deg=(10.0, 20.0, 30.0), N_cells=(5, 5, 5), default_F=100.0)
    beam = BeamConfig(wavelength_A=1.0, water_size_um=water_size_um)
    sim = Simulator(Crystal(crystal, beam, dtype=torch.float64), Detector(detector, dtype=torch.float64),
                    crystal, beam, dtype=torch.float64)
    return sim.run(pixel_batch_size=pixel_batch_size)


@pytest.mark.parametrize(
    "detector_kwargs",
    [
        dict(oversample=1),
        dict(oversample=3, oversample_omega=True),
        dict(oversample=1, detector_thick_um=450.0, detector_abs_um=300.0, detector_thicksteps=3),
        dict(oversample=2, detector_thick_um=450.0, detector_abs_um=300.0, detector_thicksteps=3, oversample_thick=True),
        dict(oversample=1, roi_xmin=4, roi_xmax=20, roi_ymin=6, roi_ymax=30),
        dict(oversample=1, water_size_um=100.0),
    ],
)
@pytest.mark.parametrize("pixel_batch_size", [1, 7, 39])
def test_chunked_equals_unchunked(detector_kwargs, pixel_batch_size):
    full = _simulate(None, **detector_kwargs)
    chunked = _simulate(pixel_batch_size, **detector_kwargs)
    torch.testing.assert_close(chunked, full, rtol=1e-12, atol=0.0)


def _c_binary():
    path = os.environ.get("NB_C_BIN")
    return path if path and Path(path).exists() else None


@pytest.mark.skipif(_c_binary() is None, reason="set NB_C_BIN to a nanoBragg C binary")
@pytest.mark.parametrize(
    "extra",
    [
        "-oversample 2",
        "-detector_thick 450 -detector_abs 300 -thicksteps 3 -oversample 1",
        "-detector_thick 450 -detector_abs 300 -thicksteps 3 -oversample 2 -oversample_thick",
    ],
)
def test_chunked_cli_matches_c(tmp_path, extra):
    """C gets no -pixel_batch_size: its strstr() parser would read it as -pixel."""
    args = ("-default_F 100 -cell 70 80 90 75 85 95 -misset 10 20 30 -lambda 1 -N 5 "
            "-detpixels 96 -pixel 0.1 -distance 100 " + extra).split()
    c_out, py_out = tmp_path / "c.bin", tmp_path / "py.bin"
    subprocess.run([_c_binary(), *args, "-floatfile", str(c_out)], check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "nanobrag_torch", *args, "-pixel_batch_size", "7",
                    "-floatfile", str(py_out)], check=True, capture_output=True)
    c_img = np.fromfile(c_out, dtype=np.float32)
    py_img = np.fromfile(py_out, dtype=np.float32)
    assert np.corrcoef(c_img, py_img)[0, 1] >= 0.9999
    assert py_img.sum() / c_img.sum() == pytest.approx(1.0, abs=1e-3)
