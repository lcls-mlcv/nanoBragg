"""-dtype float64 is double from the flags to the image, so float pixels land within rounding of C."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nanobrag_torch.__main__ import build_simulation, create_parser
from nanobrag_torch.utils.units import degrees_to_radians, radians_to_degrees

REPO = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), MKL_THREADING_LAYER="GNU", NANOBRAG_DISABLE_COMPILE="1")
BASE = "-cell 70 80 90 90 90 90 -default_F 100 -N 5 -detpixels 64 -oversample 1 -seed 3".split()

# Per-source and per-subpixel polarization are the ruling, so runs that have several of either
# ask for -oversample_polar, or drop polarization, which puts torch and C on the same formula.
CASES = {
    "plain": "",
    "triclinic": "-cell 70 85 95 80 95 105",
    "misset": "-misset 10 20 30",
    "rotated_detector": "-detector_rotx 1 -detector_roty 2 -detector_rotz 3 -twotheta 3",
    "phi": "-phi 10 -osc 1 -phisteps 3",
    "mosaic": "-mosaic 0.1 -mosaic_dom 5",
    "divergence": "-hdivrange 0.5 -hdivsteps 3 -oversample_polar",
    "dispersion": "-dispersion 1 -dispsteps 3",
    "all_of_it": "-cell 70 85 95 80 95 105 -hdivrange 0.5 -hdivsteps 3 -phi 5 -osc 2 -phisteps 3 "
                 "-mosaic 0.2 -mosaic_dom 3 -misset 3 4 5 -detector_rotx 1 -oversample 2 -oversample_polar",
}


def oracle():
    c_bin = os.environ.get("NB_C_BIN")
    if not c_bin or not Path(c_bin).exists():
        pytest.skip("set NB_C_BIN to the oracle binary")
    return str(Path(c_bin).resolve())


def test_cli_models_take_the_dtype():
    args = create_parser().parse_args(BASE + "-dtype float64".split())
    bundle = build_simulation(args)
    assert bundle.crystal.dtype == torch.float64 and bundle.detector.dtype == torch.float64
    assert abs(bundle.crystal.a[1].item()) < 1e-13  # float32 gives -3e-6 here


def test_scalar_degrees_are_converted_in_double():
    """A tensor built inside the conversion would be float32, the default dtype."""
    assert degrees_to_radians(1.0) == 1.0 / (180.0 * (1.0 / math.pi))
    assert degrees_to_radians(3.3) == pytest.approx(math.radians(3.3), rel=1e-15)
    assert radians_to_degrees(0.3) == pytest.approx(math.degrees(0.3), rel=1e-15)


@pytest.mark.parametrize("name", sorted(CASES))
def test_float_image_is_c_within_rounding(name, tmp_path):
    args = BASE + CASES[name].split() + ["-floatfile", "f.bin"]
    images = {}
    for label, cmd in (("c", [oracle()]), ("py", [sys.executable, "-m", "nanobrag_torch", "-dtype", "float64"])):
        (tmp_path / label).mkdir()
        subprocess.run(cmd + args, cwd=tmp_path / label, env=ENV, capture_output=True, text=True, timeout=300,
                       check=True)
        images[label] = np.fromfile(tmp_path / label / "f.bin", dtype=np.float32).astype(np.float64)
    c_image, py_image = images["c"], images["py"]
    bright = np.abs(c_image) > 1e-3 * np.abs(c_image).max()
    assert bright.sum() > 10
    # Both are double images rounded to float32 on the way out, so a pixel can sit one float32
    # ulp apart, 1.2e-7 at worst
    assert (np.abs(c_image - py_image)[bright] / np.abs(c_image)[bright]).max() < 1.3e-7
