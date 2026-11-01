"""The noise image against nanoBragg.c: the same float image and seed give the same bytes."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nanobrag_torch.utils.noise import c_noise_image

REPO_ROOT = Path(__file__).parents[1]
ARGS = ("-cell 70 80 90 75 85 95 -default_F 100 -N 5 -lambda 1 -distance 100 -pixel 0.1 -detpixels 96 "
        "-oversample 1 -nopgm -floatfile f.bin -intfile i.img -noisefile n.img").split()
# Photons per pixel run from under 1 to over 1e6 across these, reaching all three branches of poidev
CASES = {
    "faint": "-seed 7",
    "bright": "-seed 7 -fluence 1e27",
    "very-bright": "-seed 7 -fluence 1e30",
    "overloaded": "-seed 7 -fluence 1e33",
    "seed-zero": "-seed 0 -fluence 1e27",
    "adc": "-seed 3 -adc 10",
    "roi": "-seed 3 -roi 10 40 20 70 -fluence 1e27",
}


def c_bin():
    path = os.environ.get("NB_C_BIN")
    if not path or not Path(path).exists():
        pytest.skip("set NB_C_BIN to the oracle binary")
    return str(Path(path).resolve())


def run_c(tmp_path, flags):
    subprocess.run([c_bin()] + ARGS + flags.split(), cwd=tmp_path, check=True, capture_output=True, timeout=300)
    pixels = np.fromfile(tmp_path / "f.bin", dtype=np.float32).reshape(96, 96)
    noise = np.frombuffer((tmp_path / "n.img").read_bytes()[512:], dtype=np.uint16).reshape(96, 96)
    return pixels, noise


@pytest.mark.parametrize("flags", CASES.values(), ids=CASES.keys())
def test_noise_of_cs_float_image_is_identical(flags, tmp_path):
    words = flags.split()
    seed = int(words[words.index("-seed") + 1])
    adc = float(words[words.index("-adc") + 1]) if "-adc" in words else 40.0
    roi = None
    if "-roi" in words:
        at = words.index("-roi")
        roi = tuple(int(x) for x in words[at + 1:at + 5])
    pixels, c_noise = run_c(tmp_path, flags)
    noise, _, _ = c_noise_image(pixels, seed, adc, roi)
    assert np.array_equal(noise, c_noise), f"{(noise != c_noise).sum()} of {noise.size} pixels differ"


def test_all_three_poidev_branches_are_covered(tmp_path):
    bright = run_c(tmp_path, CASES["overloaded"])[0]
    assert (bright < 12).any() and ((bright >= 12) & (bright <= 1e6)).any() and (bright > 1e6).any()


def test_photons_and_overloads_match_cs_report(tmp_path):
    result = subprocess.run([c_bin()] + ARGS + CASES["overloaded"].split(), cwd=tmp_path, check=True,
                            capture_output=True, text=True, timeout=300)
    line = next(l for l in result.stdout.splitlines() if "photons on noise image" in l)
    pixels = np.fromfile(tmp_path / "f.bin", dtype=np.float32).reshape(96, 96)
    _, photons, overloads = c_noise_image(pixels, 7, 40.0)
    assert line == f"{photons:.0f} photons on noise image ({overloads} overloads)"
    assert overloads > 0


def test_cli_noise_image_uses_cs_generator(tmp_path):
    """Not byte-identical end to end: torch's float image differs from C's in the fifth digit."""
    c_dir, py_dir = tmp_path / "c", tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()
    _, c_noise = run_c(c_dir, "-seed 7")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), KMP_DUPLICATE_LIB_OK="TRUE", NANOBRAGG_DISABLE_COMPILE="1")
    output = subprocess.run([sys.executable, "-m", "nanobrag_torch", "-dtype", "float64"] + ARGS + ["-seed", "7"],
                            cwd=py_dir, env=env, check=True, capture_output=True, text=True, timeout=300).stdout
    assert "photons on noise image" in output
    noise = np.frombuffer((py_dir / "n.img").read_bytes()[512:], dtype=np.uint16).reshape(96, 96)
    assert (noise == c_noise).mean() > 0.99
