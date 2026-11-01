"""The noise image against nanoBragg.c: the same float image and seed give the same bytes."""

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from nanobrag_torch import __main__ as cli
from nanobrag_torch.utils.noise import c_noise_image, fast_noise_image, generate_poisson_noise

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


ENV = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), KMP_DUPLICATE_LIB_OK="TRUE", NANOBRAGG_DISABLE_COMPILE="1")
# What reaches the noise image through the CLI: every line here is a run of both programs
CLI_CASES = {
    "seed-zero": "-seed 0",
    "seed-zero-dim": "-seed 0 -fluence 1e27",
    "negative-seed": "-seed -5",  # 14303 photons in C
    "negative-seed-other-table": "-seed -5 -mosaic_seed 99",
    "negative-seed-positive-mosaic-seed": "-seed -5 -mosaic_seed -3",
    "negative-seed-mosaic-domains": "-seed -7 -mosaic 0.5 -mosaic_dom 3 -fluence 1e27",
    "seed-is-IM": "-seed 2147483647 -fluence 1e27",  # 118 photons in C
    "seed-is-minus-IM": "-seed -2147483647 -fluence 1e27",
    "seed-wraps-as-int": "-seed 2147483648",
    "adc-wraps": "-seed 9 -adc -5",
    "adc-fractional": "-seed 9 -adc -5.7",
    "adc-wraps-twice": "-seed 9 -adc -70000",
    "adc-beyond-int": "-seed 9 -adc -3e9",
    "fluence-1e70": "-seed 7 -fluence 1e70",
}


def run_torch(cwd, flags, *cli_flags):
    return subprocess.run([sys.executable, "-m", "nanobrag_torch", *cli_flags] + ARGS + flags.split(), cwd=cwd, env=ENV,
                          check=True, capture_output=True, text=True, timeout=300).stdout


def noise_pixels(path):
    return np.frombuffer(path.read_bytes()[512:], dtype=np.uint16).reshape(96, 96)


def photon_line(stdout):
    return next(line for line in stdout.splitlines() if "photons on noise image" in line)


@pytest.mark.parametrize("flags", CLI_CASES.values(), ids=CLI_CASES.keys())
def test_cli_noise_image_is_cs_byte_for_byte(flags, tmp_path):
    c_dir, py_dir = tmp_path / "c", tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()
    c_out = subprocess.run([c_bin()] + ARGS + flags.split(), cwd=c_dir, check=True, capture_output=True, text=True,
                           timeout=300).stdout
    py_out = run_torch(py_dir, flags, "-dtype", "float64")
    assert (py_dir / "n.img").read_bytes()[512:] == (c_dir / "n.img").read_bytes()[512:]
    assert photon_line(py_out) == photon_line(c_out)


def test_cli_noise_image_at_a_fluence_float32_cannot_hold(tmp_path):
    """1e45 is an image, not an inf: r_e^2*fluence is 8e15. C draws from pixels torch matches to 1e-7."""
    c_dir, py_dir = tmp_path / "c", tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()
    flags = "-seed 7 -fluence 1e45"
    c_out = subprocess.run([c_bin()] + ARGS + flags.split(), cwd=c_dir, check=True, capture_output=True, text=True,
                           timeout=300).stdout
    py_out = run_torch(py_dir, flags)
    assert (noise_pixels(py_dir / "n.img") == noise_pixels(c_dir / "n.img")).mean() > 0.99
    assert float(photon_line(py_out).split()[0]) == pytest.approx(float(photon_line(c_out).split()[0]), rel=1e-5)
    assert photon_line(py_out).split("(")[1] == photon_line(c_out).split("(")[1]


def test_cli_seed_zero_is_a_seed(tmp_path):
    """0 is not "no seed": two runs agree, and differ from a run seeded from the clock."""
    for name in ["a", "b", "clock"]:
        (tmp_path / name).mkdir()
    run_torch(tmp_path / "a", "-seed 0")
    run_torch(tmp_path / "b", "-seed 0")
    run_torch(tmp_path / "clock", "")
    assert (tmp_path / "a" / "n.img").read_bytes() == (tmp_path / "b" / "n.img").read_bytes()
    assert (tmp_path / "a" / "n.img").read_bytes() != (tmp_path / "clock" / "n.img").read_bytes()


def test_omitted_seed_is_c_minus_time():
    """C starts from seed = -time and reloads the table, so -mosaic_seed can't reach the image."""
    with mock.patch.object(sys, "argv", ["nanoBragg"] + ARGS):
        args = cli.create_parser().parse_args()
    args._argv = ARGS
    with mock.patch.object(cli.time, "time", return_value=1234.5):
        seed = cli.build_simulation(args).config["seed"]
    assert seed == 1234  # in the sign of the flag, so C's -1234
    image = np.full((8, 8), 3.0)
    assert np.array_equal(c_noise_image(image, seed)[0], c_noise_image(image, seed, mosaic_seed=99)[0])


def test_nan_and_inf_pixels_are_drawn_as_c_does():
    image = np.array([[float("nan"), 5.0, float("inf"), 3.0]])
    noise, photons, overloads = c_noise_image(image, 1)
    # NaN is not an overload and becomes pixel 0, whatever it did to the running sum
    assert noise[0, 0] == 0 and noise[0, 1] >= 40 and noise[0, 3] >= 40
    assert np.isnan(photons) and overloads <= 1


def test_a_large_negative_pixel_draws_zero():
    noise, photons, overloads = c_noise_image(np.array([[-1e30, -1000.0, -1.0, -float("inf")]]), 1)
    assert noise.tolist() == [[40, 40, 40, 40]] and photons == 0 and overloads == 0


def test_a_draw_past_float32_range_is_inf_not_a_crash():
    noise, photons, overloads = c_noise_image(np.array([[3.4e38] * 32]), 3)
    assert overloads == 32 and noise.min() == 65535 and photons > 1e39


@pytest.mark.parametrize("mean", [0.1, 1.0, 5.0, 11.9, 12.0, 12.1, 100.0, 1000.0, 3e5, 5e5, 1000001.0, 2e6, 1e7],
                         ids=lambda mean: f"{mean:g}")
def test_c_noise_image_is_poisson(mean):
    """
    Mean and variance within 5 standard errors, so the port is right and not just like C.

    A negative adc keeps the draws of a bright pixel on the image: the 16-bit clip is what
    would otherwise hide them, and the pixel is the draw plus adc.
    """
    n = 100000
    adc = 30000.0 - mean if mean > 30000 else 0.0
    noise, photons, _ = c_noise_image(np.full((1, n), mean, dtype=np.float32), 12345, adc)
    draws = noise.astype(np.float64).ravel() - adc
    if adc:
        assert noise.min() > 0 and noise.max() < 65535  # nothing wrapped or clipped
    assert abs(draws.mean() - mean) < 5 * (mean / n) ** 0.5
    assert abs(draws.var() / mean - 1) < 5 * ((2 + 1 / mean) / n) ** 0.5
    assert abs(photons / n - draws.mean()) <= 1  # the pixel truncates what the Gaussian branch draws


@pytest.mark.parametrize("mean", [7e5, 999999.0], ids=lambda mean: f"{mean:g}")
def test_c_noise_image_keeps_cs_excess_variance_below_1e6(mean):
    """The one regime that isn't Poisson. Pinned, since the docstring states it and C needs it."""
    n = 100000
    noise, _, _ = c_noise_image(np.full((1, n), mean, dtype=np.float32), 12345, 30000.0 - mean)
    draws = noise.astype(np.float64).ravel() - (30000.0 - mean)
    assert abs(draws.mean() - mean) < 5 * (1.2 * mean / n) ** 0.5
    assert 1.05 < draws.var() / mean < 1.25


def fast_noise(tmp_path, flags, name="a"):
    folder = tmp_path / name
    folder.mkdir()
    stdout = run_torch(folder, flags + " -fast_noise")
    return noise_pixels(folder / "n.img"), np.fromfile(folder / "f.bin", dtype=np.float32).reshape(96, 96), stdout


def test_cli_fast_noise_is_seeded_and_not_cs_stream(tmp_path):
    first, _, _ = fast_noise(tmp_path, "-seed 7 -fluence 1e30", "a")
    again, _, _ = fast_noise(tmp_path, "-seed 7 -fluence 1e30", "b")
    other, _, _ = fast_noise(tmp_path, "-seed 8 -fluence 1e30", "c")
    (tmp_path / "d").mkdir()
    run_torch(tmp_path / "d", "-seed 7 -fluence 1e30")
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert not np.array_equal(first, noise_pixels(tmp_path / "d" / "n.img"))


def test_cli_fast_noise_is_poisson_about_the_float_image(tmp_path):
    noise, pixels, stdout = fast_noise(tmp_path, "-seed 7 -fluence 1e30 -adc 0")
    assert pixels.max() < 60000 and (pixels > 10).sum() > 500
    bright = pixels > 10
    z = (noise[bright] - pixels[bright]) / np.sqrt(pixels[bright])
    assert abs(z.mean()) < 5 / z.size ** 0.5
    assert abs(z.var() - 1) < 5 * (2 / z.size) ** 0.5
    assert photon_line(stdout) == f"{noise.sum():.0f} photons on noise image (0 overloads)"


def test_cli_fast_noise_honours_roi_and_adc_as_the_c_path_does(tmp_path):
    noise, _, _ = fast_noise(tmp_path, "-seed 3 -roi 10 40 20 70 -fluence 1e27 -adc 11.7")
    outside = np.ones(noise.shape, dtype=bool)
    outside[20:71, 10:41] = False
    assert (noise[outside] == 0).all()
    assert noise[~outside].min() == 11  # 0 photons + 11.7, truncated as C does


def test_cli_fast_noise_wraps_negative_pixels_as_c_does(tmp_path):
    noise, _, _ = fast_noise(tmp_path, "-seed 3 -adc -5 -fluence 1e26")
    assert (noise == 65531).mean() > 0.9 and noise.min() >= 65531


def test_cli_fast_noise_survives_inf_pixels(tmp_path):
    noise, pixels, stdout = fast_noise(tmp_path, "-seed 7 -fluence 1e70")
    infinite = np.isinf(pixels)
    assert 0 < infinite.sum() < pixels.size
    # inf + inf*N(0, 1), as poidev: an overload, or NaN and so pixel 0
    assert set(np.unique(noise[infinite])) == {0, 65535}
    assert "nan photons" in photon_line(stdout)


def test_fast_noise_nan_and_negative_pixels():
    image = torch.tensor([[float("nan"), -7.0, float("inf"), 100.0]])
    noise, photons, overloads = fast_noise_image(image, 5, 40.0)
    assert noise[0, 0] == 0 and noise[0, 1] == 40 and noise[0, 3] > 40 and np.isnan(photons)
    limited, _ = generate_poisson_noise(image, seed=5, adc_offset=40.0, readout_noise=0.0)
    assert limited[0, 0] == 0 and limited[0, 1] == 40
