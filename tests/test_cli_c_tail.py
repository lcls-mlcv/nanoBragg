"""CLI behaviour nanoBragg.c defines and the parity matrix can't reach: exact arithmetic, and
runs that need an input file. Oracle tests need NB_C_BIN and skip without it."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nanobrag_torch.__main__ import create_parser, resolve_detector_size, resolve_wavelength_A
from nanobrag_torch.io.source import read_sourcefile


def _c_binary():
    path = os.environ.get("NB_C_BIN")
    if not path or not Path(path).exists():
        pytest.skip("NB_C_BIN is not set")
    return path


COMMON = ("-default_F 100 -cell 70 80 90 75 85 95 -N 5 -distance 100 -oversample 1 -nonoise -nopgm "
          "-lambda 1").split()


def _render(command, args, cwd, env=None):
    float_file = Path(cwd) / "out.bin"
    int_file = Path(cwd) / "out.img"
    float_file.unlink(missing_ok=True)
    done = subprocess.run(command + COMMON + args + ["-floatfile", str(float_file), "-intfile", str(int_file)],
                          cwd=cwd, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    header = int_file.read_bytes()[:512].decode("ascii", "replace")
    size = tuple(int(header.split(f"{key}=")[1].split(";")[0]) for key in ("SIZE1", "SIZE2"))
    return np.fromfile(float_file, dtype=np.float32), size, done


def compare_with_c(args, cwd):
    """Render with C and torch and return (c_image, torch_image, size), which must agree in size."""
    c_image, c_size, _ = _render([_c_binary()], args, cwd)
    torch_image, torch_size, _ = _render([sys.executable, "-m", "nanobrag_torch"], args, cwd)
    assert torch_size == c_size
    return c_image, torch_image, c_size


def assert_matches_c(args, cwd, corr_min=0.9999, sum_tol=2e-4):
    c_image, torch_image, _ = compare_with_c(args, cwd)
    assert np.isfinite(c_image).all()
    assert np.isfinite(torch_image).all()
    assert np.corrcoef(c_image, torch_image)[0, 1] > corr_min
    assert torch_image.sum() / c_image.sum() == pytest.approx(1.0, abs=sum_tol)


@pytest.mark.parametrize("pixel_mm", [0.1, 0.172, 0.05, 0.075])
def test_half_pixel_detsize_counts_as_c_does(pixel_mm):
    """nanoBragg.c:1126: ceil(detsize_f/pixel_size-0.5), both already divided by 1000 at parse."""
    for n in range(1, 3000):
        detsize_mm = (n + 0.5) * pixel_mm
        args = create_parser().parse_args(["-detsize", repr(detsize_mm), "-pixel", repr(pixel_mm)])
        fpixels, spixels, _, _ = resolve_detector_size(args, pixel_mm)
        expected = math.ceil((detsize_mm / 1000.0) / (pixel_mm / 1000.0) - 0.5)
        assert (fpixels, spixels) == (expected, expected), n


def test_half_pixel_detsize_of_one_pixel_matches_c(tmp_path):
    """One pixel, so the parity matrix's correlation is undefined: compare size and value."""
    c_image, torch_image, size = compare_with_c(["-pixel", "0.172", "-detsize", "0.258"], tmp_path)
    assert size == (1, 1)
    assert torch_image[0] == pytest.approx(c_image[0], rel=1e-4)


@pytest.mark.parametrize("flags, wavelength_A", [
    (["-energy", "0"], math.inf),
    (["-energy", "-0"], -math.inf),
    (["-energy", "0", "-lambda", "2"], 2.0),
    (["-lambda", "2", "-energy", "0"], math.inf),
])
def test_energy_zero_divides_as_c_does(flags, wavelength_A):
    """nanoBragg.c:859 divides doubles and the last flag in argv wins."""
    args = create_parser().parse_args(flags)
    args._argv = flags
    assert resolve_wavelength_A(args) == wavelength_A
    del args._argv
    if flags[0] == "-energy":  # without argv, -energy wins
        assert math.isinf(resolve_wavelength_A(args))


DETECTOR = "-pixel 0.5 -detpixels 48".split()


def _sources(tmp_path, text, beam=(1.0, 0.0, 0.0)):
    path = tmp_path / "sources.txt"
    path.write_bytes(text.encode())
    return read_sourcefile(path, default_wavelength_m=1e-10, beam_direction=torch.tensor(beam, dtype=torch.float64),
                           dtype=torch.float64)


def test_nan_in_a_sourcefile_column_is_a_missing_value(tmp_path):
    """nanoBragg.c:2329-2343: NaN marks a value as missing, whichever column it is in."""
    directions, weights, wavelengths = _sources(
        tmp_path, "nan 0 -1 2 1.1e-10\n-1 nan 0 nan 1.1e-10\n-1 0 nan 1 nan\n", beam=(0.0, 0.0, 1.0))
    expected = torch.tensor([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, -10.0]], dtype=torch.float64)
    torch.testing.assert_close(directions, expected / expected.norm(dim=1, keepdim=True))
    assert weights.tolist() == [2.0, 1.0, 1.0]
    assert wavelengths.tolist() == [1.1e-10, 1.1e-10, 1e-10]


@pytest.mark.parametrize("text, extra", [
    ("-1 0 0 1 nan\n-1 0 0 1 1.1e-10\n", []),
    ("nan 0 0 1 1.1e-10\n-1 0 0 1 1e-10\n", []),
    ("nan nan nan 1 1.1e-10\n0 0 -1 1 1e-10\n", ["-adxv"]),
    ("-1 0 0 nan 1.1e-10\n-1 0 0 1 1e-10\n", []),
])
def test_nan_in_a_sourcefile_matches_c(tmp_path, text, extra):
    (tmp_path / "sources.txt").write_text(text)
    assert_matches_c(DETECTOR + ["-sourcefile", "sources.txt"] + extra, tmp_path)


def _torch_run(tmp_path, source_text):
    (tmp_path / "sources.txt").write_text(source_text)
    return _render([sys.executable, "-m", "nanobrag_torch"], DETECTOR + ["-sourcefile", "sources.txt"], tmp_path)


def test_unequal_sourcefile_weights_warn_and_leave_stdout_and_image_alone(tmp_path):
    equal_image, _, equal = _torch_run(tmp_path, "-1 0 0 1 1e-10\n-1 0 0 1 1.1e-10\n")
    assert "use_source_weights" not in equal.stderr

    image, _, unequal = _torch_run(tmp_path, "-1 0 0 9 1e-10\n-1 0 0 1 1.1e-10\n")
    assert "ignores them, as nanoBragg.c does" in unequal.stderr
    assert "ExperimentModel or BeamConfig.use_source_weights" in unequal.stderr
    assert "use_source_weights" not in unequal.stdout
    assert unequal.stdout == equal.stdout
    np.testing.assert_array_equal(image, equal_image)


def test_a_weight_filled_in_by_default_counts_when_comparing_weights(tmp_path):
    _, _, done = _torch_run(tmp_path, "-1 0 0 1 1e-10\n-1 0 0\n")
    assert "use_source_weights" not in done.stderr
    _, _, done = _torch_run(tmp_path, "-1 0 0 2 1e-10\n-1 0 0\n")
    assert "use_source_weights" in done.stderr
