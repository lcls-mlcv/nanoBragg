"""SMV files the CLI writes, against nanoBragg.c and against the readers that must take them."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nanobrag_torch.io.mask import parse_smv_header, read_smv_header_text, read_smv_mask
from nanobrag_torch.io.smv import read_smv, write_smv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), KMP_DUPLICATE_LIB_OK="TRUE", NANOBRAGG_DISABLE_COMPILE="1")
BASE = "-cell 70 80 90 75 85 95 -default_F 100 -N 2 -lambda 1 -oversample 1 -nopgm -seed 7".split()
FILES = "-floatfile f.bin -intfile i.img -noisefile n.img".split()
# Each of these overflows 512 bytes in C's own format
LONG_HEADERS = {
    "rotated": "-detpixels 16 -pixel 0.1723 -distance 250.5 -detector_rotx 1.1 -detector_roty 0.7 -detector_rotz 0.3 "
               "-twotheta 12.34 -lambda 1.2345",
    "long-numbers": "-detpixels 16 -pixel 0.1339655 -distance 190.4755 -lambda 1.824215 -phi 47.25205 -osc 0.5570575 "
                    "-phisteps 1",
}


def c_binary():
    c_bin = os.environ.get("NB_C_BIN")
    if not c_bin or not Path(c_bin).exists():
        pytest.skip("set NB_C_BIN to the oracle binary")
    return str(Path(c_bin).resolve())


def render(cmd, cwd, args):
    cwd.mkdir(exist_ok=True)
    return subprocess.run(cmd + args, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=300, check=True)


def torch_cmd(*flags):
    return [sys.executable, "-m", "nanobrag_torch", *flags]


def header_text(data):
    return data[:data.index(b"}\f") + 2]


def test_the_reported_command_keeps_its_images(tmp_path):
    """It used to raise ValueError after the render, with the int and noise images lost."""
    args = BASE + FILES + ("-detpixels 2463 -pixel 0.172 -distance 250 -detector_rotx 1.1 -detector_roty 0.7 "
                           "-detector_rotz 0.3 -twotheta 12.34 -roi 0 3 0 3").split()
    render(torch_cmd(), tmp_path / "py", args)
    for name in ["i.img", "n.img"]:
        image, header = read_smv(tmp_path / "py" / name)
        assert header["HEADER_BYTES"] == "1024" and image.shape == (2463, 2463)
        assert image[:4, :4].min() >= 40 and image[4:, 4:].max() == 0


@pytest.fixture(params=sorted(LONG_HEADERS), ids=sorted(LONG_HEADERS))
def long_header_files(request, tmp_path):
    args = BASE + FILES + LONG_HEADERS[request.param].split()
    render([c_binary()], tmp_path / "c", args)
    render(torch_cmd("-dtype", "float64"), tmp_path / "py", args)
    return tmp_path


@pytest.mark.parametrize("name", ["i.img", "n.img"])
def test_long_header_grows_the_block_and_keeps_cs_text_and_pixels(long_header_files, name):
    c_file = (long_header_files / "c" / name).read_bytes()
    py_file = (long_header_files / "py" / name).read_bytes()
    c_header, py_header = header_text(c_file), header_text(py_file)

    assert 512 < len(c_header) < 1024, "this case no longer tests a long header"
    assert b"HEADER_BYTES=512;" in c_header
    # C's text with one edit, and the padding is spaces
    assert py_header == c_header.replace(b"HEADER_BYTES=512;", b"HEADER_BYTES=1024;")
    assert py_file[len(py_header):1024] == b" " * (1024 - len(py_header))
    # The pixels sit where the header says, and are C's, count for count. A one-count difference
    # here used to be put down to libm; it was -dtype float64 building float32 models
    py_pixels = np.frombuffer(py_file[1024:], dtype=np.uint16).astype(int)
    c_pixels = np.frombuffer(c_file[len(c_header):], dtype=np.uint16).astype(int)
    np.testing.assert_array_equal(py_pixels, c_pixels)


def test_torch_readers_find_the_pixels_of_a_long_header(long_header_files):
    path = long_header_files / "py" / "n.img"
    expected = np.frombuffer(path.read_bytes()[1024:], dtype=np.uint16).reshape(16, 16)

    image, header = read_smv(path)
    assert header["HEADER_BYTES"] == "1024" and header["BEAMLINE"] == "fake"
    np.testing.assert_array_equal(image, expected)

    assert parse_smv_header(str(path))["BEAMLINE"] == "fake"
    assert "BEAMLINE=fake;" in read_smv_header_text(str(path))
    mask, mask_header = read_smv_mask(str(path))
    assert mask_header["BEAMLINE"] == "fake"
    torch.testing.assert_close(mask, torch.as_tensor((expected != 0).astype(np.float32)))


def test_fabio_finds_the_pixels_of_a_long_header(long_header_files):
    fabio = pytest.importorskip("fabio")
    path = long_header_files / "py" / "i.img"
    expected = np.frombuffer(path.read_bytes()[1024:], dtype=np.uint16).reshape(16, 16)
    image = fabio.open(str(path))
    assert image.header["HEADER_BYTES"] == "1024"
    np.testing.assert_array_equal(image.data, expected)


def test_c_reads_a_long_header_back(tmp_path):
    """C's reader honours HEADER_BYTES (nanoBragg.c:4176), so the padded file is valid for C too."""
    args = BASE + FILES + LONG_HEADERS["long-numbers"].split()
    render(torch_cmd(), tmp_path / "py", args + ["-roi", "2", "9", "3", "12", "-adc", "7"])
    inside_roi = read_smv(tmp_path / "py" / "i.img")[0] != 0
    assert inside_roi.sum() == 80

    render([c_binary()], tmp_path / "c", args + ["-mask", str(tmp_path / "py" / "i.img")])
    masked = np.fromfile(tmp_path / "c" / "f.bin", dtype=np.float32).reshape(16, 16)
    np.testing.assert_array_equal(masked != 0, inside_roi)


def test_only_a_longer_text_grows_the_block(tmp_path):
    """A text of exactly 512 is C's file. HEADER_BYTES=1024 is itself one byte longer than =512."""
    path = tmp_path / "x.img"

    def header(serial_digits):
        write_smv(path, np.zeros((2, 2)), 0.1, 100.0, 1.0, 1.0, 1.0, detector_sn="0" * serial_digits)
        return path.read_bytes()[:-8]

    shortest = len(header_text(header(0)))
    assert len(header(0)) == 512
    assert len(header_text(header(512 - shortest))) == 512 and len(header(512 - shortest)) == 512
    assert len(header(513 - shortest)) == 1024 and b"HEADER_BYTES=1024;" in header(513 - shortest)
    assert len(header(1023 - shortest)) == 1024
    assert len(header(1024 - shortest)) == 1536 and b"HEADER_BYTES=1536;" in header(1024 - shortest)


@pytest.mark.parametrize("extra", [[], ["-Xbeam", "1.65", "-Ybeam", "2.4"], ["-distance", "123.4", "-pixel", "0.172"]],
                         ids=["defaults", "beam", "distance-pixel"])
def test_write_smv_without_geometry_is_cs_unrotated_mosflm_header(extra, tmp_path):
    render([c_binary()], tmp_path, BASE + "-detpixels 32 -distance 100 -pixel 0.1 -nonoise -intfile i.img -floatfile f.bin".split() + extra)
    c_file = (tmp_path / "i.img").read_bytes()
    _, c_header = read_smv(tmp_path / "i.img")

    pixel = float(c_header["PIXEL_SIZE"])
    write_smv(tmp_path / "lib.img", np.zeros((32, 32)), pixel, float(c_header["DISTANCE"]), 1.0,
              float(c_header["BEAM_CENTER_X"]), float(c_header["BEAM_CENTER_Y"]))
    assert (tmp_path / "lib.img").read_bytes()[:512] == c_file[:512]


def test_readers_keep_the_key_after_dials_origin(tmp_path):
    """DIALS_ORIGIN has no semicolon in nanoBragg.c:3186, and must not swallow XDS_ORGX."""
    render([c_binary()], tmp_path, BASE + "-detpixels 16 -nonoise -intfile i.img -floatfile f.bin".split())
    path = tmp_path / "i.img"
    lines = [line for line in path.read_bytes()[:512].decode("ascii").split("\n") if "=" in line]
    expected = dict(line.rstrip(";").split("=", 1) for line in lines)
    assert expected["XDS_ORGX"] and "," in expected["DIALS_ORIGIN"]

    assert read_smv(path)[1] == expected
    assert parse_smv_header(str(path)) == expected
    assert read_smv_mask(str(path))[1] == expected


@pytest.mark.parametrize("extra", [[], ["-roi", "3", "12", "2", "14"]], ids=["whole-detector", "roi"])
def test_every_file_of_a_masked_run_matches_c(extra, tmp_path):
    """C keeps masked pixels in the int image, at adc, and counts them as zeros in rmsd."""
    args = "-cell 70 80 90 75 85 95 -default_F 100 -N 2 -lambda 1 -detpixels 16 -pixel 0.1 -distance 100 -oversample 1".split()
    mask = np.ones((16, 16))
    mask[4:8, 2:9] = 0
    write_smv(tmp_path / "mask.img", mask, 0.1, 100.0, 1.0, 0.85, 0.85, adc_offset=0.0)
    args += ["-mask", str(tmp_path / "mask.img"), "-nonoise", "-intfile", "i.img", "-floatfile", "f.bin", "-pgmfile", "p.pgm"]
    render([c_binary()], tmp_path / "c", args + extra)
    render(torch_cmd("-dtype", "float64"), tmp_path / "py", args + extra)

    assert (tmp_path / "py" / "p.pgm").read_bytes() == (tmp_path / "c" / "p.pgm").read_bytes()
    c_image = np.frombuffer((tmp_path / "c" / "i.img").read_bytes()[512:], dtype=np.uint16).reshape(16, 16)
    py_image = np.frombuffer((tmp_path / "py" / "i.img").read_bytes()[512:], dtype=np.uint16).reshape(16, 16)
    assert c_image[mask == 0].max() == 40
    # The int image has 55000 counts across a float32 pixel, so the odd one can round apart
    assert np.abs(c_image.astype(int) - py_image.astype(int)).max() <= 1
    assert np.array_equal(c_image[mask == 0], py_image[mask == 0])


@pytest.mark.parametrize("extra", [[], ["-adxv"], ["-adxv", "-Xbeam", "5", "-Ybeam", "7"], ["-denzo"]],
                         ids=lambda extra: " ".join(extra) or "mosflm")
def test_header_keeps_a_detsize_the_pixels_dont_fill(extra, tmp_path):
    """C's detsize_s stays 30.7 with 61 pixels of 0.5, and sets BEAM_CENTER_Y and ADXV_CENTER_Y."""
    args = BASE + "-detsize_f 20.3 -detsize_s 30.7 -pixel 0.5 -distance 100 -nonoise -intfile i.img -floatfile f.bin".split()
    render([c_binary()], tmp_path / "c", args + extra)
    render(torch_cmd(), tmp_path / "py", args + extra)
    assert (tmp_path / "py" / "i.img").read_bytes()[:512] == (tmp_path / "c" / "i.img").read_bytes()[:512]


def test_pgm_of_a_faint_image_matches_c(tmp_path):
    """rmsd sets the PGM's scale, and float32 squares underflow on pixels like these."""
    args = BASE + "-detpixels 32 -pixel 0.1 -distance 100 -default_F 1e5 -fluence 1e-17 -nonoise -pgmfile p.pgm".split()
    args = [word for word in args if word != "-nopgm"]
    render([c_binary()], tmp_path / "c", args + ["-floatfile", "f.bin", "-intfile", "i.img"])
    render(torch_cmd(), tmp_path / "py", args + ["-floatfile", "f.bin", "-intfile", "i.img"])
    c_pgm, py_pgm = (tmp_path / "c" / "p.pgm").read_bytes(), (tmp_path / "py" / "p.pgm").read_bytes()

    def scale(pgm):
        return float(pgm.split(b"\n")[2].split()[-1])

    # float32 pixels this faint round apart in places, so the scale can differ in its last digit
    assert scale(py_pgm) == pytest.approx(scale(c_pgm), rel=1e-5)
    assert len(py_pgm) == len(c_pgm)
    c_pixels, py_pixels = (np.frombuffer(pgm[-32 * 32:], dtype=np.uint8) for pgm in (c_pgm, py_pgm))
    assert (c_pixels == py_pixels).mean() > 0.99 and c_pixels.max() > 100
