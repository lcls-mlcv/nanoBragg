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
    # The pixels sit where the header says, and are C's
    assert py_file[1024:] == c_file[len(c_header):]


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
