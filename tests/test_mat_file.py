"""A -mat file as C reads it: the first nine numbers, and lambda 1 A if no flag gave one."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), MKL_THREADING_LAYER="GNU", NANOBRAGG_DISABLE_COMPILE="1")
A_MATRIX = "0.01 0.002 0.001\n-0.001 0.012 0.002\n0.002 -0.001 0.009\n"
# A real MOSFLM file goes on with missets, the U matrix and the cell
TAIL = ("0.0 0.0 0.0\n0.8 0.2 0.1\n-0.1 0.9 0.2\n0.2 -0.1 0.9\n"
        "90.0 90.0 90.0 90.0 90.0 90.0\n0.0 0.0 0.0\n")


@pytest.mark.parametrize("text", [A_MATRIX, A_MATRIX + TAIL], ids=["nine_numbers", "real_file"])
@pytest.mark.parametrize("flags", ["-lambda 1.3", ""], ids=["lambda", "default_lambda"])
def test_image_is_cs(text, flags, tmp_path):
    c_bin = os.environ.get("NB_C_BIN")
    if not c_bin or not Path(c_bin).exists():
        pytest.skip("set NB_C_BIN to the oracle binary")
    (tmp_path / "x.mat").write_text(text)
    args = ("-mat ../x.mat -default_F 100 -N 3 -detpixels 32 -oversample 1 -distance 100 -nopolar "
            f"-floatfile f.bin {flags}").split()
    images = {}
    for label, cmd in (("c", [str(Path(c_bin).resolve())]),
                       ("py", [sys.executable, "-m", "nanobrag_torch", "-dtype", "float64"])):
        (tmp_path / label).mkdir()
        subprocess.run(cmd + args, cwd=tmp_path / label, env=ENV, capture_output=True, text=True, timeout=120, check=True)
        images[label] = np.fromfile(tmp_path / label / "f.bin", dtype=np.float32).astype(np.float64)
    assert images["c"].max() > 0
    assert np.abs(images["c"] - images["py"]).max() <= 2e-7 * images["c"].max()


def test_too_few_numbers_is_an_error(tmp_path):
    from nanobrag_torch.io.mosflm import read_mosflm_matrix

    (tmp_path / "x.mat").write_text("0.01 0.002 0.001\n")
    with pytest.raises(ValueError, match="9 numeric values"):
        read_mosflm_matrix(str(tmp_path / "x.mat"), 1.0)
