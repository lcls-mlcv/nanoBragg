"""C picks tricubic interpolation from the -N flags, before -samplesize has set Na."""

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), MKL_THREADING_LAYER="GNU", NANOBRAGG_DISABLE_COMPILE="1")
BASE = "-hkl ../x.hkl -cell 50 50 50 90 90 90 -detpixels 32 -pixel 0.3 -oversample 1 -distance 100 -nopolar".split()


@pytest.mark.parametrize("flags", [
    "-samplesize 0.0005",
    "-N 10 -samplesize 0.0005",
    "-N 1 -samplesize 0.0005",
    "-samplesize 0.0005 -nointerpolate",
    "-sample_x 0.0005 -sample_y 0.0005 -sample_z 0.0005 -Na 10 -Nb 2 -Nc 10",
])
def test_image_is_cs(flags, tmp_path):
    c_bin = os.environ.get("NB_C_BIN")
    if not c_bin or not Path(c_bin).exists():
        pytest.skip("set NB_C_BIN to the oracle binary")
    r = random.Random(3)
    with open(tmp_path / "x.hkl", "w") as hkl:
        for h in range(-9, 10):
            for k in range(-9, 10):
                for l in range(-9, 10):
                    hkl.write(f"{h} {k} {l} {r.uniform(1, 500):.2f}\n")
    args = BASE + flags.split() + ["-floatfile", "f.bin"]
    images = {}
    for label, cmd in (("c", [str(Path(c_bin).resolve())]),
                       ("py", [sys.executable, "-m", "nanobrag_torch", "-dtype", "float64"])):
        (tmp_path / label).mkdir()
        subprocess.run(cmd + args, cwd=tmp_path / label, env=ENV, capture_output=True, text=True, timeout=120, check=True)
        images[label] = np.fromfile(tmp_path / label / "f.bin", dtype=np.float32).astype(np.float64)
    assert images["c"].max() > 0
    assert np.abs(images["c"] - images["py"]).max() <= 2e-7 * images["c"].max()
