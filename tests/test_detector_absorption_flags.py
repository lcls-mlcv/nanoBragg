"""-detector_abs 0 / inf and -detector_thick, in argv order, as nanoBragg.c reads them.

C (nanoBragg.c:762-773): -detector_abs with "inf" in its value, or one atof() reads as 0, sets
detector_thick = 0 and detector_mu = 0; any other value sets mu = 1/(value*1e-6); -detector_thick
sets the thickness. So -detector_abs 0 then -detector_thick 100 is a transparent 100 um sensor,
which captures nothing: C's image is zero. torch's CLI turned "0"/"inf" into an attenuation depth
of 0, a perfect absorber, the opposite.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = "-cell 70 80 90 90 90 90 -default_F 100 -N 5 -detpixels 16 -distance 100 -pixel 0.5 -oversample 1 -nopgm -nonoise"
CASES = [
    "-detector_thick 100 -detector_abs 0",
    "-detector_abs 0 -detector_thick 100",
    "-detector_abs inf -detector_thick 100",
    "-detector_abs 0.0 -detector_thick 100",
    "-detector_absorb infinity -detector_thick 100 -detector_thicksteps 3",
    "-detector_thick 100 -detector_abs inf",
    "-detector_thick 100 -detector_abs 50",
    "-detector_abs 50 -detector_thick 100",
    "-detector_thick 100",
]


@pytest.mark.parametrize("flags", CASES)
def test_image_is_cs(flags, tmp_path):
    c_bin = os.environ.get("NB_C_BIN")
    if not c_bin or not Path(c_bin).exists():
        pytest.skip("set NB_C_BIN to the oracle binary")
    argv = BASE.split() + flags.split()
    subprocess.run([str(Path(c_bin).resolve())] + argv + ["-floatfile", "c.bin"], cwd=tmp_path,
                   capture_output=True, text=True, timeout=300, check=True)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    done = subprocess.run([sys.executable, "-m", "nanobrag_torch"] + argv + ["-floatfile", "torch.bin"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    c = np.fromfile(tmp_path / "c.bin", dtype=np.float32)
    torch_image = np.fromfile(tmp_path / "torch.bin", dtype=np.float32)
    if not c.any():
        assert not torch_image.any()
    else:
        np.testing.assert_allclose(torch_image, c, rtol=1e-4, atol=1e-6 * c.max())
