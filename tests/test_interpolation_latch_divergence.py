"""
A known, deliberate divergence from C: the interpolation latch.

nanoBragg.c keeps `interpolate` as one shared scalar. The first sample whose fractional
h,k,l falls within two indices of the Fhkl box edge clears it, and every later sample in
that run - the rest of the image - is then looked up nearest-neighbour instead. When the
detector corners leave the box, pixel (0,0) usually trips it, so C's `-interpolate` image
comes out bit-identical to its `-nointerpolate` image.

We do not reproduce that. The latch depends on evaluation order, which a batched
implementation does not have (and which would be racy in an OpenMP build of C), so torch
keeps interpolating every in-range sample. The two agree exactly whenever no sample is out
of range, which is what PARITY-INTERP-001 covers.

This test pins the divergence with numbers so it stays visible and cannot drift unnoticed.
If someone later decides to emulate the latch, this test should be replaced by a parity
case, not deleted quietly.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
HKL = REPO / "tests" / "golden_data" / "P1_interp.hkl"

# A detector wide enough that its corners leave the [-6, 6] Fhkl box of P1_interp.hkl.
ARGS = (
    f"-hkl {HKL} -dumpfile /dev/null -cell 50 60 70 90 90 90 -misset 10 20 30 "
    "-lambda 1 -pixel 0.1 -distance 100 -detpixels 128 -oversample 1 -default_F 0 -N 5"
).split()


def _c_binary():
    path = os.environ.get("NB_C_BIN")
    return path if path and Path(path).exists() else None


@pytest.mark.skipif(_c_binary() is None, reason="set NB_C_BIN to a nanoBragg C binary")
def test_c_latches_interpolation_off_and_torch_does_not(tmp_path):
    c_interp, c_plain, py_interp = (tmp_path / n for n in ("ci.bin", "cp.bin", "pi.bin"))
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "NANOBRAGG_DISABLE_COMPILE": "1"}

    subprocess.run([_c_binary(), *ARGS, "-interpolate", "-floatfile", str(c_interp)],
                   check=True, capture_output=True, cwd=tmp_path)
    subprocess.run([_c_binary(), *ARGS, "-nointerpolate", "-floatfile", str(c_plain)],
                   check=True, capture_output=True, cwd=tmp_path)
    subprocess.run([sys.executable, "-m", "nanobrag_torch", *ARGS, "-interpolate",
                    "-floatfile", str(py_interp)], check=True, capture_output=True,
                   cwd=tmp_path, env={**env, "PYTHONPATH": str(REPO / "src")})

    ci = np.fromfile(c_interp, dtype=np.float32).astype(np.float64)
    cp = np.fromfile(c_plain, dtype=np.float32).astype(np.float64)
    pi = np.fromfile(py_interp, dtype=np.float32).astype(np.float64)

    # C latched: asking for interpolation gave the nearest-neighbour image.
    assert np.array_equal(ci, cp), "C no longer latches interpolation off; revisit this divergence"

    # torch keeps interpolating the in-range samples, so it differs from C here.
    r = np.corrcoef(ci, pi)[0, 1]
    assert 0.95 < r < 0.9999, f"divergence moved: r = {r:.6f} (was ~0.988)"
