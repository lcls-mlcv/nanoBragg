"""nanoBragg.c reads -misset_seed N as -misset N <next> <next>. Torch warns, and says what C would do."""

import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from nanobrag_torch.__main__ import warn_c_misset_seed_order

BASE = "-cell 70 80 90 90 90 90 -default_F 100 -N 3 -detpixels 32 -oversample 1 -seed 5".split()


def warned(argv):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_c_misset_seed_order(argv)
    return [str(w.message) for w in caught]


@pytest.mark.parametrize("argv, angles", [
    ("-misset_seed 7 -distance 90 -N 3", "7 0 90"),
    ("-misset random -misset_seed 7 -distance 90 -N 3", "7 0 90"),
    ("-misset 1 2 3 -misset_seed -7 -pixel 0.1", "-7 0 0.1"),
    ("-misset_seed 7 1e2x .5 -N", "7 100 0.5"),
])
def test_warns_with_the_angles_c_would_use(argv, angles):
    messages = warned(argv.split())
    assert len(messages) == 1 and f"-misset {angles}." in messages[0]


@pytest.mark.parametrize("argv", [
    "-N 3 -distance 90",
    "-misset_seed 7 -misset random -distance 90",  # the later -misset replaces it, and this is the use C supports
    "-misset_seed 7 -misset 1 2 3 -distance 90",
    "-misset random -N 3 -misset_seed 7",  # fewer than three arguments follow: C's check fails
    "-misset_seed 7 -N",
])
def test_quiet_where_c_keeps_the_seed(argv):
    assert warned(argv.split()) == []


def test_c_does_what_the_warning_says(tmp_path):
    c_bin = os.environ.get("NB_C_BIN")
    if not c_bin or not Path(c_bin).exists():
        pytest.skip("set NB_C_BIN to the oracle binary")
    images = []
    for label, flags in (("seed", BASE + "-misset_seed 7 -distance 90 -floatfile f.bin".split()),
                         ("misset", BASE + "-distance 90 -misset 7 0 90 -floatfile f.bin".split()),
                         ("plain", BASE + "-distance 90 -floatfile f.bin".split())):
        (tmp_path / label).mkdir()
        subprocess.run([str(Path(c_bin).resolve())] + flags, cwd=tmp_path / label, capture_output=True, check=True)
        images.append(np.fromfile(tmp_path / label / "f.bin", dtype=np.float32))
    assert np.array_equal(images[0], images[1])
    assert not np.array_equal(images[0], images[2])
