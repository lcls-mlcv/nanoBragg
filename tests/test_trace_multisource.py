"""-trace_pixel with several sources: it runs, and says what C says where they print the same thing."""

import os
import re
import subprocess
import sys

import numpy as np
import pytest

NB_C_BIN = os.environ.get("NB_C_BIN")
needs_c = pytest.mark.skipif(not (NB_C_BIN and os.path.exists(NB_C_BIN)), reason="NB_C_BIN not set")

GEOMETRY = ("-cell 70 80 90 90 90 90 -default_F 100 -N 5 -detpixels 32 -oversample 1 "
            "-hdivrange 0.5 -hdivsteps 3").split()
SLOW, FAST = 20, 12


def torch_run(tmp_path, args, trace=True):
    out = tmp_path / "torch.bin"
    extra = ["-trace_pixel", str(SLOW), str(FAST)] if trace else []
    run = subprocess.run([sys.executable, "-m", "nanobrag_torch", *args, "-dtype", "float64", *extra,
                          "-floatfile", str(out)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    return run.stdout, np.fromfile(out, dtype=np.float32)


def trace_values(stdout, name):
    return [[float(x) for x in line.split()[-3:] if re.fullmatch(r"[-+0-9.e]+", x)]
            for line in stdout.splitlines() if re.match(rf"TRACE_PY: {name} ", line)]


def source_values(stdout, name):
    return [float(m.group(1)) for m in re.finditer(rf"TRACE_PY_SOURCE \d+: {name} (\S+)", stdout)]


def c_printout(tmp_path, args):
    run = subprocess.run([NB_C_BIN, *args, "-printout_pixel", str(FAST), str(SLOW), "-floatfile", "c.bin",
                          "-nopgm", "-nonoise", "-nointfile"], cwd=tmp_path, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    text = run.stdout[run.stdout.index("stol ="):]
    hkl = re.search(r"hkl= (\S+) (\S+) (\S+)", text)
    return {
        "hkl": [float(x) for x in hkl.groups()],
        "F_cell": float(re.search(r"F_cell=(\S+)", text).group(1)),
        "F_latt": float(re.search(r"F_latt=(\S+)", text).group(1)),
        "I_over_steps": float(re.search(r"I/steps\s+(\S+)", text).group(1)),
        "polar": float(re.search(r"polar\s+(\S+)", text).group(1)),
        "omega": float(re.search(r"omega\s+(\S+)", text).group(1)),
        "pixel": float(re.search(r"pixel\s+(\S+)", text).group(1)),
        "a": [float(re.search(rf"{axis}:\s+(\S+)", text).group(1)) for axis in "XYZ"],
    }


def test_trace_keeps_the_image(tmp_path):
    stdout, image = torch_run(tmp_path, GEOMETRY)
    _, untraced = torch_run(tmp_path, GEOMETRY, trace=False)
    assert image.size == 32 * 32
    np.testing.assert_array_equal(image, untraced)
    assert source_values(stdout, "source_index") == [0, 1, 2]


def test_source_contributions_add_up_to_the_pixel(tmp_path):
    stdout, _ = torch_run(tmp_path, GEOMETRY)
    pre = source_values(stdout, "I_contribution_pre_polar")
    post = source_values(stdout, "I_contribution_post_polar")
    assert sum(pre) == pytest.approx(trace_values(stdout, "I_before_scaling_pre_polar")[0][0], rel=1e-12)
    assert sum(post) == pytest.approx(trace_values(stdout, "I_before_scaling_post_polar")[0][0], rel=1e-12)
    for polar, a, b in zip(source_values(stdout, "polar"), pre, post):
        assert polar == pytest.approx(b / a, rel=1e-12)
        assert polar < 1


def test_oversampled_sources_trace(tmp_path):
    args = [("2" if x == "1" and GEOMETRY[i - 1] == "-oversample" else x) for i, x in enumerate(GEOMETRY)]
    stdout, image = torch_run(tmp_path, args)
    assert image.size == 32 * 32
    assert len(source_values(stdout, "polar")) == 3


def test_oversample_polar_undoes_nopolar_in_the_trace(tmp_path):
    """-nopolar only stops polarization being cached (nanoBragg.c:2772)."""
    stdout, _ = torch_run(tmp_path, GEOMETRY + ["-nopolar", "-oversample_polar"])
    assert trace_values(stdout, "oversample_polar")[0][0] == 1
    assert all(p < 1 for p in source_values(stdout, "polar"))
    assert trace_values(stdout, "polar")[0][0] < 1
    stdout, _ = torch_run(tmp_path, GEOMETRY + ["-nopolar"])
    assert all(p == 1 for p in source_values(stdout, "polar"))


@needs_c
def test_sources_against_c(tmp_path):
    stdout, image = torch_run(tmp_path, GEOMETRY)
    c = c_printout(tmp_path, GEOMETRY)
    # C prints I, summed over sources, and h,k,l and F_latt from the last source
    assert sum(source_values(stdout, "I_contribution_pre_polar")) == pytest.approx(c["I_over_steps"] * 3, rel=1e-9)
    assert source_values(stdout, "I_contribution_pre_polar")[-1] == pytest.approx(
        (c["F_cell"] * c["F_latt"]) ** 2, rel=1e-4)
    # C caches polar from the first source; torch polarizes each source, by ruling
    assert source_values(stdout, "polar")[0] == pytest.approx(c["polar"], rel=1e-9)
    assert trace_values(stdout, "omega_pixel")[0][0] == pytest.approx(c["omega"], rel=1e-8)
    assert trace_values(stdout, "rot_a_angstroms")[0][0] == pytest.approx(c["a"][0], rel=1e-8)
    assert image[SLOW * 32 + FAST] == pytest.approx(c["pixel"], rel=2e-5)


@needs_c
def test_single_source_trace_against_c(tmp_path):
    """Real-space vectors are in metres inside the run, and h,k,l came out 1e-10 too small."""
    args = "-cell 70 80 90 90 90 90 -default_F 100 -N 5 -detpixels 32 -oversample 1".split()
    stdout, image = torch_run(tmp_path, args)
    c = c_printout(tmp_path, args)
    np.testing.assert_allclose(trace_values(stdout, "hkl_frac")[0], c["hkl"], rtol=0, atol=1e-6)
    assert trace_values(stdout, "F_latt")[0][0] == pytest.approx(c["F_latt"], rel=1e-5)
    assert trace_values(stdout, "F_cell_nearest")[0][0] == c["F_cell"]
    assert trace_values(stdout, "polar")[0][0] == pytest.approx(c["polar"], rel=1e-9)
    assert image[SLOW * 32 + FAST] == pytest.approx(c["pixel"], rel=1e-7)
