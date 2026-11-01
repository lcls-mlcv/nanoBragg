"""Parity of the *files a run produces*, for the same argv, against the C oracle.

Why this exists
---------------
`tests/test_parity_matrix.py` appends `-floatfile <tmp>` to every one of its 171
runs (test_parity_matrix.py, `run_binary`), and no parity case names an output
flag itself. So the suite that certifies this port has never executed the
default output path — the one an actual user hits when they run an existing
nanoBragg command line.

That is how this went unnoticed: nanoBragg.c initialises all four output
filenames (nanoBragg.c:145-150) and writes them whether or not a flag was
given, while torch wrote nothing at all and still exited 0. The same command
line produced four files under C and zero under torch, with no error — a silent
no-op, which is the least debuggable failure a user can get.

This module compares *which files appear*, not just the contents of one file
the harness asked for by name. Contents are checked where they are meant to
agree exactly; the noise image is only checked for existence and size, because
torch uses `torch.poisson` rather than C's `poidev`/`ran1` and is deliberately
not bit-reproducible against C yet.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The four names C hardcodes (nanoBragg.c:145-150).
C_DEFAULT_OUTPUTS = {
    "floatimage.bin",
    "intimage.img",
    "image.pgm",
    "noiseimage.img",
}

# A small, cheap geometry. The point of these cases is the output surface, not
# the physics, so keep the detector tiny and the runs fast.
BASE_ARGS = (
    "-default_F 100 -cell 100 100 100 90 90 90 -lambda 6.2 -N 3 "
    "-pixel 0.1 -distance 100 -detpixels 32 -mosflm -seed 1"
).split()

# (id, extra argv). Each asserts that C and torch leave the same set of files.
CASES = [
    ("defaults-no-output-flags", []),
    ("nopgm", ["-nopgm"]),
    ("nonoise", ["-nonoise"]),
    # C flips these inside the argv loop, so order decides the outcome:
    # naming the file re-enables the output even after the suppressing flag.
    ("nopgm-then-pgmfile", ["-nopgm", "-pgmfile", "custom.pgm"]),
    ("pgmfile-then-nopgm", ["-pgmfile", "custom.pgm", "-nopgm"]),
    ("nonoise-then-noisefile", ["-nonoise", "-noisefile", "custom_noise.img"]),
    ("noisefile-then-nonoise", ["-noisefile", "custom_noise.img", "-nonoise"]),
    ("explicit-floatfile-only", ["-floatfile", "custom_float.bin"]),
    # A single-pixel ROI drives rmsd to 0, which is the PGM auto-exposure's
    # fallback branch. That branch was wrong (it used 1.0 where C uses the
    # resolved integer-image scale) and no case here exercised it.
    ("roi-single-pixel", ["-roi", "17", "17", "17", "17"]),
]


def c_binary():
    c_bin = os.environ.get("NB_C_BIN")
    if not c_bin or not Path(c_bin).exists():
        pytest.skip("set NB_C_BIN to the oracle binary to run output-surface parity")
    return str(Path(c_bin).resolve())


def run_in(cwd: Path, cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=300
    )


def files_written(cwd: Path) -> set:
    return {p.name for p in cwd.iterdir() if p.is_file()}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_output_file_set_matches_c(case, tmp_path):
    """C and torch must leave the same filenames behind for the same argv."""
    case_id, extra = case
    args = BASE_ARGS + extra

    c_dir = tmp_path / "c"
    py_dir = tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()

    c_res = run_in(c_dir, [c_binary()] + args)
    assert c_res.returncode == 0, f"C failed:\n{c_res.stdout}\n{c_res.stderr}"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["NANOBRAGG_DISABLE_COMPILE"] = "1"
    py_res = subprocess.run(
        [sys.executable, "-m", "nanobrag_torch"] + args,
        cwd=str(py_dir), capture_output=True, text=True, timeout=300, env=env,
    )
    assert py_res.returncode == 0, f"torch failed:\n{py_res.stdout}\n{py_res.stderr}"

    c_files = files_written(c_dir)
    py_files = files_written(py_dir)

    assert py_files == c_files, (
        f"{case_id}: output file sets differ for identical argv\n"
        f"  argv:       {' '.join(args)}\n"
        f"  C wrote:    {sorted(c_files) or '(nothing)'}\n"
        f"  torch wrote:{sorted(py_files) or '(nothing)'}\n"
        f"  missing from torch: {sorted(c_files - py_files) or 'none'}\n"
        f"  extra in torch:     {sorted(py_files - c_files) or 'none'}"
    )


def test_unflagged_run_writes_all_four_defaults(tmp_path):
    """The regression this module exists for: a bare run must not be a no-op."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["NANOBRAGG_DISABLE_COMPILE"] = "1"

    res = subprocess.run(
        [sys.executable, "-m", "nanobrag_torch"] + BASE_ARGS,
        cwd=str(tmp_path), capture_output=True, text=True, timeout=300, env=env,
    )
    assert res.returncode == 0, res.stderr

    written = files_written(tmp_path)
    assert written == C_DEFAULT_OUTPUTS, (
        "a run with no output flags must leave C's four default files; "
        f"got {sorted(written) or '(nothing)'}"
    )


@pytest.mark.parametrize("name", ["floatimage.bin", "intimage.img"])
def test_default_output_sizes_match_c(name, tmp_path):
    """Byte sizes of the deterministic outputs must agree with C.

    Limited to the float and integer images. `image.pgm` is checked separately
    and `noiseimage.img` is excluded because torch's noise is not yet
    C-reproducible (`torch.poisson` rather than `poidev`/`ran1`).
    """
    c_dir = tmp_path / "c"
    py_dir = tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()

    assert run_in(c_dir, [c_binary()] + BASE_ARGS).returncode == 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["NANOBRAGG_DISABLE_COMPILE"] = "1"
    assert subprocess.run(
        [sys.executable, "-m", "nanobrag_torch"] + BASE_ARGS,
        cwd=str(py_dir), capture_output=True, text=True, timeout=300, env=env,
    ).returncode == 0

    c_size = (c_dir / name).stat().st_size
    py_size = (py_dir / name).stat().st_size
    assert py_size == c_size, (
        f"{name}: torch wrote {py_size} bytes, C wrote {c_size}"
    )




def test_default_pgm_is_byte_identical_to_c(tmp_path):
    """The PGM preview must match C exactly, header included.

    It is the one default output that can be compared byte for byte: the float
    and integer images carry float32 rounding differences (the parity matrix
    compares those with correlation thresholds), and the noise image uses a
    different RNG. The PGM quantises to 256 grey levels, which is coarse enough
    that the rounding washes out.

    This is the check that caught two real defects at once: the scale was
    hardcoded to 1.0, leaving the default preview about 26x under-exposed
    (max=7 of 255 against C's max=182), and the header printed Python's full
    float repr where C uses %lg.
    """
    c_dir = tmp_path / "c"
    py_dir = tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()

    assert run_in(c_dir, [c_binary()] + BASE_ARGS).returncode == 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["NANOBRAGG_DISABLE_COMPILE"] = "1"
    assert subprocess.run(
        [sys.executable, "-m", "nanobrag_torch"] + BASE_ARGS,
        cwd=str(py_dir), capture_output=True, text=True, timeout=300, env=env,
    ).returncode == 0

    c_bytes = (c_dir / "image.pgm").read_bytes()
    py_bytes = (py_dir / "image.pgm").read_bytes()

    assert py_bytes[:40] == c_bytes[:40], (
        "PGM header differs:\n"
        f"  C:     {c_bytes[:40]!r}\n"
        f"  torch: {py_bytes[:40]!r}"
    )
    assert py_bytes == c_bytes, "PGM pixel data differs from C"


def test_default_pgm_uses_most_of_the_grey_range(tmp_path):
    """The preview should be exposed, not near-black.

    The earlier version of this test asserted merely "not all zero", which
    passed: at scale 1.0 a 32x32 default run still has 756 of 1024 pixels
    nonzero. The real defect is dynamic range -- max=7 out of 255 where C gets
    max=182 -- so assert that instead. C's auto-exposure targets roughly 250 at
    five times the RMSD, so anything under half the range is under-exposed.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["NANOBRAGG_DISABLE_COMPILE"] = "1"
    assert subprocess.run(
        [sys.executable, "-m", "nanobrag_torch"] + BASE_ARGS,
        cwd=str(tmp_path), capture_output=True, text=True, timeout=300, env=env,
    ).returncode == 0

    body = (tmp_path / "image.pgm").read_bytes().split(b"255\n", 1)[1]
    assert max(body) > 127, (
        f"default PGM peaks at {max(body)} of 255 — the preview is "
        f"under-exposed because the auto-scale is not applied (C uses "
        f"250/(5*rmsd) and reaches 182 on this run)"
    )


@pytest.mark.parametrize(
    "extra,label",
    [([], "defaults"), (["-roi", "17", "17", "17", "17"], "roi-single-pixel"),
     (["-scale", "3"], "explicit-scale")],
    ids=["defaults", "roi-single-pixel", "explicit-scale"],
)
def test_intimage_pixels_match_c(extra, label, tmp_path):
    """Integer-image PIXEL data must match C exactly.

    Before the scale/rounding fix, all 1024 pixels of a default run disagreed with
    C by up to 41 counts, because the adc offset was folded into the scale
    ((55000-adc)/max instead of C's 55000/max) and the result was truncated
    instead of rounded.
    """
    import numpy as np

    args = BASE_ARGS + extra
    c_dir = tmp_path / "c"
    py_dir = tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()

    assert run_in(c_dir, [c_binary()] + args).returncode == 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["NANOBRAGG_DISABLE_COMPILE"] = "1"
    # float64, as C: at 55000 counts a float32 pixel is good to about 0.02 of a count,
    # so a pixel next to a rounding boundary can land either side of it
    assert subprocess.run(
        [sys.executable, "-m", "nanobrag_torch", "-dtype", "float64"] + args,
        cwd=str(py_dir), capture_output=True, text=True, timeout=300, env=env,
    ).returncode == 0

    def pixels(path):
        return np.frombuffer(path.read_bytes()[512:], dtype=np.uint16)

    c_px = pixels(c_dir / "intimage.img")
    py_px = pixels(py_dir / "intimage.img")

    assert len(c_px) == len(py_px)
    differing = int((c_px != py_px).sum())
    worst = int(np.abs(c_px.astype(int) - py_px.astype(int)).max()) if differing else 0
    assert differing == 0, (
        f"{label}: {differing}/{len(c_px)} integer pixels differ from C, "
        f"max |delta| = {worst} counts (C peak {c_px.max()}, torch peak {py_px.max()})"
    )


@pytest.mark.parametrize("default_F", [[], ["-default_F", "7"]], ids=["no-default-F", "default-F"])
def test_fdump_from_hkl_is_byte_identical_to_c(default_F, tmp_path):
    c_bin = c_binary()
    hkl = tmp_path / "x.hkl"
    r = range(-3, 4)
    hkl.write_text("".join(f"{h} {k} {l} {50.1 + h + k + l}\n" for h in r for k in r for l in r if h != 2))
    args = ["-cell", "70", "80", "90", "75", "85", "95", "-N", "5", "-lambda", "1", "-distance", "100",
            "-pixel", "0.1", "-detpixels", "32", "-oversample", "1", "-nonoise", "-nopgm"] + default_F
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), KMP_DUPLICATE_LIB_OK="TRUE", NANOBRAGG_DISABLE_COMPILE="1")
    c_dir, py_dir = tmp_path / "c", tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()
    first = ["-hkl", str(hkl), "-floatfile", "f.bin"]
    subprocess.run([c_bin] + args + first, cwd=c_dir, env=env, check=True, capture_output=True, timeout=300)
    subprocess.run([sys.executable, "-m", "nanobrag_torch"] + args + first, cwd=py_dir, env=env, check=True,
                   capture_output=True, timeout=300)
    assert (py_dir / "Fdump.bin").read_bytes() == (c_dir / "Fdump.bin").read_bytes()

    # A run without -hkl renders the same image from the cache
    subprocess.run([sys.executable, "-m", "nanobrag_torch"] + args + ["-floatfile", "g.bin"], cwd=py_dir, env=env,
                   check=True, capture_output=True, timeout=300)
    assert (py_dir / "g.bin").read_bytes() == (py_dir / "f.bin").read_bytes()


@pytest.mark.parametrize("extra", [
    [], ["-denzo"], ["-adxv"], ["-xds"], ["-dials"],
    ["-Xbeam", "2", "-Ybeam", "4.5"], ["-Xbeam", "2", "-Ybeam", "4.5", "-adxv"], ["-Xbeam", "2", "-Ybeam", "4.5", "-xds"],
    ["-twotheta", "10"], ["-twotheta", "10", "-xds"], ["-detector_rotx", "3", "-detector_roty", "2"],
    ["-detector_rotz", "5", "-xds"], ["-ORGX", "10", "-ORGY", "12", "-xds"], ["-Xclose", "5", "-Yclose", "3", "-xds"],
    ["-phi", "5", "-osc", "1", "-phisteps", "2"], ["-distance", "123.456", "-pixel", "0.172"],
    ["-close_distance", "90", "-detector_rotx", "5"],
    # These land next to a %g rounding boundary, where float32 radians move the last digit
    ["-twotheta", "20", "-detector_rotx", "3", "-detector_roty", "4", "-detector_rotz", "5"],
    "-xds -distance 194.238 -detector_rotx -0.303 -detector_roty -6.067 -detector_rotz 0.051 -twotheta -15.657".split(),
    "-distance 286.3 -detector_rotx 6.1 -detector_roty -1.0 -detector_rotz 4.2 -twotheta -9.258".split(),
    # A zero's sign: C's dot products give +0 in DIALS_ORIGIN, and -phi -0 prints -0
    "-adxv -Xbeam 0 -Ybeam 0".split(), "-xds -ORGX 0.5 -ORGY 0.5".split(), "-dials -ORGX 0.5 -ORGY 0.5".split(),
    "-phi -0 -osc 0".split(),
], ids=lambda extra: " ".join(extra) or "defaults")
def test_smv_header_is_byte_identical_to_c(extra, tmp_path):
    args = ("-cell 70 80 90 75 85 95 -default_F 100 -N 2 -lambda 1 -distance 100 -pixel 0.1 -detpixels 32 "
            "-oversample 1 -nopgm -nonoise -intfile i.img -floatfile f.bin").split() + extra
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), KMP_DUPLICATE_LIB_OK="TRUE", NANOBRAGG_DISABLE_COMPILE="1")
    c_dir, py_dir = tmp_path / "c", tmp_path / "py"
    c_dir.mkdir()
    py_dir.mkdir()
    subprocess.run([c_binary()] + args, cwd=c_dir, env=env, check=True, capture_output=True, timeout=300)
    subprocess.run([sys.executable, "-m", "nanobrag_torch"] + args, cwd=py_dir, env=env, check=True,
                   capture_output=True, timeout=300)
    assert (py_dir / "i.img").read_bytes()[:512] == (c_dir / "i.img").read_bytes()[:512]
