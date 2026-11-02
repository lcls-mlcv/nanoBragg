"""C's auto oversample, and what torch says when that needs more memory than there is."""

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import nanobrag_torch.simulator as simulator_module
from nanobrag_torch.__main__ import build_simulation, create_parser

REPO = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), MKL_THREADING_LAYER="GNU", NANOBRAGG_DISABLE_COMPILE="1")
BASE = "-cell 70 80 90 90 90 90 -default_F 100 -distance 100 -detpixels 16".split()
HUGE = BASE + "-samplesize 0.04".split()


def bundle(flags):
    args = create_parser().parse_args(flags)
    args._argv = flags
    return build_simulation(args)


@pytest.mark.parametrize("flags", ["-samplesize 0.04", "-samplesize 0.004", "-N 3", "-N 40", "-samplesize 0.0123",
                                   "-samplesize 0.04 -lambda 2.3 -pixel 0.172", "-Na 100 -Nb 3 -Nc 7"])
def test_auto_oversample_is_cs(flags, tmp_path):
    c_bin = os.environ.get("NB_C_BIN")
    if not c_bin or not Path(c_bin).exists():
        pytest.skip("set NB_C_BIN to the oracle binary")
    argv = BASE + flags.split()
    # One pixel of ROI, as the choice comes before the render and C loops over every sample
    c = subprocess.run([str(Path(c_bin).resolve())] + argv + "-roi 0 0 0 0 -nopgm -nonoise -nointfile -floatfile f.bin".split(),
                       cwd=tmp_path, capture_output=True, text=True, timeout=120, check=True)
    expected = int(re.search(r"auto-selected (\d+)-fold oversampling", c.stdout).group(1))
    assert bundle(argv).simulator._auto_oversample() == expected


def test_warning_gives_the_memory_and_leaves_the_run_alone(monkeypatch):
    monkeypatch.setattr(simulator_module, "_memory_available_bytes", lambda device: 10 ** 9)
    simulator = bundle(BASE + "-N 3".split()).simulator
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = simulator.run(oversample=2)
    with pytest.warns(UserWarning, match=r"needs about 0\.000164 GB and 1e-06 GB is available") as caught:
        monkeypatch.setattr(simulator_module, "_memory_available_bytes", lambda device: 1000)
        warned = simulator.run(oversample=2)
    assert "2-fold oversampling" in str(caught[0].message)
    assert torch.equal(quiet, warned)


@pytest.mark.parametrize("flags, gigabytes", [
    ("-oversample 1000", 16 ** 2 * 1000 ** 2 * 160 / 1e9),
    ("-oversample 1000 -phisteps 3 -osc 1", 3 * 16 ** 2 * 1000 ** 2 * 160 / 1e9),
    ("-oversample 1000 -dtype float64", 2 * 16 ** 2 * 1000 ** 2 * 160 / 1e9),
    ("-oversample 1000 -detector_thick 100 -detector_abs 300 -detector_thicksteps 4", 4 * 16 ** 2 * 1000 ** 2 * 160 / 1e9),
])
def test_expected_memory_counts_what_multiplies_it(flags, gigabytes, monkeypatch):
    monkeypatch.setattr(simulator_module, "_memory_available_bytes", lambda device: 1000)
    simulator = bundle(BASE + flags.split()).simulator
    with pytest.warns(UserWarning) as caught:
        simulator._warn_if_memory_runs_out(16, 16, 1000, 1, None)
    assert float(re.search(r"needs about ([0-9.e+]+) GB", str(caught[0].message)).group(1)) == pytest.approx(gigabytes, rel=2e-2)


def test_pixel_batch_size_is_counted(monkeypatch):
    monkeypatch.setattr(simulator_module, "_memory_available_bytes", lambda device: 10 ** 10)
    simulator = bundle(BASE).simulator
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        simulator._warn_if_memory_runs_out(16000, 16000, 2, 1, 1)
    with pytest.warns(UserWarning):
        simulator._warn_if_memory_runs_out(16000, 16000, 2, 1, None)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is not enforced elsewhere")
def test_available_memory_sees_the_address_space_cap():
    code = ("import torch; from nanobrag_torch.simulator import _memory_available_bytes as m; "
            "print(m(torch.device('cpu')))")
    run = subprocess.run(["bash", "-c", f'ulimit -v 9000000; exec "{sys.executable}" -c "{code}"'],
                         env=ENV, capture_output=True, text=True, timeout=120, check=True)
    assert 0 < int(run.stdout) < 9000000 * 1024


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is not enforced elsewhere")
def test_oversized_run_under_a_cap_warns_and_fails_without_growing(tmp_path):
    cmd = " ".join(f'"{token}"' for token in [sys.executable, "-m", "nanobrag_torch"] + HUGE + ["-floatfile", "f.bin"])
    run = subprocess.run(["bash", "-c", f"ulimit -v 8000000; exec {cmd}"], cwd=tmp_path, env=ENV,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 1
    assert "This run needs about 59.1 GB" in run.stderr
    assert "1201-fold oversampling" in run.stderr
    assert "Traceback" not in run.stderr
    assert not (tmp_path / "f.bin").exists()
