"""The top-level package: what it exports, simulate(), and that it stays quiet."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import nanobrag_torch
from nanobrag_torch import BeamConfig, CrystalConfig, DetectorConfig, simulate
from nanobrag_torch.config import CrystalShape
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator

GUIDE = Path(__file__).parents[1] / "docs" / "api_guide.md"


def _configs():
    crystal = CrystalConfig(
        cell_a=50.0, cell_b=50.0, cell_c=50.0, cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0,
        N_cells=(5, 5, 5), default_F=100.0, shape=CrystalShape.GAUSS,
    )
    detector = DetectorConfig(distance_mm=100.0, pixel_size_mm=0.2, spixels=32, fpixels=48, oversample=1)
    return crystal, detector, BeamConfig(wavelength_A=1.0, fluence=1e24)


@pytest.mark.parametrize("name", nanobrag_torch.__all__)
def test_every_export_resolves(name):
    assert getattr(nanobrag_torch, name) is not None


def test_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError):
        nanobrag_torch.not_a_thing


def test_import_loads_no_physics():
    code = "import sys, nanobrag_torch; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_simulate_matches_the_objects_it_builds():
    crystal, detector, beam = _configs()
    by_hand = Simulator(
        Crystal(crystal), Detector(detector), crystal_config=crystal, beam_config=beam
    ).run()
    image = simulate(crystal, detector, beam)
    assert image.shape == (32, 48)
    assert image.max() > 0
    assert torch.equal(image, by_hand)


def test_simulate_applies_dtype_everywhere():
    assert simulate(*_configs(), dtype=torch.float64).dtype == torch.float64


def test_simulator_follows_its_crystal_device():
    crystal, detector, beam = _configs()
    model = Crystal(crystal)
    assert Simulator(model, Detector(detector), beam_config=beam).device == model.device


def test_structure_factor_pair_reaches_the_image():
    crystal, detector, beam = _configs()
    r = np.arange(-20, 21)
    indices = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    amplitudes = np.full(len(indices), 100.0)
    doubled = simulate(crystal, detector, beam, F=(indices, 2 * amplitudes))
    plain = simulate(crystal, detector, beam, F=(indices, amplitudes))
    assert doubled.sum().item() == pytest.approx(4 * plain.sum().item(), rel=1e-5)


def test_hkl_path_reaches_the_image_and_writes_no_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    crystal, detector, beam = _configs()
    r = range(-3, 4)
    (tmp_path / "P1.hkl").write_text(
        "".join(f"{h} {k} {l} 50.0\n" for h in r for k in r for l in r)
    )
    with_file = simulate(crystal, detector, beam, F=tmp_path / "P1.hkl")
    assert not torch.equal(with_file, simulate(crystal, detector, beam))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["P1.hkl"]


def test_a_simulation_prints_nothing():
    code = """
import torch
from nanobrag_torch import BeamConfig, CrystalConfig, DetectorConfig, simulate
crystal = CrystalConfig(N_cells=(5, 5, 5), default_F=100.0)
simulate(crystal, DetectorConfig(spixels=16, fpixels=16), BeamConfig(fluence=1e24))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout == ""


def test_guide_examples_run():
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), flags=re.S)
    assert blocks
    for block in blocks:
        exec(block, {})
