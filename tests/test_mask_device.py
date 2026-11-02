"""compute_statistics takes a mask from whatever device it was read on, and -mask works on a GPU."""

import subprocess
import sys

import numpy as np
import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator

ALLOWED = {torch.Tensor.to, torch.Tensor.shape.__get__, torch.Tensor.dtype.__get__, torch.Tensor.device.__get__}


class MaskOnAnotherDevice(torch.Tensor):
    """Stands in for a mask on another device, on a CPU-only machine: only .to() makes it usable."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func not in ALLOWED:
            raise RuntimeError("Expected all tensors to be on the same device")
        result = super().__torch_function__(func, types, args, kwargs or {})
        return result.as_subclass(torch.Tensor) if isinstance(result, torch.Tensor) else result


def simulator(device):
    crystal = Crystal(CrystalConfig(cell_a=70, cell_b=80, cell_c=90, N_cells=(5, 5, 5), default_F=100), device=device)
    detector = Detector(DetectorConfig(distance_mm=100, pixel_size_mm=0.1, spixels=8, fpixels=8, oversample=1), device=device)
    return Simulator(crystal, detector, beam_config=BeamConfig(wavelength_A=1.0, fluence=1e24), device=device)


def half_mask():
    mask = torch.ones(8, 8)
    mask[:4] = 0
    return mask


def test_statistics_use_the_mask_through_to():
    sim = simulator("cpu")
    image = sim.run()
    sim.detector.config.mask_array = half_mask().as_subclass(MaskOnAnotherDevice)
    stats = sim.compute_statistics(image)
    assert stats["N"] == 32
    assert stats["max_I"] == pytest.approx(image[4:].max().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_statistics_with_a_cpu_mask_on_a_gpu_image():
    sim = simulator("cuda")
    image = sim.run()
    sim.detector.config.mask_array = half_mask()
    assert sim.compute_statistics(image)["N"] == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cli_mask_on_a_gpu_writes_the_image(tmp_path):
    args = "-cell 70 80 90 90 90 90 -default_F 100 -N 5 -detpixels 16 -oversample 1".split()

    def run(*extra):
        done = subprocess.run([sys.executable, "-m", "nanobrag_torch", *args, *extra], cwd=tmp_path,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stdout + done.stderr

    run("-intfile", "mask.img", "-floatfile", "first.bin")
    run("-mask", "mask.img", "-floatfile", "cpu.bin")
    run("-mask", "mask.img", "-floatfile", "gpu.bin", "-device", "cuda")
    np.testing.assert_allclose(np.fromfile(tmp_path / "gpu.bin", dtype=np.float32),
                               np.fromfile(tmp_path / "cpu.bin", dtype=np.float32), rtol=1e-5, atol=1e-8)
