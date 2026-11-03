"""A tensor given to the models in another dtype or device is read at each run, not copied once.

A float32 run of a float64 leaf (or the reverse, or a CPU leaf in a CUDA run) used to keep a
converted copy made at construction: an in-place change of the leaf (.data, an optimizer step)
was not seen by the next run(), and the gradient that reached the leaf was taken at the old
value. Each case changes the leaf in place between two runs of one Simulator and compares the
second run, and its gradient, with a Simulator built afresh from the changed leaf.
"""

import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator

MODES = {
    "float32_run_float64_leaf": (dict(dtype=torch.float32, device="cpu"), dict(dtype=torch.float64, device="cpu")),
    "float64_run_float32_leaf": (dict(dtype=torch.float64, device="cpu"), dict(dtype=torch.float32, device="cpu")),
    "cuda_run_cpu_leaf": (dict(dtype=torch.float64, device="cuda"), dict(dtype=torch.float64, device="cpu")),
}

# name -> (config, field, value, changed value)
INPUTS = {
    "wavelength_A": ("beam", "wavelength_A", 1.0, 1.02),
    "polarization_factor": ("beam", "polarization_factor", 0.5, 0.8),
    "polarization_axis": ("beam", "polarization_axis", (0.0, 0.0, 1.0), (0.0, 0.6, 0.8)),
}


def simulate(name, leaf, run):
    where, field, _, _ = INPUTS[name]
    crystal = dict(cell_a=70.0, cell_b=80.0, cell_c=90.0, N_cells=(5, 5, 5), default_F=100.0,
                   misset_deg=(10.0, 20.0, 30.0))
    beam = dict(wavelength_A=1.0, fluence=1e24, polarization_factor=0.5)
    dict(crystal=crystal, beam=beam)[where][field] = leaf
    detector = DetectorConfig(distance_mm=100.0, pixel_size_mm=1.0, spixels=24, fpixels=24, oversample=1)
    return Simulator(Crystal(CrystalConfig(**crystal), **run), Detector(detector, **run),
                     beam_config=BeamConfig(**beam), **run)


def loss(image):
    weights = torch.rand(image.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    return (image.to(torch.float64) * weights.to(image.device)).sum()


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("name", sorted(INPUTS))
def test_in_place_change_of_a_leaf_of_another_dtype_or_device_is_seen(name, mode):
    run, leaf_kind = MODES[mode]
    if run["device"] == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    _, _, value, changed = INPUTS[name]
    leaf = torch.tensor(value, **leaf_kind, requires_grad=True)
    sim = simulate(name, leaf, run)
    before = sim.run()
    loss(before).backward()

    with torch.no_grad():
        leaf.copy_(torch.tensor(changed, dtype=leaf.dtype))
    leaf.grad = None
    after = sim.run()
    loss(after).backward()

    fresh_leaf = leaf.detach().clone().requires_grad_(True)
    fresh = simulate(name, fresh_leaf, run).run()
    loss(fresh).backward()

    assert not torch.equal(fresh, before), "the change must move the image for the test to mean anything"
    torch.testing.assert_close(after, fresh, rtol=0, atol=0)
    torch.testing.assert_close(leaf.grad, fresh_leaf.grad, rtol=1e-12, atol=0)
