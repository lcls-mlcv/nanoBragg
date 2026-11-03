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
SAME = {"float32_run_float32_leaf": (dict(dtype=torch.float32, device="cpu"), dict(dtype=torch.float32, device="cpu"))}

_DIRECTIONS = torch.nn.functional.normalize(
    torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.003, 0.002]], dtype=torch.float64), dim=1).tolist()
_MOVED = torch.nn.functional.normalize(
    torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.007, -0.001]], dtype=torch.float64), dim=1).tolist()

# name -> (config, field, value, changed value)
INPUTS = {
    "wavelength_A": ("beam", "wavelength_A", 1.0, 1.02),
    "polarization_factor": ("beam", "polarization_factor", 0.5, 0.8),
    "polarization_axis": ("beam", "polarization_axis", (0.0, 0.0, 1.0), (0.0, 0.6, 0.8)),
    "cell_a": ("crystal", "cell_a", 70.0, 70.5),
    "cell_gamma": ("crystal", "cell_gamma", 90.0, 91.0),
    "N_cells_a": ("crystal", "N_cells", 5.0, 6.0),
    "source_directions": ("sources", "source_directions", _DIRECTIONS, _MOVED),
    "source_wavelengths": ("sources", "source_wavelengths", [1.0e-10, 1.01e-10], [1.0e-10, 1.03e-10]),
    "source_weights": ("sources", "source_weights", [1.0, 0.5], [1.0, 1.5]),
}
SOURCES = sorted(name for name, spec in INPUTS.items() if spec[0] == "sources")


def simulate(name, leaf, run):
    where, field, _, _ = INPUTS[name]
    crystal = dict(cell_a=70.0, cell_b=80.0, cell_c=90.0, N_cells=(5, 5, 5), default_F=100.0,
                   misset_deg=(10.0, 20.0, 30.0))
    beam = dict(wavelength_A=1.0, fluence=1e24, polarization_factor=0.5)
    if where == "sources":
        beam.update(source_directions=torch.tensor(_DIRECTIONS, **run),
                    source_wavelengths=torch.tensor([1.0e-10, 1.01e-10], **run),
                    source_weights=torch.tensor([1.0, 0.5], **run), use_source_weights=True)
        where = "beam"
    if field == "N_cells":
        leaf = (leaf, 5, 5)
    dict(crystal=crystal, beam=beam)[where][field] = leaf
    detector = DetectorConfig(distance_mm=100.0, pixel_size_mm=1.0, spixels=24, fpixels=24, oversample=1)
    return Simulator(Crystal(CrystalConfig(**crystal), **run), Detector(detector, **run),
                     beam_config=BeamConfig(**beam), **run)


def loss(image):
    weights = torch.rand(image.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    return (image.to(torch.float64) * weights.to(image.device)).sum()


def check_in_place_change_is_seen(name, run, leaf_kind):
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


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("name", sorted(INPUTS))
def test_in_place_change_of_a_leaf_of_another_dtype_or_device_is_seen(name, mode):
    check_in_place_change_is_seen(name, *MODES[mode])


@pytest.mark.parametrize("name", SOURCES)
def test_in_place_change_of_a_source_leaf_of_the_run_dtype_is_seen(name):
    """Source wavelengths were taken to Angstroms once, so even this case was stale."""
    check_in_place_change_is_seen(name, *SAME["float32_run_float32_leaf"])


def test_sources_without_wavelengths_pass_the_gradient_to_the_wavelength():
    """Each source then has the primary wavelength, and its gradient reaches that leaf."""
    run = dict(dtype=torch.float64, device="cpu")
    grads = []
    for sources in (True, False):
        leaf = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        beam = dict(wavelength_A=leaf, fluence=1e24)
        if sources:
            beam.update(source_directions=torch.tensor([[-1.0, 0.0, 0.0]], **run))
        crystal = CrystalConfig(cell_a=70.0, cell_b=80.0, cell_c=90.0, N_cells=(5, 5, 5), default_F=100.0,
                                misset_deg=(10.0, 20.0, 30.0))
        detector = DetectorConfig(distance_mm=100.0, pixel_size_mm=1.0, spixels=24, fpixels=24, oversample=1)
        sim = Simulator(Crystal(crystal, **run), Detector(detector, **run), beam_config=BeamConfig(**beam), **run)
        loss(sim.run()).backward()
        grads.append(leaf.grad)
    assert grads[0] is not None
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-9, atol=0)
