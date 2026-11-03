"""Edge cases of the kept inputs (rv10 review): subclasses, digits, default source wavelengths, grids."""

import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.simulator import Simulator

DIRECTIONS = torch.nn.functional.normalize(torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.003, 0.002]],
                                                        dtype=torch.float64), dim=1)
META = dict(h_min=-10, h_max=10, k_min=-10, k_max=10, l_min=-10, l_max=10)


def crystal_config():
    return CrystalConfig(cell_a=70.123, cell_b=80.0, cell_c=90.0, N_cells=(5, 5, 5), default_F=100.0,
                         misset_deg=(10.0, 20.0, 30.0))


def detector_config():
    return DetectorConfig(distance_mm=100.0, pixel_size_mm=1.0, spixels=16, fpixels=16, oversample=1)


def with_sources(wavelength, dtype=torch.float64):
    beam = BeamConfig(wavelength_A=wavelength, fluence=1e24, source_directions=DIRECTIONS.to(dtype))
    return Simulator(Crystal(crystal_config(), dtype=dtype), Detector(detector_config(), dtype=dtype),
                     beam_config=beam, dtype=dtype)


class MyCrystal(Crystal):
    pass


def test_a_crystal_subclass_moves_to_another_dtype():
    crystal = MyCrystal(crystal_config(), dtype=torch.float32)
    crystal.to(dtype=torch.float64)
    assert crystal.cell_a.dtype == torch.float64


def test_a_missing_kept_value_is_an_attribute_error():
    crystal = Crystal.__new__(Crystal)
    assert not hasattr(crystal, "cell_a")


def test_moving_a_crystal_made_from_numbers_keeps_their_digits():
    crystal = Crystal(crystal_config(), dtype=torch.float32)
    crystal.to(dtype=torch.float64)
    assert crystal.cell_a.item() == 70.123


def test_assigning_the_wavelength_moves_default_source_wavelengths_as_set_spectrum_does():
    assigned = with_sources(1.0)
    assigned.wavelength = 1.3
    by_spectrum = with_sources(1.0)
    by_spectrum.set_spectrum(1.3)
    fresh = with_sources(1.3)
    torch.testing.assert_close(assigned._source_wavelengths, fresh._source_wavelengths, rtol=0, atol=0)
    torch.testing.assert_close(assigned.run(), fresh.run(), rtol=0, atol=0)
    torch.testing.assert_close(by_spectrum.run(), fresh.run(), rtol=0, atol=0)


def test_a_float32_wavelength_leaf_gives_default_source_wavelengths_in_double():
    leaf = torch.tensor(1.02, dtype=torch.float32, requires_grad=True)
    sim = with_sources(leaf)
    expected = leaf.detach().double() * 1e-10
    assert torch.equal(sim._source_wavelengths, expected.expand(2))
    sim._source_wavelengths.sum().backward()
    assert leaf.grad is not None


def test_a_grid_reaches_its_gradient_when_the_first_forward_ran_under_no_grad():
    hkl = (50 + 100 * torch.rand(21, 21, 21, generator=torch.Generator().manual_seed(0),
                                 dtype=torch.float64)).requires_grad_(True)
    model = ExperimentModel(crystal_config(), detector_config(), BeamConfig(wavelength_A=1.0, fluence=1e24),
                            dtype=torch.float32, hkl_data=hkl, hkl_metadata=META)
    with torch.no_grad():
        model()
    model().double().sum().backward()
    assert hkl.grad is not None and hkl.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_a_cpu_grid_in_a_cuda_model_follows_an_in_place_change():
    hkl = 50 + 100 * torch.rand(21, 21, 21, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    configs = (crystal_config(), detector_config(), BeamConfig(wavelength_A=1.0, fluence=1e24))
    model = ExperimentModel(*configs, dtype=torch.float64, device="cuda", hkl_data=hkl, hkl_metadata=META)
    model()
    with torch.no_grad():
        hkl.mul_(1.5)
    after = model()
    fresh = ExperimentModel(*configs, dtype=torch.float64, device="cuda", hkl_data=hkl, hkl_metadata=META)()
    torch.testing.assert_close(after, fresh, rtol=0, atol=0)
