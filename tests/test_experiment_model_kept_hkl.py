"""A fixed hkl_data grid given to ExperimentModel follows in-place changes, in any dtype."""

import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from nanobrag_torch.models.experiment import ExperimentModel

META = dict(h_min=-10, h_max=10, k_min=-10, k_max=10, l_min=-10, l_max=10)


def configs():
    return (CrystalConfig(cell_a=70.0, cell_b=80.0, cell_c=90.0, N_cells=(5, 5, 5), default_F=100.0,
                          misset_deg=(10.0, 20.0, 30.0)),
            DetectorConfig(distance_mm=100.0, pixel_size_mm=1.0, spixels=16, fpixels=16, oversample=1),
            BeamConfig(wavelength_A=1.0, fluence=1e24))


def grid(dtype):
    g = torch.Generator().manual_seed(0)
    return 50 + 100 * torch.rand(21, 21, 21, generator=g, dtype=dtype)


@pytest.mark.parametrize("model_dtype, grid_dtype", [(torch.float32, torch.float64), (torch.float64, torch.float32),
                                                     (torch.float64, torch.float64)])
def test_an_in_place_change_to_the_grid_is_seen(model_dtype, grid_dtype):
    hkl = grid(grid_dtype)
    model = ExperimentModel(*configs(), dtype=model_dtype, hkl_data=hkl, hkl_metadata=META)
    before = model().detach()
    with torch.no_grad():
        hkl.mul_(1.5)
    after = model().detach()
    fresh = ExperimentModel(*configs(), dtype=model_dtype, hkl_data=hkl, hkl_metadata=META)().detach()
    assert not torch.equal(after, before)
    torch.testing.assert_close(after, fresh, rtol=0, atol=0)


def test_the_gradient_reaches_a_grid_of_another_dtype_after_a_change():
    hkl = grid(torch.float64).requires_grad_(True)
    model = ExperimentModel(*configs(), dtype=torch.float32, hkl_data=hkl, hkl_metadata=META)
    model().double().sum().backward()
    first = hkl.grad.clone()
    hkl.grad = None
    with torch.no_grad():
        hkl.mul_(1.5)
    model().double().sum().backward()
    fresh_leaf = hkl.detach().clone().requires_grad_(True)
    ExperimentModel(*configs(), dtype=torch.float32, hkl_data=fresh_leaf, hkl_metadata=META)().double().sum().backward()
    assert not torch.equal(hkl.grad, first)
    torch.testing.assert_close(hkl.grad, fresh_leaf.grad, rtol=0, atol=0)
