"""Several sources with oversampling, where the kernel takes flat (batch, 3) coordinates."""
import pytest
import torch

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot, SpotMetric,
)
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator

DTYPE = torch.float64


def _image(shape, spot_metric, n_sources, oversample):
    crystal = CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(5, 5, 5),
                            default_F=100.0, shape=shape, spot_metric=spot_metric)
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=16, fpixels=16,
                              detector_convention=DetectorConvention.MOSFLM,
                              detector_pivot=DetectorPivot.BEAM, oversample=oversample)
    beam = BeamConfig(wavelength_A=1.2, fluence=1e24)
    if n_sources > 1:
        beam.source_directions = torch.tensor([[-1.0, 0.0, 0.0]] * n_sources, dtype=DTYPE)  # MOSFLM beam is +x
        beam.source_wavelengths = torch.full((n_sources,), 1.2e-10, dtype=DTYPE)
    return Simulator(Crystal(crystal, beam, dtype=DTYPE), Detector(detector, dtype=DTYPE),
                     beam_config=beam, dtype=DTYPE).run()


@pytest.mark.parametrize("oversample", [1, 2])
@pytest.mark.parametrize("spot_metric", list(SpotMetric))
@pytest.mark.parametrize("shape", list(CrystalShape))
def test_identical_sources_give_the_single_source_image(shape, spot_metric, oversample):
    single = _image(shape, spot_metric, 1, oversample)
    torch.testing.assert_close(_image(shape, spot_metric, 3, oversample), single, rtol=1e-12, atol=0.0)
