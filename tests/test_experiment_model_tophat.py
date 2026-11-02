"""TOPHAT in ExperimentModel: the image is the Simulator's, and Ncells is refused with the reason."""

import warnings

import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, CrystalShape, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.simulator import Simulator

CRYSTAL = CrystalConfig(cell_a=70, cell_b=70, cell_c=70, N_cells=(10, 10, 10), default_F=100.0,
                        shape=CrystalShape.TOPHAT)
DETECTOR = DetectorConfig(distance_mm=100, pixel_size_mm=0.1, spixels=64, fpixels=64, oversample=2)
BEAM = BeamConfig(wavelength_A=1.0, fluence=1e24)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_image_is_the_simulators(dtype):
    model = ExperimentModel(CRYSTAL, DETECTOR, BEAM, dtype=dtype).freeze()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = model()
        expected = Simulator(Crystal(CRYSTAL, dtype=dtype), Detector(DETECTOR, dtype=dtype), beam_config=BEAM,
                             dtype=dtype).run()
    assert expected.max() > 0
    assert torch.allclose(image, expected, rtol=1e-6)


def test_ncells_is_refused_with_the_reason():
    model = ExperimentModel(CRYSTAL, DETECTOR, BEAM, dtype=torch.float64).freeze()
    with pytest.raises(ValueError, match="hard cutoff that moves with N"):
        model.refine("Ncells")


def test_refining_everything_leaves_ncells_alone():
    model = ExperimentModel(CRYSTAL, DETECTOR, BEAM, dtype=torch.float64).freeze().refine()
    assert not model.crystal_params.delta_log_N_cells.requires_grad


def test_a_refinable_parameter_still_has_its_gradient():
    model = ExperimentModel(CRYSTAL, DETECTOR, BEAM, dtype=torch.float64).freeze().refine("scale")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model().sum().backward()
    assert model.trainable_parameters()[0].grad.abs().sum() > 0
