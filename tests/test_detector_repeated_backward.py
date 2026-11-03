"""A differentiable detector geometry survives repeated runs, and a default beam centre follows the pixel size.

The detector keeps its derived geometry (pix0, basis vectors, beam centre in pixels) between runs
and rebuilt it only when the config changed. With a geometry input that requires grad, those
tensors carry the graph of the run that built them, so a second backward at an unchanged value
went through a freed graph. The default beam centre was computed once in DetectorConfig from the
pixel size, so with a pixel size leaf it neither followed a change nor survived a second backward.
A fresh Simulator built from the same values is the reference throughout.
"""

import pytest
import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig, DetectorConvention
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator

DT = torch.float64
CONVENTIONS = [DetectorConvention.MOSFLM, DetectorConvention.XDS, DetectorConvention.ADXV, DetectorConvention.DENZO]


def simulator(field, value, convention, explicit_centre=False):
    detector = dict(distance_mm=100.0, pixel_size_mm=1.0, spixels=24, fpixels=24, oversample=1,
                    detector_convention=convention)
    if explicit_centre:
        detector.update(beam_center_s=12.0, beam_center_f=12.0)
    detector[field] = value
    crystal = CrystalConfig(cell_a=70.0, cell_b=80.0, cell_c=90.0, N_cells=(5, 5, 5), default_F=100.0,
                            misset_deg=(10.0, 20.0, 30.0))
    return Simulator(Crystal(crystal, dtype=DT), Detector(DetectorConfig(**detector), dtype=DT),
                     beam_config=BeamConfig(wavelength_A=1.0, fluence=1e24), dtype=DT)


def loss(image):
    return (image * torch.linspace(0, 1, image.numel(), dtype=DT).reshape(image.shape)).sum()


def fresh(field, value, convention, explicit_centre=False):
    leaf = torch.tensor(value, dtype=DT, requires_grad=True)
    image = simulator(field, leaf, convention, explicit_centre).run()
    loss(image).backward()
    return image.detach(), leaf.grad.item()


@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.name)
@pytest.mark.parametrize("field, start, moved", [("pixel_size_mm", 1.0, 1.02), ("distance_mm", 100.0, 102.0)])
@pytest.mark.parametrize("explicit_centre", [False, True], ids=["default_centre", "explicit_centre"])
@pytest.mark.parametrize("change", [False, True], ids=["same_value", "changed_in_place"])
def test_second_run_matches_a_fresh_simulator(convention, field, start, moved, explicit_centre, change):
    leaf = torch.tensor(start, dtype=DT, requires_grad=True)
    sim = simulator(field, leaf, convention, explicit_centre)
    loss(sim.run()).backward()
    leaf.grad = None
    if change:
        with torch.no_grad():
            leaf.fill_(moved)
    image = sim.run()
    loss(image).backward()
    expected_image, expected_grad = fresh(field, leaf.item(), convention, explicit_centre)
    torch.testing.assert_close(image.detach(), expected_image, rtol=1e-12, atol=0)
    assert leaf.grad.item() == pytest.approx(expected_grad, rel=1e-10)


def test_a_centre_assigned_after_construction_is_kept():
    leaf = torch.tensor(1.0, dtype=DT, requires_grad=True)
    sim = simulator("pixel_size_mm", leaf, DetectorConvention.MOSFLM)
    sim.detector.config.beam_center_s = 10.0
    sim.detector.config.beam_center_f = 11.0
    sim.run()
    assert sim.detector.config.beam_center_s == 10.0 and sim.detector.config.beam_center_f == 11.0


def test_numbers_rebuild_nothing_between_runs():
    sim = simulator("distance_mm", 100.0, DetectorConvention.MOSFLM)
    sim.run()
    version = sim.detector._geometry_version
    sim.run()
    assert sim.detector._geometry_version == version


def test_forward_only_runs_with_a_leaf_rebuild_nothing():
    """The rebuild is for a second backward: runs with no backward in between keep the geometry."""
    leaf = torch.tensor(100.0, dtype=DT, requires_grad=True)
    sim = simulator("distance_mm", leaf, DetectorConvention.MOSFLM)
    sim.run()
    version = sim.detector._geometry_version
    sim.run()
    sim.run()
    assert sim.detector._geometry_version == version
    loss(sim.run()).backward()
    sim.run()
    assert sim.detector._geometry_version == version + 1


def test_a_geometry_built_under_no_grad_is_rebuilt_for_a_grad_run():
    leaf = torch.tensor(100.0, dtype=DT, requires_grad=True)
    sim = simulator("distance_mm", leaf, DetectorConvention.MOSFLM)
    with torch.no_grad():
        with torch.no_grad():
            leaf.fill_(101.0)
        sim.run()
    loss(sim.run()).backward()
    _, expected_grad = fresh("distance_mm", 101.0, DetectorConvention.MOSFLM)
    assert leaf.grad.item() == pytest.approx(expected_grad, rel=1e-10)
