"""Refinable structure factors, one amplitude per unique reflection, in ExperimentModel (plan C2).

The symmetry expansion is utils/structure_factors.py; tests/test_diffbragg_parity.py
checks the per-ASU gradients against diffBragg's Fhkl gradients on a cctbx build,
and tests/test_compat_cctbx.py the expansion against cctbx's expand_to_p1.
"""
import math

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.compat.cctbx import set_structure_factors
from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.simulator import Simulator
from nanobrag_torch.utils.structure_factors import StructureFactors, expand_to_p1

DTYPE = torch.float64
P121 = [np.eye(3), np.diag([-1.0, 1.0, -1.0])]  # 2-fold along b


def _orbit(h, anomalous=False):
    return {tuple(x) for x in expand_to_p1([h], P121, anomalous)[0]}


def _unique_reflections(H):
    """One representative per P121 orbit of the box |h|, |k|, |l| <= H."""
    seen, reps = set(), []
    for h in range(-H, H + 1):
        for k in range(-H, H + 1):
            for l in range(-H, H + 1):  # noqa: E741
                if (h, k, l) not in seen:
                    seen |= _orbit((h, k, l))
                    reps.append((h, k, l))
    return np.array(reps)


def _structure_factors(H=2, seed=0):
    reps = _unique_reflections(H)
    amplitudes = np.random.default_rng(seed).uniform(50.0, 500.0, len(reps))
    return StructureFactors(reps, amplitudes, rotations=P121)


def _configs(pixels=32):
    crystal = CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, cell_beta=100.0,
                            misset_deg=(5.0, -3.0, 2.0), N_cells=(5, 5, 5), default_F=100.0,
                            shape=CrystalShape.GAUSS)
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=pixels, fpixels=pixels,
                              detector_convention=DetectorConvention.MOSFLM,
                              detector_pivot=DetectorPivot.BEAM, oversample=1)
    beam = BeamConfig(wavelength_A=1.2, fluence=1e24)
    return crystal, detector, beam


def test_orbits_follow_the_point_group_and_friedel_law():
    general, centric = (1, 2, 3), (1, 0, 2)  # h0l is centric in P2 (b unique)
    assert _orbit(general) == {(1, 2, 3), (-1, 2, -3), (-1, -2, -3), (1, -2, 3)}
    assert _orbit(general, anomalous=True) == {(1, 2, 3), (-1, 2, -3)}
    assert _orbit(centric) == _orbit(centric, anomalous=True) == {(1, 0, 2), (-1, 0, -2)}
    p1, source = expand_to_p1([(1, 2, 3), (0, 1, 0)], anomalous=False)  # P1: Friedel pairs only
    assert {tuple(h): int(s) for h, s in zip(p1, source)} == {
        (1, 2, 3): 0, (-1, -2, -3): 0, (0, 1, 0): 1, (0, -1, 0): 1}


def test_symmetry_mates_given_twice_are_rejected():
    with pytest.raises(ValueError, match="symmetry mates"):
        expand_to_p1([(1, 2, 3), (-1, 2, -3)], P121)
    expand_to_p1([(1, 2, 3), (-1, -2, -3)], P121, anomalous=True)  # a Bijvoet pair is two reflections
    with pytest.raises(ValueError, match="integer"):
        expand_to_p1([(1, 2, 3)], [np.eye(3) * 0.5])


def test_grid_index_uses_default_and_F000_slots():
    sf = StructureFactors([(1, 0, 0)], [7.0], F000=3.0)
    index, meta, constants = sf.grid_index(default_F=100.0)
    assert meta == dict(h_min=-1, h_max=1, k_min=0, k_max=0, l_min=0, l_max=0)
    np.testing.assert_array_equal(index[:, 0, 0], [0, 2, 0])  # (-1,0,0) and (1,0,0) share F; (0,0,0) is F000
    np.testing.assert_array_equal(constants, [100.0, 3.0])
    keep = StructureFactors([(1, 0, 0), (0, 0, 0)], [7.0, 5.0], F000=None)
    index, _, _ = keep.grid_index(default_F=100.0)
    np.testing.assert_array_equal(index[:, 0, 0], [0, 1, 0])


def test_model_image_matches_a_fixed_p1_grid():
    sf = _structure_factors()
    crystal_cfg, detector_cfg, beam_cfg = _configs()
    model = ExperimentModel(crystal_cfg, detector_cfg, beam_cfg, dtype=DTYPE, structure_factors=sf)
    assert model.fhkl_params.delta_log_F.requires_grad
    assert model.fhkl_params.delta_log_F.shape == (len(sf),)
    with torch.no_grad():
        image = model()
    p1, source = sf.p1()
    crystal = Crystal(crystal_cfg, beam_cfg, dtype=DTYPE)
    set_structure_factors(crystal, p1, sf.amplitudes[source])
    reference = Simulator(crystal, Detector(detector_cfg, dtype=DTYPE), beam_config=beam_cfg, dtype=DTYPE).run()
    assert image.sum() > 0
    torch.testing.assert_close(image, reference, rtol=1e-12, atol=0.0)


def test_each_amplitude_collects_the_gradient_of_all_its_mates():
    sf = _structure_factors()
    crystal_cfg, detector_cfg, beam_cfg = _configs()
    model = ExperimentModel(crystal_cfg, detector_cfg, beam_cfg, dtype=DTYPE, structure_factors=sf)
    w = torch.rand(32, 32, dtype=DTYPE, generator=torch.Generator().manual_seed(1))
    (model() * w).sum().backward()
    got = model.fhkl_params.delta_log_F.grad

    p1, source = sf.p1()
    crystal = Crystal(crystal_cfg, beam_cfg, dtype=DTYPE)
    set_structure_factors(crystal, p1, sf.amplitudes[source])
    grid = crystal.hkl_data.clone().requires_grad_(True)
    crystal.hkl_data = grid
    image = Simulator(crystal, Detector(detector_cfg, dtype=DTYPE), beam_config=beam_cfg, dtype=DTYPE).run()
    (dgrid,) = torch.autograd.grad((image * w).sum(), [grid])
    lo = p1.min(axis=0)
    per_mate = dgrid[tuple(torch.as_tensor(p1 - lo).T)]
    expected = torch.zeros(len(sf), dtype=DTYPE).index_add_(0, torch.as_tensor(source), per_mate)
    expected *= torch.as_tensor(sf.amplitudes)  # d/d(log F) = F·d/dF
    assert (expected != 0).sum() >= 5
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12 * float(expected.abs().max()))


def test_gradcheck_fhkl():
    model = ExperimentModel(*_configs(pixels=16), dtype=DTYPE, structure_factors=_structure_factors(H=1))
    name = "fhkl_params.delta_log_F"
    n = dict(model.named_parameters())[name].numel()
    start = torch.linspace(-0.2, 0.2, n, dtype=DTYPE).requires_grad_(True)

    def loss(value):
        return functional_call(model, {name: value}, ()).sum()

    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-8, rtol=1e-4)


def test_fhkl_step_reaches_the_persistent_crystal():
    sf = _structure_factors()
    model = ExperimentModel(*_configs(), dtype=DTYPE, structure_factors=sf)
    doubled = StructureFactors(sf.indices, 2 * sf.amplitudes, rotations=P121)
    with torch.no_grad():
        before = model()
        model.fhkl_params.delta_log_F.fill_(math.log(2.0))
        after = model()
        fresh = ExperimentModel(*_configs(), dtype=DTYPE, structure_factors=doubled)()
    assert not torch.allclose(before, after)
    torch.testing.assert_close(after, fresh, rtol=1e-12, atol=0.0)


def test_fhkl_unrefinable_without_structure_factors():
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    assert model.fhkl_params.delta_log_F.numel() == 0
    assert "fhkl_params.delta_log_F" in model.unrefinable
    with pytest.raises(ValueError, match="Fhkl is not refinable without structure factors"):
        model.refine("Fhkl")
    model.freeze("Fhkl")
    model.refine()  # everything refinable; Fhkl stays frozen
    assert not model.fhkl_params.delta_log_F.requires_grad


def test_refinable_and_fixed_structure_factors_are_exclusive():
    sf = _structure_factors()
    grid, meta = torch.ones(1, 1, 1), dict(h_min=0, h_max=0, k_min=0, k_max=0, l_min=0, l_max=0)
    with pytest.raises(ValueError, match="not both"):
        ExperimentModel(*_configs(), dtype=DTYPE, structure_factors=sf, hkl_data=grid, hkl_metadata=meta)
    model = ExperimentModel(*_configs(), dtype=DTYPE, structure_factors=sf)
    with pytest.raises(ValueError, match="refines its structure factors"):
        model.set_structure_factors(grid, meta)
