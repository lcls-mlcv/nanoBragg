"""diffBragg's off-diagonal Ncells (Nd, Ne, Nf) for the GAUSS lattice factor, hkl metric.

diffBragg (diffBragg_cpu_kernel.cpp:521-536) takes the symmetric matrix
    NABC = [[Na, Nd, Nf], [Nd, Nb, Ne], [Nf, Ne, Nc]]
and F_latt = det(NABC) * exp(-|NABC (h - h0)|^2 / 0.63 * fudge). With Nd = Ne = Nf = 0 that is
nanoBragg's GAUSS shape, Na Nb Nc exp(-sum (dh_i N_i)^2 / 0.63 * fudge), so the default changes nothing.
"""

import math

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot, SpotMetric,
)
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.simulator import Simulator

DT = torch.float64
NDEF = (1.3, -0.7, 0.9)


def configs(ncells_def=None, shape=CrystalShape.GAUSS, metric=SpotMetric.HKL):
    extra = {} if ncells_def is None else dict(N_cells_def=ncells_def)
    crystal = CrystalConfig(cell_a=40.0, cell_b=50.0, cell_c=60.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(6, 5, 4),
                            default_F=100.0, shape=shape, spot_metric=metric, **extra)
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=48, fpixels=48,
                              detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM,
                              oversample=1)
    return crystal, detector, BeamConfig(wavelength_A=1.2, fluence=1e24)


def simulator(ncells_def=None, **kwargs):
    crystal, detector, beam = configs(ncells_def, **kwargs)
    return Simulator(Crystal(crystal, beam, dtype=DT), Detector(detector, dtype=DT), beam_config=beam, dtype=DT)


def fractional_hkl(sim):
    """h = a . (s - s0) / lambda at each pixel centre: oversample 1, one source, no mosaic, phi 0."""
    coords = sim.detector.get_pixel_coords().detach().double()
    s = coords / coords.norm(dim=-1, keepdim=True)
    s0 = sim.incident_beam_direction.detach().double()
    scattering = (s - s0) / 1.2  # 1/A
    (a, b, c), _ = sim.crystal.get_rotated_real_vectors(sim.crystal.config)
    return torch.stack([(scattering * v[0, 0].detach().double()).sum(-1) for v in (a, b, c)], dim=-1).numpy()


def gauss_latt(dh, n, ndef):
    na, nb, nc = n
    nd, ne, nf = ndef
    nabc = np.array([[na, nd, nf], [nd, nb, ne], [nf, ne, nc]], dtype=float)
    v = dh @ nabc.T
    return np.linalg.det(nabc) * np.exp(-(v * v).sum(-1) / 0.63)


def test_zero_off_diagonals_change_nothing():
    plain = simulator().run()
    torch.testing.assert_close(simulator((0.0, 0.0, 0.0)).run(), plain, rtol=0, atol=0)


def test_off_diagonals_follow_diffbraggs_lattice_factor():
    sim_zero, sim_def = simulator(), simulator(NDEF)
    zero, with_def = sim_zero.run().numpy(), sim_def.run().numpy()
    hkl = fractional_hkl(sim_def)
    dh = hkl - np.round(hkl)
    expected_ratio = (gauss_latt(dh, (6, 5, 4), NDEF) / gauss_latt(dh, (6, 5, 4), (0, 0, 0))) ** 2
    lit = zero > 1e-6 * zero.max()
    assert lit.sum() > 50
    np.testing.assert_allclose(with_def[lit] / zero[lit], expected_ratio[lit], rtol=1e-9)
    assert not np.allclose(with_def, zero)


def test_off_diagonals_are_differentiable():
    crystal, detector, beam = configs()

    def image(nd, ne, nf):
        cfg = CrystalConfig(**{**crystal.__dict__, "N_cells_def": (nd, ne, nf)})
        return Simulator(Crystal(cfg, beam, dtype=DT), Detector(detector, dtype=DT), beam_config=beam, dtype=DT).run().sum()

    start = tuple(torch.tensor(v, dtype=DT, requires_grad=True) for v in NDEF)
    assert gradcheck(image, start, eps=1e-6, atol=1e-8, rtol=1e-5)


def test_reciprocal_metric_refuses_off_diagonals():
    with pytest.raises(ValueError, match="N_cells_def"):
        simulator(NDEF, metric=SpotMetric.RECIPROCAL).run()


def model(shape=CrystalShape.GAUSS, metric=SpotMetric.HKL, ncells_def=(0.0, 0.0, 0.0)):
    return ExperimentModel(*configs(ncells_def, shape=shape, metric=metric), dtype=DT)


def test_ndef_group_starts_at_the_config_and_reproduces_its_image():
    m = model(ncells_def=NDEF)
    torch.testing.assert_close(m.crystal_params.N_cells_def(), torch.tensor(NDEF, dtype=DT), rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(m(), simulator(NDEF).run(), rtol=1e-12, atol=0)


def test_ndef_group_gradcheck():
    m = model(ncells_def=NDEF)
    start = torch.tensor([0.2, -0.1, 0.3], dtype=DT, requires_grad=True)

    def loss(delta):
        return functional_call(m, {"crystal_params.delta_N_cells_def": delta}, ()).sum()

    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize("shape, metric", [(CrystalShape.SQUARE, SpotMetric.HKL), (CrystalShape.ROUND, SpotMetric.HKL),
                                           (CrystalShape.GAUSS, SpotMetric.RECIPROCAL)])
def test_ndef_is_unrefinable_outside_gauss_with_the_hkl_metric(shape, metric):
    m = model(shape=shape, metric=metric)
    assert "crystal_params.delta_N_cells_def" in m.unrefinable
    with pytest.raises(ValueError, match="Ndef"):
        m.refine("Ndef")
