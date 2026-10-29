"""Mosaic spread eta, isotropic and anisotropic, in ExperimentModel (plan C2).

The ensemble is diffBragg's (utils/mosaic.py); tests/test_diffbragg_parity.py
checks it and its eta derivatives against simtbx on a cctbx build.
"""
import math

import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.simulator import Simulator
from nanobrag_torch.utils.mosaic import diffbragg_umats

DTYPE = torch.float64


def _configs(eta=0.5, domains=10):
    crystal = CrystalConfig(cell_a=70.0, cell_b=80.0, cell_c=90.0,
                            cell_alpha=88.0, cell_beta=91.0, cell_gamma=93.0,
                            misset_deg=(5.0, -3.0, 2.0), N_cells=(6, 6, 6), default_F=100.0,
                            shape=CrystalShape.GAUSS, mosaic_spread_deg=eta, mosaic_domains=domains)
    detector = DetectorConfig(distance_mm=80.0, pixel_size_mm=0.2, spixels=48, fpixels=48,
                              detector_convention=DetectorConvention.MOSFLM,
                              detector_pivot=DetectorPivot.BEAM, oversample=1)
    beam = BeamConfig(wavelength_A=1.2, fluence=1e24)
    return crystal, detector, beam


def _angles_deg(umats):
    # |sin θ| from the antisymmetric part and cos θ from the trace: atan2 keeps small angles exact
    skew = umats - umats.transpose(-1, -2)
    sin = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1).norm(dim=-1) / 2
    cos = (umats.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.atan2(sin, cos))


@pytest.mark.parametrize("eta", [0.3, (0.2, 0.6, 0.05)])
def test_umats_are_rotations_in_plus_minus_pairs(eta):
    u = diffbragg_umats(eta, 20)
    assert u.shape == (20, 3, 3)
    eye = torch.eye(3, dtype=DTYPE).expand_as(u)
    torch.testing.assert_close(u @ u.transpose(-1, -2), eye, rtol=0, atol=1e-14)
    torch.testing.assert_close(torch.linalg.det(u), torch.ones(20, dtype=DTYPE), rtol=0, atol=1e-14)
    torch.testing.assert_close(u[0::2], u[1::2].transpose(-1, -2), rtol=0, atol=1e-14)


def test_isotropic_rotation_angles_scale_with_eta():
    a1, a2 = _angles_deg(diffbragg_umats(0.2, 50)), _angles_deg(diffbragg_umats(0.6, 50))
    torch.testing.assert_close(a2, 3 * a1, rtol=1e-9, atol=1e-12)
    assert diffbragg_umats(0.2, 50).shape == (50, 3, 3)


def test_equal_anisotropic_components_equal_isotropic():
    torch.testing.assert_close(diffbragg_umats((0.4, 0.4, 0.4), 30), diffbragg_umats(0.4, 30),
                               rtol=0, atol=1e-15)


def test_odd_domain_count_is_rejected():
    with pytest.raises(ValueError, match="even"):
        diffbragg_umats(0.3, 7)
    with pytest.raises(ValueError, match="mosaic_domains must be even"):
        ExperimentModel(*_configs(domains=7), dtype=DTYPE)


def test_model_image_uses_the_diffbragg_ensemble():
    crystal_cfg, detector_cfg, beam_cfg = _configs()
    model = ExperimentModel(crystal_cfg, detector_cfg, beam_cfg, dtype=DTYPE)
    assert model.crystal_params.delta_log_eta.requires_grad
    with torch.no_grad():
        image = model()
    crystal = Crystal(crystal_cfg, beam_cfg, dtype=DTYPE)
    crystal.mosaic_umats_override = diffbragg_umats(0.5, 10)
    reference = Simulator(crystal, Detector(detector_cfg, dtype=DTYPE), beam_config=beam_cfg, dtype=DTYPE).run()
    torch.testing.assert_close(image, reference, rtol=1e-12, atol=0.0)


def test_eta_step_reaches_the_persistent_crystal():
    model = ExperimentModel(*_configs(), dtype=DTYPE)
    with torch.no_grad():
        before = model()
        model.crystal_params.delta_log_eta.fill_(math.log(2.0))
        after = model()
        fresh = ExperimentModel(*_configs(eta=1.0), dtype=DTYPE)()
    assert not torch.allclose(before, after)
    torch.testing.assert_close(after, fresh, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("anisotropic", [False, True])
def test_gradcheck_eta(anisotropic):
    model = ExperimentModel(*_configs(), dtype=DTYPE, anisotropic_mosaicity=anisotropic)
    name = "crystal_params.delta_log_eta"
    param = dict(model.named_parameters())[name]
    assert param.shape == ((3,) if anisotropic else (1,))
    start = torch.tensor([0.05, -0.1, 0.2][: param.numel()], dtype=DTYPE, requires_grad=True)

    def loss(value):
        return functional_call(model, {name: value}, ()).sum()

    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-8, rtol=1e-4)


def test_anisotropic_components_have_distinct_gradients():
    model = ExperimentModel(*_configs(), dtype=DTYPE, anisotropic_mosaicity=True)
    model().sum().backward()
    g = model.crystal_params.delta_log_eta.grad
    assert torch.all(g.abs() > 0)
    assert len({round(float(x), 12) for x in g}) == 3


def test_eta_unrefinable_without_mosaic_spread():
    model = ExperimentModel(*_configs(eta=0.0, domains=1), dtype=DTYPE)
    assert not model.crystal_params.delta_log_eta.requires_grad
    with pytest.raises(ValueError, match="eta is not refinable without mosaic spread"):
        model.refine("eta")


def test_anisotropic_eta_lies_along_the_starting_crystal_axes():
    """diffBragg refines eta_abc along the crystal's a, b, c (transform_eta=True)."""
    crystal_cfg, detector_cfg, beam_cfg = _configs()
    model = ExperimentModel(crystal_cfg, detector_cfg, beam_cfg, dtype=DTYPE, anisotropic_mosaicity=True)
    start = Crystal(crystal_cfg, beam_cfg, dtype=DTYPE)
    for i, v in enumerate((start.a, start.b, start.c)):
        torch.testing.assert_close(model.crystal_params.eta_axes[:, i], v / v.norm(), rtol=0, atol=1e-14)
    eta = torch.tensor([0.2, 0.6, 0.05], dtype=DTYPE)
    with torch.no_grad():
        model.crystal_params.delta_log_eta.copy_(torch.log(eta / 0.5))
        umats = model.crystal_params.mosaic_umats()
    torch.testing.assert_close(umats, diffbragg_umats(eta, 10, crystal_axes=model.crystal_params.eta_axes),
                               rtol=0, atol=1e-15)
    assert (umats - diffbragg_umats(eta, 10)).abs().max() > 1e-4  # not the lab frame
