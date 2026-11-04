"""hopper's restraints (simtbx/diffBragg/hopper_utils.py, refiners/parameters.py).

For each refined parameter with a center c and a beta b, hopper adds
    ½ (log 2πb + (c - v)²/b)
to its target, v in hopper's units: G, Nabc, Ndef in cells, eta_abc in degrees, cell lengths in
Angstrom, cell angles in radians (centers given in degrees), RotXYZ in radians about 0, detz_shift
in metres, spec = (lambda_offset, lambda_scale). A fixed parameter adds nothing. Nvol adds
½ (c - det NABC)²/b whether or not Ncells refines.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.compat.cctbx import detector_config_from_dxtbx_panel
from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot, SpotMetric,
)
from nanobrag_torch.models.experiment import ExperimentModel, SerialExperiment
from nanobrag_torch.utils.refinement import refine_lbfgsb
from nanobrag_torch.utils.restraints import hopper_restraints

DT = torch.float64


def configs(**crystal):
    crystal = CrystalConfig(**{**dict(cell_a=40.0, cell_b=50.0, cell_c=60.0, cell_alpha=88.0, cell_beta=95.0,
                                      cell_gamma=91.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(6, 5, 4),
                                      default_F=100.0, shape=CrystalShape.GAUSS, mosaic_spread_deg=0.05,
                                      mosaic_domains=4, spot_metric=SpotMetric.HKL), **crystal})
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=16, fpixels=16,
                              detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM,
                              oversample=1)
    return crystal, detector, BeamConfig(wavelength_A=1.2, fluence=1e24, spot_scale=3.0)


def model(**kwargs):
    m = ExperimentModel(*configs(), dtype=DT, **kwargs)
    with torch.no_grad():  # away from the configs, so values and centers differ
        for p in m.parameters():
            p.add_(0.01 * torch.arange(1, p.numel() + 1, dtype=DT).reshape(p.shape))
    return m


def term(center, beta, value):
    return 0.5 * (math.log(2 * math.pi * beta) + (center - value) ** 2 / beta)


def values(m):
    """The model's values in hopper's units, computed here independently of restraints.py."""
    c, b = m.crystal_params.build_config(), m.beam_params.build_config()
    n = [float(v) for v in c.N_cells]
    nd = [float(v) for v in c.N_cells_def]
    misset = (m.crystal_params.max_misset_delta_deg * torch.tanh(m.crystal_params.delta_misset_raw)).detach()
    return {
        "G": float(b.spot_scale), "Nabc": n, "Ndef": nd, "eta_abc": [float(v) for v in m.crystal_params.eta()],
        "ucell_a": float(c.cell_a), "ucell_b": float(c.cell_b), "ucell_c": float(c.cell_c),
        "ucell_alpha": math.radians(float(c.cell_alpha)), "ucell_beta": math.radians(float(c.cell_beta)),
        "ucell_gamma": math.radians(float(c.cell_gamma)), "RotXYZ": [math.radians(float(v)) for v in misset],
        "spec": [float(m.beam_params.lambda_offset), math.exp(float(m.beam_params.delta_log_lambda_scale))],
        "Nvol": float(np.linalg.det(np.array([[n[0], nd[0], nd[2]], [nd[0], n[1], nd[1]], [nd[2], nd[1], n[2]]]))),
    }


CENTERS = {"G": 2.0, "Nabc": [5.0, 6.0, 7.0], "Ndef": [0.5, -0.5, 0.2], "eta_abc": [0.04, 0.05, 0.06],
           "ucell_a": 39.5, "ucell_b": 50.5, "ucell_c": 61.0, "ucell_alpha": 89.0, "ucell_beta": 94.0,
           "ucell_gamma": 90.0, "spec": [0.01, 1.02], "Nvol": 100.0}
BETAS = {"G": 4.0, "Nabc": [2.0, 3.0, 4.0], "Ndef": [1.0, 2.0, 3.0], "eta_abc": [1e-4, 2e-4, 3e-4],
         "ucell_a": 0.1, "ucell_b": 0.2, "ucell_c": 0.3, "ucell_alpha": 1e-4, "ucell_beta": 2e-4,
         "ucell_gamma": 3e-4, "RotXYZ": 1e-5, "spec": [1e-4, 1e-3], "Nvol": 50.0}


def expected(m, names):
    v = values(m)
    total = 0.0
    for name in names:
        center = CENTERS.get(name, 0.0 if name == "RotXYZ" else None)
        if name.startswith("ucell_") and name[6:] in ("alpha", "beta", "gamma"):
            center = math.radians(center)
        if name == "Nvol":
            total += 0.5 * (center - v[name]) ** 2 / BETAS[name]
            continue
        vals, cents, bets = (np.broadcast_to(np.asarray(x, dtype=float), np.shape(v[name]))
                             for x in (v[name], [0.0] * 3 if name == "RotXYZ" else center, BETAS[name]))
        total += sum(term(c, b, x) for c, b, x in zip(np.ravel(cents), np.ravel(bets), np.ravel(vals)))
    return total


@pytest.mark.parametrize("name", ["G", "Nabc", "Ndef", "ucell_a", "ucell_b", "ucell_c", "ucell_alpha",
                                  "ucell_beta", "ucell_gamma", "RotXYZ", "spec"])
def test_each_restraint_is_hoppers_term_at_the_models_value(name):
    m = model()
    centers = {name: CENTERS[name]} if name in CENTERS else {}
    got = hopper_restraints(m, centers, {name: BETAS[name]})
    assert float(got) == pytest.approx(expected(m, [name]), rel=1e-12)


def test_anisotropic_eta_restrains_each_axis():
    m = model(anisotropic_mosaicity=True)
    got = hopper_restraints(m, {"eta_abc": CENTERS["eta_abc"]}, {"eta_abc": BETAS["eta_abc"]})
    assert float(got) == pytest.approx(expected(m, ["eta_abc"]), rel=1e-12)


def test_isotropic_eta_takes_the_first_center():
    m = model()
    got = hopper_restraints(m, {"eta_abc": CENTERS["eta_abc"]}, {"eta_abc": BETAS["eta_abc"]})
    eta = float(m.crystal_params.eta()[0])
    assert float(got) == pytest.approx(term(CENTERS["eta_abc"][0], BETAS["eta_abc"][0], eta), rel=1e-12)


def test_nvol_uses_the_ndef_the_model_simulates():
    """Under the reciprocal metric Ndef is unrefinable and the lattice factor has none."""
    crystal, detector, beam = configs(spot_metric=SpotMetric.RECIPROCAL)
    m = ExperimentModel(crystal, detector, beam, dtype=DT)
    with torch.no_grad():
        m.crystal_params.delta_N_cells_def.fill_(0.7)
    n = [float(v) for v in m.crystal_params.build_config().N_cells]
    got = hopper_restraints(m, {"Nvol": 100.0}, {"Nvol": 50.0})
    assert float(got) == pytest.approx(0.5 * (100.0 - n[0] * n[1] * n[2]) ** 2 / 50.0, rel=1e-12)


def test_nvol_is_restrained_even_with_ncells_fixed():
    m = model()
    m.freeze()
    got = hopper_restraints(m, {"Nvol": CENTERS["Nvol"]}, {"Nvol": BETAS["Nvol"]})
    assert float(got) == pytest.approx(expected(m, ["Nvol"]), rel=1e-12)


def panel_model(**kwargs):
    """detz_shift refines on detectors given by their origin, as dxtbx panels are."""
    p = SimpleNamespace(get_fast_axis=lambda: (1.0, 0.0, 0.0), get_slow_axis=lambda: (0.0, -1.0, 0.0),
                        get_origin=lambda: (-1.6, 1.6, -60.0), get_pixel_size=lambda: (0.2, 0.2),
                        get_image_size=lambda: (16, 16))
    detector = detector_config_from_dxtbx_panel(p, (0.0, 0.0, -1.0), oversample=1)
    crystal, _, beam = configs()
    return ExperimentModel(crystal, detector, beam, dtype=DT, **kwargs)


def test_detz_shift_is_in_metres():
    m = panel_model()
    with torch.no_grad():
        m.detector_params.delta_detz_shift_mm.fill_(0.3)
    got = hopper_restraints(m, {"detz_shift": 1e-4}, {"detz_shift": 1e-8})
    assert float(got) == pytest.approx(term(1e-4, 1e-8, 0.3e-3), rel=1e-12)


def test_fixed_parameters_add_nothing():
    m = model()
    m.freeze().refine("scale")
    centers = {k: v for k, v in CENTERS.items() if k != "Nvol"}
    betas = {k: v for k, v in BETAS.items() if k != "Nvol"}
    assert float(hopper_restraints(m, centers, betas)) == pytest.approx(expected(m, ["G"]), rel=1e-12)


def test_only_free_cell_variables_are_restrained():
    """hopper restrains the crystal system's variables; a tied b follows a and has no restraint."""
    m = ExperimentModel(*configs(cell_b=40.0, cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0), dtype=DT,
                        crystal_system="tetragonal")
    names = [k for k in CENTERS if k.startswith("ucell_")]
    got = hopper_restraints(m, {k: CENTERS[k] for k in names}, {k: BETAS[k] for k in names})
    assert float(got) == pytest.approx(expected(m, ["ucell_a", "ucell_c"]), rel=1e-12)


def test_nothing_given_is_zero():
    assert float(hopper_restraints(model(), {}, {})) == 0.0


def test_a_center_needs_a_beta():
    with pytest.raises(ValueError, match="beta"):
        hopper_restraints(model(), {"G": 1.0}, {})


def test_unknown_names_are_refused():
    with pytest.raises(ValueError, match="Bfactor"):
        hopper_restraints(model(), {"Bfactor": 1.0}, {"Bfactor": 1.0})


class _Restraints(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self):
        return hopper_restraints(self.m, CENTERS, BETAS)


def test_restraints_are_differentiable():
    wrapped = _Restraints(model(anisotropic_mosaicity=True))
    names = ["m.crystal_params.delta_log_N_cells", "m.crystal_params.delta_N_cells_def",
             "m.beam_params.delta_log_spot_scale", "m.crystal_params.delta_log_a", "m.crystal_params.delta_alpha_raw",
             "m.crystal_params.delta_misset_raw", "m.crystal_params.delta_log_eta", "m.beam_params.lambda_offset",
             "m.beam_params.delta_log_lambda_scale"]
    params = dict(wrapped.named_parameters())
    start = tuple(params[n].detach().clone().requires_grad_(True) for n in names)
    assert gradcheck(lambda *x: functional_call(wrapped, dict(zip(names, x)), ()), start, eps=1e-7, atol=1e-6,
                     rtol=1e-6)


def test_refining_on_restraints_alone_lands_on_the_centers():
    m = model()
    m.freeze().refine("scale", "Ncells")
    centers, betas = {"G": 2.0, "Nabc": [5.0, 6.0, 7.0]}, {"G": 1.0, "Nabc": [1.0, 1.0, 1.0]}
    refine_lbfgsb(m, lambda: hopper_restraints(m, centers, betas))
    assert float(m.beam_params.build_config().spot_scale) == pytest.approx(2.0, rel=1e-6)
    np.testing.assert_allclose([float(v) for v in m.crystal_params.build_config().N_cells], [5, 6, 7], rtol=1e-6)


def test_serial_experiments_restrain_each_shot():
    shots = [model() for _ in range(2)]
    serial = SerialExperiment(shots)
    each = sum(float(hopper_restraints(s, {"G": 2.0}, {"G": 4.0})) for s in shots)
    assert float(hopper_restraints(serial, {"G": 2.0}, {"G": 4.0})) == pytest.approx(each, rel=1e-12)


def test_a_shared_detector_is_restrained_once():
    shots = [panel_model() for _ in range(3)]
    serial = SerialExperiment(shots, share_detector=True)
    with torch.no_grad():
        shots[0].detector_params.delta_detz_shift_mm.fill_(0.3)
    got = hopper_restraints(serial, {"detz_shift": 1e-4}, {"detz_shift": 1e-8})
    assert float(got) == pytest.approx(term(1e-4, 1e-8, 0.3e-3), rel=1e-12)
