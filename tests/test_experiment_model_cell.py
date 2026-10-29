"""Unit-cell refinement constrained by crystal system, and through an orientation matrix (plan C2)."""
import math

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.utils.unit_cell import B_matrix, FREE_VARIABLES, check_cell

DTYPE = torch.float64

CELLS = {
    "triclinic": (61.0, 73.0, 84.0, 87.0, 95.0, 101.0),
    "monoclinic": (70.0, 60.0, 50.0, 90.0, 110.0, 90.0),
    "orthorhombic": (55.0, 65.0, 77.0, 90.0, 90.0, 90.0),
    "tetragonal": (55.0, 55.0, 77.0, 90.0, 90.0, 90.0),
    "hexagonal": (55.0, 55.0, 77.0, 90.0, 90.0, 120.0),
    "rhombohedral": (60.0, 60.0, 60.0, 80.0, 80.0, 80.0),
    "cubic": (60.0, 60.0, 60.0, 90.0, 90.0, 90.0),
}
CELL_PARAMS = ["delta_log_a", "delta_log_b", "delta_log_c",
               "delta_alpha_raw", "delta_beta_raw", "delta_gamma_raw"]


def _metric(cell):
    a, b, c = cell[:3]
    al, be, ga = np.radians(cell[3:])
    return np.array([[a * a, a * b * np.cos(ga), a * c * np.cos(be)],
                     [a * b * np.cos(ga), b * b, b * c * np.cos(al)],
                     [a * c * np.cos(be), b * c * np.cos(al), c * c]])


def _configs(cell, A=None):
    a, b, c, al, be, ga = cell
    kwargs = {}
    if A is not None:
        kwargs = dict(mosflm_a_star=A[:, 0], mosflm_b_star=A[:, 1], mosflm_c_star=A[:, 2])
    crystal = CrystalConfig(cell_a=a, cell_b=b, cell_c=c, cell_alpha=al, cell_beta=be, cell_gamma=ga,
                            N_cells=(4, 4, 4), default_F=100.0, shape=CrystalShape.GAUSS, **kwargs)
    detector = DetectorConfig(distance_mm=80.0, pixel_size_mm=0.2, spixels=48, fpixels=48,
                              detector_convention=DetectorConvention.MOSFLM,
                              detector_pivot=DetectorPivot.BEAM, oversample=1)
    beam = BeamConfig(wavelength_A=1.2, fluence=1e24)
    return crystal, detector, beam


def _oriented(cell):
    """Configs oriented by A = U·B(cell) for a fixed, non-trivial U."""
    axis = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    U = np.eye(3) + math.sin(0.7) * K + (1 - math.cos(0.7)) * K @ K
    A = U @ B_matrix(*cell).numpy()
    return _configs(cell, A), torch.as_tensor(U)


@pytest.mark.parametrize("system", sorted(CELLS))
def test_B_matrix_reproduces_the_reciprocal_metric(system):
    cell = CELLS[system]
    B = B_matrix(*cell).numpy()
    np.testing.assert_allclose(B.T @ B, np.linalg.inv(_metric(cell)), rtol=1e-12, atol=1e-18)
    assert np.allclose(np.triu(B, 1), 0.0)  # cctbx's B is lower triangular


@pytest.mark.parametrize("system", sorted(CELLS))
def test_only_the_free_cell_variables_train_and_ties_hold(system):
    model = ExperimentModel(*_configs(CELLS[system]), dtype=DTYPE, crystal_system=system)
    params = model.crystal_params
    free = FREE_VARIABLES[system]
    trainable = {n for n in CELL_PARAMS if getattr(params, n).requires_grad}
    assert len(trainable) == len(free)
    blocked_cell = {n for n in model.unrefinable if n.split(".")[-1] in CELL_PARAMS}
    assert blocked_cell == {f"crystal_params.{n}" for n in CELL_PARAMS if n not in trainable}

    model.freeze().refine("cell")  # a group refines its refinable members only
    assert {n for n in CELL_PARAMS if getattr(params, n).requires_grad} == trainable

    with torch.no_grad():  # move every cell parameter, tied ones included
        for i, n in enumerate(CELL_PARAMS):
            getattr(params, n).fill_(0.02 * (i + 1))
    cfg = params.build_config()
    cell = [float(getattr(cfg, f"cell_{v}")) for v in ("a", "b", "c", "alpha", "beta", "gamma")]
    check_cell(system, cell)
    assert cell != list(CELLS[system])


def test_tied_parameter_cannot_be_refined():
    model = ExperimentModel(*_configs(CELLS["tetragonal"]), dtype=DTYPE, crystal_system="tetragonal")
    with pytest.raises(ValueError, match="b is not a free variable of a tetragonal cell"):
        model.refine("crystal_params.delta_log_b")
    model.freeze().refine()
    assert not model.crystal_params.delta_log_b.requires_grad


def test_cell_that_breaks_the_crystal_system_is_rejected():
    with pytest.raises(ValueError, match="not tetragonal"):
        ExperimentModel(*_configs(CELLS["orthorhombic"]), dtype=DTYPE, crystal_system="tetragonal")
    with pytest.raises(ValueError, match="unknown crystal system"):
        ExperimentModel(*_configs(CELLS["cubic"]), dtype=DTYPE, crystal_system="isometric")


def test_trigonal_refines_like_hexagonal():
    model = ExperimentModel(*_configs(CELLS["hexagonal"]), dtype=DTYPE, crystal_system="trigonal")
    assert model.crystal_params.crystal_system == "hexagonal"


@pytest.mark.parametrize("system", ["monoclinic", "tetragonal"])
def test_matrix_oriented_cell_has_live_gradients(system):
    """With an orientation matrix Crystal ignores cell_*; the cell must reach A = U·B(cell)."""
    configs, U = _oriented(CELLS[system])
    model = ExperimentModel(*configs, dtype=DTYPE, crystal_system=system)
    torch.testing.assert_close(model.crystal_params.U, U, rtol=0, atol=1e-12)

    image = model()
    assert image.sum() > 0
    image.sum().backward()
    for var in FREE_VARIABLES[system]:
        name = model.crystal_params._CELL_PARAMS[var]
        grad = getattr(model.crystal_params, name).grad
        assert grad is not None and grad.abs() > 0, f"d/d{var} is {grad}"


def test_matrix_oriented_cell_step_matches_a_fresh_orientation_matrix():
    """Moving a and beta by hand-made steps equals building A = U·B(new cell) directly."""
    cell = CELLS["monoclinic"]
    configs, U = _oriented(cell)
    model = ExperimentModel(*configs, dtype=DTYPE, crystal_system="monoclinic")
    model()  # build the persistent objects first, so the step goes through _sync
    with torch.no_grad():
        model.crystal_params.delta_log_a.fill_(math.log(1.01))
        model.crystal_params.delta_beta_raw.fill_(math.atanh(0.15))  # beta + 1.5 deg
        stepped = model()
    new_cell = (cell[0] * 1.01, cell[1], cell[2], 90.0, cell[4] + 1.5, 90.0)
    fresh = ExperimentModel(*_configs(new_cell, (U @ B_matrix(*new_cell)).numpy()), dtype=DTYPE,
                            crystal_system="monoclinic", param_init="frozen")
    with torch.no_grad():
        torch.testing.assert_close(stepped, fresh(), rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("name", ["crystal_params.delta_log_a", "crystal_params.delta_beta_raw"])
def test_gradcheck_matrix_oriented_cell(name):
    configs, _ = _oriented(CELLS["monoclinic"])
    model = ExperimentModel(*configs, dtype=DTYPE, crystal_system="monoclinic")
    start = torch.full_like(dict(model.named_parameters())[name], 0.003).requires_grad_(True)

    def loss(value):
        return functional_call(model, {name: value}, ()).sum()

    assert gradcheck(loss, (start,), eps=1e-6, atol=1e-8, rtol=1e-4)


def test_cell_that_does_not_match_the_orientation_matrix_is_rejected():
    (crystal, detector, beam), _ = _oriented(CELLS["monoclinic"])
    crystal.cell_beta = 100.0
    with pytest.raises(ValueError, match="does not match the orientation matrix"):
        ExperimentModel(crystal, detector, beam, dtype=DTYPE, crystal_system="monoclinic")


@pytest.mark.parametrize("cctbx_name, cell, expected", [
    ("Monoclinic", CELLS["monoclinic"], "monoclinic"),
    ("Cubic", CELLS["cubic"], "cubic"),
    ("Trigonal", CELLS["hexagonal"], "hexagonal"),
    ("Trigonal", CELLS["rhombohedral"], "rhombohedral"),
])
def test_crystal_system_of_dxtbx_crystal(cctbx_name, cell, expected):
    from types import SimpleNamespace
    from nanobrag_torch.compat.cctbx import crystal_system_of

    crystal = SimpleNamespace(
        get_space_group=lambda: SimpleNamespace(crystal_system=lambda: cctbx_name),
        get_unit_cell=lambda: SimpleNamespace(parameters=lambda: cell),
    )
    assert crystal_system_of(crystal) == expected
