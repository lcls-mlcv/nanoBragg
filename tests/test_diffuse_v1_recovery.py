"""V1 (plan section 5) at small size: the separability gate on the toy shots.

scripts/diffuse/v1_toy_recovery.py is the gate; it runs at full size with Poisson noise elsewhere.
"""
import importlib.util
import math
import os
from pathlib import Path

os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import pytest

pytest.importorskip("scipy")
_PATH = Path(__file__).resolve().parent.parent / "scripts" / "diffuse" / "v1_toy_recovery.py"
_spec = importlib.util.spec_from_file_location("v1_toy_recovery", _PATH)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)


def test_the_separable_toy_separates_and_the_hidden_one_is_flagged():
    separable = v1.separability(v1.TOYS["separable"], 24)
    hidden = v1.separability(v1.TOYS["hidden"], 24)
    assert max(separable.values()) <= v1.SEPARABILITY_LIMIT, separable
    assert max(hidden.values()) > 5 * v1.SEPARABILITY_LIMIT, hidden  # 14 measured: the halo trades off with the spots
    assert v1.TOYS["separable"].sigma > 5 * v1.TOYS["separable"].spot_sd
    assert v1.TOYS["hidden"].sigma < 2 * v1.TOYS["hidden"].spot_sd


@pytest.mark.parametrize("fit", sorted(v1.FITS))
def test_noise_free_recovery_of_the_separable_toy(fit):
    result = v1.run_fit(fit, size=24, poisson=False, seed=0)
    # Noise-free, the loss reaches the floating-point floor at the truth, and L-BFGS-B may then end
    # on a failed line search ("ABNORMAL") rather than on its tolerance, depending on the scipy
    # build: measured with errors of 2e-10. The recovery errors below are the test; running out of
    # iterations or evaluations is still a failure.
    assert result["converged"] or result["message"].startswith("ABNORMAL"), result["message"]
    for label, error in result["errors"].items():
        assert abs(error) < 1e-5, (label, error)
    assert math.isfinite(result["fisher_condition"]) and result["fisher_condition_scaled"] >= 1
    assert all(abs(c) <= 1 for c in result["correlations"].values())


def test_the_gate_passes_with_its_exit_code_and_writes_its_report(tmp_path):
    out = tmp_path / "v1.json"
    assert v1.main(["--size", "24", "--out", str(out)]) == 0
    import json
    report = json.loads(out.read_text())
    assert report["separable"]["passed"] and not report["hidden"]["separable"]


def test_poisson_recovery_is_within_the_fisher_errors():
    report = v1.evaluate("separable", 24, poisson=True, seeds=[3])
    assert report["passed"], (report["worst_ratio"], report["worst_recovery"])


# --grid (plan D3): the halo replaced by its DiffuseGrid, values frozen; k_d frozen (the default) or free

import dataclasses


@pytest.mark.parametrize("kd", ["frozen", "free"])
def test_on_the_grid_the_separable_toy_separates_and_the_hidden_one_is_flagged_if_k_d_is_free(kd):
    """k_d free: the hidden toy's k_d trades with the spots (5.95). k_d frozen: the diffuse term has no
    parameter of its own, so nothing trades with the spots and every ratio is about 1."""
    grid = v1.GridSpec(v1.grid_step_for(v1.TOYS["separable"], 24, kd), kd)
    separable = v1.separability(v1.TOYS["separable"], 24, grid=grid)
    hidden = v1.separability(v1.TOYS["hidden"], 24, grid=grid)
    assert max(separable.values()) <= v1.SEPARABILITY_LIMIT, separable
    if kd == "free":
        assert max(hidden.values()) > v1.SEPARABILITY_LIMIT, hidden
    else:
        assert max(hidden.values()) < 1.01, hidden


def test_the_grid_gate_with_k_d_free_requires_the_flag():
    assert v1.main(["--size", "24", "--grid", "--grid-kd", "free"]) == 0


def test_the_step_rule_is_the_bias_law_solved_for_the_target():
    """predicted z = K (step / sigma)^2 sqrt(photons); the step is the largest 1/(2n) with z <= the target."""
    toy = v1.TOYS["separable"]
    photons = v1.diffuse_photons(toy, 24)
    for kd in ("frozen", "free"):
        step = v1.grid_step_for(toy, 24, kd)
        assert abs(round(1 / (2 * step)) - 1 / (2 * step)) < 1e-9  # a half-integer divisor: nodes on the kinks
        assert v1.predicted_grid_bias(step, toy.sigma, photons, kd) <= v1.GRID_BIAS_TARGET
        coarser = 1 / (1 / step - 2)
        assert v1.predicted_grid_bias(coarser, toy.sigma, photons, kd) > v1.GRID_BIAS_TARGET
    assert v1.GRID_BIAS_TARGET <= v1.GRID_BIAS_LIMIT / 2


# fluence scales, per k_d mode: the rule's step changes with the photons and the bias stays under the limit
FLUENCES = [("frozen", 1.0), ("frozen", 4.0), ("free", 1 / 16), ("free", 1 / 4)]


@pytest.mark.parametrize("kd,scale", FLUENCES)
@pytest.mark.parametrize("fit", ["diffuse", "joint"])
def test_noise_free_on_the_grid_the_bias_follows_the_rule(kd, scale, fit):
    """Data from the exact halo, model on the grid at the rule's step for these photons."""
    base = v1.TOYS["separable"]
    toy = dataclasses.replace(base, fluence=base.fluence * scale)
    step = v1.grid_step_for(toy, 24, kd)
    result = v1.run_fit(fit, size=24, poisson=False, seed=0, toy=toy, grid=v1.GridSpec(step, kd))
    # not result["converged"]: on the interpolant's kinks L-BFGS-B may end in its line search (see the script)
    assert result["iterations"] > 10
    assert ("diffuse_params.delta_log_scale" in result["errors"]) == (kd == "free")
    worst = max(abs(z) for z in result["z"].values())
    assert worst < v1.GRID_BIAS_LIMIT, (step, result["z"])
    # the law predicts the worst bias to within a factor of 1.5 (it is fitted on the worst fit, the diffuse one)
    predicted = v1.predicted_grid_bias(step, toy.sigma, v1.diffuse_photons(toy, 24), kd)
    assert worst < 1.5 * predicted


def test_the_rule_gives_a_finer_step_for_more_photons():
    base = v1.TOYS["separable"]
    steps = [v1.grid_step_for(dataclasses.replace(base, fluence=base.fluence * k), 24) for k in (1.0, 4.0)]
    assert steps[1] < steps[0]


def test_the_grid_gate_passes_with_its_exit_code():
    assert v1.main(["--size", "24", "--grid"]) == 0


# --grid-core (D3b): the halo is the grid's core, refinable; the residual (zero for this toy) on the grid

@pytest.mark.parametrize("fit", sorted(v1.FITS))
def test_noise_free_recovery_with_the_halo_as_the_grids_core(fit):
    grid = v1.GridSpec(1 / 8, kd="frozen", core=True)
    result = v1.run_fit(fit, size=24, poisson=False, seed=0, grid=grid)
    if fit != "bragg":
        assert "diffuse_params.model.core.log_sigma" in result["errors"]
    for label, error in result["errors"].items():
        assert abs(error) < 1e-5, (label, error)


def test_the_core_gate_passes_and_flags_the_hidden_toy():
    assert v1.main(["--size", "24", "--grid", "0.125", "--grid-core"]) == 0


# --sharp (D3b audit): a Lorentzian core (kappa = 0.5 / N) with a smooth residual the grid carries,
# under the bias law for the residual: z = K_core step^2 sqrt(P)

def test_the_sharp_toys_step_follows_the_residuals_bias_law():
    toy = v1.EXTRA_TOYS["sharp"]
    photons = v1.diffuse_photons(toy, 24)
    step = v1.grid_step_for(toy, 24, core=True)
    assert v1.predicted_grid_bias(step, toy.sigma, photons, core=True) <= v1.GRID_BIAS_TARGET
    assert v1.predicted_grid_bias(1 / (1 / step - 2), toy.sigma, photons, core=True) > v1.GRID_BIAS_TARGET


@pytest.mark.parametrize("fit", ["diffuse", "joint"])
def test_noise_free_on_the_cored_grid_the_sharp_toy_meets_the_bias_rule(fit):
    toy = v1.EXTRA_TOYS["sharp"]
    step = v1.grid_step_for(toy, 24, core=True)
    result = v1.run_fit(fit, size=24, poisson=False, seed=0, toy=toy, grid=v1.GridSpec(step, core=True))
    assert "diffuse_params.model.core.log_kappa" in result["errors"]
    worst = max(abs(z) for z in result["z"].values())
    predicted = v1.predicted_grid_bias(step, toy.sigma, v1.diffuse_photons(toy, 24), core=True)
    assert worst < v1.GRID_BIAS_LIMIT and worst < 1.5 * predicted, result["z"]


def test_the_sharp_gate_passes_on_the_bias_and_reports_the_core_as_weakly_identified():
    """kappa = 0.05 is 1.25 spot sds: like the hidden toy its core trades with the spots (about 10)."""
    report = v1.evaluate("sharp", 24, poisson=False, seeds=[0],
                         grid=v1.GridSpec(v1.grid_step_for(v1.EXTRA_TOYS["sharp"], 24, core=True), core=True))
    assert report["recovered"] and not report["separable"]
    assert v1.main(["--size", "24", "--sharp"]) == 0


def test_the_cored_grids_residual_is_built_at_the_fits_start_and_the_drift_is_recorded():
    """As a real fit would: the leftover is sampled at the starting core, so the drift counts from there."""
    toy = v1.EXTRA_TOYS["sharp"]
    model = v1.build(toy, 24, diffuse=True, bragg=False, grid=v1.GridSpec(1 / 4, core=True))
    assert model.diffuse.core_drift() == 0.0
    result = v1.run_fit("diffuse", size=24, poisson=False, seed=0, toy=toy, grid=v1.GridSpec(1 / 4, core=True))
    # the fit moves the amplitude from 0.7 x and kappa from 1.3 x the truth: ln(1.3) = 0.26, ln(1 / 0.7) = 0.36
    assert result["core_drift"] == pytest.approx(math.log(1 / 0.7), abs=0.01)


def test_the_wide_sharp_toy_separates():
    """A Lorentzian's wings separate from the spots only well past them: measured 2026-11-05 (24^2, no
    spot mask), worst ratio 4.96, 2.43, 1.82, 1.45, 1.33 at kappa 0.1, 0.2, 0.3, 0.5, 0.7."""
    toy = v1.EXTRA_TOYS["sharp_wide"]
    assert toy.sigma / toy.spot_sd >= 5
    ratios = v1.separability(toy, 24, grid=v1.GridSpec(v1.grid_step_for(toy, 24, core=True), core=True))
    assert max(ratios.values()) <= v1.SEPARABILITY_LIMIT, ratios


def test_poisson_recovery_of_the_wide_sharp_toy_on_the_cored_grid():
    toy = v1.EXTRA_TOYS["sharp_wide"]
    report = v1.evaluate("sharp_wide", 24, poisson=True, seeds=[3],
                         grid=v1.GridSpec(v1.grid_step_for(toy, 24, core=True), core=True))
    assert report["passed"], (report["worst_ratio"], report["worst_recovery"])


def _photons(toy, size, oversample=1):
    """(diffuse, Bragg) photons of the joint truth image at a size."""
    import torch
    model = v1.truth_model(toy, "joint", size, oversample=oversample)
    with torch.no_grad():
        model()
    return float(model.components["diffuse"].sum()), float(model.components["bragg"].sum())


def test_photons_per_solid_angle_do_not_change_with_size():
    """The detector's extent is fixed (12.8 mm at 60 mm); a finer grid of pixels must split the same
    photons, not lose them as size^-2 (the first version scaled the fluence by (24 / size)^2 as well).

    The halo is compared at pixel centres, 24^2 against 48^2. The spots (sd 0.04 in h) are compared
    integrated (subpixels 0.033 in h apart) and at 48^2 against 96^2: at 24^2 one spot sits on the
    detector's edge, and MOSFLM's beam centre, which moves by half a pixel with the pixel size, takes
    4.8% of the Bragg photons off it (measured: 3.589e6 at 24^2 whatever the oversampling, 3.769e6 and
    3.770e6 at 48^2 and 96^2)."""
    toy = v1.TOYS["separable"]
    assert _photons(toy, 48)[0] == pytest.approx(_photons(toy, 24)[0], rel=0.05)
    assert _photons(toy, 96, oversample=2)[1] == pytest.approx(_photons(toy, 48, oversample=4)[1], rel=0.05)
