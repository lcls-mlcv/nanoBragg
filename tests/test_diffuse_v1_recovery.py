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


# --grid (plan D3): the halo replaced by its DiffuseGrid, values frozen, k_d as its amplitude

def test_on_the_grid_the_separable_toy_separates_and_the_hidden_one_is_flagged():
    separable = v1.separability(v1.TOYS["separable"], 24, grid_step=v1.GRID_STEP)
    hidden = v1.separability(v1.TOYS["hidden"], 24, grid_step=v1.GRID_STEP)
    assert max(separable.values()) <= v1.SEPARABILITY_LIMIT, separable
    assert max(hidden.values()) > v1.SEPARABILITY_LIMIT, hidden


@pytest.mark.parametrize("fit", ["diffuse", "joint"])
def test_noise_free_on_the_grid_the_bias_is_under_a_tenth_of_an_sd(fit):
    """Data from the exact halo, model on the grid: the grid's error is a bias, small against the noise."""
    result = v1.run_fit(fit, size=24, poisson=False, seed=0, grid_step=v1.GRID_STEP)
    # not result["converged"]: on the interpolant's kinks L-BFGS-B may end in its line search (see the script)
    assert result["iterations"] > 10
    assert "diffuse_params.delta_log_scale" in result["errors"]
    for label, z in result["z"].items():
        assert abs(z) < v1.GRID_BIAS_LIMIT, (label, z, result["errors"][label])


def test_the_grid_gate_passes_with_its_exit_code():
    assert v1.main(["--size", "24", "--grid"]) == 0


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
