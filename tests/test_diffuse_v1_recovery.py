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
