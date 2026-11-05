"""V1 (plan section 5) at small size: each half alone and both jointly recover a noise-free toy shot.

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


@pytest.mark.parametrize("fit", sorted(v1.FITS))
def test_noise_free_recovery(fit):
    result = v1.run_fit(fit, size=24, poisson=False, seed=0)
    assert result["converged"], result["message"]
    for label, error in result["errors"].items():
        assert abs(error) < 1e-5, (label, error)
    assert math.isfinite(result["fisher_condition"]) and result["fisher_condition_scaled"] >= 1
    assert all(abs(c) <= 1 for c in result["correlations"].values())


def test_poisson_recovery_is_within_the_fisher_errors():
    result = v1.run_fit("joint", size=24, poisson=True, seed=3)  # L-BFGS-B may end on its line search here
    assert max(abs(z) for z in result["z"].values()) < 5, result["z"]
