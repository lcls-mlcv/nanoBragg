"""scripts/diffuse/core_candidates.py's linearised bias follows V1's current rules (final D3b audit):
the diffuse-alone fit has the geometry given (so with k_d frozen there is none), scale-like parameters
(G, k_d) are judged by their absolute bias, the rest by z at 1e6 diffuse photons."""
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "diffuse" / "core_candidates.py"
_spec = importlib.util.spec_from_file_location("core_candidates", _PATH)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def test_k_d_frozen_has_only_the_joint_fit_with_g_judged_absolutely():
    model = cc.SharpCore()
    parts = cc.linear_bias_parts(model, model, 16, "frozen")
    assert set(parts["z"]) == {"joint"} and set(parts["scale"]) == {"joint"}
    assert "delta_log_spot_scale" not in parts["z"]["joint"]
    assert set(parts["scale"]["joint"]) == {"delta_log_spot_scale"}
    assert {"delta_log_a", "delta_misset_raw[0]"} <= set(parts["z"]["joint"])
    assert all(abs(v) < 1e-9 for fit in parts["z"].values() for v in fit.values())  # the exact model: no bias


def test_k_d_free_adds_a_diffuse_alone_fit_of_k_d_only():
    model = cc.SharpCore()
    parts = cc.linear_bias_parts(model, model, 16, "free")
    assert set(parts["z"]["diffuse"]) == set() and set(parts["scale"]["diffuse"]) == {"delta_log_scale"}
    assert set(parts["scale"]["joint"]) == {"delta_log_spot_scale", "delta_log_scale"}


def test_linear_bias_keeps_its_shape_for_callers():
    """{fit: {param: z}} as before (eryx's core_bias.py takes the worst value), now the non-scale z only."""
    model = cc.SharpCore()
    z = cc.linear_bias(model, model, 16, "frozen")
    assert z == cc.linear_bias_parts(model, model, 16, "frozen")["z"]
