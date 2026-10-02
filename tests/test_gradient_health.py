"""Gradient health sweep across every geometry in the parity matrix.

Why this exists
---------------
The parity matrix compares *forward images* against the C oracle. It cannot
see a broken backward pass. PR #25 fixed an `atan2(0, 0)` in
`polarization_factor` that put NaN through the whole backward pass at exactly
-forward scattering while the forward image stayed numerically perfect — 171
parity runs were green throughout, because none of them ever called
`.backward()`.

This module closes that gap. For every configuration the parity matrix already
exercises, it runs one forward pass, reduces to a scalar, calls `.backward()`,
and asserts that each differentiable parameter's gradient is finite and not
silently detached. It is a *health* check, not an accuracy check:
`test_gradients.py` owns numerical correctness via `torch.autograd.gradcheck`
at a handful of configurations; this owns "no NaN, no Inf, no dead graph"
across all of them.

Running it
----------
    NB_RUN_GRAD_SWEEP=1 KMP_DUPLICATE_LIB_OK=TRUE NANOBRAGG_DISABLE_COMPILE=1 \
    PYTHONPATH=src python3 -m pytest tests/test_gradient_health.py -q

Without `NB_RUN_GRAD_SWEEP=1` only the `SMOKE_CASES` subset runs, so the
default suite stays fast. The sweep is CPU/float64 by construction: gradient
health is a property of the graph, and float32 would confound a genuine NaN
with ordinary underflow.
"""

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch
import yaml

from nanobrag_torch.__main__ import build_simulation, create_parser

REPO_ROOT = Path(__file__).resolve().parent.parent

# Parity runs kept in the default (non-sweep) suite. One per geometry family,
# chosen to cover the distinct code paths rather than to be fast.
SMOKE_CASES = {
    "AT-PARALLEL-001-detpixels-64",
    "AT-PARALLEL-004-mosflm",
    "AT-PARALLEL-012-simple_cubic",
    "PARITY-SHAPE-001-tophat",
}

# Configurations where a parameter legitimately has no gradient. Each entry is
# a claim about the physics or the parameterisation, verified against the code,
# not a way to silence a failure — the sweep asserts the dead set *exactly*, so
# a parameter that starts or stops being differentiable here fails the test
# either way.
#
# TOPHAT: `F_latt` is `torch.where(rad_sqr * fudge < 0.3969, Na*Nb*Nc, 0)`
# (simulator.py:366-369). `rad_sqr` carries the cell/orientation dependence but
# enters only through a boolean, so F_latt contributes no cell gradient. That is
# mathematically right — a top hat is a step function, its derivative is zero
# almost everywhere — and it matches nanoBragg.c's binary cutoff, so it is
# recorded rather than fixed: "fixing" it means softening the cutoff, a
# deliberate divergence from the C oracle.
#
# IMPORTANT, and initially got wrong here: this severs F_latt's cell dependence,
# not the whole intensity's. I = (F_cell * F_latt)^2, and when F_cell comes from
# an *interpolated* hkl lookup it is evaluated at fractional h,k,l derived from
# the cell, so the cell gradients are live and comparable to GAUSS's (measured
# d/d cell_a = -8.84e-05 for TOPHAT against -8.30e-05 for GAUSS on the same
# geometry). Interpolation auto-enables whenever any N_cells <= 2, so that path
# needs no flag. `_cell_reaches_f_cell` encodes the distinction.
#
# How it fails when the cell genuinely is severed depends on what else is
# differentiable. With another live parameter (any detector term, as this sweep
# always installs) the backward pass succeeds and returns `None` for the six cell
# parameters. With *only* cell parameters differentiable the loss has no
# grad_fn at all and `.backward()` raises RuntimeError. The original audit
# finding said "graph-less image, so .backward() raises" and was right; an
# earlier version of this comment claimed otherwise.
TOPHAT_DEAD_CELL_PARAMS = frozenset(
    {"cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma"}
)


# --- What we differentiate -------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """One differentiable config field.

    `owner` selects which of the three configs holds it; `field` is the
    attribute name; `nudge` moves the value off an exact stationary point where
    one would otherwise make a zero gradient indistinguishable from a dead
    graph (90° cell angles, 0° rotations).
    """

    name: str
    owner: str  # "crystal" | "detector" | "beam"
    field: str
    nudge: float = 0.0


# KNOWN LIMIT of the nudges below: every case is evaluated 1 degree off a 90
# degree cell angle and 0.5 degrees off an unrotated, zero-twotheta detector. So
# the sweep does NOT evaluate exactly-orthogonal geometry, and cannot catch a
# gradient bug that appears only there. The nudges exist because at an exact
# stationary point a true zero gradient is indistinguishable from a dead graph,
# which would make the dead-parameter assertions meaningless. The honest
# description is "the parity geometries, perturbed off their stationary points",
# not "every configuration the parity matrix exercises". Exactly-degenerate
# geometry is covered by construction instead, in
# test_degenerate_geometry_gradient_health below.


PARAMS: List[ParamSpec] = [
    # Cell. Angles are nudged off 90°, which is a genuine stationary point.
    ParamSpec("cell_a", "crystal", "cell_a"),
    ParamSpec("cell_b", "crystal", "cell_b"),
    ParamSpec("cell_c", "crystal", "cell_c"),
    ParamSpec("cell_alpha", "crystal", "cell_alpha", nudge=-1.0),
    ParamSpec("cell_beta", "crystal", "cell_beta", nudge=-1.0),
    ParamSpec("cell_gamma", "crystal", "cell_gamma", nudge=-1.0),
    # Detector geometry. These are the parameters the NaN bug destroyed, and
    # the ones a refinement loop actually moves.
    #
    # `distance_mm` and `close_distance_mm` are a mutually exclusive pair: when
    # `-close_distance` (or an SMV `CLOSE_DISTANCE` header) is given, the
    # detector derives `distance = close_distance / ratio` and `distance_mm`
    # stops being an input. Measured: whichever one drives the configuration
    # carries the identical gradient, and the other is `None`. `_expected_dead`
    # encodes that, so the sweep checks the pair rather than each in isolation.
    ParamSpec("distance_mm", "detector", "distance_mm"),
    ParamSpec("close_distance_mm", "detector", "close_distance_mm"),
    ParamSpec("rotx", "detector", "detector_rotx_deg", nudge=0.5),
    ParamSpec("roty", "detector", "detector_roty_deg", nudge=0.5),
    ParamSpec("rotz", "detector", "detector_rotz_deg", nudge=0.5),
    ParamSpec("twotheta", "detector", "detector_twotheta_deg", nudge=0.5),
]

_OWNER_ATTR = {
    "crystal": "crystal_config",
    "detector": "detector_config",
    "beam": "beam_config",
}


# --- Case loading ----------------------------------------------------------


def load_parity_runs() -> List[tuple]:
    """Flatten parity_cases.yaml into (run_id, argv) pairs."""
    with open(REPO_ROOT / "tests" / "parity_cases.yaml") as fh:
        data = yaml.safe_load(fh)

    runs = []
    for case in data["cases"]:
        base = case["base_args"].strip().split()
        for run in case["runs"]:
            extra = run["extra_args"].strip().split()
            argv = [a.replace("{REPO}", str(REPO_ROOT)) for a in base + extra]
            runs.append((f"{case['id']}-{run['name']}", argv))
    return runs


def sweep_enabled() -> bool:
    return os.environ.get("NB_RUN_GRAD_SWEEP") == "1"


def pytest_generate_tests(metafunc):
    if "grad_case" not in metafunc.fixturenames:
        return

    runs = load_parity_runs()

    # A mistyped id used to degrade silently to `runs[:1]`, so the default suite
    # ran one arbitrary case while claiming to cover one geometry per family.
    # Two of the three original ids did not exist. Fail loudly instead.
    known = {r[0] for r in runs}
    missing = SMOKE_CASES - known
    assert not missing, (
        f"SMOKE_CASES names runs that are not in parity_cases.yaml: "
        f"{sorted(missing)} — fix the ids rather than letting the default "
        f"suite quietly shrink"
    )

    if not sweep_enabled():
        runs = [r for r in runs if r[0] in SMOKE_CASES]

    metafunc.parametrize("grad_case", runs, ids=[r[0] for r in runs])


# --- The differentiable build ----------------------------------------------


def _shrink(argv: List[str], max_pixels: int = 64) -> List[str]:
    """Cap detector size so a 171-run backward sweep finishes in minutes.

    Only `-detpixels` is touched. Beam centres are left exactly as the parity
    case sets them: the forward-scattering NaN was reachable precisely because
    the beam centre landed on a pixel centre, so rescaling centres here would
    be rescaling away the bug class this sweep exists to catch.
    """
    out = list(argv)
    for flag in ("-detpixels", "-detpixels_f", "-detpixels_s"):
        if flag in out:
            i = out.index(flag)
            if i + 1 < len(out):
                try:
                    out[i + 1] = str(min(int(out[i + 1]), max_pixels))
                except ValueError:
                    pass
    return out


def build_differentiable(argv: List[str]):
    """Build the models this argv produces, with leaf tensors in the configs.

    Returns (simulator, leaves, bundle). `leaves` maps ParamSpec.name to the
    leaf tensor placed in the config, for every parameter the configuration
    actually has — a field left at `None` by the CLI (an unset
    `-close_distance`, say) is not a parameter of that configuration and is
    not installed.
    """
    argv = _shrink(argv) + ["-dtype", "float64", "-device", "cpu"]

    parser = create_parser()
    args = parser.parse_args(argv)
    args._argv = argv

    leaves: Dict[str, torch.Tensor] = {}

    def hook(crystal_config, detector_config, beam_config):
        owners = {
            "crystal": crystal_config,
            "detector": detector_config,
            "beam": beam_config,
        }
        for spec in PARAMS:
            cfg = owners[spec.owner]
            if cfg is None or not hasattr(cfg, spec.field):
                continue
            current = getattr(cfg, spec.field)
            if isinstance(current, torch.Tensor):
                current = current.detach().item()
            try:
                value = float(current) + spec.nudge
            except (TypeError, ValueError):
                continue
            leaf = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            setattr(cfg, spec.field, leaf)
            leaves[spec.name] = leaf

    bundle = build_simulation(args, config_hook=hook)
    return bundle.simulator, leaves, bundle


def _expected_dead(leaves: Dict[str, torch.Tensor], bundle) -> set:
    """Parameters this configuration is *expected* to leave without a gradient.

    Every entry is a verified property of the configuration, not a waiver; see
    the module-level notes on the distance pair and on TOPHAT.
    """
    dead = set()

    # Exactly one of the distance pair drives the geometry.
    if "close_distance_mm" in leaves:
        dead.add("distance_mm")

    # TOPHAT severs the cell dependence of F_latt, but NOT of F_cell. When the
    # structure factors come from an interpolated hkl lookup, F_cell is
    # evaluated at fractional h,k,l derived from the cell, so the cell
    # gradients are alive and of the same magnitude as GAUSS's. Waiving them on
    # shape alone would fail the sweep on correct behaviour.
    shape = getattr(bundle.crystal_config, "shape", None)
    if shape is not None and getattr(shape, "name", "") == "TOPHAT":
        if not _cell_reaches_f_cell(bundle.crystal):
            dead |= {n for n in TOPHAT_DEAD_CELL_PARAMS if n in leaves}

    return dead


def _cell_reaches_f_cell(crystal) -> bool:
    """Whether F_cell carries a cell dependence for this crystal.

    True only for an interpolated hkl lookup: tricubic interpolation evaluates
    at fractional h,k,l, which are functions of the cell. A flat `default_F` is
    a constant, and a nearest-neighbour lookup rounds h,k,l to integers, which
    is non-differentiable in the same way the TOPHAT cutoff is.

    Note `interpolate` auto-enables for small crystals (any N_cells <= 2,
    mirroring nanoBragg.c), so this path is reachable with no flag at all.
    """
    if getattr(crystal, "hkl_data", None) is None:
        return False
    return bool(getattr(crystal, "interpolate", False))


# --- The test --------------------------------------------------------------


def test_gradient_health(grad_case):
    """One forward, one backward, per-parameter finite-gradient assertions."""
    run_id, argv = grad_case

    simulator, leaves, bundle = build_differentiable(argv)
    assert leaves, f"{run_id}: no differentiable parameters were installed"

    image = simulator.run()

    # A NaN in the forward image is a different (and louder) bug; say which one
    # we are looking at rather than letting it surface as a confusing backward
    # failure downstream.
    assert torch.isfinite(image).all(), (
        f"{run_id}: forward image is not finite "
        f"({torch.isnan(image).sum().item()} NaN, "
        f"{torch.isinf(image).sum().item()} Inf) — forward bug, not a gradient bug"
    )

    loss = image.sum()
    assert loss.requires_grad, f"{run_id}: loss is detached from every parameter"

    loss.backward()

    bad: List[str] = []
    dead: List[str] = []
    for name, leaf in sorted(leaves.items()):
        if leaf.grad is None:
            dead.append(name)
        elif not torch.isfinite(leaf.grad).all():
            bad.append(f"{name}={leaf.grad.item()!r}")

    assert not bad, (
        f"{run_id}: non-finite gradients: {', '.join(bad)}\n"
        f"argv: {shlex.join(argv)}"
    )

    expected = _expected_dead(leaves, bundle)
    got = set(dead)

    unexpectedly_dead = sorted(got - expected)
    assert not unexpectedly_dead, (
        f"{run_id}: no gradient reached: {', '.join(unexpectedly_dead)}\n"
        f"argv: {shlex.join(argv)}"
    )

    # The other direction matters too: if a parameter listed as expected-dead
    # starts producing a gradient, the semantics changed and the reasoning
    # recorded above is now stale.
    unexpectedly_live = sorted(expected - got)
    assert not unexpectedly_live, (
        f"{run_id}: expected no gradient but got one for "
        f"{', '.join(unexpectedly_live)} — the documented reason for this "
        f"configuration is stale, re-derive it before widening the sweep.\n"
        f"argv: {shlex.join(argv)}"
    )


# --- Degenerate geometries -------------------------------------------------
#
# The parity sweep above covers realistic geometries; it does NOT cover the
# exactly-degenerate ones, and measurement says it cannot. The
# forward-scattering NaN needs a sample whose direction equals the incident
# beam direction in exact floating point. Scanning `-Xbeam`/`-Ybeam` across
# MOSFLM, XDS, ADXV, DENZO and DIALS at every 0.05 mm from 1.40 to 1.80 on a
# 32x32/0.1 mm detector produces zero exactly-degenerate samples: the
# half-pixel beam-centre offset moves the axis off every pixel centre, and
# `sin^2(2theta)` rounding to 0.0 for a near-axis sample is NOT the same
# condition (E_out and B_out are ~1e-9 there, and atan2 is perfectly well
# behaved).
#
# So degeneracy is reached by construction, not by enumeration: these cases
# set `beam_center_f/s` on DetectorConfig directly, the way the geometry ends
# up when a caller drives the models as a library rather than through the CLI.
# Verified to fail with PR #25's guard reverted.

DEGENERATE_GEOMETRIES = [
    # (id, spixels, beam_centre_mm, oversample, kahn)
    ("beam-on-pixel-centre-32", 32, 1.6, 1, 1.0),
    ("beam-on-pixel-centre-64", 64, 3.2, 1, 1.0),
    ("beam-on-pixel-centre-kahn-partial", 32, 1.6, 1, 0.5),
    ("beam-on-pixel-centre-oversample-3", 32, 1.6, 3, 1.0),
]


@pytest.mark.parametrize(
    "case", DEGENERATE_GEOMETRIES, ids=[c[0] for c in DEGENERATE_GEOMETRIES]
)
def test_degenerate_geometry_gradient_health(case):
    """Gradients stay finite where the diffracted ray is exactly on the beam axis."""
    from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
    from nanobrag_torch.models import Crystal
    from nanobrag_torch.models.detector import Detector
    from nanobrag_torch.simulator import Simulator

    case_id, npx, centre, oversample, kahn = case

    distance = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    detector = Detector(
        DetectorConfig(
            distance_mm=distance,
            pixel_size_mm=0.1,
            spixels=npx,
            fpixels=npx,
            beam_center_f=centre,
            beam_center_s=centre,
        )
    )
    crystal_config = CrystalConfig(
        cell_a=100.0, cell_b=100.0, cell_c=100.0,
        cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0,
        default_F=100.0, N_cells=(5, 5, 5),
    )
    crystal = Crystal(crystal_config)
    beam = BeamConfig(wavelength_A=1.0, fluence=1e24, polarization_factor=kahn)

    simulator = Simulator(
        crystal, detector, crystal_config=crystal_config, beam_config=beam
    )
    image = simulator.run(oversample=oversample)

    assert torch.isfinite(image).all(), f"{case_id}: forward image is not finite"

    image.sum().backward()
    assert distance.grad is not None, f"{case_id}: no gradient reached distance"
    assert torch.isfinite(distance.grad), (
        f"{case_id}: d(sum I)/d(distance) = {distance.grad.item()!r} — a sample "
        f"exactly on the beam axis poisoned the backward pass while the "
        f"rendered image stayed clean"
    )




# --- The TOPHAT warning ----------------------------------------------------
#
# The warning lives on Simulator and fires at run() time, not in
# Crystal.__init__: `hkl_data` and `interpolate` are both set after the Crystal
# is built, so a construction-time check would warn in exactly the configuration
# where TOPHAT cell refinement works.

HKL_INTERP = REPO_ROOT / "tests" / "golden_data" / "P1_interp.hkl"


def _tophat_sim(shape_name="TOPHAT", requires_grad=True, hkl=False, interpolate=None,
                n_cells=(5, 5, 5)):
    from nanobrag_torch.config import (
        BeamConfig, CrystalConfig, CrystalShape, DetectorConfig,
    )
    from nanobrag_torch.models import Crystal
    from nanobrag_torch.models.detector import Detector
    from nanobrag_torch.simulator import Simulator

    def cell(v):
        return (
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            if requires_grad else v
        )

    cfg = CrystalConfig(
        cell_a=cell(20.0), cell_b=cell(20.0), cell_c=cell(20.0),
        cell_alpha=cell(89.0), cell_beta=cell(89.0), cell_gamma=cell(89.0),
        default_F=100.0, N_cells=n_cells, shape=CrystalShape[shape_name],
    )
    crystal = Crystal(cfg, dtype=torch.float64)
    if hkl:
        from nanobrag_torch.io.hkl import read_hkl_file
        arr, meta = read_hkl_file(str(HKL_INTERP))
        crystal.hkl_data = (
            arr if isinstance(arr, torch.Tensor)
            else torch.tensor(arr, dtype=torch.float64)
        )
        crystal.hkl_metadata = meta
    if interpolate is not None:
        crystal.interpolate = interpolate

    detector = Detector(
        DetectorConfig(distance_mm=100.0, pixel_size_mm=0.1, spixels=16, fpixels=16),
        dtype=torch.float64,
    )
    return Simulator(
        crystal, detector, crystal_config=cfg,
        beam_config=BeamConfig(wavelength_A=1.0, fluence=1e24), dtype=torch.float64,
    )


def test_tophat_warns_when_cell_parameters_require_grad():
    """The severed case is silent without this warning; say so up front."""
    sim = _tophat_sim()
    with pytest.warns(UserWarning, match="TOPHAT crystal shape has no gradient path"):
        sim.run(oversample=1)


def test_tophat_warning_fires_only_once_per_simulator():
    """run() is called repeatedly in a refinement loop; don't warn every step."""
    import warnings as _w

    sim = _tophat_sim()
    with pytest.warns(UserWarning):
        sim.run(oversample=1)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        sim.run(oversample=1)


def test_tophat_silent_without_differentiable_cell():
    """No warning for the ordinary forward-only TOPHAT run, which is most of them."""
    import warnings as _w

    sim = _tophat_sim(requires_grad=False)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        sim.run(oversample=1)


@pytest.mark.parametrize("shape", ["SQUARE", "ROUND", "GAUSS"])
def test_differentiable_shapes_do_not_warn(shape):
    """Only TOPHAT is affected — a false positive would train people to ignore it."""
    import warnings as _w

    sim = _tophat_sim(shape_name=shape)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        sim.run(oversample=1)


@pytest.mark.skipif(not HKL_INTERP.exists(), reason="P1_interp.hkl not present")
def test_tophat_with_interpolated_hkl_keeps_cell_gradients_and_is_silent():
    """TOPHAT + interpolated F_cell is a WORKING configuration, not a dead one.

    This is the case the first version of this module got wrong: it waived the
    cell gradients on shape alone, which would have failed the sweep on correct
    behaviour, and warned that gradients were unavailable where they are in fact
    live. Measured d/d cell_a = -8.84e-05 here against -8.30e-05 for GAUSS on
    the same geometry.
    """
    import warnings as _w

    sim = _tophat_sim(hkl=True, interpolate=True)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        image = sim.run(oversample=1)

    loss = image.sum()
    assert loss.requires_grad, "interpolated F_cell should keep the cell in the graph"
    loss.backward()

    live = [
        n for n in ("a", "b", "c", "alpha", "beta", "gamma")
        if getattr(sim.crystal, f"cell_{n}").grad is not None
        and getattr(sim.crystal, f"cell_{n}").grad.abs().item() > 0
    ]
    assert len(live) == 6, f"expected all six cell gradients live, got {sorted(live)}"


def test_cell_only_tophat_backward_raises():
    """With nothing else differentiable the loss has no grad_fn at all.

    Pins the behaviour the original audit finding described and an earlier
    version of this module denied: it is not a silent `None`, it raises.
    """
    sim = _tophat_sim()
    with pytest.warns(UserWarning):
        image = sim.run(oversample=1)
    loss = image.sum()
    assert not loss.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()
