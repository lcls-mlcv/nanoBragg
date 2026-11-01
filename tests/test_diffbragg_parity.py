"""Derivative parity against simtbx.diffBragg (plan A4; closes issue #8).

diffBragg is cctbx's derivative-enabled nanoBragg, with hand-written analytic
d(pixel)/dθ. For one dxtbx experiment both simulators render the image and its
derivatives with respect to diffBragg's RotX/Y/Z (ids 0-2) and isotropic Ncells
(id 9), and three things are compared:

* the forward image;
* autograd gradients of weighted image sums, d(w·I)/dθ, against Σ w·dI/dθ from
  diffBragg's per-pixel derivatives. Uniform w checks the total; seeded random w
  checks that every pixel's derivative is right, not just their sum;
* torch per-pixel central differences against diffBragg per pixel, on Bragg pixels.

One deliberate difference: with oversampling, diffBragg evaluates the solid angle
and polarization once at the pixel centre (diffBragg_cpu_kernel.cpp, pixel_pos_ave)
while nanoBragg.c evaluates them per subpixel. torch follows C; the size of the
difference is pinned by the last test.

The setup is ``scripts/compare_diffbragg_gradients.py`` (also a CLI). Needs a
cctbx build with simtbx; elsewhere the module is skipped. On exxa:

    source ~/code/cctbx/build/setpaths.sh
    OMP_NUM_THREADS=1 libtbx.python -m pytest tests/test_diffbragg_parity.py
"""
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

# cctbx's OpenMP runtime and torch's deadlock in one process unless one side is single threaded
os.environ.setdefault("OMP_NUM_THREADS", "1")

pytest.importorskip("dxtbx", reason="cctbx/dxtbx not installed")
pytest.importorskip("simtbx.diffBragg", reason="simtbx.diffBragg not built in this cctbx")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_diffbragg_gradients.py"
_spec = importlib.util.spec_from_file_location("compare_diffbragg_gradients", _SCRIPT)
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

PARAM_IDS = (0, 1, 2, 9)
NAMES = {0: "RotX", 1: "RotY", 2: "RotZ", 9: "Ncells"}

CASES = {
    "monoclinic": [],
    "random-orientation": ["--randomrotate", "3"],
    # diffBragg takes omega and polarization at the pixel centre, nanoBragg.c (and torch) per
    # subpixel; these switches make both sides per subpixel/unpolarised (see the last test)
    "oversample2-close": ["--oversample", "2", "--distance", "150", "--oversample-omega", "--nopolar"],
    "ncells5-triclinic": ["--ncells", "5", "--ucell", "61", "73", "84", "87", "95", "101", "--symbol", "P1"],
}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def experiment(request):
    """diffBragg image and per-pixel derivatives for one experiment (computed once per case)."""
    args = cmp.build_parser().parse_args(["--detpixels", "128"] + CASES[request.param])
    sim = cmp.build_sim(args)
    db_img, db_derivs = cmp.diffbragg_forward_and_derivs(sim, list(PARAM_IDS))
    assert db_img.sum() > 0
    return args, sim, db_img, db_derivs


def _torch_params(args, device):
    zero = dict(dtype=torch.float64, device=device)
    return [torch.zeros((), requires_grad=True, **zero) for _ in range(3)] + [
        torch.tensor(float(args.ncells), requires_grad=True, **zero)]


@pytest.mark.parametrize("device", DEVICES)
def test_image_and_autograd_match_diffbragg(experiment, device):
    args, sim, db_img, db_derivs = experiment
    args.device = device
    image = cmp.torch_image_fn(sim, args)
    params = _torch_params(args, device)
    img = image(*params)

    np.testing.assert_allclose(img.detach().cpu().numpy(), db_img, rtol=1e-9, atol=1e-12 * db_img.max())

    rng = np.random.default_rng(0)
    weights = {"uniform": np.ones_like(db_img), "random": rng.uniform(0.0, 1.0, db_img.shape)}
    for wname, w in weights.items():
        grads = torch.autograd.grad((img * torch.as_tensor(w, device=img.device)).sum(), params,
                                    retain_graph=True)
        for pid, g in zip(PARAM_IDS, grads):
            ref = float((w * db_derivs[pid]).sum())
            scale = float(np.abs(w * db_derivs[pid]).sum())
            assert abs(float(g) - ref) <= 1e-7 * scale, (
                f"d(w·I)/d{NAMES[pid]} ({wname} w): torch {float(g)!r} vs diffBragg {ref!r}")


def test_per_pixel_finite_differences_match_diffbragg(experiment):
    args, sim, db_img, db_derivs = experiment
    args.device = "cpu"
    image = cmp.torch_image_fn(sim, args)
    base = [0.0, 0.0, 0.0, float(args.ncells)]
    bragg = db_img > 1e-3 * db_img.max()
    assert bragg.sum() > 20
    with torch.no_grad():
        for i, pid in enumerate(PARAM_IDS):
            h = cmp.FD_STEPS[pid]
            plus, minus = list(base), list(base)
            plus[i] += h
            minus[i] -= h
            fd = ((image(*plus) - image(*minus)) / (2 * h)).numpy()
            ref, test = db_derivs[pid][bragg], fd[bragg]
            r = np.corrcoef(ref, test)[0, 1]
            scale = np.sum(ref * test) / np.sum(ref * ref)
            assert r > 0.99999, f"d/d{NAMES[pid]}: r = {r}"
            assert abs(scale - 1) < 1e-4, f"d/d{NAMES[pid]}: best-fit scale = {scale}"


def test_oversampled_default_differs_only_by_pixel_centre_omega_and_polarization():
    """At oversample 2 diffBragg's pixel-centre omega/polarization shifts pixels by ~1e-5;
    evaluating both per subpixel on both sides removes the difference entirely."""
    def max_rel_diff(extra):
        args = cmp.build_parser().parse_args(["--detpixels", "128", "--oversample", "2", "--distance", "150"] + extra)
        sim = cmp.build_sim(args)
        db_img, _ = cmp.diffbragg_forward_and_derivs(sim, [9])
        img = cmp.torch_image_fn(sim, args)(0.0, 0.0, 0.0, float(args.ncells)).numpy()
        bragg = db_img > 1e-3 * db_img.max()
        return np.max(np.abs(img[bragg] / db_img[bragg] - 1))

    default = max_rel_diff([])
    assert 1e-6 < default < 1e-4, default
    assert max_rel_diff(["--oversample-omega", "--nopolar"]) < 1e-12


# Unit cell (plan C3): ExperimentModel's crystal-system-constrained cell against diffBragg's
# crystal-system managers (simtbx/diffBragg/refiners/crystal_systems), which hold U fixed and
# differentiate A = U·B(cell). diffBragg has managers for these four systems.
UCELL_CASES = {
    "monoclinic": ["--randomrotate", "5"],
    "orthorhombic": ["--ucell", "55", "65", "77", "90", "90", "90", "--symbol", "P212121", "--randomrotate", "5"],
    "tetragonal": ["--ucell", "55", "55", "77", "90", "90", "90", "--symbol", "P43212", "--randomrotate", "5"],
    "hexagonal": ["--ucell", "55", "55", "77", "90", "90", "120", "--symbol", "P6522", "--randomrotate", "5"],
}
# diffBragg variable -> (ExperimentModel parameter, index of the cell value it scales)
UCELL_PARAMS = {
    "a_Ang": ("crystal_params.delta_log_a", 0),
    "b_Ang": ("crystal_params.delta_log_b", 1),
    "c_Ang": ("crystal_params.delta_log_c", 2),
    "beta_rad": ("crystal_params.delta_beta_raw", None),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("system", list(UCELL_CASES))
def test_unit_cell_gradients_match_diffbragg(system, device):
    from nanobrag_torch.compat.cctbx import crystal_system_of
    from nanobrag_torch.utils.unit_cell import FREE_VARIABLES

    args = cmp.build_parser().parse_args(["--detpixels", "128"] + UCELL_CASES[system])
    args.device = device
    sim = cmp.build_sim(args)
    db_img, db_derivs = cmp.diffbragg_ucell_forward_and_derivs(sim)
    xtal = sim.crystal.dxtbx_crystal
    model = cmp.torch_experiment_model(sim, args, crystal_system=crystal_system_of(xtal))
    assert model.crystal_params.crystal_system == system
    assert [n.split("_")[0] for n in db_derivs] == list(FREE_VARIABLES[system])

    img = model()
    np.testing.assert_allclose(img.detach().cpu().numpy(), db_img, rtol=1e-9, atol=1e-12 * db_img.max())

    # ExperimentModel moves log(a) and a tanh-bounded beta offset; at zero offset
    # d(a)/d(delta_log_a) = a and d(beta_rad)/d(delta_beta_raw) = max_angle in radians.
    cell = xtal.get_unit_cell().parameters()
    max_angle_rad = math.radians(model.crystal_params.max_angle_delta_deg)
    params = dict(model.named_parameters())
    names = list(db_derivs)
    leaves = [params[UCELL_PARAMS[n][0]] for n in names]
    factors = [cell[UCELL_PARAMS[n][1]] if UCELL_PARAMS[n][1] is not None else max_angle_rad for n in names]

    rng = np.random.default_rng(1)
    for wname, w in {"uniform": np.ones_like(db_img), "random": rng.uniform(0.0, 1.0, db_img.shape)}.items():
        grads = torch.autograd.grad((img * torch.as_tensor(w, device=img.device)).sum(), leaves, retain_graph=True)
        for name, g, factor in zip(names, grads, factors):
            ref = float((w * db_derivs[name]).sum())
            scale = float(np.abs(w * db_derivs[name]).sum())
            got = float(g) / factor
            assert abs(got - ref) <= 1e-7 * scale, f"d(w·I)/d{name} ({wname} w): torch {got!r} vs diffBragg {ref!r}"


# Mosaic spread (plan C3): ExperimentModel's eta against diffBragg's ETA_ID (19), isotropic and
# anisotropic. Both use diffBragg's deterministic domain ensemble (utils/mosaic.py); anisotropic
# eta lies along the starting crystal's a, b, c axes as in diffBragg's refinement path.
ETA_CASES = {
    "isotropic": ["--eta", "0.3"],
    "anisotropic": ["--eta", "0.2", "0.5", "0.08"],
    "anisotropic-triclinic": ["--eta", "0.4", "0.1", "0.25", "--ucell", "61", "73", "84", "87", "95", "101",
                              "--symbol", "P1", "--randomrotate", "2"],
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(ETA_CASES))
def test_eta_gradients_match_diffbragg(case, device):
    args = cmp.build_parser().parse_args(["--detpixels", "128", "--mosaic-domains", "20"] + ETA_CASES[case])
    args.device = device
    sim = cmp.build_sim(args)
    db_img, db_derivs = cmp.diffbragg_eta_forward_and_derivs(sim)
    model = cmp.torch_experiment_model(sim, args)
    assert len(db_derivs) == len(args.eta) == model.crystal_params.delta_log_eta.numel()

    img = model()
    np.testing.assert_allclose(img.detach().cpu().numpy(), db_img, rtol=1e-9, atol=1e-12 * db_img.max())

    # eta_i = eta0·exp(delta_i), so d/d(eta_i) = d/d(delta_i) / eta_i
    eta = model.crystal_params.eta().detach().cpu().numpy()
    rng = np.random.default_rng(2)
    for wname, w in {"uniform": np.ones_like(db_img), "random": rng.uniform(0.0, 1.0, db_img.shape)}.items():
        (g,) = torch.autograd.grad((img * torch.as_tensor(w, device=img.device)).sum(),
                                   [model.crystal_params.delta_log_eta], retain_graph=True)
        for i, ref_pixels in enumerate(db_derivs):
            ref = float((w * ref_pixels).sum())
            scale = float(np.abs(w * ref_pixels).sum())
            got = float(g[i]) / eta[i]
            assert abs(got - ref) <= 1e-7 * scale, f"d(w·I)/d(eta{i}) ({wname} w): torch {got!r} vs diffBragg {ref!r}"


# detz_shift (plan C3): ExperimentModel's origin shift along lab z against diffBragg's DETZ_ID (10).
# diffBragg's shift is in metres, ExperimentModel's in mm. diffBragg applies the solid angle and the
# polarization once per pixel and leaves both out of its analytic detz derivative (pinned by the
# next test), so the exact comparison turns on per-subpixel omega and turns polarization off.
DETZ_CASES = {
    "flat": ["--oversample-omega", "--nopolar"],
    "tilted": ["--oversample-omega", "--nopolar", "--tilt-deg", "12", "--randomrotate", "4"],
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(DETZ_CASES))
def test_detz_shift_gradient_matches_diffbragg(case, device):
    args = cmp.build_parser().parse_args(["--detpixels", "128"] + DETZ_CASES[case])
    args.device = device
    sim = cmp.build_sim(args)
    db_img, db_deriv = cmp.diffbragg_detz_forward_and_derivs(sim)
    model = cmp.torch_experiment_model(sim, args)
    assert model.detector_params.origin_defined

    img = model()
    np.testing.assert_allclose(img.detach().cpu().numpy(), db_img, rtol=1e-9, atol=1e-12 * db_img.max())

    rng = np.random.default_rng(3)
    for wname, w in {"uniform": np.ones_like(db_img), "random": rng.uniform(0.0, 1.0, db_img.shape)}.items():
        (g,) = torch.autograd.grad((img * torch.as_tensor(w, device=img.device)).sum(),
                                   [model.detector_params.delta_detz_shift_mm], retain_graph=True)
        ref = float((w * db_deriv).sum())
        scale = float(np.abs(w * db_deriv).sum())
        got = 1000.0 * float(g)  # d/d(metres)
        assert abs(got - ref) <= 1e-7 * scale, f"d(w·I)/d(detz_shift) ({wname} w): torch {got!r} vs diffBragg {ref!r}"


@pytest.mark.parametrize("flags, off_by", [
    ([], 0.05),                                                         # omega term: 11% low here
    (["--oversample-omega", "--tilt-deg", "12", "--randomrotate", "4"], 1e-7),  # polarization: 2e-6 high
])
def test_diffbragg_analytic_detz_derivative_misses_omega_and_polarization(flags, off_by):
    """diffBragg's analytic d/d(detz_shift) differs from its own finite difference; torch matches the FD."""
    args = cmp.build_parser().parse_args(["--detpixels", "128"] + flags)
    sim = cmp.build_sim(args)
    _, db_deriv = cmp.diffbragg_detz_forward_and_derivs(sim)
    h, sums = 2e-6, []
    for shift in (h, -h):
        sim.D.shift_origin_z(sim.detector, shift)
        sim.D.raw_pixels_roi *= 0
        sim.D.add_diffBragg_spots()
        sums.append(sim.D.raw_pixels_roi.as_numpy_array().sum())
    sim.D.shift_origin_z(sim.detector, 0.0)
    db_fd = float((sums[0] - sums[1]) / (2 * h))

    model = cmp.torch_experiment_model(sim, args)
    (g,) = torch.autograd.grad(model().sum(), [model.detector_params.delta_detz_shift_mm])
    assert 1000.0 * float(g) == pytest.approx(db_fd, rel=1e-7)
    assert abs(1 - float(db_deriv.sum()) / db_fd) > off_by


# Structure factors (FHKL). diffBragg refines a per-ASU intensity scale s_i (I ∝ s_i·F_i²) with
# Friedel mates as separate ASU entries unless centric; torch refines log F_i, so
# d/d(log F_i) = 2·d/ds_i at s = 1. With Friedel-symmetric data and anomalous=False one torch
# amplitude drives both mates, and its gradient is the sum of theirs.
_FHKL_BASE = ["--detpixels", "128", "--distance", "60", "--pixel", "0.2", "--fhkl-dmin", "6"]
FHKL_CASES = {
    "monoclinic-anomalous": (["--symbol", "P121"], True),
    "tetragonal-anomalous": (["--symbol", "P43212", "--ucell", "70", "70", "50", "90", "90", "90",
                              "--randomrotate", "2"], True),
    "hexagonal-anomalous": (["--symbol", "P6122", "--ucell", "70", "70", "50", "90", "90", "120",
                             "--randomrotate", "5"], True),
    "monoclinic-friedel": (["--symbol", "P121", "--fhkl-friedel", "--randomrotate", "1"], False),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(FHKL_CASES))
def test_fhkl_gradients_match_diffbragg(case, device):
    from nanobrag_torch.compat.cctbx import asu_structure_factors

    flags, anomalous = FHKL_CASES[case]
    args = cmp.build_parser().parse_args(_FHKL_BASE + flags)
    args.device = device
    sim = cmp.build_sim(args)
    space_group = sim.fhkl_asu_array.space_group()
    assert sim.D.hall_symbol == space_group.type().hall_symbol()
    sf = asu_structure_factors(sim.fhkl_asu_array, anomalous=anomalous)
    model = cmp.torch_experiment_model(sim, args, structure_factors=sf)
    img = model()

    rng = np.random.default_rng(5)
    for wname, w in {"uniform": np.ones((128, 128)), "random": rng.uniform(0.5, 1.5, (128, 128))}.items():
        db_img, db_grad, asu_map = cmp.diffbragg_fhkl_forward_and_gradients(sim, w)
        np.testing.assert_allclose(img.detach().cpu().numpy(), db_img, rtol=1e-9, atol=1e-12 * db_img.max())
        ref = np.array([
            2 * sum(db_grad[i] for i in {asu_map[cmp.diffbragg_asu_key(space_group, s * h)]
                                         for s in ((1,) if anomalous else (1, -1))})
            for h in sf.indices])
        (g,) = torch.autograd.grad((img * torch.as_tensor(w, device=img.device)).sum(),
                                   [model.fhkl_params.delta_log_F], retain_graph=True)
        g = g.cpu().numpy()
        assert (ref != 0).sum() > 50, "too few reflections on the detector for a meaningful check"
        np.testing.assert_array_equal(g != 0, ref != 0)
        np.testing.assert_allclose(g, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max(),
                                   err_msg=f"d(w·I)/d(log F) ({wname} w)")


# Background: diffBragg fits the planes once and holds them fixed, so what is compared is
# the image of the planes it fitted, not a gradient.
@pytest.mark.parametrize("robust", [False, True], ids=["fit-tilt", "median"])
def test_background_planes_reproduce_diffbragg_fit(robust):
    from dials.array_family import flex
    from simtbx.diffBragg.utils import get_roi_background_and_selection_flags

    pixels = 128
    rng = np.random.default_rng(11)
    Y, X = np.indices((pixels, pixels))
    data = 40 + 0.3 * X - 0.2 * Y + rng.normal(0, 2, (pixels, pixels))
    spots = rng.uniform(15, pixels - 15, (20, 2))
    refls = flex.reflection_table()
    refls["xyzobs.px.value"] = flex.vec3_double([(x, y, 0.5) for x, y in spots])
    refls["panel"] = flex.size_t(len(spots), 0)
    rois, _, tilt_abc, _, background = get_roi_background_and_selection_flags(
        refls, data[None], shoebox_sz=12, use_robust_estimation=robust, weighted_fit=False,
        allow_overlaps=True, skip_roi_with_negative_bg=False)

    args = cmp.build_parser().parse_args(["--detpixels", str(pixels)])
    model = cmp.torch_experiment_model(cmp.build_sim(args), args, background_rois=rois, tilt_abc=tilt_abc)
    got = model.background_params.image().detach().numpy()
    covered = background[0] != -1  # diffBragg's marker for a pixel in no ROI
    assert covered.sum() > 500
    assert any(a != 0 for a, _, _ in tilt_abc) != robust
    np.testing.assert_allclose(got[covered], background[0][covered], rtol=1e-12)
    np.testing.assert_array_equal(got[~covered], 0)


# Spectrum: diffBragg's lambda_offset and lambda_scale (ids 12, 13), and source weights.
_LAMBDA_BASE = ["--detpixels", "128", "--distance", "100", "--pixel", "0.2"]
LAMBDA_CASES = {
    "monochromatic": [],
    "five-channels": ["--spectrum-channels", "5"],
    "five-channels-tilted": ["--spectrum-channels", "5", "--tilt-deg", "5"],
    "five-channels-Fhkl": ["--spectrum-channels", "5", "--distance", "60", "--fhkl-dmin", "6", "--randomrotate", "1"],
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("point", [(0.0, 1.0), (0.002, 1.001)], ids=["start", "moved"])
@pytest.mark.parametrize("case", list(LAMBDA_CASES))
def test_lambda_gradients_match_diffbragg(case, point, device):
    from nanobrag_torch.compat.cctbx import asu_structure_factors

    args = cmp.build_parser().parse_args(_LAMBDA_BASE + LAMBDA_CASES[case])
    args.device = device
    sim = cmp.build_sim(args)
    sf = {}
    if args.fhkl_dmin:
        sf = dict(structure_factors=asu_structure_factors(sim.fhkl_asu_array, anomalous=True))
    db_img, _ = cmp.diffbragg_lambda_forward_and_derivs(sim)
    sim.D.lambda_coefficients = point
    db_img = cmp.diffbragg_image(sim)
    db_derivs = [d.as_numpy_array().reshape(db_img.shape).copy() for d in sim.D.get_lambda_derivative_pixels()]

    model = cmp.torch_experiment_model(sim, args, **sf)
    offset, scale = point
    with torch.no_grad():
        model.beam_params.lambda_offset.fill_(offset)
        model.beam_params.delta_log_lambda_scale.fill_(math.log(scale))
    img = model()
    np.testing.assert_allclose(img.detach().cpu().numpy(), db_img, rtol=1e-8, atol=1e-11 * db_img.max())

    # diffBragg's analytic d/d(lambda_scale) is only right at the start (see the test
    # below), so difference its image instead
    eps = 1e-6
    sim.D.lambda_coefficients = offset, scale + eps
    above = cmp.diffbragg_image(sim)
    sim.D.lambda_coefficients = offset, scale - eps
    dI_dscale = (above - cmp.diffbragg_image(sim)) / (2 * eps)

    rng = np.random.default_rng(5)
    for wname, w in {"uniform": np.ones_like(db_img), "random": rng.uniform(0.5, 1.5, db_img.shape)}.items():
        grads = torch.autograd.grad((img * torch.as_tensor(w, device=img.device)).sum(),
                                    [model.beam_params.lambda_offset, model.beam_params.delta_log_lambda_scale],
                                    retain_graph=True)
        references = (w * db_derivs[0]).sum(), scale * (w * dI_dscale).sum()  # d/d(log s) = s·d/ds
        for name, got, ref in zip(("offset", "log scale"), grads, references):
            assert ref != 0
            np.testing.assert_allclose(float(got), ref, rtol=1e-7, err_msg=f"d(w·I)/d({name}) ({wname} w)")


@pytest.mark.parametrize("offset, scale", [(0.0, 1.0), (0.002, 1.001), (0.0, 1.02)])
def test_diffbragg_analytic_lambda_scale_derivative_uses_the_moved_wavelength(offset, scale):
    # diffBragg_cpu_kernel.cpp:990 sets dg_dlambda = lambda_ang, which by then is
    # offset + scale·lambda; dlambda/dscale is the source's own lambda. Right at (0, 1)
    # and off by the ratio of the two after that. Pinned so a fix upstream shows up here.
    args = cmp.build_parser().parse_args(_LAMBDA_BASE)
    sim = cmp.build_sim(args)
    wavelength = sim.beam.spectrum[0][0]
    cmp.diffbragg_lambda_forward_and_derivs(sim)
    sim.D.lambda_coefficients = offset, scale
    db_img = cmp.diffbragg_image(sim)
    analytic = sim.D.get_lambda_derivative_pixels()[1].as_numpy_array().sum()
    eps = 1e-6
    sim.D.lambda_coefficients = offset, scale + eps
    above = cmp.diffbragg_image(sim)
    sim.D.lambda_coefficients = offset, scale - eps
    difference = ((above - cmp.diffbragg_image(sim)) / (2 * eps)).sum()
    assert db_img.sum() > 0
    np.testing.assert_allclose(analytic / difference, (offset + scale * wavelength) / wavelength, rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_source_weight_gradients_match_diffbragg_differences(device):
    # diffBragg has no gradient for source_I, so difference its image at fixed total flux
    args = cmp.build_parser().parse_args(_LAMBDA_BASE + ["--spectrum-channels", "5"])
    args.device = device
    sim = cmp.build_sim(args)
    wavelengths, fluxes = (np.array(x) for x in zip(*sim.beam.spectrum))
    model = cmp.torch_experiment_model(sim, args)
    img = model()
    np.testing.assert_allclose(img.detach().cpu().numpy(), cmp.diffbragg_image(sim), rtol=1e-9)

    w = np.random.default_rng(5).uniform(0.5, 1.5, img.shape)
    (g,) = torch.autograd.grad((img * torch.as_tensor(w, device=img.device)).sum(),
                               [model.beam_params.delta_log_source_weights])

    def weighted(delta):
        moved = fluxes * np.exp(delta)
        cmp.set_spectrum(sim, wavelengths, moved * fluxes.sum() / moved.sum())
        return (w * cmp.diffbragg_image(sim)).sum()

    eps = 1e-4
    reference = np.array([(weighted(eps * np.eye(5)[i]) - weighted(-eps * np.eye(5)[i])) / (2 * eps) for i in range(5)])
    assert (reference > 0).any() and (reference < 0).any()
    np.testing.assert_allclose(g.cpu().numpy(), reference, rtol=1e-5)


# ROI likelihood: hopper's target_func, its value and its gradient from diffBragg's Jacobian.
@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_nll_matches_hopper_target_and_gradient(device):
    from nanobrag_torch.utils.losses import gaussian_nll

    args = cmp.build_parser().parse_args(["--detpixels", "128"])
    args.device = device
    sim = cmp.build_sim(args)
    db_img, db_derivs = cmp.diffbragg_forward_and_derivs(sim, PARAM_IDS)
    rois = [(10, 50, 20, 60), (60, 118, 70, 120)]
    abc = [(0.02, -0.01, 0.5), (0.0, 0.03, 0.2)]
    model = cmp.torch_experiment_model(sim, args, background_rois=rois, tilt_abc=abc)
    rng = np.random.default_rng(7)
    data_image = db_img + rng.normal(0, 0.4, db_img.shape) + 0.3
    trusted_image = rng.uniform(size=db_img.shape) > 0.1
    sigma_rdout = 0.5

    # hopper_utils.target_func, on 1-D pixels as Modeler keeps them
    Y, X = np.indices(db_img.shape)
    background, selected = np.zeros(db_img.shape), np.zeros(db_img.shape, bool)
    for (x1, x2, y1, y2), (a, b, c) in zip(rois, abc):
        background[y1:y2, x1:x2] = a * (X[y1:y2, x1:x2] - x1) + b * (Y[y1:y2, x1:x2] - y1) + c
        selected[y1:y2, x1:x2] = True
    trusted = trusted_image[selected]
    model_pix = (db_img + background)[selected]
    resid = data_image[selected] - model_pix
    V = model_pix + sigma_rdout**2
    target = (.5 * (np.log(2 * np.pi * V) + resid**2 / V))[trusted].sum()
    common_grad_term = (0.5 / V * (1 - 2 * resid - resid**2 / V))[trusted]

    pred = model(roi_only=True)
    loss = gaussian_nll(pred, model.roi_values(data_image), sigma_rdout, trusted=model.roi_values(trusted_image))
    np.testing.assert_allclose(float(loss), target, rtol=1e-10)

    # hopper's g[p] = Σ_trusted J_p·common_grad_term, for J_p from diffBragg's derivative
    # pixels (Ncells), its image (G, d/d(log G) of the Bragg term) and the plane (offset c)
    g_n, g_scale, g_abc = torch.autograd.grad(loss, [
        model.crystal_params.delta_log_N_cells, model.beam_params.delta_log_spot_scale,
        model.background_params.delta_tilt_abc])
    first_roi = np.zeros(db_img.shape, bool)
    x1, x2, y1, y2 = rois[0]
    first_roi[y1:y2, x1:x2] = True
    references = {
        "log Ncells": (float(g_n.sum()), args.ncells * (db_derivs[9][selected][trusted] * common_grad_term).sum()),
        "log G": (float(g_scale), (db_img[selected][trusted] * common_grad_term).sum()),
        "plane c": (float(g_abc[0, 2]), (first_roi[selected][trusted] * common_grad_term).sum()),
    }
    for name, (got, ref) in references.items():
        assert ref != 0
        np.testing.assert_allclose(got, ref, rtol=1e-8, err_msg=name)


# Panels: hopper simulates (panel, fast, slow) pixels of a multi-panel detector in one call.
def _panelled(distance=150.0, device="cpu"):
    from nanobrag_torch.compat.cctbx import experiment_model_from_dxtbx

    args = cmp.build_parser().parse_args(["--detpixels", "128", "--distance", str(distance), "--pixel", "0.1",
                                          "--panels", "3", "--randomrotate", "3"])
    args.device = device
    sim = cmp.build_sim(args)
    rng = np.random.default_rng(0)
    rois = [(int(x), int(x) + 15, int(y), int(y) + 15) for x, y in rng.integers(0, 110, (45, 2))]
    model = experiment_model_from_dxtbx(
        sim.detector, sim.beam.nanoBragg_constructor_beam, sim.crystal.dxtbx_crystal,
        Ncells_abc=args.ncells, default_F=args.default_F, fluence=float(sim.D.fluence),
        spot_scale=float(sim.D.spot_scale), rois=rois, tilt_abc=[(0.0, 0.0, 0.0)] * 45, panels=[i % 3 for i in range(45)],
        device=torch.device(device),
    )
    model.set_structure_factors(torch.zeros((1, 1, 1), dtype=torch.float64),
                                dict(h_min=0, h_max=0, k_min=0, k_max=0, l_min=0, l_max=0))
    return args, sim, model


def _shoeboxes(model):
    background = model.background_params
    slow, fast = background.pixels.cpu().numpy().T
    return background.panel.cpu().numpy(), fast, slow


@pytest.mark.parametrize("device", DEVICES)
def test_panel_shoeboxes_match_diffbragg(device):
    args, sim, model = _panelled(device=device)
    panel, fast, slow = _shoeboxes(model)
    db, derivs = cmp.diffbragg_shoeboxes(sim, [9], panel, fast, slow)
    image = model(roi_only=True)
    got = image.detach().cpu().numpy()
    for i in range(3):
        assert db[panel == i].sum() > 0
        np.testing.assert_allclose(got[panel == i], db[panel == i], rtol=1e-8, atol=1e-11 * db[panel == i].max())
    w = np.random.default_rng(1).uniform(0.5, 1.5, db.shape)
    (g_n,) = torch.autograd.grad((image * torch.as_tensor(w, device=image.device)).sum(),
                                 [model.crystal_params.delta_log_N_cells])
    np.testing.assert_allclose(float(g_n.sum()), args.ncells * (w * derivs[9]).sum(), rtol=1e-8)


def test_shared_detz_shift_moves_every_panel_like_diffbragg():
    # Nothing to difference in hopper's id 10 across panels, so shift the dxtbx detector itself
    shift_mm = 0.5
    _, sim, model = _panelled()
    panel, fast, slow = _shoeboxes(model)
    _, moved_sim, _ = _panelled(distance=150.0 - shift_mm)
    db, _ = cmp.diffbragg_shoeboxes(moved_sim, [9], panel, fast, slow)
    unmoved, _ = cmp.diffbragg_shoeboxes(sim, [9], panel, fast, slow)
    assert not np.allclose(db, unmoved)
    with torch.no_grad():
        model.detector_params.delta_detz_shift_mm.fill_(shift_mm)
        got = model(roi_only=True).numpy()
    np.testing.assert_allclose(got, db, rtol=1e-8, atol=1e-11 * db.max())
