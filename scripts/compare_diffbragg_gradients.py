#!/usr/bin/env python
"""
Compare nanobrag_torch autograd derivatives against simtbx.diffBragg's analytic
derivatives for the same dxtbx models.

diffBragg is cctbx's derivative-enabled nanoBragg (analytic d(pixel)/dθ in
C++/CUDA/Kokkos) and is the natural reference for a differentiable port. This
script builds one SimData, renders it with diffBragg and with the torch port via
``nanobrag_torch.compat.cctbx.simulator_from_sim_data``, then compares:

  * the forward images (Pearson r, sum ratio);
  * d(image)/d(RotX, RotY, RotZ)  — diffBragg ids 0, 1, 2 (radians, lab axes).
    diffBragg's rotX/Y/Z_manager matrices are transposes of right-handed
    rotations applied to the real-space basis, so diffBragg RotX = θ turns the
    crystal by −θ; torch is evaluated at rotate_A(A, −θ) so the signs agree;
  * d(image)/d(Ncells)            — diffBragg id 9 (isotropic Na=Nb=Nc).

diffBragg's CPU kernel always uses a Gaussian lattice with the radius measured
in hkl space, det(NABC)·exp(−|NABC·ΔH|²/0.63), whatever xtal_shape says. The
default --shape is therefore gauss, and the torch side uses SpotMetric.HKL (the
compat layer's default) so the two lattices agree; --spot-metric reciprocal
switches torch back to bl831's rad_star and the comparison then fails by design.

For each parameter the torch gradient is obtained with autograd on the summed
image, and per-pixel with a finite difference of the torch model as a sanity
check. diffBragg per-pixel derivatives come from ``get_derivative_pixels``.

Requires a cctbx environment (dxtbx, simtbx). Run:

    libtbx.python scripts/compare_diffbragg_gradients.py --detpixels 256 --ncells 7
"""
from __future__ import division, print_function

import argparse
import math
import os
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")
# cctbx's OpenMP runtime and torch's deadlock in one process unless one side is single threaded
os.environ.setdefault("OMP_NUM_THREADS", "1")


def build_sim(args):
    from simtbx.nanoBragg.sim_data import SimData
    from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
    from scitbx.matrix import col, sqr
    from cctbx import uctbx
    from dxtbx.model import Crystal

    ucell = tuple(args.ucell)
    a_real, b_real, c_real = sqr(uctbx.unit_cell(ucell).orthogonalization_matrix()).transpose().as_list_of_lists()
    C = Crystal(a_real, b_real, c_real, args.symbol)
    if args.randomrotate is not None:
        rng = np.random.RandomState(args.randomrotate)
        axis = col(tuple(rng.normal(size=3))).normalize()
        C.set_U(axis.axis_and_angle_as_r3_rotation_matrix(float(rng.uniform(0, 2 * math.pi))))
    nb = NBcrystal(init_defaults=True)
    nb.dxtbx_crystal = C
    nb.n_mos_domains = args.mosaic_domains if args.eta else 1
    nb.mos_spread_deg = args.eta[0] if args.eta else 0
    if args.eta and len(args.eta) == 3:
        nb.anisotropic_mos_spread_deg = tuple(args.eta)
    nb.thick_mm = 0.01
    nb.Ncells_abc = (args.ncells,) * 3
    nb.miller_array = None  # constant default_F on both sides
    # NBcrystal's setter takes lower-case names; a shapetype enum silently becomes Tophat
    nb.xtal_shape = args.shape
    SIM = SimData(use_default_crystal=True)
    SIM.detector = SimData.simple_detector(args.distance, args.pixel, (args.detpixels, args.detpixels))
    if args.tilt_deg:
        # tilt the panel about (1, 1, 0) through its centre, so both axes get z components
        panel = SIM.detector[0]
        R = col((1, 1, 0)).normalize().axis_and_angle_as_r3_rotation_matrix(args.tilt_deg, deg=True)
        fast, slow, origin = col(panel.get_fast_axis()), col(panel.get_slow_axis()), col(panel.get_origin())
        centre = origin + fast * (args.pixel * args.detpixels / 2) + slow * (args.pixel * args.detpixels / 2)
        fast, slow = R * fast, R * slow
        panel.set_frame(fast, slow, centre - fast * (args.pixel * args.detpixels / 2)
                        - slow * (args.pixel * args.detpixels / 2))
    SIM.crystal = nb
    SIM.instantiate_diffBragg(oversample=args.oversample, verbose=0, interpolate=0, default_F=args.default_F,
                              auto_set_spotscale=False)
    SIM.D.oversample_omega = args.oversample_omega
    SIM.D.nopolar = args.nopolar
    return SIM


# central finite-difference steps per diffBragg parameter id
FD_STEPS = {0: 2e-4, 1: 2e-4, 2: 2e-4, 9: 1e-3}


def diffbragg_forward_and_derivs(SIM, param_ids):
    D = SIM.D
    for pid in param_ids:
        D.refine(pid)
    D.initialize_managers()
    for pid in (0, 1, 2):
        D.set_value(pid, 0.0)
    D.raw_pixels_roi *= 0
    D.add_diffBragg_spots()
    fast, slow = SIM.detector[0].get_image_size()
    # raw_pixels_roi and the derivative pixels are flat (slow-major) flex arrays
    img = D.raw_pixels_roi.as_numpy_array().reshape(slow, fast).copy()
    derivs = {pid: D.get_derivative_pixels(pid).as_numpy_array().reshape(slow, fast).copy() for pid in param_ids}
    return img, derivs


def diffbragg_ucell_forward_and_derivs(SIM):
    """diffBragg image and d(image)/d(unit-cell variable) from its crystal-system manager.

    Returns (image, {variable name: per-pixel derivative}); names are diffBragg's
    (a_Ang, c_Ang, beta_rad, ...). Raises NotImplementedError for systems diffBragg
    has no manager for (triclinic, rhombohedral, cubic).
    """
    from simtbx.diffBragg import utils as db_utils
    from simtbx.diffBragg.hopper_utils import UCELL_ID_OFFSET

    manager = db_utils.manager_from_crystal(SIM.crystal.dxtbx_crystal)
    D = SIM.D
    ids = [UCELL_ID_OFFSET + i for i in range(len(manager.variables))]
    for pid in ids:
        D.refine(pid)
    for pid, dB in zip(ids, manager.derivative_matrices):
        D.set_ucell_derivative_matrix(pid, dB)
    D.initialize_managers()
    D.raw_pixels_roi *= 0
    D.add_diffBragg_spots()
    fast, slow = SIM.detector[0].get_image_size()
    img = D.raw_pixels_roi.as_numpy_array().reshape(slow, fast).copy()
    derivs = {name: D.get_derivative_pixels(pid).as_numpy_array().reshape(slow, fast).copy()
              for name, pid in zip(manager.variable_names, ids)}
    return img, derivs


def diffbragg_eta_forward_and_derivs(SIM):
    """diffBragg image and d(image)/d(eta) (ETA_ID 19): one derivative, or three if anisotropic."""
    from simtbx.diffBragg.hopper_utils import ETA_ID

    D = SIM.D
    D.refine(ETA_ID)
    D.initialize_managers()
    D.raw_pixels_roi *= 0
    D.add_diffBragg_spots()
    fast, slow = SIM.detector[0].get_image_size()
    img = D.raw_pixels_roi.as_numpy_array().reshape(slow, fast).copy()
    pixels = D.get_aniso_eta_deriv_pixels() if D.has_anisotropic_mosaic_spread else [D.get_derivative_pixels(ETA_ID)]
    return img, [p.as_numpy_array().reshape(slow, fast).copy() for p in pixels]


def diffbragg_detz_forward_and_derivs(SIM):
    """diffBragg image and d(image)/d(detz_shift) (DETZ_ID 10; the shift is in metres)."""
    from simtbx.diffBragg.hopper_utils import DETZ_ID

    D = SIM.D
    D.refine(DETZ_ID)
    D.initialize_managers()
    D.raw_pixels_roi *= 0
    D.add_diffBragg_spots()
    fast, slow = SIM.detector[0].get_image_size()
    img = D.raw_pixels_roi.as_numpy_array().reshape(slow, fast).copy()
    return img, D.get_derivative_pixels(DETZ_ID).as_numpy_array().reshape(slow, fast).copy()


def torch_experiment_model(SIM, args, **kwargs):
    """ExperimentModel for SIM's experiment, oriented by its A matrix (float64).

    With --eta the model starts at diffBragg's eta (anisotropic for three values).
    """
    import torch
    from nanobrag_torch.compat.cctbx import (
        crystal_config_from_A, detector_config_from_dxtbx_panel, beam_config_from_dxtbx,
    )
    from nanobrag_torch.config import SpotMetric
    from nanobrag_torch.models.experiment import ExperimentModel

    xtal = SIM.crystal.dxtbx_crystal
    beam = SIM.beam.nanoBragg_constructor_beam
    det_cfg = detector_config_from_dxtbx_panel(SIM.detector[0], beam.get_s0(), oversample=max(args.oversample, 1))
    det_cfg.oversample_omega = args.oversample_omega
    beam_cfg = beam_config_from_dxtbx(beam, fluence=float(SIM.D.fluence), spot_scale=float(SIM.D.spot_scale))
    beam_cfg.nopolar = beam_cfg.nopolar or args.nopolar
    eta = list(args.eta or [0.0])
    cfg = crystal_config_from_A(xtal.get_unit_cell().parameters(), xtal.get_A(), Ncells_abc=args.ncells,
                                shape=args.shape, default_F=args.default_F,
                                spot_metric=SpotMetric(args.spot_metric),
                                mosaic_spread_deg=eta[0], mosaic_domains=args.mosaic_domains if args.eta else 1)
    model = ExperimentModel(cfg, det_cfg, beam_cfg, device=torch.device(args.device), dtype=torch.float64,
                            anisotropic_mosaicity=len(eta) == 3, **kwargs)
    if len(eta) == 3:
        with torch.no_grad():
            model.crystal_params.delta_log_eta.copy_(torch.log(torch.tensor(eta, dtype=torch.float64) / eta[0]))
    # cctbx: F(0,0,0) = F000 (0), every other reflection default_F
    model.set_structure_factors(torch.zeros((1, 1, 1), dtype=torch.float64),
                                dict(h_min=0, h_max=0, k_min=0, k_max=0, l_min=0, l_max=0))
    return model


def torch_image_fn(SIM, args):
    """Return image(rotx, roty, rotz, ncells) rendering SIM with the torch port.

    Arguments may be floats or 0-d tensors; diffBragg's RotXYZ (radians) and
    isotropic Ncells (id 9) are the parameters.
    """
    import torch
    from nanobrag_torch.compat.cctbx import (
        crystal_config_from_A, detector_config_from_dxtbx_panel, beam_config_from_dxtbx, rotate_A,
        set_structure_factors,
    )
    from nanobrag_torch.config import SpotMetric
    from nanobrag_torch.models.crystal import Crystal as TCrystal
    from nanobrag_torch.models.detector import Detector as TDetector
    from nanobrag_torch.simulator import Simulator

    dtype = torch.float64
    device = torch.device(args.device)
    xtal = SIM.crystal.dxtbx_crystal
    A0 = np.asarray(xtal.get_A()).reshape(3, 3)
    cell = xtal.get_unit_cell().parameters()
    beam = SIM.beam.nanoBragg_constructor_beam
    det_cfg = detector_config_from_dxtbx_panel(SIM.detector[0], beam.get_s0(), oversample=max(args.oversample, 1))
    det_cfg.oversample_omega = args.oversample_omega
    beam_cfg = beam_config_from_dxtbx(beam, fluence=float(SIM.D.fluence), spot_scale=float(SIM.D.spot_scale))
    beam_cfg.nopolar = beam_cfg.nopolar or args.nopolar
    detector = TDetector(det_cfg, device=device, dtype=dtype)

    def neg_deg(theta):
        return -torch.rad2deg(theta) if torch.is_tensor(theta) else -math.degrees(theta)

    def image(rotx, roty, rotz, ncells):
        # diffBragg's RotXYZ = θ is a −θ rotation in rotate_A's right-handed convention
        A = rotate_A(A0, neg_deg(rotx), neg_deg(roty), neg_deg(rotz))
        n = ncells if torch.is_tensor(ncells) else float(ncells)
        cfg = crystal_config_from_A(cell, A, Ncells_abc=(1, 1, 1), shape=args.shape, default_F=args.default_F,
                                    spot_metric=SpotMetric(args.spot_metric))
        cfg.N_cells = (n, n, n)
        crystal = TCrystal(cfg, beam_config=beam_cfg, device=device, dtype=dtype)
        # cctbx: F(0,0,0) = F000 (0), every other reflection default_F
        set_structure_factors(crystal, np.zeros((1, 3), dtype=np.int64), [0.0], default_F=args.default_F)
        sim = Simulator(crystal, detector, crystal_config=cfg, beam_config=beam_cfg, device=device, dtype=dtype)
        return sim.run()

    return image


def torch_forward_and_derivs(SIM, args):
    import torch

    dtype = torch.float64
    image = torch_image_fn(SIM, args)
    params = {
        0: torch.zeros((), dtype=dtype, requires_grad=True),
        1: torch.zeros((), dtype=dtype, requires_grad=True),
        2: torch.zeros((), dtype=dtype, requires_grad=True),
        9: torch.tensor(float(args.ncells), dtype=dtype, requires_grad=True),
    }
    img = image(params[0], params[1], params[2], params[9])
    # per-pixel derivatives via one backward per parameter is expensive; use finite differences per pixel
    # and autograd for the summed image (checks the chain end to end).
    grads_sum = torch.autograd.grad(img.sum(), list(params.values()), allow_unused=True)
    derivs = {}
    with torch.no_grad():
        base = [float(v) for v in params.values()]
        for i, pid in enumerate(params):
            h = FD_STEPS[pid]
            plus = list(base); plus[i] += h
            minus = list(base); minus[i] -= h
            derivs[pid] = ((image(*plus) - image(*minus)) / (2 * h)).cpu().numpy()
    return img.detach().cpu().numpy(), derivs, {pid: (None if g is None else float(g)) for pid, g in zip(params, grads_sum)}


def report(name, ref, test, mask):
    r = np.corrcoef(ref[mask].ravel(), test[mask].ravel())[0, 1] if mask.sum() > 2 else float("nan")
    scale = np.sum(ref[mask] * test[mask]) / max(np.sum(ref[mask] ** 2), 1e-300)
    mae = np.abs(ref[mask] - test[mask]).mean()
    print(f"{name:26s} r={r:9.6f}  best-fit scale (torch/diffBragg)={scale:9.5f}  MAE={mae:11.4g}  "
          f"sum_ref={ref.sum():11.5g} sum_torch={test.sum():11.5g}")
    return r, scale


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ucell", type=float, nargs=6, default=(70, 60, 50, 90.0, 110, 90.0))
    p.add_argument("--symbol", default="P121")
    p.add_argument("--ncells", type=int, default=7)
    p.add_argument("--shape", choices=["square", "round", "gauss"], default="gauss")
    p.add_argument("--detpixels", type=int, default=256)
    p.add_argument("--pixel", type=float, default=0.1)
    p.add_argument("--distance", type=float, default=220.0)
    p.add_argument("--oversample", type=int, default=1)
    p.add_argument("--default_F", type=float, default=1e3)
    p.add_argument("--randomrotate", type=int, default=None)
    p.add_argument("--spot-metric", dest="spot_metric", choices=["hkl", "reciprocal"], default="hkl",
                   help="GAUSS/TOPHAT radius metric for the torch side; diffBragg always uses hkl")
    p.add_argument("--oversample-omega", dest="oversample_omega", action="store_true",
                   help="solid angle per subpixel on both sides (diffBragg otherwise takes it at the pixel centre)")
    p.add_argument("--nopolar", action="store_true", help="polarization factor 1 on both sides")
    p.add_argument("--tilt-deg", dest="tilt_deg", type=float, default=0.0,
                   help="tilt the panel about (1, 1, 0) through its centre")
    p.add_argument("--eta", type=float, nargs="+", default=None,
                   help="mosaic spread in degrees: one value, or three for anisotropic (crystal a, b, c)")
    p.add_argument("--mosaic-domains", dest="mosaic_domains", type=int, default=20,
                   help="number of mosaic domains with --eta (even)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    try:
        SIM = build_sim(args)
    except ImportError as e:
        print("cctbx / dxtbx / simtbx not importable:", e, file=sys.stderr)
        sys.exit(2)

    ids = [0, 1, 2, 9]
    db_img, db_derivs = diffbragg_forward_and_derivs(SIM, ids)
    t_img, t_derivs, t_grad_sum = torch_forward_and_derivs(SIM, args)

    mask = db_img > 1e-3 * db_img.max()
    print("== forward image")
    report("image", db_img, t_img, np.ones_like(mask, dtype=bool))
    names = {0: "d/dRotX (rad)", 1: "d/dRotY (rad)", 2: "d/dRotZ (rad)", 9: "d/dNcells"}
    print("== per-pixel derivatives on Bragg pixels (torch central FD vs diffBragg analytic)")
    ok = True
    for pid in ids:
        r, scale = report(names[pid], db_derivs[pid], t_derivs[pid], mask)
        print(f"{'':26s} torch autograd d(sum)/dθ = {t_grad_sum[pid]!r}; diffBragg sum = {db_derivs[pid].sum():.6g}")
        ok &= (r > 0.99) and (0.9 < scale < 1.1)
    if args.plot:
        import pylab as plt
        fig, axes = plt.subplots(len(ids), 2, figsize=(8, 3 * len(ids)))
        for row, pid in enumerate(ids):
            axes[row, 0].imshow(db_derivs[pid]); axes[row, 0].set_title(f"diffBragg {names[pid]}")
            axes[row, 1].imshow(t_derivs[pid]); axes[row, 1].set_title(f"torch {names[pid]}")
        plt.tight_layout(); plt.show()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
