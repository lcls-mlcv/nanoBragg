#!/usr/bin/env python
"""
Compare nanobrag_torch autograd derivatives against simtbx.diffBragg's analytic
derivatives for the same dxtbx models.

diffBragg is cctbx's derivative-enabled nanoBragg (analytic d(pixel)/dθ in
C++/CUDA/Kokkos) and is the natural reference for a differentiable port. This
script builds one SimData, renders it with diffBragg and with the torch port via
``nanobrag_torch.compat.cctbx.simulator_from_sim_data``, then compares:

  * the forward images (Pearson r, sum ratio);
  * d(image)/d(RotX, RotY, RotZ)  — diffBragg ids 0, 1, 2 (radians, lab axes,
    convention A' = Rx·Ry·Rz·A);
  * d(image)/d(Ncells)            — diffBragg id 9 (isotropic Na=Nb=Nc).

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


def build_sim(args):
    from simtbx.nanoBragg import shapetype
    from simtbx.nanoBragg.sim_data import SimData
    from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
    from scitbx.matrix import sqr
    from cctbx import uctbx
    from dxtbx.model import Crystal

    ucell = tuple(args.ucell)
    a_real, b_real, c_real = sqr(uctbx.unit_cell(ucell).orthogonalization_matrix()).transpose().as_list_of_lists()
    C = Crystal(a_real, b_real, c_real, args.symbol)
    if args.randomrotate is not None:
        from scitbx.matrix import col
        rng = np.random.RandomState(args.randomrotate)
        axis = col(tuple(rng.normal(size=3))).normalize()
        C.set_U(axis.axis_and_angle_as_r3_rotation_matrix(float(rng.uniform(0, 2 * math.pi))))
    nb = NBcrystal(init_defaults=True)
    nb.dxtbx_crystal = C
    nb.n_mos_domains = 1
    nb.mos_spread_deg = 0
    nb.thick_mm = 0.01
    nb.Ncells_abc = (args.ncells,) * 3
    nb.xtal_shape = {"square": shapetype.Square, "round": shapetype.Round, "gauss": shapetype.Gauss}[args.shape]
    SIM = SimData(use_default_crystal=True)
    SIM.detector = SimData.simple_detector(args.distance, args.pixel, (args.detpixels, args.detpixels))
    SIM.crystal = nb
    SIM.instantiate_diffBragg(oversample=args.oversample, verbose=0, interpolate=0, default_F=args.default_F,
                              auto_set_spotscale=False)
    return SIM


def diffbragg_forward_and_derivs(SIM, param_ids):
    D = SIM.D
    for pid in param_ids:
        D.refine(pid)
    D.initialize_managers()
    for pid in (0, 1, 2):
        D.set_value(pid, 0.0)
    D.raw_pixels_roi *= 0
    D.add_diffBragg_spots()
    img = D.raw_pixels_roi.as_numpy_array().copy()
    derivs = {pid: D.get_derivative_pixels(pid).as_numpy_array().copy() for pid in param_ids}
    return img, derivs


def torch_forward_and_derivs(SIM, args):
    import torch
    from nanobrag_torch.compat.cctbx import (
        crystal_config_from_A, detector_config_from_dxtbx_panel, beam_config_from_dxtbx, rotate_A,
    )
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
    beam_cfg = beam_config_from_dxtbx(beam, fluence=float(SIM.D.fluence), spot_scale=float(SIM.D.spot_scale))
    detector = TDetector(det_cfg, device=device, dtype=dtype)

    def image(rotx, roty, rotz, ncells):
        A = rotate_A(A0, math.degrees(rotx) if not torch.is_tensor(rotx) else torch.rad2deg(rotx),
                     math.degrees(roty) if not torch.is_tensor(roty) else torch.rad2deg(roty),
                     math.degrees(rotz) if not torch.is_tensor(rotz) else torch.rad2deg(rotz))
        n = ncells if torch.is_tensor(ncells) else float(ncells)
        cfg = crystal_config_from_A(cell, A, Ncells_abc=(1, 1, 1), shape=args.shape, default_F=args.default_F)
        cfg.N_cells = (n, n, n)
        crystal = TCrystal(cfg, beam_config=beam_cfg, device=device, dtype=dtype)
        sim = Simulator(crystal, detector, crystal_config=cfg, beam_config=beam_cfg, device=device, dtype=dtype)
        return sim.run()

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
    steps = {0: 2e-4, 1: 2e-4, 2: 2e-4, 9: 1e-3}
    with torch.no_grad():
        base = [float(v) for v in params.values()]
        for i, pid in enumerate(params):
            h = steps[pid]
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ucell", type=float, nargs=6, default=(70, 60, 50, 90.0, 110, 90.0))
    p.add_argument("--symbol", default="C121")
    p.add_argument("--ncells", type=int, default=7)
    p.add_argument("--shape", choices=["square", "round", "gauss"], default="square")
    p.add_argument("--detpixels", type=int, default=256)
    p.add_argument("--pixel", type=float, default=0.1)
    p.add_argument("--distance", type=float, default=220.0)
    p.add_argument("--oversample", type=int, default=1)
    p.add_argument("--default_F", type=float, default=1e3)
    p.add_argument("--randomrotate", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

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
    report("image", db_img, t_img, np.ones_like(mask))
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
