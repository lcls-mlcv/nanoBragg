#!/usr/bin/env python
"""Refine a perturbed synthetic shot with diffBragg's hopper and with ExperimentModel (plan C6).

Stage ``real`` does the same for one frame of a real dataset, with hopper's solution standing
in for the truth. Stage ``hopper`` simulates a noisy still as simtbx's tst_diffBragg_hopper_refine.py does,
perturbs the crystal, the scale and Ncells, and runs hopper_utils.refine. It saves what
hopper used: the pixels, the trusted mask and the background. Stage ``torch`` refines the
same start on those pixels, so a difference in the result comes from the refinement and not
from the data selection. They are separate processes because cctbx's OpenMP and torch's
deadlock in one process unless cctbx is single threaded.

    source ~/code/cctbx/build/setpaths.sh
    OMP_NUM_THREADS=8 libtbx.python scripts/refinement_demo.py hopper --outdir DIR
    OMP_NUM_THREADS=1 PYTHONPATH=src:$PYTHONPATH libtbx.python scripts/refinement_demo.py torch --outdir DIR --device cuda
"""
import argparse
import json
import os
import sys
import time

import numpy as np

UCELL_TRUE = (55, 65, 75, 90, 95, 90)
UCELL_START = (55.1, 65.2, 74.9, 90, 94.9, 90)
NCELLS_TRUE = (12, 12, 11)
NCELLS_START = (20, 20, 20)
G_FACTOR_START = 10.0
SYMBOL = "P121"


def stage_hopper(args):
    from cctbx import uctbx
    from dxtbx.model import Experiment
    from dxtbx.model.crystal import Crystal
    from libtbx.phil import parse
    from scipy.spatial.transform import Rotation
    from scitbx.matrix import col, rec, sqr
    from simtbx.diffBragg import hopper_utils, utils
    from simtbx.diffBragg.device import DeviceWrapper
    from simtbx.diffBragg.phil import hopper_phil, philz
    from simtbx.nanoBragg import make_imageset
    from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
    from simtbx.nanoBragg.sim_data import SimData

    os.chdir(args.outdir)
    with DeviceWrapper(0):
        rotation = Rotation.random(num=1, random_state=args.seed)[0]
        rot_ang, rot_axis = rec(rotation.as_quat(), n=(4, 1)).unit_quaternion_as_axis_and_angle()
        np.random.seed(args.seed)
        perturb_axis = np.random.random(3)
        perturb_axis /= np.linalg.norm(perturb_axis)

        def crystal(ucell):
            a, b, c = sqr(uctbx.unit_cell(ucell).orthogonalization_matrix()).transpose().as_list_of_lists()
            xtal = Crystal(a, b, c, SYMBOL)
            xtal.rotate_around_origin(rot_axis, rot_ang)
            return xtal

        truth = crystal(UCELL_TRUE)
        start = crystal(UCELL_START)
        start.rotate_around_origin(col(perturb_axis), args.perturb_deg)

        nbcryst = NBcrystal()
        nbcryst.dxtbx_crystal = truth
        nbcryst.thick_mm = 0.1
        nbcryst.isotropic_ncells = False
        nbcryst.Ncells_abc = NCELLS_TRUE
        SIM = SimData(use_default_crystal=True)
        SIM.detector = SimData.simple_detector(150, 0.1, (513, 512))
        SIM.crystal = nbcryst
        SIM.instantiate_diffBragg(oversample=0, auto_set_spotscale=True)
        SIM.D.default_F = 0
        SIM.D.F000 = 0
        SIM.D.progress_meter = False
        SIM.water_path_mm = 0.005
        SIM.air_path_mm = 0.1
        SIM.add_air = True
        SIM.add_Water = True
        SIM.include_noise = True
        SIM.D.add_diffBragg_spots()
        spots = SIM.D.raw_pixels.as_numpy_array().copy()
        SIM._add_background()
        SIM.D.readout_noise_adu = 0
        SIM._add_noise()
        img = SIM.D.raw_pixels.as_numpy_array().copy()
        SIM.D.raw_pixels *= 0

        P = parse(hopper_phil + philz).extract()
        G_true = float(SIM.D.spot_scale)
        P.init.G = G_true * G_FACTOR_START
        P.init.Nabc = NCELLS_START
        P.init.detz_shift = 0
        P.types.G = P.types.Nabc = "ranged"
        E = Experiment()
        E.crystal = start
        E.detector = SIM.detector
        E.beam = SIM.D.beam
        E.imageset = make_imageset([img], E.beam, E.detector)
        refls = utils.refls_from_sims([spots], E.detector, E.beam, thresh=18)
        utils.refls_to_q(refls, E.detector, E.beam, update_table=True)
        utils.refls_to_hkl(refls, E.detector, E.beam, E.crystal, update_table=True)
        print("%d reflections" % len(refls))

        P.roi.shoebox_size = 20
        P.relative_tilt = False
        P.roi.fit_tilt = False
        P.roi.pad_shoebox_for_background_estimation = 10
        P.roi.reject_edge_reflections = False
        P.refiner.sigma_r = SIM.D.readout_noise_adu
        P.refiner.adu_per_photon = SIM.D.quantum_gain
        P.simulator.init_scale = 1
        P.simulator.beam.size_mm = SIM.beam.size_mm
        P.simulator.total_flux = SIM.D.flux
        P.use_restraints = False
        SIM.crystal.miller_array.as_mtz_dataset(column_root_label="F").mtz_object().write("demo.mtz")
        P.simulator.structure_factors.mtz_name = "demo.mtz"
        P.simulator.structure_factors.mtz_column = "F(+),F(-)"
        P.niter = 0
        P.sigmas.RotXYZ = [1, 1, 1]
        P.niter_per_J = 1
        P.method = "L-BFGS-B"
        P.ftol = 1e-10

        clock = time.perf_counter()
        Eopt, _, Mod, SIM_hopper, x = hopper_utils.refine(E, refls, P, return_modeler=True)
        seconds = time.perf_counter() - clock

        G, rotX, rotY, rotZ, Na, Nb, Nc, *_, a, b, c, al, be, ga, detz = hopper_utils.get_param_from_x(x, Mod)
        miss_opt, miss_start = utils.compare_with_ground_truth(
            *truth.get_real_space_vectors(), dxcryst_models=[Eopt.crystal, start], symbol=SYMBOL)
        result = dict(
            seconds=seconds, misset_deg=float(miss_opt), misset_start_deg=float(miss_start),
            G_over_truth=float(G) / G_true, Ncells=[float(Na), float(Nb), float(Nc)],
            ucell=[float(v) for v in (a, b, c, al, be, ga)], detz_shift_mm=float(detz) * 1e3, iterations=int(Mod.iter_num)
            if hasattr(Mod, "iter_num") else None,
        )
        print(json.dumps(result, indent=1))

        D = SIM_hopper.D
        np.savez(
            os.path.join(args.outdir, "pixels.npz"), img=img, spots=spots,
            all_data=Mod.all_data, all_trusted=Mod.all_trusted, all_background=Mod.all_background,
            all_fast=Mod.all_fast, all_slow=Mod.all_slow, all_pid=Mod.all_pid,
            all_sigma_rdout=np.broadcast_to(Mod.all_sigma_rdout, Mod.all_data.shape),
            rois=np.array(Mod.rois),
            tilt_abc=np.array(Mod.tilt_abc),
            G_true=G_true, G_start=G_true * G_FACTOR_START, fluence=float(D.fluence), oversample=int(D.oversample),
            default_F=float(D.default_F), ncells_true=NCELLS_TRUE, ncells_start=NCELLS_START,
        )
        with open(os.path.join(args.outdir, "models.json"), "w") as out:
            json.dump(dict(truth=truth.to_dict(), start=start.to_dict(), hopper=Eopt.crystal.to_dict(),
                           detector=SIM.detector.to_dict(), beam=SIM.D.beam.to_dict()), out)
        with open(os.path.join(args.outdir, "hopper.json"), "w") as out:
            json.dump(result, out, indent=1)


def stage_real(args):
    """One frame of a rotation dataset as a still; hopper's solution stands in for the truth."""
    import dxtbx
    from dials.array_family import flex
    from dxtbx.model import Experiment, ExperimentList
    from dxtbx.model.crystal import Crystal
    from libtbx.phil import parse
    from scitbx.matrix import col, sqr
    from simtbx.diffBragg import hopper_utils, utils
    from simtbx.diffBragg.device import DeviceWrapper
    from simtbx.diffBragg.phil import hopper_phil, philz
    from simtbx.nanoBragg import make_imageset

    expt = ExperimentList.from_file(args.expt, check_format=False)[0]
    refls = flex.reflection_table.from_file(args.refl)
    os.chdir(args.outdir)
    with DeviceWrapper(0):
        frame = args.frame - expt.scan.get_image_range()[0]  # scan point at the start of the frame
        angle = expt.scan.get_angle_from_array_index(frame + 0.5, deg=True)
        gonio = expt.goniometer
        S, F = sqr(gonio.get_setting_rotation()), sqr(gonio.get_fixed_rotation())
        R = col(gonio.get_rotation_axis_datum()).axis_and_angle_as_r3_rotation_matrix(angle, deg=True)
        A = S * R * F * sqr(expt.crystal.get_A_at_scan_point(frame))
        real = A.inverse()
        start = Crystal(real[:3], real[3:6], real[6:9], expt.crystal.get_space_group())

        image_file = expt.imageset.paths()[0].replace("00001", "%05d" % args.frame)
        raw = dxtbx.load(image_file).get_raw_data().as_numpy_array().astype(np.float64)
        E = Experiment()
        E.crystal = start
        E.detector = expt.detector
        E.beam = expt.beam
        E.imageset = make_imageset([raw], E.beam, E.detector)

        z = refls["xyzobs.px.value"].parts()[2]
        chosen = refls.select((z >= frame) & (z < frame + 1) & (refls["intensity.sum.value"] > 0))
        chosen = chosen.select(flex.sort_permutation(chosen["intensity.sum.value"], reverse=True))[:args.spots]
        chosen = chosen.select(chosen["xyzobs.px.value"].parts()[2] >= 0)
        utils.refls_to_q(chosen, E.detector, E.beam, update_table=True)
        utils.refls_to_hkl(chosen, E.detector, E.beam, E.crystal, update_table=True)
        print("%d reflections on frame %d of %d" % (len(chosen), args.frame, len(refls)))
        hkl = np.array(chosen["refl_hkl"] if "refl_hkl" in chosen else chosen["miller_index"].as_vec3_double())
        print("start: rms distance of the spots from integer hkl %.3f" % np.sqrt(((hkl - np.round(hkl)) ** 2).sum(1).mean()))

        P = parse(hopper_phil + philz).extract()
        P.init.G = args.init_g
        P.init.Nabc = NCELLS_START
        P.init.detz_shift = 0
        P.types.G = "positive"
        P.types.Nabc = "ranged"
        P.roi.shoebox_size = 15
        P.relative_tilt = False
        P.roi.fit_tilt = False
        P.roi.pad_shoebox_for_background_estimation = 10
        P.roi.reject_edge_reflections = False
        P.refiner.sigma_r = 1
        P.refiner.adu_per_photon = 1
        P.simulator.oversample = 1
        P.simulator.init_scale = 1
        P.simulator.beam.size_mm = 1
        P.simulator.total_flux = 1e12
        P.use_restraints = False
        P.fix.ucell = args.fix_cell
        P.simulator.structure_factors.mtz_name = args.mtz
        P.simulator.structure_factors.mtz_column = "IMEAN,SIGIMEAN"
        P.niter = 0
        P.sigmas.RotXYZ = [1, 1, 1]
        P.niter_per_J = 1
        P.method = "L-BFGS-B"
        P.ftol = 1e-10

        clock = time.perf_counter()
        Eopt, _, Mod, SIM_hopper, x = hopper_utils.refine(E, chosen, P, return_modeler=True)
        seconds = time.perf_counter() - clock
        G, rotX, rotY, rotZ, Na, Nb, Nc, *_, a, b, c, al, be, ga, detz = hopper_utils.get_param_from_x(x, Mod)
        miss_opt, _ = utils.compare_with_ground_truth(
            *start.get_real_space_vectors(), dxcryst_models=[Eopt.crystal, start],
            symbol=expt.crystal.get_space_group().info().type().lookup_symbol())
        result = dict(seconds=seconds, misset_deg=float(miss_opt), G=float(G), Ncells=[float(Na), float(Nb), float(Nc)],
                      ucell=[float(v) for v in (a, b, c, al, be, ga)])
        print(json.dumps(result, indent=1))
        D = SIM_hopper.D
        SIM_hopper.crystal.miller_array.as_mtz_dataset(column_root_label="F").mtz_object().write("demo.mtz")
        model_image = np.zeros_like(raw)
        model_image[Mod.all_slow, Mod.all_fast] = Mod.best_model
        np.savez(
            os.path.join(args.outdir, "pixels.npz"), img=raw, spots=model_image,
            all_data=Mod.all_data, all_trusted=Mod.all_trusted, all_background=Mod.all_background,
            all_fast=Mod.all_fast, all_slow=Mod.all_slow, all_pid=Mod.all_pid,
            all_sigma_rdout=np.broadcast_to(Mod.all_sigma_rdout, Mod.all_data.shape),
            rois=np.array(Mod.rois), tilt_abc=np.array(Mod.tilt_abc),
            G_true=float(G), G_start=float(args.init_g), fluence=float(D.fluence), oversample=1,
            default_F=float(D.default_F), ncells_true=[float(Na), float(Nb), float(Nc)], ncells_start=NCELLS_START,
        )
        with open(os.path.join(args.outdir, "models.json"), "w") as out:
            json.dump(dict(truth=Eopt.crystal.to_dict(), start=start.to_dict(), detector=expt.detector.to_dict(),
                           beam=expt.beam.to_dict()), out)
        with open(os.path.join(args.outdir, "hopper.json"), "w") as out:
            json.dump(result, out, indent=1)


def stage_torch(args):
    import torch
    from dxtbx.model import BeamFactory, DetectorFactory
    from dxtbx.model.crystal import CrystalFactory
    from iotbx.reflection_file_reader import any_reflection_file
    from simtbx.diffBragg import utils

    from nanobrag_torch.compat.cctbx import experiment_model_from_dxtbx
    from nanobrag_torch.utils.losses import gaussian_nll
    from nanobrag_torch.utils.refinement import refine_lbfgsb

    saved = np.load(os.path.join(args.outdir, "pixels.npz"))
    models = json.load(open(os.path.join(args.outdir, "models.json")))
    truth = CrystalFactory.from_dict(models["truth"])
    start = CrystalFactory.from_dict(models["start"])
    detector = DetectorFactory.from_dict(models["detector"])
    beam = BeamFactory.from_dict(models["beam"])
    miller = any_reflection_file(os.path.join(args.outdir, "demo.mtz")).as_miller_arrays()[0]
    G_true = float(saved["G_true"])
    device = torch.device(args.device)

    def build(crystal, Ncells, G):
        model = experiment_model_from_dxtbx(
            detector, beam, crystal, Ncells_abc=Ncells, shape="gauss", default_F=float(saved["default_F"]),
            detector_thick_mm=0.0,  # hopper's simulator ignores the sensor
            fluence=float(saved["fluence"]), spot_scale=G, oversample=max(int(saved["oversample"]), 1),
            miller_array=miller, anomalous=True, rois=saved["rois"], tilt_abc=saved["tilt_abc"], device=device,
        )
        return model

    def image_of(values, fast, slow, shape):
        image = np.zeros(shape, dtype=values.dtype)
        image[slow, fast] = values
        return image

    shape = (detector[0].get_image_size()[1], detector[0].get_image_size()[0])
    fast, slow = saved["all_fast"], saved["all_slow"]
    symbol = truth.get_space_group().info().type().lookup_symbol()
    model = build(start, tuple(float(n) for n in saved["ncells_start"]), float(saved["G_start"]))
    data = model.roi_values(image_of(saved["all_data"].astype(np.float64), fast, slow, shape))
    trusted = model.roi_values(image_of(saved["all_trusted"], fast, slow, shape))
    sigma = model.roi_values(image_of(saved["all_sigma_rdout"].astype(np.float64), fast, slow, shape))
    print("pixels", data.numel(), "trusted", int(trusted.sum()), "hopper pixels", len(saved["all_data"]))
    with torch.no_grad():
        planes = model.background_params.values()
        expected = model.roi_values(image_of(saved["all_background"].astype(np.float64), fast, slow, shape))
    print("background vs hopper, max difference", float((planes - expected).abs().max()))

    reference = build(truth, tuple(float(n) for n in saved["ncells_true"]), G_true)
    with torch.no_grad():
        pred = reference(roi_only=True)
        spots = reference.roi_values(saved["spots"])
        bragg = pred - planes
    print("Bragg model at the truth vs diffBragg's spots: max relative difference",
          float((bragg - spots).abs().max() / spots.max()))

    model.freeze().refine("scale", "misset", "Ncells", *([] if args.fix_cell else ["cell"]))
    if args.free_background:
        model.refine("background")

    def objective():
        return gaussian_nll(model(roi_only=True), data, sigma, trusted=trusted)

    def report(tag, seconds):
        with torch.no_grad():
            model(roi_only=True)
        a, b, c = (model._crystal.a, model._crystal.b, model._crystal.c)
        real = [tuple(float(v) for v in x.detach().cpu()) for x in (a, b, c)]
        misset = utils.compare_with_ground_truth(*truth.get_real_space_vectors(), dxcryst_models=[_crystal_from(real)],
                                                 symbol=symbol)[0]
        Ncells = (model.crystal_params.base_N_cells * torch.exp(model.crystal_params.delta_log_N_cells)).tolist()
        G = (model.beam_params.base_spot_scale * torch.exp(model.beam_params.delta_log_spot_scale)).item()
        cell = [getattr(model._crystal, f"cell_{n}").item() for n in ("a", "b", "c", "alpha", "beta", "gamma")]
        out = dict(stage=tag, seconds=seconds, nll=objective().item(), misset_deg=float(misset),
                   G_over_truth=G / G_true, Ncells=Ncells, ucell=cell)
        print(json.dumps(out))
        return out

    def _crystal_from(real):
        from dxtbx.model.crystal import Crystal
        return Crystal(real[0], real[1], real[2], symbol)

    log = [report("start", 0.0)]
    clock = time.perf_counter()
    fit = refine_lbfgsb(model, objective, maxiter=args.lbfgs_steps, scales=dict(scale=args.g_unit))
    seconds = time.perf_counter() - clock  # before report(), whose symmetry search takes seconds
    log.append(report("L-BFGS-B", seconds))
    log[-1].update(evaluations=int(fit.nfev), iterations=int(fit.nit), message=str(fit.message))
    print(log[-1]["evaluations"], "evaluations,", log[-1]["iterations"], "iterations:", log[-1]["message"])
    name = "torch_%s%s.json" % (args.device, "_freebg" if args.free_background else "")
    with open(os.path.join(args.outdir, name), "w") as out:
        json.dump(log, out, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["hopper", "real", "torch"])
    parser.add_argument("--expt", help="real: a scan-varying indexed or scaled .expt")
    parser.add_argument("--refl", help="real: its reflections")
    parser.add_argument("--mtz", help="real: merged intensities, IMEAN")
    parser.add_argument("--frame", type=int, default=900, help="real: image number")
    parser.add_argument("--spots", type=int, default=150, help="real: strongest reflections kept")
    parser.add_argument("--init-g", dest="init_g", type=float, default=1e4, help="real: starting scale")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--perturb-deg", dest="perturb_deg", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lbfgs-steps", dest="lbfgs_steps", type=int, default=300)
    parser.add_argument("--g-unit", dest="g_unit", type=float, default=1.0, help="optimizer unit of log G")
    parser.add_argument("--fix-cell", dest="fix_cell", action="store_true", help="hold the unit cell")
    parser.add_argument("--free-background", dest="free_background", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    args.outdir = os.path.abspath(args.outdir)
    return dict(hopper=stage_hopper, real=stage_real, torch=stage_torch)[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
