#!/usr/bin/env python
"""Refine a perturbed synthetic shot with diffBragg's hopper and with ExperimentModel (plan C6).

Stage ``real`` does the same for one frame of a rotation dataset, and stage ``still`` for one
indexed multi-panel still, with hopper's solution standing in for the truth. Stage ``hopper`` simulates a noisy still as simtbx's tst_diffBragg_hopper_refine.py does,
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
        P.simulator.detector.force_zero_thickness = True  # as the torch stage
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


def _planes_from_roi_corner(rois, background):
    """tilt_abc about each ROI's corner, from the pixels hopper subtracts.

    hopper's own tilt_abc is about the corner of the shoebox padded for the fit, which can be clipped at a panel edge.
    """
    planes, start = [], 0
    for x1, x2, y1, y2 in rois:
        Y, X = np.indices((y2 - y1, x2 - x1))
        stop = start + X.size
        design = np.column_stack((X.ravel(), Y.ravel(), np.ones(X.size)))
        planes.append(np.linalg.lstsq(design, background[start:stop], rcond=None)[0])
        start = stop
    return np.array(planes)


def _binned_spectrum(energies, weights, channels, wavelength):
    """The measured spectrum on ``channels`` wavelengths, centred on the refined ``wavelength``.

    The spectrometer's calibration fixes the shape; the centroid comes from indexing.
    """
    weights = np.clip(weights, 0, None)
    keep = weights > 0.01 * weights.max()
    energies, weights = energies[keep], weights[keep]
    centroid = (energies * weights).sum() / weights.sum()
    spread = np.sqrt((weights * (energies - centroid) ** 2).sum() / weights.sum())
    edges = np.linspace(centroid - 3 * spread, centroid + 3 * spread, channels + 1)
    binned = np.histogram(energies, edges, weights=weights)[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    used = binned > 0
    binned, centres = binned[used], centres[used]
    centres += 12398.4198 / wavelength - (centres * binned).sum() / binned.sum()
    print("spectrum: %d channels, rms %.2f eV of %.2f eV" % (used.sum(), spread, centroid))
    return 12398.4198 / centres, binned / binned.sum()


def stage_still(args):
    """One indexed still from --shot (pixels, spectrum and models exported with psana); hopper stands in for the truth."""
    import pickle

    from dials.array_family import flex
    from dxtbx.model import Experiment, ExperimentList
    from libtbx.phil import parse
    from simtbx.diffBragg import hopper_utils, utils
    from simtbx.diffBragg.device import DeviceWrapper
    from simtbx.diffBragg.phil import hopper_phil, philz
    from simtbx.nanoBragg import make_imageset

    shot = np.load(os.path.join(args.shot, "shot.npz"))
    expt = ExperimentList.from_file(os.path.join(args.shot, "shot.expt"), check_format=False)[0]
    refls = flex.reflection_table.from_file(os.path.join(args.shot, "indexed.refl"))
    mask = np.load(args.mask)["mask"]
    img = shot["img"].astype(np.float64)
    panels = list(img)
    keV_per_photon = 12.3984198 / expt.beam.get_wavelength()
    pedestal = img[mask & (np.abs(img) < 0.25 * keV_per_photon)]
    sigma_r = args.sigma_r if args.sigma_r else float(pedestal.std())
    print("%.4f keV per photon, readout sigma %.4f keV from %d pixels" % (keV_per_photon, sigma_r, pedestal.size))

    start = expt.crystal
    os.chdir(args.outdir)
    with open("mask.pickle", "wb") as out:
        pickle.dump(tuple(flex.bool(m) for m in mask), out)
    with DeviceWrapper(0):
        E = Experiment()
        E.crystal = start
        E.detector = expt.detector
        E.beam = expt.beam
        E.imageset = make_imageset(panels, E.beam, E.detector)

        chosen = refls.select(refls["intensity.sum.value"] > 0)
        chosen = chosen.select(flex.sort_permutation(chosen["intensity.sum.value"], reverse=True))[:args.spots]
        utils.refls_to_q(chosen, E.detector, E.beam, update_table=True)
        utils.refls_to_hkl(chosen, E.detector, E.beam, E.crystal, update_table=True)
        print("%d of %d indexed reflections" % (len(chosen), len(refls)))

        P = parse(hopper_phil + philz).extract()
        P.init.G = args.init_g
        P.init.Nabc = tuple(args.ncells_start)
        P.init.detz_shift = 0
        P.types.G = "positive"
        P.types.Nabc = "ranged"
        P.roi.shoebox_size = args.shoebox
        P.relative_tilt = False
        P.roi.fit_tilt = True
        P.roi.fit_tilt_using_weights = False
        P.roi.pad_shoebox_for_background_estimation = 10
        P.roi.reject_edge_reflections = True
        P.roi.reject_roi_with_hotpix = False
        P.roi.hotpixel_mask = os.path.abspath("mask.pickle")
        P.refiner.sigma_r = sigma_r
        P.refiner.adu_per_photon = keV_per_photon
        P.simulator.oversample = 1
        P.simulator.init_scale = 1
        P.simulator.beam.size_mm = 1
        P.simulator.total_flux = 1e12
        # From a panel, nanoBragg takes 2 layers of the full thickness in units 1000 times too long.
        P.simulator.detector.force_zero_thickness = not args.sensor_layers
        if args.sensor_layers:
            thick_mm = expt.detector[0].get_thickness()
            build_simulator = utils.simulator_for_refinement

            def layered_simulator(*build_args):
                SIM = build_simulator(*build_args)
                SIM.D.detector_thicksteps = args.sensor_layers
                SIM.D.detector_thick_mm = thick_mm
                SIM.D.detector_thickstep_mm = thick_mm / args.sensor_layers
                SIM.D.detector_attenuation_length_mm = 1 / expt.detector[0].get_mu()
                return SIM

            utils.simulator_for_refinement = layered_simulator
        if args.eta:
            P.init.eta_abc = [args.eta] * 3
            P.fix.eta_abc = False
            P.simulator.crystal.has_isotropic_mosaicity = True
            P.simulator.crystal.num_mosaicity_samples = args.mosaic_domains
        wavelengths = weights = None
        if args.channels:
            wavelengths, weights = _binned_spectrum(
                shot["energies"], shot["weights"], args.channels, expt.beam.get_wavelength())
            utils.save_spectra_file("spectrum.lam", wavelengths, weights)
        P.use_restraints = False
        P.fix.ucell = args.fix_cell
        P.simulator.structure_factors.mtz_name = args.mtz
        P.simulator.structure_factors.mtz_column = "IMEAN,SIGIMEAN"
        P.niter = 0
        P.sigmas.RotXYZ = [1, 1, 1]
        P.niter_per_J = 1
        P.method = "L-BFGS-B"
        P.ftol = 1e-10
        if args.hopper_maxiter:
            P.lbfgs_maxiter = args.hopper_maxiter

        clock = time.perf_counter()
        # refine() replaces simulator.spectrum.filename with this argument
        spectrum_file = os.path.abspath("spectrum.lam") if args.channels else None
        Eopt, _, Mod, SIM_hopper, x = hopper_utils.refine(E, chosen, P, spec=spectrum_file, return_modeler=True)
        seconds = time.perf_counter() - clock
        G, rotX, rotY, rotZ, Na, Nb, Nc, *_, a, b, c, al, be, ga, detz = hopper_utils.get_param_from_x(x, Mod)
        miss_opt, _ = utils.compare_with_ground_truth(
            *start.get_real_space_vectors(), dxcryst_models=[Eopt.crystal, start],
            symbol=start.get_space_group().info().type().lookup_symbol())
        eta = 0.0
        if args.eta:
            eta = float(Mod.P["eta_abc0"].get_val(x[Mod.P["eta_abc0"].xpos]))
        rois = np.array(Mod.rois)
        pids = np.array(Mod.pids)
        area = (rois[:, 1] - rois[:, 0]) * (rois[:, 3] - rois[:, 2])
        keep = np.array(Mod.selection_flags, bool)
        if area.sum() != len(Mod.all_data):
            assert area[keep].sum() == len(Mod.all_data), (area.sum(), area[keep].sum(), len(Mod.all_data))
            rois, pids = rois[keep], pids[keep]
        tilt_abc = _planes_from_roi_corner(rois, Mod.all_background)
        trusted = Mod.all_trusted
        cc = np.corrcoef(Mod.best_model[trusted] + Mod.all_background[trusted], Mod.all_data[trusted])[0, 1]
        result = dict(seconds=seconds, misset_deg=float(miss_opt), G=float(G), Ncells=[float(Na), float(Nb), float(Nc)],
                      ucell=[float(v) for v in (a, b, c, al, be, ga)], cc=float(cc), eta=eta,
                      rois=len(rois), pixels=len(Mod.all_data), trusted=int(trusted.sum()))
        print(json.dumps(result, indent=1))
        D = SIM_hopper.D
        state = {}
        for name in ("fluence", "flux", "beamsize_mm", "exposure_s", "spot_scale", "default_F", "oversample",
                     "oversample_omega", "polarization", "nopolar", "detector_thick_mm", "detector_thicksteps",
                     "detector_attenuation_length_mm", "detector_thickstep_mm", "wavelength_A", "Ncells_abc",
                     "xtal_shape", "mosaic_domains", "mosaic_spread_deg", "number_of_sources", "kahn_factor",
                     "quantum_gain", "adc_offset_adu", "point_pixel", "curved_detector"):
            try:
                state[name] = str(getattr(D, name))
            except Exception as err:
                state[name] = "unavailable: %s" % type(err).__name__
        state["channels"] = len(SIM_hopper.beam.spectrum)
        panel = SIM_hopper.detector[0] if SIM_hopper.detector is not None else None
        if panel is not None:
            state["panel_thickness"] = panel.get_thickness()
            state["panel_mu"] = panel.get_mu()
        print(json.dumps(state, indent=0))
        with open("simulator_state.json", "w") as out:
            json.dump(state, out, indent=1)
        SIM_hopper.crystal.miller_array.as_mtz_dataset(column_root_label="F").mtz_object().write("demo.mtz")
        model_image = np.zeros_like(img)
        model_image[Mod.all_pid, Mod.all_slow, Mod.all_fast] = Mod.best_model
        extra = dict(wavelengths=wavelengths, weights=weights) if args.channels else {}
        np.savez(
            os.path.join(args.outdir, "pixels.npz"), spots=model_image,
            all_data=Mod.all_data, all_trusted=Mod.all_trusted, all_background=Mod.all_background,
            all_fast=Mod.all_fast, all_slow=Mod.all_slow, all_pid=Mod.all_pid,
            all_sigma_rdout=np.broadcast_to(Mod.all_sigma_rdout, Mod.all_data.shape),
            rois=rois, tilt_abc=tilt_abc, panels=pids,
            G_true=float(G), G_start=float(args.init_g), fluence=float(D.fluence), oversample=1,
            default_F=float(D.default_F), ncells_true=[float(Na), float(Nb), float(Nc)],
            ncells_start=list(args.ncells_start), sensor_layers=args.sensor_layers,
            eta_true=eta, eta_start=args.eta, mosaic_domains=args.mosaic_domains, **extra,
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

    layers = int(saved["sensor_layers"]) if "sensor_layers" in saved.files else 0
    eta_true = float(saved["eta_true"]) if "eta_true" in saved.files else 0.0
    sensor = dict(detector_thick_mm=0.0)  # the other stages force zero thickness in hopper too
    if layers:
        sensor = dict(detector_thicksteps=layers, oversample_thick=True)

    def build(crystal, Ncells, G, eta=0.0):
        if eta:
            sensor.update(mosaic_spread_deg=eta, mosaic_domains=int(saved["mosaic_domains"]))
        model = experiment_model_from_dxtbx(
            detector, beam, crystal, Ncells_abc=Ncells, shape="gauss", default_F=float(saved["default_F"]),
            fluence=float(saved["fluence"]) / max(layers, 1), spot_scale=G, oversample=max(int(saved["oversample"]), 1),
            **sensor,
            miller_array=miller, anomalous=True, rois=saved["rois"], tilt_abc=saved["tilt_abc"], spectrum=spectrum,
            panels=saved["panels"] if len(detector) > 1 else None, device=device,
        )
        return model

    spectrum = None
    if "wavelengths" in saved.files:
        spectrum = [(float(w), float(f)) for w, f in zip(saved["wavelengths"], saved["weights"])]
    if args.spectrum_npz:  # the image doesn't depend on the channel count
        other = np.load(args.spectrum_npz)
        spectrum = [(float(w), float(f)) for w, f in zip(other["wavelengths"], other["weights"])]

    def image_of(values, fast, slow, shape):
        image = np.zeros(shape, dtype=values.dtype)
        if len(detector) > 1:
            image[saved["all_pid"], slow, fast] = values
        else:
            image[slow, fast] = values
        return image

    shape = (detector[0].get_image_size()[1], detector[0].get_image_size()[0])
    if len(detector) > 1:
        shape = (len(detector), *shape)
    fast, slow = saved["all_fast"], saved["all_slow"]
    symbol = truth.get_space_group().info().type().lookup_symbol()
    model = build(start, tuple(float(n) for n in saved["ncells_start"]), float(saved["G_start"]),
                  float(saved["eta_start"]) if eta_true else 0.0)
    data = model.roi_values(image_of(saved["all_data"].astype(np.float64), fast, slow, shape))
    trusted = model.roi_values(image_of(saved["all_trusted"], fast, slow, shape))
    sigma = model.roi_values(image_of(saved["all_sigma_rdout"].astype(np.float64), fast, slow, shape))
    print("pixels", data.numel(), "trusted", int(trusted.sum()), "hopper pixels", len(saved["all_data"]))
    with torch.no_grad():
        planes = model.background_params.values()
        expected = model.roi_values(image_of(saved["all_background"].astype(np.float64), fast, slow, shape))
    print("background vs hopper, max difference", float((planes - expected).abs().max()))

    reference = build(truth, tuple(float(n) for n in saved["ncells_true"]), G_true, eta_true)
    with torch.no_grad():
        pred = reference(roi_only=True)
        spots = reference.roi_values(saved["spots"])
        bragg = pred - planes
    print("Bragg model at the truth vs diffBragg's spots: max relative difference",
          float((bragg - spots).abs().max() / spots.max()))
    bright = spots > 0.01 * spots.max()
    ratio = (bragg[bright] / spots[bright]).cpu().numpy()
    print("torch/diffBragg on %d bright pixels: percentiles 1, 50, 99" % bright.sum(), np.percentile(ratio, [1, 50, 99]))

    model.freeze().refine("scale", "misset", "Ncells", *([] if args.fix_cell else ["cell"]), *(["eta"] if eta_true else []))
    if args.free_background:
        model.refine("background")
    if args.refine_wavelength:
        model.refine("wavelength")

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
        with torch.no_grad():
            cc = torch.corrcoef(torch.stack((model(roi_only=True)[trusted], data[trusted].to(torch.float64))))[0, 1]
        out = dict(stage=tag, seconds=seconds, nll=objective().item(), misset_deg=float(misset),
                   G_over_truth=G / G_true, Ncells=Ncells, ucell=cell, cc=cc.item(),
                   eta=model.crystal_params.eta().item() if eta_true else 0.0,
                   lambda_offset=model.beam_params.lambda_offset.item())
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
    parser.add_argument("stage", choices=["hopper", "real", "still", "torch"])
    parser.add_argument("--expt", help="real: a scan-varying indexed or scaled .expt")
    parser.add_argument("--refl", help="real: its reflections")
    parser.add_argument("--mtz", help="real: merged intensities, IMEAN")
    parser.add_argument("--shot", help="still: directory with shot.npz, shot.expt and indexed.refl")
    parser.add_argument("--mask", help="still: .npz with mask, True on good pixels (panel, slow, fast)")
    parser.add_argument("--channels", type=int, default=0, help="still: spectrum channels, 0 for one wavelength")
    parser.add_argument("--sensor-layers", dest="sensor_layers", type=int, default=0,
                        help="still: layers across the panels' sensor, 0 for no sensor")
    parser.add_argument("--eta", type=float, default=0.0, help="still: starting mosaic spread in degrees, 0 for none")
    parser.add_argument("--mosaic-domains", dest="mosaic_domains", type=int, default=26, help="still: even")
    parser.add_argument("--hopper-maxiter", dest="hopper_maxiter", type=int, default=None, help="still: L-BFGS-B cap")
    parser.add_argument("--spectrum-npz", dest="spectrum_npz", help="torch: wavelengths and weights to use instead")
    parser.add_argument("--refine-wavelength", dest="refine_wavelength", action="store_true", help="torch")
    parser.add_argument("--shoebox", type=int, default=15, help="still: shoebox size")
    parser.add_argument("--sigma-r", dest="sigma_r", type=float, default=0.0,
                        help="still: readout noise in image units, by default from the pedestal")
    parser.add_argument("--ncells-start", dest="ncells_start", type=float, nargs=3, default=NCELLS_START)
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
    return dict(hopper=stage_hopper, real=stage_real, still=stage_still, torch=stage_torch)[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
