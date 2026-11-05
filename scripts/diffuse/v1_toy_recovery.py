"""
V1 of docs/diffuse_joint_plan.md, section 5: toy recovery of Bragg and diffuse, alone and jointly.

A synthetic still from known G (spot_scale), misset, cell and GaussianHalo (amplitude, sigma) is
refined back from a start off in every parameter, with refine_lbfgsb on the Poisson likelihood
(as its deviance, the same minimum as utils.losses.poisson_nll):

- bragg:   Bragg-only image, model without a diffuse term; refines G, misset, cell;
- diffuse: diffuse-only image (F = 0), G known (with k_d frozen, G and the halo amplitude are one
           number there); refines misset, cell, amplitude, sigma;
- joint:   the sum; refines all of them. The Bragg term fixes G, so the amplitude is free.

For each fit it reports the error of every parameter (and with --poisson its z, the error over
the Fisher standard deviation), the Fisher information's condition number (raw, and of the
correlation matrix), and the correlations between parameters, at the truth. The Fisher matrix
is F = J^T diag(1/mu) J, the Poisson information, with J by central differences of the image.
The metric is spot_metric=HKL, the one the plan recommends for joint fits.

    python scripts/diffuse/v1_toy_recovery.py --size 24                      # noise-free, seconds
    python scripts/diffuse/v1_toy_recovery.py --size 128 --poisson --seed 1 --out v1.json
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import numpy as np
import torch

from nanobrag_torch.config import (BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention,
                                   DetectorPivot, SpotMetric)
from nanobrag_torch.diffuse import GaussianHalo
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.utils.refinement import refine_lbfgsb

DTYPE = torch.float64
SCALE = ("beam_params.delta_log_spot_scale", None)
MISSET = [("crystal_params.delta_misset_raw", i) for i in range(3)]
CELL = [("crystal_params.delta_log_a", None), ("crystal_params.delta_log_b", None), ("crystal_params.delta_log_c", None)]
HALO = [("diffuse_params.model.log_amplitude", None), ("diffuse_params.model.log_sigma", None)]
TRUTH = {  # raw parameter values of the synthetic shot; the start is zero offsets and START_HALO
    SCALE: math.log(1.3),
    MISSET[0]: 0.004, MISSET[1]: -0.003, MISSET[2]: 0.002,
    CELL[0]: 0.002, CELL[1]: -0.0015, CELL[2]: 0.001,
    HALO[0]: math.log(2000.0), HALO[1]: math.log(0.12),
}
START_HALO = (1400.0, 0.16)  # a halo of a few photons a pixel under spots of 1e4
FITS = {
    "bragg": dict(diffuse=False, bragg=True, refine=[SCALE, *MISSET, *CELL]),
    "diffuse": dict(diffuse=True, bragg=False, refine=[*MISSET, *CELL, *HALO], known=[SCALE]),
    "joint": dict(diffuse=True, bragg=True, refine=[SCALE, *MISSET, *CELL, *HALO]),
}


def build(size, diffuse, bragg, device="cpu"):
    crystal = CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(5, 5, 5),
                            default_F=100.0 if bragg else 0.0, shape=CrystalShape.GAUSS, spot_metric=SpotMetric.HKL)
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=12.8 / size, spixels=size, fpixels=size, oversample=1,
                              detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)
    beam = BeamConfig(wavelength_A=1.2, fluence=2e29 * (24.0 / size) ** 2)  # the same photons per solid angle
    halo = GaussianHalo(*START_HALO) if diffuse else None
    model = ExperimentModel(crystal, detector, beam, dtype=DTYPE, device=torch.device(device), diffuse=halo,
                            crystal_system="orthorhombic")
    return model.freeze()


def _parameter(model, key):
    name, component = key
    parameter = dict(model.named_parameters())[name]
    return parameter, component


def get(model, key):
    parameter, component = _parameter(model, key)
    return float(parameter if component is None else parameter[component])


def put(model, key, value):
    parameter, component = _parameter(model, key)
    with torch.no_grad():
        if component is None:
            parameter.fill_(value)
        else:
            parameter[component] = value


def image(model):
    return model(bragg_only=model.diffuse is None)


def deviance(mu, data):
    """The Poisson deviance, sum mu - d - d log(mu / d): poisson_nll less its constant, so that its
    minimum is 0 and L-BFGS-B's relative ftol is not lost against sum log(d!)."""
    positive = data > 0
    ratio = torch.where(positive, mu / torch.where(positive, data, torch.ones_like(data)), torch.ones_like(mu))
    return (mu - data - torch.where(positive, data * torch.log(ratio), torch.zeros_like(mu))).sum()


def fisher(model, keys, step=1e-6):
    """Poisson information at the model's current parameters, J by central differences."""
    with torch.no_grad():
        mu = image(model).reshape(-1)
        columns = []
        for key in keys:
            start = get(model, key)
            put(model, key, start + step)
            plus = image(model).reshape(-1)
            put(model, key, start - step)
            minus = image(model).reshape(-1)
            put(model, key, start)
            columns.append((plus - minus) / (2 * step))
    J = torch.stack(columns, dim=1)
    keep = mu > 0
    return (J[keep] / mu[keep, None]).T @ J[keep]


def run_fit(name, size, poisson, seed, device="cpu"):
    fit = FITS[name]
    truth = build(size, fit["diffuse"], fit["bragg"], device)
    for key in [SCALE, *MISSET, *CELL] + (HALO if fit["diffuse"] else []):
        put(truth, key, TRUTH[key])
    with torch.no_grad():
        expected = image(truth)
    data = expected
    if poisson:
        data = torch.poisson(expected.cpu(), generator=torch.Generator().manual_seed(seed)).to(expected.device)
    keys = fit["refine"]
    information = fisher(truth, keys).cpu().numpy()

    model = build(size, fit["diffuse"], fit["bragg"], device)
    for key in fit.get("known", []):
        put(model, key, TRUTH[key])
    for key in keys:
        model.refine(key[0])
    result = refine_lbfgsb(model, lambda: deviance(image(model), data), ftol=1e-15, gtol=1e-10, maxiter=1000)

    covariance = np.linalg.inv(information)
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)
    scaled = information / np.sqrt(np.outer(np.diag(information), np.diag(information)))
    labels = [f"{n}" + ("" if c is None else f"[{c}]") for n, c in keys]
    errors = {label: get(model, key) - TRUTH[key] for label, key in zip(labels, keys)}
    return {
        "fit": name, "size": size, "poisson": poisson, "seed": seed, "converged": bool(result.success),
        "iterations": int(result.nit), "message": str(result.message),
        "photons": {"max": float(expected.max()), "mean": float(expected.mean())},
        "errors": errors,
        "z": {label: errors[label] / s for label, s in zip(labels, sd)},
        "fisher_sd": dict(zip(labels, sd.tolist())),
        "fisher_condition": float(np.linalg.cond(information)),
        "fisher_condition_scaled": float(np.linalg.cond(scaled)),
        "correlations": {f"{labels[i]} ~ {labels[j]}": float(correlation[i, j])
                         for i in range(len(keys)) for j in range(i + 1, len(keys))},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--poisson", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fits", nargs="+", default=list(FITS))
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    results = [run_fit(name, args.size, args.poisson, args.seed, args.device) for name in args.fits]
    for r in results:
        worst = max(abs(v) for v in (r["z"] if args.poisson else r["errors"]).values())
        print(f"{r['fit']:8s} converged={r['converged']} iterations={r['iterations']} "
              f"worst {'|z|' if args.poisson else '|error|'}={worst:.2e} "
              f"cond={r['fisher_condition']:.2e} (scaled {r['fisher_condition_scaled']:.2e})")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
