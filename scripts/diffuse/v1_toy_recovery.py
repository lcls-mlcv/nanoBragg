"""
V1 of docs/diffuse_joint_plan.md, section 5: toy recovery of Bragg and diffuse, alone and jointly.

A synthetic still from known G (spot_scale), misset, cell and GaussianHalo (amplitude, sigma) is
refined back from a start off in every parameter, with refine_lbfgsb on the Poisson likelihood
(as its deviance, the same minimum as utils.losses.poisson_nll):

- bragg:   Bragg-only image, model without a diffuse term; refines G, misset, cell;
- diffuse: diffuse-only image (F = 0), G known (with k_d frozen, G and the halo amplitude are one
           number there); refines misset, cell, amplitude, sigma;
- joint:   the sum; refines all of them. The Bragg term fixes G, so the amplitude is free.

Each fit reports the errors (with --poisson, z = error / Fisher sd), the Fisher information's
condition number (raw, and of its correlation matrix) and the correlations, at the truth. The
Fisher matrix is F = J^T diag(1/mu) J, the Poisson information, with J by central differences.
The metric is spot_metric=HKL, the one the plan recommends for joint fits.

Separability, the gate: for each parameter of the joint fit, its Fisher sd in the joint fit over
its sd in the fit of the half it belongs to (G: bragg; the halo: diffuse; misset and cell: the
better of the two). Near 1 (or below) the halves do not trade off; the gate demands every
ratio <= SEPARABILITY_LIMIT and, with --poisson, |z| < Z_LIMIT in every fit and seed.

Two toys (TOYS). "separable" is the gate: a halo 7.5 times as wide as the spot (N = 10), so that
it is not hidden under it. "hidden" is a documented negative case, a halo about as wide as the spot,
where the joint fit cannot tell halo from spot: the gate requires it to be flagged as weakly
identified (a ratio above the limit), so that the metric is shown to discriminate. The exit
code is 0 only if the separable toy passes and the hidden one is flagged.

    python scripts/diffuse/v1_toy_recovery.py                                   # 24^2, noise-free
    python scripts/diffuse/v1_toy_recovery.py --size 128 --poisson --seeds 1 2 3 --out v1.json
"""
import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import numpy as np
import torch

from nanobrag_torch.config import (BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention,
                                   DetectorPivot, SpotMetric)
from nanobrag_torch.diffuse import GaussianHalo
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.utils.refinement import refine_lbfgsb

DTYPE = torch.float64
SEPARABILITY_LIMIT = 1.5
Z_LIMIT = 4.0
SCALE = ("beam_params.delta_log_spot_scale", None)
MISSET = [("crystal_params.delta_misset_raw", i) for i in range(3)]
CELL = [("crystal_params.delta_log_a", None), ("crystal_params.delta_log_b", None), ("crystal_params.delta_log_c", None)]
HALO = [("diffuse_params.model.log_amplitude", None), ("diffuse_params.model.log_sigma", None)]
GEOMETRY_TRUTH = {  # raw parameter values of the synthetic shot; the start is zero offsets
    SCALE: math.log(1.3),
    MISSET[0]: 0.004, MISSET[1]: -0.003, MISSET[2]: 0.002,
    CELL[0]: 0.002, CELL[1]: -0.0015, CELL[2]: 0.001,
}
FITS = {
    "bragg": dict(diffuse=False, bragg=True, refine=[SCALE, *MISSET, *CELL]),
    "diffuse": dict(diffuse=True, bragg=False, refine=[*MISSET, *CELL, *HALO], known=[SCALE]),
    "joint": dict(diffuse=True, bragg=True, refine=[SCALE, *MISSET, *CELL, *HALO]),
}


@dataclass(frozen=True)
class Toy:
    """One synthetic shot. The GAUSS-HKL spot is exp(-2 (N dh)^2 / 0.63) in intensity, so its sd
    in h is 0.397 / N; the halo's is sigma. amplitude is the halo's peak, e^2 per cell (F = 100
    gives the spot 1e4)."""

    n_cells: int = 10
    sigma: float = 0.3
    amplitude: float = 128.0
    fluence: float = 2e29  # at 24 x 24 pixels; scaled so that photons per solid angle do not change with size

    @property
    def spot_sd(self) -> float:
        return math.sqrt(0.63 / 4) / self.n_cells

    def truth(self) -> dict:
        return {**GEOMETRY_TRUTH, HALO[0]: math.log(self.amplitude), HALO[1]: math.log(self.sigma)}

    def start_halo(self) -> tuple:
        return 0.7 * self.amplitude, 1.3 * self.sigma


# Measured (2026-11-05, Fisher at the truth, 24^2 and 64^2 alike): the worst joint/alone sd ratio
# is set by halo sigma over spot sd. N = 5 (spot sd 0.079): sigma 0.05, 0.12, 0.3, 0.5 give 22, 14,
# 2.6, 2.3. N = 10 (0.040): 40, 2.2, 1.28, 1.23. The amplitude is scaled as sigma^-3 to keep the
# diffuse photons fixed; corr(amplitude, sigma) is about -0.9 in every joint fit.
TOYS = {
    # sigma / spot sd = 7.5: about 8 diffuse photons a pixel at 24^2 under spots of about 1e6
    "separable": Toy(n_cells=10, sigma=0.3, amplitude=128.0),
    # negative case, the first toy: sigma / spot sd = 1.5, the halo hides under the spots
    "hidden": Toy(n_cells=5, sigma=0.12, amplitude=2000.0),
}


def build(toy, size, diffuse, bragg, device="cpu"):
    n = toy.n_cells
    crystal = CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(n, n, n),
                            default_F=100.0 if bragg else 0.0, shape=CrystalShape.GAUSS, spot_metric=SpotMetric.HKL)
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=12.8 / size, spixels=size, fpixels=size, oversample=1,
                              detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)
    beam = BeamConfig(wavelength_A=1.2, fluence=toy.fluence * (24.0 / size) ** 2)
    halo = GaussianHalo(*toy.start_halo()) if diffuse else None
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


def label(key):
    name, component = key
    return name + ("" if component is None else f"[{component}]")


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


def truth_model(toy, fit, size, device="cpu"):
    model = build(toy, size, FITS[fit]["diffuse"], FITS[fit]["bragg"], device)
    truth = toy.truth()
    for key in [SCALE, *MISSET, *CELL] + (HALO if FITS[fit]["diffuse"] else []):
        put(model, key, truth[key])
    return model


def fisher_sd(toy, fit, size, device="cpu"):
    """{label: Fisher sd} of a fit's parameters at the truth, and the information matrix."""
    keys = FITS[fit]["refine"]
    information = fisher(truth_model(toy, fit, size, device), keys).cpu().numpy()
    sd = np.sqrt(np.diag(np.linalg.inv(information)))
    return dict(zip(map(label, keys), sd.tolist())), information


def separability(toy, size, device="cpu"):
    """Joint Fisher sd over the sd of the half each parameter belongs to (see the module's docstring)."""
    joint, _ = fisher_sd(toy, "joint", size, device)
    bragg, _ = fisher_sd(toy, "bragg", size, device)
    diffuse, _ = fisher_sd(toy, "diffuse", size, device)
    ratios = {}
    for name, sd in joint.items():
        alone = [half[name] for half in (bragg, diffuse) if name in half]
        ratios[name] = float(sd / min(alone))
    return ratios


def run_fit(name, size, poisson, seed, toy=TOYS["separable"], device="cpu"):
    fit = FITS[name]
    truth = truth_model(toy, name, size, device)
    with torch.no_grad():
        expected = image(truth)
    diffuse_mean = truth.components["diffuse"].mean() if fit["diffuse"] else None
    data = expected
    if poisson:
        data = torch.poisson(expected.cpu(), generator=torch.Generator().manual_seed(seed)).to(expected.device)
    keys = fit["refine"]
    information = fisher(truth, keys).cpu().numpy()

    values = toy.truth()
    model = build(toy, size, fit["diffuse"], fit["bragg"], device)
    for key in fit.get("known", []):
        put(model, key, values[key])
    for key in keys:
        model.refine(key[0])
    result = refine_lbfgsb(model, lambda: deviance(image(model), data), ftol=1e-15, gtol=1e-10, maxiter=1000)

    covariance = np.linalg.inv(information)
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)
    scaled = information / np.sqrt(np.outer(np.diag(information), np.diag(information)))
    labels = list(map(label, keys))
    errors = {lab: get(model, key) - values[key] for lab, key in zip(labels, keys)}
    photons = {"bragg_max": float(truth.components["bragg"].max()) if fit["diffuse"] and fit["bragg"]
               else float(expected.max()) if fit["bragg"] else None,
               "diffuse_mean": float(diffuse_mean) if diffuse_mean is not None else None}
    return {
        "fit": name, "size": size, "poisson": poisson, "seed": seed, "converged": bool(result.success),
        "iterations": int(result.nit), "message": str(result.message), "photons": photons,
        "errors": errors,
        "z": {lab: float(errors[lab] / s) for lab, s in zip(labels, sd)},
        "fisher_sd": dict(zip(labels, sd.tolist())),
        "fisher_condition": float(np.linalg.cond(information)),
        "fisher_condition_scaled": float(np.linalg.cond(scaled)),
        "correlations": {f"{labels[i]} ~ {labels[j]}": float(correlation[i, j])
                         for i in range(len(keys)) for j in range(i + 1, len(keys))},
    }


def evaluate(toy_name, size, poisson, seeds, device="cpu", noise_free_tolerance=1e-5):
    """The gate's verdict for one toy: separability ratios, every fit and seed, and pass/fail."""
    toy = TOYS[toy_name]
    ratios = separability(toy, size, device)
    fits = [run_fit(name, size, poisson, seed, toy, device) for seed in (seeds if poisson else [0]) for name in FITS]
    worst_ratio = float(max(ratios.values()))
    if poisson:
        worst_recovery = float(max(abs(z) for r in fits for z in r["z"].values()))
        recovered = bool(worst_recovery < Z_LIMIT)
    else:
        worst_recovery = float(max(abs(e) for r in fits for e in r["errors"].values()))
        recovered = bool(worst_recovery < noise_free_tolerance)
    separable = bool(worst_ratio <= SEPARABILITY_LIMIT)
    return {"toy": toy_name, **asdict(toy), "spot_sd": toy.spot_sd, "size": size, "poisson": poisson,
            "separability": ratios, "worst_ratio": worst_ratio, "separable": separable,
            "worst_recovery": worst_recovery, "recovered": recovered, "passed": separable and recovered,
            "fits": fits}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--poisson", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    reports = {name: evaluate(name, args.size, args.poisson, args.seeds, args.device) for name in TOYS}
    for name, report in reports.items():
        print(f"toy {name}: N={report['n_cells']} spot sd {report['spot_sd']:.3f}, halo sigma {report['sigma']}, "
              f"worst joint/alone sd {report['worst_ratio']:.2f} (limit {SEPARABILITY_LIMIT}), "
              f"worst {'|z|' if args.poisson else '|error|'} {report['worst_recovery']:.2e}")
        for r in report["fits"]:
            print(f"  {r['fit']:8s} seed {r['seed']} converged={r['converged']} scaled cond "
                  f"{r['fisher_condition_scaled']:.1e} photons {r['photons']}")
    gate = reports["separable"]["passed"]
    flagged = not reports["hidden"]["separable"]
    print(f"separable toy: {'PASS' if gate else 'FAIL'}; hidden toy "
          f"{'flagged as weakly identified (expected)' if flagged else 'NOT flagged: the metric does not discriminate'}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(reports, handle, indent=1)
    return 0 if gate and flagged else 1


if __name__ == "__main__":
    sys.exit(main())
