"""
V1 of docs/diffuse_joint_plan.md, section 5: toy recovery of Bragg and diffuse, alone and jointly.

A synthetic still from known G (spot_scale), misset, cell and GaussianHalo (amplitude, sigma) is
refined back from a start off in every parameter, with refine_lbfgsb on the Poisson likelihood
(as its deviance, the same minimum as utils.losses.poisson_nll):

- bragg:   Bragg-only image, model without a diffuse term; refines G, misset, cell;
- diffuse: diffuse-only image (F = 0), G known (with k_d frozen, G and the halo amplitude are one
           number there) and the geometry given (misset and cell at their known values, from indexing
           and the Bragg data, as a diffuse-only program has them); refines amplitude, sigma;
- joint:   the sum; refines all of them. The Bragg term fixes G, so the amplitude is free.

Each fit reports the errors (with --poisson, z = error / Fisher sd), the Fisher information's
condition number (raw, and of its correlation matrix) and the correlations, at the truth. The
Fisher matrix is F = J^T diag(1/mu) J, the Poisson information, with J by central differences.
The metric is spot_metric=HKL, the one the plan recommends for joint fits.

Separability, the gate: for each parameter of the joint fit, its Fisher sd in the joint fit over
its sd in the fit of the half it belongs to (G, misset and cell: bragg; the halo: diffuse, with
the geometry given). Near 1 (or below) the halves do not trade off; the gate demands every
ratio <= SEPARABILITY_LIMIT and, with --poisson, |z| < Z_LIMIT in every fit and seed.

Two toys (TOYS). "separable" is the gate: a halo 7.5 times as wide as the spot (N = 10), so that
it is not hidden under it. "hidden" is a documented negative case, a halo about as wide as the spot,
where the joint fit cannot tell halo from spot: the gate requires it to be flagged as weakly
identified (a ratio above the limit), so that the metric is shown to discriminate. The exit
code is 0 only if the separable toy passes and the hidden one is flagged (with --grid and k_d
frozen there is no diffuse parameter to flag, and only the separable toy counts).

    python scripts/diffuse/v1_toy_recovery.py                                   # 24^2, noise-free
    python scripts/diffuse/v1_toy_recovery.py --size 128 --poisson --seeds 1 2 3 --out v1.json

--grid [STEP] (plan D3) fits with the halo replaced by its DiffuseGrid, sampled from the true halo
at nodes STEP apart (default: grid_step_for, from the bias law below) on the mmm ASU, values
frozen. The halo's amplitude and sigma are then not parameters. With --grid-kd frozen (the
default, plan decision 3) k_d stays 1 and the grid's error can only move G and the geometry; with
--grid-kd free k_d (group diffuse_scale) stands in for the amplitude, starting at 0.7 as the
amplitude does. The data are still rendered with the exact halo, so the fits see the
grid's interpolation error as a model error: noise-free, they pass if every error is under
GRID_BIAS_LIMIT Fisher sds (the bias the grid adds to a z-score), and with --poisson by the same
|z| < Z_LIMIT and separability rule as the exact halo. The interpolant's gradient jumps at the grid's
cell faces, so at ftol 1e-15 L-BFGS-B often ends in its line search (ABNORMAL) rather than on ftol;
the errors it ends at follow the step as step^2, as the bias does.

    python scripts/diffuse/v1_toy_recovery.py --grid                            # 24^2, noise-free
    python scripts/diffuse/v1_toy_recovery.py --grid --size 64 --poisson --seeds 1 2 3
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
from torch import nn

from nanobrag_torch.config import (BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention,
                                   DetectorPivot, SpotMetric)
from nanobrag_torch.diffuse import DiffuseGrid, GaussianHalo, LorentzianHalo
from nanobrag_torch.models.experiment import ExperimentModel
from nanobrag_torch.utils.refinement import refine_lbfgsb

DTYPE = torch.float64
SEPARABILITY_LIMIT = 1.5
Z_LIMIT = 4.0
# --grid's step comes from a law, not a fit to the gate. Noise-free, the grid's worst bias over the fits,
# in Fisher sds, is z = K (step / sigma)^2 sqrt(P), P the diffuse photons of the image: the trilinear
# error goes as (step / sigma)^2, the parameter error it causes does not depend on the photons, and
# the sd falls as 1 / sqrt(P). Measured 2026-11-05 (24^2, the separable toy, P = 4835 and 4 x 4835;
# step 1/16 and 1/32): z doubles with 4x the photons and falls 3.9-4.1x per halved step. K is set
# from step 1/32 at P = 4835 (z = K * 0.7545):
#   k_d frozen (the default): z 0.014 (cell a, diffuse fit; G 0.004 in the joint one): K = 0.0186
#   k_d free: z 0.097, all of it in k_d (geometry <= 0.009): K = 0.129
# The step is the largest 1/(2n) (nodes on the half-integer kinks) whose predicted z is at most
# GRID_BIAS_TARGET, half of the pass limit GRID_BIAS_LIMIT, which was fixed before the step.
GRID_BIAS_K = {"frozen": 0.0186, "free": 0.129}
# With a core (D3b) the grid carries only the residual, whose scale is fixed (the sharp toy's smooth
# part), so the law is z = K_core step^2 sqrt(P). Measured 2026-11-05 on the sharp toy (24^2, P = 9174,
# cored grid, noise-free diffuse fit, the worst): z 0.075, 0.0165, 0.0046 at 1/2, 1/4, 1/8; K_core from
# 1/4. At 1e6 photons the rule gives 1/8. K_core depends on the toy (the Fisher sds the bias is measured
# in, not only the residual): the wide sharp toy (kappa 0.7, P = 2878) gives z 0.058 and 0.016 at 1/4
# and 1/8, K_core 0.0189, about 7x (at 1/2 it is not yet in the step^2 regime: 0.47). Each toy carries
# its own (Toy.core_bias_k); this is the narrow one's.
GRID_BIAS_K_CORE = 0.00276
GRID_BIAS_TARGET = 0.05
GRID_BIAS_LIMIT = 0.1  # noise-free with --grid: the grid's bias, in Fisher sds
GRID_D_MIN = 7.0  # Angstrom: the detector reaches 8.0 A; the margin covers the misset and cell offsets
MMM = np.array([np.diag(d) for d in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))])  # 222, Laue mmm
SCALE = ("beam_params.delta_log_spot_scale", None)
MISSET = [("crystal_params.delta_misset_raw", i) for i in range(3)]
CELL = [("crystal_params.delta_log_a", None), ("crystal_params.delta_log_b", None), ("crystal_params.delta_log_c", None)]
HALO = [("diffuse_params.model.log_amplitude", None), ("diffuse_params.model.log_sigma", None)]
K_D = ("diffuse_params.delta_log_scale", None)  # with --grid, in place of the halo's amplitude
GEOMETRY_TRUTH = {  # raw parameter values of the synthetic shot; the start is zero offsets
    SCALE: math.log(1.3),
    MISSET[0]: 0.004, MISSET[1]: -0.003, MISSET[2]: 0.002,
    CELL[0]: 0.002, CELL[1]: -0.0015, CELL[2]: 0.001,
}
FITS = {
    "bragg": dict(diffuse=False, bragg=True, refine=[SCALE, *MISSET, *CELL]),
    # diffuse alone means geometry given (decided 2026-11-06): misset and cell held at their known
    # values (from indexing and the Bragg data), as a diffuse-only program would; joint refines them
    "diffuse": dict(diffuse=True, bragg=False, refine=[*HALO], known=[SCALE, *MISSET, *CELL]),
    "joint": dict(diffuse=True, bragg=True, refine=[SCALE, *MISSET, *CELL, *HALO]),
}
GRID_FITS = {  # --grid: the halo is a frozen grid, its amplitude k_d
    "bragg": FITS["bragg"],
    "diffuse": dict(diffuse=True, bragg=False, refine=[K_D], known=[SCALE, *MISSET, *CELL]),
    "joint": dict(diffuse=True, bragg=True, refine=[SCALE, *MISSET, *CELL, K_D]),
}


GRID_FITS_KD_FROZEN = {  # --grid with k_d frozen at 1, plan decision 3's default: the grid's scale error has
    "bragg": FITS["bragg"],  # nowhere to go but G and the geometry. With the geometry given and nothing
    "joint": dict(diffuse=True, bragg=True, refine=[SCALE, *MISSET, *CELL]),  # diffuse to refine, there
}  # is no diffuse-alone fit


@dataclass(frozen=True)
class GridSpec:
    """--grid: the true halo on a DiffuseGrid, nodes ``step`` apart; k_d "frozen" (the default) or "free"."""

    step: float
    kd: str = "frozen"
    core: bool = False  # D3b: the halo is the grid's core (refinable), the grid holds the residual

    def __post_init__(self):
        if self.kd not in ("frozen", "free"):
            raise ValueError(f"kd must be 'frozen' or 'free', got {self.kd!r}")
        if self.core and self.kd == "free":
            raise ValueError("with the halo as the core its amplitude is refined, so k_d stays frozen")


def diffuse_photons(toy, size, device="cpu"):
    """The diffuse photons of the toy's diffuse-only image at the truth (with the shape factor)."""
    model = truth_model(toy, "diffuse", size, device)
    with torch.no_grad():
        model()
    return float(model.components["diffuse"].sum())


def predicted_grid_bias(step, sigma, photons, kd="frozen", core=False, core_k=GRID_BIAS_K_CORE):
    """The bias law, in Fisher sds: z = K (step / sigma)^2 sqrt(photons); with a core (the residual on
    the grid) z = K_core step^2 sqrt(photons), K_core the toy's."""
    if core:
        return core_k * step ** 2 * math.sqrt(photons)
    return GRID_BIAS_K[kd] * (step / sigma) ** 2 * math.sqrt(photons)


def grid_step_for(toy, size, kd="frozen", device="cpu", core=False):
    """The largest 1/(2n) whose predicted bias is at most GRID_BIAS_TARGET for this toy's photons."""
    photons = diffuse_photons(toy, size, device)
    if core:
        widest = math.sqrt(GRID_BIAS_TARGET / (toy.core_bias_k * math.sqrt(photons)))
        return 1.0 / (2 * math.ceil(1.0 / (2 * widest) - 1e-12))
    widest = toy.sigma * math.sqrt(GRID_BIAS_TARGET / (GRID_BIAS_K[kd] * math.sqrt(photons)))
    return 1.0 / (2 * math.ceil(1.0 / (2 * widest) - 1e-12))


CORE_HALO = [("diffuse_params.model.core.log_amplitude", None), ("diffuse_params.model.core.log_sigma", None)]
GRID_CORE_FITS = {name: {**spec, "refine": [CORE_HALO[HALO.index(k)] if k in HALO else k for k in spec["refine"]]}
                  for name, spec in FITS.items()}


def _for_toy(table, toy):
    """A Lorentzian toy's halo has log_kappa where the Gaussian's has log_sigma."""
    if toy is None or toy.kind == "gauss":
        return table
    return {name: {**spec, "refine": [(k[0].replace("log_sigma", "log_kappa"), k[1]) for k in spec["refine"]]}
            for name, spec in table.items()}


def fits(grid=None, toy=None):
    return _for_toy(_fits(grid), toy)


def _fits(grid=None):
    if grid is None:
        return FITS
    if grid.core:
        return GRID_CORE_FITS
    return GRID_FITS if grid.kd == "free" else GRID_FITS_KD_FROZEN


@dataclass(frozen=True)
class Toy:
    """One synthetic shot. The GAUSS-HKL spot is exp(-2 (N dh)^2 / 0.63) in intensity, so its sd
    in h is 0.397 / N; the halo's is sigma. amplitude is the halo's peak, e^2 per cell (F = 100
    gives the spot 1e4)."""

    n_cells: int = 10
    sigma: float = 0.3
    amplitude: float = 128.0
    fluence: float = 2e29  # the same at every size: the detector's extent is fixed, so its photons per solid angle are
    kind: str = "gauss"  # "lorentz" (D3b): LorentzianHalo(amplitude, kappa = sigma) plus Smooth(smooth)
    smooth: float = 0.0
    core_bias_k: float = GRID_BIAS_K_CORE  # the cored grid's bias law constant, measured per toy

    @property
    def spot_sd(self) -> float:
        return math.sqrt(0.63 / 4) / self.n_cells

    def truth(self) -> dict:
        kappa = {(k[0].replace("log_sigma", "log_kappa"), None): math.log(self.sigma) for k in (HALO[1], CORE_HALO[1])}
        return {**GEOMETRY_TRUTH, HALO[0]: math.log(self.amplitude), HALO[1]: math.log(self.sigma), K_D: 0.0,
                CORE_HALO[0]: math.log(self.amplitude), CORE_HALO[1]: math.log(self.sigma), **kappa}

    def halo(self, amplitude=None, sigma=None):
        """The exact diffuse model at (amplitude, sigma), the truth by default."""
        amplitude = self.amplitude if amplitude is None else amplitude
        sigma = self.sigma if sigma is None else sigma
        if self.kind == "gauss":
            return GaussianHalo(amplitude, sigma)
        return SharpHalo(amplitude, sigma, self.smooth)

    def core(self):
        """The core the cored grid refines: the halo itself, or the Lorentzian part of a sharp toy."""
        return GaussianHalo(self.amplitude, self.sigma) if self.kind == "gauss" else LorentzianHalo(self.amplitude, self.sigma)

    def start_halo(self) -> tuple:
        return 0.7 * self.amplitude, 1.3 * self.sigma


# Measured (2026-11-05, Fisher at the truth, 24^2 and 64^2 alike): the worst joint/alone sd ratio
# is set by halo sigma over spot sd. N = 5 (spot sd 0.079): sigma 0.05, 0.12, 0.3, 0.5 give 22, 14,
# 2.6, 2.3. N = 10 (0.040): 40, 2.2, 1.28, 1.23. The amplitude is scaled as sigma^-3 to keep the
# diffuse photons fixed; corr(amplitude, sigma) is about -0.9 in every joint fit.
class Smooth(nn.Module):
    """A smooth, mmm-symmetric part: amplitude (2 + cos(2 pi h / 3) cos(2 pi k / 4) cos(2 pi l / 5))."""

    def __init__(self, amplitude):
        super().__init__()
        self.amplitude = amplitude

    def forward(self, hkl):
        waves = [torch.cos(2 * math.pi * hkl[:, i] / p) for i, p in enumerate((3.0, 4.0, 5.0))]
        return self.amplitude * (2 + waves[0] * waves[1] * waves[2])


class SharpHalo(LorentzianHalo):
    """D3b's sharp toy: a Lorentzian core (kappa = 0.5 / N, eryx's k floor) plus a fixed smooth part."""

    def __init__(self, amplitude, kappa, smooth):
        super().__init__(amplitude, kappa)
        self.smooth = Smooth(smooth)

    def forward(self, hkl):
        return super().forward(hkl) + self.smooth(hkl)


TOYS = {
    # sigma / spot sd = 7.5: about 8 diffuse photons a pixel at 24^2 under spots of about 1e6
    "separable": Toy(n_cells=10, sigma=0.3, amplitude=128.0),
    # negative case, the first toy: sigma / spot sd = 1.5, the halo hides under the spots
    "hidden": Toy(n_cells=5, sigma=0.12, amplitude=2000.0),
}
# D3b (--sharp): a one-phonon-like core, kappa = 0.5 / N, with a smooth part the grid must carry
# and its separable variant. A Lorentzian's wings separate from the spots only well past them: measured
# 2026-11-05 (24^2, no spot mask, Fisher at the truth, amplitude scaled as kappa^-2), worst joint/alone
# sd 4.96, 3.17, 2.43, 1.82, 1.57, 1.45, 1.33 at kappa 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7 (2.5 to 17.6
# spot sds). kappa = 0.7 is the first with a margin under the limit.
EXTRA_TOYS = {"sharp": Toy(n_cells=10, sigma=0.05, amplitude=3000.0, kind="lorentz", smooth=10.0),
              "sharp_wide": Toy(n_cells=10, sigma=0.7, amplitude=3000.0 * (0.05 / 0.7) ** 2, kind="lorentz",
                                smooth=10.0, core_bias_k=0.0189)}


_GRIDS = {}


def toy_grid(toy, step, device="cpu"):
    """The true halo on a DiffuseGrid (mmm ASU, values frozen), built once per toy and step."""
    key = (toy.kind, toy.amplitude, toy.sigma, toy.smooth, float(step), str(device))
    if key not in _GRIDS:
        halo = toy.halo()
        _GRIDS[key] = DiffuseGrid.from_model(halo, step=step, d_min=GRID_D_MIN, cell=(30.0, 35.0, 40.0),
                                             point_group=MMM, refinable=False, device=torch.device(device))
    return _GRIDS[key]


def cored_grid(toy, step, device="cpu"):
    """A grid whose core is refinable, built as a real fit would be: the model and its core at the fit's
    starting parameters (0.7 x the amplitude, 1.3 x the width), so theta_b is the start and core_drift()
    counts from there. The residual is zero for a Gaussian toy (the machinery's check: it fits as the
    exact halo) and the smooth part for a sharp one, which does not depend on the core's parameters."""
    amplitude, sigma = toy.start_halo()
    core = toy.core().to(device)
    with torch.no_grad():
        core.log_amplitude.fill_(math.log(amplitude))
        (core.log_sigma if toy.kind == "gauss" else core.log_kappa).fill_(math.log(sigma))
    return DiffuseGrid.from_model(toy.halo(amplitude, sigma), step=step, d_min=GRID_D_MIN,
                                  cell=(30.0, 35.0, 40.0), point_group=MMM, refinable=False,
                                  device=torch.device(device), core=core)


def build(toy, size, diffuse, bragg, device="cpu", grid=None, oversample=1):
    n = toy.n_cells
    crystal = CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(n, n, n),
                            default_F=100.0 if bragg else 0.0, shape=CrystalShape.GAUSS, spot_metric=SpotMetric.HKL)
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=12.8 / size, spixels=size, fpixels=size, oversample=oversample,
                              detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)
    beam = BeamConfig(wavelength_A=1.2, fluence=toy.fluence)
    if not diffuse:
        halo = None
    elif grid is None:
        halo = toy.halo(*toy.start_halo())
    else:
        halo = toy_grid(toy, grid.step, device) if not grid.core else cored_grid(toy, grid.step, device)
    model = ExperimentModel(crystal, detector, beam, dtype=DTYPE, device=torch.device(device), diffuse=halo,
                            crystal_system="orthorhombic")
    if diffuse and grid is not None and grid.kd == "free":
        put(model, K_D, math.log(toy.start_halo()[0] / toy.amplitude))
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


def truth_model(toy, fit, size, device="cpu", grid=None, oversample=1):
    spec = fits(grid, toy)[fit]
    model = build(toy, size, spec["diffuse"], spec["bragg"], device, grid, oversample=oversample)
    truth = toy.truth()
    if grid is None:
        diffuse_keys = HALO
    else:
        diffuse_keys = CORE_HALO if grid.core else [K_D] if grid.kd == "free" else []
    if toy.kind != "gauss":
        diffuse_keys = [(k[0].replace("log_sigma", "log_kappa"), k[1]) for k in diffuse_keys]
    diffuse_keys = diffuse_keys if spec["diffuse"] else []
    for key in [SCALE, *MISSET, *CELL] + diffuse_keys:
        put(model, key, truth[key])
    return model


def fisher_sd(toy, fit, size, device="cpu", grid=None):
    """{label: Fisher sd} of a fit's parameters at the truth, and the information matrix."""
    keys = fits(grid, toy)[fit]["refine"]
    information = fisher(truth_model(toy, fit, size, device, grid), keys).cpu().numpy()
    sd = np.sqrt(np.diag(np.linalg.inv(information)))
    return dict(zip(map(label, keys), sd.tolist())), information


def separability(toy, size, device="cpu", grid=None):
    """Joint Fisher sd over the sd of the half each parameter belongs to (see the module's docstring)."""
    joint, _ = fisher_sd(toy, "joint", size, device, grid)
    bragg, _ = fisher_sd(toy, "bragg", size, device, grid)
    diffuse = fisher_sd(toy, "diffuse", size, device, grid)[0] if "diffuse" in fits(grid, toy) else {}
    ratios = {}
    for name, sd in joint.items():
        alone = [half[name] for half in (bragg, diffuse) if name in half]
        ratios[name] = float(sd / min(alone))
    return ratios


# Where a parameter's Fisher sd at the truth exceeds PROFILE_SD (log units) the likelihood is not
# quadratic over the range a fit can wander, and the Fisher z misdescribes it (measured: the 128^2
# sharp_wide fits before the fluence fix, log kappa sd 1.18, ran to z 223 along a valley). For such a
# parameter the gate uses the profile-likelihood z at the truth, and the report gives the 1-sd profile
# interval (2 dNLL = 1) beside the Fisher z.
PROFILE_SD = 0.5


def get_truth(toy, wanted):
    """The true raw value of the parameter labelled ``wanted``."""
    for key, value in toy.truth().items():
        if label(key) == wanted:
            return value
    raise KeyError(wanted)


def _refit(toy, size, name, device, grid, data, start, fixed=None):
    """The deviance minimum with every parameter of the fit refined from ``start`` except ``fixed``
    (key, value), held at its value: one point of a profile."""
    fit = fits(grid, toy)[name]
    values = toy.truth()
    model = build(toy, size, fit["diffuse"], fit["bragg"], device, grid)
    for key in fit.get("known", []):
        put(model, key, values[key])
    for key in fit["refine"]:
        put(model, key, start[key])
    if fixed is not None:
        put(model, fixed[0], fixed[1])
    for key in fit["refine"]:
        if fixed is None or key != fixed[0]:
            model.refine(key[0])
    refine_lbfgsb(model, lambda: deviance(image(model), data), ftol=1e-15, gtol=1e-10, maxiter=1000)
    with torch.no_grad():
        return float(deviance(image(model), data))


def profile(toy, size, name, device, grid, data, fitted, key, sd, nll_min):
    """The profile z at the truth and the 1-sd interval of one parameter (deviance is the NLL up to a
    constant, so 2 dNLL = 1 marks one sd). The interval is bracketed from the estimate in steps of the
    Fisher sd, doubling up to 16 sds, then bisected five times; an end not reached is +-inf."""
    estimate, truth = fitted[key], toy.truth()[key]
    at_truth = _refit(toy, size, name, device, grid, data, fitted, (key, truth))
    best = min(nll_min, at_truth)
    delta = max(at_truth - best, 0.0)
    z = math.copysign(math.sqrt(2 * delta), estimate - truth)
    interval = []
    for direction in (-1, 1):
        inside, width = 0.0, sd
        outside = None
        while width <= 16 * sd:
            if 2 * (_refit(toy, size, name, device, grid, data, fitted, (key, estimate + direction * width)) - best) >= 1:
                outside = width
                break
            inside, width = width, 2 * width
        if outside is None:
            interval.append(direction * math.inf)
            continue
        for _ in range(5):
            mid = 0.5 * (inside + outside)
            if 2 * (_refit(toy, size, name, device, grid, data, fitted, (key, estimate + direction * mid)) - best) >= 1:
                outside = mid
            else:
                inside = mid
        interval.append(estimate + direction * 0.5 * (inside + outside))
    return {"z": z, "delta_nll": delta, "interval": interval}


def run_fit(name, size, poisson, seed, toy=TOYS["separable"], device="cpu", grid=None):
    """One fit. The data come from the exact halo; with a GridSpec the model fitted (and its Fisher
    information) uses the halo's grid."""
    fit = fits(grid, toy)[name]
    exact = truth_model(toy, name, size, device)
    with torch.no_grad():
        expected = image(exact)
    photons = {"bragg_max": float(exact.components["bragg"].max()) if fit["diffuse"] and fit["bragg"]
               else float(expected.max()) if fit["bragg"] else None,
               "diffuse_mean": float(exact.components["diffuse"].mean()) if fit["diffuse"] else None}
    truth = exact if grid is None else truth_model(toy, name, size, device, grid)
    data = expected
    if poisson:
        data = torch.poisson(expected.cpu(), generator=torch.Generator().manual_seed(seed)).to(expected.device)
    keys = fit["refine"]
    information = fisher(truth, keys).cpu().numpy()

    values = toy.truth()
    model = build(toy, size, fit["diffuse"], fit["bragg"], device, grid)
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
    z = {lab: float(errors[lab] / s) for lab, s in zip(labels, sd)}
    profiles = {}
    if any(s > PROFILE_SD for s in sd):
        fitted = {key: get(model, key) for key in keys}
        with torch.no_grad():
            nll_min = float(deviance(image(model), data))
        for lab, key, s in zip(labels, keys, sd):
            if s > PROFILE_SD:
                profiles[lab] = profile(toy, size, name, device, grid, data, fitted, key, float(s), nll_min)
    return {
        "fit": name, "size": size, "poisson": poisson, "seed": seed, "grid": None if grid is None else asdict(grid),
        "converged": bool(result.success),
        "iterations": int(result.nit),
        # the cored grid's drift at the fit's end (data for the V2 drift tolerance)
        "core_drift": model.diffuse.core_drift() if fit["diffuse"] and grid is not None and grid.core else None, "message": str(result.message), "photons": photons,
        "errors": errors,
        "z": z,
        "profile": profiles,  # the parameters whose Fisher sd exceeds PROFILE_SD: z, delta_nll, interval
        "z_gate": {lab: profiles[lab]["z"] if lab in profiles else value for lab, value in z.items()},
        "fisher_sd": dict(zip(labels, sd.tolist())),
        "fisher_condition": float(np.linalg.cond(information)),
        "fisher_condition_scaled": float(np.linalg.cond(scaled)),
        "correlations": {f"{labels[i]} ~ {labels[j]}": float(correlation[i, j])
                         for i in range(len(keys)) for j in range(i + 1, len(keys))},
    }


def evaluate(toy_name, size, poisson, seeds, device="cpu", noise_free_tolerance=1e-5, grid=None):
    """The gate's verdict for one toy: separability ratios, every fit and seed, and pass/fail."""
    toy = {**TOYS, **EXTRA_TOYS}[toy_name]
    ratios = separability(toy, size, device, grid)
    runs = [run_fit(name, size, poisson, seed, toy, device, grid)
            for seed in (seeds if poisson else [0]) for name in fits(grid, toy)]
    worst_ratio = float(max(ratios.values()))
    if poisson:
        worst_recovery = float(max(abs(z) for r in runs for z in r["z_gate"].values()))
        recovered = bool(worst_recovery < Z_LIMIT)
    elif grid is not None and (not grid.core or toy.smooth > 0):  # the grid's bias (a nonzero residual), in sds
        worst_recovery = float(max(abs(z) for r in runs for z in r["z_gate"].values()))
        recovered = bool(worst_recovery < GRID_BIAS_LIMIT)
    else:
        worst_recovery = float(max(abs(e) for r in runs for e in r["errors"].values()))
        recovered = bool(worst_recovery < noise_free_tolerance)
    separable = bool(worst_ratio <= SEPARABILITY_LIMIT)
    return {"toy": toy_name, **asdict(toy), "spot_sd": toy.spot_sd, "size": size, "poisson": poisson,
            "grid": None if grid is None else asdict(grid),
            "separability": ratios, "worst_ratio": worst_ratio, "separable": separable,
            "worst_recovery": worst_recovery, "recovered": recovered, "passed": separable and recovered,
            "fits": runs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--poisson", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid", type=float, nargs="?", const=0.0, default=None, metavar="STEP",
                        help="fit with the halo's DiffuseGrid, nodes STEP apart (default: grid_step_for)")
    parser.add_argument("--grid-kd", choices=["frozen", "free"], default="frozen",
                        help="with --grid: k_d frozen at 1 (plan decision 3, the default) or refined")
    parser.add_argument("--grid-core", action="store_true",
                        help="with --grid: the machinery check of D3b: the Gaussian halo as the grid's refinable "
                             "core, with a zero residual")
    parser.add_argument("--sharp", action="store_true",
                        help="also the sharp toys (D3b): a Lorentzian core plus a smooth part on a cored grid, the "
                             "step from the residual's bias law; kappa = 0.5 / N (a flagged negative case) and "
                             "kappa = 0.7 (separable, gated by the usual rules)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    grid = None
    if args.grid is not None:  # one step for both toys, the separable one's: the hidden toy is judged on separability
        step = args.grid or grid_step_for(TOYS["separable"], args.size, args.grid_kd, args.device)
        grid = GridSpec(step, args.grid_kd, core=args.grid_core)
    reports = {name: evaluate(name, args.size, args.poisson, args.seeds, args.device, grid=grid)
               for name in TOYS}
    for name, report in reports.items():
        noise_free = "|error|" if args.grid is None else "|bias| / sd"
        print(f"toy {name}: N={report['n_cells']} spot sd {report['spot_sd']:.3f}, halo sigma {report['sigma']}, "
              f"worst joint/alone sd {report['worst_ratio']:.2f} (limit {SEPARABILITY_LIMIT}), "
              f"worst {'|z|' if args.poisson else noise_free} {report['worst_recovery']:.2e}"
              + ("" if grid is None else f" (grid step 1/{round(1 / grid.step)}, k_d {grid.kd})"))
        for r in report["fits"]:
            print(f"  {r['fit']:8s} seed {r['seed']} converged={r['converged']} scaled cond "
                  f"{r['fisher_condition_scaled']:.1e} photons {r['photons']}")
    gate = reports["separable"]["passed"]
    if args.sharp:
        for name in ("sharp", "sharp_wide"):
            toy = EXTRA_TOYS[name]
            spec = GridSpec(grid_step_for(toy, args.size, core=True, device=args.device), core=True)
            report = evaluate(name, args.size, args.poisson, args.seeds, args.device, grid=spec)
            reports[name] = report
            drifts = [r["core_drift"] for r in report["fits"] if r["core_drift"] is not None]
            if name == "sharp":
                # kappa = 0.5 / N is 1.25 spot sds: a negative case, as the hidden toy is. It must be
                # flagged (ratio about 10), and noise-free the grid's bias is gated; with --poisson its z
                # is reported, not gated, as the exact halo is weakly identified there too (measured
                # 2026-11-05, 24^2, seed 3, joint: z 15.7 exact, 34 on the grid, both ABNORMAL).
                ok = (not report["separable"]) and (args.poisson or report["recovered"])
                rule = "flagged as weakly identified (expected)" + ("" if args.poisson else ", bias gated")
            else:  # separable: the usual rules, separability and recovery (bias, or |z| with --poisson)
                ok = report["passed"]
                rule = "separability and recovery gated"
            print(f"toy {name}: kappa {toy.sigma} ({toy.sigma / toy.spot_sd:.1f} spot sds), cored grid step "
                  f"1/{round(1 / spec.step)}, worst joint/alone sd {report['worst_ratio']:.2f}, worst "
                  f"{'|z|' if args.poisson else '|bias| / sd'} {report['worst_recovery']:.2e}, core drift at the "
                  f"end {min(drifts):.3f}-{max(drifts):.3f}; {rule}: {'PASS' if ok else 'FAIL'}")
            gate = gate and ok
    flagged = not reports["hidden"]["separable"]
    # With the grid's shape and k_d both frozen the diffuse term has no parameter of its own, so
    # nothing can trade off with the spots: the hidden toy cannot be flagged and is not required to be.
    needs_flag = grid is None or grid.kd == "free" or grid.core
    if needs_flag:
        verdict = 'flagged as weakly identified (expected)' if flagged else 'NOT flagged: the metric does not discriminate'
    else:
        verdict = "not applicable (k_d frozen: no diffuse parameter to trade with the spots)"
    print(f"separable toy: {'PASS' if gate else 'FAIL'}; hidden toy {verdict}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(reports, handle, indent=1)
    return 0 if gate and (flagged or not needs_flag) else 1


if __name__ == "__main__":
    sys.exit(main())
