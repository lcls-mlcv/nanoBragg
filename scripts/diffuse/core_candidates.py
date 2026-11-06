"""
D3b, measure before building (docs/diffuse_joint_plan.md, section 2.2): how fine must a grid be near
integer hkl for a sharp halo core, and which refinement meets the bias law's target at affordable
memory.

The toy stands in for a one-phonon halo with a k floor: about every lattice point h0,

    I(h) = A(h0) kappa^2 / (d^T M d + kappa^2) + B (2 + cos(2 pi h / 3) cos(2 pi k / 4) cos(2 pi l / 5)),

d = h - h0, kappa = 0.5 / N (eryx's floor, N = 10 here), M = diag(1, 1.3, 0.8) (anisotropic), and
A(h0) = A0 (1 + 0.3 cos(2 pi h0 / 5) cos(2 pi k0 / 6) cos(2 pi l0 / 7)). It is even in each index, so
every candidate is stored on the positive octant (mmm, the V1 toy's Laue group) and read at |hkl|.

Candidates, prototypes on one nonuniform tensor-product trilinear interpolator:

- uniform: nodes step apart (DiffuseGrid's scheme), the baseline;
- warped: per axis, nodes graded towards the integers, d_j = sign(t) w (exp(c |t|) - 1) for t uniform
  on [-1/2, 1/2] with n nodes a unit, c set by d(1/2) = 1/2: spacing about w c / n at the lattice
  point, growing geometrically; a tensor grid, so the refinement fills planes and lines too;
- local: the uniform grid plus, about every lattice point, a fine cube |d_i| <= r at step s_f; a
  query inside a cube reads it (the two interpolants differ by the coarse error at the cube's faces);
- core: an analytic core A(h0) kappa^2 / (d^T M' d + kappa^2), supplied by the model with A(h0) a
  per-lattice-point table, plus a uniform grid of the residual I - core. M' = M is the core the model
  knows; M' = 1 (isotropic) a core of the wrong shape, whose residual is still sharp.

For each, at the V1 toy's geometry (128^2, GAUSS-HKL, k_d frozen): the linearised bias of the fit's
parameters, b = -F^-1 J^T W (mu_candidate - mu_exact), F = J^T W J (Poisson, J by central differences
at the truth), in Fisher sds and scaled to 1e6 diffuse photons (z grows as sqrt(P)); the lysozyme
memory (79 x 79 x 38 A to 2 A, Laue 4/mmm: the box over 16, the lattice points of the sphere over 16);
and the lookup time of 2^20 points.

    python scripts/diffuse/core_candidates.py --out core.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import numpy as np
import torch
from torch import nn

_spec = importlib.util.spec_from_file_location("v1_toy_recovery", Path(__file__).with_name("v1_toy_recovery.py"))
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

DTYPE = torch.float64
KAPPA = 0.05
METRIC = (1.0, 1.3, 0.8)
TARGET_PHOTONS = 1e6
TARGET_Z = v1.GRID_BIAS_TARGET
EXTENT = (30.0 / v1.GRID_D_MIN, 35.0 / v1.GRID_D_MIN, 40.0 / v1.GRID_D_MIN)
LYSOZYME = dict(cell=(79.0, 79.0, 38.0), d_min=2.0, ops=16)


def amplitude(h0, a0):
    return a0 * (1 + 0.3 * torch.cos(2 * math.pi * h0[..., 0] / 5) * torch.cos(2 * math.pi * h0[..., 1] / 6)
                 * torch.cos(2 * math.pi * h0[..., 2] / 7))


def lorentz(d, metric, kappa=KAPPA):
    m = torch.as_tensor(metric, dtype=d.dtype, device=d.device)
    return kappa ** 2 / ((d * d * m).sum(-1) + kappa ** 2)


class SharpCore(nn.Module):
    def __init__(self, a0=1000.0, b=5.0):
        super().__init__()
        self.a0, self.b = a0, b

    def core(self, hkl, metric=METRIC):
        h0 = torch.round(hkl)
        return amplitude(h0, self.a0) * lorentz(hkl - h0, metric)

    def smooth(self, hkl):
        waves = [torch.cos(2 * math.pi * hkl[..., i] / p) for i, p in enumerate((3.0, 4.0, 5.0))]
        return self.b * (2 + waves[0] * waves[1] * waves[2])

    def forward(self, hkl):
        return self.core(hkl) + self.smooth(hkl)


class TensorGrid(nn.Module):
    """Trilinear on per-axis node lists (sorted, from 0), read at |hkl|; zero beyond the last node."""

    def __init__(self, axes, fn, chunk=1 << 18):
        super().__init__()
        self.axes = [torch.as_tensor(a, dtype=DTYPE) for a in axes]
        nx, ny, nz = (len(a) for a in self.axes)
        values = torch.empty(nx * ny * nz, dtype=DTYPE)
        with torch.no_grad():
            for start in range(0, values.numel(), chunk):
                flat = torch.arange(start, min(start + chunk, values.numel()))
                ijk = (flat // (ny * nz), (flat // nz) % ny, flat % nz)
                values[start:start + flat.numel()] = fn(torch.stack([a[i] for a, i in zip(self.axes, ijk)], dim=-1))
        self.values = values.reshape(nx, ny, nz)

    def forward(self, hkl):
        x = hkl.abs()
        index, weights = [], []
        for i, axis in enumerate(self.axes):
            j = torch.searchsorted(axis, x[:, i].detach(), right=True) - 1
            j = j.clamp(0, len(axis) - 2)
            lo, hi = axis[j], axis[j + 1]
            f = (x[:, i] - lo) / (hi - lo)
            inside = (x[:, i] <= axis[-1])
            index.append(j)
            weights.append((f, inside))
        out = torch.zeros(hkl.shape[0], dtype=hkl.dtype)
        valid = weights[0][1] & weights[1][1] & weights[2][1]
        for c in range(8):
            o = [(c >> 2) & 1, (c >> 1) & 1, c & 1]
            w = torch.ones_like(out)
            for i in range(3):
                f = weights[i][0]
                w = w * (f if o[i] else 1 - f)
            out = out + w * self.values[index[0] + o[0], index[1] + o[1], index[2] + o[2]]
        return torch.where(valid, out, torch.zeros_like(out))


def uniform_axes(step, extent=EXTENT):
    return [torch.arange(0, math.ceil(e / step) + 2) * step for e in extent]


def warped_unit(n, w):
    """Nodes on [0, 1/2] of one half unit: t uniform, d = w (exp(c t) - 1), d(1/2) = 1/2."""
    c = 2 * math.log(1 + 0.5 / w)
    t = torch.linspace(0, 0.5, n // 2 + 1, dtype=DTYPE)
    return w * (torch.exp(c * t) - 1)


def warped_axes(n, w, extent=EXTENT):
    half = warped_unit(n, w)
    unit = torch.cat([half, 1 - half.flip(0)[1:]])  # nodes on [0, 1], dense at both integers
    axes = []
    for e in extent:
        cells = math.ceil(e) + 1
        axes.append(torch.unique(torch.cat([k + unit for k in range(cells)])))
    return axes


class LocalRefined(nn.Module):
    def __init__(self, model, coarse_step, radius, fine_step, extent=EXTENT):
        super().__init__()
        self.coarse = TensorGrid(uniform_axes(coarse_step, extent), model)
        self.radius, self.fine_step = radius, fine_step
        self.n = int(round(2 * radius / fine_step)) + 1
        self.points = [math.ceil(e) + 1 for e in extent]
        d = torch.linspace(-radius, radius, self.n, dtype=DTYPE)
        local = torch.stack(torch.meshgrid(d, d, d, indexing="ij"), dim=-1).reshape(-1, 3)
        h0 = torch.stack(torch.meshgrid(*[torch.arange(p, dtype=DTYPE) for p in self.points], indexing="ij"),
                         dim=-1).reshape(-1, 3)
        with torch.no_grad():
            self.fine = torch.stack([model(p + local) for p in h0]).reshape(*self.points, self.n, self.n, self.n)

    def forward(self, hkl):
        x = hkl.abs()
        h0 = torch.round(x.detach())
        d = x - h0
        inside = (d.abs() <= self.radius).all(-1) & (h0 < torch.tensor(self.points, dtype=DTYPE)).all(-1)
        u = (d + self.radius) / self.fine_step
        j = torch.floor(u.detach()).long().clamp(0, self.n - 2)
        f = u - j
        p = h0.long().clamp(min=0)
        p = torch.minimum(p, torch.tensor(self.points) - 1)
        out = torch.zeros(hkl.shape[0], dtype=hkl.dtype)
        for c in range(8):
            o = [(c >> 2) & 1, (c >> 1) & 1, c & 1]
            w = torch.ones_like(out)
            for i in range(3):
                w = w * (f[:, i] if o[i] else 1 - f[:, i])
            out = out + w * self.fine[p[:, 0], p[:, 1], p[:, 2], j[:, 0] + o[0], j[:, 1] + o[1], j[:, 2] + o[2]]
        return torch.where(inside, out, self.coarse(hkl))


class CorePlusResidual(nn.Module):
    def __init__(self, model, step, metric, kappa=KAPPA):
        super().__init__()
        self.model, self.metric, self.kappa = model, metric, kappa
        self.residual = TensorGrid(uniform_axes(step), lambda h: model(h) - self.core(h))

    def core(self, hkl):
        h0 = torch.round(hkl)
        return amplitude(h0, self.model.a0) * lorentz(hkl - h0, self.metric, self.kappa)

    def forward(self, hkl):
        return self.core(hkl) + self.residual(hkl)


def linear_bias(exact, candidate, size, kd):
    """z of every refined parameter, scaled to TARGET_PHOTONS diffuse photons: diffuse and joint fits."""
    toy = v1.TOYS["separable"]
    out = {}
    for fit in ("diffuse", "joint"):
        shot = v1.build(toy, size, diffuse=True, bragg=fit == "joint")
        truth = toy.truth()
        for key in [v1.SCALE, *v1.MISSET, *v1.CELL]:
            v1.put(shot, key, truth[key])
        keys = [*v1.MISSET, *v1.CELL] + ([v1.SCALE] if fit == "joint" else []) + ([v1.K_D] if kd == "free" else [])
        shot.diffuse_params.model = exact
        with torch.no_grad():
            mu = v1.image(shot).reshape(-1)
            photons = float(shot.components["diffuse"].sum())
            columns = []
            for key in keys:
                start = v1.get(shot, key)
                v1.put(shot, key, start + 1e-6)
                plus = v1.image(shot).reshape(-1)
                v1.put(shot, key, start - 1e-6)
                minus = v1.image(shot).reshape(-1)
                v1.put(shot, key, start)
                columns.append((plus - minus) / 2e-6)
            shot.diffuse_params.model = candidate
            mu_c = v1.image(shot).reshape(-1)
        J = torch.stack(columns, 1)
        keep = mu > 0
        W = 1 / mu[keep]
        F = (J[keep] * W[:, None]).T @ J[keep]
        cov = torch.linalg.inv(F)
        b = -cov @ (J[keep] * W[:, None]).T @ (mu_c - mu)[keep]
        z = b / cov.diagonal().sqrt() * math.sqrt(TARGET_PHOTONS / photons)
        out[fit] = {v1.label(k).split(".")[-1]: float(v) for k, v in zip(keys, z)}
    return out


def lysozyme_memory(kind, **p):
    """Values stored for lysozyme on the 4/mmm ASU (8 bytes each), analytically."""
    a = np.array(LYSOZYME["cell"]) / LYSOZYME["d_min"]
    ops = LYSOZYME["ops"]
    points = 4 / 3 * math.pi * np.prod(a) / ops  # lattice points in the resolution sphere, ASU
    if kind in ("uniform", "core"):
        n = np.prod(2 * np.ceil(a / p["step"]) + 3) / ops
        if kind == "core":
            n += points  # the amplitude table
    elif kind == "warped":
        n = np.prod(2 * (np.ceil(a) + 1) * p["n"] + 1) / ops
    elif kind == "local":
        n = np.prod(2 * np.ceil(a / p["coarse_step"]) + 3) / ops + points * (2 * round(p["radius"] / p["fine_step"]) + 1) ** 3
    return float(n * 8 / 2 ** 20)


def lookup_ns(module, n=1 << 20):
    hkl = (torch.rand(n, 3, dtype=DTYPE, generator=torch.Generator().manual_seed(0)) * 2 - 1) * 4
    module(hkl[:1024])
    start = time.perf_counter()
    with torch.no_grad():
        module(hkl)
    return 1e9 * (time.perf_counter() - start) / n


CANDIDATES = [
    ("uniform", dict(step=1 / 8)), ("uniform", dict(step=1 / 16)), ("uniform", dict(step=1 / 32)),
    ("uniform", dict(step=1 / 64)),
    ("warped", dict(n=16, w=0.02)), ("warped", dict(n=24, w=0.02)), ("warped", dict(n=32, w=0.02)),
    ("warped", dict(n=24, w=0.01)), ("warped", dict(n=32, w=0.05)),
    ("local", dict(coarse_step=1 / 16, radius=0.25, fine_step=1 / 64)),
    ("local", dict(coarse_step=1 / 16, radius=0.125, fine_step=1 / 128)),
    ("local", dict(coarse_step=1 / 32, radius=0.125, fine_step=1 / 128)),
    ("core", dict(step=1 / 8, metric=METRIC)), ("core", dict(step=1 / 16, metric=METRIC)),
    ("core", dict(step=1 / 16, metric=(1.0, 1.0, 1.0))), ("core", dict(step=1 / 32, metric=(1.0, 1.0, 1.0))),
    # a core shape wrong by about 5% and 10% in M (the model's core approximate)
    ("core", dict(step=1 / 8, metric=(1.0, 1.25, 0.84))), ("core", dict(step=1 / 16, metric=(1.0, 1.25, 0.84))),
    ("core", dict(step=1 / 16, metric=(1.0, 1.18, 0.88))), ("core", dict(step=1 / 8, metric=(1.0, 1.3, 0.8), kappa=0.055)),
]


def build(kind, model, p):
    if kind == "uniform":
        return TensorGrid(uniform_axes(p["step"]), model)
    if kind == "warped":
        return TensorGrid(warped_axes(p["n"], p["w"]), model)
    if kind == "local":
        return LocalRefined(model, p["coarse_step"], p["radius"], p["fine_step"])
    return CorePlusResidual(model, p["step"], p["metric"], p.get("kappa", KAPPA))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--kd", choices=["frozen", "free"], default="frozen")
    parser.add_argument("--only", nargs="*", help="kinds to run")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    model = SharpCore()
    rows = []
    for kind, p in CANDIDATES:
        if args.only and kind not in args.only:
            continue
        start = time.perf_counter()
        candidate = build(kind, model, p)
        built = time.perf_counter() - start
        bias = linear_bias(model, candidate, args.size, args.kd)
        worst = max(abs(z) for fit in bias.values() for z in fit.values())
        row = {"kind": kind, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}, "worst_z": worst,
               "bias": bias, "lysozyme_MB": lysozyme_memory(kind, **p), "lookup_ns": lookup_ns(candidate),
               "build_s": built}
        rows.append(row)
        where = max(((fit, k) for fit in bias for k in bias[fit]), key=lambda t: abs(bias[t[0]][t[1]]))
        print(f"{kind:8s} {p}  worst z at 1e6 photons {worst:.3f} ({where[0]} {where[1]})  "
              f"lysozyme {row['lysozyme_MB']:.0f} MB  lookup {row['lookup_ns']:.0f} ns", flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
