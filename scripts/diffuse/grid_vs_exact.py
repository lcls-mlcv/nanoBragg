"""
Grid vs exact (docs/diffuse_joint_plan.md, sections 2.2 and 7, D3): how far a DiffuseGrid's image is from
the exact per-pixel model's, as a function of the grid step over the halo's width, against the
Poisson noise of the diffuse term at the V1 photon level; and the memory of a lysozyme-sized grid.

"error" renders the V1 toy's diffuse-only shot (scripts/diffuse/v1_toy_recovery.py: GAUSS-HKL,
N = 10, 30 x 35 x 40 A, the V1 fluence) with GaussianHalo(amplitude, sigma), amplitude scaled as
sigma^-3 from V1's 128 at 0.3 so that the diffuse photons stay about fixed, and with the halo's
DiffuseGrid (mmm ASU) at each step. Per case:

- max_rel, rms_rel: max |grid - exact| / max exact, and rms(grid - exact) / rms(exact);
- max_z: max |grid - exact| / sqrt(max(exact, grid)), the worst pixel's error in Poisson sds;
- deviance: 2 sum (grid - exact + exact log(exact / grid)), the Poisson deviance of the grid's
  image against data at the exact one's expectation (twice the expected log-likelihood loss), the
  whole image's detectability: well under 1, the grid error cannot be told from the noise. It
  counts a pixel the grid lights where the exact image is dark as its photons; chi2 =
  sum (grid - exact)^2 / exact, its small-error limit, divides those by about zero;
- deviance_scaled, best_scale: the deviance after the best-fit scale c of the grid image, the part
  of the error a free k_d cannot absorb (c = sum exact / sum grid);
- photons: mean diffuse photons per pixel.

"bench" times 2^20 evaluations of the lysozyme grid (step 1/4), ASU and P1, on the CPU in float64
and, if available, on MPS in float32.

A smooth model (cosines, periods 3, 4 and 5 in h, k, l) is the control.

"memory" builds the lysozyme grid (79 x 79 x 38 A, d_min 2 A, P4_3 2_1 2: Laue 4/mmm) at a step, ASU
and P1, each in a fresh process, and reports nodes, stored values, bytes held and the peak RSS.

    python scripts/diffuse/grid_vs_exact.py error --size 64 --out grid_vs_exact.json
    python scripts/diffuse/grid_vs_exact.py memory --step 0.25
"""
import argparse
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import numpy as np
import torch
from torch import nn

from nanobrag_torch.diffuse import DiffuseGrid, GaussianHalo

_spec = importlib.util.spec_from_file_location("v1_toy_recovery", Path(__file__).with_name("v1_toy_recovery.py"))
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

SIGMAS = (0.05, 0.1, 0.2, 0.3)
STEPS = (1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32)
MAX_NODES = 40_000_000  # the box at 1/32 to 7 A is 33M nodes
FOUR_FOLD = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
TWO_FOLD_A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


def point_group_422():
    ops = [np.eye(3, dtype=np.int64)]
    for _ in range(3):
        ops.append(ops[-1] @ FOUR_FOLD)
    return np.array(ops + [R @ TWO_FOLD_A for R in ops])


class Smooth(nn.Module):
    def __init__(self, amplitude):
        super().__init__()
        self.amplitude = amplitude

    def forward(self, hkl):
        waves = [torch.cos(2 * math.pi * hkl[:, i] / p) for i, p in enumerate((3.0, 4.0, 5.0))]
        return self.amplitude * (2 + waves[0] * waves[1] * waves[2])


def diffuse_image(model, size, toy):
    shot = v1.build(toy, size, diffuse=True, bragg=False)
    shot.diffuse_params.model = model  # the block's model, swapped: everything else as V1
    with torch.no_grad():
        shot()
    return shot.components["diffuse"]


def compare(exact, grid):
    delta = grid - exact
    lit = torch.maximum(exact, grid) > 0
    positive = exact > 0
    log_term = torch.where(positive, exact * torch.log(torch.where(positive, exact, 1.0) / grid.clamp(min=1e-300)), 0.0)
    return {"max_rel": float(delta.abs().max() / exact.max()),
            "rms_rel": float(delta.pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()),
            "max_z": float((delta.abs()[lit] / torch.maximum(exact, grid)[lit].sqrt()).max()),
            "deviance": float(2 * (delta + log_term).sum()),
            **_scaled_deviance(exact, grid, positive),
            "photons": float(exact.mean()), "photons_total": float(exact.sum())}


def _scaled_deviance(exact, grid, positive):
    """The deviance after the best-fit scale c of the grid image (what a free k_d absorbs):
    dD/dc = 0 gives c = sum(exact) / sum(grid)."""
    scale = exact.sum() / grid.sum()
    fitted = scale * grid
    log_term = torch.where(positive, exact * torch.log(torch.where(positive, exact, 1.0) / fitted.clamp(min=1e-300)), 0.0)
    return {"best_scale": float(scale), "deviance_scaled": float(2 * (fitted - exact + log_term).sum())}


def error_table(size):
    toy = v1.TOYS["separable"]
    rows = []
    cases = [("halo", sigma, GaussianHalo(toy.amplitude * (toy.sigma / sigma) ** 3, sigma)) for sigma in SIGMAS]
    cases.append(("smooth", None, Smooth(20.0)))
    for kind, sigma, model in cases:
        exact = diffuse_image(model, size, toy)
        for step in STEPS:
            half = np.ceil(np.array([30.0, 35.0, 40.0]) / v1.GRID_D_MIN / step) + 1
            if np.prod(2 * half + 1) > MAX_NODES:
                continue
            start = time.perf_counter()
            grid = DiffuseGrid.from_model(model, step=step, d_min=v1.GRID_D_MIN, cell=(30.0, 35.0, 40.0),
                                          point_group=v1.MMM, refinable=False)
            built = time.perf_counter() - start
            row = {"model": kind, "sigma": sigma, "step": step, "step_over_sigma": step / sigma if sigma else None,
                   "values": grid.values.numel(), "build_s": built, **compare(exact, diffuse_image(grid, size, toy))}
            rows.append(row)
            print(f"{kind:6s} sigma {sigma!s:5s} step 1/{round(1 / step):<3d} step/sigma "
                  f"{row['step_over_sigma'] or float('nan'):5.2f}  max_rel {row['max_rel']:.2e} "
                  f"rms_rel {row['rms_rel']:.2e}  max_z {row['max_z']:.2e}  deviance {row['deviance']:.2e} "
                  f"(scaled {row['deviance_scaled']:.2e}, c {row['best_scale']:.4f})  "
                  f"photons/px {row['photons']:.3g}", flush=True)
    return rows


def memory_one(step, symmetric):
    """Build the lysozyme grid in this process; RSS is the process's peak, so run one per process."""
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    grid = DiffuseGrid.from_model(GaussianHalo(100.0, 0.1), step=step, d_min=2.0, cell=(79.0, 79.0, 38.0),
                                  point_group=point_group_422() if symmetric else None, refinable=False)
    built = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    hkl = torch.rand(1 << 16, 3, dtype=torch.float64) * 20 - 10
    start = time.perf_counter()
    grid(hkl)
    lookup = (time.perf_counter() - start) / hkl.shape[0]
    return {"step": step, "storage": "ASU 4/mmm" if symmetric else "P1", "nodes": int(np.prod(grid.shape)),
            "shape": grid.shape, "values": grid.values.numel(), "held_MB": grid.nbytes() / 2 ** 20,
            "peak_rss_MB": peak * scale / 2 ** 20, "rss_before_MB": rss_before * scale / 2 ** 20,
            "build_s": built, "lookup_us_per_point": lookup * 1e6}


def bench(step, n=1 << 20):
    rows = []
    hkl = (torch.rand(n, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 2 - 1) * torch.tensor(
        [39.0, 39.0, 18.5], dtype=torch.float64)
    devices = [("cpu", torch.float64)] + ([("mps", torch.float32)] if torch.backends.mps.is_available() else [])
    for storage in ("asu", "p1"):
        grid = DiffuseGrid.from_model(GaussianHalo(100.0, 0.1), step=step, d_min=2.0, cell=(79.0, 79.0, 38.0),
                                      point_group=point_group_422() if storage == "asu" else None, refinable=False)
        for device, dtype in devices:
            g = grid.to(device=device, dtype=dtype)
            q = hkl.to(dtype=dtype).to(device)
            g(q[:1024])  # warm up
            times = []
            for _ in range(3):
                if device == "mps":
                    torch.mps.synchronize()
                start = time.perf_counter()
                g(q)
                if device == "mps":
                    torch.mps.synchronize()
                times.append(time.perf_counter() - start)
            rows.append({"storage": storage, "device": device, "dtype": str(dtype), "n": n, "best_s": min(times),
                         "ns_per_point": 1e9 * min(times) / n})
            print(rows[-1], flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    error = sub.add_parser("error")
    error.add_argument("--size", type=int, default=64)
    error.add_argument("--out")
    timing = sub.add_parser("bench")
    timing.add_argument("--step", type=float, default=0.25)
    timing.add_argument("--out")
    memory = sub.add_parser("memory")
    memory.add_argument("--step", type=float, nargs="+", default=[0.25])
    memory.add_argument("--p1", action="store_true", help="also P1 (one value per node)")
    memory.add_argument("--one", choices=["asu", "p1"], help=argparse.SUPPRESS)
    memory.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "error":
        report = error_table(args.size)
    elif args.command == "bench":
        report = bench(args.step)
    elif args.one:
        print(json.dumps(memory_one(args.step[0], args.one == "asu")))
        return 0
    else:
        report = []
        for step in args.step:
            for storage in ["asu"] + (["p1"] if args.p1 else []):
                out = subprocess.run([sys.executable, __file__, "memory", "--step", str(step), "--one", storage],
                                     check=True, capture_output=True, text=True).stdout
                report.append(json.loads(out.strip().splitlines()[-1]))
                print(report[-1], flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
