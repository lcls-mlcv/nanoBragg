"""How refinement of one shared Fhkl scales with the number of shots (SerialExperiment).

Synthetic stills, nothing measured: a tetragonal crystal with lysozyme's cell (P4_3 2_1 2 point
group 422), Wilson-distributed amplitudes, random orientations, ROIs around each shot's strongest
spots and Poisson noise. The start has every log F off by --f-noise and every shot's scale off by
--scale-noise; refine_lbfgsb then fits Fhkl and the per-shot scales (shot 0's held: the overall
scale is degenerate with F), one objective term per shot so each graph is freed before the next.

Reports, per run: evaluations, seconds per evaluation, peak GPU memory, and the agreement of the
fitted F with the truth on the reflections the ROIs see (CC of log F and an R factor, at the start
and at the end).

    NANOBRAGG_DISABLE_COMPILE=1 PYTHONPATH=src python scripts/benchmarks/serial_scaling.py \
        --shots 100 --device cuda --out serial_100.json
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as nnf

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot, SpotMetric,
)
from nanobrag_torch.models.experiment import ExperimentModel, SerialExperiment
from nanobrag_torch.utils.losses import gaussian_nll
from nanobrag_torch.utils.refinement import git_commit, refine_lbfgsb
from nanobrag_torch.utils.structure_factors import StructureFactors

CELL = (79.1, 79.1, 37.9)  # tetragonal lysozyme
DT = torch.float64


def point_group_422() -> np.ndarray:
    four = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    two = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    group = {tuple(np.eye(3, dtype=int).ravel())}
    while True:
        grown = group | {tuple((np.array(g).reshape(3, 3) @ m).ravel()) for g in group for m in (four, two)}
        if grown == group:
            return np.array([np.array(g).reshape(3, 3) for g in sorted(group)])
        group = grown


def unique_reflections(d_min: float, rotations: np.ndarray) -> np.ndarray:
    """One representative per orbit under the point group and Friedel, inside d_min."""
    a, b, c = CELL
    n = [int(x / d_min) + 1 for x in CELL]
    seen, keep = set(), []
    for h in range(-n[0], n[0] + 1):
        for k in range(-n[1], n[1] + 1):
            for l in range(-n[2], n[2] + 1):  # noqa: E741
                if (h, k, l) == (0, 0, 0) or (h / a) ** 2 + (k / b) ** 2 + (l / c) ** 2 > 1 / d_min ** 2:
                    continue
                orbit = {tuple(s * (r @ (h, k, l))) for r in rotations for s in (1, -1)}
                key = max(orbit)
                if key not in seen:
                    seen.add(key)
                    keep.append(key)
    return np.array(keep)


def detector(args) -> DetectorConfig:
    return DetectorConfig(distance_mm=args.distance, pixel_size_mm=0.172, spixels=args.pixels, fpixels=args.pixels,
                          oversample=1, detector_convention=DetectorConvention.MOSFLM,
                          detector_pivot=DetectorPivot.BEAM)


def shot_model(args, misset, sf, rois=None, spot_scale=1.0) -> ExperimentModel:
    crystal = CrystalConfig(cell_a=CELL[0], cell_b=CELL[1], cell_c=CELL[2], misset_deg=tuple(misset),
                            N_cells=(args.ncells,) * 3, default_F=0.0, shape=CrystalShape.GAUSS,
                            spot_metric=SpotMetric.HKL)
    beam = BeamConfig(wavelength_A=args.wavelength, fluence=1e24, spot_scale=spot_scale)
    return ExperimentModel(crystal, detector(args), beam, dtype=DT, device=args.device, structure_factors=sf,
                           background_rois=rois)


def strongest_spots(image: torch.Tensor, count: int, half: int) -> list:
    """Boxes (x1, x2, y1, y2) around the brightest local maxima, no two overlapping."""
    peaks = (image == nnf.max_pool2d(image[None, None], 2 * half + 1, 1, half)[0, 0]) & (image > 0)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    order = torch.argsort(image[ys, xs], descending=True)
    boxes, taken = [], []
    S, F = image.shape
    for i in order.tolist():
        y, x = int(ys[i]), int(xs[i])
        if any(abs(y - ty) <= 2 * half and abs(x - tx) <= 2 * half for ty, tx in taken):
            continue
        if half <= x < F - half and half <= y < S - half:
            taken.append((y, x))
            boxes.append((x - half, x + half + 1, y - half, y + half + 1))
        if len(boxes) == count:
            break
    return boxes


def main():
    P = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    P.add_argument("--shots", type=int, default=20)
    P.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    P.add_argument("--pixels", type=int, default=512)
    P.add_argument("--distance", type=float, default=150.0)
    P.add_argument("--wavelength", type=float, default=1.3)
    P.add_argument("--d-min", dest="d_min", type=float, default=2.5)
    P.add_argument("--ncells", type=float, default=20.0)
    P.add_argument("--spots", type=int, default=60, help="ROIs per shot")
    P.add_argument("--half", type=int, default=4, help="ROI half width, pixels")
    P.add_argument("--peak", type=float, default=200.0, help="photons at the median ROI peak")
    P.add_argument("--f-noise", dest="f_noise", type=float, default=0.3, help="start log F error (sd)")
    P.add_argument("--scale-noise", dest="scale_noise", type=float, default=0.3, help="start log G error (sd)")
    P.add_argument("--maxiter", type=int, default=200)
    P.add_argument("--fhkl-scale", dest="fhkl_scale", type=float, help="refine_lbfgsb scale for Fhkl (default its own)")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--out")
    args = P.parse_args()
    args.device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    rotations = point_group_422()
    hkl = unique_reflections(args.d_min, rotations)
    true_F = np.sqrt(rng.exponential(1.0, len(hkl))) * 100.0  # acentric Wilson
    sf = StructureFactors(hkl, true_F, rotations=rotations)
    missets = rng.uniform(-180.0, 180.0, (args.shots, 3))

    # Truth: full images, ROIs at the strongest spots, then the scale that puts the median
    # ROI peak at --peak photons.
    rois, peaks = [], []
    with torch.no_grad():
        for m in missets:
            image = shot_model(args, m, sf)()
            boxes = strongest_spots(image, args.spots, args.half)
            rois.append(boxes)
            peaks += [float(image[y1:y2, x1:x2].max()) for x1, x2, y1, y2 in boxes]
    spot_scale = args.peak / float(np.median(peaks))
    true_logG = rng.normal(0.0, 0.2, args.shots)
    true_logG[0] = 0.0

    def serial():
        return SerialExperiment([shot_model(args, m, sf, r, spot_scale) for m, r in zip(missets, rois)],
                                share_detector=True)

    truth = serial()
    with torch.no_grad():
        for shot, g in zip(truth.shots, true_logG):
            shot.beam_params.delta_log_spot_scale.fill_(float(g))
        data = [torch.poisson(d.clamp_min(0)) for d in truth(roi_only=True)]
    del truth

    model = serial()
    model.freeze().refine("Fhkl", "scale")
    model.shots[0].freeze("scale")
    with torch.no_grad():
        model.shots[0].fhkl_params.delta_log_F.add_(
            torch.as_tensor(rng.normal(0.0, args.f_noise, len(hkl)), dtype=DT, device=args.device))
        for shot, g in zip(model.shots[1:], true_logG[1:]):
            shot.beam_params.delta_log_spot_scale.fill_(float(g + rng.normal(0.0, args.scale_noise)))

    calls = {"n": 0, "seconds": 0.0}

    def term(i):
        def f():
            t0 = time.perf_counter()
            value = gaussian_nll(model([i], roi_only=True)[0], data[i], 1.0)
            calls["seconds"] += time.perf_counter() - t0  # forward only; backward runs after
            return value
        return f

    fhkl = model.shots[0].fhkl_params
    # the reflections the ROIs see: those the first gradient reaches
    loss = sum(term(i)() for i in range(args.shots))
    loss.backward()
    seen = (fhkl.delta_log_F.grad.abs() > 0).cpu().numpy()
    fhkl.delta_log_F.grad = None

    def agreement():
        fit = fhkl.amplitudes().detach().cpu().numpy()[: len(hkl)][seen]
        truth_f = true_F[seen]
        cc = float(np.corrcoef(np.log(fit), np.log(truth_f))[0, 1])
        k = (fit * truth_f).sum() / (fit * fit).sum()
        return {"cc_logF": cc, "R": float(np.abs(k * fit - truth_f).sum() / truth_f.sum())}

    start = agreement()
    trajectory = []

    def record():
        """A last term of nothing trainable: notes the loss and F error at each evaluation."""
        trajectory.append({"evaluation": len(trajectory) + 1, **agreement()})
        return torch.zeros((), dtype=DT, device=args.device)
    if args.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    scales = None if args.fhkl_scale is None else {"Fhkl": args.fhkl_scale}
    result = refine_lbfgsb(model, [term(i) for i in range(args.shots)] + [record], scales=scales, maxiter=args.maxiter)
    if args.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    evaluations = int(result.nfev)
    fitted_logG = np.array([float(s.beam_params.delta_log_spot_scale) for s in model.shots])
    report = {
        "commit": git_commit(), "args": {k: str(v) for k, v in vars(args).items()},
        "unique_reflections": len(hkl), "seen_reflections": int(seen.sum()),
        "roi_pixels": int(sum(d.numel() for d in data)), "evaluations": evaluations, "iterations": int(result.nit),
        "converged": bool(result.success), "message": str(result.message),
        "wall_seconds": wall, "seconds_per_evaluation": wall / max(evaluations, 1),
        "forward_seconds_per_evaluation": calls["seconds"] / max(evaluations, 1),
        "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9 if args.device.type == "cuda" else None,
        "start": start, "end": agreement(),
        "scale_rms_error": float(np.sqrt(np.mean((fitted_logG - true_logG) ** 2))),
        "trajectory": trajectory,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "trajectory"}, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
