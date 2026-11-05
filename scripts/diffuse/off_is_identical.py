"""
Gate for the diffuse extension: with no diffuse model, nothing changes, bit for bit.

Run the same script against two trees and compare:

    PYTHONPATH=<base tree>/src   python scripts/diffuse/off_is_identical.py write base.npz
    PYTHONPATH=<branch tree>/src python scripts/diffuse/off_is_identical.py write branch.npz
    PYTHONPATH=<either>/src      python scripts/diffuse/off_is_identical.py compare base.npz branch.npz

"write" stores, for a fixed set of Simulator, ExperimentModel and SerialExperiment cases, the
image, the gradient of a weighted image sum with respect to every trainable parameter (and the
Simulator's geometry leaves) and the state_dict keys. "compare" demands np.array_equal on every
array and identical key lists. It uses only API that exists without the extension, so the base
tree can run it. Pass --device cuda to repeat on a GPU.
"""
import argparse
import os
import sys

os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import numpy as np
import torch

from nanobrag_torch.config import (BeamConfig, CrystalConfig, CrystalShape, DetectorConfig,
                                   DetectorConvention, DetectorPivot)
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel, SerialExperiment
from nanobrag_torch.simulator import Simulator

P = 24


def _crystal(**kw):
    base = dict(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(5, 5, 5),
                default_F=100.0)
    return CrystalConfig(**{**base, **kw})


def _detector(**kw):
    base = dict(distance_mm=60.0, pixel_size_mm=0.4, spixels=P, fpixels=P, oversample=1,
                detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)
    return DetectorConfig(**{**base, **kw})


def _beam(sources=0):
    beam = BeamConfig(wavelength_A=1.2, fluence=1e24)
    if sources:
        beam.source_directions = torch.tensor([[-1.0, 0.0, 0.0]] * sources, dtype=torch.float64)
        beam.source_wavelengths = torch.linspace(1.19, 1.21, sources, dtype=torch.float64) * 1e-10
        beam.source_weights = torch.arange(1, sources + 1, dtype=torch.float64)
    return beam


SIMULATOR_CASES = {
    "square": dict(crystal=dict(shape=CrystalShape.SQUARE)),
    "gauss": dict(crystal=dict(shape=CrystalShape.GAUSS)),
    "mosaic-phi": dict(crystal=dict(shape=CrystalShape.GAUSS, mosaic_spread_deg=0.3, mosaic_domains=5,
                                    phi_start_deg=1.0, osc_range_deg=0.5, phi_steps=3)),
    "sources": dict(beam=dict(sources=3)),
    "sensor": dict(detector=dict(detector_thick_um=300.0, detector_abs_um=200.0, detector_thicksteps=3)),
    "oversample": dict(detector=dict(oversample=3)),
}

MODEL_CASES = {
    "whole": dict(),
    "roi": dict(model=dict(background_rois=[(2, 12, 3, 9), (14, 22, 10, 20)]), run=dict(roi_only=True)),
    "panels": dict(panels=[(4.8, 4.8), (-5.0, 4.8)]),
    "gauss-eta": dict(crystal=dict(shape=CrystalShape.GAUSS, mosaic_spread_deg=0.2, mosaic_domains=4)),
}


def _weights(shape, device, dtype):
    generator = torch.Generator().manual_seed(1234)
    return torch.rand(shape, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)


def _store(out, prefix, image, named):
    out[f"{prefix}/image"] = image.detach().cpu().numpy()
    for name, tensor in named:
        if tensor.grad is not None:
            out[f"{prefix}/grad/{name}"] = tensor.grad.detach().cpu().numpy()


def simulator_cases(out, device, dtype):
    for name, case in SIMULATOR_CASES.items():
        distance = torch.tensor(60.0, dtype=dtype, device=device, requires_grad=True)
        detector = _detector(**case.get("detector", {}))
        detector.distance_mm = distance
        crystal = Crystal(_crystal(**case.get("crystal", {})), device=device, dtype=dtype)
        sim = Simulator(crystal, Detector(detector, device=device, dtype=dtype),
                        beam_config=_beam(**case.get("beam", {})), device=device, dtype=dtype)
        image = sim.run()
        (image * _weights(image.shape, device, dtype)).sum().backward()
        _store(out, f"sim/{name}/{dtype}", image, [("distance_mm", distance)])


def model_cases(out, device, dtype):
    for name, case in MODEL_CASES.items():
        detector = ([_detector(beam_center_f=f, beam_center_s=s) for f, s in case["panels"]]
                    if "panels" in case else _detector())
        model = ExperimentModel(_crystal(**case.get("crystal", {})), detector, _beam(), device=device,
                                dtype=dtype, **case.get("model", {}))
        image = model(**case.get("run", {}))
        (image * _weights(image.shape, device, dtype)).sum().backward()
        _store(out, f"model/{name}/{dtype}", image, model.named_parameters())
        out[f"model/{name}/{dtype}/state_dict_keys"] = np.array(sorted(model.state_dict()))
    shots = [ExperimentModel(_crystal(misset_deg=m), _detector(), _beam(), device=device, dtype=dtype)
             for m in [(5.0, -3.0, 2.0), (1.0, 2.0, 3.0)]]
    serial = SerialExperiment(shots, share_detector=True)
    images = serial()
    sum((i * _weights(i.shape, device, dtype)).sum() for i in images).backward()
    _store(out, f"serial/{dtype}", torch.stack(images), serial.named_parameters())
    out[f"serial/{dtype}/state_dict_keys"] = np.array(sorted(serial.state_dict()))


def write(path, device):
    device = torch.device(device)
    out = {}
    for dtype in (torch.float32, torch.float64):
        simulator_cases(out, device, dtype)
        model_cases(out, device, dtype)
    np.savez(path, **out)
    print(f"wrote {len(out)} arrays to {path}")


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    failures = []
    if sorted(a.files) != sorted(b.files):
        failures.append(f"different entries: only in first {sorted(set(a.files) - set(b.files))}, "
                        f"only in second {sorted(set(b.files) - set(a.files))}")
    for key in sorted(set(a.files) & set(b.files)):
        if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
            failures.append(key)
    for failure in failures:
        print("DIFFERS:", failure)
    print(f"{len(a.files)} arrays compared, {len(failures)} differ")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("path")
    w.add_argument("--device", default="cpu")
    c = sub.add_parser("compare")
    c.add_argument("first")
    c.add_argument("second")
    args = parser.parse_args(argv)
    if args.command == "write":
        write(args.path, args.device)
        return 0
    return compare(args.first, args.second)


if __name__ == "__main__":
    sys.exit(main())
