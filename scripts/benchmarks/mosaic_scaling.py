"""Mosaic-domain scaling benchmark: torch (CPU/GPU, eager/compiled) vs nanoBragg.c.

Answers "does the mosaic axis actually buy us anything on the GPU?" by sweeping
``mosaic_domains`` with everything else held fixed and splitting the torch wall time
into (a) umat generation -- the serial ran1 python loop in utils/c_random.py --
(b) crystal rotation setup and (c) the physics kernel.

Each sweep point runs in its own subprocess so peak RSS / CUDA peak memory are clean.

    # torch sweep, eager GPU
    NANOBRAGG_DISABLE_COMPILE=1 python scripts/benchmarks/mosaic_scaling.py \
        --device cuda --domains 1,10,50,100,500,1000
    # compiled GPU (warm timings only; first call pays 6-12 s of compilation)
    python scripts/benchmarks/mosaic_scaling.py --device cuda --domains 1,10,100
    # C reference
    python scripts/benchmarks/mosaic_scaling.py --mode c \
        --c-binary ../wt-oracle/nanoBragg --domains 1,10,50,100

Output is one JSON object per sweep point on stdout plus a human-readable table.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Fixed physical setup shared by the torch and C legs so the comparison is apples to apples.
CELL = (100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
NCELLS = 5
DEFAULT_F = 100.0
LAMBDA_A = 1.0
DISTANCE_MM = 100.0
PIXEL_MM = 0.1
MOSAIC_SPREAD_DEG = 0.5
OSC_RANGE_DEG = 1.0  # only used when phi_steps > 1

# ru_maxrss is bytes on darwin, kibibytes on linux
_RSS_SCALE = 1e9 if sys.platform == "darwin" else 1e6


# --------------------------------------------------------------------------- torch


def _build(device, dtype, npix, domains, phi_steps, oversample):
    from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
    from nanobrag_torch.models import Crystal, Detector
    from nanobrag_torch.simulator import Simulator

    cc = CrystalConfig(
        cell_a=CELL[0], cell_b=CELL[1], cell_c=CELL[2],
        cell_alpha=CELL[3], cell_beta=CELL[4], cell_gamma=CELL[5],
        N_cells=(NCELLS, NCELLS, NCELLS),
        default_F=DEFAULT_F,
        mosaic_spread_deg=MOSAIC_SPREAD_DEG,
        mosaic_domains=domains,
        osc_range_deg=(OSC_RANGE_DEG if phi_steps > 1 else 0.0),
        phi_steps=phi_steps,
    )
    dc = DetectorConfig(
        distance_mm=DISTANCE_MM, pixel_size_mm=PIXEL_MM,
        spixels=npix, fpixels=npix, oversample=oversample,
    )
    crystal = Crystal(cc, device=device, dtype=dtype)
    detector = Detector(dc, device=device, dtype=dtype)
    sim = Simulator(crystal, detector, beam_config=BeamConfig(wavelength_A=LAMBDA_A),
                    device=device, dtype=dtype)
    return sim, crystal, cc


def run_torch(args) -> dict:
    import torch

    dtype = getattr(torch, args.dtype)
    device = args.device
    cuda = device.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    sim, crystal, cc = _build(device, dtype, args.npix, args.domains,
                              args.phi_steps, args.oversample)
    batch = args.batch if args.batch and args.batch > 0 else None

    out = {
        "mode": "torch", "device": device, "dtype": args.dtype, "npix": args.npix,
        "domains": args.domains, "phi_steps": args.phi_steps,
        "oversample": args.oversample, "pixel_batch_size": batch,
        "compile_disabled": os.environ.get("NANOBRAGG_DISABLE_COMPILE", "0") == "1",
    }

    # --- (a) umat generation on its own: the serial python ran1 loop ---------
    from nanobrag_torch.utils.c_random import mosaic_rotation_umats
    spread_rad = torch.deg2rad(torch.tensor(MOSAIC_SPREAD_DEG, device=device, dtype=dtype))
    sync()
    ts = []
    for _ in range(max(3, args.repeats)):
        t = time.perf_counter()
        mosaic_rotation_umats(spread_rad, args.domains, seed=-12345678,
                              dtype=dtype, device=torch.device(device))
        sync()
        ts.append(time.perf_counter() - t)
    out["t_umat_s"] = min(ts)

    # --- (b) full rotation setup (umats + phi + rotate + reciprocal recompute) ---
    sync()
    ts = []
    for _ in range(max(3, args.repeats)):
        t = time.perf_counter()
        crystal.get_rotated_real_vectors(cc)
        sync()
        ts.append(time.perf_counter() - t)
    out["t_rot_setup_s"] = min(ts)

    # --- (c) full forward pass ----------------------------------------------
    try:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        sync()
        t = time.perf_counter()
        img = sim.run(pixel_batch_size=batch)
        sync()
        out["t_first_s"] = time.perf_counter() - t

        ts = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            img = sim.run(pixel_batch_size=batch)
            sync()
            ts.append(time.perf_counter() - t)
        ts.sort()
        out["t_run_s"] = ts[0]
        out["t_run_median_s"] = ts[len(ts) // 2]
        out["t_run_max_s"] = ts[-1]
        out["t_kernel_s"] = ts[0] - out["t_rot_setup_s"]
        out["checksum"] = float(img.double().sum())
        out["peak_gpu_gb"] = (torch.cuda.max_memory_allocated() / 1e9) if cuda else None
        out["ok"] = True
    except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else RuntimeError) as exc:
        out["ok"] = False
        out["error"] = f"OOM: {type(exc).__name__}"
    # ru_maxrss is bytes on darwin, kibibytes on linux
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_SCALE
    return out


# ------------------------------------------------------------------------------- C


def run_c(args) -> dict:
    cmd = [
        args.c_binary,
        "-default_F", str(DEFAULT_F),
        "-cell", *[str(v) for v in CELL],
        "-N", str(NCELLS),
        "-lambda", str(LAMBDA_A),
        "-distance", str(DISTANCE_MM),
        "-detpixels", str(args.npix),
        "-pixel", str(PIXEL_MM),
        "-mosaic", str(MOSAIC_SPREAD_DEG),
        "-mosaic_domains", str(args.domains),
        "-oversample", str(args.oversample),
        "-nonoise",
        "-nopgm",
        "-floatfile", os.path.join(args.workdir, "c_float.bin"),
        "-intfile", os.path.join(args.workdir, "c_int.img"),
    ]
    if args.phi_steps > 1:
        cmd += ["-osc", str(OSC_RANGE_DEG), "-phisteps", str(args.phi_steps)]
    ts, rss = [], 0
    for _ in range(args.repeats):
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        t = time.perf_counter()
        proc = subprocess.run(cmd, cwd=args.workdir, capture_output=True, text=True)
        ts.append(time.perf_counter() - t)
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        rss = max(rss, after, before)
        if proc.returncode != 0:
            return {"mode": "c", "domains": args.domains, "ok": False,
                    "error": proc.stderr[-400:]}
    ts.sort()
    return {
        "mode": "c", "device": "cpu-1thread", "dtype": "float64",
        "npix": args.npix, "domains": args.domains, "phi_steps": args.phi_steps,
        "oversample": args.oversample,
        "t_run_s": ts[0], "t_run_median_s": ts[len(ts) // 2], "t_run_max_s": ts[-1],
        "peak_rss_gb": rss / _RSS_SCALE, "ok": True,
    }


# --------------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="torch", choices=["torch", "c"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--npix", type=int, default=1024)
    ap.add_argument("--phi-steps", type=int, default=1)
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--batch", type=int, default=0, help="pixel_batch_size; 0 = None")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--domains", default="1,10,50,100,500,1000")
    ap.add_argument("--c-binary", default="")
    ap.add_argument("--workdir", default="/tmp")
    ap.add_argument("--tag", default="")
    ap.add_argument("--child", type=int, default=0,
                    help="internal: run exactly this one domain count in-process")
    args = ap.parse_args()

    if args.child:
        args.domains = args.child
        res = run_c(args) if args.mode == "c" else run_torch(args)
        res["tag"] = args.tag
        print("JSON " + json.dumps(res))
        return 0

    rows = []
    for d in [int(x) for x in args.domains.split(",")]:
        cmd = [sys.executable, os.path.abspath(__file__), "--child", str(d),
               "--mode", args.mode, "--device", args.device, "--dtype", args.dtype,
               "--npix", str(args.npix), "--phi-steps", str(args.phi_steps),
               "--oversample", str(args.oversample), "--batch", str(args.batch),
               "--repeats", str(args.repeats), "--workdir", args.workdir,
               "--tag", args.tag or f"{args.mode}-{args.device}-{args.dtype}"]
        if args.c_binary:
            cmd += ["--c-binary", os.path.abspath(args.c_binary)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("JSON ")]
        if not line:
            print(f"domains={d}: FAILED\n{proc.stdout[-800:]}\n{proc.stderr[-1500:]}",
                  file=sys.stderr)
            rows.append({"domains": d, "ok": False,
                         "error": proc.stderr.strip().splitlines()[-1][:200]
                         if proc.stderr.strip() else "no output"})
            continue
        row = json.loads(line[-1][5:])
        rows.append(row)
        print("JSON " + json.dumps(row), flush=True)

    hdr = (f"{'dom':>6} {'run_s':>9} {'med_s':>9} {'max_s':>9} {'umat_s':>9} "
           f"{'rot_s':>9} {'kern_s':>9} {'peakGPU':>8} {'peakRSS':>8}")
    print("\n# " + (args.tag or args.mode) + f"  {args.npix}^2 phi={args.phi_steps} "
          f"osamp={args.oversample} batch={args.batch or 'None'}")
    print(hdr)
    for r in rows:
        if not r.get("ok"):
            print(f"{r['domains']:>6} {'FAIL/OOM':>9}  {r.get('error','')[:60]}")
            continue
        g = lambda k: (f"{r[k]:9.4f}" if r.get(k) is not None else f"{'-':>9}")
        pg = f"{r['peak_gpu_gb']:8.2f}" if r.get("peak_gpu_gb") else f"{'-':>8}"
        pr = f"{r['peak_rss_gb']:8.2f}" if r.get("peak_rss_gb") else f"{'-':>8}"
        print(f"{r['domains']:>6} {g('t_run_s')} {g('t_run_median_s')} {g('t_run_max_s')} "
              f"{g('t_umat_s')} {g('t_rot_setup_s')} {g('t_kernel_s')} {pg} {pr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
