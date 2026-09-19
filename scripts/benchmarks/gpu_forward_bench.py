"""
Forward-pass throughput benchmark for nanobrag_torch on CPU or GPU.

    NANOBRAGG_DISABLE_COMPILE=1 python scripts/benchmarks/gpu_forward_bench.py --device cuda   # eager
    python scripts/benchmarks/gpu_forward_bench.py --device cuda                              # torch.compile

Reports first-call time (includes compile warm-up), best warm time of 3, Mpix/s and peak GPU memory.
"""
import argparse, os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from nanobrag_torch.config import CrystalConfig, DetectorConfig, BeamConfig
from nanobrag_torch.models import Crystal, Detector
from nanobrag_torch.simulator import Simulator

CASES = [("1024^2 mos1 phi1", 1024, 1, 1), ("2048^2 mos1 phi1", 2048, 1, 1), ("1024^2 mos10 phi1", 1024, 10, 1),
         ("1024^2 mos10 phi10", 1024, 10, 10), ("2048^2 mos10 phi5", 2048, 10, 5), ("4096^2 mos1 phi1", 4096, 1, 1)]

def run(dev, npix, mos, phisteps, dtype, label, batch):
    cc = CrystalConfig(N_cells=(5, 5, 5), default_F=100.0, mosaic_spread_deg=(0.5 if mos > 1 else 0.0), mosaic_domains=mos,
                       osc_range_deg=(1.0 if phisteps > 1 else 0.0), phi_steps=phisteps)
    dc = DetectorConfig(distance_mm=100, pixel_size_mm=0.1, spixels=npix, fpixels=npix, oversample=1)
    sim = Simulator(Crystal(cc, device=dev, dtype=dtype), Detector(dc, device=dev, dtype=dtype),
                    beam_config=BeamConfig(wavelength_A=1.0), device=dev, dtype=dtype)
    cuda = dev.startswith("cuda")
    if cuda: torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.perf_counter(); img = sim.run(pixel_batch_size=batch)
    if cuda: torch.cuda.synchronize()
    first = time.perf_counter() - t
    ts = []
    for _ in range(3):
        t = time.perf_counter(); img = sim.run(pixel_batch_size=batch)
        if cuda: torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    warm = min(ts)
    mem = torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan")
    print(f"{label:22s} {str(dtype)[6:]:8s} batch={str(batch):5s} first {first:7.2f}s warm {warm:7.3f}s "
          f"{npix*npix/warm/1e6:8.1f} Mpix/s  peakGPU {mem:5.2f} GB  max {float(img.max()):.4g}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtypes", default="float32,float64")
    ap.add_argument("--batch", type=int, default=None, help="pixel_batch_size (rows per chunk)")
    args = ap.parse_args()
    print("compile disabled:", os.environ.get("NANOBRAGG_DISABLE_COMPILE", "0"), "| device:", args.device,
          torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "")
    for dt in args.dtypes.split(","):
        dtype = getattr(torch, dt)
        for label, npix, mos, phi in CASES:
            try:
                run(args.device, npix, mos, phi, dtype, label, args.batch)
            except torch.cuda.OutOfMemoryError:
                print(f"{label:22s} {dt:8s} OOM"); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
