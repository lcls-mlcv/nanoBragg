"""Refine each lattice's Ncells and eta across rotation frames; G free per frame and lattice.

Orientation, cell and detector stay at DIALS' scan-varying solution, F is the PDB's F_calc.
Frames listed under --test never inform Ncells or eta: they only get their own G, and their
own orientation with --misset, fitted to the same likelihood as the other frames, then are scored.

    OMP_NUM_THREADS=2 NANOBRAGG_DISABLE_COMPILE=1 PYTHONPATH=src:$PYTHONPATH libtbx.python \
        scripts/rotation_refinement.py --expt sv.expt --refl sv.refl --pdb model.pdb \
        --images 'data/H4_7_%05d.cbf' --hot hot.npz --out result.json
"""
import argparse, copy, json, math, os, time
import numpy as np, torch
import iotbx.pdb
from dials.array_family import flex
from dxtbx import load
from dxtbx.model.experiment_list import ExperimentList
from scipy.optimize import nnls
from scitbx.matrix import sqr
from nanobrag_torch.compat.cctbx import _axis_angle_matrix, asu_structure_factors, experiment_model_from_dxtbx
from nanobrag_torch.models.experiment import SerialExperiment
from nanobrag_torch.utils.losses import gaussian_nll
from nanobrag_torch.utils.refinement import refine_lbfgsb

P = argparse.ArgumentParser()
P.add_argument("--expt", required=True, help="scan-varying refined experiments, one per lattice")
P.add_argument("--refl", required=True, help="their reflections")
P.add_argument("--pdb", required=True)
P.add_argument("--images", required=True, help="printf template, 1-based image numbers")
P.add_argument("--hot", help=".npz with hot: True on pixels to drop")
P.add_argument("--fit", default="100,300,500,700,900,1100,1300,1500")
P.add_argument("--test", default="200,600,1000,1400")
P.add_argument("--spots", type=int, default=100)
P.add_argument("--half", type=int, default=7)
P.add_argument("--N", type=float, nargs=3, default=(40, 20, 16))
P.add_argument("--eta", type=float, default=0.2)
P.add_argument("--domains", type=int, default=26)
P.add_argument("--phi-steps", dest="phi_steps", type=int, default=4)
P.add_argument("--sign", type=float, default=1.0)
P.add_argument("--sensor-layers", dest="layers", type=int, default=0)
P.add_argument("--misset", action="store_true", help="refine each frame and lattice's orientation too")
P.add_argument("--rmin", type=float, default=400.0, help="pixels from the beam; F_calc has no bulk solvent inside about 8.5 A")
P.add_argument("--flash", type=float, default=20.0,
               help="drop a box whose peak is this many times the same box on both neighbouring frames; 0 keeps all")
P.add_argument("--maxiter", type=int, default=100)
P.add_argument("--start-only", dest="start_only", action="store_true")
P.add_argument("--out", default="rotation_refinement.json")
A = P.parse_args()
device = torch.device("cuda")
fit, test = [[int(x) for x in s.split(",") if x] for s in (A.fit, A.test)]

E = ExperimentList.from_file(A.expt, check_format=False)
scan, gonio, beam, det = E[0].scan, E[0].goniometer, E[0].beam, E[0].detector
axis = np.array(gonio.get_rotation_axis())  # in the lab, after the setting rotation
datum = np.array(gonio.get_rotation_axis_datum())
fc = iotbx.pdb.input(A.pdb).xray_structure_simple().structure_factors(
    d_min=1.7, algorithm="fft").f_calc().amplitudes()
sf = asu_structure_factors(fc, anomalous=False)
refl = flex.reflection_table.from_file(A.refl)
xo, yo, zo = (np.array(v) for v in refl["xyzobs.px.value"].parts())
intensity = np.array(refl["intensity.sum.value"])
lattice = np.array(refl["id"])
hot = np.load(A.hot)["hot"] if A.hot else np.False_
bx, by = det[0].get_beam_centre_px(beam.get_s0())
osc = scan.get_oscillation()[1]


def lab_crystal(exp, i):
    A_i = sqr(exp.crystal.get_A_at_scan_point(i))
    R = sqr(_axis_angle_matrix(datum, math.radians(scan.get_angle_from_array_index(i))).flatten().tolist())
    c = copy.deepcopy(exp.crystal)
    c.reset_scan_points()
    c.set_A(sqr(gonio.get_setting_rotation()) * R * sqr(gonio.get_fixed_rotation()) * A_i)
    return c


def read(i):
    return load(A.images % (i + 1)).get_raw_data().as_numpy_array().astype(np.float64)


def rois_of(i, img, good):
    """Strongest indexed spots of frame i, one tilted plane each; returns rois, tilt_abc.

    Real spots here rock over three to five frames, so a box bright on one frame alone is a
    cosmic ray that DIALS happened to index, and the model can never fit it.
    """
    sel = np.where((zo >= i - 0.5) & (zo < i + 1.5) & (lattice >= 0) & (np.hypot(xo - bx, yo - by) > A.rmin))[0]
    sel = sel[np.argsort(-intensity[sel])]
    H, W = img.shape
    h, rois, tilts = A.half, [], []
    used = np.zeros(img.shape, bool)
    side = np.maximum(read(i - 1), read(i + 1)) if A.flash else None
    flashes = []
    for j in sel:
        x, y = int(xo[j]), int(yo[j])
        x1, x2, y1, y2 = x - h, x + h + 1, y - h, y + h + 1
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H or used[y1:y2, x1:x2].any() or good[y1:y2, x1:x2].mean() < 0.9:
            continue
        used[y1:y2, x1:x2] = True
        box = good[y1:y2, x1:x2]
        if A.flash and img[y1:y2, x1:x2][box].max() > A.flash * max(1.0, side[y1:y2, x1:x2][box].max()):
            flashes.append((x, y, img[y1:y2, x1:x2][box].max()))
            continue
        yy, xx = np.mgrid[0:y2 - y1, 0:x2 - x1]
        keep = good[y1:y2, x1:x2].ravel().copy()
        M = np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size)), 1).astype(float)
        v = img[y1:y2, x1:x2].ravel()
        for _ in range(5):  # planes through the pixels outside the spot
            abc = np.linalg.lstsq(M[keep], v[keep], rcond=None)[0]
            r = v - M @ abc
            s = max(1.0, 1.4826 * np.median(np.abs(r[keep])))
            keep = good[y1:y2, x1:x2].ravel() & (r < 2.5 * s)
        rois.append((x1, x2, y1, y2))
        tilts.append(abc)
        if len(rois) == A.spots:
            break
    if flashes:
        print("frame", i, "dropped flashes (x, y, peak):", flashes, flush=True)
    return rois, tilts


frames = {}
models = []
sensor = dict(detector_thick_mm=0.0)
if A.layers:
    sensor = dict(detector_thicksteps=A.layers, oversample_thick=True)
t0 = time.time()
for i in fit + test:
    img = read(i)
    good = (img >= 0) & (img < 1e5) & ~hot
    rois, tilts = rois_of(i, img, good)
    start = len(models)
    for l, exp in enumerate(E):
        models.append(experiment_model_from_dxtbx(
            det, beam, lab_crystal(exp, i), Ncells_abc=tuple(A.N), shape="gauss", mosaic_spread_deg=A.eta,
            mosaic_domains=A.domains, oversample=1, rois=rois, **sensor,
            tilt_abc=tilts if l == 0 else np.zeros((len(rois), 3)),
            phi_start_deg=A.sign * osc / A.phi_steps / 2, osc_range_deg=A.sign * osc, phi_steps=A.phi_steps,
            spindle_axis=axis, structure_factors=sf, device=device))
    frames[i] = dict(img=img, good=good, models=models[start:], rois=len(rois))
print("built", len(models), "models in", round(time.time() - t0), "s; rois", {i: f["rois"] for i, f in frames.items()}, flush=True)
serial = SerialExperiment(models, share_detector=True)
serial.freeze()

for l in range(len(E)):  # one Ncells and eta per lattice, across every frame
    ref = frames[fit[0]]["models"][l]
    for f in frames.values():
        f["models"][l].crystal_params.delta_log_N_cells = ref.crystal_params.delta_log_N_cells
        f["models"][l].crystal_params.delta_log_eta = ref.crystal_params.delta_log_eta

for f in frames.values():
    m0 = f["models"][0]
    f["data"] = m0.roi_values(f["img"])
    f["trusted"] = m0.roi_values(f["good"])
    with torch.no_grad():
        f["bg"] = m0.background_params.values().clone()


def predict(f):
    return sum(m(roi_only=True) for m in f["models"])


def set_G(f):
    """Non-negative least squares for one frame's lattice scales."""
    with torch.no_grad():
        parts = []
        for k, m in enumerate(f["models"]):
            G0 = torch.exp(m.beam_params.delta_log_spot_scale).item()
            q = m(roi_only=True) - (f["bg"] if k == 0 else 0)
            parts.append((q / G0)[f["trusted"]].cpu().numpy())
        X = np.stack(parts, 1)
        y = (f["data"] - f["bg"])[f["trusted"]].cpu().numpy()
        norm = X.max(0).clip(1e-30)
        w = nnls(X / norm, y)[0] / norm
        floor = 1e-4 * w.max()
        for m, g in zip(f["models"], w):
            m.beam_params.delta_log_spot_scale.data.fill_(math.log(max(g, floor)))


def G_of(f):
    return [torch.exp(m.beam_params.delta_log_spot_scale).item() for m in f["models"]]


def score(f):
    with torch.no_grad():
        p, d, t = predict(f), f["data"].to(torch.float64), f["trusted"]
        cc = torch.corrcoef(torch.stack((p[t].to(torch.float64), d[t])))[0, 1].item()
        nll = gaussian_nll(p, f["data"], 1.0, trusted=t).item() / int(t.sum())
    return round(cc, 4), round(nll, 3)


for f in frames.values():
    set_G(f)
start = {i: score(f) for i, f in frames.items()}
print("start (cc, nll/px):", start, flush=True)
print("start G:", {i: [float(f"{g:.3g}") for g in G_of(f)] for i, f in frames.items()}, flush=True)
print("peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
result = dict(args=vars(A), start=start)
if not A.start_only:
    for f in (frames[i] for i in fit):
        for m in f["models"]:
            m.refine("Ncells", "eta", "scale", *(["misset"] if A.misset else []))

    def term(f):
        return lambda: gaussian_nll(predict(f), f["data"], 1.0, trusted=f["trusted"])

    t0 = time.time()
    r = refine_lbfgsb(serial, [term(frames[i]) for i in fit], maxiter=A.maxiter, scales=dict(scale=0.1))
    seconds = time.time() - t0
    for i in test:  # Ncells and eta stay as the fitted frames left them
        f = frames[i]
        set_G(f)
        serial.freeze()
        for m in f["models"]:
            m.refine("scale", *(["misset"] if A.misset else []))

        def held_out():
            return gaussian_nll(predict(f), f["data"], 1.0, trusted=f["trusted"])

        refine_lbfgsb(serial, held_out, maxiter=A.maxiter, scales=dict(scale=0.1))
    end = {i: score(f) for i, f in frames.items()}
    lat = []
    for l in range(len(E)):
        cp = frames[fit[0]]["models"][l].crystal_params
        lat.append(dict(N=(cp.base_N_cells * torch.exp(cp.delta_log_N_cells)).tolist(), eta=cp.eta().tolist()))
    result.update(end=end, lattices=lat, seconds=seconds, nfev=int(r.nfev), nit=int(r.nit), message=str(r.message),
                  G={i: G_of(f) for i, f in frames.items()},
                  peak_GiB=torch.cuda.max_memory_allocated() / 2**30)
    print(json.dumps({k: result[k] for k in ("end", "lattices", "seconds", "nfev", "nit", "message", "peak_GiB")}), flush=True)
json.dump(result, open(A.out, "w"), indent=1, default=str)
