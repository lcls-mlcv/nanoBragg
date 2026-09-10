"""
CCTBX / dxtbx compatibility layer for nanobrag_torch.

This module mirrors the wiring that ``simtbx.nanoBragg.sim_data.SimData`` and
``nanoBragg::set_dxtbx_detector_panel`` perform in cctbx, so that a dxtbx
Detector / Beam / Crystal (and a cctbx miller array) can drive the PyTorch
simulator without going through the nanoBragg CLI conventions.

Design rules
------------
* cctbx is an optional dependency. Every function accepts duck-typed dxtbx
  objects (anything exposing the same getters), so the layer can be unit
  tested without cctbx installed. Real dxtbx / cctbx objects work unchanged.
* Units follow nanoBragg.cpp: dxtbx works in mm and Å; the torch configs take
  mm for the detector, Å and degrees for the cell, Å⁻¹ for reciprocal vectors.
* The dxtbx panel is ingested exactly as ``set_dxtbx_detector_panel`` does:
  fdet/sdet from the panel axes, odet = fdet × sdet (flipped if the geometry is
  left handed), pix0 = origin / 1000, Fclose/Sclose/close_distance from pix0,
  CUSTOM convention (no MOSFLM half-pixel offset), SAMPLE pivot, no rotations.
* The crystal orientation is ingested as the transpose of dxtbx ``get_A()``:
  rows a*, b*, c* in Å⁻¹, which is what ``Amatrix_dials2nanoBragg`` feeds to
  ``nanoBragg.Amatrix``.

Public API
----------
detector_config_from_dxtbx_panel(panel, s0, ...)
crystal_config_from_dxtbx(crystal, ...)          # or crystal_config_from_A(...)
beam_config_from_dxtbx(beam, ...)
structure_factors_from_miller_array(miller_array)
set_structure_factors(crystal, indices, amplitudes, default_F)
set_mosaic_blocks(crystal, umats)
isotropic_umats(mos_spread_deg, n_domains, seed)
simulator_from_dxtbx(detector, beam, crystal, ...)
simulator_from_sim_data(SIM, ...)
MultiPanelSimulator
to_raw_pixels(image)
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..config import (
    BeamConfig,
    CrystalConfig,
    CrystalShape,
    DetectorConfig,
    DetectorConvention,
    DetectorPivot,
)
from ..models.crystal import Crystal
from ..models.detector import Detector
from ..simulator import Simulator

Vec3 = Tuple[float, float, float]

_SHAPE_MAP = {
    "square": CrystalShape.SQUARE,
    "round": CrystalShape.ROUND,
    "gauss": CrystalShape.GAUSS,
    "gauss_argchk": CrystalShape.GAUSS,
    "tophat": CrystalShape.TOPHAT,
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length vector cannot be normalised")
    return v / n


def _as_tuple3(v) -> Vec3:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return (float(v[0]), float(v[1]), float(v[2]))


def _xtal_shape(shape) -> CrystalShape:
    if isinstance(shape, CrystalShape):
        return shape
    if shape is None:
        return CrystalShape.SQUARE
    key = str(shape).lower()
    # cctbx exposes enum-like objects whose str() looks like "shapetype.Gauss"
    key = key.split(".")[-1]
    if key not in _SHAPE_MAP:
        raise ValueError(f"unknown crystal shape {shape!r}; expected one of {sorted(_SHAPE_MAP)}")
    return _SHAPE_MAP[key]


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
def detector_config_from_dxtbx_panel(
    panel,
    s0,
    *,
    oversample: int = -1,
    detector_thicksteps: Optional[int] = None,
    detector_thick_mm: Optional[float] = None,
    detector_attenuation_length_mm: Optional[float] = None,
    verbose: bool = False,
) -> DetectorConfig:
    """
    Build a DetectorConfig that reproduces ``nanoBragg::set_dxtbx_detector_panel``.

    Args:
        panel: dxtbx Panel (or duck type) with get_fast_axis(), get_slow_axis(),
            get_origin() [mm], get_pixel_size() [mm], get_image_size() (fast, slow),
            and optionally get_thickness() [mm], get_mu() [1/mm], get_gain().
        s0: incident wavevector (any length); only its direction is used, and it
            is passed to panel.get_beam_centre() if that method exists.
        oversample: sub-pixel sampling; -1 lets the simulator auto-select like C.
        detector_thicksteps: number of absorption layers (None keeps the config default).
        detector_thick_mm / detector_attenuation_length_mm: override the panel's
            sensor thickness (mm) and attenuation length (mm). When omitted they
            are taken from panel.get_thickness() and 1/panel.get_mu() as cctbx does.
    """
    fdet = _unit(panel.get_fast_axis())
    sdet = _unit(panel.get_slow_axis())
    odet = _unit(np.cross(fdet, sdet))
    pix0_m = np.asarray(panel.get_origin(), dtype=np.float64) / 1000.0

    close_distance_m = float(np.dot(pix0_m, odet))
    righthanded = True
    if close_distance_m < 0:
        # nanoBragg.cpp: "dxtbx model seems to be lefthanded. Inverting odet_vector."
        odet = -odet
        close_distance_m = float(np.dot(pix0_m, odet))
        righthanded = False
        if verbose:
            print("WARNING: dxtbx panel is left handed; inverting odet_vector")

    # Point of closest approach, expressed along the panel axes (metres).
    fclose_m = -float(np.dot(pix0_m, fdet))
    sclose_m = -float(np.dot(pix0_m, sdet))

    pixel_size_mm = float(panel.get_pixel_size()[0])
    fpixels, spixels = (int(x) for x in panel.get_image_size())

    thick_mm = detector_thick_mm
    attn_mm = detector_attenuation_length_mm
    if thick_mm is None and hasattr(panel, "get_thickness"):
        thick_mm = float(panel.get_thickness())
    if attn_mm is None and hasattr(panel, "get_mu"):
        mu = float(panel.get_mu())
        if mu > 0:
            attn_mm = 1.0 / mu
    thick_um = 0.0 if thick_mm is None else 1000.0 * thick_mm
    abs_um = None if attn_mm is None else 1000.0 * attn_mm

    beam_vector = _as_tuple3(_unit(np.asarray(s0, dtype=np.float64)))

    cfg = DetectorConfig(
        distance_mm=1000.0 * close_distance_m,
        close_distance_mm=1000.0 * close_distance_m,
        pixel_size_mm=pixel_size_mm,
        spixels=spixels,
        fpixels=fpixels,
        # CUSTOM convention + SAMPLE pivot: pix0 = -Fclose*fdet - Sclose*sdet + close_distance*odet
        beam_center_f=1000.0 * fclose_m,
        beam_center_s=1000.0 * sclose_m,
        beam_center_source="explicit",
        detector_convention=DetectorConvention.CUSTOM,
        detector_pivot=DetectorPivot.SAMPLE,
        custom_fdet_vector=_as_tuple3(fdet),
        custom_sdet_vector=_as_tuple3(sdet),
        custom_odet_vector=_as_tuple3(odet),
        custom_beam_vector=beam_vector,
        oversample=oversample,
        detector_thick_um=thick_um,
        detector_abs_um=abs_um,
    )
    if detector_thicksteps is not None:
        cfg = replace(cfg, detector_thicksteps=int(detector_thicksteps))
    cfg._dxtbx_righthanded = righthanded  # informational only
    return cfg


def _expected_pix0_m(cfg: DetectorConfig) -> np.ndarray:
    """pix0 implied by a CUSTOM/SAMPLE config (metres); used by the self-check."""
    f = np.asarray(cfg.custom_fdet_vector)
    s = np.asarray(cfg.custom_sdet_vector)
    o = np.asarray(cfg.custom_odet_vector)
    return (
        -cfg.beam_center_f / 1000.0 * f
        - cfg.beam_center_s / 1000.0 * s
        + cfg.close_distance_mm / 1000.0 * o
    )


# --------------------------------------------------------------------------- #
# crystal
# --------------------------------------------------------------------------- #
def crystal_config_from_A(
    unit_cell: Sequence[float],
    A,
    *,
    Ncells_abc: Union[int, Sequence[int]] = (10, 10, 10),
    shape="square",
    mosaic_spread_deg: float = 0.0,
    mosaic_domains: int = 1,
    mosaic_seed: Optional[int] = None,
    default_F: float = 0.0,
    fudge: float = 1.0,
    misset_deg: Vec3 = (0.0, 0.0, 0.0),
) -> CrystalConfig:
    """
    CrystalConfig from unit-cell parameters (Å, deg) and a dxtbx-style A matrix.

    ``A`` is the 3x3 matrix with q = A·(h,k,l) in Å⁻¹, i.e. its *columns* are the
    reciprocal basis vectors a*, b*, c* (dxtbx ``Crystal.get_A()`` flattened row
    major). Its transpose is what cctbx passes to ``nanoBragg.Amatrix``. A torch
    tensor (possibly requiring grad) is accepted so orientation gradients flow.
    """
    if isinstance(A, torch.Tensor):
        A_t = A.reshape(3, 3)
        a_star, b_star, c_star = A_t[:, 0], A_t[:, 1], A_t[:, 2]
    else:
        A_np = np.asarray(A, dtype=np.float64).reshape(3, 3)
        a_star, b_star, c_star = A_np[:, 0].copy(), A_np[:, 1].copy(), A_np[:, 2].copy()

    if isinstance(Ncells_abc, (int, float)):
        Ncells_abc = (int(Ncells_abc),) * 3
    Ncells_abc = tuple(int(n) for n in Ncells_abc)
    if len(Ncells_abc) == 1:
        Ncells_abc = Ncells_abc * 3

    a, b, c, al, be, ga = (float(x) for x in unit_cell)
    return CrystalConfig(
        cell_a=a, cell_b=b, cell_c=c, cell_alpha=al, cell_beta=be, cell_gamma=ga,
        misset_deg=tuple(misset_deg),
        mosflm_a_star=a_star, mosflm_b_star=b_star, mosflm_c_star=c_star,
        mosaic_spread_deg=float(mosaic_spread_deg),
        mosaic_domains=int(mosaic_domains),
        mosaic_seed=mosaic_seed,
        N_cells=Ncells_abc,
        default_F=float(default_F),
        shape=_xtal_shape(shape),
        fudge=float(fudge),
    )


def crystal_config_from_dxtbx(crystal, **kwargs) -> CrystalConfig:
    """
    CrystalConfig from a dxtbx Crystal (``get_unit_cell()``, ``get_A()``).

    Keyword arguments are forwarded to :func:`crystal_config_from_A`. Like
    ``Amatrix_dials2nanoBragg`` this refuses non-primitive settings when the
    crystal exposes a space group, because nanoBragg's Fhkl box is P1.
    """
    if hasattr(crystal, "get_space_group"):
        try:
            cb_op = crystal.get_space_group().info().change_of_basis_op_to_primitive_setting()
            if not cb_op.is_identity_op():
                raise ValueError(
                    "convert the dxtbx crystal to its primitive setting first "
                    "(cctbx: crystal.change_basis(cb_op))"
                )
        except AttributeError:
            pass  # duck-typed crystal without a real sgtbx space group
    cell = tuple(crystal.get_unit_cell().parameters())
    return crystal_config_from_A(cell, crystal.get_A(), **kwargs)


def rotate_A(A, rotx_deg=0.0, roty_deg=0.0, rotz_deg=0.0) -> torch.Tensor:
    """
    Apply diffBragg's RotX·RotY·RotZ small-rotation convention to an A matrix:
    A' = Rx(θx)·Ry(θy)·Rz(θz)·A, with rotations about the lab axes. Angles may
    be tensors requiring grad; the result is a (3,3) tensor.
    """
    A_t = torch.as_tensor(np.asarray(A, dtype=np.float64).reshape(3, 3)) if not isinstance(A, torch.Tensor) else A.reshape(3, 3)
    dtype = A_t.dtype
    angles = [torch.as_tensor(x, dtype=dtype) for x in (rotx_deg, roty_deg, rotz_deg)]
    tx, ty, tz = (torch.deg2rad(x) for x in angles)
    one, zero = torch.ones((), dtype=dtype), torch.zeros((), dtype=dtype)
    Rx = torch.stack([torch.stack([one, zero, zero]),
                      torch.stack([zero, torch.cos(tx), -torch.sin(tx)]),
                      torch.stack([zero, torch.sin(tx), torch.cos(tx)])])
    Ry = torch.stack([torch.stack([torch.cos(ty), zero, torch.sin(ty)]),
                      torch.stack([zero, one, zero]),
                      torch.stack([-torch.sin(ty), zero, torch.cos(ty)])])
    Rz = torch.stack([torch.stack([torch.cos(tz), -torch.sin(tz), zero]),
                      torch.stack([torch.sin(tz), torch.cos(tz), zero]),
                      torch.stack([zero, zero, one])])
    return Rx @ Ry @ Rz @ A_t.to(dtype)


# --------------------------------------------------------------------------- #
# structure factors
# --------------------------------------------------------------------------- #
def structure_factors_from_miller_array(miller_array) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a cctbx miller array into (indices, amplitudes) the way NBcrystal does:
    expand to P1, generate Bijvoet mates, take amplitudes if the data are complex.
    Duck-typed arrays only need ``.indices()`` and ``.data()``.
    """
    ma = miller_array
    if hasattr(ma, "expand_to_p1"):
        ma = ma.expand_to_p1()
    if hasattr(ma, "generate_bijvoet_mates"):
        ma = ma.generate_bijvoet_mates()
    if hasattr(ma, "is_complex_array") and ma.is_complex_array():
        ma = ma.amplitudes()
    idx = np.asarray([tuple(h) for h in ma.indices()], dtype=np.int64).reshape(-1, 3)
    amp = np.asarray(list(ma.data()), dtype=np.float64)
    return idx, amp


def set_structure_factors(
    crystal: Crystal,
    indices,
    amplitudes,
    default_F: Optional[float] = None,
) -> None:
    """
    Load a P1 (h,k,l,F) list into ``crystal.hkl_data`` as the dense
    [h-h_min][k-k_min][l-l_min] box nanoBragg uses. Missing reflections take
    ``default_F`` (defaults to the crystal config's default_F). No symmetry
    expansion is applied; do that upstream (see structure_factors_from_miller_array).
    """
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    amp = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if idx.shape[0] != amp.shape[0]:
        raise ValueError("indices and amplitudes must have the same length")
    if default_F is None:
        default_F = float(crystal.config.default_F)
    if idx.shape[0] == 0:
        crystal.hkl_data = None
        crystal.hkl_metadata = None
        return
    hmin, kmin, lmin = (int(x) for x in idx.min(axis=0))
    hmax, kmax, lmax = (int(x) for x in idx.max(axis=0))
    grid = np.full((hmax - hmin + 1, kmax - kmin + 1, lmax - lmin + 1), default_F, dtype=np.float64)
    grid[idx[:, 0] - hmin, idx[:, 1] - kmin, idx[:, 2] - lmin] = amp
    crystal.hkl_data = torch.as_tensor(grid, device=crystal.device, dtype=crystal.dtype)
    crystal.hkl_metadata = {
        "h_min": hmin, "h_max": hmax, "k_min": kmin, "k_max": kmax, "l_min": lmin, "l_max": lmax,
    }


# --------------------------------------------------------------------------- #
# mosaicity
# --------------------------------------------------------------------------- #
def _axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit(axis)
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def isotropic_umats(mos_spread_deg: float, n_domains: int, seed: int = 777, isotropic: bool = True) -> torch.Tensor:
    """
    NumPy port of ``SimData.Umats`` (the "double_random legacy method"): for each
    of ``n_domains`` draws, a random axis on the sphere and a Gaussian angle with
    sigma = mos_spread (deg → rad); with ``isotropic`` the −angle partner is added,
    giving 2·n_domains matrices. Uses numpy's RNG, so it is statistically but not
    bitwise identical to the scitbx generator.

    Returns a (M, 3, 3) float64 tensor.
    """
    rng = np.random.RandomState(seed)
    sigma = math.radians(mos_spread_deg)
    angles = rng.normal(0.0, sigma, size=n_domains) if mos_spread_deg > 0 else np.zeros(n_domains)
    mats = []
    for ang in angles:
        axis = rng.normal(size=3)
        mats.append(_axis_angle_matrix(axis, float(ang) if mos_spread_deg > 0 else 0.0))
        if isotropic and mos_spread_deg > 0:
            mats.append(_axis_angle_matrix(axis, -float(ang)))
    return torch.as_tensor(np.stack(mats), dtype=torch.float64)


def umats_from_cctbx(umat_nm) -> torch.Tensor:
    """Convert a flex.mat3_double (or any iterable of 9-tuples / 3x3) to (M,3,3)."""
    mats = [np.asarray(list(m) if not hasattr(m, "elems") else m.elems, dtype=np.float64).reshape(3, 3) for m in umat_nm]
    return torch.as_tensor(np.stack(mats), dtype=torch.float64)


def set_mosaic_blocks(crystal: Crystal, umats) -> None:
    """
    Use explicit mosaic rotation matrices, like ``nanoBragg.set_mosaic_blocks``.
    ``umats`` is (M,3,3). The crystal's ``mosaic_domains`` is set to M so the
    C-style ``steps`` normalisation stays consistent; ``mosaic_spread_deg`` is
    left untouched (it is informational once blocks are explicit).
    """
    u = torch.as_tensor(umats, dtype=crystal.dtype, device=crystal.device)
    if u.ndim != 3 or u.shape[-2:] != (3, 3):
        raise ValueError("umats must have shape (M, 3, 3)")
    crystal.mosaic_umats_override = u
    crystal.config.mosaic_domains = int(u.shape[0])


# --------------------------------------------------------------------------- #
# beam
# --------------------------------------------------------------------------- #
def beam_config_from_dxtbx(
    beam,
    *,
    spectrum: Optional[Iterable[Tuple[float, float]]] = None,
    fluence: Optional[float] = None,
    flux: Optional[float] = None,
    exposure_s: Optional[float] = None,
    beamsize_mm: Optional[float] = None,
    spot_scale: float = 1.0,
) -> BeamConfig:
    """
    BeamConfig from a dxtbx Beam: wavelength, polarization fraction (used as the
    Kahn factor, as nanoBragg.cpp does) and polarization normal.

    ``spectrum`` is an optional list of (wavelength_Å, flux) pairs like
    ``NBbeam.spectrum``; each entry becomes a source along −unit_s0. Note that
    nanoBragg (C and cctbx CPU kernel) weights sources equally regardless of
    flux, and so does the torch simulator; the fluxes are kept in
    ``source_weights`` for reference only.
    """
    wavelength_A = float(beam.get_wavelength())
    kwargs = dict(wavelength_A=wavelength_A, spot_scale=float(spot_scale))
    if hasattr(beam, "get_polarization_fraction"):
        frac = float(beam.get_polarization_fraction())
        kwargs["polarization_factor"] = frac
        kwargs["nopolar"] = not (0.0 < frac <= 1.0)
    if hasattr(beam, "get_polarization_normal"):
        kwargs["polarization_axis"] = _as_tuple3(_unit(beam.get_polarization_normal()))
    if fluence is not None:
        kwargs["fluence"] = float(fluence)
    if flux is not None:
        kwargs["flux"] = float(flux)
    if exposure_s is not None:
        kwargs["exposure"] = float(exposure_s)
    if beamsize_mm is not None:
        kwargs["beamsize_mm"] = float(beamsize_mm)

    if spectrum is not None:
        spec = [(float(w), float(f)) for w, f in spectrum]
        if spec:
            s0 = _unit(np.asarray(beam.get_s0() if hasattr(beam, "get_s0") else beam.get_unit_s0(), dtype=np.float64))
            n = len(spec)
            kwargs["source_directions"] = torch.as_tensor(np.tile(-s0, (n, 1)), dtype=torch.float64)
            kwargs["source_wavelengths"] = torch.as_tensor([w * 1e-10 for w, _ in spec], dtype=torch.float64)
            kwargs["source_weights"] = torch.as_tensor([f for _, f in spec], dtype=torch.float64)
    return BeamConfig(**kwargs)


# --------------------------------------------------------------------------- #
# simulators
# --------------------------------------------------------------------------- #
def _s0_of(beam) -> np.ndarray:
    if hasattr(beam, "get_s0"):
        return np.asarray(beam.get_s0(), dtype=np.float64)
    return np.asarray(beam.get_unit_s0(), dtype=np.float64)


def simulator_from_dxtbx(
    detector,
    beam,
    crystal,
    *,
    panel_id: int = 0,
    Ncells_abc=(10, 10, 10),
    shape="square",
    mosaic_spread_deg: float = 0.0,
    mosaic_domains: int = 1,
    umats=None,
    miller_array=None,
    structure_factors: Optional[Tuple] = None,
    default_F: float = 0.0,
    oversample: int = -1,
    fluence: Optional[float] = None,
    spot_scale: float = 1.0,
    device=None,
    dtype=torch.float64,
) -> Simulator:
    """
    Build a Simulator for one dxtbx panel from dxtbx Detector, Beam and Crystal.

    ``structure_factors`` is an (indices, amplitudes) pair for P1 data; or pass a
    cctbx ``miller_array`` and it is expanded with
    :func:`structure_factors_from_miller_array`. ``umats`` (M,3,3) uses explicit
    mosaic blocks instead of sampling.
    """
    panel = detector[panel_id] if hasattr(detector, "__getitem__") else detector
    s0 = _s0_of(beam)
    det_cfg = detector_config_from_dxtbx_panel(panel, s0, oversample=oversample)
    cry_cfg = crystal_config_from_dxtbx(
        crystal, Ncells_abc=Ncells_abc, shape=shape, mosaic_spread_deg=mosaic_spread_deg,
        mosaic_domains=mosaic_domains, default_F=default_F,
    )
    beam_cfg = beam_config_from_dxtbx(beam, fluence=fluence, spot_scale=spot_scale)

    crystal_model = Crystal(cry_cfg, beam_config=beam_cfg, device=device, dtype=dtype)
    if miller_array is not None:
        structure_factors = structure_factors_from_miller_array(miller_array)
    if structure_factors is not None:
        set_structure_factors(crystal_model, *structure_factors, default_F=default_F)
    if umats is not None:
        set_mosaic_blocks(crystal_model, umats)
    detector_model = Detector(det_cfg, device=device, dtype=dtype)
    return Simulator(crystal_model, detector_model, crystal_config=cry_cfg, beam_config=beam_cfg,
                     device=device, dtype=dtype)


def simulator_from_sim_data(SIM, *, panel_id: Optional[int] = None, device=None, dtype=torch.float64, **overrides) -> Simulator:
    """
    Build a Simulator from a ``simtbx.nanoBragg.sim_data.SimData`` instance,
    reading the same members ``SimData.instantiate_nanoBragg`` uses:
    ``SIM.detector``, ``SIM.beam.nanoBragg_constructor_beam``, ``SIM.crystal``
    (NBcrystal: dxtbx_crystal, Ncells_abc, xtal_shape, mos_spread_deg,
    n_mos_domains, miller_array) and, if ``SIM.D`` exists, its ``default_F``,
    ``oversample``, ``fluence`` and ``spot_scale``. Keyword overrides win.
    """
    nb = SIM.crystal
    beam = SIM.beam.nanoBragg_constructor_beam if hasattr(SIM.beam, "nanoBragg_constructor_beam") else SIM.beam
    kw = dict(
        panel_id=int(SIM.panel_id if panel_id is None and hasattr(SIM, "panel_id") else (panel_id or 0)),
        Ncells_abc=nb.Ncells_abc,
        shape=nb.xtal_shape,
        mosaic_spread_deg=(nb.mos_spread_deg if nb.n_mos_domains > 1 else 0.0),
        mosaic_domains=(nb.n_mos_domains if nb.n_mos_domains > 1 else 1),
        miller_array=nb.miller_array,
    )
    D = getattr(SIM, "D", None)
    if D is not None:
        kw["default_F"] = float(D.default_F)
        try:
            kw["oversample"] = int(D.oversample)
        except Exception:
            pass
        try:
            kw["fluence"] = float(D.fluence)
        except Exception:
            pass
        try:
            kw["spot_scale"] = float(D.spot_scale)
        except Exception:
            pass
        try:
            umats = D.get_mosaic_blocks()
            if len(umats) > 1:
                kw["umats"] = umats_from_cctbx(umats)
        except Exception:
            pass
    kw.update(overrides)
    return simulator_from_dxtbx(SIM.detector, beam, nb.dxtbx_crystal, device=device, dtype=dtype, **kw)


class MultiPanelSimulator:
    """
    Loop over the panels of a dxtbx Detector the way cctbx does with
    ``panel_id`` / ``set_dxtbx_detector_panel``: one Simulator per panel sharing
    a single Crystal, so crystal parameters (and their gradients) are shared.

    ``run()`` returns a tensor of shape (n_panels, slow, fast).
    """

    def __init__(self, detector, beam, crystal, *, device=None, dtype=torch.float64, **kwargs):
        self.panels: List[Simulator] = []
        first = simulator_from_dxtbx(detector, beam, crystal, panel_id=0, device=device, dtype=dtype, **kwargs)
        self.panels.append(first)
        s0 = _s0_of(beam)
        oversample = kwargs.get("oversample", -1)
        for pid in range(1, len(detector)):
            det_cfg = detector_config_from_dxtbx_panel(detector[pid], s0, oversample=oversample)
            det = Detector(det_cfg, device=first.device, dtype=dtype)
            self.panels.append(Simulator(first.crystal, det, crystal_config=first.crystal.config,
                                         beam_config=first.beam_config, device=first.device, dtype=dtype))

    @property
    def crystal(self) -> Crystal:
        return self.panels[0].crystal

    def run(self, **kwargs) -> torch.Tensor:
        return torch.stack([sim.run(**kwargs) for sim in self.panels], dim=0)


def to_raw_pixels(image: torch.Tensor):
    """
    Return the image as nanoBragg's ``raw_pixels`` would present it: a slow-major
    float64 array. Returns a ``scitbx.array_family.flex.double`` when cctbx is
    importable, else a NumPy array.
    """
    arr = np.ascontiguousarray(image.detach().cpu().numpy().astype(np.float64))
    try:
        from scitbx.array_family import flex  # type: ignore
        return flex.double(arr)
    except Exception:
        return arr
