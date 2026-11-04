"""hopper's restraints on refined parameters, for adding to a likelihood.

diffBragg's hopper (simtbx/diffBragg/hopper_utils.py, refiners/parameters.py) adds, for each
refined parameter with a restraint center c and variance beta b,

    ½ (log 2πb + (c - v)²/b),

and nothing for a fixed parameter. ``centers`` and ``betas`` take hopper's phil names and units,
so values carry over from a hopper run:

    G            spot_scale
    Nabc         Na, Nb, Nc (cells)
    Ndef         Nd, Ne, Nf (cells)
    eta_abc      mosaic spread (degrees); an isotropic eta takes the first entry
    ucell_a/b/c  cell lengths (Angstrom)
    ucell_alpha/beta/gamma
                 cell angles: centers in degrees, betas and values in radians, as hopper
                 converts them. Only the crystal system's free variables are restrained.
    RotXYZ       the misset offsets from the start (radians), restrained to 0: hopper takes
                 betas.RotXYZ and no center. Here they are offsets of the misset angles, which
                 equal hopper's rotations to first order.
    detz_shift   the origin shift along z (metres)
    spec         (lambda_offset in Angstrom, lambda_scale)
    Nvol         det NABC: ½ (c - det)²/b, without the log term, and whether or not Ncells
                 refines, as hopper does

A SerialExperiment restrains each shot, and a parameter its shots share once. hopper's Fhkl,
B-factor and diffuse restraints are not here.
"""
import math
from typing import Callable, Dict, List, Sequence, Tuple, Union

import torch

from .unit_cell import FREE_VARIABLES

Number = Union[float, Sequence[float]]

_ANGLES = ("alpha", "beta", "gamma")


def _terms(shot) -> Dict[str, List[Tuple[torch.Tensor, Callable[[], torch.Tensor]]]]:
    """name -> [(parameter, value in hopper's units)] for one ExperimentModel."""
    crystal, beam, detector = shot.crystal_params, shot.beam_params, shot.detector_params

    def cell(var):
        def value():
            v = getattr(crystal.build_config(), f"cell_{var}")
            v = torch.as_tensor(v, dtype=crystal.dtype, device=crystal.device).reshape(1)
            return torch.deg2rad(v) if var in _ANGLES else v
        return value

    def detz():
        base = detector._base_config.origin_shift_z_mm
        return ((0.0 if base is None else base) + detector.delta_detz_shift_mm).reshape(1) * 1e-3

    terms = {
        "G": [(beam.delta_log_spot_scale, lambda: (beam.base_spot_scale * torch.exp(beam.delta_log_spot_scale)).reshape(1))],
        "Nabc": [(crystal.delta_log_N_cells, lambda: crystal.base_N_cells * torch.exp(crystal.delta_log_N_cells))],
        "Ndef": [(crystal.delta_N_cells_def, crystal.N_cells_def)],
        "eta_abc": [(crystal.delta_log_eta, crystal.eta)],
        "RotXYZ": [(crystal.delta_misset_raw,
                    lambda: torch.deg2rad(crystal.max_misset_delta_deg * torch.tanh(crystal.delta_misset_raw)))],
        "detz_shift": [(detector.delta_detz_shift_mm, detz)],
        "spec": [(beam.lambda_offset, lambda: beam.lambda_offset.reshape(1)),
                 (beam.delta_log_lambda_scale, lambda: torch.exp(beam.delta_log_lambda_scale).reshape(1))],
    }
    free = FREE_VARIABLES[crystal.crystal_system]
    for var, pname in crystal._CELL_PARAMS.items():
        terms[f"ucell_{var}"] = [(getattr(crystal, pname), cell(var))] if var in free else []
    return terms


_NAMES = ("G", "Nabc", "Ndef", "eta_abc", "RotXYZ", "detz_shift", "spec", "Nvol") + tuple(
    f"ucell_{v}" for v in ("a", "b", "c") + _ANGLES)


def _nvol(shot) -> torch.Tensor:
    crystal = shot.crystal_params
    n = crystal.base_N_cells * torch.exp(crystal.delta_log_N_cells)
    # the off-diagonals the lattice factor uses: none where Ndef is unrefinable (CrystalStageAParams)
    d = crystal.base_N_cells_def if "delta_N_cells_def" in crystal.unrefinable else crystal.N_cells_def()
    return n[0] * n[1] * n[2] - n[0] * d[1] ** 2 - n[1] * d[2] ** 2 - n[2] * d[0] ** 2 + 2 * d[0] * d[1] * d[2]


def _entries(x: Number, count: int, what: str) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    if x.numel() == 1:
        return x.expand(count)
    if x.numel() < count:
        raise ValueError(f"{what} needs {count} entries, got {x.numel()}")
    return x[:count]


def hopper_restraints(model, centers: Dict[str, Number], betas: Dict[str, Number]) -> torch.Tensor:
    """Sum of hopper's restraint terms on ``model`` (ExperimentModel or SerialExperiment).

    Add it to the likelihood inside the objective:

        refine_lbfgsb(model, lambda: gaussian_nll(model(roi_only=True), data, sigma)
                                     + hopper_restraints(model, centers, betas))
    """
    centers, betas = dict(centers or {}), dict(betas or {})
    unknown = sorted((set(centers) | set(betas)) - set(_NAMES))
    if unknown:
        raise ValueError(f"no hopper restraint named {', '.join(unknown)}; known: {', '.join(_NAMES)}")
    if "RotXYZ" in centers:
        raise ValueError("RotXYZ is restrained to 0, as in hopper: give betas.RotXYZ and no center")
    unpaired = sorted(set(centers) ^ (set(betas) - {"RotXYZ"}))
    if unpaired:
        raise ValueError(f"restraining {', '.join(unpaired)} needs both a center and a beta, as hopper requires")

    shots = list(getattr(model, "shots", [model]))
    first = next(iter(shots[0].parameters()))
    total = torch.zeros((), dtype=first.dtype, device=first.device)
    seen = set()
    for shot in shots:
        terms = _terms(shot)
        for name in betas:
            if name == "Nvol":
                key = (name, id(shot.crystal_params))
                if key not in seen:
                    seen.add(key)
                    total = total + 0.5 * (float(centers[name]) - _nvol(shot)) ** 2 / float(betas[name])
                continue
            # spec's two entries are two parameters; every other name is one parameter
            at = 0
            for parameter, value in terms[name]:
                v = value()
                count = v.numel()
                start, at = at, at + count
                key = (name, id(parameter))
                if not parameter.requires_grad or key in seen:
                    continue
                seen.add(key)
                beta = _entries(betas[name], at, f"betas.{name}")[start:].to(v)
                if name == "RotXYZ":
                    center = torch.zeros_like(v)
                else:
                    center = _entries(centers[name], at, f"centers.{name}")[start:].to(v)
                    if name in tuple(f"ucell_{a}" for a in _ANGLES):
                        center = torch.deg2rad(center)
                total = total + (0.5 * (torch.log(2 * math.pi * beta) + (center - v) ** 2 / beta)).sum()
    return total
