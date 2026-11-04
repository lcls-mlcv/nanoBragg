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
    RotXYZ       the misset (radians), restrained to 0; hopper ignores centers.RotXYZ and so
                 does this. hopper rotates about the lab axes on top of the orientation; the
                 misset equals that to first order only from a zero base misset (a crystal
                 given by its orientation matrix, as from dxtbx), so anything else is refused.
    detz_shift   the origin shift along z (metres)
    spec         (lambda_offset in Angstrom, lambda_scale)
    Nvol         det NABC: ½ (c - det)²/b, without the log term, and whether or not Ncells
                 refines, as hopper does

A SerialExperiment restrains each shot, and a parameter its shots share once.

Differences from hopper: calling this is hopper's use_restraints=True (hopper ignores centers
and betas without it). hopper's has_isotropic_ncells restrains Na only; Ncells here are always
three. Not here, and refused by name: Fhkl, Finit, Friedel, B, diffuse_gamma, diffuse_sigma,
the deprecated betas.ucell, and hopper's per-ROI scale restraint (center 1, beta 1e12).
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from .unit_cell import FREE_VARIABLES

Number = Union[float, Sequence[float]]

_ANGLES = ("alpha", "beta", "gamma")
_NAMES = ("G", "Nabc", "Ndef", "eta_abc", "RotXYZ", "detz_shift", "spec", "Nvol") + tuple(
    f"ucell_{v}" for v in ("a", "b", "c") + _ANGLES)
_DEGREE_CENTERS = tuple(f"ucell_{a}" for a in _ANGLES)


def _vector(values, like: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.as_tensor(v, dtype=like.dtype, device=like.device).reshape(()) for v in values])


def _values(shot, names) -> Dict[str, List[Tuple[Optional[torch.Tensor], torch.Tensor]]]:
    """name -> [(parameter, its value in hopper's units)] for one ExperimentModel.

    Values come from the blocks' build_config, the configs the simulation runs on. Nvol has
    no parameter of its own.
    """
    crystal, beam, detector = shot.crystal_params, shot.beam_params, shot.detector_params
    c, b = crystal.build_config(), beam.build_config()
    like = crystal.delta_log_N_cells
    n, nd = _vector(c.N_cells, like), _vector(c.N_cells_def, like)
    values = {
        "G": [(beam.delta_log_spot_scale, _vector([b.spot_scale], like))],
        "Nabc": [(crystal.delta_log_N_cells, n)],
        "Ndef": [(crystal.delta_N_cells_def, nd)],
        "eta_abc": [(crystal.delta_log_eta, crystal.eta())],
        "spec": [(beam.lambda_offset, beam.lambda_offset.reshape(1)),
                 (beam.delta_log_lambda_scale, torch.exp(beam.delta_log_lambda_scale).reshape(1))],
        "Nvol": [(None, torch.linalg.det(torch.stack([torch.stack([n[0], nd[0], nd[2]]),
                                                      torch.stack([nd[0], n[1], nd[1]]),
                                                      torch.stack([nd[2], nd[1], n[2]])])).reshape(1))],
    }
    if "RotXYZ" in names and crystal.delta_misset_raw.requires_grad:
        base = _vector([crystal.base_misset_x, crystal.base_misset_y, crystal.base_misset_z], like)
        if bool((base != 0).any()):
            raise ValueError(
                "RotXYZ restrains rotations about the lab axes on top of the orientation, as hopper's are; "
                "the misset equals those only from a zero base misset (a crystal given by its orientation "
                "matrix, as from dxtbx). Leave betas.RotXYZ out for this crystal.")
        values["RotXYZ"] = [(crystal.delta_misset_raw, torch.deg2rad(_vector(c.misset_deg, like)))]
    if "detz_shift" in names and detector.origin_defined:
        shift = detector.build_config().origin_shift_z_mm
        values["detz_shift"] = [(detector.delta_detz_shift_mm, _vector([shift], like) * 1e-3)]
    free = FREE_VARIABLES[crystal.crystal_system]
    for var, pname in crystal._CELL_PARAMS.items():
        if var in free:
            value = _vector([getattr(c, f"cell_{var}")], like)
            values[f"ucell_{var}"] = [(getattr(crystal, pname), torch.deg2rad(value) if var in _ANGLES else value)]
    return values


def _entries(x: Number, count: int, what: str) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    if x.numel() == 1:
        return x.expand(count)
    if x.numel() < count:
        raise ValueError(f"{what} needs {count} entries, got {x.numel()}")
    return x[:count]


def hopper_restraints(model, centers: Dict[str, Number], betas: Dict[str, Number]) -> torch.Tensor:
    """Sum of hopper's restraint terms on ``model``, in the model's dtype.

    Args:
        model: ExperimentModel or SerialExperiment.
        centers: Restraint targets by hopper's ``centers`` names (module docstring).
        betas: Restraint variances by hopper's ``betas`` names; RotXYZ needs no center.

    Add it to the likelihood inside the objective:

        refine_lbfgsb(model, lambda: gaussian_nll(model(roi_only=True), data, sigma)
                                     + hopper_restraints(model, centers, betas))
    """
    centers, betas = dict(centers or {}), dict(betas or {})
    unknown = sorted((set(centers) | set(betas)) - set(_NAMES))
    if unknown:
        raise ValueError(f"no hopper restraint named {', '.join(unknown)}; known: {', '.join(_NAMES)}")
    centers.pop("RotXYZ", None)  # hopper restrains RotXYZ to 0 and ignores its center
    unpaired = sorted(set(centers) ^ (set(betas) - {"RotXYZ"}))
    if unpaired:
        raise ValueError(f"restraining {', '.join(unpaired)} needs both a center and a beta, as hopper requires")

    shots = list(getattr(model, "shots", [model]))
    first = next(iter(shots[0].parameters()))
    total = torch.zeros((), dtype=first.dtype, device=first.device)
    seen = set()
    for shot in shots:
        if not betas:
            break
        values = _values(shot, betas)
        for name in betas:
            if name == "Nvol":  # hopper: whether or not Ncells refines, and no log term
                key = (name, id(shot.crystal_params))
                if key not in seen:
                    seen.add(key)
                    total = total + 0.5 * (float(centers[name]) - values[name][0][1][0]) ** 2 / float(betas[name])
                continue
            # spec's two entries are two parameters; every other name is one parameter
            end = 0
            for parameter, value in values.get(name, []):
                start, end = end, end + value.numel()
                key = (name, id(parameter))
                if not parameter.requires_grad or key in seen:
                    continue
                seen.add(key)
                beta = _entries(betas[name], end, f"betas.{name}")[start:].to(value)
                if name == "RotXYZ":
                    center = torch.zeros_like(value)
                else:
                    center = _entries(centers[name], end, f"centers.{name}")[start:].to(value)
                    if name in _DEGREE_CENTERS:
                        center = torch.deg2rad(center)
                total = total + (0.5 * (torch.log(2 * math.pi * beta) + (center - value) ** 2 / beta)).sum()
    return total
