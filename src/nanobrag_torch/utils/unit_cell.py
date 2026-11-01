"""Unit cells constrained by crystal system, and the cctbx B matrix, in torch.

A crystal system fixes some cell angles and ties some lengths together, so a
refinement should move only its free variables:

    triclinic     a b c alpha beta gamma
    monoclinic    a b c beta              (unique axis b; alpha = gamma = 90)
    orthorhombic  a b c                   (all angles 90)
    tetragonal    a c                     (b = a; all angles 90)
    hexagonal     a c                     (b = a; alpha = beta = 90, gamma = 120;
                                           trigonal groups in hexagonal axes too)
    rhombohedral  a alpha                 (b = c = a; beta = gamma = alpha;
                                           R groups in rhombohedral axes)
    cubic         a                       (b = c = a; all angles 90)

diffBragg's crystal-system managers (simtbx/diffBragg/refiners/crystal_systems)
cover monoclinic, orthorhombic, tetragonal and hexagonal with the same variables.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

CELL_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")

FREE_VARIABLES = {
    "triclinic": ("a", "b", "c", "alpha", "beta", "gamma"),
    "monoclinic": ("a", "b", "c", "beta"),
    "orthorhombic": ("a", "b", "c"),
    "tetragonal": ("a", "c"),
    "hexagonal": ("a", "c"),
    "rhombohedral": ("a", "alpha"),
    "cubic": ("a",),
}
# trigonal space groups in hexagonal axes refine like hexagonal ones
_ALIASES = {"trigonal": "hexagonal"}

# Each cell parameter is a free variable, a copy of one, or a fixed angle (degrees).
_TIES = {
    "triclinic": {},
    "monoclinic": {"alpha": 90.0, "gamma": 90.0},
    "orthorhombic": {"alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    "tetragonal": {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    "hexagonal": {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 120.0},
    "rhombohedral": {"b": "a", "c": "a", "beta": "alpha", "gamma": "alpha"},
    "cubic": {"b": "a", "c": "a", "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
}


def normalize_crystal_system(name: str) -> str:
    key = _ALIASES.get(str(name).lower(), str(name).lower())
    if key not in FREE_VARIABLES:
        raise ValueError(
            f"unknown crystal system {name!r}; expected one of {sorted(FREE_VARIABLES) + sorted(_ALIASES)}"
        )
    return key


def check_cell(system: str, cell: Sequence[float], angle_tol_deg: float = 1e-3, length_rtol: float = 1e-6) -> None:
    """Raise ValueError if the cell (a, b, c, alpha, beta, gamma) breaks the system's constraints."""
    system = normalize_crystal_system(system)
    values = dict(zip(CELL_NAMES, (float(x) for x in cell)))
    for name, tie in _TIES[system].items():
        if isinstance(tie, str):
            is_length = name in ("a", "b", "c")
            ok = math.isclose(values[name], values[tie], rel_tol=length_rtol if is_length else 0.0,
                              abs_tol=0.0 if is_length else angle_tol_deg)
            want = f"{tie} = {values[tie]:g}"
        else:
            ok = abs(values[name] - tie) <= angle_tol_deg
            want = f"{tie:g}"
        if not ok:
            raise ValueError(
                f"cell {tuple(values.values())} is not {system}: {name} = {values[name]:g}, expected {want}"
            )


def constrained_cell(system: str, free: dict, fixed: dict) -> tuple:
    """Six cell parameters from the free variables (``free``) and the system's ties.

    ``fixed`` supplies the base value of every cell parameter; ties to a fixed angle
    use the constant from the crystal system rather than the base value.
    """
    system = normalize_crystal_system(system)
    ties = _TIES[system]
    cell = {}
    for name in CELL_NAMES:
        tie = ties.get(name)
        if name in free:
            cell[name] = free[name]
        elif isinstance(tie, str):
            cell[name] = free[tie]
        elif tie is not None:
            cell[name] = tie
        else:
            cell[name] = fixed[name]
    return tuple(cell[n] for n in CELL_NAMES)


def B_matrix(a, b, c, alpha, beta, gamma, *, dtype=torch.float64, device=None) -> torch.Tensor:
    """cctbx/dxtbx B matrix (angles in degrees): columns are a*, b*, c* in Å⁻¹.

    B is the transpose of cctbx's fractionalization matrix, the inverse of the
    orthogonalization matrix that puts a along x and b in the xy plane. A dxtbx
    crystal has A = U·B, so holding U fixed while the cell changes is diffBragg's
    unit-cell derivative. Accepts tensors requiring grad.
    """
    params = [x for x in (a, b, c, alpha, beta, gamma) if isinstance(x, torch.Tensor)]
    if params:
        device = params[0].device if device is None else device
        dtype = params[0].dtype
    a, b, c, alpha, beta, gamma = (torch.as_tensor(x, dtype=dtype, device=device) for x in (a, b, c, alpha, beta, gamma))
    ca, cb, cg = torch.cos(torch.deg2rad(alpha)), torch.cos(torch.deg2rad(beta)), torch.cos(torch.deg2rad(gamma))
    sg = torch.sin(torch.deg2rad(gamma))
    volume = a * b * c * torch.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    zero = torch.zeros_like(a)
    orthogonalization = torch.stack([
        torch.stack([a, b * cg, c * cb]),
        torch.stack([zero, b * sg, c * (ca - cb * cg) / sg]),
        torch.stack([zero, zero, volume / (a * b * sg)]),
    ])
    eye = torch.eye(3, dtype=dtype, device=device)
    fractionalization = torch.linalg.solve_triangular(orthogonalization, eye, upper=True)
    return fractionalization.T
