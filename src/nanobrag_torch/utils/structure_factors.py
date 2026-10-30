"""
Structure factors held per asymmetric-unit reflection and expanded to nanoBragg's P1 grid.

nanoBragg looks F up in a dense [h-h_min][k-k_min][l-l_min] box. Refining F means
refining one amplitude per unique reflection, then filling every symmetry mate in
the box from it. The expansion is fixed by the space group, so it is computed once
here, as an index from every box entry into the vector

    [F_asu_0, ..., F_asu_{M-1}, default_F, F000]

and each step the box is one gather, ``values[index]``, through which gradients
reach the M amplitudes.

The symmetry is given as the point group's rotation matrices R (cctbx
``space_group.all_ops()[i].r()``): the mates of h are h·R. Without
``anomalous`` the mates also include -h·R (Friedel's law), so a Friedel pair is
one amplitude; with it they are separate unless the reflection is centric. This
is cctbx's anomalous_flag, and diffBragg's ASU map uses anomalous_flag=True.
No cctbx is needed here; compat/cctbx.py builds a StructureFactors from a
miller array.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def _rotations(rotations) -> np.ndarray:
    R = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    Ri = np.rint(R).astype(np.int64)
    if not np.array_equal(Ri, R):
        raise ValueError("reciprocal-space symmetry rotations must be integer matrices")
    return np.unique(Ri, axis=0)


def expand_to_p1(asu_indices, rotations=None, anomalous: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Every symmetry mate of every reflection, and which input reflection it came from.

    Returns (p1_indices (N, 3), source (N,)) with each P1 index once. Raises
    ValueError if two input reflections are symmetry mates of each other.
    """
    hkl = np.asarray(asu_indices, dtype=np.int64).reshape(-1, 3)
    R = _rotations(np.eye(3) if rotations is None else rotations)
    if not anomalous:
        R = np.unique(np.concatenate([R, -R]), axis=0)
    mates = np.einsum("ni,rij->nrj", hkl, R).reshape(-1, 3)
    source = np.repeat(np.arange(hkl.shape[0]), R.shape[0])
    p1, first, inverse = np.unique(mates, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    lo = np.full(p1.shape[0], np.iinfo(np.int64).max)
    hi = np.full(p1.shape[0], -1)
    np.minimum.at(lo, inverse, source)
    np.maximum.at(hi, inverse, source)
    clash = np.flatnonzero(lo != hi)
    if clash.size:
        i, j = lo[clash[0]], hi[clash[0]]
        raise ValueError(
            f"reflections {tuple(hkl[i])} and {tuple(hkl[j])} are symmetry mates; "
            "give one amplitude per unique reflection"
        )
    return p1, source[first]


@dataclass
class StructureFactors:
    """Amplitudes of unique reflections plus the symmetry that generates the rest.

    ``indices`` (M, 3) and ``amplitudes`` (M,) list one reflection per symmetry
    orbit (any representative). ``rotations`` are the point group's integer
    reciprocal-space rotations (default: P1, identity only). Reflections in the
    P1 box but not generated here take ``default_F`` (default: the crystal
    config's); (0,0,0) takes ``F000`` (cctbx's Fhkl_tuple default 0; None keeps
    the listed or default value). Only the M listed amplitudes are refinable.
    """

    indices: np.ndarray
    amplitudes: np.ndarray
    rotations: Optional[np.ndarray] = None
    anomalous: bool = False
    default_F: Optional[float] = None
    F000: Optional[float] = 0.0
    _p1: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1, 3)
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.float64).reshape(-1)
        if self.indices.shape[0] != self.amplitudes.shape[0]:
            raise ValueError("indices and amplitudes must have the same length")
        if self.indices.shape[0] == 0:
            raise ValueError("StructureFactors needs at least one reflection")

    def __len__(self) -> int:
        return self.indices.shape[0]

    def p1(self) -> Tuple[np.ndarray, np.ndarray]:
        """(P1 indices, index of the unique reflection each one takes its amplitude from)."""
        if self._p1 is None:
            self._p1 = expand_to_p1(self.indices, self.rotations, self.anomalous)
        return self._p1

    def grid_index(self, default_F: float) -> Tuple[np.ndarray, dict, np.ndarray]:
        """The P1 box as indices into [F_0 .. F_{M-1}, default_F, F000].

        Returns (index (H, K, L) int64, hkl_metadata, constants [default_F, F000]).
        """
        p1, source = self.p1()
        lo, hi = p1.min(axis=0), p1.max(axis=0)
        M = len(self)
        index = np.full(tuple(hi - lo + 1), M, dtype=np.int64)
        index[tuple((p1 - lo).T)] = source
        F000 = default_F if self.F000 is None else float(self.F000)
        if self.F000 is not None and np.all(lo <= 0) and np.all(hi >= 0):
            index[tuple(-lo)] = M + 1
        metadata = {
            "h_min": int(lo[0]), "h_max": int(hi[0]), "k_min": int(lo[1]), "k_max": int(hi[1]),
            "l_min": int(lo[2]), "l_max": int(hi[2]),
        }
        return index, metadata, np.array([default_F, F000], dtype=np.float64)
