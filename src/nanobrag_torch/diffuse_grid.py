"""
A diffuse model held on a regular grid of fractional Miller indices and interpolated trilinearly.

An extension, not part of nanoBragg.c (see nanobrag_torch.diffuse). A physical diffuse model is
expensive per evaluation and need not carry hkl in its graph; a grid of its values, sampled once,
is cheap to evaluate at every pixel sample and differentiable both in hkl (so in the geometry:
orientation, cell, detector, wavelength) and in the values themselves (docs/diffuse_joint_plan.md,
section 2.2).

Nodes sit at hkl = origin + step * (i, j, k), i < shape[0] and so on (the layout of eryx's
HKLGrid, read here from its fields without importing eryx). Two storages:

- P1: one value per node, the (nx, ny, nz) box itself.
- symmetric, with ``point_group`` (the integer reciprocal-space matrices that StructureFactors
  takes as ``rotations``; with Friedel's law unless ``anomalous``): one value per orbit of
  nodes, the Laue ASU, on a box centred on hkl = 0. Node m (hkl = m * step, m integer) has the
  mates m R, so a node's value is found from the largest key of its mates, through the same
  operators (structure_factors.point_group_operators) as StructureFactors' expansion; on a grid
  of step 1 the orbits are those of expand_to_p1 (tests/test_diffuse_grid.py). The operators
  must form a group (checked: generators alone are refused, not closed silently) and must map
  nodes to nodes: steps equal on axes an operator mixes. Every lookup must find its orbit's key
  exactly, or raises RuntimeError.

The argument is ``point_group``, not ``rotations``, because None means something else here:
P1 storage with no operators at all, where StructureFactors' rotations=None is the identity
with Friedel's law (here point_group=np.eye(3)). ``anomalous`` needs a point group.

Outside the box the grid is zero (``outside="zero"``, the default): it is the trilinear
interpolant of its values padded with zeros, so a point less than a step outside sees the
boundary nodes with that step's weights, and the term vanishes continuously. ``outside="error"``
raises ValueError for any point beyond the box, or not finite, instead. In zero mode a point at
+-inf is outside (0) and a NaN gives NaN. A box built for a resolution (``from_model(d_min=...)``)
covers the whole resolution sphere plus a node.

Device and dtype: the result is on the query's device and in its dtype, as for the other models.
The interpolation runs where the values are (a large grid is not copied per call): a query on
another device is moved there, in the values' dtype, and the result moved back.

With a ``core`` (D3b), a cheap closed-form DiffuseModel for the sharp part near the lattice points
that the model supplies (DiffuseModel.core), the grid holds model - core and its value is
core(hkl) + the residual's interpolant. The core is a submodule, so its parameters refine with the
values (group ``diffuse``); the residual was sampled at the core's parameters when built, so a core
that moves far from them should be followed by a rebuild. Measured (plan section 2.2): on a
1/k^2-like core with a k floor of 0.05, a uniform grid needs a step finer than 1/64 (30 GB for
lysozyme) to keep the bias under 0.05 sd at 1e6 photons; with the model's own core, 1/8 to 1/16
(59 to 468 MB) does.

Trilinear interpolation is exact for multilinear functions and its error elsewhere is bounded by
(step^2 / 8) times the second derivatives, per axis. Its gradient in hkl is constant within a cell
and jumps at the cell faces; finite-difference checks must not straddle one.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from .utils.structure_factors import point_group_operators

__all__ = ["DiffuseGrid"]

_CORNERS = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.long)
_LOOKUP_CHUNK = 1 << 16  # nodes per pass of the symmetry lookup: bounds its (n, operators, 3) work
_BUILD_CHUNK = 1 << 19  # nodes per pass when the orbits are listed: 0.2 GB per (n, 16, 3) temporary


def _check_group(ops: np.ndarray) -> None:
    """Refuse an operator set not closed under products (its orbits would not partition the nodes)."""
    members = {R.tobytes() for R in ops}
    for A in ops:
        for B in ops:
            if (A @ B).tobytes() not in members:
                raise ValueError(f"the operators are not a group: {A.tolist()} @ {B.tolist()} = {(A @ B).tolist()} "
                                 "is missing; give the whole point group, not its generators")


def _triple(value, name) -> np.ndarray:
    array = np.asarray(torch.as_tensor(value).detach().cpu().numpy() if torch.is_tensor(value) else value,
                       dtype=np.float64).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, 3)
    if array.size != 3:
        raise ValueError(f"{name} needs 1 or 3 values, got {array.size}")
    return array


class DiffuseGrid(nn.Module):
    """A DiffuseModel whose values sit on a regular hkl grid, interpolated trilinearly (see the module).

    values: the (nx, ny, nz) box of values, or None for zeros (with ``shape``). With ``point_group``
        a box is averaged over the symmetry mates inside it; a 1D tensor is taken as the ASU
        values themselves, in the order of ``value_hkl()``.
    origin, step: hkl of node (0, 0, 0) and the node spacing, per axis (or one number), in Miller
        units. With ``point_group`` the box must be centred: origin = -half * step, shape = 2 half + 1.
    point_group, anomalous: the point group's integer rotations, as StructureFactors' ``rotations``
        and ``anomalous``; None stores P1 (no operators, no Friedel mates; anomalous then refused).
    refinable: values as an nn.Parameter (the group ``diffuse`` in ExperimentModel), else a buffer.
    outside: "zero" (the grid is zero beyond its box) or "error".

    Construct from a model with from_model() and from an (origin, step, values) grid, such as
    eryx's HKLGrid, with from_values() or from_hkl_grid().
    """

    def __init__(self, values, origin, step, *, shape: Optional[Sequence[int]] = None, point_group=None,
                 anomalous: bool = False, refinable: bool = True, outside: str = "zero",
                 dtype: torch.dtype = torch.float64, device=None, core=None):
        super().__init__()
        self.core = core  # an nn.Module core registers as a submodule
        if outside not in ("zero", "error"):
            raise ValueError(f"outside must be 'zero' or 'error', got {outside!r}")
        self.outside = outside
        self._origin = _triple(origin, "origin")
        self._step = _triple(step, "step")
        if not (self._step > 0).all():
            raise ValueError(f"the step must be positive, got {self._step.tolist()}")
        if values is not None:
            values = torch.as_tensor(values)
            if values.is_floating_point():
                dtype = values.dtype
            device = values.device if device is None else device
        if values is not None and values.ndim == 3:
            shape = tuple(values.shape)
        if shape is None:
            raise ValueError("give the values as a box, or their shape")
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError(f"a grid needs at least 2 nodes per axis, got shape {self.shape}")
        if point_group is None and anomalous:
            raise ValueError("anomalous needs a point group: point_group=None stores P1, with no Friedel mates")
        self.symmetric = point_group is not None
        self._half = None
        if self.symmetric:
            self._setup_symmetry(point_group, anomalous, device)
        n_values = int(self._keys.numel()) if self.symmetric else self.shape
        stored = torch.zeros(n_values, dtype=dtype, device=device)
        if values is not None:
            values = values.detach().to(dtype=dtype, device=stored.device)
            if not self.symmetric:
                stored = values.clone()
            elif values.ndim == 1:
                if values.numel() != stored.numel():
                    raise ValueError(f"{values.numel()} ASU values for {stored.numel()} orbits")
                stored = values.clone()
            else:
                stored = self._average_over_mates(values)
        if refinable:
            self.values = nn.Parameter(stored)
        else:
            self.register_buffer("values", stored)
        if core is not None:
            self._setup_core(core)

    def _setup_core(self, core) -> None:
        """Checks a core and records its parameters at build, theta_b (see forward)."""
        if not np.allclose(0.5 / self._step, np.round(0.5 / self._step), rtol=0, atol=1e-9):
            raise ValueError(f"with a core, 1 / (2 step) must be an integer (step {self._step.tolist()}): the "
                             "nearest-lattice-point core has a kink on the half-integer planes, which nodes must lie on")
        if self.symmetric:
            g = torch.Generator().manual_seed(0)
            device = self.values.device
            half = torch.as_tensor(self._half * self._step, dtype=torch.float64)
            hkl = ((torch.rand(256, 3, generator=g, dtype=torch.float64) * 2 - 1) * half).to(device)
            with torch.no_grad():
                reference = torch.as_tensor(core(hkl)).double()
                for R in self._ops.to(torch.float64):
                    mate = torch.as_tensor(core(hkl @ R)).double()
                    if not torch.allclose(mate, reference, rtol=1e-9, atol=1e-12 * float(reference.abs().max())):
                        raise ValueError(f"the core is not invariant under the point group (operator {R.long().tolist()}): "
                                         "the grid stores one residual per orbit, so the core must have its symmetry")
        self._core_reference = ({name: p.detach().clone().cpu() for name, p in core.named_parameters()}
                                if isinstance(core, nn.Module) else {})
        self.core_drift_tolerance = 0.05  # placeholder, to be set on V2 (plan section 2.2)
        self._core_drift_warned = False

    def core_drift(self) -> float:
        """The largest change of any core parameter since the residual was sampled (theta - theta_b, in the
        parameters' own units: for log parameters a relative change)."""
        if self.core is None or not self._core_reference:
            return 0.0
        params = dict(self.core.named_parameters())
        return max(float((params[name].detach().cpu() - ref).abs().max()) for name, ref in self._core_reference.items())

    # -- construction ---------------------------------------------------------------------------

    def _setup_symmetry(self, rotations, anomalous, device):
        half = (np.asarray(self.shape) - 1) / 2
        centred = (np.asarray(self.shape) % 2 == 1) & np.isclose(self._origin, -half * self._step, rtol=0,
                                                                  atol=1e-9 * self._step)
        if not centred.all():
            raise ValueError("a symmetric grid must be centred on hkl = 0: origin = -half * step, "
                             f"shape = 2 half + 1; got origin {self._origin.tolist()}, shape {self.shape}")
        ops = point_group_operators(rotations, anomalous)
        _check_group(ops)
        for R in ops:
            i, j = np.nonzero(R)
            if not np.allclose(self._step[i], self._step[j], rtol=1e-12, atol=0):
                raise ValueError(f"operator {R.tolist()} mixes axes with different steps {self._step.tolist()}: "
                                 "it would not map nodes to nodes")
        self._half = half.astype(np.int64)
        self._bound = int(self._half.max() * np.abs(ops).sum(axis=1).max())
        self._width = 2 * self._bound + 1
        self.register_buffer("_ops", torch.as_tensor(ops, dtype=torch.long, device=device), persistent=False)
        # One key per orbit with a node in the box. A node whose own key is its orbit's key lists it,
        # once; an orbit whose representative lies outside the box (an operator that does not keep
        # the box, as a hexagonal one in a rectangular box) is listed by each of its nodes and
        # made unique at the end. Memory is the orbits, not the nodes.
        half = torch.as_tensor(self._half, device=device)
        parts = []
        total = int(np.prod(self.shape))
        for start in range(0, total, _BUILD_CHUNK):
            ijk = self._flat_to_ijk(torch.arange(start, min(start + _BUILD_CHUNK, total), device=device))
            keys = self._canonical_keys(ijk)
            own = self._encode(ijk - half)
            outside = (self._decode(keys).abs() > half).any(dim=-1)
            parts.append(keys[(keys == own) | outside])
        self.register_buffer("_keys", torch.unique(torch.cat(parts)), persistent=False)

    def _encode(self, m: torch.Tensor) -> torch.Tensor:
        shifted = m + self._bound
        return (shifted[..., 0] * self._width + shifted[..., 1]) * self._width + shifted[..., 2]

    def _decode(self, key: torch.Tensor) -> torch.Tensor:
        w = self._width
        return torch.stack([key // (w * w), (key // w) % w, key % w], dim=-1) - self._bound

    @classmethod
    def from_values(cls, values, origin, step, **kwargs) -> "DiffuseGrid":
        """A grid from a box of values at hkl = origin + step * (i, j, k) (eryx's HKLGrid layout).

        With ``point_group`` the values are averaged over the symmetry mates inside the box, so a
        symmetric box is stored exactly. Other keywords as DiffuseGrid, except ``core``: the values here
        are a model's, and adding a core to them would count it twice. A residual (model - core) with
        its core goes to DiffuseGrid(values, origin, step, core=...) directly.
        """
        if kwargs.get("core") is not None:
            raise ValueError("from_values takes a model's values; a residual with its core goes to "
                             "DiffuseGrid(values, origin, step, core=...)")
        return cls(torch.as_tensor(values), origin, step, **kwargs)

    @classmethod
    def from_hkl_grid(cls, grid, **kwargs) -> "DiffuseGrid":
        """from_values(grid.values, grid.origin, grid.step): any object with those fields."""
        return cls.from_values(grid.values, grid.origin, grid.step, **kwargs)

    @classmethod
    def from_model(cls, model, step: Union[float, Sequence[float]], extent: Optional[Sequence[float]] = None,
                   d_min: Optional[float] = None, cell: Optional[Sequence[float]] = None,
                   chunk: Optional[int] = 1 << 18, dtype: torch.dtype = torch.float64, device=None, core=None,
                   **kwargs) -> "DiffuseGrid":
        """Sample a DiffuseModel on the box |h_i| <= extent_i, centred on hkl = 0, nodes step apart.

        extent: half-widths in Miller units, or ``d_min`` with ``cell`` (a, b, c, ...: Angstrom), for
            which extent_i = a_i / d_min, the largest |h_i| on the resolution sphere in any cell.
            The box reaches one node past the extent. A step dividing 1/2 puts nodes on the lattice
            points and on the half-integer planes.
        chunk: nodes per model call, so that a large grid never holds the model's work for all of
            them at once (None: one call). The model runs under no_grad at the values it stores:
            with a point group, once per orbit (at a node inside the box, value_hkl()), so it must have
            the point group's symmetry.
        core: a DiffuseModel for the model's sharp part, or True for ``model.core()``. The grid then
            stores model - core (sampled under no_grad) and adds the core back at every query.
        Other keywords (point_group, anomalous, refinable, outside) as DiffuseGrid.
        """
        if core is True:
            if not callable(getattr(model, "core", None)):
                raise ValueError("core=True needs a model with a core() method")
            core = model.core()
        step = _triple(step, "step")
        if extent is None:
            if d_min is None or cell is None:
                raise ValueError("from_model needs an extent, or d_min with the cell")
            extent = np.asarray(cell, dtype=np.float64)[:3] / float(d_min)
        extent = _triple(extent, "extent")
        half = np.ceil(extent / step - 1e-9).astype(np.int64) + 1
        grid = cls(None, -half * step, step, shape=tuple(2 * half + 1), dtype=dtype, device=device, core=core,
                   **kwargs)
        total = grid.values.numel()
        chunk = total if chunk is None else int(chunk)
        with torch.no_grad():
            flat_values = grid.values.view(-1)
            for start in range(0, total, chunk):
                stop = min(start + chunk, total)
                hkl = grid.value_hkl(torch.arange(start, stop, device=flat_values.device))
                sampled = torch.as_tensor(model(hkl))
                if core is not None:
                    sampled = sampled - torch.as_tensor(core(hkl)).to(sampled)
                flat_values[start:stop] = sampled.to(dtype=flat_values.dtype, device=flat_values.device)
        return grid

    # -- geometry of the nodes --------------------------------------------------------------------

    @property
    def origin(self) -> torch.Tensor:
        return torch.as_tensor(self._origin, dtype=torch.float64)

    @property
    def step(self) -> torch.Tensor:
        return torch.as_tensor(self._step, dtype=torch.float64)

    def grid(self) -> "DiffuseGrid":
        """The DiffuseModel protocol's optional grid(): a grid is its own."""
        return self

    def nbytes(self) -> int:
        """Bytes held: the values, the orbit keys of a symmetric grid, and the core's parameters and buffers."""
        held = self.values.numel() * self.values.element_size()
        if self.symmetric:
            held += self._keys.numel() * self._keys.element_size()
        if isinstance(self.core, nn.Module):
            held += sum(t.numel() * t.element_size() for t in list(self.core.parameters()) + list(self.core.buffers()))
        return held

    def _flat_to_ijk(self, flat: torch.Tensor) -> torch.Tensor:
        nx, ny, nz = self.shape
        return torch.stack([flat // (ny * nz), (flat // nz) % ny, flat % nz], dim=-1)

    def _canonical_keys(self, ijk: torch.Tensor) -> torch.Tensor:
        """The largest key of the mates m R of each node m = ijk - half: one key per orbit."""
        m = ijk - torch.as_tensor(self._half, device=ijk.device)
        ops = self._ops
        mates = sum(m[:, None, i, None] * ops[None, :, i, :] for i in range(3))  # (n, ops, 3)
        return self._encode(mates).max(dim=1).values

    def value_index(self, ijk: torch.Tensor) -> torch.Tensor:
        """Index into values.view(-1) of nodes (..., 3) given as box indices (i, j, k) inside the box."""
        ijk = torch.as_tensor(ijk, dtype=torch.long, device=(self._keys if self.symmetric else self.values).device)
        flat = ijk.reshape(-1, 3)
        if not self.symmetric:
            ny, nz = self.shape[1], self.shape[2]
            index = (flat[:, 0] * ny + flat[:, 1]) * nz + flat[:, 2]
        else:
            index = torch.cat([self._find(self._canonical_keys(part))
                               for part in flat.split(_LOOKUP_CHUNK)]) if flat.shape[0] else flat[:, 0]
        return index.reshape(ijk.shape[:-1])

    def _find(self, keys: torch.Tensor) -> torch.Tensor:
        """Positions of orbit keys in _keys, each of which must be there exactly."""
        index = torch.searchsorted(self._keys, keys).clamp(max=max(self._keys.numel() - 1, 0))
        if self._keys.numel() == 0 or not bool((self._keys[index] == keys).all()):
            raise RuntimeError("a node's orbit is not among the grid's orbits: the operators or keys are inconsistent")
        return index

    def value_hkl(self, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """hkl (n, 3) of stored values (all, or those at ``index`` into values.view(-1)): the nodes of a
        P1 box, or for each orbit its node inside the box of largest key (every listed orbit has one), so
        that a model defined only on the box (another grid, outside="error") can be sampled."""
        device = self.values.device
        if index is None:
            index = torch.arange(self.values.numel(), device=device)
        step = torch.as_tensor(self._step, dtype=self.values.dtype, device=device)
        if not self.symmetric:
            origin = torch.as_tensor(self._origin, dtype=self.values.dtype, device=device)
            return origin + step * self._flat_to_ijk(index).to(self.values.dtype)
        canonical = self._decode(self._keys[index])
        half = torch.as_tensor(self._half, device=device)
        mates = sum(canonical[:, None, i, None] * self._ops[None, :, i, :] for i in range(3))  # (n, ops, 3)
        keys = torch.where((mates.abs() <= half).all(dim=-1), self._encode(mates), torch.full_like(mates[..., 0], -1))
        inside = mates[torch.arange(mates.shape[0], device=device), keys.argmax(dim=1)]
        return step * inside.to(self.values.dtype)

    def _average_over_mates(self, box: torch.Tensor) -> torch.Tensor:
        total = box.numel()
        sums = torch.zeros(self._keys.numel(), dtype=box.dtype, device=box.device)
        counts = torch.zeros_like(sums)
        flat_box = box.reshape(-1)
        for start in range(0, total, _BUILD_CHUNK):
            flat = torch.arange(start, min(start + _BUILD_CHUNK, total), device=box.device)
            index = self.value_index(self._flat_to_ijk(flat))
            sums.index_add_(0, index, flat_box[start:start + flat.numel()])
            counts.index_add_(0, index, torch.ones_like(flat_box[start:start + flat.numel()]))
        return sums / counts

    # -- evaluation ---------------------------------------------------------------------------------

    def forward(self, hkl: torch.Tensor) -> torch.Tensor:
        """Scattering per cell at fractional hkl (n, 3): trilinear in the nodes, plus the core if any.

        With a core the residual is frozen: the grid returns core(theta) + [model(theta_b) - core(theta_b)],
        theta_b the core's parameters when the residual was sampled. Its gradient in theta is the core's
        alone; any dependence of the rest of the model on theta is not in it, and a core that is not the
        model's exact sharp term leaves a sharp residual fixed at theta_b while the core moves.
        core_drift() is |theta - theta_b|; past core_drift_tolerance (0.05, a placeholder to be set on V2)
        a warning is given, once: rebuild the grid there. With refinable values the residual's values and
        the core's amplitude are nearly degenerate (both scale the halo near the lattice points); refine
        one of them. Outside the box the residual is zero (or an error) and the core remains.
        """
        result = self._interpolate(hkl)
        if self.core is not None:
            if not self._core_drift_warned and self.core_drift() > self.core_drift_tolerance:
                self._core_drift_warned = True
                warnings.warn(f"the core's parameters drifted {self.core_drift():.3g} from where the residual was "
                              f"sampled (tolerance {self.core_drift_tolerance}): rebuild the grid", UserWarning)
            result = result + torch.as_tensor(self.core(hkl)).to(device=result.device).to(dtype=result.dtype)
        return result

    def _interpolate(self, hkl: torch.Tensor) -> torch.Tensor:
        values = self.values
        query_device, query_dtype = hkl.device, hkl.dtype
        if hkl.device != values.device:
            hkl = hkl.to(dtype=values.dtype).to(device=values.device)
        dtype, device = hkl.dtype, hkl.device
        origin = torch.as_tensor(self._origin, dtype=dtype, device=device)
        step = torch.as_tensor(self._step, dtype=dtype, device=device)
        shape = torch.as_tensor(self.shape, device=device)
        u = (hkl - origin) / step
        finite = torch.isfinite(u).all(dim=-1)
        if self.outside == "error":
            if not bool(finite.all()):
                raise ValueError(f"hkl {hkl[~finite][0].tolist()} is not finite (outside='error')")
            beyond = (u < 0) | (u > (shape - 1).to(dtype))
            if bool(beyond.any()):
                first = hkl[beyond.any(dim=-1)][0].tolist()
                raise ValueError(f"hkl {first} is outside the grid (origin {self._origin.tolist()}, "
                                 f"step {self._step.tolist()}, shape {self.shape})")
        nan = torch.isnan(hkl).any(dim=-1)
        u = torch.where(finite[:, None], u, torch.full_like(u, -2.0))  # +-inf: outside (NaN restored below)
        cell = torch.floor(u.detach())
        fraction = u - cell
        corners = cell.long()[:, None, :] + _CORNERS.to(device)  # (n, 8, 3)
        inside = ((corners >= 0) & (corners < shape)).all(dim=-1)
        with torch.no_grad():
            index = self.value_index(torch.minimum(corners.clamp(min=0), shape - 1))
        node = values.reshape(-1)[index].to(dtype)
        node = torch.where(inside, node, torch.zeros((), dtype=dtype, device=device))
        sides = torch.stack([1 - fraction, fraction], dim=-1)  # (n, 3, 2)
        corner_sides = _CORNERS.to(device)
        weights = sides[:, 0, corner_sides[:, 0]] * sides[:, 1, corner_sides[:, 1]] * sides[:, 2, corner_sides[:, 2]]
        result = (weights * node).sum(dim=-1)
        if bool(nan.any()):
            result = torch.where(nan, torch.full_like(result, float("nan")), result)
        if result.device != query_device or result.dtype != query_dtype:
            result = result.to(device=query_device).to(dtype=query_dtype)  # device first: MPS has no float64
        return result

    def extra_repr(self) -> str:
        storage = f"ASU of {self.values.numel()} orbits" if self.symmetric else "P1"
        return (f"shape={self.shape}, origin={self._origin.tolist()}, step={self._step.tolist()}, {storage}, "
                f"outside={self.outside!r}, refinable={isinstance(self.values, nn.Parameter)}, "
                f"core={'yes' if self.core is not None else 'no'}")
