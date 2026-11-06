"""DiffuseGrid (plan D3): a diffuse model held on a regular grid of fractional hkl, interpolated trilinearly.

The grid alone: it reproduces its nodes and every multilinear function, is zero outside its box
(or raises), stores the Laue ASU through the point group's integer operators (the orbits of
nanobrag_torch.utils.structure_factors.expand_to_p1) and is differentiable in hkl and in its
values. Images and gradients through the Simulator and ExperimentModel are in
tests/test_diffuse_grid_image.py.
"""
import itertools
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import numpy as np
import pytest
import torch
from torch import nn
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.diffuse import DiffuseGrid, DiffuseModel, GaussianHalo
from nanobrag_torch.utils.structure_factors import expand_to_p1, point_group_operators

DTYPE = torch.float64


def group(*generators):
    """The closure of integer 3x3 matrices under products (h R convention: rows act on the right)."""
    ops = {tuple(np.eye(3, dtype=np.int64).ravel())}
    frontier = list(ops)
    while frontier:
        new = []
        for a in frontier:
            for g in generators:
                p = tuple((np.array(a).reshape(3, 3) @ np.array(g)).ravel())
                if p not in ops:
                    ops.add(p)
                    new.append(p)
        frontier = new
    return np.array(sorted(ops)).reshape(-1, 3, 3)


FOUR_FOLD = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
TWO_FOLD_A = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
PG_422 = group(FOUR_FOLD, TWO_FOLD_A)  # P4_3 2_1 2's point group; with Friedel's law, Laue 4/mmm
HEX_SIX = [[1, 1, 0], [-1, 0, 0], [0, 0, 1]]  # (h, k, l) -> (h - k, h, l): mixes h and k with a sum
PG_6 = group(HEX_SIX)


def random_hkl(n, extent, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (2 * torch.rand(n, 3, generator=g, dtype=DTYPE) - 1) * torch.as_tensor(extent, dtype=DTYPE)


def test_point_groups_used_here_are_groups():
    assert len(PG_422) == 8 and len(PG_6) == 6
    assert len(point_group_operators(PG_422)) == 16  # with Friedel's law
    assert len(point_group_operators(PG_422, anomalous=True)) == 8


def test_a_grid_is_a_diffuse_model_and_a_module():
    grid = DiffuseGrid(torch.ones(3, 3, 3, dtype=DTYPE), origin=(-1, -1, -1), step=1.0)
    assert isinstance(grid, DiffuseModel) and isinstance(grid, nn.Module)
    assert grid.grid() is grid


# 1. Interpolation: nodes, multilinear functions, outside the box.

def _box(origin, step, shape):
    axes = [origin[i] + step[i] * torch.arange(shape[i], dtype=DTYPE) for i in range(3)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def test_nodes_are_reproduced_exactly():
    origin, step, shape = (-1.0, -0.5, 0.25), (0.5, 0.25, 0.125), (5, 6, 7)
    values = torch.rand(shape, dtype=DTYPE, generator=torch.Generator().manual_seed(1))
    grid = DiffuseGrid(values, origin, step)
    nodes = _box(origin, step, shape).reshape(-1, 3)
    torch.testing.assert_close(grid(nodes), values.reshape(-1), rtol=0, atol=1e-15)


def test_multilinear_functions_are_interpolated_exactly():
    origin, step, shape = (-2.0, -2.0, -1.0), (0.5, 0.25, 0.5), (9, 17, 5)

    def f(hkl):
        h, k, l = hkl.unbind(-1)
        return 3.0 + 2 * h - k + 0.5 * l + h * k - 0.25 * k * l + 0.1 * h * k * l

    grid = DiffuseGrid(f(_box(origin, step, shape)), origin, step)
    inside = random_hkl(500, (1.99, 1.99, 0.99))
    torch.testing.assert_close(grid(inside), f(inside), rtol=0, atol=1e-12)


def test_outside_the_box_is_zero_and_the_edge_cells_go_to_zero_continuously():
    """Zero-extension: the grid is the trilinear interpolant of its values padded with zeros, so a
    point within one step outside the box sees the boundary nodes with the weights of that step."""
    grid = DiffuseGrid(torch.ones(3, 3, 3, dtype=DTYPE), origin=(-1, -1, -1), step=1.0)
    far = torch.tensor([[5.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 1.0 + 1e-12 + 1.0]], dtype=DTYPE)
    assert torch.equal(grid(far), torch.zeros(3, dtype=DTYPE))
    edge = torch.tensor([[1.25, 0.0, 0.0], [0.0, -1.5, 0.0], [1.5, 1.5, 0.0]], dtype=DTYPE)
    torch.testing.assert_close(grid(edge), torch.tensor([0.75, 0.5, 0.25], dtype=DTYPE), rtol=0, atol=1e-15)


def test_outside_error_refuses_points_beyond_the_box():
    grid = DiffuseGrid(torch.ones(3, 3, 3, dtype=DTYPE), origin=(-1, -1, -1), step=1.0, outside="error")
    torch.testing.assert_close(grid(torch.tensor([[1.0, -1.0, 0.3]], dtype=DTYPE)), torch.ones(1, dtype=DTYPE))
    with pytest.raises(ValueError, match="outside the grid"):
        grid(torch.tensor([[1.0 + 1e-9, 0.0, 0.0]], dtype=DTYPE))
    with pytest.raises(ValueError, match="outside"):
        DiffuseGrid(torch.ones(2, 2, 2), origin=(0, 0, 0), step=1.0, outside="clamp")


def test_bad_grids_are_refused():
    with pytest.raises(ValueError, match="step"):
        DiffuseGrid(torch.ones(2, 2, 2), origin=(0, 0, 0), step=0.0)
    with pytest.raises(ValueError, match="nodes"):
        DiffuseGrid(torch.ones(1, 2, 2), origin=(0, 0, 0), step=1.0)


def test_output_follows_the_query_dtype_and_values_keep_theirs():
    grid = DiffuseGrid(torch.ones(3, 3, 3, dtype=DTYPE), origin=(-1, -1, -1), step=1.0)
    out = grid(torch.zeros(4, 3, dtype=torch.float32))
    assert out.dtype == torch.float32 and grid.values.dtype == DTYPE


# 2. Refinable or frozen.

def test_values_are_a_parameter_or_a_buffer():
    values = torch.ones(3, 3, 3, dtype=DTYPE)
    refinable = DiffuseGrid(values, origin=(-1, -1, -1), step=1.0)
    frozen = DiffuseGrid(values, origin=(-1, -1, -1), step=1.0, refinable=False)
    assert [n for n, _ in refinable.named_parameters()] == ["values"]
    assert list(frozen.parameters()) == [] and "values" in dict(frozen.named_buffers())
    assert values.data_ptr() not in (refinable.values.data_ptr(), frozen.values.data_ptr())  # a copy


# 3. Symmetry: the Laue ASU, through the integer operators.

def _p1_and_asu(rotations, extent=(2.0, 2.0, 1.5), step=0.25, model=None):
    model = model or GaussianHalo(5.0, 0.2)
    p1 = DiffuseGrid.from_model(model, step=step, extent=extent)
    asu = DiffuseGrid.from_model(model, step=step, extent=extent, rotations=rotations)
    return p1, asu


class _HexagonalBlob(nn.Module):
    """exp(-(h^2 - h k + k^2 + l^2) / 2): the hexagonal metric, invariant under 6/m (GaussianHalo's
    Miller-unit distance is not: it has only the symmetry of the square lattice)."""

    def forward(self, hkl):
        h, k, l = hkl.unbind(-1)
        return torch.exp(-(h * h - h * k + k * k + l * l) / 2)


@pytest.mark.parametrize("rotations,ops,most,model", [(PG_422, 16, 1.3, None), (PG_6, 12, 2.0, _HexagonalBlob())])
def test_asu_grid_is_the_p1_grid_with_fewer_values(rotations, ops, most, model):
    """The box is closed under 4/mmm (n / 16 orbits, plus those on mirrors); a hexagonal operator
    takes some nodes out of a rectangular box, so their orbits are only partly in it (728 for n / 12 = 451)."""
    p1, asu = _p1_and_asu(rotations, step=0.25, model=model)
    assert asu.values.ndim == 1 and p1.values.shape == p1.shape
    n_box = int(np.prod(p1.shape))
    assert n_box / ops <= asu.values.numel() < most * n_box / ops + 30
    hkl = random_hkl(2000, (2.2, 2.2, 1.7), seed=3)
    torch.testing.assert_close(asu(hkl), p1(hkl), rtol=0, atol=1e-14)


@pytest.mark.parametrize("rotations,anomalous", [(PG_422, False), (PG_422, True), (PG_6, False), (None, False)])
def test_asu_orbits_are_those_of_expand_to_p1(rotations, anomalous):
    """On a grid of step 1 the nodes are Miller indices: the grid's orbits are StructureFactors' orbits."""
    grid = DiffuseGrid.from_model(GaussianHalo(1.0, 0.3), step=1.0, extent=(3, 3, 2), rotations=rotations,
                                  anomalous=anomalous, refinable=False)
    nodes = _box(grid.origin.tolist(), grid.step.tolist(), grid.shape).reshape(-1, 3).round().long()
    ids = grid.value_index(nodes - grid.origin.round().long()).numpy()
    # one representative per grid orbit, expanded by StructureFactors' expansion, covers the box once
    first = np.unique(ids, return_index=True)[1]
    if rotations is None and not anomalous:
        assert grid.values.ndim == 3  # P1 storage: no operators and no Friedel's law
        return
    p1, source = expand_to_p1(nodes.numpy()[first], rotations, anomalous)
    lookup = {tuple(h): s for h, s in zip(p1, source)}
    by_source = np.array([lookup[tuple(h)] for h in nodes.numpy()])
    # the same partition: grid id and expand_to_p1 source determine each other
    pairs = np.unique(np.stack([ids, by_source]), axis=1)
    assert len(np.unique(pairs[0])) == len(np.unique(pairs[1])) == pairs.shape[1]


def test_p1_storage_is_the_default_and_friedel_alone_halves_it():
    p1, friedel = _p1_and_asu(None)
    assert p1.values.ndim == 3
    friedel = DiffuseGrid.from_model(GaussianHalo(5.0, 0.2), step=0.25, extent=(2, 2, 1.5), rotations=np.eye(3))
    assert friedel.values.numel() == (int(np.prod(p1.shape)) + 1) // 2


def test_symmetric_grids_must_be_centred_with_equal_steps_on_mixed_axes():
    with pytest.raises(ValueError, match="centred"):
        DiffuseGrid.from_values(torch.ones(4, 4, 4), origin=(-1.0, -1.0, -1.0), step=0.5, rotations=PG_422)
    with pytest.raises(ValueError, match="step"):
        DiffuseGrid.from_values(torch.ones(5, 9, 5), origin=(-1.0, -1.0, -1.0), step=(0.5, 0.25, 0.5),
                                rotations=PG_422)
    # unequal steps are fine on axes the operators do not mix (l here)
    DiffuseGrid.from_values(torch.ones(5, 5, 9), origin=(-1.0, -1.0, -1.0), step=(0.5, 0.5, 0.25), rotations=PG_422)


def test_from_values_averages_symmetry_mates():
    values = torch.rand(5, 5, 5, dtype=DTYPE, generator=torch.Generator().manual_seed(4))
    grid = DiffuseGrid.from_values(values, origin=(-1.0, -1.0, -1.0), step=0.5, rotations=np.eye(3))
    expected = 0.5 * (values + values.flip(0, 1, 2))  # Friedel mates: -h is the flipped box
    nodes = _box((-1.0, -1.0, -1.0), (0.5, 0.5, 0.5), (5, 5, 5)).reshape(-1, 3)
    torch.testing.assert_close(grid(nodes), expected.reshape(-1), rtol=0, atol=1e-15)


class _HKLGridLike:
    """eryx.torchbragg.HKLGrid's fields (duck-typed: torchBragg does not import eryx)."""

    def __init__(self, origin, step, values):
        self.origin, self.step, self.values = origin, step, values
        self.shape = tuple(values.shape)


def test_from_values_takes_an_hkl_grid_triple():
    origin = torch.tensor([-1.0, -1.5, -0.5], dtype=DTYPE)
    step = torch.tensor([0.5, 0.5, 0.25], dtype=DTYPE)
    values = torch.rand(5, 7, 5, dtype=DTYPE, generator=torch.Generator().manual_seed(5))
    other = _HKLGridLike(origin, step, values)
    grid = DiffuseGrid.from_values(other.values, other.origin, other.step)
    assert torch.equal(DiffuseGrid.from_hkl_grid(other).values, grid.values)
    torch.testing.assert_close(grid(_box(origin.tolist(), step.tolist(), (5, 7, 5)).reshape(-1, 3)), values.reshape(-1))


# 4. Construction from a model: extent from d_min, chunks.

class _Counting(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.calls = model, []

    def forward(self, hkl):
        self.calls.append(hkl.shape[0])
        return self.model(hkl)


@pytest.mark.parametrize("rotations", [None, PG_422])
def test_from_model_evaluates_in_chunks_and_without_a_graph(rotations):
    halo = GaussianHalo(5.0, 0.2)
    counting = _Counting(halo)
    small = DiffuseGrid.from_model(counting, step=0.25, extent=(1, 1, 1), rotations=rotations, chunk=37)
    assert max(counting.calls) <= 37 and len(counting.calls) > 1
    whole = DiffuseGrid.from_model(halo, step=0.25, extent=(1, 1, 1), rotations=rotations, chunk=None)
    assert torch.equal(small.values, whole.values)
    assert not small.values.grad_fn and small.values.requires_grad


def test_from_model_takes_its_extent_from_d_min_and_the_cell():
    """|h| <= a / d_min on the resolution sphere, a the real-space length: the box covers it, plus a node."""
    grid = DiffuseGrid.from_model(GaussianHalo(1.0, 0.3), step=0.5, d_min=10.0, cell=(30.0, 35.0, 40.0, 90, 90, 90))
    half = (-grid.origin / grid.step).round().long().tolist()
    assert half == [7, 8, 9]  # ceil(3 / 0.5) + 1, ceil(3.5 / 0.5) + 1, ceil(4 / 0.5) + 1
    with pytest.raises(ValueError, match="extent"):
        DiffuseGrid.from_model(GaussianHalo(1.0, 0.3), step=0.5)


def test_from_model_matches_the_model_at_the_nodes():
    halo = GaussianHalo(5.0, 0.2)
    grid = DiffuseGrid.from_model(halo, step=0.125, extent=(1, 1, 1), rotations=PG_422)
    nodes = _box(grid.origin.tolist(), grid.step.tolist(), grid.shape).reshape(-1, 3)
    torch.testing.assert_close(grid(nodes), halo(nodes).detach(), rtol=0, atol=1e-14)


# 5. Gradients in hkl and in the values.

def _away_from_cell_edges(grid, hkl, margin=0.05):
    u = (hkl - grid.origin) / grid.step
    return ((u - u.floor()) > margin).all() and ((u.ceil() - u) > margin).all()


@pytest.mark.parametrize("rotations", [None, PG_422])
def test_gradcheck_in_hkl_and_in_values(rotations):
    grid = DiffuseGrid.from_model(GaussianHalo(5.0, 0.3), step=0.25, extent=(1, 1, 1), rotations=rotations)
    g = torch.Generator().manual_seed(6)
    cells = torch.randint(-3, 3, (8, 3), generator=g).to(DTYPE)
    hkl = ((cells + 0.1 + 0.8 * torch.rand(8, 3, generator=g, dtype=DTYPE)) * 0.25).requires_grad_()
    # trilinear is piecewise linear in hkl; gradcheck's steps of 1e-6 must not cross a cell edge
    assert _away_from_cell_edges(grid, hkl.detach())
    assert gradcheck(grid, (hkl,), eps=1e-6, atol=1e-9)
    values = grid.values.detach().clone().requires_grad_()
    assert gradcheck(lambda v: functional_call(grid, {"values": v}, (hkl.detach(),)), (values,), eps=1e-6, atol=1e-9)


def test_the_hkl_gradient_is_the_cells_slope():
    grid = DiffuseGrid(torch.arange(8, dtype=DTYPE).reshape(2, 2, 2), origin=(0, 0, 0), step=0.5)
    hkl = torch.tensor([[0.1, 0.2, 0.3]], dtype=DTYPE, requires_grad=True)
    grid(hkl).sum().backward()
    u = hkl.detach()[0] / 0.5
    v = torch.arange(8, dtype=DTYPE).reshape(2, 2, 2)
    expected = torch.stack([_partial(v, u, axis) for axis in range(3)]) / 0.5  # d/du of the trilinear form
    torch.testing.assert_close(hkl.grad[0], expected, rtol=0, atol=1e-14)


def _partial(v, u, axis):
    total = torch.zeros((), dtype=DTYPE)
    for corner in itertools.product((0, 1), repeat=3):
        weight = torch.ones((), dtype=DTYPE)
        for a in range(3):
            if a == axis:
                weight = weight * (1.0 if corner[a] else -1.0)
            else:
                weight = weight * (u[a] if corner[a] else 1 - u[a])
        total = total + weight * v[corner]
    return total


def test_values_gradient_is_the_trilinear_weights_summed_over_mates():
    grid = DiffuseGrid.from_model(GaussianHalo(5.0, 0.3), step=0.5, extent=(1, 1, 1), rotations=np.eye(3))
    hkl = torch.tensor([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=DTYPE)  # Friedel mates of each other
    grid(hkl).sum().backward()
    assert grid.values.grad.sum().item() == pytest.approx(2.0, abs=1e-14)  # the weights sum to 1 per point
