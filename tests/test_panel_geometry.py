"""Per-panel-group geometry, diffBragg's panel refinement (plan step 3b).

diffBragg (diffBragg.cpp, rotate_fs_ss_vecs_3D and update_dxtbx_geoms) moves a panel by
    R = R_O(rot_O) R_F(rot_F) R_S(rot_S), Rodrigues rotations about the panel's own odet, fdet, sdet,
    f' = R f, s' = R s, origin' = pivot + R (origin - pivot) + (shift_X, shift_Y, shift_Z),
where the pivot of a group is the origin of its first panel (ensemble_refine_launcher.py). The
reference below applies that to the dxtbx panels with numpy and builds a model from the moved panels.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from nanobrag_torch.compat.cctbx import detector_config_from_dxtbx_panel
from nanobrag_torch.config import BeamConfig, CrystalConfig, CrystalShape
from nanobrag_torch.models.experiment import ExperimentModel

DT = torch.float64
S0 = (0.0, 0.0, -1.0)
ROT, SHIFT = "panel_group_params.delta_panel_rot_rad", "panel_group_params.delta_panel_shift_mm"


def panel(fast, slow, origin, size=(32, 32), pixel=0.2):
    return SimpleNamespace(get_fast_axis=lambda: tuple(fast), get_slow_axis=lambda: tuple(slow),
                           get_origin=lambda: tuple(origin), get_pixel_size=lambda: (pixel, pixel),
                           get_image_size=lambda: size)


def three_panels():
    tilt = math.radians(8.0)
    tilted_fast = (math.cos(tilt), 0.0, math.sin(tilt))
    return [panel((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (-3.2, 3.2, -60.0)),
            panel((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (3.6, 3.2, -60.0)),
            panel(tilted_fast, (0.0, -1.0, 0.0), (-3.2, -4.0, -59.0))]


def crystal():
    return CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=(5.0, -3.0, 2.0), N_cells=(5, 5, 5),
                         default_F=100.0, shape=CrystalShape.GAUSS)


def model(panels, **kwargs):
    configs = [detector_config_from_dxtbx_panel(p, S0, oversample=1) for p in panels]
    return ExperimentModel(crystal(), configs, BeamConfig(wavelength_A=1.2, fluence=1e24), dtype=DT, **kwargs)


def rodrigues(axis, angle):
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def moved(p, rot, shift_mm, pivot_mm):
    f, s, origin = (np.asarray(v, dtype=float) for v in (p.get_fast_axis(), p.get_slow_axis(), p.get_origin()))
    o = np.cross(f, s)
    R = rodrigues(o, rot[0]) @ rodrigues(f, rot[1]) @ rodrigues(s, rot[2])
    new_origin = pivot_mm + R @ (origin - pivot_mm) + np.asarray(shift_mm, dtype=float)
    return panel(R @ f, R @ s, new_origin, p.get_image_size(), p.get_pixel_size()[0])


def set_groups(m, rot, shift):
    with torch.no_grad():
        m.panel_group_params.delta_panel_rot_rad.copy_(torch.as_tensor(rot, dtype=DT))
        m.panel_group_params.delta_panel_shift_mm.copy_(torch.as_tensor(shift, dtype=DT))


def test_zero_offsets_give_the_panels_as_given():
    with torch.no_grad():
        torch.testing.assert_close(model(three_panels(), panel_groups=[0, 0, 1])(), model(three_panels())(),
                                   rtol=1e-12, atol=0)


def test_group_moves_match_the_panels_moved_as_diffbragg_moves_them():
    panels = three_panels()
    groups = [0, 0, 1]
    rot = [[0.004, -0.003, 0.002], [-0.002, 0.005, 0.001]]
    shift = [[0.15, -0.1, 0.3], [-0.05, 0.2, -0.4]]
    m = model(panels, panel_groups=groups)
    set_groups(m, rot, shift)
    pivots = {g: np.asarray(panels[groups.index(g)].get_origin(), dtype=float) for g in set(groups)}
    reference = model([moved(p, rot[g], shift[g], pivots[g]) for p, g in zip(panels, groups)], param_init="frozen")
    with torch.no_grad():
        torch.testing.assert_close(m(), reference(), rtol=1e-9, atol=1e-12 * reference().abs().max().item())


def test_panels_of_a_group_share_its_parameters_and_others_do_not_move():
    panels = three_panels()
    m = model(panels, panel_groups=[0, 0, 1])
    assert m.panel_group_params.delta_panel_rot_rad.shape == (2, 3)
    with torch.no_grad():
        before = m().clone()
    set_groups(m, [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    with torch.no_grad():
        after = m()
    torch.testing.assert_close(after[:2], before[:2], rtol=0, atol=0)
    assert not torch.equal(after[2], before[2])


def test_panel_geometry_gradcheck():
    m = model(three_panels(), panel_groups=[0, 0, 1])
    rot = torch.tensor([[0.003, -0.002, 0.001], [0.002, 0.001, -0.003]], dtype=DT, requires_grad=True)
    shift = torch.tensor([[0.1, -0.1, 0.2], [0.05, 0.1, -0.2]], dtype=DT, requires_grad=True)

    def loss(r, t):
        return functional_call(m, {ROT: r, SHIFT: t}, ()).sum()

    assert gradcheck(loss, (rot, shift), eps=1e-7, atol=1e-8, rtol=1e-4)


def test_without_panel_groups_the_parameters_are_unrefinable_and_change_nothing():
    m = model(three_panels())
    assert ROT in m.unrefinable and SHIFT in m.unrefinable
    for group in ("panel_rotation", "panel_shift"):
        with pytest.raises(ValueError, match="panel_groups"):
            m.refine(group)


def test_panel_groups_take_over_the_shared_detector_offsets():
    m = model(three_panels(), panel_groups=[0, 1, 2])
    for name in ("detector_params.delta_rotx_raw", "detector_params.delta_beam_f_raw",
                 "detector_params.delta_detz_shift_mm"):
        assert name in m.unrefinable
    m.freeze().refine("panel_rotation", "panel_shift")
    assert m.panel_group_params.delta_panel_rot_rad.requires_grad
    assert m.panel_group_params.delta_panel_shift_mm.requires_grad
