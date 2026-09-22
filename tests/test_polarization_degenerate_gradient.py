"""Exactly-forward scattering must not poison the backward pass.

At 2theta = 0 the diffracted ray lies along the incident beam, so it has no
component in the E-B plane and E_out == B_out == 0.0 exactly. C evaluates
atan2(0, 0), which IEEE-754 defines as +0, and moves on (nanoBragg.c:4102).

torch's forward value is fine too, but d/dx atan2(y, x) = y/(x^2 + y^2) is
0/0 = NaN there, and sin^2(2theta) being 0 at that point does not rescue it
because 0 * NaN = NaN. The NaN then spreads through the entire backward pass,
so every geometry gradient for the whole image becomes NaN while the rendered
image still looks perfectly clean.

Reachable with an ordinary configuration, measured on a 32x32 detector: beam
centre on the geometric detector centre (1.6 mm = 32 * 0.1 / 2), oversample 1,
Kahn factor 1.0 -> d(sum I)/d(distance) = nan.
"""

import torch

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.simulator import Simulator
from nanobrag_torch.utils.physics import polarization_factor

# Beam along +x, polarization axis along +z -- the non-degenerate arrangement.
# (Putting the axis parallel to the beam zeroes cross(axis, incident) for every
# pixel, which is a different degeneracy and not what the simulator produces.)
AXIS = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
INCIDENT = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)


def test_exact_forward_scattering_gradient_is_finite():
    diffracted = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64,
                              requires_grad=True)

    polar = polarization_factor(1.0, INCIDENT, diffracted, AXIS)
    polar.sum().backward()

    assert not torch.isnan(polar).any(), "forward value should be finite"
    assert diffracted.grad is not None
    assert not torch.isnan(diffracted.grad).any(), (
        f"atan2(0,0) poisoned the backward pass: grad={diffracted.grad}")


def test_degenerate_value_still_matches_c():
    """psi must remain 0 there, as C's atan2(0,0) gives, so guarding the inputs
    cannot change any rendered image."""
    diffracted = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    # 2theta = 0 and psi = 0 => 0.5 * (1 + 1 - K * cos(0) * 0) = 1.0
    polar = polarization_factor(1.0, INCIDENT, diffracted, AXIS)
    assert torch.allclose(polar, torch.ones_like(polar), atol=1e-9), polar


def test_near_axis_and_ordinary_scattering_unaffected():
    for offset in (1e-15, 1e-9, 1e-6, 0.176):
        d = torch.tensor([[1.0, offset, 0.0]], dtype=torch.float64,
                         requires_grad=True)
        polar = polarization_factor(1.0, INCIDENT, d, AXIS)
        polar.sum().backward()
        assert not torch.isnan(d.grad).any(), f"NaN at offset {offset}"


def test_simulator_geometry_gradient_is_finite_with_beam_on_pixel_centre():
    """The end-to-end reproduction: clean image, NaN gradient, before the fix."""
    distance = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    detector = Detector(DetectorConfig(
        distance_mm=distance,
        pixel_size_mm=0.1,
        spixels=32,
        fpixels=32,
        beam_center_f=1.6,
        beam_center_s=1.6,
    ))
    crystal_config = CrystalConfig(
        cell_a=100.0, cell_b=100.0, cell_c=100.0,
        cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0,
        default_F=100.0, N_cells=(5, 5, 5),
    )
    crystal = Crystal(crystal_config)
    beam = BeamConfig(wavelength_A=1.0, fluence=1e24, polarization_factor=1.0)

    simulator = Simulator(crystal, detector, crystal_config=crystal_config,
                          beam_config=beam)
    # oversample=1 puts a sample exactly at the pixel centre; the auto-selected
    # 2-fold oversampling offsets every sample by half a subpixel and hides this.
    image = simulator.run(oversample=1)
    image.sum().backward()

    assert not torch.isnan(image).any(), "image should be clean either way"
    assert distance.grad is not None
    assert not torch.isnan(distance.grad), (
        "detector-distance gradient is NaN: a pixel sampled exactly on the "
        "beam axis poisoned the backward pass")
