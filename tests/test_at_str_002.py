"""
Acceptance test AT-STR-002: Tricubic interpolation with fallback.

Tests tricubic interpolation of structure factors with proper fallback behavior
when neighborhood goes out of bounds.
"""

import torch
import tempfile
import os
from pathlib import Path
from io import StringIO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.config import CrystalConfig


def test_tricubic_interpolation_enabled():
    """
    AT-STR-002: Test that tricubic interpolation is enabled and works correctly.

    Setup: Enable -interpolate; choose fractional h,k,l within a grid with complete 4×4×4 neighborhoods.
    Expectation: F_cell SHALL be tricubically interpolated between neighbors.

    INTERP-PARITY-001: the grid used to be h,k,l ∈ [-2,2]. C's safety test is
    `(h-h_min+3) > h_range || h-2 < h_min` (with h_range = h_max-h_min+1), i.e. it
    demands a margin of *two* whole indices on each side, not the one the 4×4×4
    gather needs — on a [-2,2] grid that leaves only h == 0 exactly. Widened to
    [-4,4] so the fractional queries below sit inside C's window.
    """
    # Create a simple HKL file with a 9x9x9 grid of known values
    hkl_content = StringIO()

    # Generate a grid with values that vary smoothly
    for h in range(-4, 5):
        for k in range(-4, 5):
            for l in range(-4, 5):
                # Create a smooth function: F = 100 + 10*h + 5*k + 2*l
                F = 100.0 + 10.0 * h + 5.0 * k + 2.0 * l
                hkl_content.write(f"{h} {k} {l} {F:.1f}\n")

    # Write HKL to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.hkl', delete=False) as f:
        f.write(hkl_content.getvalue())
        hkl_file = f.name

    try:
        # Create crystal with interpolation enabled
        config = CrystalConfig(
            cell_a=100.0,
            cell_b=100.0,
            cell_c=100.0,
            cell_alpha=90.0,
            cell_beta=90.0,
            cell_gamma=90.0,
            N_cells=(5, 5, 5),
            default_F=0.0
        )

        crystal = Crystal(config, device='cpu')
        crystal.load_hkl(hkl_file)
        crystal.interpolate = True  # Force interpolation on

        # Test fractional indices that should be interpolated
        # h=0.5, k=0.5, l=0.5 should interpolate between the 8 nearest integer points
        h = torch.tensor(0.5)
        k = torch.tensor(0.5)
        l = torch.tensor(0.5)

        F_interp = crystal.get_structure_factor(h, k, l)

        # The interpolation should give us a value between the surrounding points
        # For our smooth function, the exact value at (0.5, 0.5, 0.5) should be
        # approximately 100 + 10*0.5 + 5*0.5 + 2*0.5 = 100 + 5 + 2.5 + 1 = 108.5
        #
        # However, tricubic interpolation will give a slightly different value
        # based on the 4x4x4 neighborhood. We just check it's in a reasonable range.
        assert 105.0 < F_interp.item() < 112.0, \
            f"Interpolated value {F_interp.item()} outside expected range"

        # Test another fractional point
        h = torch.tensor(-0.25)
        k = torch.tensor(0.75)
        l = torch.tensor(-0.5)

        F_interp2 = crystal.get_structure_factor(h, k, l)

        # This should also give a reasonable interpolated value
        # Expected roughly: 100 + 10*(-0.25) + 5*0.75 + 2*(-0.5) = 100 - 2.5 + 3.75 - 1 = 100.25
        assert 97.0 < F_interp2.item() < 104.0, \
            f"Second interpolated value {F_interp2.item()} outside expected range"

        # INTERP-PARITY-001 regression: interpolation must actually change the
        # answer. The simulator used to hand the ROUNDED h0,k0,l0 to this lookup,
        # and a 4-point Lagrange polynomial evaluated on one of its own nodes
        # returns that node's value exactly — so -interpolate and -nointerpolate
        # produced bit-identical images. Querying at a fractional point with the
        # flag off must give the nearest-neighbour value, and it must differ.
        crystal.interpolate = False
        F_nearest = crystal.get_structure_factor(h, k, l)
        crystal.interpolate = True

        h0, k0, l0 = torch.round(h), torch.round(k), torch.round(l)
        expected_nn = 100.0 + 10.0 * h0.item() + 5.0 * k0.item() + 2.0 * l0.item()
        assert torch.allclose(F_nearest, torch.tensor(expected_nn, dtype=F_nearest.dtype)), \
            f"Nearest-neighbour lookup should be F{(h0.item(), k0.item(), l0.item())}={expected_nn}, got {F_nearest.item()}"
        assert not torch.allclose(F_interp2, F_nearest), \
            "Tricubic interpolation returned the nearest-neighbour value — the fractional " \
            "h,k,l are not reaching polin3 (dead-code regression)"

    finally:
        # Clean up temp file
        os.unlink(hkl_file)

    print("✓ Tricubic interpolation test passed")


def test_tricubic_out_of_bounds_fallback():
    """
    AT-STR-002: Test out-of-bounds fallback behavior.

    Setup: Enable interpolation but query a point where C's safety test fails.
    Expectation (INTERP-PARITY-001, verified against nanoBragg.c):
    - SHALL print a one-time warning
    - SHALL fall back to the NEAREST-NEIGHBOUR lookup for that evaluation, which
      yields default_F only when the rounded index is also outside the Fhkl box.
      C writes `F_cell = default_F` in the out-of-range branch, but that value is
      immediately overwritten: clearing `interpolate` makes the very next
      `if(! interpolate){ ... F_cell = Fhkl[h0-h_min]... }` block run.
    - SHALL NOT mutate the crystal's interpolate flag. C *does* latch its shared
      `interpolate` scalar to 0 there, which silently renders the rest of the
      image nearest-neighbour; that is evaluation-order dependent (and racy under
      OpenMP) and is deliberately not reproduced in the vectorised port. Samples
      that are in range keep being interpolated.
    """
    # Create an HKL grid (9x9x9). C's safety window is [h_min+2, h_max-2] = [-2, 2],
    # so queries outside that band exercise the fallback while queries inside it
    # prove the flag was not latched off.
    hkl_content = StringIO()

    for h in range(-4, 5):
        for k in range(-4, 5):
            for l in range(-4, 5):
                F = 100.0 + h + k + l
                hkl_content.write(f"{h} {k} {l} {F:.1f}\n")

    # Write HKL to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.hkl', delete=False) as f:
        f.write(hkl_content.getvalue())
        hkl_file = f.name

    try:
        # Create crystal with specific default_F value
        config = CrystalConfig(
            cell_a=100.0,
            cell_b=100.0,
            cell_c=100.0,
            cell_alpha=90.0,
            cell_beta=90.0,
            cell_gamma=90.0,
            N_cells=(5, 5, 5),
            default_F=999.0  # Distinctive value to verify fallback
        )

        crystal = Crystal(config, device='cpu')
        crystal.load_hkl(hkl_file)
        crystal.interpolate = True  # Force interpolation on initially

        # (a) Outside C's safety window (h > h_max-2 = 2) but the rounded index
        #     h0 = 3 is still inside the Fhkl box → nearest-neighbour F(3,0,0).
        F_oob = crystal.get_structure_factor(
            torch.tensor(3.2), torch.tensor(0.0), torch.tensor(0.0)
        )
        assert torch.allclose(F_oob, torch.tensor(103.0, dtype=F_oob.dtype)), \
            f"Out-of-window query should fall back to nearest-neighbour F(3,0,0)=103.0, got {F_oob.item()}"

        # (b) Out of the Fhkl box entirely → default_F.
        F_far = crystal.get_structure_factor(
            torch.tensor(6.2), torch.tensor(0.0), torch.tensor(0.0)
        )
        assert torch.allclose(F_far, torch.tensor(999.0, dtype=F_far.dtype)), \
            f"Query outside the Fhkl box should return default_F=999.0, got {F_far.item()}"

        # The flag must survive: no mid-run mutation of model state.
        assert crystal.interpolate, \
            "Out-of-range query must not mutate crystal.interpolate"

        # An in-window query afterwards is still interpolated, not rounded.
        h = torch.tensor(0.5)
        k = torch.tensor(0.5)
        l = torch.tensor(0.5)
        F_in = crystal.get_structure_factor(h, k, l)
        crystal.interpolate = False
        F_nn = crystal.get_structure_factor(h, k, l)
        crystal.interpolate = True
        assert not torch.allclose(F_in, F_nn), \
            "In-range query after an out-of-range one fell back to nearest-neighbour: " \
            "the interpolate flag was latched off"

    finally:
        # Clean up temp file
        os.unlink(hkl_file)

    print("✓ Out-of-bounds fallback test passed")


def test_auto_enable_interpolation():
    """
    AT-STR-002: Test auto-enable logic for small crystals.

    Setup: Create crystal with Na, Nb, or Nc ≤ 2.
    Expectation: Interpolation SHALL be automatically enabled.
    """
    # Test with small crystal (Na=2)
    config1 = CrystalConfig(
        cell_a=100.0,
        cell_b=100.0,
        cell_c=100.0,
        cell_alpha=90.0,
        cell_beta=90.0,
        cell_gamma=90.0,
        N_cells=(2, 5, 5),  # Na=2 should trigger auto-enable
        default_F=100.0
    )

    crystal1 = Crystal(config1, device='cpu')
    assert crystal1.interpolate, "Interpolation should be auto-enabled for Na=2"

    # Test with Nb=1
    config2 = CrystalConfig(
        cell_a=100.0,
        cell_b=100.0,
        cell_c=100.0,
        cell_alpha=90.0,
        cell_beta=90.0,
        cell_gamma=90.0,
        N_cells=(5, 1, 5),  # Nb=1 should trigger auto-enable
        default_F=100.0
    )

    crystal2 = Crystal(config2, device='cpu')
    assert crystal2.interpolate, "Interpolation should be auto-enabled for Nb=1"

    # Test with larger crystal (should NOT auto-enable)
    config3 = CrystalConfig(
        cell_a=100.0,
        cell_b=100.0,
        cell_c=100.0,
        cell_alpha=90.0,
        cell_beta=90.0,
        cell_gamma=90.0,
        N_cells=(5, 5, 5),  # All > 2, should not auto-enable
        default_F=100.0
    )

    crystal3 = Crystal(config3, device='cpu')
    assert not crystal3.interpolate, "Interpolation should NOT be auto-enabled for large crystal"

    print("✓ Auto-enable interpolation test passed")


if __name__ == "__main__":
    # Run all tests
    test_tricubic_interpolation_enabled()
    test_tricubic_out_of_bounds_fallback()
    test_auto_enable_interpolation()

    print("\n✅ All AT-STR-002 tests passed!")