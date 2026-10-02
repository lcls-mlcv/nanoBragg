"""Editing DetectorConfig after construction must reach the detector geometry.

Why this exists
---------------
`Detector.distance` and `Detector.pixel_size` are properties that read
`self.config` on every access, added so a caller can install a differentiable
tensor after construction (DBEX-GRADIENT-001). The rest of the geometry --
beam centres in pixels, the three basis vectors, and pix0_vector -- was
computed once in `__init__` and never again, and `get_pixel_coords` guarded its
cache by comparing those frozen attributes against cached copies of
themselves. That comparison could not fail, so a config edit was silently
ignored and the detector kept returning the geometry of the original
configuration.

Measured before the fix, on a 32x32 / 0.1 mm detector: mutating `config` and
re-reading `get_pixel_coords()` returned unchanged coordinates for all eight
geometry fields, and calling `invalidate_cache()` repaired only two of them
(`distance_mm`, `pixel_size_mm`) because it recomputed pix0_vector alone --
never the basis vectors or the beam-centre conversion.

This bites finite-difference validation and any optimizer that writes into
config between steps, and it fails silently in the worst way: plausible
numbers from the wrong geometry.
"""

import pytest
import torch

from nanobrag_torch.config import DetectorConfig
from nanobrag_torch.models.detector import Detector

BASE = dict(distance_mm=100.0, pixel_size_mm=0.1, spixels=32, fpixels=32)

# (field, initial, mutated). Every field here feeds the derived geometry, so
# editing it must change the pixel coordinates.
GEOMETRY_FIELDS = [
    ("distance_mm", 100.0, 200.0),
    ("pixel_size_mm", 0.1, 0.2),
    ("detector_rotx_deg", 0.0, 5.0),
    ("detector_roty_deg", 0.0, 5.0),
    ("detector_rotz_deg", 0.0, 5.0),
    ("detector_twotheta_deg", 0.0, 5.0),
    ("beam_center_f", 1.6, 2.0),
    ("beam_center_s", 1.6, 2.0),
]


def _detector(**overrides):
    cfg = dict(BASE)
    cfg.update(overrides)
    return Detector(DetectorConfig(**cfg))


@pytest.mark.parametrize(
    "field,initial,mutated",
    GEOMETRY_FIELDS,
    ids=[f[0] for f in GEOMETRY_FIELDS],
)
def test_mutating_config_changes_pixel_coords(field, initial, mutated):
    """The bug: this returned stale coordinates for all eight fields."""
    det = _detector(**{field: initial})
    before = det.get_pixel_coords().clone()

    setattr(det.config, field, mutated)
    after = det.get_pixel_coords()

    assert not torch.allclose(before, after), (
        f"editing config.{field} from {initial} to {mutated} left "
        f"get_pixel_coords() unchanged — the detector is using stale geometry"
    )


@pytest.mark.parametrize(
    "field,initial,mutated",
    GEOMETRY_FIELDS,
    ids=[f[0] for f in GEOMETRY_FIELDS],
)
def test_invalidate_cache_also_rebuilds_derived_geometry(field, initial, mutated):
    """invalidate_cache() repaired only 2 of 8 fields before this change."""
    det = _detector(**{field: initial})
    before = det.get_pixel_coords().clone()

    setattr(det.config, field, mutated)
    det.invalidate_cache()
    after = det.get_pixel_coords()

    assert not torch.allclose(before, after), (
        f"invalidate_cache() did not rebuild the geometry after config.{field} "
        f"changed — it recomputes pix0_vector but not the basis vectors or the "
        f"beam-centre conversion"
    )


@pytest.mark.parametrize(
    "field,initial,mutated",
    [f for f in GEOMETRY_FIELDS if f[0] != "pixel_size_mm"],
    ids=[f[0] for f in GEOMETRY_FIELDS if f[0] != "pixel_size_mm"],
)
def test_mutated_matches_freshly_constructed(field, initial, mutated):
    """A mutated detector must agree with one built with the same values.

    `pixel_size_mm` is excluded deliberately, and the reason is not a defect:
    `DetectorConfig.__post_init__` derives the *default* beam centre in mm from
    the pixel size (1.65 mm at 0.1 mm, 3.30 mm at 0.2 mm), and editing one
    field of an already-constructed dataclass does not re-run __post_init__.
    So a fresh detector at 0.2 mm is a genuinely different configuration --
    same pixel size, different beam centre in mm -- not the same one rebuilt.
    `test_pixel_size_mutation_matches_fresh_with_same_beam_centre` pins the
    comparison that actually holds.
    """
    det = _detector(**{field: initial})
    det.get_pixel_coords()
    setattr(det.config, field, mutated)

    assert torch.allclose(
        det.get_pixel_coords(), _detector(**{field: mutated}).get_pixel_coords(),
        atol=1e-12,
    ), f"mutating config.{field} disagrees with a detector built at {mutated}"


def test_pixel_size_mutation_matches_fresh_with_same_beam_centre():
    """With the beam centre pinned, a pixel-size edit does match a fresh build."""
    det = _detector(pixel_size_mm=0.1, beam_center_f=1.65, beam_center_s=1.65)
    det.get_pixel_coords()
    det.config.pixel_size_mm = 0.2

    fresh = _detector(pixel_size_mm=0.2, beam_center_f=1.65, beam_center_s=1.65)

    assert torch.allclose(det.get_pixel_coords(), fresh.get_pixel_coords(), atol=1e-12)


def test_repeated_reads_still_use_the_cache():
    """The fix must not turn every read into a full recompute."""
    det = _detector()
    first = det.get_pixel_coords()
    version = det._geometry_version
    second = det.get_pixel_coords()

    assert second is first, "unchanged config should return the cached tensor"
    assert det._geometry_version == version, "cache was rebuilt with no config change"


def test_fresh_leaf_with_the_same_value_rebuilds_the_graph():
    """A refinement loop installing a new leaf each step must not get a stale graph.

    The value is identical, so a contents-only fingerprint would treat this as
    unchanged and hand back coordinates attached to the *previous* iteration's
    graph -- gradients would flow to a leaf the caller has already discarded.

    Note the detector is built float64 to match the leaf: installing a leaf
    whose dtype differs from the detector's drops the gradient in
    `as_tensor_preserving_grad`, which is a separate issue from caching.
    """
    det = Detector(DetectorConfig(**BASE), dtype=torch.float64)
    first_leaf = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    det.config.distance_mm = first_leaf
    coords_a = det.get_pixel_coords()
    coords_a.sum().backward()
    assert first_leaf.grad is not None

    second_leaf = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    det.config.distance_mm = second_leaf
    coords_b = det.get_pixel_coords()

    assert coords_b is not coords_a, (
        "a new leaf holding the same value returned the cached tensor, which is "
        "still attached to the previous leaf's graph"
    )
    coords_b.sum().backward()
    assert second_leaf.grad is not None, "gradient did not reach the new leaf"


# --- Entry points other than get_pixel_coords -------------------------------


def test_geometry_fields_are_real_dataclass_fields():
    """A typo in _GEOMETRY_FIELDS would silently stop watching a field.

    `_current_geometry_fingerprint` reads with `getattr(..., None)`, so a renamed
    or misspelled entry degrades to "always None" rather than raising — which is
    the same silent staleness this module exists to prevent.
    """
    import dataclasses

    real = {f.name for f in dataclasses.fields(DetectorConfig)}
    watched = set(Detector._GEOMETRY_FIELDS)
    assert watched <= real, f"not DetectorConfig fields: {sorted(watched - real)}"


def test_curved_detector_planar_coords_follow_config():
    """get_planar_pixel_coords() is the simulator's curved-mode entry point.

    It bypassed the fingerprint entirely, so curved detectors kept the staleness
    bug: it reads the derived pix0_vector and basis vectors directly.
    """
    det = Detector(DetectorConfig(**BASE, curved_detector=True))
    before = det.get_planar_pixel_coords().clone()

    det.config.distance_mm = 200.0
    after = det.get_planar_pixel_coords()

    assert not torch.allclose(before, after), (
        "get_planar_pixel_coords() ignored a config edit — curved detectors are "
        "still using stale geometry"
    )


def test_simulator_picks_up_detector_config_mutation():
    """The Simulator snapshots pixel coords, so the fix has to reach it too.

    Before this, mutating the detector config and re-running the *same* Simulator
    returned a bit-identical image — the exact mutate-then-rerun pattern that
    finite-difference checks and config-writing optimizers use.
    """
    from nanobrag_torch.config import BeamConfig, CrystalConfig
    from nanobrag_torch.models import Crystal
    from nanobrag_torch.simulator import Simulator

    ccfg = CrystalConfig(
        cell_a=100.0, cell_b=100.0, cell_c=100.0,
        cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0,
        default_F=100.0, N_cells=(3, 3, 3),
    )
    det = Detector(DetectorConfig(**BASE))
    sim = Simulator(
        Crystal(ccfg), det, crystal_config=ccfg,
        beam_config=BeamConfig(wavelength_A=6.2, fluence=1e24),
    )

    first = sim.run(oversample=1).sum().item()
    det.config.distance_mm = 200.0
    second = sim.run(oversample=1).sum().item()

    assert first != second, (
        f"same Simulator returned {second} after the detector distance doubled; "
        f"it is still reading the snapshot taken at construction"
    )

    fresh_det = Detector(DetectorConfig(**{**BASE, "distance_mm": 200.0}))
    fresh = Simulator(
        Crystal(ccfg), fresh_det, crystal_config=ccfg,
        beam_config=BeamConfig(wavelength_A=6.2, fluence=1e24),
    ).run(oversample=1).sum().item()

    assert second == pytest.approx(fresh, rel=1e-12), (
        f"mutated Simulator gives {second}, freshly built gives {fresh}"
    )


def test_invalidate_cache_does_not_leave_a_redundant_recompute():
    """invalidate_cache() used to clear the fingerprint, forcing a second rebuild."""
    det = Detector(DetectorConfig(**BASE))
    det.get_pixel_coords()

    det.config.distance_mm = 200.0
    det.invalidate_cache()
    version_after_invalidate = det._geometry_version

    det.get_pixel_coords()
    assert det._geometry_version == version_after_invalidate, (
        "get_pixel_coords() rebuilt the geometry again after invalidate_cache() "
        "had already done it"
    )
