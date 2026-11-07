"""
Spot masks predicted from geometry, for diffuse and joint fits (plan section 2.2, V2).

The reference is the Bragg image itself: with one F for every reflection and GAUSS spots under the
HKL metric (no side lobes, unlike SQUARE's sinc), each recorded spot is a local maximum, so the
predicted centres and the image's maxima must coincide.
"""
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import torch
import torch.nn.functional as nnf

from src.nanobrag_torch.config import BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, SpotMetric
from src.nanobrag_torch.models.crystal import Crystal
from src.nanobrag_torch.models.detector import Detector
from src.nanobrag_torch.simulator import Simulator
from src.nanobrag_torch.spot_mask import predicted_spot_mask, strided_pixels

CRYSTAL = dict(N_cells=(5, 5, 5), default_F=100.0, cell_a=50.0, cell_b=60.0, cell_c=70.0,
               misset_deg=(10.0, 20.0, 30.0), shape=CrystalShape.GAUSS, spot_metric=SpotMetric.HKL)
DETECTOR = dict(spixels=64, fpixels=64, distance_mm=100.0, pixel_size_mm=0.5)
BEAM = dict(wavelength_A=1.0, fluence=1e24)


def simulator(crystal=None, detector=None, beam=None):
    return Simulator(Crystal(CrystalConfig(**{**CRYSTAL, **(crystal or {})}), dtype=torch.float64),
                     Detector(DetectorConfig(**{**DETECTOR, **(detector or {})}), dtype=torch.float64),
                     beam_config=BeamConfig(**{**BEAM, **(beam or {})}), dtype=torch.float64)


def local_maxima(image, floor):
    """Pixels that are the maximum of their 3x3 neighbourhood and above floor."""
    pooled = nnf.max_pool2d(image[None, None], 3, stride=1, padding=1)[0, 0]
    return (image == pooled) & (image > floor)


def test_centres_are_the_bragg_images_maxima():
    sim = simulator()
    image = sim.run(oversample=1)
    # A GAUSS-HKL spot n sds from its pixel peaks at exp(-n^2/2) of a centred one: 1.1% at the default
    # 3 sds. Above 2% (2.8 sds) every recorded spot must be predicted.
    maxima = local_maxima(image, 2e-2 * image.max())
    assert maxima.sum() >= 5, "the test geometry must record several spots"
    centres = ~predicted_spot_mask(sim, radius_px=1.0)
    assert not (maxima & ~centres).any(), "a recorded spot was not predicted"
    on_maxima = local_maxima(image, 0.0)[centres].float().mean()
    assert on_maxima > 0.9, f"only {on_maxima:.2f} of the predicted centres are maxima of the image"


def test_only_the_direct_beam_spot_by_hand():
    # A 4 A cubic cell: the first reflection needs |s| = 0.25/A, past this detector's corners (0.225/A),
    # so the one spot is 000 at the direct beam, and the mask is a disk about one pixel. The beam falls on
    # the corner of four pixels, equally near 000: a spot still has one centre.
    sim = simulator(crystal=dict(cell_a=4.0, cell_b=4.0, cell_c=4.0, misset_deg=(0.0, 0.0, 0.0)))
    assert predicted_spot_mask(sim, radius_px=0.0).all()
    one = ~predicted_spot_mask(sim, radius_px=1.0)
    assert one.sum() == 1
    (i, j), = torch.nonzero(one).tolist()
    nine = ~predicted_spot_mask(sim, radius_px=2.0)          # distance < 2: the 3 x 3 block, not (+-2, 0)
    expected = torch.zeros_like(nine)
    expected[i - 1:i + 2, j - 1:j + 2] = True
    assert torch.equal(nine, expected)


import pytest


@pytest.mark.parametrize("spread", [dict(mosaic_spread_deg=1.0, mosaic_domains=8, mosaic_seed=7),
                                    dict(osc_range_deg=1.0, phi_steps=5)])
def test_more_orientations_mask_a_superset(spread):
    still = ~predicted_spot_mask(simulator(), radius_px=1.5)
    spread_out = ~predicted_spot_mask(simulator(crystal=spread), radius_px=1.5)
    assert not (still & ~spread_out).any(), "a spot of the central orientation was dropped"
    assert spread_out.sum() > still.sum()


def resolution_A(sim):
    """Bragg's law at each pixel centre, d = lambda / (2 sin theta), theta half the angle from the beam."""
    pixels = sim._cached_pixel_coords_meters
    cos_2theta = (pixels / pixels.norm(dim=-1, keepdim=True)) @ sim.incident_beam_direction
    return 1.0 / (2.0 * torch.sqrt((1 - cos_2theta) / 2))          # lambda = 1 A


def test_spots_beyond_dmin_are_not_predicted():
    full = ~predicted_spot_mask(simulator(), radius_px=1.0)
    d = resolution_A(simulator())
    dmin = float(d[full].median())                                 # about half the spots are beyond it
    cut = ~predicted_spot_mask(simulator(beam=dict(dmin=dmin)), radius_px=1.0)
    assert (d[cut] >= dmin * 0.999).all()
    assert cut.sum() < full.sum()
    well_inside = d > dmin * 1.05
    assert torch.equal(cut & well_inside, full & well_inside)


def test_strided_pixels_by_hand():
    keep = torch.ones(4, 4, dtype=torch.bool)
    keep[1, 1] = False
    # kept, row-major: (0,0) (0,1) (0,2) (0,3) (1,0) (1,2) (1,3) (2,0) ... ; every second from the first
    expected = [[0, 0], [0, 2], [1, 0], [1, 3], [2, 1], [2, 3], [3, 1], [3, 3]]
    assert strided_pixels(keep, stride=2).tolist() == expected
    assert strided_pixels(keep, stride=1).shape == (15, 2)


from src.nanobrag_torch.diffuse import GaussianHalo
from src.nanobrag_torch.models.experiment import ExperimentModel


def experiment(mask_array=None):
    detector = DetectorConfig(**DETECTOR, mask_array=mask_array)
    return ExperimentModel(CrystalConfig(**CRYSTAL), detector, BeamConfig(**BEAM), dtype=torch.float64,
                           diffuse=GaussianHalo(amplitude=40.0, sigma=0.15))


def test_experiment_masks_zero_both_components_and_keep_an_existing_mask():
    beamstop = torch.ones(64, 64)
    beamstop[30:34, 30:34] = 0
    model = experiment(mask_array=beamstop)
    masks = model.spot_masks(radius_px=2.0, apply=True)
    assert len(masks) == 1 and masks[0].dtype == torch.bool
    assert torch.equal(masks[0], predicted_spot_mask(simulator(), radius_px=2.0))
    with torch.no_grad():
        model()
    removed = ~masks[0] | (beamstop == 0)
    for name in ("bragg", "diffuse"):
        assert (model.components[name][removed] == 0).all(), name
        assert (model.components[name][~removed] > 0).any(), name


def test_experiment_mask_stays_fixed_while_the_geometry_moves():
    model = experiment()
    before = model.spot_masks(radius_px=2.0, apply=True)[0]
    with torch.no_grad():
        model.crystal_params.delta_misset_raw.add_(torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64))
        model()
    # The halo is positive at every pixel, so the diffuse image is zero exactly where the mask removes.
    assert torch.equal(model.components["diffuse"] == 0, ~before)
    assert not torch.equal(model.spot_masks(radius_px=2.0)[0], before), "the geometry did not move the spots"
    assert not before.requires_grad
