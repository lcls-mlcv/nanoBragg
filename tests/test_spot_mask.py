"""
Spot masks predicted from geometry, for diffuse and joint fits (plan section 2.2, V2).

The reference is the Bragg image itself: with one F for every reflection and GAUSS spots under the
HKL metric (no side lobes, unlike SQUARE's sinc), each recorded spot is a local maximum, so the
predicted centers and the image's maxima must coincide.
"""
import os
os.environ.setdefault("NANOBRAGG_DISABLE_COMPILE", "1")

import pytest
import torch
import torch.nn.functional as nnf

from nanobrag_torch.config import BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, SpotMetric
from nanobrag_torch.diffuse import GaussianHalo
from nanobrag_torch.models.crystal import Crystal
from nanobrag_torch.models.detector import Detector
from nanobrag_torch.models.experiment import ExperimentModel, SerialExperiment
from nanobrag_torch.simulator import Simulator
from nanobrag_torch.spot_mask import predicted_spot_mask, strided_pixels

CRYSTAL = dict(N_cells=(5, 5, 5), default_F=100.0, cell_a=50.0, cell_b=60.0, cell_c=70.0,
               misset_deg=(10.0, 20.0, 30.0), shape=CrystalShape.GAUSS, spot_metric=SpotMetric.HKL)
DETECTOR = dict(spixels=64, fpixels=64, distance_mm=100.0, pixel_size_mm=0.5)
BEAM = dict(wavelength_A=1.0, fluence=1e24)


def simulator(crystal=None, detector=None, beam=None, dtype=torch.float64):
    return Simulator(Crystal(CrystalConfig(**{**CRYSTAL, **(crystal or {})}), dtype=dtype),
                     Detector(DetectorConfig(**{**DETECTOR, **(detector or {})}), dtype=dtype),
                     beam_config=BeamConfig(**{**BEAM, **(beam or {})}), dtype=dtype)


def local_maxima(image, floor):
    """Pixels that are the maximum of their 3x3 neighborhood and above floor."""
    pooled = nnf.max_pool2d(image[None, None], 3, stride=1, padding=1)[0, 0]
    return (image == pooled) & (image > floor)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_centers_are_the_bragg_images_maxima(dtype):
    sim = simulator(dtype=dtype)
    image = sim.run(oversample=1)
    # A GAUSS-HKL spot n sds from its pixel peaks at exp(-n^2/2) of a centered one: 1.1% at the default
    # 3 sds. Above 2% (2.8 sds) every recorded spot must be predicted.
    maxima = local_maxima(image, 2e-2 * image.max())
    assert maxima.sum() >= 5, "the test geometry must record several spots"
    centers = ~predicted_spot_mask(sim, radius_px=1.0)
    # A spot on a pixel corner (here 000, the direct beam) records equal maxima in up to four pixels, of which
    # one is its center: r = 1.5 reaches the diagonal neighbors.
    assert not (maxima & predicted_spot_mask(sim, radius_px=1.5)).any(), "a recorded spot was not predicted"
    on_maxima = local_maxima(image, 0.0)[centers].float().mean()
    assert on_maxima > 0.9, f"only {on_maxima:.2f} of the predicted centers are maxima of the image"


def test_only_the_direct_beam_spot_by_hand():
    # A 4 A cubic cell: the first reflection needs |s| = 0.25/A, past this detector's corners (0.225/A),
    # so the one spot is 000 at the direct beam, and the mask is a disk about one pixel. The beam falls on
    # the corner of four pixels, equally near 000: a spot still has one center.
    sim = simulator(crystal=dict(cell_a=4.0, cell_b=4.0, cell_c=4.0, misset_deg=(0.0, 0.0, 0.0)))
    assert predicted_spot_mask(sim, radius_px=0.0).all()
    one = ~predicted_spot_mask(sim, radius_px=1.0)
    assert one.sum() == 1
    (i, j), = torch.nonzero(one).tolist()
    nine = ~predicted_spot_mask(sim, radius_px=2.0)          # distance < 2: the 3 x 3 block, not (+-2, 0)
    expected = torch.zeros_like(nine)
    expected[i - 1:i + 2, j - 1:j + 2] = True
    assert torch.equal(nine, expected)


@pytest.mark.parametrize("spread", [dict(mosaic_spread_deg=1.0, mosaic_domains=8, mosaic_seed=7),
                                    dict(osc_range_deg=1.0, phi_steps=5)])
def test_more_orientations_mask_a_superset(spread):
    still = ~predicted_spot_mask(simulator(), radius_px=1.5)
    spread_out = ~predicted_spot_mask(simulator(crystal=spread), radius_px=1.5)
    assert not (still & ~spread_out).any(), "a spot of the central orientation was dropped"
    assert spread_out.sum() > still.sum()


def resolution_A(sim):
    """Bragg's law at each pixel center, d = lambda / (2 sin theta), theta half the angle from the beam."""
    pixels = sim.detector.get_pixel_coords()
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


def experiment(mask_array=None, diffuse=None, **kwargs):
    detector = DetectorConfig(**DETECTOR, mask_array=mask_array)
    return ExperimentModel(CrystalConfig(**CRYSTAL), detector, BeamConfig(**BEAM), dtype=torch.float64,
                           diffuse=diffuse or GaussianHalo(amplitude=40.0, sigma=0.15), **kwargs)


def test_experiment_masks_zero_both_components_and_keep_an_existing_mask():
    beamstop = torch.ones(64, 64)
    beamstop[30:34, 30:34] = 0
    model = experiment(mask_array=beamstop)
    masks = model.spot_masks(radius_px=2.0)
    assert len(masks) == 1 and masks[0].dtype == torch.bool
    model.set_mask_arrays([masks[0] & (beamstop > 0)])
    # wiring: the model's geometry is the bare Simulator's
    assert torch.equal(masks[0], predicted_spot_mask(simulator(), radius_px=2.0))
    with torch.no_grad():
        model()
    removed = ~masks[0] | (beamstop == 0)
    for name in ("bragg", "diffuse"):
        assert (model.components[name][removed] == 0).all(), name
        assert (model.components[name][~removed] > 0).any(), name


def test_experiment_mask_stays_fixed_while_the_geometry_moves():
    model = experiment()
    before = model.spot_masks(radius_px=2.0)[0]
    model.set_mask_arrays([before])
    with torch.no_grad():
        dict(model.named_parameters())["crystal_params.delta_misset_raw"].add_(
            torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64))
        model()
    # The halo is positive at every pixel, so the diffuse image is zero exactly where the mask removes.
    assert torch.equal(model.components["diffuse"] == 0, ~before)
    assert not torch.equal(model.spot_masks(radius_px=2.0)[0], before), "the geometry did not move the spots"
    assert not before.requires_grad


@pytest.mark.parametrize("dmin", [0.0, 7.0])
def test_several_sources_mask_the_union_of_each_alone(dmin):
    # Two sources on the default beam at 1.0 and 1.3 A, with phi steps: each view must pair its pixels with its
    # own source's wavelength, so the mask is the union of the two single-source masks (also under dmin).
    crystal = dict(osc_range_deg=1.0, phi_steps=3)
    directions = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    both = dict(source_directions=directions, source_wavelengths=torch.tensor([1.0e-10, 1.3e-10], dtype=torch.float64),
                source_weights=torch.ones(2, dtype=torch.float64), dmin=dmin)
    together = ~predicted_spot_mask(simulator(crystal=crystal, beam=both), radius_px=1.0)
    alone = [~predicted_spot_mask(simulator(crystal=crystal, beam=dict(wavelength_A=w, dmin=dmin)), radius_px=1.0)
             for w in (1.0, 1.3)]
    assert alone[0].sum() > 0 and alone[1].sum() > 0 and not torch.equal(alone[0], alone[1])
    assert torch.equal(together, alone[0] | alone[1])


def two_panels(mask_array=None, **kwargs):
    # Two panels, the second shifted sideways: the multi-panel ROI path caches its pixels (with their masks)
    panels = [DetectorConfig(**DETECTOR, beam_center_f=f, beam_center_s=16.0, mask_array=mask_array)
              for f in (16.0, -16.0)]
    return ExperimentModel(CrystalConfig(**CRYSTAL), panels, BeamConfig(**BEAM), dtype=torch.float64,
                           diffuse=GaussianHalo(amplitude=40.0, sigma=0.15), **kwargs)


@pytest.mark.parametrize("make", [lambda: experiment(background_rois=[(0, 63, 0, 63)]),
                                  lambda: two_panels(background_rois=[(0, 63, 0, 63)] * 2, background_panels=[0, 1])])
def test_masks_set_after_a_forward_reach_the_roi_path_and_replace_each_other(make):
    model = make().freeze()                                       # geometry given, as in a diffuse fit
    with torch.no_grad():
        assert (model(roi_only=True) > 0).all()                    # the ROI pixels are cached now
        for radius in (6.0, 2.0):                                  # a sweep: each mask replaces the last
            masks = model.spot_masks(radius_px=radius)
            model.set_mask_arrays(masks)
            values = model(roi_only=True)
            kept = torch.cat([m[tuple(model.background_params.pixels[model.background_params.panel == i].T)]
                              for i, m in enumerate(masks)])
            assert (~kept).any() and kept.any(), radius
            assert torch.equal(values == 0, ~kept), radius


def test_shots_sharing_a_detector_refuse_a_mask_of_their_own():
    halo = GaussianHalo(amplitude=40.0, sigma=0.15)
    shots = [experiment(diffuse=halo), experiment(diffuse=halo)]
    serial = SerialExperiment(shots, share_detector=True)
    for shot in serial.shots:
        with pytest.raises(ValueError, match="shared"):
            shot.set_mask_arrays(shot.spot_masks(radius_px=2.0))


LYSOZYME_LIKE = dict(cell_a=79.0, cell_b=79.0, cell_c=38.0, N_cells=(20, 20, 20), misset_deg=(5.0, 7.0, 11.0))
FINE = dict(spixels=128, fpixels=128, distance_mm=100.0, pixel_size_mm=0.4)


def test_spots_smaller_than_a_pixel_are_predicted():
    # V2a's regime: a spot (sd 0.397/20 = 0.02 in h) far smaller than a pixel (about 0.3 in h), recorded by the
    # kernel's subpixels. Every spot of the oversampled image must be masked, and the mask must hold its photons.
    sim = simulator(crystal=LYSOZYME_LIKE, detector=FINE)
    image = sim.run(oversample=3)
    maxima = local_maxima(image, 2e-2 * image.max())
    assert maxima.sum() >= 20
    keep = predicted_spot_mask(sim, radius_px=2.0)
    assert not (maxima & keep).any(), f"{int((maxima & keep).sum())} of {int(maxima.sum())} recorded spots kept"
    masked_share = float(image[~keep].sum() / image.sum())
    assert masked_share > 0.95, f"the mask holds only {masked_share:.3f} of the Bragg photons"


def test_curved_detector_spots_are_predicted():
    sim = simulator(detector=dict(curved_detector=True, distance_mm=50.0, pixel_size_mm=0.25))
    image = sim.run(oversample=1)
    maxima = local_maxima(image, 2e-2 * image.max())
    assert maxima.sum() >= 5
    keep = predicted_spot_mask(sim, radius_px=1.5)
    assert not (maxima & keep).any(), f"{int((maxima & keep).sum())} of {int(maxima.sum())} recorded spots kept"


@pytest.mark.parametrize("crystal", [dict(spot_metric=SpotMetric.RECIPROCAL), dict(shape=CrystalShape.SQUARE),
                                     dict(fudge=0.25), dict(N_cells_def=(1.0, 0.0, 0.0))])
def test_spot_widths_other_than_gauss_hkl_are_refused(crystal):
    with pytest.raises(ValueError, match="GAUSS"):
        predicted_spot_mask(simulator(crystal=crystal), radius_px=1.0)


@pytest.mark.parametrize("panels", [1, 2])
def test_a_beamstop_edited_in_place_reaches_the_roi_path(panels):
    # A caller's DetectorConfig mask is the model's own tensor, so editing it in place must reach both paths
    # (Simulator tracks a mask's version as well as its identity; the multi-panel ROI cache must too).
    beamstop = torch.ones(64, 64)
    if panels == 1:
        model = experiment(mask_array=beamstop, background_rois=[(0, 63, 0, 63)])
    else:
        model = two_panels(mask_array=beamstop, background_rois=[(0, 63, 0, 63)] * 2, background_panels=[0, 1])
    model.freeze()
    with torch.no_grad():
        assert (model(roi_only=True) > 0).all()
        beamstop[10:20, 10:20] = 0
        values = model(roi_only=True)
    slow, fast = model.background_params.pixels.T
    shadowed = (slow >= 10) & (slow < 20) & (fast >= 10) & (fast < 20)
    assert torch.equal(values == 0, shadowed)            # the one beamstop tensor is on every panel


def test_set_mask_arrays_copies_and_checks_the_shape():
    model = experiment()
    mask = torch.ones(64, 64, dtype=torch.bool)
    model.set_mask_arrays([mask])
    mask[:] = False                                            # the caller's tensor is not the model's
    with torch.no_grad():
        model()
    assert (model.components["diffuse"] > 0).all()
    with pytest.raises(ValueError, match="shape"):
        model.set_mask_arrays([torch.ones(10, 10)])


def test_pixels_spanning_much_of_a_miller_unit_are_refused():
    # Each pixel considers the lattice point nearest its center hkl, which misses spots once a pixel spans about a
    # Miller unit (a 250 A cell with 0.5 mm pixels at 100 mm: about 1.2 a pixel). Refused rather than mis-masked.
    big = simulator(crystal=dict(cell_a=250.0, cell_b=250.0, cell_c=250.0, N_cells=(20, 20, 20)))
    with pytest.raises(ValueError, match="Miller"):
        predicted_spot_mask(big, radius_px=2.0)
