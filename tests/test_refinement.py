"""refine_lbfgsb (plan C6). scripts/refinement_demo.py compares it with hopper on one shot."""
import numpy as np
import pytest
import torch

from nanobrag_torch.config import (
    BeamConfig, CrystalConfig, CrystalShape, DetectorConfig, DetectorConvention, DetectorPivot,
)
from nanobrag_torch.models.experiment import ExperimentModel, SerialExperiment
from nanobrag_torch.utils.losses import gaussian_nll
from nanobrag_torch.utils.refinement import DEFAULT_SCALES, refine_lbfgsb, scaled_parameters
from nanobrag_torch.utils.structure_factors import StructureFactors

pytest.importorskip("scipy")
DTYPE = torch.float64
ROIS = [(2, 12, 3, 9), (15, 24, 10, 22), (26, 31, 1, 6)]


def _model(misset=(5.0, -3.0, 2.0), **kwargs):
    crystal = CrystalConfig(cell_a=30.0, cell_b=35.0, cell_c=40.0, misset_deg=misset, N_cells=(5, 5, 5),
                            default_F=100.0, shape=CrystalShape.GAUSS)
    detector = DetectorConfig(distance_mm=60.0, pixel_size_mm=0.4, spixels=32, fpixels=32, oversample=1,
                              detector_convention=DetectorConvention.MOSFLM, detector_pivot=DetectorPivot.BEAM)
    return ExperimentModel(crystal, detector, BeamConfig(wavelength_A=1.2, fluence=1e30), dtype=DTYPE, **kwargs)


def test_every_group_has_a_scale():
    assert set(DEFAULT_SCALES) == set(ExperimentModel.PARAMETER_GROUPS)


def test_scaled_parameters_are_the_trainable_ones_each_once():
    model = _model()
    model.freeze().refine("misset", "scale")
    assert sorted(scale for _, scale in scaled_parameters(model)) == [0.002, 1.0]
    assert sorted(scale for _, scale in scaled_parameters(model, dict(scale=3.0))) == [0.002, 3.0]


def test_nothing_to_refine_is_an_error():
    model = _model()
    model.freeze()
    with pytest.raises(ValueError, match="nothing to refine"):
        refine_lbfgsb(model, lambda: model().sum())


def test_float32_parameters_warn():
    model = _model(background_rois=ROIS).float()
    model.freeze()
    model.refine("scale")
    with pytest.warns(UserWarning, match="float64"):
        refine_lbfgsb(model, lambda: model.beam_params.delta_log_spot_scale ** 2, maxiter=1)


def test_recovers_a_shot_from_a_bad_scale_and_ncells():
    truth = _model(background_rois=ROIS)
    bump_misset = torch.tensor([0.003, -0.004, 0.002], dtype=DTYPE)
    with torch.no_grad():
        truth.crystal_params.delta_misset_raw.copy_(bump_misset)
        truth.beam_params.delta_log_spot_scale.fill_(-2.0)
        truth.crystal_params.delta_log_N_cells.fill_(-0.3)
        data = truth(roi_only=True)
    model = _model(background_rois=ROIS)
    model.freeze().refine("misset", "scale", "Ncells")
    result = refine_lbfgsb(model, lambda: gaussian_nll(model(roi_only=True), data, 1.0), ftol=1e-14, gtol=1e-10)
    assert result.nit < 200
    # Not to the last digit: log V in hopper's target puts its minimum just under the truth
    # even without noise (here NLL 757.77 fitted, 757.84 at the truth).
    torch.testing.assert_close(model.crystal_params.delta_misset_raw, bump_misset, rtol=0, atol=1e-4)
    torch.testing.assert_close(model.beam_params.delta_log_spot_scale, truth.beam_params.delta_log_spot_scale,
                               rtol=0, atol=0.003)
    torch.testing.assert_close(model.crystal_params.delta_log_N_cells, truth.crystal_params.delta_log_N_cells,
                               rtol=0, atol=0.001)


def test_serial_refinement_shares_fhkl_and_keeps_per_shot_scales():
    hkl = np.array([(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4)  # noqa: E741
                    if (h, k, l) > (0, 0, 0)])
    sf = StructureFactors(hkl, np.random.default_rng(0).uniform(50, 500, len(hkl)))
    missets = [(5.0, -3.0, 2.0), (20.0, 10.0, -5.0), (-30.0, 7.0, 15.0)]

    def serial():
        return SerialExperiment([_model(m, structure_factors=sf, background_rois=ROIS) for m in missets])

    truth = serial()
    scales = (0.1, -0.2, 0.3)
    F = torch.as_tensor(np.random.default_rng(1).uniform(-0.2, 0.2, len(sf)))
    with torch.no_grad():
        truth.shots[0].fhkl_params.delta_log_F.copy_(F)
        for shot, scale in zip(truth.shots, scales):
            shot.beam_params.delta_log_spot_scale.fill_(scale)
        data = truth(roi_only=True)
    model = serial()
    model.freeze().refine("Fhkl", "scale")
    chosen = scaled_parameters(model)
    assert len(chosen) == 4  # one Fhkl, three scales
    # the overall scale is degenerate between shots and F: hold the first shot's
    model.shots[0].freeze("scale")
    with torch.no_grad():
        model.shots[0].beam_params.delta_log_spot_scale.fill_(scales[0])

    def objective():
        return sum(gaussian_nll(pred, d, 1.0) for pred, d in zip(model(roi_only=True), data))

    start = float(objective())
    refine_lbfgsb(model, objective, maxiter=400)
    assert float(objective()) < 0.05 * start
    fitted = [float(shot.beam_params.delta_log_spot_scale) for shot in model.shots]
    np.testing.assert_allclose(fitted, scales, atol=0.02)


def test_a_list_of_terms_matches_their_sum():
    hkl = np.array([(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4)  # noqa: E741
                    if (h, k, l) > (0, 0, 0)])
    sf = StructureFactors(hkl, np.random.default_rng(0).uniform(50, 500, len(hkl)))
    missets = [(5.0, -3.0, 2.0), (20.0, 10.0, -5.0), (-30.0, 7.0, 15.0)]

    def run(as_list):
        model = SerialExperiment(
            [_model(m, structure_factors=sf, background_rois=ROIS) for m in missets], share_detector=True)
        with torch.no_grad():
            data = [d * 1.3 + 1.0 for d in model(roi_only=True)]
        model.freeze().refine("Fhkl", "scale", "misset", "distance")

        def term(i):
            return gaussian_nll(model([i], roi_only=True)[0], data[i], 1.0)

        terms = [lambda i=i: term(i) for i in range(len(missets))]
        result = refine_lbfgsb(model, terms if as_list else lambda: sum(t() for t in terms), maxiter=3)
        return result.fun, result.x, result.jac

    together, apart = run(False), run(True)
    assert apart[0] == pytest.approx(together[0], rel=1e-10)
    np.testing.assert_allclose(apart[1], together[1], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(apart[2], together[2], rtol=1e-6, atol=1e-6)


def test_a_model_of_rois_holds_no_full_image_for_its_mask():
    model = _model(background_rois=ROIS)
    model(roi_only=True)
    assert model._simulator._cached_roi_mask.untyped_storage().nbytes() <= 8


def _two_terms():
    model = _model(background_rois=ROIS)
    model.freeze().refine("scale")
    return model, lambda: model(roi_only=True).sum()


def test_each_term_frees_its_graph_before_the_next_starts():
    model, term = _two_terms()
    earlier = []

    def second():
        with pytest.raises(RuntimeError, match="second time"):
            earlier[-1].backward()
        return term()

    def first():
        earlier.append(term())
        return earlier[-1]

    refine_lbfgsb(model, [first, second], maxiter=1)


def test_a_term_of_nothing_trainable_counts_in_the_loss():
    model, term = _two_terms()
    together = refine_lbfgsb(model, [term, lambda: torch.tensor(5.0, dtype=DTYPE)], maxiter=1)
    model, term = _two_terms()
    assert together.fun == pytest.approx(refine_lbfgsb(model, lambda: term() + 5.0, maxiter=1).fun)


def test_an_empty_list_of_terms_is_an_error():
    model, _ = _two_terms()
    with pytest.raises(ValueError, match="empty"):
        refine_lbfgsb(model, [])


def test_git_commit_names_the_checkout(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path

    from nanobrag_torch.utils import refinement

    repo = Path(refinement.__file__).resolve().parent
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    assert refinement.git_commit().split("+")[0] == head.stdout.strip()

    monkeypatch.setattr(refinement, "__file__", str(tmp_path / "refinement.py"))
    assert refinement.git_commit() == "unknown"


def test_rotation_refinement_records_the_commit():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "scripts" / "rotation_refinement.py").read_text()
    assert "result = dict(commit=git_commit()," in script
