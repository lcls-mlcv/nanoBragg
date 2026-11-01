# Python API guide

Everything below comes from the top-level package. `import nanobrag_torch` is cheap: modules load on first use.

## Simulate an image

```python
from nanobrag_torch import BeamConfig, CrystalConfig, DetectorConfig, simulate

crystal = CrystalConfig(
    cell_a=57.0, cell_b=70.3, cell_c=102.7, cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0,
    N_cells=(10, 10, 10), default_F=100.0,
)
detector = DetectorConfig(distance_mm=150.0, pixel_size_mm=0.1, spixels=64, fpixels=64)
beam = BeamConfig(wavelength_A=1.3, fluence=1e24)

image = simulate(crystal, detector, beam)
```

`image` has shape `(spixels, fpixels)`, in photons per pixel.

Pass structure factors with `F=`:

- a path to an `h k l F` text file: `simulate(crystal, detector, beam, F="P1.hkl")`
- a P1 pair of arrays: `simulate(crystal, detector, beam, F=(indices, amplitudes))`

`device=` and `dtype=` apply to every object: `simulate(crystal, detector, beam, device="cuda", dtype=torch.float64)`. The default is CPU and `torch.float32`.

## From dxtbx

`simulator_from_dxtbx` and `experiment_model_from_dxtbx` take dxtbx detector, beam and crystal objects, so they need a cctbx install. Install cctbx and dxtbx with conda; this package has no pip extra for them.

## Refine

`ExperimentModel` is a `torch.nn.Module` whose parameters are the diffBragg refinables. All of them train by default; `freeze()` and `refine()` choose the subset. Refine in `torch.float64`; `refine_lbfgsb` warns on anything else.

`gaussian_nll` is hopper's target and `poisson_nll` is the counting likelihood. `SerialExperiment` holds many shots that share structure factors. `scripts/refinement_demo.py` is a worked example on synthetic and real stills.

## Output

The library prints nothing. It logs to the `nanobrag_torch` logger: oversampling choices at INFO, reflection ranges at DEBUG. To see them:

```python
import logging

logging.basicConfig()
logging.getLogger("nanobrag_torch").setLevel(logging.INFO)
```

The `nanoBragg` command line prints these messages to stdout at INFO.
