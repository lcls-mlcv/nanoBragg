# Plan: Bragg and diffuse in one differentiable model

Status: plan settled 2026-11-04. D1 (the hook, branch `feat/diffuse-joint` off `torch`
3d863fc) is open as PR #72. D2 (ExperimentModel, `feat/diffuse-experiment`, stacked on D1)
is implemented and has been through the audit and review of 2026-11-05. Its V1 gate,
`scripts/diffuse/v1_toy_recovery.py`, runs at small size in the tests; the full size runs on
the GPU gate machine. D3 (`DiffuseGrid`, `feat/diffuse-grid`, stacked on D2) is implemented; its
grid-vs-exact measurement and the decision it informs are in section 2.2.

Names: nanoBragg = the C program, diffBragg = simtbx, torchBragg = this package
(`nanobrag_torch`), eryx = the diffuse-scattering library (PyTorch, GPLv2).

## 0. Principles

1. **Separable first, joint second.** Each side keeps working alone:
   - torchBragg with no diffuse model is the code that is in `torch` today. Images and
     gradients are byte-identical, and the parity matrix is unchanged.
   - eryx fits merged 3D maps without torchBragg, as it does now. Its map-space results
     (lysozyme, insulin) must not change.
   - A joint model reports its components separately (`bragg`, `diffuse`, `background`).
     Either side can be frozen or left out. torchBragg can render a diffuse-only image, so
     each half can be compared with programs that do only Bragg (diffBragg/hopper, DIALS)
     or only diffuse (mdx2, mdx-lib, EVAL/DECaF).
2. **C-first holds with diffuse off.** The diffuse term is an extension that does not exist
   in nanoBragg.c. It lives outside the compiled kernel and adds no work, parameter or state
   when it is absent. The only divergences from C stay the ones in
   `docs/bugs/verified_c_bugs.md`.
3. **Measure before fixing.** Every scale, normalization and gradient claim below gets a test
   that fails before the change and passes after.
4. **Licenses.** torchBragg is MIT and eryx is GPLv2. No eryx code is copied into
   torchBragg. torchBragg defines a small protocol, and eryx implements it, depending on
   torchBragg. mdx-lib (CC BY-NC-SA) is cited, never copied.
5. **No unpublished data.** Validation uses synthetic cases and public data: lysozyme SBGrid
   747 frames and the CXIDB 128 map, and insulin from Zenodo 10515006.

## 1. Where things stand

### torchBragg (`torch` 3d863fc)
- `Simulator` is the C-parity port: compiled per-sample kernel, oversampling, sources, mosaic,
  phi, sensor layers and polarization.
- `ExperimentModel` holds the refinable state as zero-offset `nn.Parameter`s in groups:
  cell, misset, distance, detz_shift, detector_rotation, twotheta, beam_center, fluence,
  scale (G), Ncells, Ndef, panel_rotation, panel_shift, eta, Fhkl (per-ASU log F), background
  (diffBragg tilt planes), wavelength and source_weights.
  - It supports `refine()`/`freeze()` and `roi_only` (pixels inside the background ROIs).
  - Multi-panel models run as one kernel call.
  - `bragg_only` leaves out the background.
- `SerialExperiment` shares one Fhkl across shots and, optionally, one detector.
- Supporting pieces: `utils/losses` (gaussian_nll as hopper, poisson_nll), `utils/restraints`
  (hopper's centers and betas), `utils/refinement.refine_lbfgsb` (scaled L-BFGS-B over
  closures), and the .expt/.refl export (#71).
- **The diffuse hook is not in `torch`.** Branch `diffuse-on-tail` (31d0b2c and 3f653f1)
  adds it, plus the MPS fix 7b344a4 from a local branch (rebased in D1 as 4defbe5, 6995bd8,
  0805f4e). All three cherry-pick onto 3d863fc
  with no conflict, and `tests/test_diffuse_term.py` passes there (20 tests). What it does:
  - `Simulator(diffuse=fn, diffuse_orientations="all"|"first")`, where
    `fn(hkl (n,3) float64) -> (n,)` is scattering per unit cell in electrons².
  - It evaluates at pixel centers, outside the kernel, in float64 (float32 on MPS).
  - photons = N_cells · mean over sources and orientations(I_cell · polar) · Ω · capture ·
    r_e² · fluence · spot_scale. This is the Bragg tail without F_latt².
  - **Changed in D1 after review:** that pixel-center tail departed from Bragg's per-sample
    rules. Measured gaps:
    - sensor without `oversample_thick`: up to 1.6–1.9×;
    - dmin ignored;
    - `oversample` ignored: 19% for a σ = 0.05 halo;
    - static `kahn_factor`;
    - a shape bug with `panel_pixels` and a sensor.

    The term now goes through Bragg's own path. The uncompiled kernel takes
    N_cells·I(hkl) in place of (F_cell·F_latt)² at every sample, and hkl arrives in the
    run's dtype. A one-cell Bragg image with F = √I(hkl) is the test oracle, to 1e-12, in 10
    configurations.
  - It fills `Simulator.components`, and a ROI mask applies to both parts.
  - `ExperimentModel` does not use it: it builds its own Simulators, has no diffuse group,
    and the ROI path ignores it. (Done in D2, section 4.)

### eryx (`mdx2-compare` 4ab7329)
- **Detector-facing model:** `total_scattering.TotalScattering(crystal, network, share,
  mesh, n_cells, solvent, residual, ...)`.
  - It is an `nn.Module`; `forward(hkl) -> (n,)` gives diffuse per cell in e² at fractional
    hkl.
  - Components: one-phonon from an atomic GNM over a contact network (lattice dynamics,
    C(k) read off an m³ mesh, exact k near lattice points with a floor of 0.5/max(n_cells)),
    plus a residual-disorder term.
  - The absolute level is set by the B-factors through `share` ("budget" = lattice U equals
    a share of U_B).
  - Parameters: log γ per contact class (restraint Σ log² γ fixes the gauge), solvent log k
    and log B, optional log share, and extra_U (one uniform U shift for Bragg and diffuse).
- **Map-facing models**, the ones behind the published ladders:
  - `contact_springs[_multi]`: Meisburger-style rigid bodies with springs per interface or
    contact; `cell_transform` gives a fixed W(q), and the intensity is Wᴴ C W.
  - `ShellFit` projects out a scale per shell and an offset per radial bin, so these
    intensities are relative.
- **Not refinable on either path:** coordinates, per-atom B and ANISOU, occupancy, cell
  and orientation.
  - The cell comes only from CRYST1.
  - `Sampling.F`, `amplitudes()` and `cell_transform()` run under no_grad.
  - The composable path reads b_iso only (ANISOU ignored).
- **Device:** eigenproblems run on CPU in float64 (`_SpectralMatrix` has a first-order-only
  eigh backward). Arrays indexed by q go on MPS (float32) or CUDA.

### How we coupled them, and what changed since
- eryx `tests/test_torchbragg_integration.py` passes a `TotalScattering` as `diffuse=`.
- `write_bragg()` fills torchBragg's `hkl_data` with eryx's |F|, as a detached numpy grid.
- Bragg and diffuse share one absolute scale (G, fluence, N_cells), with one B shift for
  both (eryx 9301099 and 2b17e38).
- Gradients reach only the eryx parameters:
  - `TotalScattering._prepared` detaches hkl (`total_scattering.py:66`), so orientation, cell
    and detector get no gradient through the diffuse term.
  - The Bragg F is a constant grid.
- eryx imports torchBragg through `sys.path` (`src.nanobrag_torch`), which breaks in other
  checkouts: the test skips silently in the mdx2-compare worktree.
- `_prepared` caches a `Sampling` keyed on the hkl bytes, with no bound. Every orientation,
  phi step or parameter step that moves the pixels adds an entry: fine for one image, a leak
  under refinement.
- Since then:
  - torchBragg gained Ndef, panel-group geometry, restraints, serial shared-Fhkl scaling and
    export.
  - eryx gained Poisson-weighted `merge_map.py`, the per-term ladder (bodies, ANISOU, water,
    F4/F5 solvent), and `ShellFit(direct=True)`.
  - Both ladders are map-space and relative-scale. None of the new eryx terms is reachable
    from the detector-facing `TotalScattering`.

## 2. Coupling design

**Recommendation: a protocol in torchBragg, with the adapter in eryx.**

```python
# nanobrag_torch/diffuse.py (MIT)
class DiffuseModel(Protocol):          # in practice an nn.Module
    def __call__(self, hkl: Tensor) -> Tensor: ...       # (n,3) fractional -> (n,) e^2 per cell
    # added in D3, optional (not in the runtime check):
    # def grid(self) -> "DiffuseGrid": ...               # values on a reciprocal grid, see 2.2
    # DiffuseGrid.from_model(model, step, d_min=, cell=) samples any model; DiffuseGrid.grid() is itself
```

- torchBragg ships only the protocol, reference toy models and a grid interpolator:
  - toy models: `ConstantDiffuse` and `GaussianHalo` (a Gaussian about every lattice point);
  - torchBragg's tests and CI use only these, with no eryx dependency.
- eryx ships `eryx.torchbragg.Diffuse(nn.Module)`, an adapter over TotalScattering or the
  multi-body model. eryx gains a `pyproject.toml`, with nanobrag-torch as an optional
  dependency (`pip install -e`), which replaces the sys.path import.
- Rejected:
  - eryx as a hard dependency of torchBragg: GPL, gemmi and a conda stack in an MIT package
    aimed at LCLS users.
  - A vendored adapter: GPL code in an MIT repo.

### 2.1 Units and scale
- Both terms are per unit cell in electrons² and share r_e²·fluence·G·Ω·polar·capture.
- Integrated over a Bragg peak in hkl, the SQUARE lattice factor gives N_cells·|F|²
  (∫sincg² = N per axis). The diffuse term carries N_cells·I_d per unit hkl volume. So the
  diffuse-to-Bragg ratio does not depend on N_cells, as it should for a mosaic-block size.
- **To measure, not assume:** the GAUSS and ROUND shapes (diffBragg's default) have their
  own integrals, so the tie may be off by a constant there. Test D2-T1 integrates F_latt² per
  shape and records the factor. If it is not 1, the tie divides by it, recorded as an
  extension rather than a change to the C path.
- **Measured in D2** (`tests/test_diffuse_shape_integral.py`, through the kernel function, one
  peak in its cell of hkl, factor = ∫F_latt² / N_cells):

  | Shape | Factor |
  |---|---|
  | SQUARE | 1 (to 1e-15) |
  | GAUSS, `spot_metric=HKL` (diffBragg, cctbx) | (0.315π)^(3/2) = 0.9844 |
  | TOPHAT, HKL | (4/3)π·0.63³ = 1.0474 |
  | ROUND | 0.8972 (N = 5), 0.9496 (10), 0.9748 (20): 1 untruncated (0.7236² = π/6), but the sinc3² tail is cut at the cell, about 1 − 0.5/N |
  | GAUSS or TOPHAT, `RECIPROCAL` (the default, bl831 C) | the HKL factor × V/N_cells², V in Å³: 2.65 for 30×35×40 Å and 5³ cells, 0.23 for 237,000 Å³ and 10³ |

  The C path and the Simulator are unchanged (N_cells·I_d for every shape).
- **Decided 2026-11-05: the tie is corrected for the shape in `ExperimentModel`.** The
  diffuse term is multiplied by the factor, `nanobrag_torch.diffuse.bragg_peak_factor`, so
  integrated Bragg over diffuse is |F|²/I_d for every shape. SQUARE is 1 exactly, so it
  behaves as before.
  - The factor is the integral over the peak's own cell of hkl, |h − h0| < 1/2, as the
    kernel assigns points to their nearest lattice point.
    - GAUSS and TOPHAT: the closed forms (0.315π/fudge)^(3/2) and (4/3)π(0.3969/fudge)^(3/2),
      over √det M, apply where the cell cuts under 1e-10 of the Gaussian or none of the ball.
      Elsewhere it computes the cut integral (Gauss–Legendre in 3D, or the ball's chord
      integrated over h, k).
    - ROUND has no closed form. It is a radial integral of sinc3² times the area of the sphere
      inside the cell, with work linear in N: 12 ms at N = 500.
    - All of these are differentiable in N, the cell and Ndef.
    - Audit, 2026-11-05: the first version ignored the cut. Under RECIPROCAL that was 9% off
      for 79×79×38 Å with N = 5, and a factor of 50 off for a 100 Å cell with N = 3.
  - It is recomputed at every forward from the current N_cells and cell, with their gradients.
    It is not folded into k_d's starting value because it moves as they refine. It shows as
    `model.diffuse_shape_factor()`, and `diffuse_shape_factor=False` sets it to 1.
  - Against the kernel's measured integral (`tests/test_diffuse_shape_integral.py`):
    - SQUARE 1e-15, GAUSS-HKL 1e-9 (4e-5 at N = 1–2);
    - ROUND 5e-6;
    - RECIPROCAL GAUSS 5e-5 where the cell cuts the peak; there the kernel's midpoint grid is
      the limit, converging as h² towards the factor;
    - TOPHAT 1e-5 to 2e-3, the grid's ball.
- **The RECIPROCAL metric and joint fits.** Under bl831's metric the factor is about
  V/N_cells² while the peak sits inside the cell, so with k_d frozen the integrated diffuse
  term scales as V/N_cells, against N_cells for the Bragg term. That is faithful to C's
  metric, whose radius is not dimensionless, but unphysical when N_cells or the cell refines:
  the fit moves Bragg and diffuse pixels together in a way no crystal does.
  **Recommendation: use `spot_metric=SpotMetric.HKL` (diffBragg's) for joint fits.** There the
  factor is a constant: 0.9844 for GAUSS, 1 for SQUARE.
- **Relative scale `k_d`** (group `diffuse_scale`, log, default frozen at 1):
  - Frozen, the absolute tie is a physical constraint.
  - Freed, the halves separate: a diffuse fit that does not lean on the Bragg scale.
- Isotropic scattering (water, air, Compton, solvent ring, the isotropic part of TDS) is
  neither model's business in a detector fit.
  - It needs a **radial background** b(|q|): a refinable spline per shot or sweep, a new
    background kind beside the tilt planes.
  - Its gauge against eryx's isotropic content is fixed by projecting the shell mean out of
    the diffuse model, as ShellFit does, so b(|q|) owns everything radial.

### 2.2 Per-pixel vs reciprocal grid
| | Per pixel (the hook today) | Reciprocal grid + interpolation |
|---|---|---|
| eryx work per step | one evaluation per pixel × orientation × source; Sampling rebuilt per hkl set | one evaluation of the ASU grid, independent of geometry |
| Geometry gradients | need eryx to keep hkl in the graph | free: the interpolation is torch, differentiable in hkl and in the grid values |
| Exactness | exact at pixel centers | interpolation error near lattice points, where halos are sharp |
| Memory | cache per hkl set, unbounded today | one grid; for lysozyme to 2 Å at 13×11×11 it is about 1e7–1e8 voxels before symmetry |

**Recommendation:** use the grid for refinement and the exact per-pixel path as the
reference.
- Gate: grid vs exact on synthetic images, error well under the Poisson noise of the diffuse
  term.
- The grid is stored on the Laue ASU and expanded through the same integer point-group
  gather that `StructureFactors` already uses.
- It can be finer near integer hkl, or carry an analytic 1/k² core: decide after measuring
  the error.
- **Done in D3** (`nanobrag_torch/diffuse_grid.py`, `DiffuseGrid`):
  - Values at hkl = origin + step·(i, j, k), eryx's HKLGrid layout, taken from its fields
    (`from_values`, `from_hkl_grid`) without importing eryx, or sampled from any model
    (`from_model(model, step, d_min=, cell=)`, extent a/d_min per axis plus a node, in chunks).
  - A parameter (group `diffuse`) or, with `refinable=False`, a buffer.
  - P1 storage, or one value per orbit of the point group's integer operators (with Friedel's
    law), the operators StructureFactors uses (`point_group_operators`). On a step-1 grid the
    orbits are expand_to_p1's. A node's value is found from the largest key of its mates, so a
    symmetric grid holds the values and one int64 key per orbit, no index box.
  - Zero outside the box (the trilinear interpolant of the values padded with zeros, so continuous),
    or `outside="error"`.
  - Trilinear in hkl and the values. The gradient in hkl is constant within a cell; every
    finite-difference test asserts that no sample changes cell.
- **Measured, 2026-11-05** (`scripts/diffuse/grid_vs_exact.py error --size 64`: the V1 toy's
  diffuse-only shot, the amplitude scaled as σ⁻³ from V1's 128 at σ = 0.3, mmm ASU). Deviance is
  2Σ(g − μ + μ log(μ/g)) of the grid image g against the exact μ, the whole image's
  detectability (about 530 diffuse photons over the image at σ = 0.3):

  | σ | step | step/σ | max rel | rms rel | max \|Δ\|/√μ | deviance |
  |---|---|---|---|---|---|---|
  | 0.3 | 1/4 | 0.83 | 0.16 | 0.11 | 0.094 | 5.4 |
  | 0.3 | 1/8 | 0.42 | 0.057 | 0.032 | 0.032 | 0.42 |
  | 0.3 | 1/16 | 0.21 | 0.016 | 0.0083 | 0.0087 | 0.028 |
  | 0.3 | 1/32 | 0.10 | 0.0039 | 0.0020 | 0.0021 | 0.0017 |
  | 0.2 | 1/16 | 0.31 | 0.034 | 0.018 | 0.035 | 0.27 |
  | 0.2 | 1/32 | 0.16 | 0.0085 | 0.0045 | 0.0086 | 0.016 |
  | 0.1 | 1/16 | 0.62 | 0.11 | 0.071 | 0.33 | 8.4 |
  | 0.1 | 1/32 | 0.31 | 0.029 | 0.017 | 0.085 | 0.54 |
  | 0.05 | 1/32 | 0.62 | 0.074 | 0.052 | 0.54 | 9.4 |
  | smooth | 1/4 | | 0.020 | 0.0097 | 0.012 | 0.045 |
  | smooth | 1/8 | | 0.0047 | 0.0025 | 0.0030 | 0.0029 |

  - Below step/σ ≈ 0.6 the error falls as step², as the trilinear bound says. Above it the halo is
    aliased and the error does not fall (step/σ 1.25 is no better than 2.5). The tests check the
    bound itself: (step²/8)·Σ max |∂²I| times the image of a constant.
  - Per pixel the grid error is under the Poisson sd once step/σ ≤ 0.6. Over the whole image it
    is undetectable (deviance < 1) once step/σ ≤ 0.3–0.4, at about 500 photons.
  - V1 with `--grid` (data from the exact halo, its grid as the model, k_d in place of the
    amplitude): the bias goes into k_d and falls as step², 1.44, 0.40 and 0.097 Fisher sds at
    step/σ 0.42, 0.21 and 0.10. Misset and cell stay under 0.14 sd at every step, so geometry is
    not where the grid's error shows.
- **Decided (decision 2), on these numbers:**
  - Uniform trilinear is enough for geometry and for the V1 toy. `--grid` defaults to step 1/32
    (step/σ = 0.10): every bias is under 0.1 sd, and the Poisson gate gives the exact halo's
    worst |z| (3.12, seeds 1–3, 24²).
  - No refined or analytic core in D3. The numbers do say one is needed at real photon counts:
    the deviance and the amplitude bias in sds grow as the diffuse photons do (bias ∝
    step²·√photons). At 1e5–1e6 diffuse photons a shot, step/σ must be about 0.02–0.05. For a
    halo core as sharp as eryx's k floor (0.5/N_cells, σ of about 0.05), uniform nodes would
    need a step near 1/500, which is not affordable. The refined core is decided on V2, with
    eryx's real halo shape.
- **Memory** (`grid_vs_exact.py memory`: lysozyme 79×79×38 Å to 2 Å, P4₃2₁2 so Laue 4/mmm, the box
  ±(a/d_min + one node)):

  | step | nodes | ASU values (held) | P1 (held) | peak RSS, ASU / P1 | build, ASU / P1 | lookup per point, ASU / P1 |
  |---|---|---|---|---|---|---|
  | 1/4 | 15.8M | 1.00M (15 MB) | 120 MB | 1.19 / 0.48 GB | 1.7 / 0.25 s | 0.92 / 0.12 µs |
  | 1/8 | 124M | 7.8M (119 MB) | 944 MB | 1.37 / 1.36 GB | 13 / 1.8 s | 0.99 / 0.13 µs |

  - Recommended: 1/8 on the ASU (119 MB) as the uniform base for lysozyme, with the core decided
    on V2.
  - The ASU build peak is chunk work: 2¹⁹ nodes × 16 operators, transient.
  - The ASU lookup is about 8× the P1's (16 operators per corner).
- Pixels: diffuse fits do not need every pixel. A ROI set with a strided subset between
  spots (mask out ±n px around predicted spots, then stride) keeps one shot at 1e4–1e5
  pixels.

## 3. Joint objective

L = Σ_shots NLL(pixels | bragg + k_d·diffuse + background) + restraints.

- In image space both terms predict the same counts, so there is **no Bragg/diffuse weight
  to choose**: the noise model weights them. Poisson NLL is the default for diffuse pixels,
  which sit at a few photons, and gaussian_nll (hopper) stays available.
- A map-space objective (eryx ShellFit on merged maps) stays the separable reference. It is
  not mixed into L by an arbitrary weight. If a hybrid is ever wanted, its weight comes from
  the map σ, which is relative today, so that needs a calibration first.

| Parameter | Bragg | Diffuse | Joint |
|---|---|---|---|
| G (scale), fluence | yes | via the tie | shared (k_d frozen) |
| N_cells | spot shape | k floor (eryx n_cells) | open: tie eryx's floor to N_cells, or keep it fixed |
| cell | yes | no today (CRYST1) | shared once eryx takes A* as a tensor (E2) |
| misset, phi, detector, beam, spectrum | yes | through hkl | shared (grid path gives this for free) |
| eta, Ndef | yes | through orientation averaging | shared |
| Fhkl (free per ASU) | yes | no | Bragg only |
| B / U (overall, or extra_U) | through F if F comes from a model | sets the absolute level | shared only when F comes from the model (mode b below) |
| springs γ, solvent k/B, residual, LLM σ | — | yes | diffuse only |
| background planes, radial b(\|q\|) | yes | yes | shared nuisance |

How the Bragg F is supplied:
- **a. Free per-ASU Fhkl** (diffBragg's way). This is the default for comparison with
  hopper.
- **b. F from the structure model**: eryx `amplitudes()`, made differentiable, gives one
  structure for both halves and simulates full maps from a PDB.
- **c. b times a restrained per-ASU correction.** Later, if wanted.

Gauges:
- G vs k_d: fixed by freezing k_d, or by the tie.
- γ overall vs the B budget: eryx already restrains it with Σ log² γ.
- Radial b(|q|) vs isotropic diffuse: fixed by projection (2.1).
- Fhkl vs G: as now, through hopper restraints.
- Wavelength vs cell: as now, never both.

## 4. eryx refinables in torchBragg

`ExperimentModel(diffuse=module)` registers the module as a submodule:
- **Group `diffuse`** holds all of the module's own trainable parameters, and appears only
  when a model is given. `PARAMETER_GROUPS` gains it conditionally, and the partition test
  learns about optional groups.
- **Group `diffuse_scale`** holds k_d, frozen by default.
- **Group `radial_background`** holds the spline knots.
- `SerialExperiment` shares one diffuse module across shots, as it shares Fhkl. The radial
  background stays per shot.
- **Done in D2:** the module is `diffuse_params.model` and k_d is
  `diffuse_params.delta_log_scale`; the groups are in the instance's `PARAMETER_GROUPS`
  (`ExperimentModel.OPTIONAL_GROUPS` names them). A parameter the module froze itself is
  unrefinable. Shots share the whole block, k_d included: one diffuse-to-Bragg ratio.
  `model.components` gives bragg, diffuse, background and total, on the ROI path too.
  `bragg_only=True` leaves out the diffuse term as well as the background (decided
  2026-11-05), so `roi_centroids` and the export see the Bragg term alone.
  `param_init="frozen"` freezes the given module's parameters in place.

Gradient checks (/tdd, CPU float64, central finite differences plus gradcheck on a 16² case):
- each eryx parameter (log γ per class, solvent log k and log B, log share, extra_U, LLM σ)
  through the full image;
- k_d and the radial knots;
- through the diffuse term into misset, cell, distance, beam center and wavelength, on the
  grid path and, after E1, the exact path;
- one CUDA and one MPS float32 check against CPU;
- a check that no gradient needs `create_graph`: eryx's eigh backward is first-order only,
  so L-BFGS is fine and Hessian-vector products are not.

## 5. Validation ladder

Each rung uses the result of the one before; real data only after synthetic recovery.

- **V0 Off is off.**
  - Parity matrix (306 cases) unchanged.
  - `scripts/diffuse/off_is_identical.py` covers:
    - 9 Simulator cases, among them `pixels=`, ROUND and TOPHAT, with gradients for distance,
      cell a and wavelength;
    - 9 ExperimentModel cases and a SerialExperiment, with gradients for every trainable
      parameter and their `state_dict` keys. The model cases are whole image, ROI,
      multi-panel, ROI across panels, GAUSS with mosaic, ROUND, TOPHAT, and `bragg_only` on
      the whole image and on the ROI.
  - It runs in float32 and float64, eager and compiled, writes on the base tree and on the
    branch, and compares with `np.array_equal`.
  - The .expt/.refl export reads only model state, which this gate already covers; its own
    tests run in the cctbx gate.
  - The bytes are specific to the machine (BLAS, CPU), so this runs on both gate machines for
    every PR, rather than as a stored golden file in CI. A 1-ulp change to the scale makes
    119 of the 262 arrays in the first version differ.
- **V1 Toy recovery** (analytic halo, in torchBragg tests):
  - noise-free and Poisson images, Bragg+diffuse from known G, misset, cell and halo width;
  - recover each half alone and jointly;
  - report the Fisher-matrix condition number and the correlations between shared
    parameters.
  - **Gate design (2026-11-05)**, `scripts/diffuse/v1_toy_recovery.py`:
    - **Fits:** bragg (G, misset, cell), diffuse (misset, cell, halo amplitude and σ, with G
      known, since with k_d frozen G and the amplitude are one number there), and joint (all).
      They run on the Poisson deviance, GAUSS with the HKL metric.
    - **Separability:** for each joint parameter, its Fisher sd in the joint fit over its sd
      in the fit of its own half (G: bragg; halo: diffuse; misset and cell: the better of the
      two). Pass if every ratio is ≤ 1.5 and |z| < 4 in every fit and seed (noise-free: errors
      < 1e-5).
    - **Exit code:** 0 only if the separable toy passes and the negative toy is flagged
      (ratio > 1.5), so the metric is shown to tell them apart.
    - **What sets the ratio:** halo σ over the spot's sd in h, 0.397/N (Fisher at the truth,
      the same at 24² and 64²).
      - N = 5 (spot sd 0.079): σ = 0.05, 0.12, 0.3, 0.5 give 22, 14, 2.6, 2.3.
      - N = 10 (spot sd 0.040): 40, 2.2, 1.28, 1.23.
      - corr(amplitude, σ) is about −0.9 in every joint fit.
    - **The gate toy:** N = 10, σ = 0.3 (7.5 spot sds). Peak 128 e² per cell under F = 100.
      At 24² that is 8.4 diffuse photons a pixel under spots of 9e5 (about 4800 diffuse photons
      over the image). The fluence is the same at every size, so photons per solid angle are
      fixed: 1.2 a pixel at 64², 0.30 at 128², the same total.
      - **Corrected 2026-11-05.** The first version also scaled the fluence by (24/size)². That
        cut the photons as size⁻²: 0.17 a pixel and 680 over the image at 64², 28× fewer at 128².
        The diagnosis is the D3b sharp_wide failure at 128² (101 diffuse photons), in section 2.2
        of the D3b plan.
      - Ratios are 1.28 (amplitude), 1.11 (σ) and 1.00 for G, misset and cell. They do not
        depend on the photons.
      - Poisson |z| ≤ 2.3 over seeds 1–3 at 64² was measured at the old, lower photon level and
        is to be re-measured.
    - **Negative case:** the first toy, N = 5 and σ = 0.12 (1.5 spot sds). The halo hides under
      the spots: ratios 14 (amplitude) and 4.5 (σ). At 128² with seed 1 the joint fit
      collapsed σ to 0.025 (z = −4.9). That was at the old photon level, 28× too low, and is to
      be re-measured. The report flags it as weakly identified.
- **V2 eryx synthetic recovery:**
  - lysozyme from a public PDB in the SBGrid 747 cell, stills and a rotation wedge, with
    known springs, solvent, G and misset;
  - recover jointly, and with each half frozen;
  - grid vs exact error.
- **V3 Separability cross-check:**
  - merge the V2 images with `merge_map.py` and fit with ShellFit (map space alone);
  - the springs must agree with the image-space joint fit within the bootstrap SE;
  - a Bragg-only fit to images that contain diffuse must still recover F (through the
    radial and tilt backgrounds), and show how much the diffuse term moves it.
- **V4 Lysozyme real** (SBGrid 747 frames, CXIDB 128 map):
  - baselines, already done: Bragg-only box-sum CC about 0.93 against independent F (flat
    from exact F_calc); eryx map-space CC 0.952 against CXIDB (reduction) and the ladder
    numbers;
  - joint: does the held-out NLL drop; does the Bragg CC move; does the joint diffuse
    model predict the CXIDB map as well as the map-space fit?
  - **Before V4:** a /diagnosing-bugs pass on the 0.93 ceiling, because halos and spot tails
    compete for the same pixels.
- **V5 Insulin** (Zenodo, I213 with the reindexing ambiguity, cosym first): the same as V4,
  against the 0.900 ladder.

## 6. The byte-identical guarantee

- The diffuse code runs only `if diffuse is not None`. Nothing enters the compiled kernel, no
  buffer or parameter is created, and no RNG is drawn.
- Gates on every PR:
  - the V0 gate script, base vs branch, on both machines;
  - the parity matrix, the full suite, gradients, cctbx and the GPU suite on the GPU gate
    machine, run serially;
  - diffuse tests in CI with the toy models;
  - the eryx integration test in eryx's own suite.

## 7. Branches, PRs, order

torchBragg (lcls-mlcv, PRs into `torch`, one at a time; the maintainer merges):

| PR | Content | Gate |
|---|---|---|
| D1 | the hook: 31d0b2c, 3f653f1, 7b344a4 rebased; `diffuse.py` protocol plus toy models; V0 gate script | V0, full gates |
| D2 | `ExperimentModel(diffuse=)`, groups diffuse/diffuse_scale, ROI path, components, `SerialExperiment` sharing, D2-T1 shape-integral test, gradient checks | /tdd, V1 |
| D3 | `DiffuseGrid`: ASU grid, symmetry gather, differentiable trilinear interpolation, grid-vs-exact test. Done: section 2.2 (error table, memory, decision); `v1_toy_recovery.py --grid` passes, V0 0 differ | V1 on the grid |
| D4 | radial background b(\|q\|) | V1 with background |
| D5 | Bragg F from the structure model (mode b): F(hkl) from atom tensors in place of the grid; overall B/U, then rigid-body ADPs | V2 |
| (track) | spot-broadening diagnosis, from the start in parallel; the fix is decided from its result | gates V4 |

eryx (pushed to `backup` only):

| Branch | Content |
|---|---|
| E1 | `pyproject.toml`; `eryx.torchbragg.Diffuse` adapter; hkl kept in the graph on the exact path; bounded cache; `grid()` export; integration test resolves the import and fails loudly instead of skipping |
| E2 | cell as a tensor (A* differentiable); amplitudes() differentiable in atom xyz and U (for D5, and so levels 3-4 later add restraints, not a rewrite); the multi-body/ShellFit ladder terms reachable from the adapter |
| E3 | `analysis/joint/`: V2–V5 scripts and READMEs |

Order: D1 → E1 + D2 → V1 → D3 → V2 → D4 → V3 → E2 + D5 → V4 → V5. E2 + D5 (mode b) is the
major goal and on the critical path. The spot-broadening diagnosis runs in parallel from
the start and must finish before V4. This plan document goes in with D1.
Each PR gets /code-review and /scientific-code-auditor before I ask for a merge.

## 8. Decisions (settled 2026-11-04)

1. **Coupling.** The protocol lives in torchBragg and the adapter in eryx. The licenses are
   expected to align (both MIT) later. The adapter can then move into torchBragg as an
   optional extra (`nanobrag_torch[eryx]`), still with no hard eryx dependency.
2. **Evaluation.** A reciprocal grid with interpolation for refinement; exact per-pixel
   evaluation as the reference. The fineness of the grid near integer hkl is set by the
   measured grid-vs-exact error. Measured in D3 (section 2.2): uniform trilinear at step/σ ≤ 0.1
   passes V1, and D3 builds no refined core. Real photon counts need step/σ of about 0.02–0.05,
   so the refined or analytic core is decided on V2.
3. **Scale.** Bragg and diffuse are tied absolutely by default. A relative k_d (group
   `diffuse_scale`) is frozen at 1 and can be freed for separable fits. The peak-integral
   factor was measured (D2-T1). Decided 2026-11-05: `ExperimentModel` multiplies the diffuse
   term by it, recomputed from the current crystal, with an opt-out (section 2.1). k_d
   stays one value shared by all shots.
4. **Isotropic part.** A radial background b(|q|) owns everything radial. The diffuse model's
   shell mean is projected out by default, and the projection can be switched off for a
   future absolute model.
5. **Bragg F.** Mode a (free per-ASU Fhkl) first, then mode b (F from the structure model),
   which is **the major goal**. Mode c is deferred.
6. **Structure parameters in mode b.** Overall B/U and solvent first, then rigid-body ADPs.
   Per-atom ADPs and coordinates are the long-term goal: E2 builds F from atom tensors,
   differentiable in xyz and U, so that they can be added without a redesign.
7. **N_cells vs eryx k_floor.** Independent by default; a tie is optional
   (`k_floor_from="Ncells"`) for synthetic tests. The halo-core sensitivity is measured on V2;
   if it is large, k_floor becomes a refinable diffuse parameter.
8. **Spot broadening.** The Bragg CC limit of about 0.93 on lysozyme is a hypothesis to be
   diagnosed, not a known cause. Candidates are box truncation, partiality across frames,
   absorption, per-frame scale, the comparison target and a detector PSF. Diagnosis runs
   first, in parallel; the fix is chosen from what it finds. It gates V4.
9. **Where things live.** Library code goes in its repos. The joint analyses (V2-V5) go in
   eryx `analysis/joint/` (private), and large outputs stay off both repos. torchBragg gets
   one synthetic demo with the toy model and an api_guide section.
10. **PRs.** D1 is its own PR (hook, protocol, toy models, V0) and carries this plan.
