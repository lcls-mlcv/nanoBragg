# Plan: Bragg and diffuse in one differentiable model

Status: plan settled 2026-11-04. D1 (the hook, branch `feat/diffuse-joint` off `torch`
3d863fc) is open as PR #72. D2 (ExperimentModel, `feat/diffuse-experiment`, stacked on D1)
is implemented and has been through the audit and review of 2026-11-05. Its V1 gate,
`scripts/diffuse/v1_toy_recovery.py`, runs at small size in the tests; the full size runs on
the GPU gate machine. D3 (`DiffuseGrid`, `feat/diffuse-grid`, stacked on D2) is implemented; its
grid-vs-exact measurement and the decision it informs are in section 2.2.

Update 2026-11-06: D1, D2 and D3 are merged (#72-#74); D3b (the analytic core) is PR #75. V2 is
designed (section 5, V2a/V2b, settled with the user 2026-11-06) and its first piece, the spot-mask
helper (`nanobrag_torch.spot_mask`, branch `feat/diffuse-spot-mask` stacked on D3b), is built. The
best eryx models (the map-space ladders' top rungs) reach the protocol through a new eryx item, E1c
(section 7). The spot-broadening diagnosis is done and its fixes are planned before V4 (section 8,
decision 8).

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
   - The end goal (the user, 2026-11-06): the two halves merged and refined hand in hand, with
     either one switched off when it misbehaves or when a user wants only Bragg or only diffuse.
     V2 tests the switch both ways (section 5).
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
    # added in D3b, optional (not in the runtime check):
    # def core(self) -> DiffuseModel: ...                # the model's own sharp term near the lattice
    #                                                    # points, closed form; from_model(core=True)
    #                                                    # grids only model - core, see 2.2
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
    With a point group, each orbit is sampled at a node inside the box, so a bounded source
    (another grid, `outside="error"`) works under hexagonal operators too.
  - A parameter (group `diffuse`) or, with `refinable=False`, a buffer.
  - Storage:
    - `point_group=None`: P1, with no operators and no Friedel mates.
    - With a point group: one value per orbit of the integer operators (Friedel's law unless
      `anomalous`), the ones StructureFactors uses (`point_group_operators`).
    - The argument is not called `rotations` because StructureFactors' `rotations=None` means
      the identity with Friedel's law; here that is `point_group=np.eye(3)`. `anomalous`
      without a point group is refused.
    - The operators must form a group: generators alone are refused, not closed silently. The
      audit found that {I, 4-fold} read another orbit's value for 180 of 729 nodes.
    - Every lookup must find its orbit's key exactly.
    - On a step-1 grid the orbits are expand_to_p1's. A symmetric grid holds the values and one
      int64 key per orbit, with no index box.
  - Outside the box:
    - Zero (the trilinear interpolant of the values padded with zeros, so continuous). ±inf is
      outside and NaN stays NaN.
    - Or `outside="error"`, which also refuses non-finite hkl.
  - The result is on the query's device and in its dtype. The interpolation runs where the
    values are.
  - float32 at lysozyme-sized indices (u ≈ 640 at step 1/8): worst error 3.9e-5 absolute
    (4e-6 relative), under the bound 2·640·ε₃₂ times a cell's change in value.
  - Trilinear in hkl and the values. The gradient in hkl is constant within a cell; every
    finite-difference test asserts that no sample changes cell.
- **Measured, 2026-11-05** (`scripts/diffuse/grid_vs_exact.py error --size 64`: the V1 toy's
  diffuse-only shot, the amplitude scaled as σ⁻³ from V1's 128 at σ = 0.3, mmm ASU).
  - Deviance is 2Σ(g − μ + μ log(μ/g)) of the grid image g against the exact μ: the whole
    image's detectability, at about 530 diffuse photons over the image for σ = 0.3.
  - "scaled" is the deviance after the best-fit scale c = Σμ/Σg, the part a free k_d cannot
    absorb. The grid image is low by c − 1.

  | σ | step | step/σ | max rel | rms rel | max \|Δ\|/√μ | deviance | scaled (c) |
  |---|---|---|---|---|---|---|---|
  | 0.3 | 1/4 | 0.83 | 0.16 | 0.11 | 0.094 | 5.4 | 2.2 (1.083) |
  | 0.3 | 1/8 | 0.42 | 0.057 | 0.032 | 0.032 | 0.42 | 0.16 (1.022) |
  | 0.3 | 1/16 | 0.21 | 0.016 | 0.0083 | 0.0087 | 0.028 | 0.010 (1.006) |
  | 0.3 | 1/32 | 0.10 | 0.0039 | 0.0020 | 0.0021 | 0.0017 | 0.0006 (1.001) |
  | 0.2 | 1/16 | 0.31 | 0.034 | 0.018 | 0.035 | 0.27 | 0.21 (1.008) |
  | 0.2 | 1/32 | 0.16 | 0.0085 | 0.0045 | 0.0086 | 0.016 | 0.013 (1.002) |
  | 0.1 | 1/4 | 2.5 | 0.19 | 0.21 | 0.57 | 254 | 226 (0.871) |
  | 0.1 | 1/8 | 1.25 | 0.23 | 0.23 | 0.72 | 103 | 98 (1.064) |
  | 0.1 | 1/16 | 0.62 | 0.11 | 0.071 | 0.33 | 8.4 | 7.8 (1.022) |
  | 0.1 | 1/32 | 0.31 | 0.029 | 0.017 | 0.085 | 0.54 | 0.51 (1.005) |
  | 0.05 | 1/32 | 0.62 | 0.074 | 0.052 | 0.54 | 9.4 | 9.2 (1.009) |
  | smooth | 1/4 | | 0.020 | 0.0097 | 0.012 | 0.045 | 0.045 |
  | smooth | 1/8 | | 0.0047 | 0.0025 | 0.0030 | 0.0029 | 0.0029 |

  - Below step/σ ≈ 0.6 the error falls as step², as the trilinear bound says. Above it the halo
    is aliased: from step/σ 2.5 to 1.25 the max error rises (0.19 to 0.23) and the deviance
    falls only 2.5×. The tests check the bound itself: (step²/8)·Σ max |∂²I| times the image of
    a constant.
  - Per pixel the grid error is under the Poisson sd once step/σ ≤ 0.6. Over the whole image it
    is undetectable (deviance < 1) once step/σ ≤ 0.3–0.4, at about 500 photons.
  - A free scale removes about 60% of the deviance at σ = 0.3, and under 25% at σ ≤ 0.2.
  - **V1 with `--grid`** (`v1_toy_recovery.py`): the data come from the exact halo, and the
    model is its grid with σ and the shape known. Noise-free bias, at 24², in Fisher sds:
    - k_d frozen (plan decision 3, the default): the bias goes into the cell and G. Cell a takes
      0.055 at step 1/16 and 0.014 at 1/32 (diffuse fit); G takes 0.016 and 0.004 (joint fit).
      Geometry stays under 0.1 sd at every step measured.
    - k_d free: the bias goes into k_d, 0.397 and 0.097, and geometry stays ≤ 0.04.
    - The law z = K (step/σ)² √P (P the diffuse photons) holds: 4× the photons doubles z, and
      halving the step divides it by 3.9–4.1. K is 0.0186 (k_d frozen) and 0.129 (k_d free),
      from step 1/32 at P = 4835.
    - The step is the largest 1/(2n) with predicted z ≤ 0.05, half the pass limit of 0.1 sd,
      which was not changed. For the toy that is 1/18 (k_d frozen) and 1/46 (k_d free).
      - Measured there: noise-free 0.051 and 0.046 sd.
      - With Poisson noise (seeds 1–3): worst |z| 3.14 and 3.12, against 3.12 for the exact halo.
    - Tests at two fluences per mode show the rule holding as the photons change.
    - With k_d and the grid both frozen the diffuse term has no parameter, so the hidden toy
      cannot be flagged (ratios 1.00). The gate requires the flag only with k_d free (6.06).
- **Decided (decision 2), on these numbers:**
  - Uniform trilinear is enough for the V1 toy, with the step from the law above.
  - No refined or analytic core in D3. The law says one is needed at real photon counts: at
    1e5–1e6 diffuse photons a shot, z ≤ 0.05 needs step/σ of about 0.05–0.09 with k_d frozen
    and 0.02–0.035 with it free. For a halo core as sharp as eryx's k floor (0.5/N_cells, σ of
    about 0.05), uniform nodes would need a step of 1/220–1/390 (frozen) or 1/570–1/1000 (free),
    which is not affordable.
    The refined core is decided on V2, with eryx's real halo shape.
- **Memory** (`grid_vs_exact.py memory`: lysozyme 79×79×38 Å to 2 Å, P4₃2₁2 so Laue 4/mmm, the box
  ±(a/d_min + one node)):

  | step | nodes | ASU values (held) | P1 (held) | peak RSS, ASU / P1 | build, ASU / P1 |
  |---|---|---|---|---|---|
  | 1/4 | 15.8M | 1.00M (15 MB) | 120 MB | 1.19 / 0.48 GB | 1.7 / 0.25 s |
  | 1/8 | 124M | 7.8M (119 MB) | 944 MB | 1.37 / 1.36 GB | 13 / 1.8 s |

  - Lookup (`grid_vs_exact.py bench`, 2²⁰ points, step 1/4), ASU vs P1: 1015 vs 70 ns a point
    on the CPU (float64), and 318 vs 15 ns on MPS (float32). The ASU costs 15–20× per lookup
    for 8× less memory.
  - Recommended for lysozyme: 1/8 as the uniform base, with the core decided on V2. Use the ASU
    (119 MB) where memory binds, and P1 (944 MB) where the lookup cost binds, as on a GPU.
  - The ASU build peak is chunk work: 2¹⁹ nodes × 16 operators, transient.
- **D3b, the core near integer hkl** (`feat/diffuse-grid-core`, measured 2026-11-05 with
  `scripts/diffuse/core_candidates.py`).
  - The toy: a one-phonon-like core with a k floor, A(h₀)·κ²/(dᵀMd + κ²), with κ = 0.5/N = 0.05,
    M = diag(1, 1.3, 0.8) and A varying ±30% over h₀, plus a smooth part.
  - The fit: V1 geometry at 128², k_d frozen. The bias is linearised,
    b = −F⁻¹JᵀW(μ_grid − μ_exact), in sds scaled to 1e6 diffuse photons. The target is z ≤ 0.05.
  - Memory is for lysozyme to 2 Å on the 4/mmm ASU.
  - **Current metric (re-measured 2026-11-06, final D3b audit).** These are V1's current rules,
    `core_candidates.linear_bias_parts`:
    - with k_d frozen there is no diffuse-alone fit, since the geometry is given and the grid has
      nothing else to refine;
    - the joint fit's G is judged by its absolute bias (all ≤ 6.4e-6 in log, against 1e-3), the
      rest in sds at 1e6 diffuse photons.

    | candidate | worst z, current metric | lysozyme memory |
    |---|---|---|
    | uniform 1/8, 1/16, 1/32, 1/64 | 0.49, 0.155, 0.062, 0.015 | 59 MB, 0.47, 3.7, 30 GB |
    | warped, w 0.02, 16; 24; 32 nodes a unit | 0.161; 0.059; 0.031 | 0.53; 1.8; 4.2 GB |
    | warped, 24 nodes (w 0.01); 32 nodes (w 0.05) | 0.072; 0.027 | 1.8; 4.2 GB |
    | local cubes (coarse 1/16, r 0.25, fine 1/64) | 0.004 | 2.6 GB |
    | local cubes (coarse 1/16 or 1/32, r 0.125, fine 1/128) | 0.011; 0.011 | 2.6; 5.9 GB |
    | **model's core + residual grid, 1/8; 1/16** | **0.001; 0.000** | **59; 468 MB** |
    | core, M off by errors that cancel (−4%/+5%), 1/8; 1/16 | 0.014; 0.002 | |
    | core, M off by errors that cancel (−9%/+10%), 1/16 | 0.005 | |
    | core, M all +5%, 1/8; 1/16; M all +10%, 1/16 | 0.070; 0.002; 0.004 | |
    | core, κ +2%, 1/8; 1/16; 1/32 | 0.059; 0.003; 0.001 | |
    | core, κ +5%, 1/8; 1/16; 1/32; κ +10%, 1/8 | 0.149; 0.007; 0.002; 0.300 | |
    | core, isotropic (M = 1), 1/16; 1/32 | 0.013; 0.006 | |
    | truth with q-dependent amplitude, 1/8; 1/16; 1/32 | 0.124; 0.032; 0.008 | |
    | truth with a quartic shape, ε 0.1, 1/8; 1/16; 1/32 | 0.287; 0.030; 0.009 | |
    | truth with a quartic shape, ε 0.3, 1/8; 1/16; 1/32 | 0.775; 0.082; 0.026 | |
    | off-diagonal truth, diagonal core, 1/8; 1/16; 1/32 | 0.258; 0.503; 0.629 | |
    | off-diagonal truth, full-metric core, 1/8; 1/16 | 0.001; 0.000 | |

    - The ranking stands: the core is the only candidate under the target at 59 MB.
    - Every candidate does better than in the old metric. Local cubes (2.6 GB), warped grids at
      32 nodes (4.2 GB) and uniform 1/64 (30 GB) now meet z ≤ 0.05 as well.
    - The old diffuse-alone fit, which refined the geometry and judged G in sds, was the binding
      and stricter one.
  - The table below is the **old metric** (2026-11-05): a diffuse fit that refined the geometry,
    with G in sds. It is kept for the record.

  | candidate | worst z at 1e6 photons (old metric) | lysozyme memory | lookup |
  |---|---|---|---|
  | uniform 1/8, 1/16, 1/32, 1/64 | 29.8, 5.2, 1.14, 0.38 | 59 MB, 0.47, 3.7, 30 GB | 66–96 ns |
  | warped tensor grid, w 0.02, 16; 24; 32 nodes a unit | 11.5; 4.9; 2.6 | 0.53; 1.8; 4.2 GB | 63–72 ns |
  | warped tensor grid, 24 nodes (w 0.01); 32 nodes (w 0.05) | 6.9; 1.75 | 1.8; 4.2 GB | 69–79 ns |
  | local fine cubes (coarse 1/16, r 0.25, fine 1/64) | 0.39 | 2.6 GB | 180 ns |
  | local fine cubes (coarse 1/32, r 0.125, fine 1/128) | 0.28 | 5.9 GB | 177 ns |
  | **model's core + residual grid, 1/8; 1/16** | **0.057; 0.014** | **59; 468 MB** | 79 ns |
  | core with M off by errors that cancel, −4%/+5% (1.25, 0.84), 1/8; 1/16 | 0.14; 0.060 | 59; 468 MB | |
  | core with M off by errors that cancel, −9%/+10% (1.18, 0.88), 1/16 | 0.12 | 468 MB | |
  | core with M off, all +5% (1.05, 1.365, 0.84), 1/8; 1/16 | 2.29; 0.39 | 59; 468 MB | |
  | core with M off, all +10%, 1/16 | 0.74 | 468 MB | |
  | core with κ +2%, 1/8; 1/16 | 1.78; 0.29 | 59; 468 MB | |
  | core with κ +5%, 1/8; κ +10%, 1/8 | 4.5; 9.0 | 59 MB | |
  | isotropic core (M = 1), 1/16; 1/32 | 0.32; 0.20 | 0.47; 3.7 GB | |

  Realistic departures of the truth from the supplied core A(h₀)/(dᵀMd + κ²), z at 1/8, 1/16, 1/32:

  | the truth departs by | z |
  |---|---|
  | amplitude varying within the halo (q² and Debye–Waller at hkl, B = 20 Å², the core taking A(h₀)) | 5.9, 1.9, 0.55 |
  | non-quadratic shape \|d\|²f(d̂), f = 1 + ε(Σd̂⁴ − 3/5)/0.4, ε = 0.1 | 4.6, 1.4, 0.23 |
  | the same, ε = 0.3 | 11.1, 3.1, 0.66 |
  | off-diagonal metric, the core given its diagonal | 12.5, 16.1, 22.3 |
  | off-diagonal metric, the core given it in full | 0.056, 0.014 (1/8, 1/16) |

  - **Decided and built: the analytic core.** It is the only candidate that meets the target at
    affordable memory. A tensor grid refines whole planes and lines to reach the points. Fine
    cubes cost one cube per reflection and still leave the coarse grid's error at their faces.
    - `DiffuseGrid.from_model(model, ..., core=core_module | True)`: True takes
      `model.core()`, an optional method of the protocol. The grid stores model − core and
      returns core(hkl) plus the interpolated residual.
    - The core is a submodule: its parameters join the group `diffuse`, with gradients in hkl,
      the core and the values (FD and gradcheck).
    - `LorentzianHalo(amplitude, κ, metric)` is the reference core.
  - **What eryx's `core()` must reach, in the current metric (2026-11-06).** This replaces the
    old-metric rule, "residual under about 1% of the core at 1/16, 4% at 1/32", which came from
    the stricter diffuse fit.
    - **Anisotropy:** the core must carry the full (off-diagonal) metric of its own elastic term.
      An off-diagonal truth with a diagonal core fails at every step (0.26, 0.50, 0.63 at 1/8,
      1/16, 1/32), getting worse with a finer step because the residual stays sharp.
    - **At step 1/16, tolerated:**
      - κ within 5% (0.007);
      - a q-dependent amplitude across the halo, frozen at h₀ (0.032);
      - a quartic shape error of ε ≤ 0.1 (0.030).
      ε = 0.3 fails (0.082).
    - **At step 1/32, tolerated:** ε = 0.3 (0.026), κ within 5% (0.002) and the q amplitude
      (0.008).
    - **At step 1/8:** κ within about 2% (0.059 at +2%, already over), and neither the q
      amplitude (0.124) nor the shape (0.29) is tolerated.
    - Roughly, the core's bias is the uniform grid's (0.49, 0.155, 0.062 at 1/8–1/32) times the
      leftover's share of the core. That share must stay under about 10% at 1/8, 30% at 1/16 and
      80% at 1/32. The κ rows follow this only within about 3×, so the measured rows above are
      the requirement.
    - The pilot's recommendation, an order-3 core at 1/32, sits inside these with margin.
    - This is an E1/E2 item. Until it exists the D3b decision rests on a core that is not yet
      shown to exist for eryx.
  - **LorentzianHalo** takes a full 3 × 3 metric (built). An off-diagonal truth with a diagonal
    core gives z = 0.26–0.63 in the current metric (12–22 in the old one); with the full metric,
    0.001 and 0.000 (old metric 0.056 and 0.014). It stays a reference core for
    tests. A per-lattice-point amplitude would not make it a usable stand-in: the amplitude varies
    within the halo.
  - **The frozen residual.** The grid returns core(θ) + [model(θ_b) − core(θ_b)], θ_b the core's
    parameters at build.
    - Its gradient in θ is the core's alone.
    - `core_drift()` exposes |θ − θ_b|. Past `core_drift_tolerance` a warning is given once.
      The tolerance is a placeholder of 0.05, to be set on V2.
    - **How V2a sets it (decided 2026-11-06).** At points along each V2a grid fit (start, middle,
      end) the residual is rebuilt at the current θ; the stale-residual image's linearised bias is
      measured (non-scale parameters in sds, G and k_d absolutely) and fitted against the drift.
      Drift is the largest |Δθ_i|/s_i, with a drift scale s_i measured per parameter (log γ and
      log B_sol need not cost the same). The threshold is the drift at which the worst non-scale
      bias reaches 0.05 sd or log G's reaches 1e-3, rounded down. Past it the fit **rebuilds the
      residual** at the current θ and continues (minutes at 1/16 on the GPU gate machine); the
      one-time warning stays as the fallback where no rebuild is wired. This needs a small
      torchBragg addition (a residual rebuild on `DiffuseGrid`, or a callback in the refine loop).
    - A test with a smooth part that depends on θ shows the stale error growing with the drift.
    - With refinable values, the residual's values and the core's amplitude are nearly
      degenerate.
  - **Checks:**
    - A core not invariant under the point group is refused at build, by a sampled check.
    - A core needs 1/(2·step) integer, so that nodes sit on its half-integer kink.
    - `from_values` refuses `core=`: a residual with its core goes to `DiffuseGrid(values, ...,
      core=)`.
    - Outside the box only the core remains.
  - **Cost:** 2²⁰ lookups, step 1/4, without → with the core: ASU 992 → 1040 ns and P1 71 → 72 ns
    on the CPU; ASU 330 → 331 ns and P1 15 → 17 ns on MPS. float32: the core's error is under
    2·ulp(40)·√m/κ.
  - Image table (`grid_vs_exact.py error --size 128 --only sharp sharp+core`; deviance scaled
    to 1e6 photons):
    - without the core, max rel 0.20 / 0.14 / 0.10 and deviance 1.2e4 / 2.0e3 / 350 at steps
      1/8, 1/16, 1/32;
    - with it, max rel 7.5e-5 / 2.0e-5 / 5.1e-6 and deviance 0.73 / 0.053 / 0.003.
  - V1 `--grid --grid-core` is a machinery check: the halo is the refinable core, with a zero
    residual. It recovers to 1e-5 noise-free in all three fits, and flags the hidden toy.
  - V1 `--sharp`: a Lorentzian core plus a smooth part, on a cored grid.
    - The grid is built as a real fit would build it: from the model and its core at the fit's
      starting parameters, so `core_drift()` counts from the start.
    - The step comes from the residual's own bias law, z = K_core·step²·√P. K_core is measured
      per toy, since it depends on the Fisher sds the bias is measured in.
    - **κ = 0.5/N = 0.05**, the negative case.
      - Under the current rule (2026-11-06; the scale-like parameters judged absolutely, the
        geometry given in the diffuse-alone fit): K_core = 0.00045, C_core = 5e-6.
      - The rule picks 1/2. That is the coarsest step the 1/(2n) rule allows, not the law's answer
        (the law's widest step is 1.08). It is checked directly: at 1/2 the noise-free bias is
        0.0084 sd at 24² and 0.019 at 64².
      - Old rule: K_core 0.00276 (z 0.075, 0.0165, 0.0046 at 1/2, 1/4, 1/8), step 1/4, bias
        0.0165 sd.
      - The core is 1.25 spot sds, so it is weakly identified: ratio 10, like the hidden toy.
        It must be flagged.
      - With Poisson noise its z is reported, not gated (seed 3, joint: 15.7 exact, 15.8 on the
        grid).
    - **κ = 0.7**, separable, with the usual rules. A Lorentzian's wings separate from the spots
      only well past them: with no spot mask the worst ratio is 4.96, 3.17, 2.43, 1.82, 1.57,
      1.45, 1.33 at κ 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7 (2.5 to 17.6 spot sds), against 1.28
      for a Gaussian at 7.5 spot sds.
      - Current rule: K_core = 0.00548 and C_core = 2.4e-5, so the rule gives 1/4. Noise-free bias:
        0.018 sd at 24² and 0.019 at 64².
      - Old rule: K_core 0.0189 (z 0.058, 0.016 at 1/4, 1/8; 0.47 at 1/2, not yet in the step²
        regime), step 1/6, bias 0.024 sd.
      - Ratio 1.33. Poisson worst |z| 3.06 (seeds 1–3, 24², old rule).
    - Drift each fit ends with: 0.19–0.57 where the fit recovers. These drifts cannot set the V2
      threshold: the sharp toys' residual does not depend on θ, so drift costs them nothing. The
      θ-dependent case is covered only by the unit test with a shared amplitude. (The start is 0.7× the amplitude and 1.3× the width, ln 1/0.7 = 0.36), and up to 14.7 in
      the narrow toy's collapsed Poisson fit. The placeholder of 0.05 warns in every one of these
      fits.) V2 must set it from drifts that cost something.
  - **The 128² sharp_wide failure** (2026-11-06, diagnosed with /diagnosing-bugs).
    - **The root cause was V1's fluence scaling.** The fluence was scaled by (24/size)² while
      each pixel's solid angle also falls as size⁻², so 128² had 28× fewer photons than 24².
      sharp_wide had 101 diffuse photons and a Fisher sd of log κ of 1.18. Fixed on D2
      (feat/diffuse-experiment); 24² is unchanged.
    - **Ruled out, at seed 2:** the cored grid (the exact model failed the same way, log A +30),
      a coarse step (1/8 failed too) and the start (starting at the truth failed too).
    - **At 101 photons the likelihood has degenerate limits that beat the truth.**
      - κ → 0 puts a near-point spike on pixels with a count (μ 0.73 on a count of 1) and moves
        misset by about 1 rad: deviance 389 against 402 at the truth.
      - At 4× the photons the cell collapses instead (log a −50), so every pixel sits in one
        halo, a nearly flat image: 5 below the truth.
    - **After the fix**, at 2900 photons at 128², κ is identified (sd 0.22) and five of the six
      fits recover. The seed-3 diffuse fit does not, with the exact model as with the grid. A
      κ = 0.7 halo is nearly featureless, so the diffuse-only fit hardly constrains the geometry:
      misset and log a have Fisher sds 0.45–1.1. The fit finds the collapsed-cell basin, 10 NLL
      below the truth.
      - Under the profile rule below, log a gives z −4.5 with an interval unbounded over 16 sds.
        **Corrected 2026-11-06:** the "misset[1] z −0.5" recorded here was an artifact. The
        profile refit re-optimised the held misset angle through the rest of the misset tensor
        (fixed: a gradient mask now holds it).
      - misset[2] has sd 0.45, under the rule's threshold, and keeps its Fisher z of 21.
      - **Re-run 2026-11-06 under the current definitions** (the diffuse-alone fit with the
        geometry given; the per-parameter grid rule, cored step 1/4): sharp_wide passes at 128²
        with Poisson seeds 1–3. Worst |z| is 3.23, the ratio 1.33, and no parameter needed the
        profile rule (44 s on the Mac).
      - **The 128² gate fails on this toy under the old definition. It is not tuned.** What would change it is a choice
        about the toy: fix the geometry in a diffuse-only fit of a near-featureless halo, or
        separate a narrower core with the spot mask (below) rather than by width.
  - **Profile likelihood in the gate.** Where a parameter's Fisher sd at the truth exceeds 0.5
    (log or radian units), V1 refits with it held at the truth.
    - z_p = sign(p̂ − p)·√(2ΔNLL).
    - The 1-sd profile interval (2ΔNLL = 1) is reported beside the Fisher z.
    - The gate uses z_p, with the same |z| < 4, for those parameters.
  - **κ for real data:** bounded below by eryx's k_floor, 0.5/N_cells. The κ → 0 limit above is
    unphysical there; the toy leaves κ free to expose it.
- Pixels: diffuse fits do not need every pixel. A ROI set with a strided subset between
  spots (mask out ±n px around predicted spots, then stride) keeps one shot at 1e4–1e5
  pixels.
- **Spot masks (decided 2026-11-05).** Near the lattice points the diffuse core and the Bragg
  peak trade off: the core's amplitude and width are the parameters the joint fit cannot tell
  from the spot. Separability has to come from the halo's wings.
  - Diffuse and joint fits therefore mask ±n pixels around the predicted spots.
  - V2 reports the separability ratio and the grid bias against the mask radius n.
  - What exists: `DetectorConfig.mask_array`, a per-pixel mask that the Simulator and the ROI
    path (`ExperimentModel._roi_mask_values`) apply.
  - **Built for V2 (2026-11-06, `feat/diffuse-spot-mask`, stacked on D3b):**
    `nanobrag_torch.spot_mask.predicted_spot_mask(simulator, radius_px, n_sd=3)`.
    - Spots are predicted from the geometry, not from an image (a threshold on a still drops weak
      partials). Each pixel centre's fractional hkl is computed as the kernel computes it, and its
      distance to the nearest integer hkl is taken in the HKL metric, Σ(N_i d_i)², in GAUSS-HKL
      spot sds (0.397/N).
    - An integer hkl within n_sd sds of some pixel is a spot. Its centre is the nearest pixel, and
      ties (a beam on a pixel corner) go to the lowest index.
    - The mask is the union over phi steps, mosaic domains and sources, with no spot beyond dmin.
      It is a fixed bool tensor with no gradient.
    - `strided_pixels(keep, stride)` gives a `pixels=` subset of the kept pixels.
    - `ExperimentModel.spot_masks(radius_px, n_sd, apply=False)` builds one per panel at the
      current geometry. `apply=True` multiplies it into each panel's `mask_array` (a beamstop mask
      already there is kept), so both components drop those pixels, on the whole-image and ROI
      paths. It stays fixed while the geometry refines, until called again.
    - Tests (`tests/test_spot_mask.py`): the predicted centres are the Bragg image's maxima; a
      hand-counted direct-beam case; mosaic and phi give a superset; dmin; strided pixels;
      ExperimentModel masks zero both components and stay fixed while misset moves.
- **The spot-mask pilot, and the grid recommended for eryx (decided 2026-11-06).**
  - The pilot: eryx `torchbragg-core` ebe9e81, `core_bias.py --mask-radius`. Lysozyme 193l,
    n_cells 20, mesh 4, k_floor 0.025, V1 geometry at 128², k_d frozen. z is scaled to 1e6
    diffuse photons.
  - **Masks are an analysis option, not part of the grid's accuracy budget.** They do not fix
    the grid bias, and they hurt separability.

    | mask radius (px) | 0 | 1 | 2 | 3 | 4 | 6 |
    |---|---|---|---|---|---|---|
    | worst z, order-3 core, 1/16 | 0.32 | 0.34 | 0.41 | 0.41 | 0.40 | 0.22 |
    | worst z, order-3 core, 1/32 | 0.100 | 0.105 | 0.126 | 0.127 | 0.125 | 0.054 |
    | diffuse photons removed | 0 | 2.0% | 7.4% | 12.4% | 17.1% | 29.5% |
    | worst joint/alone ratio | 1.06 | 1.08 | 1.21 | 1.61 | 1.59 | 1.62 |

  - **The worst parameter is the joint Bragg scale G.** Its Fisher sd is tiny because of the
    bright spots, so a negligible bias is many sds. In absolute terms it is 2.6e-4 in log G at
    1/16 and 8e-5 at 1/32 (0.03% and 0.008%).
  - **Per-parameter targets (decided).** Scale-like parameters get an absolute tolerance: the
    bias in G must be under 0.1%. Diffuse and geometry parameters keep z ≤ 0.05 sd.
  - With G judged that way, order-3 core, unmasked:
    - at 1/32 the worst z is 0.037 (joint log a), which meets the target;
    - at 1/16 it is 0.145, which does not;
    - an order-2 core at 1/32 gives 0.59.
  - **Recommended grid for eryx: order-3 core plus step 1/32.** The lysozyme ASU is about
    3.7 GB, and the one-time sampling takes about 34 min on 64 cores. The optical-block
    correction is shelved unless V2 shows a disagreement.
  - **Superseded 2026-11-06 (current rules, eryx `core_bias.py --check`; worst non-scale z at
    1e6 diffuse photons / |Δ log G|):** order 3 at 1/16 gives 0.145 / 2.6e-4 unmasked and
    **0.033 / 4.2e-4 with a 2-px mask (7% of the diffuse photons), which passes**; order 3 at 1/32
    gives **0.037 / 8e-5 unmasked (passes)**. Two setups therefore meet the targets: **A**, order 3
    at 1/16 with a 2-px spot mask (about 470 MB), the default; and **B**, order 3 at 1/32 unmasked
    (about 3.7 GB), the cross-check. V2a chooses between them (section 5).
  - Caveat: the pilot's diffuse-only fit still refined the geometry (the old V1 definition).
    The joint numbers are unaffected.

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
    - **Fits:** bragg (G, misset, cell), diffuse (halo amplitude and σ, with G known and the
      geometry given), and joint (all). They run on the Poisson deviance, GAUSS with the HKL
      metric.
      - G is known in the diffuse fit because with k_d frozen, G and the amplitude are one number
        there.
    - **Diffuse alone means geometry given (decided 2026-11-06).** The diffuse-only fit holds
      misset and cell at their known values, from indexing and the Bragg data, as a diffuse-only
      program has them; the joint fit refines them. Each half is analysable alone and jointly.
      - The reason: a halo whose image is nearly featureless does not constrain the geometry.
        The 128² sharp_wide diffuse-only fit (κ = 0.7) had geometry Fisher sds of 0.45–1.1, and
        it found a collapsed-cell basin 10 NLL below the truth.
      - With k_d and the grid frozen there is nothing diffuse to refine, so `--grid` (k_d
        frozen) has no diffuse-alone fit.
    - **Separability.** For each joint parameter, its Fisher sd in the joint fit over its sd in
      the fit of its own half: G, misset and cell against Bragg alone; the halo against diffuse
      alone with the geometry given. Pass if every ratio is ≤ 1.5 and |z| < 4 in every fit and
      seed (noise-free: errors < 1e-5).
    - **Grid bias, per-parameter targets (decided 2026-11-06, section 2.2).**
      - Scale-like parameters (the Bragg scale G) get an absolute tolerance: bias under 0.1%.
        Their Fisher sd is set by the bright spots, so a bias in sds says little about accuracy.
      - Diffuse and geometry parameters keep the noise-free grid bias ≤ 0.05 sd (pass limit
        0.1 sd).
      - Implemented in V1 (2026-11-06). Both laws are re-derived without the scale-like
        parameters, judged absolutely:
        - z = K·(s/σ)²·√P: K 0.00215 (k_d frozen) and 0.00107 (free); with a core K_core 0.00045
          (sharp) and 0.00548 (sharp_wide).
        - |Δlog scale| = C·(s/σ)²: C 2.8e-4 (G) and 0.134 (k_d, free); with a core 5e-6 and
          2.4e-5.
        - Steps the rule now picks (24² and 64² alike): 1/6 (k_d frozen, was 1/18), 1/56 (k_d
          free, set by k_d's absolute bias, was 1/46), 1/2 (sharp, was 1/4), 1/4 (sharp_wide,
          was 1/6).
        - Gates at 24² and 64² pass. With k_d frozen at the chosen step (1/6), the worst non-scale
          bias is 0.041 and 0.031 sd; that figure excludes the scale-like parameters.
        - G's bias is 7.3e-5 in log, inside its absolute bound of 1e-3, and it is 0.095 sd. In V1
          the bound is about 1.3 G sds (G's joint sd is 7.7e-4 at 24²), so the "bright spots" reason
          for an absolute bound comes from the eryx pilot rather than from V1. G is reported in sds
          as well.
      - Measured, old (diffuse alone refining the geometry too) → new, at 24²:
        - separable 1.28 → 1.29 exact, 1.00 → 1.09 on the grid with k_d free;
        - sharp_wide 1.33 → 1.33;
        - the negative cases stay flagged: hidden 14.1 → 14.1 (grid, k_d free: 4.25 → 4.82),
          sharp 10.0 → 11.5.
        - The geometry's joint/Bragg-alone ratio is ≤ 1.04 in every toy.
      - The limit is unchanged.
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
- **V2 eryx synthetic recovery.** Settled with the user 2026-11-06. It is split in two: V2a tests
  the machinery on the eryx model that has an absolute scale and a `core()` today, and V2b repeats
  it on the best eryx model once E1c makes that a `DiffuseModel`.
  - **V2a, on `TotalScattering`** (atomic GNM one-phonon plus the residual-disorder term):
    - **Data:** lysozyme 193l (public), stills only (a rotation wedge moves to a later V2 step),
      4-8 shots at random orientations, 256² at 0.4 mm and 100 mm, 1 Å, N_cells 20, mesh 4,
      k_floor 0.025, GAUSS spots under the HKL metric. Diffuse photons per shot at 1e5 and 1e6.
      - Truth images at oversample 3, since pixel centres alias the sharp halo cores (below); fits
        use the same oversample, so grid and exact are compared like for like.
      - A low-resolution cutoff of d_max about 20 Å, as a beamstop: the 000 halo and the low-order
        00l halos make the image's brightest diffuse pixels there (below), which real data do not
        record.
    - **Truth and refined parameters:** springs at `interface` level, log γ drawn at about ±0.4
      with mean 0 and refined with the Σ log² γ restraint as the gauge (grouped, or `class` level,
      if 193l has too many interfaces); `share` (refine_share, truth 0.5), the diffuse term's
      absolute level, which tests the tie against G with k_d frozen; solvent log k and log B;
      `extra_U` (truth 0, refined). Fhkl is fixed at the truth from `write_bragg`, so `extra_U`
      acts on the diffuse term only until mode b, and its correlation with `share` is reported. One
      extra joint run frees per-ASU Fhkl (mode a) on the recorded reflections, to measure how much
      the diffuse term moves F (V3's question) and whether separability survives.
    - **Fits:** Bragg alone (G, misset, cell, N_cells), diffuse alone (geometry given, per V1), and
      joint (all), each on the exact per-pixel path and on the cored grid, setups A and B
      (section 2.2).
    - **Exact reference (option c, decided):** the linearised bias b = −F⁻¹JᵀW(μ_grid − μ_exact),
      from exact forwards, for every fit; plus full nonlinear exact fits as the direct check. These
      became affordable on 2026-11-06 (below).
    - **Pass rules**, applied to A and B:
      1. Grid vs exact, linearised, at 1e5 and 1e6 diffuse photons: non-scale parameters (springs,
         share, solvent, extra_U, misset, cell, N_cells) ≤ 0.05 sd; G and k_d < 1e-3 in log. The
         full exact and grid fits agree within 0.1 sd.
      2. Recovery: noise-free errors < 1e-5 relative; Poisson |z| < 4 over seeds 1-3 in all three
         fits, with the profile-likelihood z where a Fisher sd exceeds 0.5 (as V1).
      3. Separability: joint/alone sd ratio ≤ 1.5 for every parameter, reported against mask
         radius 0, 1, 2, 3, 4, 6 px (`spot_mask`); gated only at the chosen setup's radius.
      4. The switch: with the diffuse term off, the Bragg-alone fit on images that contain diffuse
         recovers the truth or reports its bias; with Bragg off, the diffuse-alone fit recovers
         the truth.
    - **Choosing eryx's grid:** if A passes rules 1-3 it is the default and B the recorded
      cross-check; if only B passes, B; if neither, stop and diagnose before D4, with no tuning.
    - **Core drift:** V2a sets the threshold and the rebuild (section 2.2, the frozen residual).
    - **Measured before V2a (2026-11-06, the exact path, 193l as in progress figure 9):**
      - Exact-path cost (CPU, 32 threads). The backward was about 200× the forward and grew as the
        rows squared: at 128², 1271 s for forward plus backward through the eryx parameters and
        1216 s for the backward through hkl. Cause: `OnePhononTerm` read C[k] and F[index] once per
        wavevector, and each read's backward fills a zero tensor the size of all of C; with mesh 4
        every pixel has its own k. Fixed in eryx `torchbragg-backward` (a72b1cb, c04ccc8: one
        gather and one split; values and gradients bit-identical, pinned by a test). Now 10.7 s
        and 6.2 s at 128²; 46 s and 39 s at 256².
      - Figure 9's square box around the direct beam is physical: the forward acoustic halo of 000
        and the low-order 00l halos, where |F(q)|² is near |F(000)|². The model is continuous
        across the 000 cell's faces (relative jump 2e-7, as for G ≠ 0) and falls smoothly along
        (0, 0, l): 1.60e5 at l = 0.05, 1.38e5 at 0.51, 5.3e4 at 0.95 e² per cell. It looks square
        because the halos tile the reciprocal cells and the figure's clip saturates them. It spans
        about 10 px (4 mm at 100 mm, d ≳ 20 Å), partly behind a typical beamstop: hence d_max.
      - Figure 9's single-pixel spikes are real sharp halo cores aliased by pixel-centre sampling:
        114 pixels at 256², |hkl − G| 0.02-0.17 (a pixel spans 0.28 in h), at 1.39× (median, up to
        5.2×) the oversample-3 average. The image totals agree (4641 against 4640 photons); 1% of
        pixels differ by more than 7%. Hence oversample 3 for the truth. Oversample 5 was not
        measured.
  - **V2b, on the best eryx model** (after E1c): V2a's design and rules repeated on `SpringDiffuse`,
    before V3 and V4.

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
    compete for the same pixels. **Done 2026-11-04** (decision 8): the ceiling came from how the
    data were set up and fitted, not from the forward physics. Its three fixes are V4
    prerequisites, and a pixel-level joint fit adds a fourth item:
    1. Train only on reflections whose whole rocking curve lies in the training frames, or model
       the frames that hold it (a script change). Measured: lin CC 0.928 → 0.959.
    2. A rotation-summed box objective: an NLL on each reflection's box sums added over all the
       frames it occupies (boxes on ±2.5 frames), as a loss utility in torchBragg `utils/losses`,
       grouped by reflection id across shots. Measured on top of 1 and 3: lin 0.973 → 0.984, log
       0.989 → 0.995. Its L-BFGS took about 5× the iterations: it needs preconditioning or a better
       F step scale.
    3. Keep misset honest: hold it to the spindle-phase component, or add a centroid or pixel term
       so the rotation about the beam is pinned by spot positions. The box-sum objective left that
       rotation nearly free, and it drifted 0.2° (2.1 px rms at the detector). Measured: +0.012 lin.
    4. **For the joint fit, which is a pixel NLL:** data spots were wider than the model's (σ about
       0.65 px against 0.36 px). A box sum is blind to this, but a pixel NLL is not: spot tails
       would leak into the diffuse pixels around each spot. Add a cheap per-shot Gaussian detector
       PSF (σ about 0.5 px) to the Bragg term before V4, after fixing the orientation (item 3),
       since a 2 px rotation offset dominates a 0.3 px width mismatch. Measure on V4's data first;
       the spot mask (section 2.2) is the fallback.
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
| D3b | finer grid near integer hkl, decided 2026-11-05: built before V2, to meet the bias law at 1e5–1e6 diffuse photons for sharp cores at affordable memory. Built as an analytic core supplied by the model plus a residual grid (section 2.2). Recommended for eryx (2026-11-06): order-3 core plus 1/32, per-parameter targets (G absolute < 0.1%, others ≤ 0.05 sd), masks not part of the budget | `core_candidates.py` (bias at 1e6 photons per candidate and core error), `grid_vs_exact.py error --only sharp sharp+core` at 128², V1 `--grid --grid-core` (machinery) and `--sharp` (noise-free bias under the residual's law), FD and gradcheck, V0 |
| D4 | radial background b(\|q\|) | V1 with background |
| D5 | Bragg F from the structure model (mode b): F(hkl) from atom tensors in place of the grid; overall B/U, then rigid-body ADPs | V2 |
| (track) | spot-broadening diagnosis: done 2026-11-04. Its fixes, before V4 (section 5, V4): whole rocking curves in training, a rotation-summed box loss in `utils/losses`, misset held to the spindle phase or pinned by centroids, and a per-shot detector PSF for the pixel-level joint fit | gates V4 |
| V2 helper | `spot_mask` (predicted-spot masks, `ExperimentModel.spot_masks`), with the plan's V2 design; stacked on D3b. Then a residual rebuild for the core drift (section 2.2) | /tdd, V0 |

eryx (pushed to `backup` only):

| Branch | Content |
|---|---|
| E1 | `pyproject.toml`; `eryx.torchbragg.Diffuse` adapter; hkl kept in the graph on the exact path; bounded cache; `grid()` export; integration test resolves the import and fails loudly instead of skipping |
| E1b | (decided 2026-11-05, before V2) the exact `core()` for D3b's cored grid: the one-phonon leading term computed from the model's own parameters. Accuracy it must reach, in the current metric (section 2.2, 2026-11-06; the old metric said a leftover under 1% at 1/16, 4% at 1/32). It must carry the full, off-diagonal anisotropy of its elastic term (a diagonal core fails at every step). At 1/16 it tolerates κ within 5%, an amplitude frozen at h₀ and a 10% quartic shape error. At 1/32 it tolerates a 30% shape error. At 1/8 it needs κ within about 2% and the q-dependent amplitude and shape. Its k_floor must be exact, which also bounds κ from below in a fit (κ ≥ 0.5/N_cells: the toy's κ → 0 spikes are unphysical). Gate: core_candidates-style bias at 1e6 photons ≤ 0.05 sd on eryx's own model. Pilot (torchbragg-core ebe9e81, 2026-11-06): the order-3 core at 1/32 meets it unmasked, with worst z 0.037 once G is judged absolutely (bias 8e-5 in log G); 1/16 gives 0.145 and order 2 gives 0.59 |
| E2 | cell as a tensor (A* differentiable); amplitudes() differentiable in atom xyz and U (for D5, and so levels 3-4 later add restraints, not a rewrite); the multi-body/ShellFit ladder terms reachable from the adapter |
| E1c | (decided 2026-11-06) `SpringDiffuse`: the best map-space model as a torchBragg `DiffuseModel` with a `core()`. Start from the insulin top rung (`MultiSpringModel` level 4, ASU bodies, F4 solvent and waters, crystal-2 CC 0.901). It needs: a torch-native `cell_transform`, differentiable in hkl (today it takes numpy); an absolute spring scale tied to the lattice share of the ADPs as `TotalScattering` has (ShellFit leaves the overall spring size undetermined); a k_floor from N_cells; and a `SpringAcousticCore` from the spring blocks at k → 0, with W rephased to body centres. For lysozyme, the body model is rebuilt from 193l with our own solvent, not from the mdx-lib export (CC BY-NC-SA) the map-space ladder used. The absolute scale is the science question: if the ADP-share tie fails for these models, k_d is freed for them. Then V2b |
| E3 | `analysis/joint/`: V2–V5 scripts and READMEs |
| (fix) | `torchbragg-backward` (2026-11-06): the exact path's backward made linear in the rows (section 5, V2a) |

Order: D1 → E1 + D2 → V1 → D3 → D3b → E1b → V2a → E1c → V2b → D4 → V3 → E2 + D5 → V4 → V5. E2 + D5
(mode b) is the major goal and on the critical path. The spot-broadening diagnosis is done; its
fixes must be in before V4. This plan document goes in with D1.
Each PR gets /code-review and /scientific-code-auditor before I ask for a merge.

## 8. Decisions (settled 2026-11-04)

1. **Coupling.** The protocol lives in torchBragg and the adapter in eryx. The licenses are
   expected to align (both MIT) later. The adapter can then move into torchBragg as an
   optional extra (`nanobrag_torch[eryx]`), still with no hard eryx dependency.
2. **Evaluation.** A reciprocal grid with interpolation for refinement; exact per-pixel
   evaluation as the reference. The fineness of the grid near integer hkl is set by the
   measured grid-vs-exact error. Measured in D3 (section 2.2): uniform trilinear with the step
   from the bias law (1/18 for the V1 toy with k_d frozen) passes V1, and D3 builds no refined
   core. (These D3 numbers predate the per-parameter rule: see the 2026-11-06 addendum below,
   under which the step is 1/6.) Real photon counts (1e5–1e6) need step/σ of about 0.05–0.09 (k_d frozen) or 0.02–0.035
   (free).
   **Decided 2026-11-05, after the D3 audit:**
   - The finer grid near integer hkl is built as its own PR, D3b, before V2.
   - The V1 grid pass rule stays: bias under 0.1 Fisher sd, with the step from the bias law.
   - L-BFGS-B's ABNORMAL stops on the grid are accepted for now; an AdamW check comes later.
   - Refining the grid values with a smoothness restraint is deferred until after V2.
   **Decided 2026-11-06, after D3b and the spot-mask pilot (section 2.2):**
   - The grid is the model's analytic core plus a residual grid. For eryx it is an order-3 core
     at step 1/32: about 3.7 GB for the lysozyme ASU, sampled once in about 34 min on 64 cores.
   - Accuracy is judged per parameter: the scale G absolutely (bias < 0.1%), the diffuse and
     geometry parameters by z ≤ 0.05 sd.
   - Spot masks are an analysis option, not part of the accuracy budget.
   - V2 compares grid and per-pixel fits directly. The per-pixel path stays the reference and
     the fallback.
   **Revised 2026-11-06, under the current rules (section 2.2):** setup A (order 3 at 1/16 with a
   2-px spot mask) is the default and B (order 3 at 1/32, unmasked) the cross-check; V2a's pass
   rules pick between them (section 5).
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
   **Diagnosed 2026-11-04.** The CC limit is not forward physics or detector-plane broadening.
   Three causes stack: reflections whose peak frame was outside the training frames (35 of 593,
   fitted from their tails); a spurious 0.2° misset about the beam, which the box-sum objective
   cannot see; and per-frame partiality the model cannot predict well enough (frame-centroid
   error 0.21 frames against a rocking width of 0.5). Truncation, phi sampling, Lorentz and
   polarization, and the background were refuted. With all three fixed, lin CC 0.984 and log
   0.995 against a metric ceiling of about 0.998. **Decided 2026-11-06:** the fixes and a detector
   PSF are V4 prerequisites (section 5, V4).
9. **Where things live.** Library code goes in its repos. The joint analyses (V2-V5) go in
   eryx `analysis/joint/` (private), and large outputs stay off both repos. torchBragg gets
   one synthetic demo with the toy model and an api_guide section.
10. **PRs.** D1 is its own PR (hook, protocol, toy models, V0) and carries this plan.
