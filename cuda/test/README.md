# nanoBragg GPU↔CPU parity harness

A permanent, in-repo parity harness. Point it at a GPU kernel binary and read
PASS/FAIL. The test corpus is *compiled* from a compact source spec, so coverage
is countable and reproducible.

## Four-part layout

**Committed source** — `cuda/test/`, pristine; the only committed part:

    cuda/test/
      run.sh            entrypoint / orchestrator
      gen_cpu.sh        builds + behaviorally verifies the CPU reference
      Makefile          out-of-source build (-> cuda/test/build/)
      src/gen_suites.c   the compiler   (spec -> cells)
      src/metrics.c     the comparator (float32 image pair -> corr/sum_ratio/…)
      spec/             base.json, dimensions.jsonl, groups.json, plan.json
      suites/            compiled + LOCKED cells (committed source-of-truth)

**Build output** — `cuda/test/build/`, gitignored compiled tools:

    cuda/test/build/
      gen_suites, metrics   compiled C tools
      .devprobe            device-assertion probe (built on demand)

**Run output** — `cuda/testrun/`, gitignored and disposable:

    cuda/testrun/
      cpu/             CPU reference images (cached, keyed by cpu_args hash)
      out/             transient GPU output images
      scratch/         per-render isolated working dirs + logs
      frozen/          persistent baked/deathstar oracles (never auto-deleted)
      cpu_build/       from-source CPU-reference build scratch
      results.tsv      run output
      cpu_manifest.txt CPU-reference provenance

**Data inputs** — `cuda/testdata/`, gitignored and machine-local:

    cuda/testdata/
      crystals/*.hkl     crystal structure factors
      A.mat, scaled.hkl  MOSFLM-matrix inputs
      nanoBragg_root     the CPU oracle binary
      dummy.stol         amorphous-table stub (guards suite)

`make` compiles `src/*.c` into `cuda/test/build/`. `run.sh` reads `spec/` +
`suites/`, invokes the tools, resolves each cell's `{data_root}` inputs against
`cuda/testdata/`, and writes cpu/out/results into `cuda/testrun/`. The only thing
that ever writes into the committed `cuda/test/` is a deliberate `gen_suites`
regeneration of `suites/`, reviewed and committed like a lockfile.

## The compile model

`spec/` is SOURCE. `gen_suites` is the COMPILER. `suites/*.jsonl` are the COMPILED,
LOCKED output — regenerate deliberately and review the diff. A cell is a scenario:
an exact GPU/CPU CLI plus id/axes/tags/gate/cost metadata. Different plans compile
to different suites.

    make                                                   # build tools -> cuda/test/build
    build/gen_suites grid320 spec suites/grid320.jsonl       # compile the 320-suite

`plan.json` names the suites:

- `grid320` — Cartesian cross of crystal × crystal_size × grid320 × orientation
  (4×4×5×4 = 320). Reproduces the canonical parity grid cell-for-cell.
- `coverage` — main-effects: a baseline cell, then one cell per (dimension, value)
  that differs from that dimension's baseline (only that dimension changed), plus
  hand-authored feature/interaction scenarios (detector thickness, `-energy`,
  `-roi`, non-square detector, alternate beam centers, custom basis, powder, the
  curved-detector and cancellation stress cells).
- `guards` — hand-authored cells the GPU kernel must REFUSE (see "Reject gate" below).
- `perf` — hand-authored cells tagged for timing (see "Perf suite" below).
- `pairwise` — a 12-run Plackett-Burman covering array (strength-2, every pair of
  factor levels co-occurs) over 8 risky axes: curved_det × N × lambda × pixel ×
  oversample × detector-thickness × misset × dispersion.

Untested axes needing external data files (documented gaps, not yet cells):
`-mask` (an SMV mask image) and `-sourcefile` (a source-list file).

## Running

    ./gen_cpu.sh                                   # establish/verify the CPU reference
    ./run.sh grid320 <gpu-kernel-binary>           # render, compare, gate, TSV
    column -t < ../testrun/results.tsv             # human-readable view

`run.sh <suite> <kernel-binary> [workdir]` asserts the device before rendering a
single pixel (see below), builds the C tools out-of-source, and for each cell:
renders the GPU kernel (`gpu_args`) and the CPU reference (`cpu_args`, cached by
hash), compares with `metrics`, applies the gate, and appends a TSV row
`cell corr sum_ratio ms verdict note`. It emits a final
`TIER <suite> PASS|FAIL n_pass/n_total` line and exits nonzero on any FAIL.

`NB_RANGE="lo-hi"` restricts a run to a 1-based inclusive cell-index range so a
long suite can run as several foreground invocations that append to one
`results.tsv`; the TIER tally is computed over the whole file.

## The gate (uniform, all precisions)

    PASS iff corr >= 0.9999 AND sum_ratio in [0.999, 1.001]

Pearson correlation + sum-ratio, identical for fp32/df64/fp64. fp32 is held to the
same gate deliberately, so float failures stay visible. `metrics` also prints
`max_rel worst_pixel_frac peak_max_rel worst_is_peak` as diagnostics (not gated).

## Reject gate / guards suite

Some CLI options are not supported on the GPU kernel; the correct behavior is an
explicit refusal (`exit 9`), never a silent no-op that renders something else.
A cell whose plan line carries `"gate":"reject"` compiles with `"gate_type":"reject"`
(a routine parity cell emits no `gate_type` field at all, so `grid320`/`coverage`
stay byte-identical). `run.sh` routes a reject-gate cell down a separate path: no
CPU oracle, no `metrics` — it renders the GPU kernel once and classifies by exit
code: `9` → `REJECT` (expected), `0` → `FAIL` (silent no-op regression: it
produced an image it should have refused), anything else → `BLOCKED`. `REJECT` is
its own tally bucket (not folded into `BLOCKED`) and participates in golden
comparison exactly like `PASS`/`FAIL` (a golden `REJECT` vs an actual `FAIL` or
`BLOCKED` is a flip). The `guards` suite has eight cells, one per GPU-unsupported
option that must be refused: `guard_interpolate` (explicit `-interpolate`),
`guard_gauss` (`-gauss_xtal`), `guard_tophat` (`-binary_spots`), `guard_fudge`
(`-fudge` ≠ 1), `guard_stol` / `guard_4stol` / `guard_Q` (amorphous-background
tables), and `guard_osthick0` (explicit `-oversample_thick 0`).

## Perf suite (min-of-5, warn-not-fail)

The `perf` suite is ordinary parity cells tagged for timing, kept at small
compute-K (K ≤ 640) so their CPU oracles stay cheap. `run.sh` detects
`SUITE_NAME=perf` and renders each cell's GPU kernel 5 times, keeping the MIN
`ms` (warm/cold jitter) while still comparing against the CPU oracle once for
`corr`/`sum_ratio`. The suite is warn-not-fail: a parity miss prints a
`# perf-warn: <cell> corr=.. sr=..` diagnostic line but never gates — `TIER` is
forced `PASS` and `run.sh` always exits 0 for this suite. No `perf` golden is
ever required.

## Invariants (every cell)

- Detector locked at 2048×2048.
- Desktop RTX 5090 only. This box has two GPUs and the CUDA runtime enumerates
  them INVERTED vs nvidia-smi; under `CUDA_DEVICE_ORDER=FASTEST_FIRST` the desktop
  card sorts first, so `CUDA_VISIBLE_DEVICES=0` selects it. `run.sh` proves the
  active device is the 170-SM RTX 5090 and prints it as proof; a mismatch is a
  hard refusal, not a warning.
- Deterministic: explicit step counts; `-nonoise` always (noise is host-only and
  non-deterministic — a separate regression, not here).
- Thickness cells pass `-oversample_thick` to the CPU side only (group `cpu_extra`).
- All cells pass an explicit `-hkl`, and renders run in an isolated working
  directory so nanoBragg's `Fdump.bin` auto-write/auto-load (a hardcoded filename
  in the CWD) cannot collide across cells.

## CPU reference provenance (gen_cpu.sh)

The CPU oracle must be `main` + `fix/phi0-stale-rotation` + `fix/subpixel-oversampling`.
Provenance is BEHAVIORAL, not by branch name: two canary cells — a phi-sweep cell
(sensitive to the phi0 fix) and an `-oversample_thick`, oversample>1 cell
(sensitive to the subpixel fix) — are rendered and compared against a "gold"
binary built from source (`main` + both branch diffs applied to `nanoBragg.c`) on
every invocation. Building gold from source makes the check non-circular (it
trusts no cached image) and self-adapting: if the fixes are already merged into
`main` the diffs are empty and gold == main, so the check still passes and the
eventual PR merge needs no edits here. If `cuda/testdata/nanoBragg_root` already
reproduces gold on both canaries it is reused; otherwise the freshly built gold is
installed. Provenance is recorded in `testrun/cpu_manifest.txt`. A parity result
measured against a reference that lacks either fix is invalid.

## Layer 2 — performance ledger

`run.sh` writes per-run results to an ephemeral `results.tsv` (Layer 1: one run,
one file, overwritten/appended to next time). Layer 2 accumulates every run's
provenance and per-cell rows permanently into two typed TSVs under `ledger/`:

    ledger/runs.tsv     # run_id timestamp host gpu_name driver cpu_model commit
                        #   suite precision kernel_md5 cpu_ref_md5
    ledger/results.tsv  # run_id cell verdict corr sum_ratio ms

joined by `run_id`. `ledger_append.sh <results.tsv> [ledger_dir]` parses a
finished `results.tsv`'s `# key=value` trailer, derives
`run_id = <UTCcompact>-<kernel_md5[0:8]>-<host>`, and appends one `runs.tsv` row
plus one `results.tsv` row per cell. It is idempotent: re-running it on a
`results.tsv` already ingested (same `run_id`) prints `already ingested: <run_id>`
and appends nothing.

`commit` in both the ledger and `run.sh`'s own summary trailer is a **build-time
bake**: `metrics` is compiled with `-DNB_BUILD_COMMIT=<HEAD at compile time>`
(see `Makefile`) and reports it via `metrics --build-commit`, rather than
re-reading `git HEAD` at render time. Since `run.sh` runs `make` on every
invocation, this equals HEAD-at-render on a stable branch; `gen_suites` itself
stays commit-agnostic (its output is reviewed by diff, not stamped).

## Data

Uses local `.hkl`/`.mat` inputs under `{data_root}` (repo-relative `cuda/testdata`):
the crystals at `{data_root}/crystals/*.hkl` and the AMAT inputs at `{data_root}/A.mat`
(105 B) and `{data_root}/scaled.hkl`. These large inputs are gitignored and populated
per-box (symlinks or copies); this harness commits no crystal data and touches no PDB.
Cells name only the `{data_root}` token — never a machine-specific path — and `run.sh`
resolves it to an absolute path at run time.
