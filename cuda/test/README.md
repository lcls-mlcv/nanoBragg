# nanoBragg GPU↔CPU parity harness

A permanent, in-repo parity harness. Point `nbrunsuite` at a candidate binary (a
CUDA `nanoBraggCUDA` build) and it renders each case against a trusted reference
binary (the CPU `nanoBragg`), compares images, applies a per-suite gate, and
reports PASS/FAIL/REJECT plus any flips against a committed baseline. Build the
candidate with `cuda/build.sh` so its `-version` — and the ledger row — identify
exactly which binary was tested. The test
corpus is *compiled* from a compact JSON spec, so coverage is countable and
reproducible.

Full design rationale lives in `NBTOOLS-SPEC.md` (this doc is the practical
how-to; that one is the authoritative spec).

## The tool family

All binaries are `nb`-prefixed to avoid `PATH` collisions.

| Binary | Job |
|---|---|
| `nbgensuite` | compiles `spec/` → `suites/*.jsonl`, baking each case's `args_hash` and a per-suite gate header |
| `nbcache` | maintains the machine-global rendered-reference image cache (`--status` / `--gc` / `--path` / `--set-budget-gb`) |
| `nbmetrics` | compares two float32 images → FP32 (gated) corr/sum_ratio + FP16 (reported-only) corr/sum_ratio/ULP-diff |
| `nbrunsuite` | runs a suite end-to-end: picks the GPU, renders candidate + reference (cache-aware), calls `nbmetrics`, applies the gate, reports flips |

Shared logic is linked source, not standalone tools: `argkey.{c,h}` (canonicalize
an arg list for hashing), `case_core.{c,h}` (parses `base.json` + `suites/*.jsonl`),
`cache_core.{c,h}` (image-cache path/gc/eviction/status).

## Directory layout

```
cuda/test/
├── inputs/     render feedstock (crystals/*.hkl, matrix/amat.mat, dummy.stol) — untracked, real files
├── reference/  the CPU oracle binary (nanoBragg_root) — untracked
├── workdir/    per-run candidate images + results.tsv — untracked (--workdir target)
├── expected/   <suite>.<precision>.tsv verdict baselines — committed
├── ledger/     curated run provenance (runs.tsv, results.tsv) — committed, opt-in via --append-to-ledger
├── spec/       base.json + suite spec (dimensions.jsonl, groups.json, plan.json) — committed
├── suites/     compiled *.jsonl (grid320, coverage, guards, pairwise, perf) — committed
├── src/        nb* tool sources — committed
├── build/      compiled binaries — untracked
└── Makefile · NBTOOLS-SPEC.md · README.md
```

`inputs/`, `reference/`, `workdir/`, and `build/` are gitignored and machine-local
(a fresh clone needs them populated before the harness can render anything). All
harness-relative paths (`input_root`, `expected/`, `ledger/`, the default
`reference/nanoBragg_root`) are resolved from `base.json`'s own on-disk location,
not the process's working directory, so the tools run correctly from any CWD.

A case's stored CLI carries the literal `{input_root}` token (e.g.
`{input_root}/crystals/193L.hkl`); it is expanded to an absolute path at run
time, keeping `suites/*.jsonl` and the cache key machine-independent.

## Building

Prerequisites (system packages):

- a C11 compiler (`cc`/`gcc`)
- `json-c-devel` (JSON parsing, `<json-c/json.h>` + `-ljson-c`) — Rocky/RHEL:
  `sudo dnf install json-c-devel`; Debian/Ubuntu: `sudo apt install libjson-c-dev`
- `libmd-devel` (MD5 hashing, `<md5.h>` + `-lmd`) — Rocky/RHEL:
  `sudo dnf install libmd-devel` (may need CRB/EPEL); Debian/Ubuntu:
  `sudo apt install libmd-dev`
- for `nbrunsuite` only: the CUDA toolkit directory, to supply `<nvml.h>` at
  compile time and the `libnvidia-ml.so` link stub (auto-detected via
  `$CUDA_HOME`, `$CUDA_PATH`, an `nvcc` on `PATH`, or `/usr/local/cuda`); at
  **runtime** it needs only the NVIDIA driver's `libnvidia-ml.so.1` (no
  `cudart`). `nbgensuite`/`nbcache`/`nbmetrics` need none of this.

```
make            # builds nbgensuite, nbcache, nbmetrics into build/, and
                # nbrunsuite too if a CUDA toolkit dir was found
make test       # unit tests for argkey/case_core/cache_core/nbmetrics
make clean      # rm -rf build/
```

If no CUDA toolkit dir is found, `make` and `make test` still succeed with just
the CUDA-free core; `make nbrunsuite` prints an explanation and fails.

## Suites

`spec/` is source, `nbgensuite` is the compiler, `suites/*.jsonl` are the
compiled output (re-run `nbgensuite <suite> spec suites/<suite>.jsonl` and
review the diff to regenerate deliberately):

    build/nbgensuite grid320 spec suites/grid320.jsonl

Compiled suites, each with a gate type baked into its header:

- `grid320` (320 cases, `absolute` gate) — the canonical crystal × crystal-size ×
  orientation parity grid.
- `coverage` (74 cases, `absolute`) — main-effects sweep plus hand-authored
  feature/interaction cases.
- `guards` (8 cases, `reject` gate) — CLI options the GPU kernel must refuse
  (e.g. `-interpolate`, `-gauss_xtal`, `-fudge` ≠ 1, the amorphous-background
  tables); a `9` exit is REJECT (expected), `0` is FAIL (silent no-op
  regression), anything else is BLOCKED.
- `pairwise` (12 cases, `absolute`) — a strength-2 covering array over curved
  detector × N × lambda × pixel × oversample × thickness × misset × dispersion.
- `perf` (7 cases, `perf` gate) — timing cases, min-of-5 candidate render,
  warn-not-fail (a parity miss prints a diagnostic but never fails the tier).

## Running a suite

```
build/nbrunsuite --suite grid320 \
    --candidate /path/to/nanoBraggCUDA \
    --gpu "NVIDIA GeForce RTX 5090" \
    --precision fp32 \
    --workdir workdir
```

Per case this renders the candidate, renders or reuses the cached reference
image, compares with `nbmetrics`, applies the suite's gate, and appends a row
to `<workdir>/results.tsv`. It finishes with a tier line, e.g.:

```
# TIER grid320 PASS 320/320  (PASS=320 FAIL=0 BLOCKED=0 REJECT=0 SKIP=0)
# expected grid320.fp32: 0 flips (320 cells match) -> suite PASS
```

A nonzero flip count (a case that used to PASS now FAILs, or vice versa,
relative to the committed `expected/grid320.fp32.tsv`) fails the tier even if
every case still individually passes its gate — the tier verdict is a **flip**
check against the baseline, not a bare pass count.

`nbrunsuite` full flag reference (`--help`):

```
usage: nbrunsuite --suite NAME --candidate PATH --gpu "NAME|index|uuid" --workdir PATH
  options:
    --reference PATH        trusted binary (default reference/nanoBragg_root)
    --precision fp32|df64   candidate -precision single|double; selects expected baseline
    --cases N-M             run only cases N..M (1-based inclusive)
    --seed [--force]        re-baseline expected/ (canary-gated; refuses verdict flips)
    --append-to-ledger [--tag NAME]
    --list-gpus             print the GPU table and exit
    --keep-candidate-images / --keep-reference-images
    --refresh-cache / --no-cache
    --cache-dir PATH / --budget-gb N
    --build-commit          print the git HEAD this was built at and exit
  test hooks: --skip-device  --expected-dir DIR  --ledger-dir DIR  --canary-fast
```

## Device selection — desktop RTX 5090 only

This box has two GPUs whose CUDA-runtime enumeration order does not match
`nvidia-smi`. `nbrunsuite` sidesteps that entirely: it enumerates devices via
**NVML** (`nvmlDeviceGetName`/`GetUUID`/…, no CUDA context, no visibility pin),
matches `--gpu` against name/index/UUID by **exact** equality, and sets
`CUDA_VISIBLE_DEVICES=GPU-<uuid>` only in the rendered child's environment —
its own process never sets a visibility variable. Pin by UUID for an
index-proof match:

```
$ build/nbrunsuite --list-gpus
# 2 GPU(s) via NVML
index  name                               sm      mem       uuid                                       pci.bus_id
0      NVIDIA GeForce RTX 5090 Laptop GPU sm_120  23.9GiB   GPU-62609c5d-...                            00000000:01:00.0
1      NVIDIA GeForce RTX 5090            sm_120  31.8GiB   GPU-c5d63746-...                            00000000:09:00.0

$ build/nbrunsuite --suite grid320 --candidate ... --gpu GPU-c5d63746-6d55-532a-7d24-9dff88db9873 ...
```

`--skip-device` bypasses NVML device selection entirely (`--gpu` becomes
optional), so the render→cache→metrics→gate→flip pipeline can be exercised
with a CPU stand-in candidate on a box with no eligible GPU.

## Re-baselining (`--seed`)

`expected/<suite>.<precision>.tsv` is the committed verdict baseline that flips
are measured against. `--seed` records the current run's verdicts as the new
baseline, but **refuses to change any existing verdict** (prints the flips,
exits nonzero) unless combined with `--force`. Before writing anything, `--seed`
runs a reference canary: it builds a from-source "gold" binary from `main` +
`base.json`'s `reference_fix_branches` and renders three fix-sensitive cases
(phi0, subpixel-oversampling, curved-detector); `--seed` aborts unless the
`--reference` binary reproduces gold on all three, so a stale or wrong CPU
oracle can't silently poison the baseline.

## Image cache

Rendered reference images are cached machine-globally at
`$XDG_CACHE_HOME/nanobragg` (default `~/.cache/nanobragg`), keyed by
`<reference_md5>/<args_hash>`, evicted by cost (`K`) + recency under a
20 GB default budget (`--budget-gb`, or `nbcache --set-budget-gb N`).
`nbcache --status` reports size/entry/orphan counts; `--gc` applies eviction;
`--path -- <exe> <args...>` (fed the `{input_root}`-token form) prints the
cache path for a given reference invocation.

## Ledger

`--append-to-ledger [--tag NAME]` is opt-in: it appends this run's provenance
to `ledger/runs.tsv` (`run_id timestamp host gpu_name driver cpu_model commit
suite precision kernel_md5 cpu_ref_md5`) and one row per case to
`ledger/results.tsv` (`run_id cell verdict corr sum_ratio ms`). Routine runs
need not touch it; `<workdir>/results.tsv` is the ephemeral per-run record.

## Superseded

This is a from-scratch rewrite of an earlier shell-based harness (`run.sh`,
`gen_cpu.sh`, `metrics.c`/`gen_suites.c`, the `golden/*.tsv` baselines, and the
`{data_root}` token). None of that is part of the current flow: `expected/`
replaces `golden/`, `{input_root}` replaces `{data_root}`, and the CPU-oracle
provenance canary that `gen_cpu.sh` used to run standalone is now built into
`nbrunsuite --seed`.
