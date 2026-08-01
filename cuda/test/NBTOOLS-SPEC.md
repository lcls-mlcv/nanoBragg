# nb\* tool family — design spec (v1)

**Status:** IMPLEMENTED — this document describes the harness **as built** (all tools, libs,
suites, and `expected/` exist; the Makefile builds them). This is a **fresh** harness: `run.sh` is
**not** a reference and is not something to reproduce — it is retired/deleted. The validated `golden/*.tsv`
are **migrated** to `expected/` (same FP32 gate → verdicts carry); `--seed` is for deliberate
re-baselining, guarded by the reference canary (§6). Features that are designed but not
in the first cut are collected in **§ Phase 2**.

**Scope:** a legible C tool family that renders a candidate binary and a trusted reference binary
per case, compares the images, applies a per-suite gate, and reports. v1 is single-axis (absolute
vs the reference) with an FP32 gate; the FP16 view is reported, not gated.

---

## 1. Two vocabularies that never cross

- **cache = images.** Rendered *reference* images, machine-global, automatic.
- **ledger = data.** Numbers, verdicts, provenance. Never images.

Any name with "cache" is images; any with "ledger" is data. They never mix.

---

## 2. Tool family

`nb`-prefixed to avoid `PATH` collisions.

| Binary | Job |
|---|---|
| **`nbrunsuite`** | run a suite: render candidate, render/reuse reference, compare, gate, report |
| **`nbcache`** | image cache: `--status` / `--gc` / `--path` / `--set-…` |
| **`nbgensuite`** | compile a spec into suites (was `gen_suites`); bakes each case's `args_hash` |
| **`nbmetrics`** | compare two float32 images → FP32 (gated) + FP16 (reported) metrics |

Shared logic is linked source (no standalone binaries, no `.so`):

| Module | Job | Links |
|---|---|---|
| `argkey.{c,h}` | canonicalize an arg list → stable-hash input (was `canon`) | libc |
| `case_core.{c,h}` | parse `base.json` + `suites/*.jsonl` (cases, cost, gate header) | libc |
| `cache_core.{c,h}` | image-cache path assembly + gc + eviction + status + config | json-c + libmd |

`case_core` is the shared parser all three consuming tools link — the one place `base.json` +
`suites/*.jsonl` are read, so no tool re-implements it.

---

## 3. Vocabulary & schema

| Use | Retired | Schema field |
|---|---|---|
| **candidate** — binary under test | "kernel" | `candidate_args` (was `gpu_args`) |
| **reference** — trusted binary; output = truth | "oracle"/"cpu-bin" | `reference_args` (was `cpu_args`) |
| **case** — one scenario (command line + expected result) | "cell" | — |
| **suite** — a named group of cases, with a gate header | (kept) | — |
| **expected** — recorded per-case verdict | "golden" | `expected/<suite>.<precision>.tsv` |
| render cost (wall-time proxy) | — | **`K` ≡ `cost.compute`** (see §9) |
| reference fingerprint | — | `reference_md5` (was `cpu_ref_md5`) |

The reference defaults to the CPU `nanoBragg` but the interface never assumes CPU. Its correctness
is guarded by the `--seed` reference canary (§6, §15), built from `reference_fix_branches` in
`base.json` — the logic of the retired `gen_cpu.sh`, not its script.

**`args_hash` keys on `reference_args`.** The cache stores *reference* images, so the key is the
md5 of the canonical reference command line — **not** `candidate_args` (they differ, e.g.
thickness cases append `-oversample_thick` to the reference side only).

---

## 4. CLI conventions

`getopt_long` (`--flags`), uniform, no env-var interface (`$XDG_CACHE_HOME` honored for the cache
location is an OS convention, not our surface). Mode is a flag, not a bare subcommand. A nanoBragg
payload on our command line ends option parsing with `--`: `nbcache --path -- <exe> <params...>`.

---

## 5. Device selection

Two cards on this box differ by name/memory/UUID (desktop `NVIDIA GeForce RTX 5090`, 32 GiB,
`GPU-c5d63746-…`, pci `09:00.0`; laptop `…RTX 5090 Laptop GPU`, 24 GiB, `GPU-62609c5d-…`, pci
`01:00.0`). `nbrunsuite` handles devices through **NVML** — the NVIDIA driver's management library,
the same one `nvidia-smi` wraps — never `nvidia-smi` and never the CUDA runtime (`cudart`):

1. **Enumerate** — `nvmlInit_v2` → `nvmlDeviceGetCount_v2` → per device `nvmlDeviceGetName`,
   `nvmlDeviceGetUUID` (the `GPU-…` string used verbatim as the pin), `nvmlDeviceGetCudaComputeCapability`,
   `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetPciInfo_v3`. NVML sees every card with **no CUDA context
   and no visibility pin**. `--list-gpus` prints this table (index · name · sm · memory · UUID · pci).
2. **Resolve** — `--gpu <index|exact-name|uuid>` matches the enumeration by **exact** equality (never
   `strstr` — the desktop name is a strict prefix of the laptop's). The matched device's name/UUID are
   ground-truth from NVML, so there is **no separate post-render probe**. Refuses on no-match / multi-match.
3. **Pin the child only** — `nbrunsuite`'s own process never sets `CUDA_VISIBLE_DEVICES` /
   `CUDA_DEVICE_ORDER`. They are set **only in the nanoBragg child's environment** at fork/exec:
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=GPU-<uuid>` (order-independent — a UUID pins
   exactly one physical card).

`--skip-device` bypasses selection (CPU-mock pipeline testing). Resolution is by UUID/name, so it is
immune to the CUDA-runtime index inversion — only the printed `index` label tracks NVML's order.

---

## 6. `nbrunsuite`

**Required:** `--suite NAME` · `--candidate PATH` (`-c`) · `--gpu "NAME|index|uuid"` · `--workdir PATH`.

| Flag | Default | Meaning |
|---|---|---|
| `--reference PATH` | `reference/nanoBragg_root` (harness-root-relative, §11) | trusted binary |
| `--precision fp32\|df64` | `fp32` | candidate path → `-precision single\|double`; selects `expected` baseline (`fp64` dropped — df64 *is* double) |
| `--cases N-M` | all | run only cases N–M (1-based, inclusive) — batching; results accumulate in the workdir, only the final batch (`hi ≥ n_total`) computes the tier verdict |
| `--seed [--force]` | off | (re-)baseline `expected` (deliberate); **refuses** to change any verdict without `--force`; gated by the reference canary (below) |
| `--append-to-ledger [--tag NAME]` | off | append this run's **data** to the curated ledger (opt-in) |
| `--list-gpus` | — | print devices and exit |
| `--keep-candidate-images` / `--keep-reference-images` | off | retain images in the workdir |
| `--refresh-cache` / `--no-cache` | off | overwrite the cache / bypass it |
| `--cache-dir PATH` / `--budget-gb N` | XDG / 20 | image-cache location / size budget |
| `--build-commit` | — | print the git HEAD baked at build time and exit |
| `--ledger-dir DIR` *(test hook)* | `<harness-root>/ledger` | write the appended ledger under DIR instead of the default |
| `--canary-fast` *(test hook)* | off | shrink the `--seed` canary geometry (detpixels/N/steps) so the build+render+refuse plumbing verifies in seconds |

Cache use is via `cache_core`: `cache_gc()` at suite start, `cache_lookup()`/`cache_store()` per
case (keyed on the **baked** `args_hash` — `nbrunsuite` reads it, never recomputes). No `--vs`
/ drift comparison / session log in v1 (§ Phase 2).

**perf suite.** A suite whose header sets gate type `perf` is run specially: each case's candidate
is rendered **min-of-5** (keep the fastest `ms`), compared once for corr/sum_ratio, and is
**warn-not-fail** — a parity miss prints a diagnostic but the tier is forced PASS and no `expected`
is required.

**`--seed` (re-baseline).** Records this run's verdicts as the `expected` baseline — a *deliberate*
act. It **refuses to change any existing verdict** (prints the flips, exits nonzero) unless
`--force`. Before writing, the **reference canary** runs: it builds a from-source "gold" from
`main` + `reference_fix_branches` (`base.json`: `fix/phi0-stale-rotation`,
`fix/subpixel-oversampling`, `fix/curved-det-flag-guard`) and renders three fix-sensitive cases
(phi0 / subpixel / curved); `--seed` **aborts unless the reference reproduces gold on all three**,
so a wrong oracle can't silently poison baselines. (Where a fix is already merged into `main`, gold
== main and the canary still passes.)

---

## 7. `nbcache`, `nbgensuite`, `nbmetrics`

**`nbcache`** — image cache: `--status` / `--gc` / `--path -- <exe> <params...>` / `--set-budget-gb`.
`--path` must be handed the `{input_root}`-**token** form (or `--input-root` to re-tokenize) to
reproduce the baked key.

**`nbgensuite`** — compiles `spec/` → `suites/*.jsonl`. Bakes each case's `args_hash` = md5 of the
canonical (`argkey`) `{input_root}`-token **`reference_args`**. Writes the **suite gate header**
(§10). Emits per case `K = cost.compute` (the wall-time proxy). Links `argkey` + `case_core` + libmd.

**`nbmetrics`** — compares two float32 images and reports two metric sets from the one FP32 pair:
- **FP32** corr / sum_ratio / diagnostics — **the gated metrics**.
- **FP16 (reported, not gated)** — round both to FP16, then FP16 corr / sum_ratio + the count of
  pixels differing after FP16 rounding (`≥1 ULP` ≡ not bitwise-equal at FP16). **Overflow-safe:**
  float32 values above FP16 max (≈65504) and any Inf/NaN are **excluded** from the FP16 metrics
  (not clamped), so they never NaN-contaminate.

Input parsing → `getopt_long`; the FP32 stdout contract is stable (FP16 columns appended).
`nbmetrics --build-commit` (git HEAD baked at build time) is retained for the ledger provenance.

---

## 8. Comparison: FP32 gates, FP16 reports

The FP32 metrics feed the gate (§10). The FP16 metrics are the "what MX consumes" view — reported
alongside, never gated (FP16 ε ≈ 5e-4 far exceeds the corr bar, so gating on FP16 is meaningless).
The cache stores the **FP32** reference (16 MB, the superset); FP16 is derived at compare time by
rounding both images — one stored image, both metric sets. FP16-ULP-diff is where df64 vs fp32
shows (fp32 compute error ≪ FP16 ε except under large-N accumulation drift).

---

## 9. Image cache

**Location:** `$XDG_CACHE_HOME/nanobragg` (default `~/.cache/nanobragg`), `--cache-dir` override —
bootstrap from env/flag, not the config file. Machine-global, shared across worktrees, outlives
any project.

**Config (`settings.json`):** JSON, parsed by `cache_core`. Holds `budget_gb`
(flag > settings > default 20 GB). Written by `nbcache --set-…`.

**Layout:**

```
<cache-dir>/<reference_md5>/<AB>/<args_hash>.bin   (image, evictable)
<cache-dir>/<reference_md5>/<AB>/<args_hash>.meta  ({k, actual_seconds}, persists past eviction)
```

`args_hash` = md5 of canonical (`argkey`) `{input_root}`-token **`reference_args`**, baked by
`nbgensuite`; **all three consumers** (bake / store-lookup / `--gc` live-set) use that same baked
hash. `reference_md5` = md5 of the reference binary, a directory level. `<AB>` = 2-hex shard.
Reference versions coexist as sibling dirs.

**Lookup:** `.bin` present → **hit** (load, skip render). Absent → **miss** (render, `cache_store`).
Existence == a valid hit (`reference_md5` in the path).

**`.meta` (recorder):** each render writes `{k, actual_seconds}` atomically. **`K` ≡
`cost.compute`** (the wall-time-linear proxy; anchors from measured `compute_k`). Not
`cost.precision` (= Na·Nb·Nc, which does *not* track render time). `.meta` **survives eviction**
so the calibration dataset accumulates. In v1 it is *recorded only*; the estimator that consumes it
is **Phase 2**.

**Eviction:** automatic, **cost (`K`) + recency (mtime)**, under the budget — cheapest-and-coldest
first, so **expensive references are the last evicted** ("live forever" in practice by cost rank).
This is *not* a hard guarantee: under extreme budget pressure an expensive reference can still be
evicted and then re-renders on next use (a `pin` for a hard guarantee is Phase 2). Orphan-gc drops
entries no current case references; old `<reference_md5>` dirs age out by the same cost+mtime budget
eviction (the args_hash live-set is reference-independent, so gc doesn't mark whole dirs dead — the
budget caps disk). No wall-clock TTL. Dormant on today's all-cheap corpus (all cases `routine`,
max `K` ≈ 1152 ≪ budget).

**Concurrency:** every write is `<name>.tmp.<pid>` then `rename` (atomic), so a reader never sees a
partial file. `.bin` content is deterministic (racing writers write identical bytes); `.meta`
`actual_seconds` is not — it's harmless last-write-wins per entry (the dataset is the *set* of
`.meta`, not accumulation within one). A race costs at most a redundant render, never a wrong
result. `--gc` never evicts an entry younger than the run in progress.

**No compression.** Measured ~1.1× on real reference images (float32 physical data is high-entropy);
`.bin`s stored raw, disk managed by budget/gc.

---

## 10. The gate — per-suite, typed, FP32, single-axis (v1)

**Where it lives:** a **suite gate header** written by `nbgensuite`, not per case. Resolution:
`base.json` default → suite header (states only differences) → per-case override (rare). Effective
gate = `case ?? suite-header ?? base`.

**Gate types (v1):**
- `absolute` — `corr_min`, `sum_ratio` range on the **FP32** metrics. Default (`base.json`):
  `corr ≥ 0.9999`, `sum_ratio ∈ [0.999, 1.001]`. Both fp32 and df64 use it; fp32's legitimate
  misses are absorbed by the `expected`-flip layer (a recorded FAIL is not an alarm; a *flip* is).
- `reject` — exit-code (guards; the kernel must refuse): `9` = REJECT (expected), `0` = FAIL
  (silent no-op — it rendered what it must refuse), any other = BLOCKED.
- `perf` — timing, warn-not-fail (§6); no `expected`.

**Verdict & flips.** The FP32 gate → PASS/FAIL/REJECT per case; the suite verdict is **flip
detection** vs `expected` (pass→fail / fail→pass / absent / missing / SKIP-not-a-flip / REJECT).
`expected` is the migrated, validated golds; `--seed` (deliberate, canary-gated) re-baselines them.

*(The `relative` gate type — reproduce the seed corr/sum_ratio within a tolerance, for an fp32
"no-regression" gate — is **Phase 2**. v1 fp32 = `absolute` + flip.)*

---

## 11. Directory layout & data flow

The harness is self-contained under `cuda/test/`:

```
cuda/test/
├── inputs/          render feedstock — real files, NO symlinks · UNTRACKED
│   ├── crystals/    193L.hkl · 2PLV.hkl · 3NIR.hkl · scaled.hkl      (-hkl)
│   ├── matrix/      amat.mat                                          (-mat)
│   └── dummy.stol   flat table for the -stol/-4stol/-Q reject guards
├── reference/       nanoBragg_root · UNTRACKED
├── workdir/         per-run candidate images + results.tsv · UNTRACKED (conventional --workdir)
├── expected/        <suite>.<precision>.tsv verdict baselines · COMMITTED
├── ledger/          curated run data · COMMITTED (opt-in)
├── spec/            base.json + suite spec · COMMITTED
├── suites/          compiled *.jsonl · COMMITTED
├── src/ · build/    nb* sources (committed) · built binaries (UNTRACKED)
└── Makefile · NBTOOLS-SPEC.md · README.md
```

`.gitignore` covers the four generated/large dirs (`inputs/ reference/ workdir/ build/`). The image
cache is NOT under the repo — it is machine-global `~/.cache/nanobragg` (§9).

**Path anchoring.** Harness-relative paths (`input_root`, the `--reference` default, `expected/`,
`ledger/`, `suites/`) are relative to the **harness root** `cuda/test/`, discovered from `base.json`'s
own on-disk location (`dirname²(realpath(base.json))`) — never the process CWD. So `base.json` stores
`input_root = "inputs"` (= `cuda/test/inputs`), and the tools run from any working directory.

| Output | Where | Lifetime |
|---|---|---|
| reference images (FP32; FP16 derived) | image cache `~/.cache/nanobragg` | global, GC-bounded |
| candidate images | workdir | transient (deleted unless `--keep-*`) |
| this-run numbers (FP32 gated + FP16 reported, ms, verdict) + provenance trailer | `<workdir>/results.tsv` | ephemeral |
| `expected` (verdict) | committed `expected/<suite>.<precision>.tsv` | versioned; migrated golds, re-baselined by `--seed` |
| curated ledger (data) — tagged runs | `ledger/` | append-only, **opt-in** (`--append-to-ledger`) |

Cache = images (auto). Ledger = data (opt-in). No session log / drift axis in v1 (§ Phase 2).

---

## 12. `argkey` (was canon)

Canonicalizes an arg list so the cache key is order-independent, no arity table. A token is a
**flag** iff it starts with `-`(+dashes) then a **letter** (`-cell`, `--misset`); everything else
is a **value** (`74`, `/path`, `-10`, lone `-`). A **bundle** = a flag + its trailing non-flag
tokens; bundles **stable-sort by flag name**, values within a bundle keep order. Applied only to
the hash input; stored/executed args untouched. Never drops or merges tokens → distinct arg sets
can't collide (worst case a spurious re-render, never a wrong hit).

**Caveat:** the `-`+letter rule treats a *digit-leading* flag (e.g. nanoBragg's `-4stol`) as a
value. This is benign for keying (still deterministic, collision-free) and the only known such flag
is used solely in `reject`/guard cases (no cached reference). The model matches nanoBragg's
*value-carrying* flags; it is not claimed to classify every nanoBragg token. Built + tested as
`canon.c`; the extraction into `argkey.{c,h}` is a **refactor** — `canonicalize()` must *return a
string* (not write stdout) for the md5 consumer — with no standalone binary.

---

## 13. Build sequence

*Retrospective: this records the build order that was followed to produce the harness as it now
stands — it is the history of how the pieces landed, not an open to-do list.*

0. **Reorg** (one-time, before any tool work) — restructure to the §11 layout:
   `cuda/testdata/` → `cuda/test/inputs/` (dereference every symlink to a real file; crystals under
   `crystals/`, `A.mat` → `matrix/amat.mat`, `dummy.stol` at root); `nanoBragg_root` →
   `cuda/test/reference/`; establish `cuda/test/workdir/` (old `cuda/testrun/` outputs discarded —
   transient); `.gitignore` `inputs/ reference/ workdir/ build/`. Rename `data_root`→`input_root` and
   the `{data_root}`→`{input_root}` token across `base.json`, `suites/*.jsonl`, and the resolver
   (`input_root` = `inputs`, anchored at the harness root `cuda/test/` from `base.json`'s location —
   CWD-independent). The binary move is cache-safe (keyed on `reference_md5` content); the token
   rename re-keys cache entries, free pre-build.
1. **argkey** — refactor `canon.c` into `argkey.{c,h}` (`canonicalize()` returns a string, not
   stdout); drop the standalone binary. **Rewrite the Makefile** for the object→link layout (§14).
2. **case_core** — the shared `base.json` + `suites/*.jsonl` parser.
3. **nbgensuite** — rename `gen_suites`; link `argkey` + `case_core` + libmd; bake `args_hash` over
   `reference_args`; write suite gate headers; emit `K = cost.compute`; schema renames.
4. **cache_core / nbcache** — path/gc/eviction (`K` + mtime, no pin) / status / config / `.meta`
   recorder / atomic writes; cache at `~/.cache/nanobragg` with the `.bin`+`.meta` layout.
5. **nbmetrics** — rename `metrics`; FP32 gated + FP16 reported (overflow-safe) + ULP-diff; getopt_long.
6. **nbrunsuite** — device selection (§5), cache lookup, render, `nbmetrics`, typed per-suite gate
   (absolute/reject/perf) + flips, `--seed`, `--append-to-ledger`, ledger trailer.
7. **Migrate** the validated `golden/*.tsv` → `expected/` (rename + reformat; same FP32 gate, so
   verdicts and seed numbers carry). Validate self-consistency (stable verdicts, cache hit-rate,
   device assertion, `--seed` canary passes). **Delete `run.sh`.**

There is no "reproduce run.sh" step — run.sh is not a reference. `golden/` data is kept (as
`expected/`), not re-seeded from scratch.

## 14. Build / link mechanics

Objects-then-link; static; no `.so`. JSON is parsed/emitted with the **json-c** system library
(`<json-c/json.h>`, `-ljson-c`), not a bundled parser. `argkey.o` (libc); `case_core.o` (`-ljson-c`);
`cache_core.o` (`-ljson-c -lmd`). `nbgensuite`: `argkey.o` + `case_core.o` + `-ljson-c -lmd`.
`nbcache`: `cache_core.o` + `argkey.o` + `case_core.o` + `-ljson-c -lmd`. `nbrunsuite`: the same
**plus NVML** for device enumeration (`<nvml.h>`, `-lnvidia-ml`), built as a **separate CUDA-gated
target** (needs the toolkit dir for `nvml.h` + the link stub; the CUDA-free core still builds without
it). `nbmetrics`: its own build (`-lm`).

**Build prerequisites** (system packages — the harness is deliberately *not* self-contained, unlike
nanoBragg): a C compiler, **`json-c-devel`** (JSON) and **`libmd-devel`** (md5). A package being
installed on one box is not proof the distro ships it by default, so these are listed as explicit
install requirements. `nbrunsuite` also needs the CUDA toolkit dir for `nvml.h` + the `-lnvidia-ml`
link stub, but at **runtime** only the NVIDIA driver's `libnvidia-ml.so` (not `cudart`); the candidate
`nanoBragg` is built separately with `nvcc`. The CUDA-free core tools (`nbgensuite`/`nbcache`/`nbmetrics`)
need none of this.

## 15. Invariants

- Detector 2048×2048.
- Desktop RTX 5090 only; device selected by UUID and confirmed by name (§5), incl. the post-render
  binary-printed name.
- Deterministic renders (explicit step counts, `-nonoise`).
- Thickness cases pass `-oversample_thick` to the **reference** side only (so `reference_args ≠
  candidate_args` — see §3).
- Each render runs in an isolated working directory so `Fdump.bin` can't leak between cases.
- Inputs live under `cuda/test/inputs/` as **real files, never symlinks** (harness decoupled from any
  external data tree); the reference oracle lives under `cuda/test/reference/`.
- The FP32 absolute gate (`corr ≥ 0.9999 AND sum_ratio ∈ [0.999, 1.001]`, per-suite overridable)
  and the per-precision `expected` baselines (migrated validated golds).
- **Reference-oracle correctness:** the reference must reproduce a from-source gold built from
  `main` + `reference_fix_branches` (`base.json`) on the three fix-sensitive canaries; `--seed`
  enforces this (north-star check).

## 16. Phase 2 (deferred — designed, not in v1)

- **`relative` gate type** — reproduce the seed corr/sum_ratio within a defined tolerance (fp32
  "no-regression"). Needs the tolerance pinned and the seed numbers stored in `expected`.
- **Drift comparison (`--vs` / `--tag`)** — beyond the absolute gate, a two-axis (quality + speed)
  delta vs an explicitly named baseline. No "previous" (it drifts run-to-run) — always a named point.
  - **`--vs`** (bare) → the suite's **gold**: quality drift (Δcorr/Δsum_ratio vs the committed seed
    numbers) **and** the CPU-vs-GPU speedup (candidate `ms` vs the reference's CPU time in `.meta`).
    Always available; machine-independent on the quality axis.
  - **`--vs session:NAME` / `--vs ledger:NAME`** → a tagged snapshot: quality **+ Δms**. Explicit
    store prefix — session tags are disposable churn baselines, ledger tags are pinned/persistent.
  - **`--tag NAME`** snapshots this run's numbers (session-scoped by default; `--append-to-ledger`
    persists to the ledger).
  - Speed comparisons control the changed variable via recorded provenance (`candidate_md5`,
    `gpu_name`, `precision`, `ms`): CPU-vs-GPU (`.meta`), new-card-vs-old-card (tag, kernel fixed),
    kernel-A-vs-kernel-B (tag, hardware fixed).
  - Stores: **session log** (auto, disposable, project-local) + **curated ledger** (opt-in,
    persistent). Both need schema/location pinned before build; the minimal v1-pullable slice is
    bare `--vs` (gold) + `--tag`/`--vs session:`.
- **Cost estimator** — the `seconds_per_K` refit (robust median of `actual_seconds / K` over all
  `.meta`) + `--reference-budget` skip + the `.bin`-gone/`.meta`-present "known miss" path. v1
  records `.meta`; this consumes it, once an expensive reference actually exists.
- **Input data-prep / download** — `inputs/` is untracked large feedstock, so a fresh clone lacks it.
  A committed "Preparing input data" doc (the §11 `inputs/` tree + a `filename → md5 + size`
  verification manifest + an acquisition source) is deferred; for now inputs are prepared manually.
  The manifest also pins the canonical `amat.mat` (resolving the historical 369 B / 105 B `A.mat`
  ambiguity).
- **Retired** (not carried into the nb\* tools): `NB_DIFF` (two-candidate differential probe) and
  `NB_SCORE_ONLY` (re-score without render).
