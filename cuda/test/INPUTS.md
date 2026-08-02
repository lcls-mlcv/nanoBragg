# Harness inputs — obtaining the render inputs and the reference oracle

The parity harness renders against two untracked, gitignored trees under
`cuda/test/`: the render inputs in `inputs/` and the trusted CPU oracle in
`reference/`. A fresh clone must populate both before any suite can render.
This document says how to obtain each and how to verify it.

All paths below are relative to the harness root `cuda/test/`.

## Render inputs

The crystal structure-factor tables (`-hkl`) and the orientation matrix (`-mat`)
are distributed as a versioned GitHub **release asset**, not committed to git (the
largest, `2PLV.hkl`, is ~126 MB). Download and extract the current input set. Plain HTTPS — needs only `curl` and
`tar`, no other tooling:

```
curl -L -o test-inputs-v1.tar.gz https://github.com/bl831/nanoBragg/releases/download/test-inputs-v1/test-inputs-v1.tar.gz
tar -xzf test-inputs-v1.tar.gz -C cuda/test/
```

Or, if you have the GitHub CLI:

```
gh release download test-inputs-v1 --repo bl831/nanoBragg --pattern 'test-inputs-v1.tar.gz'
tar -xzf test-inputs-v1.tar.gz -C cuda/test/
```

Either way that populates `inputs/crystals/` and `inputs/matrix/`. The release is versioned independently of
product releases and is bumped only when the input data changes. Verify the
extracted tree against this manifest:

| Path | Size (bytes) | md5 | What it is |
|---|---|---|---|
| `inputs/crystals/2PLV.hkl` | 126487008 | `a7beb944672798abae7ebebf81fa6184` | `-hkl` table, poliovirus (~126 MB) |
| `inputs/crystals/193L.hkl` | 11198088 | `83959896d6f8246ae042b55dad1f8606` | `-hkl` table, lysozyme; the `--seed` canary crystal |
| `inputs/crystals/3NIR.hkl` | 1429002 | `239deae7083615cb7fbbddeae6da1e52` | `-hkl` table, crambin |
| `inputs/crystals/scaled.hkl` | 1350993 | `af3457b2d8a04a9a43c9c887a61fd1f1` | `-hkl` table, canonical fixture |
| `inputs/matrix/amat.mat` | 105 | `4fe7d5ffe4fc7a91323e9093efee7ef0` | `-mat` orientation matrix (3×3 A-matrix) |

### Provenance

The crystal `.hkl` tables were computed from deposited PDB coordinates
(`2PLV`, `193L`, `3NIR` — available from RCSB by ID) with gemmi's density/FFT
structure-factor engine (`gemmi sfcalc`), at d_min:

```
2PLV = 3.4 Å    193L = 1.33 Å    3NIR = 1.1 Å
```

Each table is then expanded to the full P1 sphere (all symmetry-equivalent
reflections plus Friedel mates), with (0,0,0) dropped, and written as `h k l F`.
Exact regeneration takes the gemmi structure-factor step plus a symmetry-aware P1
expansion and a reformat — not a single command — so the canonical bytes are the
release asset above; this note is provenance, not a byte-exact recipe.

`scaled.hkl` and `amat.mat` are canonical fixtures whose origin is not recorded —
they exist only as the distributed bytes.

## Reference oracle (`--reference`)

`--reference PATH` is required (there is no default) and names the trusted CPU
oracle every case is compared against. The oracle is the top-level `nanoBragg.c`
built from `main` with three fixes applied to that source:

- `fix/phi0-stale-rotation`
- `fix/subpixel-oversampling`
- `fix/curved-det-flag-guard`

Build it with:

    gcc -O3 -fopenmp nanoBragg.c -o nanoBragg_root -lm

These three fixes are exactly what the `--seed` reference canary re-checks: it
builds a from-source "gold" from `base.json`'s `reference_fix_branches` and refuses
to re-baseline `expected/` unless `--reference` reproduces gold on the phi0 /
subpixel / curved canaries. The canary — not a byte match — is the correctness gate.

### Current oracle binary (verification only)

| Path | Size (bytes) | md5 |
|---|---|---|
| `reference/nanoBragg_root` | 116504 | `bd647dbabb172780ae623e390cc49751` |

The md5 is **not** a hard pin: the same source compiles to different bytes across
compilers, versions, and flags. Treat it as a sanity check on a populated
`reference/` tree, and rely on the `--seed` canary for correctness.

### Canonical oracle

The committed `expected/<suite>.<precision>.tsv` verdict baselines were rendered
against `reference/nanoBragg_root` (md5 `bd647dbabb172780ae623e390cc49751`). That
md5 is also the image cache's top-level subdirectory key
(`<reference_md5>/<args_hash>`), so a different oracle binary keys into a separate
cache namespace and would need its own re-baseline.

> TODO / owner-review: a separate oracle build (md5 `b6a565d9…`) exists per the
> project's North Star; the canonical choice of oracle should be confirmed. This
> document records only that the committed `expected/` baselines used
> `reference/nanoBragg_root` (`bd647dba…`).
