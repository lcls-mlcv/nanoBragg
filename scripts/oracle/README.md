# C oracle

Parity tests compare the PyTorch CLI against a C build of nanoBragg. That build is
**bl831/main plus three fix branches** — `fix/phi0-stale-rotation`,
`fix/subpixel-oversampling`, `fix/curved-det-flag-guard` — which are unmerged upstream
but are what bl831's own GPU harness uses as its CPU reference.

Because the merge exists nowhere public, reproduce it here rather than copying a binary
between machines:

```bash
NB_C_BIN=$(scripts/oracle/build_oracle.sh | tail -1)
NB_RUN_PARALLEL=1 NB_SKIP_INFRA_GATE=1 NB_C_BIN=$NB_C_BIN pytest tests/test_parity_matrix.py -n 8
```

The script clones bl831, merges the three branches, checks the merged `nanoBragg.c`
against a pinned md5 (`0a8675a1ce4b19e0d0039542e1a2099d`, from bl831/main `0d24e6e`,
2026-08-04) and builds with `-O2 -fno-fast-math -ffp-contract=off` — no fast-math and no
FMA contraction, so float results are comparable across machines. It writes
`ORACLE_PROVENANCE` next to the binary recording the base sha, branches, checksum and
build flags.

If the checksum warning fires, bl831 has moved. Re-measure the thresholds in
`tests/parity_cases.yaml` against the new build before trusting them; do not silently
update the pin.

**Never push to bl831.** This repo's `main` tracks bl831/main by fast-forward only.

## Known gaps in the oracle

`bl831/main` is the accuracy reference; its CUDA build is for speed comparison only (it
still uses the pre-2023 hkl spot metric, hard-codes `fudge=1`, and rejects
GAUSS/TOPHAT, `-interpolate`, `-fudge` and `-stol`).

C argv parsing uses `strstr`, which the parity cases have to work around:

- `-misset_seed N` must come **before** `-misset random`, or C parses it as fixed misset
  angles `(N, ...)`. `-seed` does not seed the random misset at all.
- `-twotheta_axis` also matches `-twotheta`, so it sets the twotheta angle and the SAMPLE pivot.
- Any flag containing `-pixel` (for example a PyTorch-only `-pixel_batch_size`) is read as
  `-pixel`. Never pass PyTorch-only flags to the C binary.

Beam intensity is likewise easy to get wrong, because three flags feed one number:

- `exposure` defaults to 1 s and `beamsize` to 1e-4 m (0.1 mm), and C sets
  `fluence = flux*exposure/beamsize^2` whenever `flux != 0`. **`-flux` alone therefore
  changes the scale**, and it silently **overrides an explicit `-fluence`** no matter which
  order the two appear in. To set the fluence directly, pass `-fluence` and no `-flux`.
- The default fluence with no flags is 1.259e29 photons/m^2, which is *not* what
  `-flux 1e12` gives (that is 1e20). A run with `-flux` and one without differ by ~9 orders
  of magnitude, so these flags are worth pinning in any comparison (PARITY-FLUX-001 does).
- C also recomputes `flux = fluence/exposure*beamsize^2` afterwards, which only affects what
  it prints, never the image.
