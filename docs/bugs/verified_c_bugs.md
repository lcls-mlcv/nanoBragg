# Bugs in nanoBragg.c that torchBragg does not reproduce

Plain `nanoBragg.c` is wrong in the cases below, and torchBragg follows the physics. The parity
reference is the oracle from `scripts/oracle/build_oracle.sh`: bl831/main plus bl831's unmerged
fix branches `fix/phi0-stale-rotation` and `fix/subpixel-oversampling`, which correct both.

Run against plain C, these tests fail because of C: all PARITY-CONVENTION-002 `-phi` runs,
PARITY-MOSAIC-001 `mosaic-with-phi`, the `-phistep`/`-osc`/`-phisteps` runs of PARITY-CLIFLAGS-001,
AT-PARALLEL-020 and `-oversample_thick`.

## C-PARITY-001: phi = 0 reuses the previous pixel's rotation

`nanoBragg.c:2783` rotates the crystal only `if( phi != 0.0 )`. At phi = 0 it keeps `ap`,
`bp`, `cp` from the last step of the previous pixel, so a sweep starting at phi = 0 renders
the final orientation twice and never renders phi = 0. Only the first pixel, per thread
under OpenMP, is correct. Any nonzero start avoids it.

| Case (cell 70 80 90 75 85 95, N 5) | torchBragg / C, sum | Against fixed C |
|---|---|---|
| `-phisteps 5` | 1.029, corr 0.9971 | 1.00000, corr 1.000000 |
| `-phistep 0.5` | 1.044, corr 0.9930 | 1.00000, corr 1.000000 |
| `-osc 2.0` | 1.094, corr 0.9684 | 1.00000, corr 1.000000 |
| `-phisteps 5 -phi 1e-4` | 1.00000, corr 1.000000 | 1.00000, corr 1.000000 |

The oracle without C is `test_sweep_from_zero_is_the_mean_of_its_steps`.

## C-PARITY-002: per-sample corrections multiply the running sum

`nanoBragg.c:3014-3018` adds a sample to `I`, then with `-oversample_thick`, `-oversample_polar`
or `-oversample_omega` multiplies all of `I`, so every earlier sample is corrected again by
each later one. `-oversample_omega` keeps little more than the last sample, because omega is tiny.

| Case, `-oversample 2` | torchBragg / C, sum | Against fixed C |
|---|---|---|
| `-oversample_thick`, 4 layers | 35.8, corr 0.9657 | 1.00001, corr 1.000000 |
| `-oversample_omega` | 3.98, corr 0.9699 | 0.99997, corr 1.000000 |
| `-oversample_polar` | 1.002, corr 0.999999 | 0.99997, corr 1.000000 |

## Difference left in the reference: one polarization for all sources

`nanoBragg.c:2772` computes polarization once per pixel, from the first source, and reuses it
for the others. torchBragg computes it per source, which is what C does with
`-oversample_polar`. The difference grows with the angle between sources: 2.4% in sum for two
sources 17 degrees apart (`-sourcefile` with the second at -1 0.3 0), none with `-oversample_polar`.
The oracle keeps this, because it is an approximation and not an error in the arithmetic.

## SMV header longer than 512 bytes

Rotated or long-numbered geometry can take the `%g` header text past 512 bytes. `nanoBragg.c:3179-3198` writes it all under `HEADER_BYTES=512`, so the pixels start late and any reader misplaces them; torchBragg pads to the next multiple of 512 and declares that size, and every other file stays byte-identical to C.
