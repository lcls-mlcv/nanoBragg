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

## The forward beam on a sample point under `-round_xtal`

`sinc3` in `nanoBragg.c` returns 1 at exactly x = 0, and takes `3*(sin(x)/x-cos(x))/(x*x)` everywhere else. Below x = 1e-8 the bracket cancels to 0, so a sample on the forward beam gets F_latt = 0 unless its hkl is exactly 000. Which of the two C gets turns on the last bit of the pixel position. Torch's arithmetic for that position differs in the last bit, so it cannot follow, and below x = 1e-6, where the closed form has no digits left, torchBragg returns the series value, 1. That takes 42 of 48 geometries to byte identity with C. In the other 6, and in the sweep's case 45, C has 0 at that one sample, which is the whole pixel at `-oversample 1` and 1/9 of it at `-oversample 3`. It happens only with `-round_xtal` and a pixel or subpixel centre on the beam.

## `-misset_seed` without a later `-misset`

`nanoBragg.c` matches flags with `strstr`, so `-misset_seed N` is also `-misset N <next> <next>`, and `-misset_seed 7 -distance 90` renders a fixed misset of 7 0 90 degrees, unless a later `-misset` replaces it or fewer than three arguments follow. Those angles come from the text of unrelated flags, so torchBragg reads the flag as a seed and warns with the `-misset` that gives C's orientation.

# Bugs in nanoBragg.c that torchBragg reproduces

These are wrong physics that stay as C has them, by ruling, so that images remain comparable. The numbers below were measured on both programs, and the two agree.

## Water background is quadratic in fluence

`I_bg` already holds `r_e_sqr*fluence` (`nanoBragg.c:2515`), and the pixel loop multiplies the sum it starts from by `r_e_sqr*fluence` again (`:3045`). Doubling `-fluence` gives 4.0 times the background, and four times gives 16.0. The mean of a 64 x 64 `-water 100` image is 13.9 at `-fluence 1e24` and 1.39e13 at 1e30, 1e12 times higher for 1e6 times the fluence.

## Sensor capture is not conserved across `-thicksteps` and `-oversample_thick`

With N layers given, the step is T/(N-1), so N layers absorb over T*N/(N-1) (`nanoBragg.c:1626`). Without `-oversample_thick` every layer takes the capture fraction of the last one (`:3048`). Sum of the image over that of a thin sensor, for T*mu = 1, where near normal incidence 1 - 1/e = 0.632 is right:

| Flags | Sum |
|---|---|
| `-oversample_thick` | 0.632 |
| `-oversample_thick -thicksteps 2` | 0.865 |
| `-oversample_thick -thicksteps 3` | 0.777 |
| `-oversample_thick -thicksteps 11` | 0.667 |
| none | 0.477 |
| `-thicksteps 5` | 0.407 |

## `-divsteps` alone means a divergence of 1 radian

With a step count and no range, `-hdivrange` and `-vdivrange` both become 1 radian. `-divsteps 3` gives hdivstep = 0.333 rad and four sources at +/-0.167 rad on each axis, and the image sum falls to 0.50 of the same run without the flag. Only an info line reports the range.
