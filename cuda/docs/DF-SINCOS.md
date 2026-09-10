# Why the Curved-Detector Trig Rebuilds sin and cos in Pairs

The curved-detector option puts every pixel the same distance from the sample by rotating each
pixel's position about two detector axes, through small angles (the pixel's offset along an axis
over the distance). That rotation needs a sine and a cosine of each angle. There are two ways to
get them: the hardware `sinf`/`cosf` on a single-precision angle — fast, and correct almost
everywhere; or a float-only sin/cos rebuilt in double-float pairs — accurate to ~14 digits at the
same speed class. Compare whatever the kernel does against these two and you know which side of the
line it is on.

Why not simply use true-double `sincos`? Consumer GPUs run double at roughly 1/64 the float rate,
so it is ~2× slower on the curved path, and its double temporaries cost registers that every
launch pays for whether it draws a curved pixel or not. The question is whether a float-only
version can be made accurate enough. It can.

## The straightforward version — and what breaks

The df ("double-float") rotation already carries the vectors as (hi, lo) pairs, but the angle and
its trig were single floats:

    const float sinphi = sinf(phi);
    const float cosphi = cosf(phi);
    ...
    newv[1] = df_add(df_mul_f(v1, cosphi), df_mul_f(cross, sinphi));   /* pair * float */

The vectors are pairs, the trig scalars are floats, and the `_f` multiplies round each product
back to float — so the rotation is only ever as good as `sinf(phi)`.

`sinf` is fast here: the angle is small (≲ 1.8 rad), so none of the huge-argument machinery
applies. It is simply *rounded* — a float32 holds ~7 digits, so it returns the nearest one to the
true value, about 1 ulp off (~6×10⁻⁸):

    true sin(0.5) = 0.4794255 386042…    <- what you want
    sinf(0.5)     = 0.4794255 495        <- one float32: right to ~7 digits, then the nearest
                                            float, and the "386042…" tail is simply gone

That looks harmless. The damage is done at the next step. The rotation multiplies that sine by a
pixel coordinate (~176 mm), and **a product is only as precise as its least-precise factor.** The
coordinate is carried as a pair (14 digits), but the sine has 7, so the product has 7:

    176.128 mm × true sine = 84.44026126 mm
    176.128 mm × sinf sine = 84.44026318 mm    -> off by 1.9 nm

All the care spent carrying the vectors as pairs is thrown away at that multiply, because the sine
had only 7 digits to give — a precise ruler read against a coarse protractor. At short wavelength
and a large crystal (N ≈ 1000 unit cells) the interference fringes are needle-sharp, and that
1.9 nm shift is about a tenth of a fringe: summed over the detector, a few tenths of a percent of
the total flux moves between pixels. The spots stay put — correlation to the double reference holds
at 0.9999988, so it is redistribution, not displacement — but it is enough to fail a 0.1% flux
check.

The trap, and the reason this page exists: **correcting the hardware call does not work.** Nudging
`sinf(hi)` by the low word of the angle — `sin(hi + lo) ≈ sinf(hi) + lo·cosf(hi)` — still fails,
because the error is `sinf`'s own rounding, not the angle's missing low bits. You cannot correct an
error baked into the value you started from. To beat `sinf`'s rounding you stop calling it and
evaluate sine and cosine yourself, in arithmetic that carries more than one float.

## The pair version

Two classical pieces — range reduction and polynomial evaluation — both done in df arithmetic so no
float rounding creeps back in. Here is the whole function, `df_sincos`, in four steps.

**1. Range reduction.** A polynomial only approximates sine near zero, so fold the angle into
[−π/4, π/4]: subtract the nearest multiple of π/2 and remember which quarter-turn you removed.

    const float2 PIO2        = make_float2(1.570796371e+00f, -4.371138829e-08f); // pi/2 as a PAIR
    const float  TWO_OVER_PI = 6.366197467e-01f;                                  // 2/pi
    const float  kf = rintf(phi.x * TWO_OVER_PI);        // which quarter-turn phi is in (= k)
    const float2 r  = df_sub(phi, df_mul_f(PIO2, kf));   // r = phi - k*(pi/2), now in [-pi/4, pi/4]
    const float2 r2 = df_mul(r, r);                      // r^2, reused by both polynomials

The catch: π/2 is irrational, so `k·(π/2)` does not fit in a float, and subtracting a rounded copy
of it would reintroduce exactly the error we are avoiding. Storing π/2 as a (hi, lo) pair and
subtracting in df keeps the cancellation exact — `r` comes out as a pair.

> W. J. Cody and W. Waite, *Software Manual for the Elementary Functions*, Prentice-Hall (1980) —
> the split-constant reduction. The free, readable version is K. C. Ng, *Argument Reduction: Good
> to the Last Bit* (1992), https://www.validlab.com/arg.pdf.

**2. sin(r), Horner in pairs.** The Taylor series `sin(r) = r·(1 − r²/6 + r⁴/120 − …)`, evaluated
inner-to-outer. Every coefficient is a pair — its `.y` is the bit the float cannot hold — and every
step is a `df_add`/`df_mul`, so low words carry through instead of rounding off:

    const float2 SC1=make_float2(-1.666666716e-01f, 4.967053879e-09f); // -1/6   as (hi, lo)
    const float2 SC2=make_float2( 8.333333768e-03f,-4.346172033e-10f); //  1/120
    const float2 SC3=make_float2(-1.984127011e-04f, 2.725596875e-12f); // -1/5040
    const float2 SC4=make_float2( 2.755731884e-06f, 3.793571224e-14f); //  1/362880
    float2 s = df_add(df_mul(SC4, r2), SC3);   // SC4*r^2 + SC3
    s = df_add(df_mul(s, r2), SC2);            // *r^2 + SC2
    s = df_add(df_mul(s, r2), SC1);            // *r^2 + SC1
    s = df_add_f(df_mul(s, r2), 1.0f);         // *r^2 + 1
    const float2 sin_r = df_mul(r, s);         // r * (1 + c1*r^2 + c2*r^4 + ...)

Four terms hold the truncation error near 2×10⁻⁸ over the reduced interval — still under the
hardware's 6×10⁻⁸ rounding, and past what the flux check needs. (Fewer terms work because Taylor
is exact at r = 0; near a Bragg peak the angles are tiny, so origin-fidelity, not uniform-interval
accuracy, is what the fringe sum depends on.)

**3. cos(r), the same machinery.** Series `cos(r) = 1 − r²/2 + r⁴/24 − …`:

    const float2 CC1=make_float2(-5.000000000e-01f, 0.0f);             // -1/2
    const float2 CC2=make_float2( 4.166666791e-02f,-1.241763470e-09f); //  1/24
    // ... CC3 = -1/720, CC4 = 1/40320 ...
    float2 c = df_add(df_mul(CC4, r2), CC3);
    c = df_add(df_mul(c, r2), CC2);
    c = df_add(df_mul(c, r2), CC1);
    const float2 cos_r = df_add_f(df_mul(c, r2), 1.0f);

**4. Quadrant fold.** Steps 2–3 gave sin and cos of the reduced `r`; we want them for the original
`phi = r + k·(π/2)`. Each quarter-turn just swaps sin↔cos and flips a sign, in a fixed pattern:

    const int q = ((int) kf) & 3;                          // k mod 4
    const float2 neg_sin = make_float2(-sin_r.x, -sin_r.y);
    const float2 neg_cos = make_float2(-cos_r.x, -cos_r.y);
    if      (q == 0) { *sinphi = sin_r;   *cosphi = cos_r;   }  // phi = r
    else if (q == 1) { *sinphi = cos_r;   *cosphi = neg_sin; }  // phi = r + pi/2
    else if (q == 2) { *sinphi = neg_sin; *cosphi = neg_cos; }  // phi = r + pi
    else             { *sinphi = neg_cos; *cosphi = sin_r;  }   // phi = r + 3pi/2

The two results go back through pointers — `sinphi` and `cosphi` — because there are two of them.

The rotation is then the same formula as before with pair trig, the `_f` multiplies promoted to
full pair multiplies:

    float2 sinphi, cosphi;
    df_sincos(phi, &sinphi, &cosphi);                 /* phi and its trig are pairs now */
    ...
    newv[1] = df_add(df_mul(v1, cosphi), df_mul(cross, sinphi));   /* pair * pair */

Nothing collapses to float anywhere in the rotation.

## Why float-only, and what it costs

That polynomial is dozens of float operations — far more *instructions* than a single `double`
`sincos`. It is still faster, because not all instructions cost the same: consumer GPUs run double
at ~1/64 the float rate (roughly 2 FP64 lanes per SM beside 128 float lanes), so a pile of float
ops on the wide pipe beats a handful of double ops on the narrow one. Measured on the hardest cell
(2048², this box):

    bare sinf/cosf    4.10 ms    7 digits   fails
    df_sincos         4.46 ms   14 digits   passes   (+9% over the broken-but-fast version)
    double sincos     9.10 ms   14 digits   passes   (+122%)

Full precision for a 9% bump instead of 122% — and no extra registers, because df stays in 32-bit
lanes, while the double temporaries would raise the kernel's register high-water mark by four, a
cost every launch pays whether or not it takes the curved branch.

- **The angle must arrive as an accurate pair** — here `offset / distance` formed with a df divide.
  Hand it a float widened with a zero low word and you get float accuracy straight back.
- **A pair is not a double** — ~14 digits versus 16. On the hardest case the pair lands a few parts
  in 10⁴ from unity where true double lands a few parts in 10⁶; both clear a 10⁻³ check, but a 10⁻⁵
  check on this path would need FP64.
- **Scope** — curved detector at fine sampling only. The flat path never calls it, and the
  single-precision kernel keeps bare `sinf`/`cosf`: small angle, one rounding, no sharp fringe
  downstream for it to shift.

## In one sentence

`sinf` gives the right answer rounded to a float; on a needle-sharp fringe that rounding is a tenth
of a fringe, and correcting the *angle* cannot remove an error that lives in the *function's* last
bit — so you rebuild sin and cos in pairs.

## The pattern

When a float result is not accurate enough, ask whether the error is in the *input* or the
*operation*. If it is the input's magnitude, peel it off exactly and a float becomes a precision
instrument for what remains. If it is the operation's own rounding, you cannot patch the output —
you redo the operation in more precision, which on this hardware is a pair of floats, not a double.
