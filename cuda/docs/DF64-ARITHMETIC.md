# How to Approximate a Double Using Two Floats

Consumer GPU hardware deliberately nerfed double precision: FP64 instructions run at 1/32 to
1/64 of FP32 speed on recent consumer parts and they only have 2 FP64 pipelines compared to
32 or 64 for FP32. When a computation genuinely needs double-like
accuracy there, the only fast units available are the float ones — so the precision has to be
rebuilt out of floats. The trick: store each value as a *pair* of floats and spend a few extra
float instructions per operation, recovering nearly the accuracy of a double. It is an
approximation — a very good one — and this page explains how it works.

The technique was invented by T. J. Dekker in 1971, long before GPUs. We picked it up from
Andrew Thall's GPU write-up, which is the one reference to read for more depth:

> Andrew Thall, *Extended-Precision Floating-Point Numbers for GPU Computation* (2007)
> https://andrewthall.org/papers/df64_qf128.pdf

Thall named the float-pair type **df64** ("double-float"), and that's the shorthand used
throughout this repo's code and docs. Each type in this story goes by three names, used
interchangeably in this document and in the code:

- **float32 = fp32 = float** — the 32-bit type; full speed on every GPU
- **float64 = fp64 = double** — the 64-bit type; the one consumer GPUs throttle
- **df64 = double-float = float2** — the pair of floats acting as one nearly-double value
  (`float2` is the CUDA type it lives in: `.x` = hi, `.y` = lo)

To avoid any confusion, here is exactly what is and is not a double in everything below:

- Every variable in these routines is declared `float` (or `float2`, which is simply two
  floats side by side). Not one of them is declared `double`.
- Every instruction executed is an FP32 instruction — float add, float multiply, or float
  fused multiply-add. The GPU's FP64 units are never touched.
- The df64 pair is **not** a double. It is two floats whose sum *approximates* the value a
  double would hold — good to ~14 decimal digits, where a real double carries ~16.
- Real doubles appear in exactly one place: on the host CPU, before and after the kernel
  runs. A double is split into a pair on the way in, and a pair can be reassembled into a
  double — exactly, with nothing lost — on the way out. Both conversions are shown at the end
  of section 1.

## 1. The representation: a hi/lo pair

A df64 value stores a number x as two float32s whose sum is the number:

    x = hi + lo

`hi` carries the largest portion: it is the closest a single float32 can get to x. `lo` carries
the remainder: the small part that got rounded away, which happens to fit in a float32 of its
own. Because `lo` is so much smaller than `hi`, their digits don't overlap, and together the pair
holds about twice the digits of one float.

In code the pair is CUDA's built-in `float2` — no custom type needed. Stripped of its
decoration macros, the CUDA header defines it as exactly what it sounds like, and a companion
header supplies the constructor:

    // vector_types.h — two floats, 8 bytes, aligned and moved as one unit
    struct float2 { float x; float y; };

    // vector_functions.h — returns {.x = x, .y = y}
    float2 make_float2(float x, float y);

The df64 convention, used in every example below and throughout the kernels, is:

    float2 v = make_float2(hi, lo);      // v.x = hi,  v.y = lo
    // the number represented is v.x + v.y

So wherever a `float2` appears in the df64 code, read it as one ~14-digit number, not as two
unrelated floats.

Where the digit counts come from: a float's 32 bits are three separate fields — 1 sign bit, 8
exponent bits, and 23 significand bits (24 counting the leading 1 that normalized values get for
free). The exponent bits are *not* part of the 24: the exponent only sets the scale — where the
point sits, hence the ~10^±38 range — while precision, how many digits you can trust, comes
entirely from the significand. Each decimal digit costs about 3.3 bits (10 ≈ 2^3.3), so 24
significand bits buy ~7 decimal digits. The same accounting across the types:

| type | exponent field (sets range) | significand bits (set precision) | ≈ decimal digits |
|---|---|---|---|
| float (FP32) | 8 bits, ~10^±38 | 24 | ~7 |
| double (FP64) | 11 bits, ~10^±308 | 53 | ~15–16 |
| df64 pair | 8 bits, ~10^±38 (still float's) | ~49 (24 + 24, +1 from `lo`'s own sign) | ~14–15 |

So the pair lands just shy of a real double — close enough for most purposes — while every
instruction executed is an ordinary float add, multiply, or FMA.

Getting in and out of the pair is cheap, and the round trip through double is exact. The
double-to-pair split is a real function in the kernel source, run on the host (where doubles
are full speed) before anything is uploaded to the GPU:

    // double -> df64: hi grabs all the digits a float can hold; lo catches what it missed
    float2 make_real_double(double v) {
        float hi = (float) v;                    // nearest float to v: the first ~7 digits
        float lo = (float) (v - (double) hi);    // what the cast rounded away: the next ~7
        return make_float2(hi, lo);
    }

The other three conversions are one-liners:

    // float -> df64: nothing was rounded away, the remainder is zero
    float2 x = make_float2(f, 0.0f);

    // df64 -> float: keep the big half; x.y is below float precision by construction
    float f = x.x;

    // df64 -> double: promote both halves and add -- exact, recovers all ~49 bits
    double v = (double) x.x + (double) x.y;

The reason this works at all is one guarantee of IEEE-754 arithmetic: when the hardware rounds
the result of a float32 add or multiply, the amount it rounded off is *itself exactly
representable as a float32* — and a short fixed sequence of float32 operations can recover it
exactly. Those recovery sequences are the "error-free transformations" below.

## 2. The error-free transformations (EFTs)

Only two primitives in the whole scheme touch rounding error directly; everything else is
composed from them. The code's names are `df_two_sum` and `df_two_prod`, and both *return a
float2*: the rounded result lands in `.x` (hi) and the captured error in `.y` (lo) — which is
exactly the invariant from section 1. An EFT doesn't just measure what the hardware rounded
off; its output IS a ready-made df64.

**df_two_sum** (Knuth) — add two floats and capture the exact rounding error. Works for any
inputs, branch-free:

    // guarantee: s + e == a + b EXACTLY (s = the correctly rounded float sum)
    float2 df_two_sum(float a, float b) {
        float s  = a + b;
        float bb = s - a;
        float e  = (a - (s - bb)) + (b - bb);
        return make_float2(s, e);        // s -> hi, e -> lo: a valid df64
    }

**df_two_prod** — the multiply counterpart: capture the exact rounding error of one float
multiply. With hardware FMA it is two instructions:

    // guarantee: p + e == a * b EXACTLY
    float2 df_two_prod(float a, float b) {
        float p = a * b;
        float e = fmaf(a, b, -p);        // the multiply's exact error, in one instruction
        return make_float2(p, e);
    }

`fmaf(a, b, c)` — fused multiply-add — is the star here: one full-speed instruction that
computes a·b + c with a *single* rounding at the very end, holding the intermediate product
a·b exact inside the unit. Hand it c = −p and what comes back is the exact product minus the
rounded product: precisely the error the multiply threw away. (The kernels call the CUDA
intrinsic `__fmaf_rn` — the same operation with round-to-nearest spelled out, so no compiler
setting can reinterpret it.)

**df_quick_two_sum** (Dekker) — df_two_sum in half the operations, valid only when |a| ≥ |b|:

    // guarantee: s + e == a + b exactly -- but VALID ONLY IF |a| >= |b|
    float2 df_quick_two_sum(float a, float b) {
        float s = a + b;
        float e = b - (s - a);
        return make_float2(s, e);
    }

Note there is no runtime check on that requirement — adding one (compare, maybe swap) would
put a data-dependent branch in the kernel, and divergent paths are exactly what CUDA code
avoids. So the rule in this codebase: on operands of unknown size order, always the
branch-free df_two_sum; df_quick_two_sum appears in exactly one situation — the final
repackaging step inside the pair operations below, where the first argument is already known
to dominate *by construction*, so the precondition holds without ever being tested.

These are exact identities of round-to-nearest IEEE-754 arithmetic, not approximations. Exact
error capture is what keeps pair arithmetic from drifting the way naive float code does.

## 3. Arithmetic on pairs

Full df64 operations are short compositions of the EFTs, and the names pair up: df_two_sum
grows into **df_add**, df_two_prod into **df_mul**. These are the kernel's actual routines
(`.x` = hi, `.y` = lo throughout). Each ends by repackaging its result into a clean hi/lo
float2 — the one legitimate df_quick_two_sum spot, because by then the big part is known:

    // df64 + df64                          (~20 float ops)
    float2 df_add(float2 x, float2 y) {
        float2 s = df_two_sum(x.x, y.x);     // add the high halves, error captured
        float2 t = df_two_sum(x.y, y.y);     // and the low halves -- keeps accuracy when x and y nearly cancel
        s.y += t.x;
        s = df_quick_two_sum(s.x, s.y);      // repackage: s.x dominates by construction
        s.y += t.y;
        return df_quick_two_sum(s.x, s.y);
    }

    // df64 * df64                          (~7 float ops)
    float2 df_mul(float2 x, float2 y) {
        float2 p = df_two_prod(x.x, y.x);                  // exact product of the high halves
        float  e = fmaf(x.x, y.y, fmaf(x.y, y.x, p.y));    // cross terms x.hi*y.lo + x.lo*y.hi
        return df_quick_two_sum(p.x, e);                   // (lo*lo would land below rounding: dropped)
    }

The literature also has a cheaper "sloppy" df_add that skips the low-half df_two_sum — three
ops saved, but its accuracy guarantee dies when opposite-sign operands nearly cancel. This
codebase pays for the safe version. The `_f` variants (df_add_f, df_mul_f) are the same shapes
with a plain float as the second operand: its lo is zero, so terms drop out and they run a few
ops cheaper.

**Could the 20-flop add be cheaper? We checked (literature review, 2026-07), and the answer
is a proven no — with one rehabilitation:**

- Zhang & Aiken (SC'25, DOI 10.1145/3712285.3759876) machine-verified, by exhaustive search
  over every branch-free floating-point network up to the relevant size, that nothing smaller
  than the 20-op structure computes a fully general, cancellation-safe double-word sum. Their
  own "optimal" network is the same 20 operations rescheduled for instruction-level
  parallelism. 20 flops is the floor, not folklore.
- Muller & Rideau (ACM TOMS 48(1), 2022, DOI 10.1145/3484514) proved the 11-flop sloppy add
  satisfies the *same* 3u² error bound as the 20-flop add, for all inputs, under the modified
  relative-error metric |error|/(|x|+|y|) — the metric that matters in long chained
  computations. This result is why the QD and SLEEF libraries default to the sloppy add in
  production. (Same-sign operands were already proven safe in the classical metric: Graillat,
  Lauter, Tang, Yamanaka & Oishi, ACM TOMS 41(4), 2015.)

The practical rule that falls out: pay 20 when arbitrary operands may cancel and the result
must be accurate relative to *itself*; pay 11 when the operands provably can't cancel, or
when accuracy relative to the operands' magnitude is the actual requirement.

**Division** has no EFT of its own, and doesn't need one. The pattern — used for df_div and
df_sqrt both — is guess-measure-correct: take a plain float guess, measure the miss in df64
(so the measurement loses nothing), and add the correction. One refinement turns ~7 correct
digits into ~14, exactly the gap between one float and the pair. The kernel's division:

    // df64 / df64
    float2 df_div(float2 x, float2 y) {
        float  q     = x.x / y.x;                    // float quotient guess: first ~7 digits
        float2 resid = df_sub(x, df_mul_f(y, q));    // the part of x the guess failed to cover
        float  corr  = resid.x / y.x;                // quotient of the miss: the next ~7 digits
        return df_add_f(make_float2(q, 0.0f), corr);
    }

It is long division in float-sized bites: quotient block, exact remainder, next quotient
block. (df_sub is df_add with y's signs flipped.) Square root (`df_sqrt`) is the same pattern
seeded from the hardware `rsqrtf`. Both operations are in the papers: Dekker's original gives
division, and Thall's write-up works out df64 division and square root in full.

## 4. The dot product — deferring the renormalization

A dot product is a sum of products. For the Miller index it is `x[1]·y[1] + x[2]·y[2] + x[3]·y[3]`, where every number is a (hi, lo) pair. The obvious way to compute it reuses the tools from section 3 — multiply each pair of pairs with `df_mul`, add each product into a running total with `df_add`:

    float2 sum = {0, 0};
    sum = df_add(sum, df_mul(x[1], y[1]));
    sum = df_add(sum, df_mul(x[2], y[2]));
    sum = df_add(sum, df_mul(x[3], y[3]));
    return sum;

That is the readable statement of what the dot means, and for three terms it is perfectly correct. It is also doing more work than it needs to. Every `df_mul` and every `df_add` ends by *renormalizing* — a final `df_quick_two_sum` that repackages its (hi, lo) result into a clean, non-overlapping pair for the next operation to consume. That cleanup earns its keep when you are handing a value to someone else. In the middle of a sum it is wasted: the tidy pair goes straight into the next `df_add`, which disturbs the low word all over again. You pay to clean up after every term and then immediately un-clean it.

The faster arrangement stops cleaning up between terms. Carry two running quantities: one high word holding the sum so far, and one plain float where every leftover correction piles up. For each term, take the product's rounding error and the addition's leftover and drop both onto that pile — no repackaging. Renormalize exactly once, after the last term. This is the **Dot2** algorithm of Ogita, Rump, and Oishi.

> T. Ogita, S. M. Rump, S. Oishi, *Accurate Sum and Dot Product* (2005)
> https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf

Same answer, roughly **39 float operations instead of 81** for the three-term dot — a bit under half the work — because the per-term repackaging is gone.

**The one wrinkle for our pairs.** Plain Dot2 assumes each input is a single number, so each product has exactly one rounding error to carry. Here each input is a (hi, lo) pair, so the exact product of two pairs has more pieces:

    a·b = a_hi·b_hi + (a_hi·b_lo + a_lo·b_hi) + a_lo·b_lo

The leading `a_hi·b_hi` is the ordinary product. The two middle **cross terms** are the extra baggage the pair representation carries; we fold them onto the correction pile with a fused multiply-add, next to the rounding error already sitting there. The last term, `a_lo·b_lo`, is too small to matter — below the pair's ~14-digit resolution — and is dropped, the same drop `df_mul` already makes. That fold is the only change needed to fit the published single-number algorithm to our pairs, and every operation stays on the float pipe.

### One term at a time

The running total starts at (0, 0). Every term then runs the same short routine. Writing the
running total as the pair (sum_hi, sum_lo) and the term being added as a·b — where a and b are
each pairs, a = a_hi + a_lo and b = b_hi + b_lo:

    p       = df_two_prod(a_hi, b_hi)                    // exact product of the high words
    e       = fmaf(a_hi, b_lo, fmaf(a_lo, b_hi, p_lo))   // add the two cross terms onto its error
    s       = df_two_sum(sum_hi, p_hi)                   // add into the running high word
    sum_hi  = s_hi                                       // advance the high word
    sum_lo += s_lo + e                                   // pool both leftovers, unrenormalized

1. **Multiply the high words exactly.** `df_two_prod` returns the product of the two high words
   as a pair: `p_hi` is the rounded product, `p_lo` the bit rounding threw away.
2. **Fold in the cross terms.** Because a and b are pairs, the true product also carries
   `a_hi·b_lo + a_lo·b_hi`; two fused multiply-adds add those onto `p_lo`, giving one per-term
   correction `e` that holds everything the high-word product left out.
3. **Add into the running total.** `df_two_sum` adds `p_hi` to the running high word and hands
   back the rounded sum `s_hi` plus the exact remainder `s_lo` — the part of that addition that
   did not fit.
4. **Advance the high word.** `s_hi` becomes the new running high word.
5. **Pool the leftovers.** Both corrections — the addition's remainder `s_lo` and the term's
   correction `e` — are added onto the running low word. Nothing is repackaged into a clean
   pair; the low word just holds a growing pile of corrections.

Steps 4 and 5 are the whole point: the high word walks forward while the corrections wait
together in the low word, so no term pays for a renormalization. After the last term, one final
`df_quick_two_sum` folds the low word back into the high word and returns a clean pair — the
single cleanup the whole arrangement was built to reach.

**Why deferring is safe.** It is fair to worry that skipping the per-term cleanup loses accuracy. It does not, for one reason: the two-sum is exact for numbers of *any* sign. When two terms are large, opposite, and nearly cancel — the case that wrecks a naive float sum — the two-sum's leftover captures exactly the digits the cancellation would otherwise destroy, and they land safely on the correction pile. Deferring keeps every one of those corrections; it simply adds them all in at the end instead of after each term. That is the line between deferring the renormalization (safe) and a "sloppy" add that skips the correction entirely (not safe — that one throws digits away).

## 5. What the pair buys — and what it does not

Each df64 operation is accurate to about 14 decimal digits. Compared to a real float64:

- **The exponent range is still float32's.** Overflow near 3.4×10³⁸, and precision fades for
  magnitudes below ~10⁻³¹ (where `lo` falls off the bottom of the float32 range). A real float64
  reaches ~10±³⁰⁸.
- **Results are not bit-identical to float64.** Each operation is slightly differently rounded,
  and those tiny differences compound. df64 tracks a float64 computation to ~14 digits — it does
  not reproduce it exactly.
- **It costs roughly 10–20 float32 ops per operation.** That is the whole economic argument: on
  hardware where float64 runs at half the float32 rate, just use float64; on consumer GPUs,
  where float64 runs at 1/32–1/64 of the float32 rate, paying ~10–20 float32 ops for
  double-quality arithmetic wins decisively.

## 6. What breaks it

The EFTs are exact only under strict round-to-nearest IEEE-754 rules:

- **Value-unsafe compiler optimization.** On paper, `e = b - (s - a)` simplifies to `e = b` —
  destroying exactly the rounding information being recovered. Fast-math style flags
  (reassociation, flush-to-zero) silently zero out the error terms. These routines must be
  compiled with value-safe floating point.
- **Non-compliant hardware.** Not a concern on anything modern: every mainstream GPU since ~2010
  provides correctly rounded float32 add, multiply, and FMA. (Thall's 2006-era GPUs did not,
  which is why his paper audits rounding op-by-op.)
