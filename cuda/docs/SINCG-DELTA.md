# Why sincg Gets a Fraction, Not the Whole Miller Index

sincg is the N-slit interference function sin(N·x)/sin(x) — the closed form for N unit cells
diffracting along one axis. The lattice factor multiplies it across the three axes (h, k, l)
in the innermost loop, so it is on the hottest path in the kernel. This page compares two
versions of the same five-line function: the straightforward one, which is correct in double
and fails in float, and the delta-reduced one, which is exact, fast, and float-safe. Whatever
spelling the kernel uses at any given moment, compare it against these two and you will know
which side of the line it is on and why.

The motivation is hardware: consumer GPUs deliberately throttle double precision — a handful
of FP64 lanes beside 128 float lanes per SM, roughly 1/64 the throughput — so running the
double version of sin here is a losing move, and a float variant is the preference for
speed. The question is how a float variant gets to be precise enough. The delta version is
the answer.

## The straightforward version

The reference implementation (the CPU code, which always runs it in double):

    /* Fourier transform of a grating */
    double sincg(double x, double N) {
        if(x==0.0) return N;
        return sin(x*N)/sin(x);
    }

    if(Na>1){ F_latt *= sincg(M_PI*h, Na); }

The argument is the whole Miller index converted to an angle, x = π·h, so the numerator
computes sin(π·N·h). Nothing bounds it: sharper crystals (bigger N) and higher resolution
(bigger h) grow it without limit — 10⁴–10⁵ radians on realistic cells. The guard handles the
removable singularity at the origin (the direct beam): as x → 0 the ratio tends to N.

### What happens when this version runs in float

A float carries ~7 significant digits *of whatever magnitude it holds* — precision is
relative. And sine needs exactly one thing from its argument: the position within the current
turn, a number between 0 and 2π. The turn count is dead weight.

At h ≈ 137.0017 and N = 1000, π·N·h ≈ 430,403 radians — about 68,500 full turns plus 5.34
radians of position. Floats at that magnitude are spaced **0.031 radians apart**, so the
moment the multiply produces this value it is rounded to ±0.016 radians: nearly the whole
digit budget went to counting turns, and the position — the only part sine needs — got the
scraps. It worsens with magnitude: near 10⁷ radians the float spacing reaches a full radian,
and past ~7×10⁷ the spacing is 8 radians — more than a whole turn — so the float can no
longer even say which revolution it is on.

The math library cannot rescue this. `sinf`'s huge-argument routine — the Payne–Hanek
reduction, after M. H. Payne and R. N. Hanek, who published the algorithm in 1983 (full
citation below) — unwinds the turns of the *stored* value exactly: roughly 120 instructions,
with double-precision steps and warp-divergent branches inside. But the stored value was
already wrong at creation. Slow
*and* garbage. So in float this version fails twice, and the traditional remedy is to run it
in double: correct, but on consumer GPUs every double operation queues for 2 FP64 lanes per
SM beside 128 float lanes (~1/64 the throughput). Correctness at a 64× toll.

## The delta version

First, the identity that licenses it. Sine doesn't care how many times you've gone around
the circle. Split the Miller index into integer and fractional parts, h = h₀ + δ with
δ = h − rint(h) and |δ| ≤ ½. The denominator collapses:

    sin(π·h) = sin(π·h₀ + π·δ) = ± sin(π·δ)

because π·h₀ is a whole number of half-turns — sin(π·h₀) = 0, cos(π·h₀) = ±1 for integer h₀.
The numerator collapses the same way, and this is the step that makes the trick work: N is an
integer too, so N·h₀ is *also* a whole number of half-turns:

    sin(π·N·h) = sin(π·N·h₀ + π·N·δ) = ± sin(π·N·δ)

Divide, then square (F_latt enters the intensity squared, so the signs wash out):

    sincg(πh, N)² = [ sin(πNδ) / sin(πδ) ]²        — exact, zero approximation

One ordering rule makes the peel itself exact: it must happen in *index units*, before any
multiply by π. Integers are exactly representable, so `h − rintf(h)` loses nothing; after
converting to radians, a "whole turn" is a multiple of an irrational number and nothing can
be subtracted exactly.

The delta version is then the same five lines with a smaller input:

    /* removable singularity: as delta->0, sin(pi*N*delta)/sin(pi*delta) -> N.
       Guard delta==0 (Bragg peak / integer h) to avoid 0/0 = NaN. */
    float sincg_delta(float delta, float N) {
        const float PIf = 3.14159265358979323846f;
        if (delta == 0.0f) return N;
        return sinf(PIf * N * delta) / sinf(PIf * delta);
    }

    float dh = h - rintf(h);              /* peel whole turns in index units -- exact */
    if (Na > 1) F_latt *= sincg_delta(dh, Na);

Same ratio, same guard (now it also visibly prevents 0/0 = NaN exactly on a Bragg peak). The
function never got smarter; its input got smaller. The angles reaching `sinf` are now at most
π·N/2 and π/2 — and near a peak they are tiny, e.g. π·N·δ ≈ 5.34 radians in the example
above, where floats are spaced ~5×10⁻⁷ apart: every digit describes position on the circle.

A note on where π lives. Mathematically it makes no difference whether the caller converts to
radians and passes π·δ in, or the function takes the bare fraction and converts internally —
the two spellings agree to the last bit or two, nothing more. The compiler broke the tie:
measured on this kernel, the version with the multiply *inside* the function compiled to
faster code with fewer registers, and moving the conversion out to the callers lost that
optimization. Keeping it inside also puts π in exactly one place — callers deal only in
index-space fractions and never handle radians at all.

### Why it is faster — twice

- **Off the slow path.** `sinf` on a small argument is a handful of instructions; the
  unreduced version trips the ~120-instruction unwinding routine at three call sites per
  innermost iteration.
- **Off the slow pipe — the bigger win.** With the argument small, float trig becomes
  *accurate*, so double trig is no longer needed at all: the entire trig workload leaves the
  1/64-rate FP64 pipe for the full-speed float pipe. The straightforward version's real cost
  was never just the slow path — it was that its float form couldn't be trusted, which forced
  the whole evaluation into double.

### What it costs

- **δ must be accurate upstream.** The reduction fixes *how* the function is evaluated; the
  h,k,l computation fixes *what* δ is. The peel itself is exact, so δ inherits exactly the
  accuracy h arrived with. Computed in plain float, that is float accuracy — adequate for
  modest crystals, but at large N and high resolution the peak structure is finer than the
  ~7 digits a float spends on a full-size h can place, so the h,k,l chain must carry extra
  precision (double, or a double-equivalent built from pairs of floats) for δ to be worth
  reducing. The identity pays either way.
- **A misuse hazard.** A delta-taking function computes happily on anything; handing it a
  whole index rebuilds the huge angle internally and silently reintroduces both failures —
  plausible-looking images, one axis quietly wrong, several-fold slower, no compiler error.
  The parameter name, the peel sitting next to the call, and this page are the guard: the
  argument is always the peeled fraction, |δ| ≤ ½.

## The comparison in one sentence

430,403 radians and 5.34 radians point at the same spot on the circle; one spends a float's
entire digit budget saying "68,500 turns plus…" and the other spends every digit on the
"plus." The straightforward version needs double because of the *magnitude* it stores, not
the math it does; the delta version never builds the magnitude.

## Attribution and further reading

Reducing an argument before evaluating a periodic function is classical numerical practice;
none of it is original here. The specific pieces:

- M. H. Payne, R. N. Hanek, "Radian reduction for trigonometric functions," *ACM SIGNUM
  Newsletter* 18(1), 1983 — the huge-argument reduction algorithm that accurate math
  libraries (including CUDA's `sinf` slow path) implement.
- K. C. Ng, *Argument Reduction for Huge Arguments: Good to the Last Bit*, SunPro (1992),
  https://www.validlab.com/arg.pdf — the accessible treatment of the same problem, and the
  one reference to read for depth.
- The N-slit interference function itself is standard diffraction physics (the grating
  function); the reference implementation is nanoBragg's CPU `sincg`.

What is specific to this code is only the observation that the peel can be done *exactly*,
in index units, because Miller indices split into an integer part and a fraction — turning
heroic range reduction into one subtraction.

## The pattern

The same idea recurs wherever this code needs precision on float hardware: peel the big part
off *exactly*, and a plain float becomes a precision instrument for the small part that
remains. A float's digits go to whatever magnitude it holds — spend them on the part that
matters.
