/* nbmetrics.c -- parity comparator for two float32 nanoBragg images.
 *
 * Reports TWO metric sets from one FP32 image pair:
 *
 *   FP32 (gated)    -- corr, sum_ratio, and diagnostics, computed exactly as
 *                       the harness gate expects. Column order/format is
 *                       stable; keep it exact.
 *   FP16 (reported) -- both images rounded to IEEE binary16 (round-to-nearest-
 *                       even), then FP16 corr, FP16 sum_ratio, and the count
 *                       of pixels whose FP16 bit pattern differs (>=1 ULP).
 *                       Never gated -- FP16 epsilon (~5e-4) dwarfs the FP32
 *                       gate bar, so gating on it would be meaningless.
 *
 * Overflow safety: any pixel where either image's FP32 value is non-finite
 * (Inf/NaN) or exceeds the FP16 max magnitude (65504) is EXCLUDED from the
 * FP16 metrics (not clamped) -- the same exclusion mask is applied to both
 * images so the FP16 pass never NaN-contaminates. FP32 metrics are
 * unaffected and use every pixel as before.
 *
 * Usage:
 *   nbmetrics [--width W] [--height H] <candidate.bin> <reference.bin>
 *   nbmetrics --build-commit   (print the commit HEAD was at when this
 *                                binary was compiled, for the Layer-2
 *                                ledger's run provenance; see Makefile)
 *
 * Reads two little-endian float32 images, promotes to double, and prints:
 *
 *   corr sum_ratio max_rel worst_pixel_frac peak_max_rel worst_is_peak \
 *   fp16_corr fp16_sum_ratio fp16_diff_count fp16_excluded_count
 *
 *   corr              Pearson correlation candidate vs reference (nan if either image is constant)
 *   sum_ratio         sum(candidate)/sum(reference)   (nan if sum(reference)==0)
 *   max_rel           worst per-pixel |cand-ref|/max(|cand|,|ref|) over den>1e-12
 *   worst_pixel_frac  reference value at the max_rel pixel / reference image max
 *   peak_max_rel      worst per-pixel rel error over bright pixels (reference >= 1% of reference max)
 *   worst_is_peak     PEAK if worst_pixel_frac >= 0.01 else DIM
 *   fp16_corr         Pearson correlation of the two images after FP16 rounding,
 *                     over non-excluded pixels (nan if either is constant or all excluded)
 *   fp16_sum_ratio    sum(fp16 candidate)/sum(fp16 reference) over non-excluded pixels
 *   fp16_diff_count   # non-excluded pixels whose FP16 bit pattern differs (>=1 ULP)
 *   fp16_excluded_count  # pixels excluded from the FP16 pass (overflow/Inf/NaN in either image)
 *
 * Formats: %.7f %.6f %.3e %.3e %.3e %s %.7f %.6f %zu %zu. The gate uses
 * FP32 corr + sum_ratio only; the FP16 columns are reported, never gated.
 *
 * FP32 reductions use long double accumulators so the printed corr/sum_ratio
 * match numpy's pairwise-summed values to the printed precision. FP16
 * reductions likewise use long double accumulators over the rounded values.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>

#ifndef NB_BUILD_COMMIT
#define NB_BUILD_COMMIT "unknown"
#endif

#define FP16_MAX 65504.0f

static float *read_f32(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nbmetrics: cannot open %s\n", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long bytes = ftell(f);
    if (bytes < 0) { fclose(f); return NULL; }
    rewind(f);
    size_t n = (size_t)bytes / sizeof(float);
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) { fclose(f); fprintf(stderr, "nbmetrics: OOM (%zu floats)\n", n); return NULL; }
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) { free(buf); fprintf(stderr, "nbmetrics: short read on %s\n", path); return NULL; }
    *count = n;
    return buf;
}

/* IEEE-754 binary32 -> binary16 bit pattern, round-to-nearest-even.
 * Overflow (|value| > FP16 max, after rounding would hit Inf) and Inf/NaN
 * inputs are handled per IEEE FP16 semantics here; the caller applies the
 * spec's exclusion mask (§7) BEFORE trusting any FP16 metric derived from
 * these bits, so overflow never silently clamps into a gated number. */
static uint16_t f32_to_f16_bits(float value) {
    uint32_t x;
    memcpy(&x, &value, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t abs_x = x & 0x7FFFFFFFu;
    int32_t exp = (int32_t)((abs_x >> 23) & 0xFFu) - 127 + 15; /* rebias to half's bias */
    uint32_t mant = abs_x & 0x7FFFFFu;

    if (((abs_x >> 23) & 0xFFu) == 0xFFu) {
        if (mant) return (uint16_t)(sign | 0x7C00u | 0x0200u); /* NaN */
        return (uint16_t)(sign | 0x7C00u);                      /* Inf */
    }

    if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u); /* overflow -> Inf */
    }

    if (exp <= 0) {
        if (exp < -24) return (uint16_t)sign; /* far underflow -> signed zero */
        uint64_t mant_ext = (uint64_t)(mant | 0x800000u); /* restore implicit leading 1 */
        int shift = 14 - exp;                             /* in [14, 38] */
        uint64_t half_mant = mant_ext >> shift;
        uint64_t remainder = mant_ext & (((uint64_t)1 << shift) - 1);
        uint64_t halfway = (uint64_t)1 << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mant & 1))) half_mant++;
        if (half_mant == 0x400u) return (uint16_t)(sign | (1u << 10)); /* rounds up into smallest normal */
        return (uint16_t)(sign | (uint32_t)half_mant);
    }

    /* normalized: round 23-bit mantissa down to 10 bits, RNE */
    uint32_t half_mant = mant >> 13;
    uint32_t remainder = mant & 0x1FFFu;
    uint32_t halfway = 0x1000u;
    if (remainder > halfway || (remainder == halfway && (half_mant & 1))) {
        half_mant++;
        if (half_mant == 0x400u) {
            half_mant = 0;
            exp++;
            if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); /* rounded up to Inf */
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
}

static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1;
            do { e++; mant <<= 1; } while (!(mant & 0x400u));
            mant &= 0x3FFu;
            uint32_t fexp = (uint32_t)(127 - 15 - e);
            bits = sign | (fexp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t fexp = exp - 15u + 127u;
        bits = sign | (fexp << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Per §7: a pixel is excluded from the FP16 pass if either image's FP32
 * value is non-finite or exceeds FP16's representable magnitude -- applied
 * identically to both images so no NaN/Inf ever reaches an FP16 metric. */
static int fp16_excluded(float cand, float ref) {
    return !isfinite(cand) || !isfinite(ref) ||
           fabsf(cand) > FP16_MAX || fabsf(ref) > FP16_MAX;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--width W] [--height H] <candidate.bin> <reference.bin>\n"
        "       %s --build-commit\n",
        prog, prog);
}

int main(int argc, char **argv) {
    long width = 2048, height = 2048;
    int build_commit_flag = 0;

    enum { OPT_WIDTH = 1000, OPT_HEIGHT, OPT_BUILD_COMMIT, OPT_HELP };
    static struct option long_opts[] = {
        {"width",        required_argument, 0, OPT_WIDTH},
        {"height",       required_argument, 0, OPT_HEIGHT},
        {"build-commit", no_argument,       0, OPT_BUILD_COMMIT},
        {"help",         no_argument,       0, OPT_HELP},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (c) {
            case OPT_WIDTH:  width = strtol(optarg, NULL, 10); break;
            case OPT_HEIGHT: height = strtol(optarg, NULL, 10); break;
            case OPT_BUILD_COMMIT: build_commit_flag = 1; break;
            case OPT_HELP: usage(argv[0]); return 0;
            default: usage(argv[0]); return 2;
        }
    }

    if (build_commit_flag) {
        puts(NB_BUILD_COMMIT);
        return 0;
    }

    if (argc - optind != 2) {
        usage(argv[0]);
        return 2;
    }
    const char *cand_path = argv[optind];
    const char *ref_path = argv[optind + 1];

    size_t ncand = 0, nref = 0;
    float *cand = read_f32(cand_path, &ncand);
    float *ref = read_f32(ref_path, &nref);
    if (!cand || !ref) { free(cand); free(ref); return 2; }

    if (ncand != nref)
        fprintf(stderr, "SIZE_MISMATCH candidate=%zu reference=%zu\n", ncand, nref);
    size_t n = ncand < nref ? ncand : nref;

    if (width > 0 && height > 0) {
        size_t expected = (size_t)width * (size_t)height;
        if (ncand != expected)
            fprintf(stderr, "WARNING: candidate size %zu != width*height (%ld*%ld=%zu)\n",
                    ncand, width, height, expected);
        if (nref != expected)
            fprintf(stderr, "WARNING: reference size %zu != width*height (%ld*%ld=%zu)\n",
                    nref, width, height, expected);
    }

    /* ---- FP32 metrics (gated) -- unchanged math from the prior metrics.c ---- */

    long double sg = 0.0L, sc = 0.0L;
    double gmin = 0.0, gmax = 0.0, cmin = 0.0, cmax_signed = 0.0, cmax = 0.0;
    for (size_t i = 0; i < n; i++) {
        double g = (double)cand[i];
        double cc = (double)ref[i];
        sg += (long double)g;
        sc += (long double)cc;
        if (i == 0) {
            gmin = gmax = g;
            cmin = cmax_signed = cc;
            cmax = cc;
        } else {
            if (g < gmin) gmin = g;
            if (g > gmax) gmax = g;
            if (cc < cmin) cmin = cc;
            if (cc > cmax_signed) cmax_signed = cc;
            if (cc > cmax) cmax = cc;
        }
    }

    double sum_ratio = (sc != 0.0L) ? (double)(sg / sc) : NAN;

    int g_const = (n == 0) || (gmin == gmax);
    int c_const = (n == 0) || (cmin == cmax_signed);
    double corr;
    if (!g_const && !c_const) {
        long double mg = sg / (long double)n;
        long double mc = sc / (long double)n;
        long double cov = 0.0L, vg = 0.0L, vc = 0.0L;
        for (size_t i = 0; i < n; i++) {
            long double dg = (long double)cand[i] - mg;
            long double dc = (long double)ref[i] - mc;
            cov += dg * dc;
            vg += dg * dg;
            vc += dc * dc;
        }
        corr = (double)(cov / sqrtl(vg * vc));
    } else {
        corr = NAN;
    }

    double max_rel = -1.0;
    size_t worst_idx = 0;
    double peak_max_rel = 0.0;
    double bright_thresh = 0.01 * cmax;
    for (size_t i = 0; i < n; i++) {
        double g = (double)cand[i];
        double cc = (double)ref[i];
        double ag = fabs(g), ac = fabs(cc);
        double den = ag > ac ? ag : ac;
        double rel = (den > 1e-12) ? fabs(g - cc) / den : 0.0;
        if (rel > max_rel) { max_rel = rel; worst_idx = i; }
        if (cmax > 0.0 && cc >= bright_thresh) {
            if (rel > peak_max_rel) peak_max_rel = rel;
        }
    }
    if (max_rel < 0.0) max_rel = 0.0;

    double worst_pixel_frac = 0.0;
    if (cmax > 0.0 && n > 0)
        worst_pixel_frac = (double)ref[worst_idx] / cmax;

    const char *worst_is_peak = (worst_pixel_frac >= 0.01) ? "PEAK" : "DIM";

    /* ---- FP16 metrics (reported, not gated), overflow-safe ---- */

    long double sg16 = 0.0L, sc16 = 0.0L;
    float g16min = 0.0f, g16max = 0.0f, c16min = 0.0f, c16max = 0.0f;
    size_t n16 = 0, diff16 = 0, excluded16 = 0;
    for (size_t i = 0; i < n; i++) {
        if (fp16_excluded(cand[i], ref[i])) { excluded16++; continue; }
        uint16_t gb = f32_to_f16_bits(cand[i]);
        uint16_t cb = f32_to_f16_bits(ref[i]);
        if (gb != cb) diff16++;
        float gh = f16_bits_to_f32(gb);
        float ch = f16_bits_to_f32(cb);
        sg16 += (long double)gh;
        sc16 += (long double)ch;
        if (n16 == 0) {
            g16min = g16max = gh;
            c16min = c16max = ch;
        } else {
            if (gh < g16min) g16min = gh;
            if (gh > g16max) g16max = gh;
            if (ch < c16min) c16min = ch;
            if (ch > c16max) c16max = ch;
        }
        n16++;
    }

    double sum_ratio16 = (sc16 != 0.0L) ? (double)(sg16 / sc16) : NAN;

    int g16_const = (n16 == 0) || (g16min == g16max);
    int c16_const = (n16 == 0) || (c16min == c16max);
    double corr16;
    if (!g16_const && !c16_const) {
        long double mg16 = sg16 / (long double)n16;
        long double mc16 = sc16 / (long double)n16;
        long double cov16 = 0.0L, vg16 = 0.0L, vc16 = 0.0L;
        for (size_t i = 0; i < n; i++) {
            if (fp16_excluded(cand[i], ref[i])) continue;
            float gh = f16_bits_to_f32(f32_to_f16_bits(cand[i]));
            float ch = f16_bits_to_f32(f32_to_f16_bits(ref[i]));
            long double dg = (long double)gh - mg16;
            long double dc = (long double)ch - mc16;
            cov16 += dg * dc;
            vg16 += dg * dg;
            vc16 += dc * dc;
        }
        corr16 = (double)(cov16 / sqrtl(vg16 * vc16));
    } else {
        corr16 = NAN;
    }

    printf("%.7f %.6f %.3e %.3e %.3e %s %.7f %.6f %zu %zu\n",
           corr, sum_ratio, max_rel, worst_pixel_frac, peak_max_rel, worst_is_peak,
           corr16, sum_ratio16, diff16, excluded16);

    free(cand);
    free(ref);
    return 0;
}
