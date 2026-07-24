/* metrics.c -- parity comparator for two float32 nanoBragg images.
 *
 * The corr/sum_ratio definitions are fixed by the harness gate; keep them exact.
 *
 *   Usage: metrics <gpu_float32.bin> <cpu_float32.bin>
 *          metrics --build-commit    (print the commit HEAD was at when this
 *                                     binary was compiled, for the Layer-2
 *                                     ledger's run provenance; see Makefile)
 *
 * Reads two little-endian float32 images, promotes to double, and prints:
 *
 *   corr sum_ratio max_rel worst_pixel_frac peak_max_rel worst_is_peak
 *
 *   corr             Pearson correlation gpu vs cpu (nan if either image is constant)
 *   sum_ratio        sum(gpu)/sum(cpu)          (nan if sum(cpu)==0)
 *   max_rel          worst per-pixel |g-c|/max(|g|,|c|) over den>1e-12
 *   worst_pixel_frac cpu value at the max_rel pixel / cpu image max
 *   peak_max_rel     worst per-pixel rel error over bright pixels (cpu >= 1% of cpu max)
 *   worst_is_peak    PEAK if worst_pixel_frac >= 0.01 else DIM
 *
 * Formats: %.7f %.6f %.3e %.3e %.3e %s. The gate uses corr + sum_ratio only.
 *
 * Reductions use long double accumulators so the printed corr/sum_ratio match
 * numpy's pairwise-summed values to the printed precision.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef NB_BUILD_COMMIT
#define NB_BUILD_COMMIT "unknown"
#endif

static float *read_f32(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "metrics: cannot open %s\n", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long bytes = ftell(f);
    if (bytes < 0) { fclose(f); return NULL; }
    rewind(f);
    size_t n = (size_t)bytes / sizeof(float);
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) { fclose(f); fprintf(stderr, "metrics: OOM (%zu floats)\n", n); return NULL; }
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) { free(buf); fprintf(stderr, "metrics: short read on %s\n", path); return NULL; }
    *count = n;
    return buf;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--build-commit") == 0) {
        puts(NB_BUILD_COMMIT);
        return 0;
    }
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <gpu_float32.bin> <cpu_float32.bin>\n", argv[0]);
        return 2;
    }

    size_t ng = 0, nc = 0;
    float *gf = read_f32(argv[1], &ng);
    float *cf = read_f32(argv[2], &nc);
    if (!gf || !cf) { free(gf); free(cf); return 2; }

    if (ng != nc)
        fprintf(stderr, "SIZE_MISMATCH gpu=%zu cpu=%zu\n", ng, nc);
    size_t n = ng < nc ? ng : nc;

    /* Pass 1: sums, means, extremes (for constant-image / cmax detection). */
    long double sg = 0.0L, sc = 0.0L;
    double gmin = 0.0, gmax = 0.0, cmin = 0.0, cmax_signed = 0.0, cmax = 0.0;
    for (size_t i = 0; i < n; i++) {
        double g = (double)gf[i];
        double c = (double)cf[i];
        sg += (long double)g;
        sc += (long double)c;
        if (i == 0) {
            gmin = gmax = g;
            cmin = cmax_signed = c;
            cmax = c;
        } else {
            if (g < gmin) gmin = g;
            if (g > gmax) gmax = g;
            if (c < cmin) cmin = c;
            if (c > cmax_signed) cmax_signed = c;
            if (c > cmax) cmax = c;
        }
    }

    double sum_ratio = (sc != 0.0L) ? (double)(sg / sc) : NAN;

    /* Pearson corr: nan if either image is constant (numpy std == 0). */
    int g_const = (n == 0) || (gmin == gmax);
    int c_const = (n == 0) || (cmin == cmax_signed);
    double corr;
    if (!g_const && !c_const) {
        long double mg = sg / (long double)n;
        long double mc = sc / (long double)n;
        long double cov = 0.0L, vg = 0.0L, vc = 0.0L;
        for (size_t i = 0; i < n; i++) {
            long double dg = (long double)gf[i] - mg;
            long double dc = (long double)cf[i] - mc;
            cov += dg * dc;
            vg += dg * dg;
            vc += dc * dc;
        }
        corr = (double)(cov / sqrtl(vg * vc));
    } else {
        corr = NAN;
    }

    /* Pass 2: per-pixel relative error, worst pixel, bright-pixel worst. */
    double max_rel = -1.0;
    size_t worst_idx = 0;
    double peak_max_rel = 0.0;
    double bright_thresh = 0.01 * cmax;
    for (size_t i = 0; i < n; i++) {
        double g = (double)gf[i];
        double c = (double)cf[i];
        double ag = fabs(g), ac = fabs(c);
        double den = ag > ac ? ag : ac;
        double rel = (den > 1e-12) ? fabs(g - c) / den : 0.0;
        if (rel > max_rel) { max_rel = rel; worst_idx = i; }
        if (cmax > 0.0 && c >= bright_thresh) {
            if (rel > peak_max_rel) peak_max_rel = rel;
        }
    }
    if (max_rel < 0.0) max_rel = 0.0; /* n == 0 guard */

    double worst_pixel_frac = 0.0;
    if (cmax > 0.0 && n > 0)
        worst_pixel_frac = (double)cf[worst_idx] / cmax;

    const char *worst_is_peak = (worst_pixel_frac >= 0.01) ? "PEAK" : "DIM";

    printf("%.7f %.6f %.3e %.3e %.3e %s\n",
           corr, sum_ratio, max_rel, worst_pixel_frac, peak_max_rel, worst_is_peak);

    free(gf);
    free(cf);
    return 0;
}
