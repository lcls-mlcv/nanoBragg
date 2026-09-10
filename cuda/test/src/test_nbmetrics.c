/* test_nbmetrics.c -- unit test for nbmetrics (image-compare tool).
 *
 * Writes tiny 4x4 float32 fixture pairs, invokes the built nbmetrics binary
 * against each pair, and checks the printed FP32+FP16 columns:
 *
 *   (a) identical images       -> corr=1, sum_ratio=1, fp16_diff_count=0
 *   (b) known additive delta   -> corr close to 1, sum_ratio != 1
 *   (c) overflow/Inf/NaN pixel -> FP16 metrics finite (pixel excluded),
 *                                 FP32 metrics still computed (line printed)
 *
 * Usage: test_nbmetrics [path/to/nbmetrics]   (default: build/nbmetrics,
 * i.e. run from cuda/test/)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int write_f32(const char *path, const float *vals, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 0; }
    size_t got = fwrite(vals, sizeof(float), n, f);
    fclose(f);
    return got == n;
}

/* Runs `nbmetrics --width 4 --height 4 <cand> <ref>` and parses the 10
 * whitespace-separated output fields into out[10] as doubles (worst_is_peak,
 * field index 5, is parsed separately into is_peak). Returns 0 on failure
 * (nonzero exit, short read, unparsable field). */
static int run_nbmetrics(const char *exe, const char *cand, const char *ref,
                          double out[10], char is_peak[16]) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s --width 4 --height 4 %s %s", exe, cand, ref);
    FILE *p = popen(cmd, "r");
    if (!p) { perror("popen"); return 0; }

    char line[1024];
    if (!fgets(line, sizeof(line), p)) { pclose(p); fprintf(stderr, "no output from: %s\n", cmd); return 0; }
    int status = pclose(p);
    if (status != 0) { fprintf(stderr, "nonzero exit (%d) from: %s\n", status, cmd); return 0; }

    /* corr sum_ratio max_rel worst_pixel_frac peak_max_rel worst_is_peak fp16_corr fp16_sum_ratio fp16_diff_count fp16_excluded_count */
    int matched = sscanf(line, "%lf %lf %lf %lf %lf %15s %lf %lf %lf %lf",
                          &out[0], &out[1], &out[2], &out[3], &out[4], is_peak,
                          &out[6], &out[7], &out[8], &out[9]);
    if (matched != 10) { fprintf(stderr, "expected 10 fields, got %d: %s", matched, line); return 0; }
    return 1;
}

static int failures = 0;

static void check(int cond, const char *label) {
    if (cond) {
        printf("PASS: %s\n", label);
    } else {
        printf("FAIL: %s\n", label);
        failures++;
    }
}

int main(int argc, char **argv) {
    const char *exe = (argc > 1) ? argv[1] : "build/nbmetrics";
    const char *cand_path = "/tmp/nbmetrics_test_cand.bin";
    const char *ref_path = "/tmp/nbmetrics_test_ref.bin";
    double out[10];
    char is_peak[16];

    /* ---- (a) identical images ---- */
    {
        float ref[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        float cand[16]; memcpy(cand, ref, sizeof(ref));
        if (!write_f32(cand_path, cand, 16) || !write_f32(ref_path, ref, 16)) return 2;
        if (!run_nbmetrics(exe, cand_path, ref_path, out, is_peak)) { check(0, "(a) identical: ran"); }
        else {
            check(fabs(out[0] - 1.0) < 1e-6, "(a) identical: corr == 1");
            check(fabs(out[1] - 1.0) < 1e-6, "(a) identical: sum_ratio == 1");
            check(out[8] == 0.0, "(a) identical: fp16_diff_count == 0");
            check(out[9] == 0.0, "(a) identical: fp16_excluded_count == 0");
            check(isfinite(out[6]) && fabs(out[6] - 1.0) < 1e-3, "(a) identical: fp16_corr == 1");
        }
    }

    /* ---- (b) known additive delta (+0.1 on every pixel) ---- */
    {
        float ref[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        float cand[16];
        for (int i = 0; i < 16; i++) cand[i] = ref[i] + 0.1f;
        if (!write_f32(cand_path, cand, 16) || !write_f32(ref_path, ref, 16)) return 2;
        double sum_ref = 136.0, sum_cand = 136.0 + 16 * 0.1;
        double expect_ratio = sum_cand / sum_ref; /* ~1.011765 */
        if (!run_nbmetrics(exe, cand_path, ref_path, out, is_peak)) { check(0, "(b) delta: ran"); }
        else {
            check(out[0] > 0.999, "(b) delta: corr in expected range (>0.999)");
            check(fabs(out[1] - expect_ratio) < 1e-4, "(b) delta: sum_ratio matches expected != 1");
            check(fabs(out[1] - 1.0) > 1e-3, "(b) delta: sum_ratio != 1");
        }
    }

    /* ---- (c) overflow: pixel > 65504, an Inf, and a NaN in candidate ---- */
    {
        float ref[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        float cand[16]; memcpy(cand, ref, sizeof(ref));
        cand[0] = 100000.0f;                 /* > FP16 max (65504) */
        cand[1] = (float)INFINITY;
        cand[2] = (float)NAN;
        /* cand[3..15] left identical to ref -> those 13 pixels should be
         * the only ones counted in the FP16 pass. */
        if (!write_f32(cand_path, cand, 16) || !write_f32(ref_path, ref, 16)) return 2;
        if (!run_nbmetrics(exe, cand_path, ref_path, out, is_peak)) { check(0, "(c) overflow: ran (FP32 still computed)"); }
        else {
            check(1, "(c) overflow: ran (FP32 still computed, line parsed)");
            check(isfinite(out[6]), "(c) overflow: fp16_corr is finite (no NaN contamination)");
            check(isfinite(out[7]), "(c) overflow: fp16_sum_ratio is finite (no NaN contamination)");
            check(out[9] == 3.0, "(c) overflow: fp16_excluded_count == 3");
            check(out[8] == 0.0, "(c) overflow: fp16_diff_count == 0 (remaining 13 pixels identical)");
        }
    }

    remove(cand_path);
    remove(ref_path);

    if (failures == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURE(S)\n", failures);
    return 1;
}
