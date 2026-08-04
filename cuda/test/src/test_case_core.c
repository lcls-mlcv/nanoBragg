/* test_case_core.c -- unit test for case_core (own main).
 *
 * Run from the worktree root (paths are repo-relative). Parses the real
 * spec/base.json and each suites/<name>.jsonl, asserting field values, gate types,
 * case counts, and {input_root} token expansion. Exits nonzero on any failure.
 */
#define _XOPEN_SOURCE 700   /* realpath(path, NULL) under -std=c11 */
#include "case_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (cond) { printf("  ok  : " __VA_ARGS__); printf("\n"); } \
    else { printf("  FAIL: " __VA_ARGS__); printf("\n"); failures++; } \
} while (0)

static int deq(double a, double b) { return fabs(a - b) < 1e-12; }

/* find a case by id in a suite */
static const cc_case *find_case(const cc_suite *s, const char *id) {
    for (int i = 0; i < s->n; i++)
        if (s->cases[i].id && strcmp(s->cases[i].id, id) == 0) return &s->cases[i];
    return NULL;
}

static int load_and_count(const char *path, const cc_base *base, const char *label, int want) {
    cc_suite *s = cc_load_suite(path, base);
    int got = s ? s->n : -1;
    CHECK(got == want, "%s count == %d (got %d)", label, want, got);
    if (s) cc_suite_free(s);
    return got;
}

int main(void) {
    printf("== base.json ==\n");
    cc_base *b = cc_load_base("cuda/test/spec/base.json");
    if (!b) { printf("  FAIL: cc_load_base returned NULL\n"); return 1; }

    CHECK(b->input_root && strcmp(b->input_root, "inputs") == 0,
          "input_root == \"inputs\" (got \"%s\")", b->input_root ? b->input_root : "(null)");
    CHECK(deq(b->corr_min, 0.9999), "corr_min == 0.9999 (got %.6f)", b->corr_min);
    CHECK(deq(b->sum_ratio_min, 0.999), "sum_ratio_min == 0.999 (got %.6f)", b->sum_ratio_min);
    CHECK(deq(b->sum_ratio_max, 1.001), "sum_ratio_max == 1.001 (got %.6f)", b->sum_ratio_max);

    CHECK(b->n_reference_fix_branches == 3, "reference_fix_branches count == 3 (got %d)",
          b->n_reference_fix_branches);
    if (b->n_reference_fix_branches == 3) {
        CHECK(strcmp(b->reference_fix_branches[0], "fix/phi0-stale-rotation") == 0,
              "reference_fix_branches[0] == fix/phi0-stale-rotation");
        CHECK(strcmp(b->reference_fix_branches[1], "fix/subpixel-oversampling") == 0,
              "reference_fix_branches[1] == fix/subpixel-oversampling");
        CHECK(strcmp(b->reference_fix_branches[2], "fix/curved-det-flag-guard") == 0,
              "reference_fix_branches[2] == fix/curved-det-flag-guard");
    }
    CHECK(b->n_base_flags == 3, "base_flags count == 3 (got %d)", b->n_base_flags);
    CHECK(b->n_base_geometry_dims == 3, "base_geometry_dims count == 3 (got %d)",
          b->n_base_geometry_dims);

    printf("== suite counts (asserted against wc -l) ==\n");
    load_and_count("cuda/test/suites/grid320.jsonl",  b, "grid320",  320);
    load_and_count("cuda/test/suites/coverage.jsonl", b, "coverage",  74);
    load_and_count("cuda/test/suites/guards.jsonl",   b, "guards",     8);
    load_and_count("cuda/test/suites/pairwise.jsonl", b, "pairwise",  12);
    load_and_count("cuda/test/suites/perf.jsonl",     b, "perf",       7);

    printf("== case spot-checks ==\n");
    cc_suite *guards = cc_load_suite("cuda/test/suites/guards.jsonl", b);
    const cc_case *gi = find_case(guards, "guard_interpolate");
    CHECK(gi != NULL, "guards has case guard_interpolate");
    if (gi) {
        CHECK(gi->gate_type == CC_GATE_REJECT, "guard_interpolate gate_type == reject");
        CHECK(gi->candidate_args && strstr(gi->candidate_args, "-interpolate"),
              "guard_interpolate candidate_args carries -interpolate");
        CHECK(gi->reference_args != NULL, "guard_interpolate reference_args present");
    }

    cc_suite *grid = cc_load_suite("cuda/test/suites/grid320.jsonl", b);
    const cc_case *g0 = &grid->cases[0];
    CHECK(g0->gate_type == CC_GATE_ABSOLUTE, "grid320[0] gate_type == absolute (from suite header)");
    CHECK(g0->K >= 1, "grid320[0] K (cost.compute) >= 1 (got %ld)", g0->K);
    CHECK(deq(g0->corr_min, 0.9999), "grid320[0] effective corr_min == 0.9999 (got %.6f)", g0->corr_min);

    cc_suite *perf = cc_load_suite("cuda/test/suites/perf.jsonl", b);
    CHECK(perf->cases[0].gate_type == CC_GATE_PERF,
          "perf[0] gate_type == perf (from suite header)");

    printf("== args_hash (baked cache key) ==\n");
    /* every case carries a 32-hex md5 over its reference_args */
    int hash_ok = 1;
    for (int i = 0; i < grid->n; i++) {
        const char *h = grid->cases[i].args_hash;
        if (!h || strlen(h) != 32) { hash_ok = 0; break; }
        for (const char *p = h; *p; p++)
            if (!((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f'))) { hash_ok = 0; break; }
        if (!hash_ok) break;
    }
    CHECK(hash_ok, "grid320: every case args_hash is 32 lowercase-hex chars");
    if (gi) CHECK(gi->args_hash && strlen(gi->args_hash) == 32,
                  "guard_interpolate args_hash present (32 hex, got \"%s\")",
                  gi->args_hash ? gi->args_hash : "(null)");

    printf("== {input_root} token expansion ==\n");
    char *abs = cc_input_root_abs(b);
    CHECK(abs != NULL, "cc_input_root_abs resolved (got \"%s\")", abs ? abs : "(null)");
    if (abs && gi) {
        char *expanded = cc_resolve_input_root(gi->candidate_args, abs);
        CHECK(strstr(expanded, "{input_root}") == NULL, "no {input_root} token remains after expand");
        CHECK(strstr(expanded, abs) != NULL, "expanded args contain absolute inputs path");
        /* the abs path must end in /inputs and be absolute */
        CHECK(abs[0] == '/', "absolute path is rooted at /");
        CHECK(strstr(abs, "cuda/test/inputs") != NULL, "absolute path ends in cuda/test/inputs");
        free(expanded);
    }

    printf("== CWD independence (input_root anchors on base.json's own path, not CWD) ==\n");
    /* Resolve base.json to an absolute path *before* leaving the worktree root,
       then reload from a completely different working directory using that
       absolute path. cc_input_root_abs() must still land on the exact same
       .../cuda/test/inputs path -- proving the harness root comes from
       base.json's own location, not getcwd(). */
    char *base_abs_path = realpath("cuda/test/spec/base.json", NULL);
    CHECK(base_abs_path != NULL, "resolved absolute path to base.json (got \"%s\")",
          base_abs_path ? base_abs_path : "(null)");

    char orig_cwd[4096];
    CHECK(getcwd(orig_cwd, sizeof(orig_cwd)) != NULL, "captured original cwd");

    if (base_abs_path) {
        CHECK(chdir("/tmp") == 0, "chdir(/tmp) succeeded");

        cc_base *b2 = cc_load_base(base_abs_path);
        CHECK(b2 != NULL, "cc_load_base(absolute path) succeeded from /tmp");
        if (b2) {
            char *abs2 = cc_input_root_abs(b2);
            CHECK(abs2 != NULL, "cc_input_root_abs resolved from /tmp (got \"%s\")",
                  abs2 ? abs2 : "(null)");
            if (abs2 && abs)
                CHECK(strcmp(abs2, abs) == 0,
                      "input_root abs path from /tmp == input_root abs path from worktree root "
                      "(\"%s\" == \"%s\")", abs2, abs);
            free(abs2);
            cc_base_free(b2);
        }

        CHECK(chdir(orig_cwd) == 0, "restored original cwd");
    }
    free(base_abs_path);
    free(abs);

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "TEST PASSED", failures);
    return failures ? 1 : 0;
}
