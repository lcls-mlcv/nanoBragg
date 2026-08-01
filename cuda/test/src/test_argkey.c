/*
 * test_argkey.c -- unit tests for the argkey canonicalizer.
 *
 * Proves: (a) order-independence, (b) bundle stable-sort by flag name,
 * (c) values keep intra-bundle order, (d) the "-4stol" digit-leading caveat,
 * (e) byte-for-byte agreement with the original canon.c (driven live via a
 * reference binary whose path is argv[1], plus a hardcoded expected string).
 *
 * Usage: test_argkey [path-to-canon-reference-binary]
 * The (e-live) sub-check is skipped (not failed) when no path is given.
 */
#define _POSIX_C_SOURCE 200809L   /* popen/pclose under -std=c11 */

#include "argkey.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(const char *name, int cond) {
    printf("%-52s %s\n", name, cond ? "PASS" : "FAIL");
    if (!cond) failures++;
}

/* Canonicalize an argv-style array and compare to expected. */
static int canon_eq(char **tok, int n, const char *expected) {
    char *got = argkey_canonicalize(tok, n);
    int ok = got && strcmp(got, expected) == 0;
    if (!ok) fprintf(stderr, "  got:      \"%s\"\n  expected: \"%s\"\n",
                     got ? got : "(null)", expected);
    free(got);
    return ok;
}

/* Two token arrays canonicalize to the same string. */
static int canon_same(char **a, int na, char **b, int nb) {
    char *ca = argkey_canonicalize(a, na);
    char *cb = argkey_canonicalize(b, nb);
    int ok = ca && cb && strcmp(ca, cb) == 0;
    if (!ok) fprintf(stderr, "  a: \"%s\"\n  b: \"%s\"\n",
                     ca ? ca : "(null)", cb ? cb : "(null)");
    free(ca); free(cb);
    return ok;
}

/* Space-join tokens into a caller-freed string (for building a shell line). */
static char *join(char **tok, int n) {
    size_t total = 1;
    for (int i = 0; i < n; i++) total += strlen(tok[i]) + 1;
    char *s = malloc(total);
    s[0] = '\0';
    for (int i = 0; i < n; i++) {
        if (i) strcat(s, " ");
        strcat(s, tok[i]);
    }
    return s;
}

/* Run "<canon_ref> <tokens...>", capture stdout, strip one trailing newline. */
static char *run_canon_ref(const char *canon_ref, char **tok, int n) {
    char *args = join(tok, n);
    size_t clen = strlen(canon_ref) + 1 + strlen(args) + 1;
    char *cmd = malloc(clen);
    snprintf(cmd, clen, "%s %s", canon_ref, args);

    FILE *fp = popen(cmd, "r");
    free(args); free(cmd);
    if (!fp) return NULL;

    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    size_t got;
    while ((got = fread(buf + len, 1, cap - len, fp)) > 0) {
        len += got;
        if (len == cap) { cap *= 2; buf = realloc(buf, cap); }
    }
    pclose(fp);
    if (len && buf[len - 1] == '\n') len--;   /* drop canon's trailing newline */
    buf[len] = '\0';
    return buf;
}

int main(int argc, char **argv) {
    const char *canon_ref = (argc > 1) ? argv[1] : NULL;

    /* (a) Order-independence: permuting whole bundles canonicalizes equal. */
    {
        char *p1[] = { "-hkl", "P", "-cell", "74", "74", "34", "90", "90", "90",
                       "-Na", "100", "-lambda", "1.0", "-nonoise" };
        char *p2[] = { "-nonoise", "-lambda", "1.0", "-Na", "100",
                       "-cell", "74", "74", "34", "90", "90", "90", "-hkl", "P" };
        check("(a) order-independence",
              canon_same(p1, (int)(sizeof p1 / sizeof *p1),
                         p2, (int)(sizeof p2 / sizeof *p2)));
    }

    /* (b) Bundles stable-sort by flag name (uppercase before lowercase). */
    {
        char *t[] = { "-nonoise", "-hkl", "P", "-cell", "74", "82", "34",
                      "-Nc", "100", "-Na", "100", "-Nb", "100" };
        check("(b) stable-sort by flag name",
              canon_eq(t, (int)(sizeof t / sizeof *t),
                       "-Na 100 -Nb 100 -Nc 100 -cell 74 82 34 -hkl P -nonoise"));
    }

    /* (b2) Duplicate flags keep original relative order (stability). */
    {
        char *fwd[] = { "-foo", "1", "-foo", "2" };
        char *rev[] = { "-foo", "2", "-foo", "1" };
        int distinct = !canon_same(fwd, 4, rev, 4);
        check("(b2) duplicate flags stay distinct (stable)", distinct);
        check("(b2) forward dup order preserved",
              canon_eq(fwd, 4, "-foo 1 -foo 2"));
    }

    /* (c) Values keep intra-bundle order. */
    {
        char *t[] = { "-cell", "74", "82", "34", "90", "90", "90" };
        int keep = canon_eq(t, 7, "-cell 74 82 34 90 90 90");
        char *swap[] = { "-cell", "82", "74", "34", "90", "90", "90" };
        int differ = !canon_same(t, 7, swap, 7);
        check("(c) values keep intra-bundle order", keep && differ);
    }

    /* (d) "-4stol" (digit-leading) is treated as a VALUE, not a flag: it does
     * not start its own bundle and joins the leading value run (key ""). */
    {
        char *t[] = { "-4stol", "0.5", "-cell", "74", "-lambda", "1.0" };
        check("(d) -4stol digit-leading treated as value",
              canon_eq(t, 6, "-4stol 0.5 -cell 74 -lambda 1.0"));
    }

    /* (d2) Negative-number and lone-dash tokens are values too. */
    {
        char *t[] = { "-misset", "-10", "-.5", "-" };
        check("(d2) -10 / -.5 / lone - are values",
              canon_eq(t, 4, "-misset -10 -.5 -"));
    }

    /* (e) A representative nanoBragg line: hardcoded expected + live vs canon. */
    {
        char *line[] = {
            "-hkl", "{input_root}/crystals/193L.hkl",
            "-cell", "74", "74", "34", "90", "90", "90",
            "-Na", "100", "-Nb", "100", "-Nc", "100",
            "-default_F", "100", "-lambda", "1.0", "-nonoise"
        };
        int n = (int)(sizeof line / sizeof *line);
        const char *expected =
            "-Na 100 -Nb 100 -Nc 100 -cell 74 74 34 90 90 90 "
            "-default_F 100 -hkl {input_root}/crystals/193L.hkl -lambda 1.0 -nonoise";
        check("(e) representative line == expected", canon_eq(line, n, expected));

        if (canon_ref) {
            char *ref = run_canon_ref(canon_ref, line, n);
            char *got = argkey_canonicalize(line, n);
            int ok = ref && got && strcmp(ref, got) == 0;
            if (!ok) fprintf(stderr, "  canon_ref: \"%s\"\n  argkey:    \"%s\"\n",
                             ref ? ref : "(null)", got ? got : "(null)");
            check("(e-live) argkey == canon.c (stdout, newline-stripped)", ok);
            free(ref); free(got);
        } else {
            printf("%-52s SKIP (no canon_ref path)\n",
                   "(e-live) argkey == canon.c");
        }
    }

    printf("\n%s (%d failure%s)\n",
           failures ? "TESTS FAILED" : "ALL TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
