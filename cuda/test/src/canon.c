/*
 * canon.c -- canonicalize a command-line argument list for stable hashing.
 *
 * Reads the argument list either from this program's own argv (everything
 * after argv[0]) or, when there is no argv payload and stdin is not a
 * terminal, from stdin (split on whitespace runs). Either way it prints a
 * canonical re-ordering to stdout: each flag is kept together with the value
 * tokens that follow it (a "bundle"), and the bundles are sorted by flag
 * name. Values inside a bundle keep their original order, so distinct
 * configs never collapse -- "-cell 74 82 ..." stays put and never merges with
 * "-cell 82 74 ...".
 *
 *   canon -lambda 1.0 -cell 74 74 34 90 90 90 -nonoise
 *   -> -cell 74 74 34 90 90 90 -lambda 1.0 -nonoise
 *
 *   printf '%s' '-lambda 1.0 -cell 74 74 34 90 90 90 -nonoise' | canon
 *   -> -cell 74 74 34 90 90 90 -lambda 1.0 -nonoise
 *
 * No per-flag arity table is needed. A token is a FLAG iff it starts with '-'
 * followed by a letter (so "-cell" and "--misset" are flags; "-10", "-.5",
 * "-1e3" are negative-number VALUES and a lone "-" is a value). A bundle is a
 * flag plus every following non-flag token up to the next flag.
 *
 * The sort is stable: duplicate flags keep their original relative order, so
 * "-foo 1 -foo 2" and "-foo 2 -foo 1" stay distinct. Any tokens before the first
 * flag form one leading bundle that sorts first.
 *
 * Feed the output to md5 to key the reference-image cache without the key
 * depending on the order the flags were written in:  canon <args...> | md5sum
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

/* A token is a flag iff it is '-' (or leading dashes) then a letter. */
static int is_flag(const char *t) {
    if (t[0] != '-') return 0;
    const char *p = t + 1;
    while (*p == '-') p++;                 /* allow long options: --misset */
    return isalpha((unsigned char)*p);
}

/* A flag and its trailing values as the slice [start, start+len) of tok. key
 * is the flag token ("" for a leading value run). idx keeps the sort stable. */
typedef struct { const char *key; int start; int len; int idx; } bundle_t;

static int cmp_bundle(const void *a, const void *b) {
    const bundle_t *x = a, *y = b;
    int c = strcmp(x->key, y->key);
    if (c) return c;
    return (x->idx > y->idx) - (x->idx < y->idx);   /* tie-break by original order */
}

/* Bundles tok[0..n) by flag, stable-sorts the bundles by flag name, and
 * prints the canonical form: space-joined tokens, one trailing newline,
 * nothing for empty input. */
static void canonicalize(char **tok, int n) {
    bundle_t *b = malloc((n > 0 ? n : 1) * sizeof *b);
    int nb = 0, i = 0;

    if (i < n && !is_flag(tok[i])) {         /* leading non-flag run (rare) */
        int s = i;
        while (i < n && !is_flag(tok[i])) i++;
        b[nb].key = ""; b[nb].start = s; b[nb].len = i - s; b[nb].idx = nb; nb++;
    }
    while (i < n) {                          /* one bundle per flag */
        int s = i++;
        while (i < n && !is_flag(tok[i])) i++;
        b[nb].key = tok[s]; b[nb].start = s; b[nb].len = i - s; b[nb].idx = nb; nb++;
    }

    qsort(b, nb, sizeof *b, cmp_bundle);

    int first = 1;
    for (int k = 0; k < nb; k++)
        for (int j = 0; j < b[k].len; j++) {
            if (!first) putchar(' ');
            fputs(tok[b[k].start + j], stdout);
            first = 0;
        }
    if (nb) putchar('\n');

    free(b);
}

/* space/tab/newline only -- not the full isspace() set */
static int is_ws(char c) { return c == ' ' || c == '\t' || c == '\n'; }

/* Reads all of stdin into *out_buf and splits it on whitespace runs into
 * *out_tok (pointers into *out_buf; empty runs are skipped). Returns the
 * token count. Caller owns and frees both *out_tok and *out_buf. */
static int read_stdin_tokens(char ***out_tok, char **out_buf) {
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    size_t got;
    while ((got = fread(buf + len, 1, cap - len, stdin)) > 0) {
        len += got;
        if (len == cap) {
            cap *= 2;
            buf = realloc(buf, cap);
        }
    }
    buf[len] = '\0';

    int cnt = 0, cap_tok = 64;
    char **tok = malloc(cap_tok * sizeof *tok);

    size_t i = 0;
    while (i < len) {
        while (i < len && is_ws(buf[i])) i++;
        if (i >= len) break;
        size_t s = i;
        while (i < len && !is_ws(buf[i])) i++;
        if (i < len) { buf[i] = '\0'; i++; }
        if (cnt == cap_tok) { cap_tok *= 2; tok = realloc(tok, cap_tok * sizeof *tok); }
        tok[cnt++] = buf + s;
    }

    *out_tok = tok;
    *out_buf = buf;
    return cnt;
}

int main(int argc, char **argv) {
    int n = argc - 1;                       /* payload = argv[1..argc-1] */
    char **tok = argv + 1;

    if (n == 0 && !isatty(0)) {              /* no argv payload, stdin piped */
        char *buf, **stok;
        int sn = read_stdin_tokens(&stok, &buf);
        canonicalize(stok, sn);
        free(stok);
        free(buf);
        return 0;
    }

    canonicalize(tok, n);
    return 0;
}
