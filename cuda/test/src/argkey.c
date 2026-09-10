/*
 * argkey.c -- canonicalize a command-line argument list for stable hashing.
 *
 * See argkey.h for the public contract and the flag/value/bundle model.
 * Extracted from the standalone canon.c: the canonicalization is byte-for-byte
 * identical, but the entry point returns a heap-allocated string (no trailing
 * newline) for the md5 consumer instead of writing to stdout.
 */
#include "argkey.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

char *argkey_canonicalize(char **tok, int n) {
    if (n < 0) n = 0;

    bundle_t *b = malloc((n > 0 ? n : 1) * sizeof *b);
    if (!b) return NULL;
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

    /* Size the output: sum of token lengths + one space between tokens. */
    size_t total = 0;
    int count = 0;
    for (int k = 0; k < nb; k++)
        for (int j = 0; j < b[k].len; j++) {
            total += strlen(tok[b[k].start + j]);
            count++;
        }
    if (count > 0) total += (size_t)(count - 1);   /* inter-token spaces */
    total += 1;                                     /* NUL */

    char *out = malloc(total);
    if (!out) { free(b); return NULL; }

    char *p = out;
    int first = 1;
    for (int k = 0; k < nb; k++)
        for (int j = 0; j < b[k].len; j++) {
            const char *t = tok[b[k].start + j];
            if (!first) *p++ = ' ';
            size_t l = strlen(t);
            memcpy(p, t, l);
            p += l;
            first = 0;
        }
    *p = '\0';

    free(b);
    return out;
}
