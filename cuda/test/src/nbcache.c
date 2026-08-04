/* nbcache.c -- image-cache maintenance tool for the nanoBragg parity harness.
 *
 * The cache stores rendered REFERENCE images, machine-global and shared across
 * worktrees (NBTOOLS-SPEC §9). This tool is a thin CLI over cache_core:
 *
 *   nbcache --status                    report size / budget / eviction weights
 *   nbcache --gc                        trim the cache back to its budget
 *   nbcache --path -- <exe> <params...> print the cache path for a reference render
 *   nbcache --set-budget-gb N           write the budget to settings.json
 *   nbcache --build-commit              print the git HEAD this was built at
 *
 *   options: --cache-dir DIR  --budget-gb N  --input-root ABS
 *
 * --status reports what is in effect: entry count and footprint against the
 * resolved budget, the three eviction weights with where each came from, the
 * timestamp axis the ranking uses, and a dry-run victim count.
 *
 * --gc is pure score-to-budget: it evicts only while the cache exceeds the
 * budget, lowest score first. It has no run in flight, so nothing is protected
 * and every entry is a candidate; under budget it deletes nothing.
 *
 * --path recomputes the key from a reference command line handed in
 * {input_root}-token form (or absolute form + --input-root to re-tokenize),
 * proving bake and lookup agree -- the same hash nbgensuite baked and nbrunsuite
 * looks up (the §9 invariant).
 *
 * Eviction is correctness-neutral: an evicted image simply re-renders on next
 * use, so a wrong eviction costs time, never a wrong verdict.
 */
#define _XOPEN_SOURCE 700
#include "cache_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <getopt.h>

#ifndef NB_BUILD_COMMIT
#define NB_BUILD_COMMIT "unknown"
#endif

static void die(const char *m) { fprintf(stderr, "nbcache: %s\n", m); exit(2); }

/* ---- --path re-tokenize ------------------------------------------------- */
/* Replace every occurrence of `abs` in `tok` with the {input_root} token, so an
 * expanded (absolute) reference command line reproduces the token-form baked key.
 * Returns a malloc'd string (caller frees). */
static char *retokenize(const char *tok, const char *abs) {
    const char *repl = "{input_root}";
    size_t al = strlen(abs), rl = strlen(repl);
    size_t count = 0;
    for (const char *p = strstr(tok, abs); p && al; p = strstr(p + al, abs)) count++;
    char *out = malloc(strlen(tok) + count * (rl > al ? rl - al : 0) + 1);
    if (!out) die("OOM");
    char *w = out; const char *r = tok, *p;
    while (al && (p = strstr(r, abs))) {
        size_t lead = (size_t)(p - r);
        memcpy(w, r, lead); w += lead;
        memcpy(w, repl, rl); w += rl;
        r = p + al;
    }
    strcpy(w, r);
    return out;
}

/* ---- modes -------------------------------------------------------------- */

/* The three eviction weights, each tagged with where its effective value came
 * from (§9: compile default, overridable only by hand-editing settings.json).
 * The tag is decided by comparing against the compile default, so a hand-edited
 * value that equals the default reads as "compile default" -- the two are
 * indistinguishable on disk and identical in effect. Grace is reported in days,
 * the unit settings.json takes. */
static void print_weights(const cache_weights *w) {
    double grace_days = w->grace_seconds / 86400.0;
    printf("# evict_cost_weight: %g (%s)\n", w->cost_weight,
           w->cost_weight == CACHE_DEFAULT_COST_WEIGHT ? "compile default" : "settings.json");
    printf("# evict_decay: %g (%s)\n", w->decay,
           w->decay == CACHE_DEFAULT_DECAY ? "compile default" : "settings.json");
    printf("# evict_grace_days: %g (%s)\n", grace_days,
           grace_days == CACHE_DEFAULT_GRACE_DAYS ? "compile default" : "settings.json");
}

static int do_status(const char *cache_dir, long budget_gb) {
    cache_weights w;
    cache_weights_resolve(cache_dir, &w);

    int n, rank_by_mtime;
    cache_entry *e = cache_scan(cache_dir, &n, &rank_by_mtime);
    if (!e) {
        printf("# cache: %s -- no cache yet; budget %ld GB\n", cache_dir, budget_gb);
        print_weights(&w);
        return 0;
    }
    cache_status st;
    cache_compute_status(e, n, budget_gb, &st);
    printf("# cache: %s\n", cache_dir);
    printf("# entries: %d (%.3f GB) / budget %ld GB%s\n",
           st.n_entries, (double)st.total_bytes / (double)(1LL << 30), budget_gb,
           st.total_bytes > st.budget_bytes ? "  [OVER BUDGET]" : "");
    print_weights(&w);
    /* Which timestamp ranks victims: atime normally, mtime where the cache
     * filesystem is mounted noatime and atime can never advance. */
    printf("# ranking: %s\n", rank_by_mtime
           ? "mtime (age-since-written) -- cache filesystem is mounted noatime"
           : "atime (last use)");

    int would = cache_gc(cache_dir, e, n, (long long)budget_gb << 30, NULL, 0, &w,
                         (long)time(NULL), rank_by_mtime, 0);
    printf("# would evict: %d; dry run -- nbcache --gc to apply\n", would);
    free(e);
    return 0;
}

/* Score-to-budget only: no run is in flight, so the protect set is empty and
 * every entry is a candidate. Under budget this deletes nothing. */
static int do_gc(const char *cache_dir, long budget_gb) {
    int n, rank_by_mtime;
    cache_entry *e = cache_scan(cache_dir, &n, &rank_by_mtime);
    if (!e) {
        printf("# cache: %s -- no cache yet; nothing to gc\n", cache_dir);
        return 0;
    }

    cache_weights w;
    cache_weights_resolve(cache_dir, &w);
    int total = cache_gc(cache_dir, e, n, (long long)budget_gb << 30, NULL, 0, &w,
                         (long)time(NULL), rank_by_mtime, 1);
    printf("# cache: %s -- scanned %d, evicted %d; budget %ld GB\n",
           cache_dir, n, total, budget_gb);

    free(e);
    return 0;
}

static int do_path(const char *cache_dir, const char *input_root,
                   char **payload, int npayload) {
    if (npayload < 1) die("--path needs '-- <exe> [params...]'");
    const char *exe = payload[0];
    char **params = payload + 1;
    int nparams = npayload - 1;

    char ref_md5[33];
    if (cache_reference_md5(exe, ref_md5) != 0) {
        fprintf(stderr, "nbcache: cannot read reference binary '%s'\n", exe);
        return 2;
    }

    /* Re-tokenize absolute inputs back to {input_root} when asked, so an expanded
     * command line reproduces the token-form baked key. */
    char **tok = params;
    char **owned = NULL;
    if (input_root) {
        owned = malloc((size_t)nparams * sizeof *owned);
        for (int i = 0; i < nparams; i++) owned[i] = retokenize(params[i], input_root);
        tok = owned;
    }

    char args_hash[33];
    if (cache_args_hash(tok, nparams, args_hash) != 0) die("OOM (args_hash)");

    char path[4096];
    if (!cache_entry_path(cache_dir, ref_md5, args_hash, ".bin", path, sizeof path))
        die("path assembly failed");
    printf("%s\n", path);

    if (owned) { for (int i = 0; i < nparams; i++) free(owned[i]); free(owned); }
    return 0;
}

/* ---- main --------------------------------------------------------------- */

static void usage(void) {
    fprintf(stderr,
        "usage: nbcache <mode> [options]\n"
        "  modes:  --status | --gc | --path -- <exe> <params...>\n"
        "          --set-budget-gb N | --build-commit\n"
        "  options: --cache-dir DIR  --budget-gb N  --input-root ABS\n");
}

int main(int argc, char **argv) {
    enum { M_NONE, M_STATUS, M_GC, M_PATH, M_SETBUDGET, M_BUILDCOMMIT };
    int mode = M_NONE;
    long set_budget = 0, flag_budget = -1;
    const char *cache_dir_flag = NULL, *input_root = NULL;

    enum { OPT_STATUS = 1000, OPT_GC, OPT_PATH, OPT_SETBUDGET, OPT_BUILDCOMMIT,
           OPT_CACHEDIR, OPT_BUDGET, OPT_INPUTROOT, OPT_HELP };
    static struct option lo[] = {
        {"status",        no_argument,       0, OPT_STATUS},
        {"gc",            no_argument,       0, OPT_GC},
        {"path",          no_argument,       0, OPT_PATH},
        {"set-budget-gb", required_argument, 0, OPT_SETBUDGET},
        {"build-commit",  no_argument,       0, OPT_BUILDCOMMIT},
        {"cache-dir",     required_argument, 0, OPT_CACHEDIR},
        {"budget-gb",     required_argument, 0, OPT_BUDGET},
        {"input-root",    required_argument, 0, OPT_INPUTROOT},
        {"help",          no_argument,       0, OPT_HELP},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "", lo, NULL)) != -1) {
        switch (c) {
            case OPT_STATUS:       if (mode) die("one mode only"); mode = M_STATUS; break;
            case OPT_GC:           if (mode) die("one mode only"); mode = M_GC; break;
            case OPT_PATH:         if (mode) die("one mode only"); mode = M_PATH; break;
            case OPT_BUILDCOMMIT:  if (mode) die("one mode only"); mode = M_BUILDCOMMIT; break;
            case OPT_SETBUDGET:    if (mode) die("one mode only"); mode = M_SETBUDGET;
                                   set_budget = strtol(optarg, NULL, 10); break;
            case OPT_CACHEDIR:     cache_dir_flag = optarg; break;
            case OPT_BUDGET:       flag_budget = strtol(optarg, NULL, 10); break;
            case OPT_INPUTROOT:    input_root = optarg; break;
            case OPT_HELP:         usage(); return 0;
            default:               usage(); return 2;
        }
    }

    if (mode == M_NONE) { usage(); return 2; }
    if (mode == M_BUILDCOMMIT) { puts(NB_BUILD_COMMIT); return 0; }

    char *cache_dir = cache_dir_resolve(cache_dir_flag);
    if (!cache_dir) die("cannot resolve cache dir (no --cache-dir / XDG_CACHE_HOME / HOME)");

    int rc = 0;
    if (mode == M_SETBUDGET) {
        if (cache_set_budget_gb(cache_dir, set_budget) != 0) die("cannot write settings.json");
        printf("# cache: %s -- budget_gb set to %ld\n", cache_dir, set_budget);
    } else if (mode == M_PATH) {
        rc = do_path(cache_dir, input_root, argv + optind, argc - optind);
    } else {
        long budget_gb = cache_budget_resolve(cache_dir, flag_budget);
        if (mode == M_STATUS) rc = do_status(cache_dir, budget_gb);
        else                  rc = do_gc(cache_dir, budget_gb);
    }

    free(cache_dir);
    return rc;
}
