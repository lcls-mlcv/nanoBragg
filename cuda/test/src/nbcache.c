/* nbcache.c -- image-cache maintenance tool for the nanoBragg parity harness.
 *
 * The cache stores rendered REFERENCE images, machine-global and shared across
 * worktrees (NBTOOLS-SPEC §9). This tool is a thin CLI over cache_core:
 *
 *   nbcache --status                    report size / entry count vs budget
 *   nbcache --gc                        evict orphans + over-budget entries
 *   nbcache --path -- <exe> <params...> print the cache path for a reference render
 *   nbcache --set-budget-gb N           write the budget to settings.json
 *   nbcache --build-commit              print the git HEAD this was built at
 *
 *   options: --cache-dir DIR  --budget-gb N  --input-root ABS  --suites-dir DIR
 *
 * --status / --gc need the live set (which args_hashes current cases reference):
 * it is read from the baked args_hash of every case in --suites-dir via
 * case_core, never recomputed -- the same hash nbgensuite baked and nbrunsuite
 * looks up (the §9 invariant). --path recomputes the key from a reference command
 * line handed in {input_root}-token form (or absolute form + --input-root to
 * re-tokenize), proving bake and lookup agree.
 *
 * gc is correctness-neutral: an evicted image simply re-renders on next use, so a
 * wrong eviction costs time, never a wrong verdict.
 */
#define _XOPEN_SOURCE 700
#include "cache_core.h"
#include "case_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <getopt.h>

#ifndef NB_BUILD_COMMIT
#define NB_BUILD_COMMIT "unknown"
#endif

#define DEFAULT_SUITES_DIR "cuda/test/suites"

static void die(const char *m) { fprintf(stderr, "nbcache: %s\n", m); exit(2); }

/* ---- live set ----------------------------------------------------------- */
/* Collect the baked args_hash (+ K) of every case across all suites/<name>.jsonl
 * in suites_dir. Strings are strdup'd so the caller owns them independent of the
 * case_core DOM. Returns count; *hashes / *ks receive malloc'd parallel arrays. */
static int build_live_set(const char *suites_dir, char ***hashes, long **ks) {
    DIR *d = opendir(suites_dir);
    if (!d) { *hashes = NULL; *ks = NULL; return 0; }

    int cap = 512, n = 0;
    char **h = malloc((size_t)cap * sizeof *h);
    long *k = malloc((size_t)cap * sizeof *k);
    struct dirent *e;
    char path[4096];
    while ((e = readdir(d))) {
        size_t nl = strlen(e->d_name);
        if (nl < 6 || strcmp(e->d_name + nl - 6, ".jsonl")) continue;
        snprintf(path, sizeof path, "%s/%s", suites_dir, e->d_name);
        cc_suite *s = cc_load_suite(path, NULL);   /* base only feeds gate defaults */
        if (!s) continue;
        for (int i = 0; i < s->n; i++) {
            const char *ah = s->cases[i].args_hash;
            if (!ah || strlen(ah) != CACHE_HEXLEN) continue;
            if (n == cap) { cap *= 2; h = realloc(h, (size_t)cap * sizeof *h);
                            k = realloc(k, (size_t)cap * sizeof *k); }
            h[n] = strdup(ah);
            k[n] = s->cases[i].K;
            n++;
        }
        cc_suite_free(s);
    }
    closedir(d);
    *hashes = h; *ks = k;
    return n;
}

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

static int do_status(const char *cache_dir, long budget_gb,
                     const char *suites_dir) {
    char **live; long *live_k;
    int nlive = build_live_set(suites_dir, &live, &live_k);

    int n;
    cache_entry *e = cache_scan(cache_dir, (const char *const *)live, live_k, nlive, &n);
    if (!e) {
        printf("# cache: %s -- no cache yet; budget %ld GB\n", cache_dir, budget_gb);
        goto done;
    }
    cache_status st;
    cache_compute_status(e, n, budget_gb, &st);
    printf("# cache: %s\n", cache_dir);
    printf("# entries: %d (%.3f GB) / budget %ld GB%s\n",
           st.n_entries, (double)st.total_bytes / (double)(1LL << 30), budget_gb,
           st.total_bytes > st.budget_bytes ? "  [OVER BUDGET]" : "");
    printf("# live: %d  orphan: %d\n", st.n_entries - st.n_orphans, st.n_orphans);

    int oe, be;
    cache_gc(cache_dir, e, n, (long long)budget_gb << 30, time(NULL), 0, &oe, &be);
    printf("# would evict: %d (orphans=%d + over-budget=%d); dry run -- nbcache --gc to apply\n",
           oe + be, oe, be);
    free(e);
done:
    for (int i = 0; i < nlive; i++) free(live[i]);
    free(live); free(live_k);
    return 0;
}

static int do_gc(const char *cache_dir, long budget_gb, const char *suites_dir) {
    char **live; long *live_k;
    int nlive = build_live_set(suites_dir, &live, &live_k);

    /* Safety guard: an empty live set means the suites failed to parse; refuse to
     * delete (every entry would look like an orphan). */
    if (nlive == 0) {
        fprintf(stderr, "# cache: ABORT -- live set empty (suites parse failure "
                        "in %s?); cache untouched\n", suites_dir);
        free(live); free(live_k);
        return 0;
    }

    int n;
    cache_entry *e = cache_scan(cache_dir, (const char *const *)live, live_k, nlive, &n);
    if (!e) {
        printf("# cache: %s -- no cache yet; nothing to gc\n", cache_dir);
        for (int i = 0; i < nlive; i++) free(live[i]);
        free(live); free(live_k);
        return 0;
    }

    int oe, be;
    int total = cache_gc(cache_dir, e, n, (long long)budget_gb << 30, time(NULL), 1, &oe, &be);
    printf("# cache: %s -- scanned %d, evicted %d (orphans=%d, over-budget=%d); budget %ld GB\n",
           cache_dir, n, total, oe, be, budget_gb);

    free(e);
    for (int i = 0; i < nlive; i++) free(live[i]);
    free(live); free(live_k);
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
        "  options: --cache-dir DIR  --budget-gb N  --input-root ABS  --suites-dir DIR\n");
}

int main(int argc, char **argv) {
    enum { M_NONE, M_STATUS, M_GC, M_PATH, M_SETBUDGET, M_BUILDCOMMIT };
    int mode = M_NONE;
    long set_budget = 0, flag_budget = -1;
    const char *cache_dir_flag = NULL, *input_root = NULL;
    const char *suites_dir = DEFAULT_SUITES_DIR;

    enum { OPT_STATUS = 1000, OPT_GC, OPT_PATH, OPT_SETBUDGET, OPT_BUILDCOMMIT,
           OPT_CACHEDIR, OPT_BUDGET, OPT_INPUTROOT, OPT_SUITES, OPT_HELP };
    static struct option lo[] = {
        {"status",        no_argument,       0, OPT_STATUS},
        {"gc",            no_argument,       0, OPT_GC},
        {"path",          no_argument,       0, OPT_PATH},
        {"set-budget-gb", required_argument, 0, OPT_SETBUDGET},
        {"build-commit",  no_argument,       0, OPT_BUILDCOMMIT},
        {"cache-dir",     required_argument, 0, OPT_CACHEDIR},
        {"budget-gb",     required_argument, 0, OPT_BUDGET},
        {"input-root",    required_argument, 0, OPT_INPUTROOT},
        {"suites-dir",    required_argument, 0, OPT_SUITES},
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
            case OPT_SUITES:       suites_dir = optarg; break;
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
        if (mode == M_STATUS) rc = do_status(cache_dir, budget_gb, suites_dir);
        else                  rc = do_gc(cache_dir, budget_gb, suites_dir);
    }

    free(cache_dir);
    return rc;
}
