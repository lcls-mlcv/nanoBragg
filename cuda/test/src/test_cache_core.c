/* test_cache_core.c -- unit test for cache_core + the nbcache --path key.
 *
 * Synthetic only: every cache operation runs against a fresh mkdtemp() temp dir,
 * never the real ~/.cache/nanobragg, and no nanoBragg render is invoked. Covers
 * the NBTOOLS-SPEC §9 invariants: path/key agreement (baked args_hash ==
 * nbcache --path), the eviction score and its budget/protect contract, atomic
 * writes, the .meta cost.actual record, and config resolution.
 *
 *   Usage: test_cache_core [nbcache_binary]
 * If nbcache_binary is given, the key-agreement check also runs the tool via
 * popen and the `--gc` trim is exercised through the real binary; otherwise the
 * library entry points (cache_args_hash, cache_gc) are verified alone and the two
 * CLI-level checks report themselves skipped.
 * Run from the worktree root (spec/suites paths are repo-relative).
 */
#define _XOPEN_SOURCE 700
#include "cache_core.h"
#include "case_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (cond) { printf("  ok  : " __VA_ARGS__); printf("\n"); } \
    else { printf("  FAIL: " __VA_ARGS__); printf("\n"); failures++; } \
} while (0)

/* Synthetic clock: entry timestamps are expressed as NOW - <n> * DAY, so the
 * scores under test are independent of the real time of day. */
#define NOW 2000000000L
#define DAY 86400L

/* ---- small filesystem/util helpers ------------------------------------- */

static int file_exists(const char *p) { struct stat s; return stat(p, &s) == 0; }

static int dir_exists(const char *p) { struct stat s; return stat(p, &s) == 0 && S_ISDIR(s.st_mode); }

static int deq(double a, double b) { return fabs(a - b) < 1e-9; }

/* Split a command line on spaces into a malloc'd argv (tokens malloc'd). The
 * baked reference_args carry no embedded spaces, matching nbgensuite's token
 * list, so a plain space split reproduces it. */
static char **split_ws(const char *s, int *n) {
    char **tok = malloc(256 * sizeof *tok);
    int cnt = 0;
    const char *p = s;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        const char *start = p;
        while (*p && *p != ' ') p++;
        size_t len = (size_t)(p - start);
        tok[cnt] = malloc(len + 1);
        memcpy(tok[cnt], start, len);
        tok[cnt][len] = 0;
        cnt++;
    }
    *n = cnt;
    return tok;
}
static void free_tok(char **tok, int n) { for (int i = 0; i < n; i++) free(tok[i]); free(tok); }

/* Write a .bin of `nbytes` and stamp its atime/mtime. Uses the library's own
 * atomic writer and path assembly. */
static void make_bin(const char *cache_dir, const char *ref_md5,
                     const char *args_hash, long nbytes, long atime, long mtime) {
    char *buf = malloc((size_t)nbytes);
    memset(buf, 'x', (size_t)nbytes);
    char bin[4096];
    cache_entry_path(cache_dir, ref_md5, args_hash, ".bin", bin, sizeof bin);
    cache_write_atomic(bin, buf, (size_t)nbytes);
    free(buf);
    struct timeval tv[2] = { { atime, 0 }, { mtime, 0 } };
    utimes(bin, tv);
}

/* A full synthetic entry: the .meta recording cost.actual plus the stamped .bin. */
static void make_entry(const char *cache_dir, const char *ref_md5,
                       const char *args_hash, long nbytes, double cost_actual,
                       long atime, long mtime) {
    cache_write_meta(cache_dir, ref_md5, args_hash, cost_actual);
    make_bin(cache_dir, ref_md5, args_hash, nbytes, atime, mtime);
}

/* A full synthetic entry whose .bin is SPARSE: ftruncate gives the file the
 * apparent size the budget arithmetic reads (st_size) without writing a byte, so
 * a GB-scale cache -- the granularity `nbcache --budget-gb` works in -- costs no
 * disk. atime and mtime are stamped equal, so the ranking axis (atime vs the
 * noatime mtime fallback) cannot change which entry is the victim. */
static void make_sparse_entry(const char *cache_dir, const char *ref_md5,
                              const char *args_hash, long long nbytes,
                              double cost_actual, long stamp) {
    cache_write_meta(cache_dir, ref_md5, args_hash, cost_actual);   /* creates the shard dir */
    char bin[4096];
    cache_entry_path(cache_dir, ref_md5, args_hash, ".bin", bin, sizeof bin);
    int fd = open(bin, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) { if (ftruncate(fd, (off_t)nbytes)) {} close(fd); }
    struct timeval tv[2] = { { stamp, 0 }, { stamp, 0 } };
    utimes(bin, tv);
}

static void write_text(const char *path, const char *text) {
    cache_write_atomic(path, text, strlen(text));
}

static void write_settings(const char *cache_dir, const char *text) {
    char p[4096]; snprintf(p, sizeof p, "%s/settings.json", cache_dir);
    write_text(p, text);
}

/* File contents with the trailing newline trimmed, so a CHECK can print them
 * inline. Caller frees. */
static char *read_text(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    char *buf = malloc(8192);
    size_t got = fread(buf, 1, 8191, f);
    fclose(f);
    while (got && buf[got - 1] == '\n') got--;
    buf[got] = 0;
    return buf;
}

/* Recursively count files whose name contains ".tmp." under dir. */
static int count_tmp(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    int n = 0; struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char p[4096]; snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        struct stat s;
        if (stat(p, &s) == 0 && S_ISDIR(s.st_mode)) n += count_tmp(p);
        else if (strstr(e->d_name, ".tmp.")) n++;
    }
    closedir(d);
    return n;
}

/* ======================================================================== */
/* 1. PATH / KEY AGREEMENT -- the §9 invariant                              */
/* ======================================================================== */

/* library-level: md5(argkey(reference_args)) must reproduce the baked hash */
static void check_key_lib(const cc_case *c, const char *label) {
    int nt; char **tok = split_ws(c->reference_args, &nt);
    char got[33];
    cache_args_hash(tok, nt, got);
    CHECK(c->args_hash && strcmp(got, c->args_hash) == 0,
          "%s: cache_args_hash == baked (%s vs %s)", label, got,
          c->args_hash ? c->args_hash : "(null)");
    free_tok(tok, nt);
}

/* tool-level: nbcache --path must emit a path whose <args_hash> is the baked one */
static void check_key_cli(const char *nbcache_bin, const char *cache_dir,
                          const char *dummy_ref, const cc_case *c, const char *label) {
    char cmd[16384];
    snprintf(cmd, sizeof cmd, "%s --path --cache-dir %s -- %s %s",
             nbcache_bin, cache_dir, dummy_ref, c->reference_args);
    FILE *fp = popen(cmd, "r");
    if (!fp) { CHECK(0, "%s: popen nbcache --path failed", label); return; }
    char out[4096] = {0};
    if (!fgets(out, sizeof out, fp)) out[0] = 0;
    pclose(fp);
    out[strcspn(out, "\n")] = 0;
    /* extract the trailing <args_hash>.bin basename */
    char *slash = strrchr(out, '/');
    char *base = slash ? slash + 1 : out;
    char *dot = strstr(base, ".bin");
    if (dot) *dot = 0;
    CHECK(c->args_hash && strcmp(base, c->args_hash) == 0,
          "%s: nbcache --path args_hash == baked (%s vs %s)", label, base,
          c->args_hash ? c->args_hash : "(null)");
    /* the shard is the first two hex of the hash */
    if (slash) {
        char *sh = slash - 3;   /* ".../XX/<hash>" -> back up over "/XX" */
        CHECK(sh >= out && sh[0] == '/' && sh[1] == base[0] && sh[2] == base[1],
              "%s: shard == first two hex of args_hash", label);
    }
}

static void test_key_agreement(const char *nbcache_bin, const char *tmp_base) {
    printf("== 1. path/key agreement (baked vs cache_core / nbcache --path) ==\n");
    cc_base *b = cc_load_base("cuda/test/spec/base.json");
    if (!b) { CHECK(0, "cc_load_base"); return; }

    cc_suite *grid = cc_load_suite("cuda/test/suites/grid320.jsonl", b);
    cc_suite *guards = cc_load_suite("cuda/test/suites/guards.jsonl", b);
    if (!grid || !guards) { CHECK(0, "load grid320/guards"); return; }

    const cc_case *g = &grid->cases[0];               /* a grid320 case */
    const cc_case *gd = &guards->cases[0];            /* a guards case */
    check_key_lib(g,  "grid320[0]");
    check_key_lib(gd, "guards[0]");

    if (nbcache_bin) {
        char cdir[512], dummy[4096];
        snprintf(cdir, sizeof cdir, "%s/keycache", tmp_base);
        snprintf(dummy, sizeof dummy, "%s/dummy_ref", tmp_base);
        FILE *f = fopen(dummy, "w"); if (f) { fputs("ref\n", f); fclose(f); }
        check_key_cli(nbcache_bin, cdir, dummy, g,  "grid320[0]");
        check_key_cli(nbcache_bin, cdir, dummy, gd, "guards[0]");
    } else {
        printf("  note: nbcache binary not supplied -- CLI --path check skipped\n");
    }
    cc_suite_free(grid); cc_suite_free(guards); cc_base_free(b);
}

/* ======================================================================== */
/* 2. SCORE -- cost vs staleness, with the default exponents                */
/* ======================================================================== */

static void test_score(void) {
    printf("== 2. eviction score (cost^a / (stale+s0)^b) ==\n");
    cache_weights w;
    cache_weights_resolve("/nonexistent-cache-dir", &w);   /* compile defaults */

    /* cost dominates: 100 s unread for 200 days outranks 1 s read today, so a
     * pure last-use ranking would evict the wrong one. */
    double expensive_stale = cache_score(100.0, NOW - 200 * DAY, NOW, &w);
    double cheap_fresh     = cache_score(1.0,   NOW,             NOW, &w);
    CHECK(expensive_stale > cheap_fresh,
          "expensive+stale (100 s, 200 d) outranks cheap+fresh (1 s, 0 d): %.3e > %.3e",
          expensive_stale, cheap_fresh);

    /* staleness dominates once the gap is wide enough: 2x the cost does not save
     * an image unread for 10 years, so a pure cost ranking would evict the wrong
     * one. Both directions together pin the sign of each exponent -- that cost
     * raises the score and staleness lowers it -- not their magnitudes. */
    double pricier_ancient = cache_score(2.0, NOW - 3650 * DAY, NOW, &w);
    CHECK(cheap_fresh > pricier_ancient,
          "cheap+fresh (1 s, 0 d) outranks pricier+ancient (2 s, 3650 d): %.3e > %.3e",
          cheap_fresh, pricier_ancient);

    /* no .meta (cost 0) scores 0 -- the first victim at any age */
    CHECK(cache_score(0.0, NOW, NOW, &w) == 0.0, "cost.actual 0 scores 0");

    /* a future timestamp (clock skew) clamps to zero staleness, not a negative
     * one that would invert the rank */
    CHECK(deq(cache_score(4.0, NOW + 900 * DAY, NOW, &w),
              cache_score(4.0, NOW, NOW, &w)),
          "future atime clamps to stale=0");

    /* monotone in each axis at fixed other axis */
    CHECK(cache_score(9.0, NOW - DAY, NOW, &w) > cache_score(3.0, NOW - DAY, NOW, &w),
          "score rises with cost at equal staleness");
    CHECK(cache_score(9.0, NOW - DAY, NOW, &w) > cache_score(9.0, NOW - 900 * DAY, NOW, &w),
          "score falls with staleness at equal cost");
}

/* ======================================================================== */
/* 3. BUDGET EVICTION -- ascending score, .bin only                         */
/* ======================================================================== */

/* Four entries whose score order is NOT the cost order and NOT the age order:
 * h2 is 4x the cost of h1 but evicts first (far colder), and h4 is the coldest
 * of all yet survives (far most expensive). */
static const char *REF   = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
static const char *H1    = "e1000000000000000000000000000001";  /* 0.5 s,   1 d */
static const char *H2    = "e2000000000000000000000000000002";  /* 2.0 s, 400 d */
static const char *H3    = "e3000000000000000000000000000003";  /* 20 s,  100 d */
static const char *H4    = "e4000000000000000000000000000004";  /* 100 s, 800 d */

static void make_ladder(const char *cdir) {
    make_entry(cdir, REF, H1, 1000,   0.5, NOW -   1 * DAY, NOW - 1 * DAY);
    make_entry(cdir, REF, H2, 1000,   2.0, NOW - 400 * DAY, NOW - 1 * DAY);
    make_entry(cdir, REF, H3, 1000,  20.0, NOW - 100 * DAY, NOW - 1 * DAY);
    make_entry(cdir, REF, H4, 1000, 100.0, NOW - 800 * DAY, NOW - 1 * DAY);
}

static void test_budget_eviction(const char *tmp_base) {
    printf("== 3. budget eviction (ascending score, .bin only) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/evict", tmp_base);
    make_ladder(cdir);

    cache_weights w; cache_weights_resolve(cdir, &w);
    int n;
    cache_entry *e = cache_scan(cdir, &n, NULL);
    CHECK(n == 4, "scan found 4 entries (got %d)", n);

    /* 4x1000 on disk, budget 2500 -> the two lowest scores go */
    int evicted = cache_gc(cdir, e, n, 2500, NULL, 0, &w, NOW, 0, 1);
    CHECK(evicted == 2, "evicted exactly 2 (got %d)", evicted);

    char p[4096];
    cache_entry_path(cdir, REF, H2, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "h2 evicted (4x the cost of h1, but 400 d unread)");
    cache_entry_path(cdir, REF, H1, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "h1 evicted (cheapest, though read yesterday)");
    cache_entry_path(cdir, REF, H3, ".bin", p, sizeof p);
    CHECK(file_exists(p), "h3 survives");
    cache_entry_path(cdir, REF, H4, ".bin", p, sizeof p);
    CHECK(file_exists(p), "h4 survives (coldest of all, but the most expensive)");

    /* the measured render time outlives its image */
    cache_entry_path(cdir, REF, H1, ".meta", p, sizeof p);
    CHECK(file_exists(p), "evicted h1 .meta survives");
    cache_entry_path(cdir, REF, H2, ".meta", p, sizeof p);
    CHECK(file_exists(p), "evicted h2 .meta survives");
    free(e);
}

/* ======================================================================== */
/* 4. UNDER BUDGET / NEGATIVE BUDGET -- nothing is evicted                  */
/* ======================================================================== */

static void test_under_budget(const char *tmp_base) {
    printf("== 4. under budget and negative budget (no eviction) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/under", tmp_base);
    make_ladder(cdir);

    cache_weights w; cache_weights_resolve(cdir, &w);
    int n;
    cache_entry *e = cache_scan(cdir, &n, NULL);
    int evicted = cache_gc(cdir, e, n, 10000, NULL, 0, &w, NOW, 0, 1);
    CHECK(evicted == 0, "4000 bytes under a 10000 byte budget -> 0 evicted (got %d)", evicted);

    char p[4096];
    const char *all[] = { H1, H2, H3, H4 };
    for (int i = 0; i < 4; i++) {
        cache_entry_path(cdir, REF, all[i], ".bin", p, sizeof p);
        CHECK(file_exists(p), "under budget: %.4s.. kept despite its score", all[i]);
    }
    free(e);

    /* A negative budget disables eviction outright: the cache is over every
     * non-negative limit here, yet nothing goes. */
    char ndir[512]; snprintf(ndir, sizeof ndir, "%s/negbudget", tmp_base);
    make_ladder(ndir);
    e = cache_scan(ndir, &n, NULL);
    evicted = cache_gc(ndir, e, n, -1, NULL, 0, &w, NOW, 0, 1);
    CHECK(evicted == 0, "negative budget -> 0 evicted (got %d)", evicted);
    for (int i = 0; i < 4; i++) {
        cache_entry_path(ndir, REF, all[i], ".bin", p, sizeof p);
        CHECK(file_exists(p), "negative budget: %.4s.. kept", all[i]);
    }
    free(e);
}

/* ======================================================================== */
/* 5. PROTECT SET -- a run's declared hashes are never victims              */
/* ======================================================================== */

static void test_protect(const char *tmp_base) {
    printf("== 5. protect set (declared hashes never evicted) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/protect", tmp_base);
    make_ladder(cdir);

    cache_weights w; cache_weights_resolve(cdir, &w);
    int n;
    cache_entry *e = cache_scan(cdir, &n, NULL);

    /* h2 has the lowest score but is declared; the next two scores go instead */
    const char *protect[] = { H2 };
    int evicted = cache_gc(cdir, e, n, 2500, protect, 1, &w, NOW, 0, 1);
    CHECK(evicted == 2, "evicted exactly 2 (got %d)", evicted);

    char p[4096];
    cache_entry_path(cdir, REF, H2, ".bin", p, sizeof p);
    CHECK(file_exists(p), "protected h2 survives despite the lowest score");
    cache_entry_path(cdir, REF, H1, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "unprotected h1 evicted");
    cache_entry_path(cdir, REF, H3, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "unprotected h3 evicted (higher score than protected h2)");
    cache_entry_path(cdir, REF, H4, ".bin", p, sizeof p);
    CHECK(file_exists(p), "h4 survives (budget reached)");
    free(e);
}

/* ======================================================================== */
/* 6. NOATIME FALLBACK -- rank on mtime when atime cannot advance           */
/* ======================================================================== */

static void test_noatime_fallback(const char *tmp_base) {
    printf("== 6. noatime fallback (rank on mtime) ==\n");
    /* Equal cost, opposed timestamps: hA is fresh by atime and ancient by mtime,
     * hB the reverse, so the ranking axis alone decides the victim. */
    const char *hA = "0a000000000000000000000000000001";
    const char *hB = "0b000000000000000000000000000002";
    cache_weights w; cache_weights_resolve("/nonexistent-cache-dir", &w);
    char p[4096];

    char adir[512]; snprintf(adir, sizeof adir, "%s/rank_atime", tmp_base);
    make_entry(adir, REF, hA, 1000, 1.0, NOW -   1 * DAY, NOW - 800 * DAY);
    make_entry(adir, REF, hB, 1000, 1.0, NOW - 800 * DAY, NOW -   1 * DAY);
    int n;
    cache_entry *e = cache_scan(adir, &n, NULL);
    cache_gc(adir, e, n, 1500, NULL, 0, &w, NOW, 0, 1);   /* rank on atime */
    cache_entry_path(adir, REF, hB, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "atime ranking: the atime-cold entry is the victim");
    cache_entry_path(adir, REF, hA, ".bin", p, sizeof p);
    CHECK(file_exists(p), "atime ranking: the atime-fresh entry survives");
    free(e);

    char mdir[512]; snprintf(mdir, sizeof mdir, "%s/rank_mtime", tmp_base);
    make_entry(mdir, REF, hA, 1000, 1.0, NOW -   1 * DAY, NOW - 800 * DAY);
    make_entry(mdir, REF, hB, 1000, 1.0, NOW - 800 * DAY, NOW -   1 * DAY);
    e = cache_scan(mdir, &n, NULL);
    cache_gc(mdir, e, n, 1500, NULL, 0, &w, NOW, 1, 1);   /* rank on mtime */
    cache_entry_path(mdir, REF, hA, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "mtime ranking: the mtime-cold entry is the victim");
    cache_entry_path(mdir, REF, hB, ".bin", p, sizeof p);
    CHECK(file_exists(p), "mtime ranking: the mtime-fresh entry survives");
    free(e);
}

/* ======================================================================== */
/* 7. ATOMIC WRITE -- no partial, re-store identical                        */
/* ======================================================================== */

static void test_atomic(const char *tmp_base) {
    printf("== 7. atomic write (tmp+rename, no partial) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/atomic", tmp_base);
    const char *ref = "abababababababababababababababab";  /* synthetic reference_md5 (32 hex) */
    const char *h   = "a7000000000000000000000000000000";
    char bin[4096]; cache_entry_path(cdir, ref, h, ".bin", bin, sizeof bin);

    char data1[4096]; memset(data1, 'A', sizeof data1);
    CHECK(cache_write_atomic(bin, data1, sizeof data1) == 0, "first write ok");
    CHECK(file_exists(bin), "target exists after write");
    CHECK(count_tmp(cdir) == 0, "no .tmp.<pid> file left behind");

    /* content readback */
    FILE *f = fopen(bin, "rb"); char rb[4096]; size_t got = f ? fread(rb, 1, sizeof rb, f) : 0; if (f) fclose(f);
    CHECK(got == sizeof data1 && memcmp(rb, data1, sizeof data1) == 0, "content matches after first write");

    /* re-store the same deterministic bytes -> identical file, still no tmp */
    CHECK(cache_write_atomic(bin, data1, sizeof data1) == 0, "re-store ok");
    CHECK(count_tmp(cdir) == 0, "no tmp after re-store");
    f = fopen(bin, "rb"); got = f ? fread(rb, 1, sizeof rb, f) : 0; if (f) fclose(f);
    CHECK(got == sizeof data1 && memcmp(rb, data1, sizeof data1) == 0, "re-store wrote identical bytes");
}

/* ======================================================================== */
/* 8. .meta -- cost.actual round-trip; an unrecognized shape reads as 0     */
/* ======================================================================== */

static void test_meta(const char *tmp_base) {
    printf("== 8. .meta cost.actual (round-trip, unreadable shape) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/meta", tmp_base);
    const char *hFast = "0f000000000000000000000000000001";  /* sub-second render */
    const char *hSlow = "0f000000000000000000000000000002";
    make_entry(cdir, REF, hFast, 1000, 0.001234, NOW - DAY, NOW - DAY);
    make_entry(cdir, REF, hSlow, 1000, 987.654321, NOW - DAY, NOW - DAY);

    char meta[4096];
    cache_entry_path(cdir, REF, hFast, ".meta", meta, sizeof meta);
    char *txt = read_text(meta);
    CHECK(txt && strstr(txt, "\"cost\"") && strstr(txt, "\"actual\""),
          "written .meta is {\"cost\":{\"actual\":...}}: %s", txt ? txt : "(unreadable)");
    free(txt);

    int n;
    cache_entry *e = cache_scan(cdir, &n, NULL);
    double fast = -1, slow = -1;
    for (int i = 0; i < n; i++) {
        if (!strcmp(e[i].args_hash, hFast)) fast = e[i].cost_actual;
        if (!strcmp(e[i].args_hash, hSlow)) slow = e[i].cost_actual;
    }
    CHECK(deq(fast, 0.001234), "sub-second cost.actual round-trips (got %.6f)", fast);
    CHECK(deq(slow, 987.654321), "long cost.actual round-trips (got %.6f)", slow);
    free(e);

    /* An unrecognized .meta shape reads as 0 and is therefore the first victim. */
    char ldir[512]; snprintf(ldir, sizeof ldir, "%s/meta_unreadable", tmp_base);
    const char *hOld = "0e000000000000000000000000000001";
    const char *hNew = "0e000000000000000000000000000002";
    cache_entry_path(ldir, REF, hOld, ".meta", meta, sizeof meta);
    write_text(meta, "{\"k\":42,\"actual_seconds\":9.500000}\n");
    make_bin(ldir, REF, hOld, 1000, NOW - DAY, NOW - DAY);
    make_entry(ldir, REF, hNew, 1000, 0.5, NOW - DAY, NOW - DAY);

    e = cache_scan(ldir, &n, NULL);
    double old_cost = -1;
    for (int i = 0; i < n; i++)
        if (!strcmp(e[i].args_hash, hOld)) old_cost = e[i].cost_actual;
    CHECK(old_cost == 0.0, "unreadable .meta reads cost.actual 0 (got %.6f)", old_cost);

    cache_weights w; cache_weights_resolve(ldir, &w);
    int evicted = cache_gc(ldir, e, n, 1500, NULL, 0, &w, NOW, 0, 1);
    char p[4096];
    cache_entry_path(ldir, REF, hOld, ".bin", p, sizeof p);
    CHECK(evicted == 1 && !file_exists(p), "cost 0 evicts first (evicted %d)", evicted);
    cache_entry_path(ldir, REF, hNew, ".bin", p, sizeof p);
    CHECK(file_exists(p), "the entry with a readable cost survives");
    free(e);
}

/* ======================================================================== */
/* 9. STATUS -- total size + count vs budget                               */
/* ======================================================================== */

static void test_status(const char *tmp_base) {
    printf("== 9. status (size + count vs budget) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/status", tmp_base);
    const char *ref = "0123456789abcdef0123456789abcdef";  /* synthetic reference_md5 (32 hex) */
    const char *hA = "50000000000000000000000000000001";
    const char *hB = "60000000000000000000000000000002";
    make_entry(cdir, ref, hA, 4096, 1.0, NOW - DAY, NOW - DAY);
    make_entry(cdir, ref, hB, 2048, 1.0, NOW - DAY, NOW - DAY);

    int n;
    cache_entry *e = cache_scan(cdir, &n, NULL);
    cache_status st; cache_compute_status(e, n, 11, &st);
    CHECK(st.n_entries == 2, "status n_entries == 2 (got %d)", st.n_entries);
    CHECK(st.total_bytes == 6144, "status total_bytes == 6144 (got %lld)", st.total_bytes);
    CHECK(st.budget_bytes == (11LL << 30), "status budget_bytes == 11 GB (got %lld)", st.budget_bytes);
    free(e);
}

/* ======================================================================== */
/* 10. CONFIG -- budget resolution, weight resolution, merge-write          */
/* ======================================================================== */

static void check_weights(const char *cdir, double a, double b, double s0,
                          const char *label) {
    cache_weights w; cache_weights_resolve(cdir, &w);
    CHECK(deq(w.cost_weight, a) && deq(w.decay, b) && deq(w.grace_seconds, s0),
          "%s -> a=%g b=%g s0=%g (got a=%g b=%g s0=%g)",
          label, a, b, s0, w.cost_weight, w.decay, w.grace_seconds);
}

static void test_config(const char *tmp_base) {
    printf("== 10. config (budget + weights + merge-write) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/config", tmp_base);

    /* budget: default -> settings -> flag */
    CHECK(cache_settings_budget_gb(cdir) == -1, "no settings.json -> -1 (absent)");
    CHECK(cache_budget_resolve(cdir, -1) == 11, "resolve default == 11 (no settings, no flag)");
    CHECK(cache_set_budget_gb(cdir, 7) == 0, "set-budget-gb 7 ok");
    CHECK(count_tmp(cdir) == 0, "no settings.json.tmp left behind");
    CHECK(cache_settings_budget_gb(cdir) == 7, "settings.json budget_gb == 7");
    CHECK(cache_budget_resolve(cdir, -1) == 7, "resolve == settings (7) when no flag");
    CHECK(cache_budget_resolve(cdir, 42) == 42, "resolve == flag (42) overrides settings");

    /* weights: compile defaults, hand-written overrides, per-key validation */
    double s30 = 30.0 * 86400.0, s7 = 7.0 * 86400.0;
    char wdir[512];
    snprintf(wdir, sizeof wdir, "%s/w_default", tmp_base);
    check_weights(wdir, 1.8, 1.2, s30, "no settings.json");

    snprintf(wdir, sizeof wdir, "%s/w_set", tmp_base);
    write_settings(wdir, "{\"evict_cost_weight\":2.5,\"evict_decay\":0.5,\"evict_grace_days\":7}\n");
    check_weights(wdir, 2.5, 0.5, s7, "hand-written settings.json");

    snprintf(wdir, sizeof wdir, "%s/w_type", tmp_base);
    write_settings(wdir, "{\"evict_cost_weight\":\"two\",\"evict_decay\":0.5,\"evict_grace_days\":7}\n");
    check_weights(wdir, 1.8, 0.5, s7, "wrong type on evict_cost_weight");

    snprintf(wdir, sizeof wdir, "%s/w_neg", tmp_base);
    write_settings(wdir, "{\"evict_cost_weight\":2.5,\"evict_decay\":-1.0,\"evict_grace_days\":7}\n");
    check_weights(wdir, 2.5, 1.2, s7, "negative evict_decay");

    snprintf(wdir, sizeof wdir, "%s/w_zero", tmp_base);
    write_settings(wdir, "{\"evict_cost_weight\":2.5,\"evict_decay\":0.5,\"evict_grace_days\":0}\n");
    check_weights(wdir, 2.5, 0.5, s30, "zero evict_grace_days");

    snprintf(wdir, sizeof wdir, "%s/w_inf", tmp_base);
    write_settings(wdir, "{\"evict_cost_weight\":1e400,\"evict_decay\":0.5,\"evict_grace_days\":7}\n");
    check_weights(wdir, 1.8, 0.5, s7, "non-finite evict_cost_weight");

    snprintf(wdir, sizeof wdir, "%s/w_corrupt", tmp_base);
    write_settings(wdir, "not json at all\n");
    check_weights(wdir, 1.8, 1.2, s30, "corrupt settings.json");

    /* merge-write: --set-budget-gb replaces budget_gb and nothing else */
    char mdir[512]; snprintf(mdir, sizeof mdir, "%s/merge", tmp_base);
    write_settings(mdir, "{\"budget_gb\":5,\"evict_cost_weight\":2.5,"
                         "\"evict_decay\":0.5,\"evict_grace_days\":7,\"note\":\"hand\"}\n");
    CHECK(cache_set_budget_gb(mdir, 9) == 0, "merge: set-budget-gb 9 ok");
    CHECK(cache_settings_budget_gb(mdir) == 9, "merge: budget_gb updated to 9");
    check_weights(mdir, 2.5, 0.5, s7, "merge: hand-added weights preserved");
    char mp[4096]; snprintf(mp, sizeof mp, "%s/settings.json", mdir);
    char *txt = read_text(mp);
    CHECK(txt && strstr(txt, "\"note\"") && strstr(txt, "hand"),
          "merge: unrelated key preserved verbatim: %s", txt ? txt : "(unreadable)");
    free(txt);
    CHECK(count_tmp(mdir) == 0, "merge: no settings.json.tmp left behind");

    /* a corrupt settings.json starts from an empty object rather than failing */
    char bdir[512]; snprintf(bdir, sizeof bdir, "%s/merge_corrupt", tmp_base);
    write_settings(bdir, "{ this is not json\n");
    CHECK(cache_set_budget_gb(bdir, 3) == 0, "merge: corrupt settings.json rewritten ok");
    CHECK(cache_settings_budget_gb(bdir) == 3, "merge: budget_gb == 3 after corrupt rewrite");

    /* cache_dir_resolve: flag wins; XDG honored; string-only (no disk touch) */
    char *d = cache_dir_resolve("/some/explicit/dir");
    CHECK(d && strcmp(d, "/some/explicit/dir") == 0, "cache_dir_resolve: --cache-dir flag wins"); free(d);
    setenv("XDG_CACHE_HOME", "/xdgroot", 1);
    d = cache_dir_resolve(NULL);
    CHECK(d && strcmp(d, "/xdgroot/nanobragg") == 0, "cache_dir_resolve: XDG_CACHE_HOME honored"); free(d);
    unsetenv("XDG_CACHE_HOME");
}

/* ======================================================================== */
/* 11. TWO SUITES -- one suite's gc must not wipe another suite's warm cache */
/* ======================================================================== */

/* Suite A is warm and expensive (a long render, read yesterday); suite B is
 * cheap and cold. gc runs with an EMPTY protect set -- the `nbcache --gc` case,
 * no run in flight -- so nothing is shielded by declaration and only the budget
 * and the score decide. Under budget that must delete nothing at all; over
 * budget it must take the cheap-and-cold entries and stop at the budget, never
 * reaching into the expensive suite. */
static void test_two_suite_no_wipe(const char *tmp_base) {
    printf("== 11. two suites (a gc with no run in flight keeps the warm cache) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/two_suite", tmp_base);
    const char *A1 = "a1000000000000000000000000000001";  /* suite A: 900 s,   1 d */
    const char *A2 = "a2000000000000000000000000000002";
    const char *B1 = "b1000000000000000000000000000001";  /* suite B:   2 s, 120 d */
    const char *B2 = "b2000000000000000000000000000002";
    make_entry(cdir, REF, A1, 3000, 900.0, NOW -   1 * DAY, NOW -   1 * DAY);
    make_entry(cdir, REF, A2, 3000, 900.0, NOW -   1 * DAY, NOW -   1 * DAY);
    make_entry(cdir, REF, B1, 1000,   2.0, NOW - 120 * DAY, NOW - 120 * DAY);
    make_entry(cdir, REF, B2, 1000,   2.0, NOW - 120 * DAY, NOW - 120 * DAY);
    const char *all[] = { A1, A2, B1, B2 };

    cache_weights w; cache_weights_resolve(cdir, &w);
    char p[4096];
    int n;

    /* 8000 bytes on disk, budget 11000 -> nothing is a victim */
    cache_entry *e = cache_scan(cdir, &n, NULL);
    CHECK(n == 4, "scan found both suites (4 entries, got %d)", n);
    int evicted = cache_gc(cdir, e, n, 11000, NULL, 0, &w, NOW, 0, 1);
    CHECK(evicted == 0, "under budget with an empty protect set -> 0 evicted (got %d)", evicted);
    for (int i = 0; i < 4; i++) {
        cache_entry_path(cdir, REF, all[i], ".bin", p, sizeof p);
        CHECK(file_exists(p), "under budget: %.2s.. image survives", all[i]);
    }
    free(e);

    /* budget 6000 -> free 2000: the two cheap-and-cold images, then stop */
    e = cache_scan(cdir, &n, NULL);
    evicted = cache_gc(cdir, e, n, 6000, NULL, 0, &w, NOW, 0, 1);
    CHECK(evicted == 2, "over budget -> exactly the 2 cheap-and-cold evicted (got %d)", evicted);
    cache_entry_path(cdir, REF, A1, ".bin", p, sizeof p);
    CHECK(file_exists(p), "warm suite A image a1.. survives the trim");
    cache_entry_path(cdir, REF, A2, ".bin", p, sizeof p);
    CHECK(file_exists(p), "warm suite A image a2.. survives the trim");
    cache_entry_path(cdir, REF, B1, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "cheap suite B image b1.. evicted");
    cache_entry_path(cdir, REF, B2, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "cheap suite B image b2.. evicted");
    for (int i = 0; i < 4; i++) {
        cache_entry_path(cdir, REF, all[i], ".meta", p, sizeof p);
        CHECK(file_exists(p), "over budget: %.2s.. .meta survives", all[i]);
    }
    /* A surviving .meta pins its shard, so the emptied-dir prune cannot remove a
     * directory an entry still lives in -- the property that lets the prune run
     * next to a process writing new entries. */
    for (int i = 2; i < 4; i++) {
        snprintf(p, sizeof p, "%s/%s/%.2s", cdir, REF, all[i]);
        CHECK(dir_exists(p), "over budget: %.2s shard dir survives the prune", all[i]);
    }
    free(e);
}

/* ======================================================================== */
/* 12. CLI -- `nbcache --gc` is a score-to-budget trim, not a cache wipe     */
/* ======================================================================== */

/* Section 11 pins the library contract; this pins the TOOL, which is what a user
 * actually runs. `nbcache --gc` has no suite in flight and so passes an empty
 * protect set: if it ever traded that for "keep only what some suite declares",
 * a warm expensive cache would be wiped and nothing above would notice. Same
 * fixture as section 11, scaled to the --budget-gb GB granularity via sparse
 * .bin, and driven through the real binary. */

/* Run the tool and return the evicted count it reports, or -1 if unreadable. */
static int run_nbcache_gc(const char *nbcache_bin, const char *cache_dir, long budget_gb) {
    char cmd[8192];
    snprintf(cmd, sizeof cmd, "%s --gc --cache-dir %s --budget-gb %ld",
             nbcache_bin, cache_dir, budget_gb);
    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;
    char line[4096], last[4096] = {0};
    while (fgets(line, sizeof line, fp)) snprintf(last, sizeof last, "%s", line);
    pclose(fp);
    const char *p = strstr(last, "evicted ");
    return p ? atoi(p + 8) : -1;
}

static void test_cli_gc(const char *nbcache_bin, const char *tmp_base) {
    printf("== 12. nbcache --gc (CLI: score-to-budget trim, empty protect set) ==\n");
    if (!nbcache_bin) {
        printf("  note: nbcache binary not supplied -- CLI --gc check skipped\n");
        return;
    }
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/cli_gc", tmp_base);
    const char *C1 = "c1000000000000000000000000000001";  /* warm suite: 900 s,   1 d */
    const char *C2 = "c2000000000000000000000000000002";
    const char *D1 = "d1000000000000000000000000000001";  /* cheap suite:  2 s, 120 d */
    const char *D2 = "d2000000000000000000000000000002";
    const char *all[] = { C1, C2, D1, D2 };
    const long long GB = 1LL << 30;

    /* The tool scores against the real clock, so the timestamps are real too. */
    long now = (long)time(NULL);
    make_sparse_entry(cdir, REF, C1, 3 * GB, 900.0, now -   1 * DAY);
    make_sparse_entry(cdir, REF, C2, 3 * GB, 900.0, now -   1 * DAY);
    make_sparse_entry(cdir, REF, D1, 1 * GB,   2.0, now - 120 * DAY);
    make_sparse_entry(cdir, REF, D2, 1 * GB,   2.0, now - 120 * DAY);

    char p[4096];

    /* 8 GB on disk, budget 11 GB -> the trim is a no-op */
    int evicted = run_nbcache_gc(nbcache_bin, cdir, 11);
    CHECK(evicted == 0, "under budget: nbcache --gc reports 0 evicted (got %d)", evicted);
    for (int i = 0; i < 4; i++) {
        cache_entry_path(cdir, REF, all[i], ".bin", p, sizeof p);
        CHECK(file_exists(p), "under budget: %.2s.. image survives nbcache --gc", all[i]);
    }

    /* budget 6 GB -> free 2 GB: the two cheap-and-cold images, then stop */
    evicted = run_nbcache_gc(nbcache_bin, cdir, 6);
    CHECK(evicted == 2, "over budget: nbcache --gc reports 2 evicted (got %d)", evicted);
    cache_entry_path(cdir, REF, C1, ".bin", p, sizeof p);
    CHECK(file_exists(p), "warm image c1.. survives nbcache --gc");
    cache_entry_path(cdir, REF, C2, ".bin", p, sizeof p);
    CHECK(file_exists(p), "warm image c2.. survives nbcache --gc");
    cache_entry_path(cdir, REF, D1, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "cheap image d1.. evicted by nbcache --gc");
    cache_entry_path(cdir, REF, D2, ".bin", p, sizeof p);
    CHECK(!file_exists(p), "cheap image d2.. evicted by nbcache --gc");
    for (int i = 0; i < 4; i++) {
        cache_entry_path(cdir, REF, all[i], ".meta", p, sizeof p);
        CHECK(file_exists(p), "nbcache --gc: %.2s.. .meta survives", all[i]);
    }
}

/* ======================================================================== */

int main(int argc, char **argv) {
    const char *nbcache_bin = (argc > 1) ? argv[1] : NULL;

    char tmpl[] = "/tmp/nbcache_test_XXXXXX";
    char *base = mkdtemp(tmpl);
    if (!base) { fprintf(stderr, "mkdtemp failed\n"); return 2; }
    printf("temp cache base: %s (real ~/.cache untouched)\n\n", base);

    test_key_agreement(nbcache_bin, base);
    test_score();
    test_budget_eviction(base);
    test_under_budget(base);
    test_protect(base);
    test_noatime_fallback(base);
    test_atomic(base);
    test_meta(base);
    test_status(base);
    test_config(base);
    test_two_suite_no_wipe(base);
    test_cli_gc(nbcache_bin, base);

    /* clean up the temp tree */
    char rm[4200]; snprintf(rm, sizeof rm, "rm -rf %s", base); if (system(rm)) {}

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "TEST PASSED", failures);
    return failures ? 1 : 0;
}
