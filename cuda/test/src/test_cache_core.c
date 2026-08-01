/* test_cache_core.c -- unit test for cache_core + the nbcache --path key.
 *
 * Synthetic only: every cache operation runs against a fresh mkdtemp() temp dir,
 * never the real ~/.cache/nanobragg, and no nanoBragg render is invoked. Covers
 * the NBTOOLS-SPEC §9 invariants: path/key agreement (baked args_hash ==
 * nbcache --path), cost+recency eviction, atomic writes, orphan-gc (prune only
 * empty dirs), and budget config resolution.
 *
 *   Usage: test_cache_core [nbcache_binary]
 * If nbcache_binary is given, the key-agreement check also runs the tool via
 * popen; otherwise it verifies the library entry point (cache_args_hash) alone.
 * Run from the worktree root (spec/suites paths are repo-relative).
 */
#define _XOPEN_SOURCE 700
#include "cache_core.h"
#include "case_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (cond) { printf("  ok  : " __VA_ARGS__); printf("\n"); } \
    else { printf("  FAIL: " __VA_ARGS__); printf("\n"); failures++; } \
} while (0)

/* ---- small filesystem/util helpers ------------------------------------- */

static int file_exists(const char *p) { struct stat s; return stat(p, &s) == 0; }

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

/* Write a synthetic cache entry: .bin of `nbytes`, .meta with k, then stamp both
 * files' mtime. Uses the library's own atomic writer and path assembly. */
static void make_entry(const char *cache_dir, const char *ref_md5,
                       const char *args_hash, long nbytes, long k, long mtime) {
    char *buf = malloc((size_t)nbytes);
    memset(buf, 'x', (size_t)nbytes);
    char bin[4096], meta[4096];
    cache_entry_path(cache_dir, ref_md5, args_hash, ".bin", bin, sizeof bin);
    cache_write_atomic(bin, buf, (size_t)nbytes);
    free(buf);
    cache_write_meta(cache_dir, ref_md5, args_hash, k, (double)k * 0.01);
    cache_entry_path(cache_dir, ref_md5, args_hash, ".meta", meta, sizeof meta);
    struct timeval tv[2] = { { mtime, 0 }, { mtime, 0 } };
    utimes(bin, tv);
    utimes(meta, tv);
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
/* 2. EVICTION ORDERING -- cheapest-and-coldest first, expensive/hot survive */
/* ======================================================================== */

static void test_eviction(const char *tmp_base) {
    printf("== 2. eviction ordering (cost + recency) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/evict", tmp_base);
    const char *ref = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";  /* synthetic reference_md5 (32 hex) */
    /* args_hashes, K, mtime -- cheap aligned with cold, expensive with hot. */
    const char *h1 = "e1000000000000000000000000000001"; long k1 = 1,    m1 = 1000000000;
    const char *h2 = "e2000000000000000000000000000002"; long k2 = 10,   m2 = 1000000100;
    const char *h3 = "e3000000000000000000000000000003"; long k3 = 100,  m3 = 1000000200;
    const char *h4 = "e4000000000000000000000000000004"; long k4 = 1000, m4 = 1000000300;
    make_entry(cdir, ref, h1, 1000, k1, m1);
    make_entry(cdir, ref, h2, 1000, k2, m2);
    make_entry(cdir, ref, h3, 1000, k3, m3);
    make_entry(cdir, ref, h4, 1000, k4, m4);

    const char *live[] = { h1, h2, h3, h4 };
    long live_k[] = { k1, k2, k3, k4 };
    int n;
    cache_entry *e = cache_scan(cdir, live, live_k, 4, &n);
    CHECK(n == 4, "scan found 4 entries (got %d)", n);

    /* budget 2500 bytes: 4x1000 = 4000 on disk -> evict 2 cheapest until <=2500 */
    long run_start = 2000000000;   /* all entries older -> all evictable */
    int oe, be;
    int total = cache_gc(cdir, e, n, 2500, run_start, 1, &oe, &be);
    CHECK(be == 2 && oe == 0, "budget eviction removed exactly 2, 0 orphans (got be=%d oe=%d)", be, oe);
    CHECK(total == 2, "total evicted == 2 (got %d)", total);

    char p[4096];
    cache_entry_path(cdir, ref, h1, ".bin", p, sizeof p); CHECK(!file_exists(p), "cheapest+coldest h1 evicted");
    cache_entry_path(cdir, ref, h2, ".bin", p, sizeof p); CHECK(!file_exists(p), "next-cheapest h2 evicted");
    cache_entry_path(cdir, ref, h3, ".bin", p, sizeof p); CHECK(file_exists(p),  "expensive h3 survives");
    cache_entry_path(cdir, ref, h4, ".bin", p, sizeof p); CHECK(file_exists(p),  "most-expensive+hot h4 survives");
    /* budget eviction keeps .meta for the calibration dataset */
    cache_entry_path(cdir, ref, h1, ".meta", p, sizeof p); CHECK(file_exists(p), "evicted h1 .meta survives (budget eviction keeps .meta)");
    free(e);

    /* tie-break: equal K, colder mtime evicted first */
    char tdir[512]; snprintf(tdir, sizeof tdir, "%s/evict_tie", tmp_base);
    const char *t_cold = "7c000000000000000000000000000000"; long tm_cold = 1000000000;
    const char *t_hot  = "7d000000000000000000000000000000"; long tm_hot  = 1000009999;
    make_entry(tdir, ref, t_cold, 1000, 50, tm_cold);
    make_entry(tdir, ref, t_hot,  1000, 50, tm_hot);
    const char *tlive[] = { t_cold, t_hot }; long tlk[] = { 50, 50 };
    int tn; cache_entry *te = cache_scan(tdir, tlive, tlk, 2, &tn);
    cache_gc(tdir, te, tn, 1500, run_start, 1, &oe, &be);   /* room for 1 -> evict 1 */
    cache_entry_path(tdir, ref, t_cold, ".bin", p, sizeof p); CHECK(!file_exists(p), "tie-break: colder evicted first");
    cache_entry_path(tdir, ref, t_hot,  ".bin", p, sizeof p); CHECK(file_exists(p),  "tie-break: hotter survives");
    free(te);
}

/* ======================================================================== */
/* 3. ATOMIC WRITE -- no partial, re-store identical                        */
/* ======================================================================== */

static void test_atomic(const char *tmp_base) {
    printf("== 3. atomic write (tmp+rename, no partial) ==\n");
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
/* 4. ORPHAN-GC -- drop unreferenced, prune only empty dirs, young survives  */
/* ======================================================================== */

static void test_orphan_gc(const char *tmp_base) {
    printf("== 4. orphan-gc (prune only empty dirs) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/orphan", tmp_base);
    const char *refA = "aaaa0000000000000000000000000000";  /* all orphan -> emptied */
    const char *refB = "bbbb0000000000000000000000000000";  /* all live   -> kept */
    const char *refC = "cccc0000000000000000000000000000";  /* mixed      -> kept */

    const char *live_h  = "b0000000000000000000000000000001";  /* referenced */
    const char *live_h2 = "c0000000000000000000000000000002";  /* referenced (in refC) */
    const char *orphA   = "a0000000000000000000000000000003";  /* unreferenced */
    const char *orphC   = "a0000000000000000000000000000004";  /* unreferenced (in refC) */
    const char *young_o = "a0000000000000000000000000000005";  /* unreferenced but young */

    long past = 1000000000, future = 3000000000L, run_start = 2000000000;
    make_entry(cdir, refA, orphA,   1000, 5, past);
    make_entry(cdir, refA, young_o, 1000, 5, future);   /* young orphan: must survive */
    make_entry(cdir, refB, live_h,  1000, 5, past);
    make_entry(cdir, refC, live_h2, 1000, 5, past);
    make_entry(cdir, refC, orphC,   1000, 5, past);

    const char *live[] = { live_h, live_h2 };
    long live_k[] = { 5, 5 };
    int n;
    cache_entry *e = cache_scan(cdir, live, live_k, 2, &n);
    CHECK(n == 5, "scan found 5 entries (got %d)", n);

    int oe, be;
    cache_gc(cdir, e, n, -1 /* budget phase off */, run_start, 1, &oe, &be);
    CHECK(oe == 2 && be == 0, "2 orphans removed, 0 budget (got oe=%d be=%d); young orphan spared", oe, be);
    free(e);

    char p[4096];
    /* orphan entries: both .bin AND .meta gone */
    cache_entry_path(cdir, refA, orphA, ".bin",  p, sizeof p); CHECK(!file_exists(p), "orphan .bin dropped");
    cache_entry_path(cdir, refA, orphA, ".meta", p, sizeof p); CHECK(!file_exists(p), "orphan .meta dropped");
    /* young orphan spared */
    cache_entry_path(cdir, refA, young_o, ".bin", p, sizeof p); CHECK(file_exists(p), "young orphan survives (younger than run)");
    /* refA still holds the young orphan -> NOT pruned */
    snprintf(p, sizeof p, "%s/%s", cdir, refA); CHECK(file_exists(p), "refA kept (still holds the young orphan)");
    /* live entries untouched */
    cache_entry_path(cdir, refB, live_h,  ".bin", p, sizeof p); CHECK(file_exists(p), "live refB entry kept");
    snprintf(p, sizeof p, "%s/%s", cdir, refB); CHECK(file_exists(p), "refB dir kept (non-empty)");
    /* refC mixed: orphan removed, live kept, dir kept */
    cache_entry_path(cdir, refC, orphC,   ".bin", p, sizeof p); CHECK(!file_exists(p), "refC orphan dropped");
    cache_entry_path(cdir, refC, live_h2, ".bin", p, sizeof p); CHECK(file_exists(p),  "refC live entry kept");
    snprintf(p, sizeof p, "%s/%s", cdir, refC); CHECK(file_exists(p), "refC dir kept (non-empty sibling)");

    /* Now make refA fully orphan (drop the young guard by moving run_start) and
     * confirm an EMPTIED reference dir is pruned. */
    cache_entry *e2 = cache_scan(cdir, live, live_k, 2, &n);
    cache_gc(cdir, e2, n, -1, 4000000000L /* run_start far future -> nothing young */, 1, &oe, &be);
    free(e2);
    snprintf(p, sizeof p, "%s/%s", cdir, refA); CHECK(!file_exists(p), "refA pruned once emptied");
    snprintf(p, sizeof p, "%s/%s", cdir, refB); CHECK(file_exists(p), "refB still kept (non-empty)");
    snprintf(p, sizeof p, "%s/%s", cdir, refC); CHECK(file_exists(p), "refC still kept (non-empty)");
}

/* ======================================================================== */
/* 5. STATUS -- total size + count vs budget                               */
/* ======================================================================== */

static void test_status(const char *tmp_base) {
    printf("== 5. status (size + count vs budget) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/status", tmp_base);
    const char *ref = "0123456789abcdef0123456789abcdef";  /* synthetic reference_md5 (32 hex) */
    const char *hL = "50000000000000000000000000000001";  /* live */
    const char *hO = "60000000000000000000000000000002";  /* orphan */
    make_entry(cdir, ref, hL, 4096, 5, 1000000000);
    make_entry(cdir, ref, hO, 2048, 5, 1000000000);

    const char *live[] = { hL }; long lk[] = { 5 };
    int n; cache_entry *e = cache_scan(cdir, live, lk, 1, &n);
    cache_status st; cache_compute_status(e, n, 20, &st);
    CHECK(st.n_entries == 2, "status n_entries == 2 (got %d)", st.n_entries);
    CHECK(st.total_bytes == 6144, "status total_bytes == 6144 (got %lld)", st.total_bytes);
    CHECK(st.n_orphans == 1, "status n_orphans == 1 (got %d)", st.n_orphans);
    CHECK(st.budget_bytes == (20LL << 30), "status budget_bytes == 20 GB (got %lld)", st.budget_bytes);
    free(e);
}

/* ======================================================================== */
/* 6. CONFIG -- settings.json write + resolution order flag>settings>default  */
/* ======================================================================== */

static void test_config(const char *tmp_base) {
    printf("== 6. config (budget resolution flag>settings>default) ==\n");
    char cdir[512]; snprintf(cdir, sizeof cdir, "%s/config", tmp_base);

    /* fresh dir: no settings.json -> default 20 */
    CHECK(cache_settings_budget_gb(cdir) == -1, "no settings.json -> -1 (absent)");
    CHECK(cache_budget_resolve(cdir, -1) == 20, "resolve default == 20 (no settings, no flag)");

    /* --set-budget-gb writes settings.json atomically */
    CHECK(cache_set_budget_gb(cdir, 7) == 0, "set-budget-gb 7 ok");
    CHECK(count_tmp(cdir) == 0, "no settings.json.tmp left behind");
    CHECK(cache_settings_budget_gb(cdir) == 7, "settings.json budget_gb == 7");
    CHECK(cache_budget_resolve(cdir, -1) == 7, "resolve == settings (7) when no flag");
    CHECK(cache_budget_resolve(cdir, 42) == 42, "resolve == flag (42) overrides settings");

    /* cache_dir_resolve: flag wins; XDG honored; string-only (no disk touch) */
    char *d = cache_dir_resolve("/some/explicit/dir");
    CHECK(d && strcmp(d, "/some/explicit/dir") == 0, "cache_dir_resolve: --cache-dir flag wins"); free(d);
    setenv("XDG_CACHE_HOME", "/xdgroot", 1);
    d = cache_dir_resolve(NULL);
    CHECK(d && strcmp(d, "/xdgroot/nanobragg") == 0, "cache_dir_resolve: XDG_CACHE_HOME honored"); free(d);
    unsetenv("XDG_CACHE_HOME");
}

/* ======================================================================== */

int main(int argc, char **argv) {
    const char *nbcache_bin = (argc > 1) ? argv[1] : NULL;

    char tmpl[] = "/tmp/nbcache_test_XXXXXX";
    char *base = mkdtemp(tmpl);
    if (!base) { fprintf(stderr, "mkdtemp failed\n"); return 2; }
    printf("temp cache base: %s (real ~/.cache untouched)\n\n", base);

    test_key_agreement(nbcache_bin, base);
    test_eviction(base);
    test_atomic(base);
    test_orphan_gc(base);
    test_status(base);
    test_config(base);

    /* clean up the temp tree */
    char rm[4200]; snprintf(rm, sizeof rm, "rm -rf %s", base); if (system(rm)) {}

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "TEST PASSED", failures);
    return failures ? 1 : 0;
}
