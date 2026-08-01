/* cache_core.h -- image-cache path assembly, config, atomic writes, scan, gc.
 *
 * The image cache stores rendered REFERENCE images (NBTOOLS-SPEC §9), one per
 * case, machine-global and shared across worktrees. This module owns everything
 * about the cache on disk; the nbcache tool and nbrunsuite drive it.
 *
 * Layout (§9):
 *   <cache-dir>/<reference_md5>/<AB>/<args_hash>.bin   image, evictable
 *   <cache-dir>/<reference_md5>/<AB>/<args_hash>.meta  {"k","actual_seconds"}, survives eviction
 * <AB> = first 2 hex of args_hash (shard). reference_md5 = md5 of the reference
 * binary FILE (a directory level, so reference versions coexist as siblings).
 * args_hash = md5(argkey_canonicalize(reference_args in {input_root}-token form)),
 * the SAME baked key nbgensuite writes and nbrunsuite looks up (§9 invariant).
 *
 * Location & budget bootstrap from env/flag, NOT the config file (§9): the cache
 * dir is $XDG_CACHE_HOME/nanobragg (default ~/.cache/nanobragg) or --cache-dir;
 * budget_gb resolves flag > <cache-dir>/settings.json > default 20.
 *
 * Every write is <name>.tmp.<pid> then rename(2), so a reader never sees a
 * partial file; .bin content is deterministic (racing writers write identical
 * bytes), .meta is harmless last-write-wins.
 */
#ifndef NB_CACHE_CORE_H
#define NB_CACHE_CORE_H

#include <stddef.h>

#define CACHE_DEFAULT_BUDGET_GB 20L
#define CACHE_HEXLEN 32              /* md5 hex digest length (no NUL) */

/* ---- location & config (bootstrap from env/flag, never the config file) ---- */

/* Resolve the cache root: flag_override (non-NULL wins) > $XDG_CACHE_HOME/nanobragg
 * > $HOME/.cache/nanobragg. Returns a malloc'd path the caller frees, or NULL if
 * none can be determined. */
char *cache_dir_resolve(const char *flag_override);

/* budget_gb from <cache_dir>/settings.json, or -1 if the file or key is absent.
 * Guards a missing file so json.c's exit-on-open-error cannot fire. */
long cache_settings_budget_gb(const char *cache_dir);

/* Effective budget (§9): flag_gb (>=0 wins) > settings.json > default 20. */
long cache_budget_resolve(const char *cache_dir, long flag_gb);

/* Write {"budget_gb":N} to <cache_dir>/settings.json atomically. 0 ok, -1 err. */
int cache_set_budget_gb(const char *cache_dir, long budget_gb);

/* ---- keys & path assembly ---- */

/* md5 of the reference binary FILE -> out[33] (32 hex + NUL). 0 ok, -1 on read
 * error. */
int cache_reference_md5(const char *ref_path, char out[33]);

/* args_hash = md5(argkey_canonicalize(tok[0..n))) -> out[33]. tok[] are the
 * {input_root}-token reference_args (one argv word each). 0 ok, -1 on failure. */
int cache_args_hash(char **tok, int n, char out[33]);

/* Assemble <cache_dir>/<ref_md5>/<AB>/<args_hash><ext> into out (cap). <AB> =
 * first 2 hex of args_hash. ext is ".bin" or ".meta". Returns out, or NULL if
 * cap is too small or args_hash is malformed. */
char *cache_entry_path(const char *cache_dir, const char *ref_md5,
                       const char *args_hash, const char *ext,
                       char *out, size_t cap);

/* ---- atomic writes ---- */

/* Write buf[0..len) to path via <path>.tmp.<pid> then rename(2); parent dirs are
 * created. 0 ok, -1 err. */
int cache_write_atomic(const char *path, const void *buf, size_t len);

/* Record the .meta {"k":K,"actual_seconds":S} beside the entry, atomically. */
int cache_write_meta(const char *cache_dir, const char *ref_md5,
                     const char *args_hash, long k, double actual_seconds);

/* ---- scan / status / gc ---- */

typedef struct {
    char ref_md5[CACHE_HEXLEN + 1];   /* owning reference dir */
    char args_hash[CACHE_HEXLEN + 1]; /* entry key */
    long mtime;                        /* recency (.bin st_mtime) */
    long long size;                    /* .bin bytes */
    long k;                            /* cost: live-set K if live, else .meta k */
    int orphan;                        /* 1 = args_hash not in the live set */
} cache_entry;

/* Scan every <ref_md5>/<AB>/<hash>.bin under cache_dir. live[]/live_k[] are the baked
 * args_hashes of current cases and their K (parallel arrays, nlive long; pass
 * NULL/0 to treat every entry as an orphan). A live entry takes K from live_k[];
 * an orphan from its .meta if present, else 0. Returns a malloc'd array (caller
 * frees) with *n set, or NULL if cache_dir cannot be opened (no cache yet). */
cache_entry *cache_scan(const char *cache_dir,
                        const char *const *live, const long *live_k, int nlive,
                        int *n);

typedef struct {
    long long total_bytes;
    int n_entries;
    int n_orphans;
    long budget_gb;
    long long budget_bytes;
} cache_status;

void cache_compute_status(const cache_entry *e, int n, long budget_gb,
                          cache_status *out);

/* Plan (do_apply=0) or apply (do_apply=1) gc under budget_bytes (a negative
 * budget disables the budget phase, leaving orphan removal). run_start protects
 * entries younger than the run in progress: an entry with mtime >= run_start is
 * never evicted (§9). Apply order: (1) orphans -> unlink .bin AND .meta; (2) if
 * still over budget, live entries cheapest-then-coldest -> unlink .bin only (the
 * .meta survives for calibration); (3) prune emptied <AB> and <ref_md5> dirs.
 * Returns the number of .bin entries evicted; *out_orphans / *out_budget receive
 * the split. Entry sizes are zeroed in place as they are evicted. */
int cache_gc(const char *cache_dir, cache_entry *e, int n, long long budget_bytes,
             long run_start, int do_apply, int *out_orphans, int *out_budget);

#endif /* NB_CACHE_CORE_H */
