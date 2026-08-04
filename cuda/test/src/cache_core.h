/* cache_core.h -- image-cache path assembly, config, atomic writes, scan, gc.
 *
 * The image cache stores rendered REFERENCE images (NBTOOLS-SPEC §9), one per
 * case, machine-global and shared across worktrees. This module owns everything
 * about the cache on disk; the nbcache tool and nbrunsuite drive it.
 *
 * Layout (§9):
 *   <cache-dir>/settings.json                          budget + eviction weights
 *   <cache-dir>/<reference_md5>/<AB>/<args_hash>.bin   image, evictable
 *   <cache-dir>/<reference_md5>/<AB>/<args_hash>.meta  {"cost":{"actual":S}}
 * <AB> = first 2 hex of args_hash (shard). reference_md5 = md5 of the reference
 * binary FILE (a directory level, so reference versions coexist as siblings).
 * args_hash = md5(argkey_canonicalize(reference_args in {input_root}-token form)),
 * the SAME baked key nbgensuite writes and nbrunsuite looks up (§9 invariant).
 * The .meta records the measured render time and survives eviction of its .bin.
 *
 * Location & budget bootstrap from env/flag, NOT the config file (§9): the cache
 * dir is $XDG_CACHE_HOME/nanobragg (default ~/.cache/nanobragg) or --cache-dir;
 * budget_gb resolves flag > <cache-dir>/settings.json > CACHE_DEFAULT_BUDGET_GB.
 *
 * Eviction (§9) fires only under budget pressure and ranks by
 *   score = cost.actual^a / (stale + s0)^b
 * so cheap-and-long-unread images go first; the hashes a run declares are
 * protected. Only the .bin is unlinked. The weights a/b/s0 are compile defaults
 * overridable solely by hand-editing settings.json -- no CLI, never written back.
 *
 * Every write is <name>.tmp.<pid> then rename(2), so a reader never sees a
 * partial file; .bin content is deterministic (racing writers write identical
 * bytes), .meta is harmless last-write-wins.
 */
#ifndef NB_CACHE_CORE_H
#define NB_CACHE_CORE_H

#include <stddef.h>

#define CACHE_DEFAULT_BUDGET_GB 11L
#define CACHE_HEXLEN 32              /* md5 hex digest length (no NUL) */

/* Eviction weight defaults (§9): exponent on cost, exponent on staleness, and
 * the grace floor added to staleness (days in settings.json, seconds here). */
#define CACHE_DEFAULT_COST_WEIGHT 1.8
#define CACHE_DEFAULT_DECAY       1.2
#define CACHE_DEFAULT_GRACE_DAYS  30.0

/* ---- location & config (bootstrap from env/flag, never the config file) ---- */

/* Resolve the cache root: flag_override (non-NULL wins) > $XDG_CACHE_HOME/nanobragg
 * > $HOME/.cache/nanobragg. Returns a malloc'd path the caller frees, or NULL if
 * none can be determined. */
char *cache_dir_resolve(const char *flag_override);

/* budget_gb from <cache_dir>/settings.json, or -1 if the file or key is absent.
 * Checks the file exists (access) before parsing, so an absent settings.json
 * returns -1 rather than surfacing a parser open error. */
long cache_settings_budget_gb(const char *cache_dir);

/* Effective budget (§9): flag_gb (>=0 wins) > settings.json > compile default. */
long cache_budget_resolve(const char *cache_dir, long flag_gb);

/* Set budget_gb in <cache_dir>/settings.json atomically, read-merge-write: every
 * other key in the file (the hand-edited eviction weights) is preserved verbatim.
 * A missing or unparseable file starts from an empty object. 0 ok, -1 err. */
int cache_set_budget_gb(const char *cache_dir, long budget_gb);

/* ---- eviction weights & score ---- */

typedef struct {
    double cost_weight;      /* a -- exponent on cost.actual                    */
    double decay;            /* b -- exponent on staleness                      */
    double grace_seconds;    /* s0 -- grace floor added to staleness            */
} cache_weights;

/* Resolve the weights: compile defaults, overridden by the settings.json keys
 * evict_cost_weight / evict_decay / evict_grace_days (days -> seconds). A key of
 * the wrong JSON type, non-finite, or <= 0 falls back to its own default and
 * leaves the others alone. Always yields a usable set. */
void cache_weights_resolve(const char *cache_dir, cache_weights *out);

/* score = cost_actual^a / (stale + s0)^b, stale = max(0, now - last_use) so a
 * future timestamp from clock skew cannot invert the rank. Low score evicts
 * first; a cost_actual of 0 (no .meta) scores 0. Size-independent (§9). */
double cache_score(double cost_actual, long last_use, long now,
                   const cache_weights *w);

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

/* Record the .meta {"cost":{"actual":S}} beside the entry, atomically. S is the
 * measured render time in seconds, at ~microsecond resolution. */
int cache_write_meta(const char *cache_dir, const char *ref_md5,
                     const char *args_hash, double cost_actual);

/* ---- scan / status / gc ---- */

typedef struct {
    char ref_md5[CACHE_HEXLEN + 1];   /* owning reference dir                   */
    char args_hash[CACHE_HEXLEN + 1]; /* entry key                              */
    long atime;                       /* .bin st_atime -- last use              */
    long mtime;                       /* .bin st_mtime -- ranks under noatime   */
    long long size;                   /* .bin bytes                             */
    double cost_actual;               /* .meta measured render seconds, else 0  */
} cache_entry;

/* Scan every <ref_md5>/<AB>/<hash>.bin under cache_dir, reading cost.actual from
 * each sibling .meta (absent or unparseable -> 0, which ranks lowest).
 * *out_rank_by_mtime is set to 1 when the cache filesystem is mounted noatime --
 * atime never advances there, so gc must rank on mtime (age-since-written); the
 * scan warns once on stderr when it detects this. Returns a malloc'd array
 * (caller frees) with *n set, or NULL if cache_dir cannot be opened. */
cache_entry *cache_scan(const char *cache_dir, int *n, int *out_rank_by_mtime);

typedef struct {
    long long total_bytes;
    int n_entries;
    long budget_gb;
    long long budget_bytes;
} cache_status;

void cache_compute_status(const cache_entry *e, int n, long budget_gb,
                          cache_status *out);

/* Plan (do_apply=0) or apply (do_apply=1) eviction. Evicts only while the total
 * .bin footprint exceeds budget_bytes; under budget, or with a negative budget,
 * nothing is evicted. Candidates are the entries whose args_hash is NOT in
 * protect[0..nprotect) -- the run's declared hashes are never victims -- removed
 * in ascending cache_score() (ties by args_hash then ref_md5, so runs are
 * reproducible) until within budget or exhausted. Only the .bin is unlinked; the
 * .meta survives. rank_by_mtime picks mtime over atime as the last-use timestamp
 * (see cache_scan). Emptied <AB> and <ref_md5> dirs are pruned. An apply zeroes
 * each evicted entry's size in e[]; a plan leaves e[] untouched, so the same
 * array can be handed straight to the applying call. Returns the number of .bin
 * evicted -- the same count either way. */
int cache_gc(const char *cache_dir, cache_entry *e, int n, long long budget_bytes,
             const char *const *protect, int nprotect, const cache_weights *w,
             long now, int rank_by_mtime, int do_apply);

#endif /* NB_CACHE_CORE_H */
