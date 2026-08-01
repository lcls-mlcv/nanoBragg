/* cache_core.c -- image-cache path assembly, config, atomic writes, scan, gc.
 * See cache_core.h. */
#define _XOPEN_SOURCE 700   /* realpath / strdup under -std=c11 */
#include "cache_core.h"
#include "argkey.h"
#include <json-c/json.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <md5.h>

/* ========================================================================= */
/* small helpers                                                             */
/* ========================================================================= */

/* First `len` chars of s are all lowercase-hex. */
static int is_hex(const char *s, int len) {
    for (int i = 0; i < len; i++) {
        char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 0;
    }
    return 1;
}

/* strictly: exactly `len` lowercase-hex chars followed by NUL. */
static int is_hex_exact(const char *s, int len) {
    if ((int)strlen(s) != len) return 0;
    return is_hex(s, len);
}

/* Create every component of `dir` (mkdir -p). Ignores EEXIST. 0 ok, -1 err. */
static int mkdirs(const char *dir) {
    char buf[4096];
    size_t n = strlen(dir);
    if (n >= sizeof buf) return -1;
    memcpy(buf, dir, n + 1);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

/* ========================================================================= */
/* location & config                                                         */
/* ========================================================================= */

char *cache_dir_resolve(const char *flag_override) {
    if (flag_override && flag_override[0]) return strdup(flag_override);

    const char *xdg = getenv("XDG_CACHE_HOME");
    char buf[4096];
    if (xdg && xdg[0]) {
        snprintf(buf, sizeof buf, "%s/nanobragg", xdg);
        return strdup(buf);
    }
    const char *home = getenv("HOME");
    if (home && home[0]) {
        snprintf(buf, sizeof buf, "%s/.cache/nanobragg", home);
        return strdup(buf);
    }
    return NULL;
}

long cache_settings_budget_gb(const char *cache_dir) {
    char path[4096];
    snprintf(path, sizeof path, "%s/settings.json", cache_dir);
    if (access(path, R_OK) != 0) return -1;          /* absent settings.json */
    struct json_object *root = json_object_from_file(path);
    if (!root || !json_object_is_type(root, json_type_object)) {
        if (root) json_object_put(root);
        return -1;
    }
    struct json_object *bv = NULL;
    long budget = -1;
    if (json_object_object_get_ex(root, "budget_gb", &bv) && bv) {
        enum json_type t = json_object_get_type(bv);
        if (t == json_type_int || t == json_type_double)
            budget = (long)json_object_get_int64(bv);
    }
    json_object_put(root);
    return budget;
}

long cache_budget_resolve(const char *cache_dir, long flag_gb) {
    if (flag_gb >= 0) return flag_gb;
    long s = cache_settings_budget_gb(cache_dir);
    if (s >= 0) return s;
    return CACHE_DEFAULT_BUDGET_GB;
}

int cache_set_budget_gb(const char *cache_dir, long budget_gb) {
    if (mkdirs(cache_dir) != 0) return -1;
    char path[4096], json[128];
    snprintf(path, sizeof path, "%s/settings.json", cache_dir);
    int len = snprintf(json, sizeof json, "{\"budget_gb\":%ld}\n", budget_gb);
    return cache_write_atomic(path, json, (size_t)len);
}

/* ========================================================================= */
/* keys & path assembly                                                      */
/* ========================================================================= */

int cache_reference_md5(const char *ref_path, char out[33]) {
    return MD5File(ref_path, out) ? 0 : -1;   /* streams the file; 32 hex + NUL */
}

int cache_args_hash(char **tok, int n, char out[33]) {
    char *canon = argkey_canonicalize(tok, n);
    if (!canon) return -1;
    MD5Data((const unsigned char *)canon, strlen(canon), out);
    free(canon);
    return 0;
}

char *cache_entry_path(const char *cache_dir, const char *ref_md5,
                       const char *args_hash, const char *ext,
                       char *out, size_t cap) {
    if (!is_hex_exact(args_hash, CACHE_HEXLEN)) return NULL;
    int len = snprintf(out, cap, "%s/%s/%c%c/%s%s",
                       cache_dir, ref_md5, args_hash[0], args_hash[1],
                       args_hash, ext);
    if (len < 0 || (size_t)len >= cap) return NULL;
    return out;
}

/* ========================================================================= */
/* atomic writes                                                             */
/* ========================================================================= */

int cache_write_atomic(const char *path, const void *buf, size_t len) {
    /* create the parent directory chain */
    char parent[4096];
    size_t pn = strlen(path);
    if (pn >= sizeof parent) return -1;
    memcpy(parent, path, pn + 1);
    char *slash = strrchr(parent, '/');
    if (slash) { *slash = 0; if (parent[0] && mkdirs(parent) != 0) return -1; }

    char tmp[4200];
    if ((size_t)snprintf(tmp, sizeof tmp, "%s.tmp.%d", path, (int)getpid()) >= sizeof tmp)
        return -1;

    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    const char *p = (const char *)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, p + off, len - off);
        if (w < 0) { close(fd); unlink(tmp); return -1; }
        off += (size_t)w;
    }
    if (close(fd) != 0) { unlink(tmp); return -1; }
    if (rename(tmp, path) != 0) { unlink(tmp); return -1; }
    return 0;
}

int cache_write_meta(const char *cache_dir, const char *ref_md5,
                     const char *args_hash, long k, double actual_seconds) {
    char path[4096];
    if (!cache_entry_path(cache_dir, ref_md5, args_hash, ".meta", path, sizeof path))
        return -1;
    char json[128];
    int len = snprintf(json, sizeof json,
                       "{\"k\":%ld,\"actual_seconds\":%.6f}\n", k, actual_seconds);
    return cache_write_atomic(path, json, (size_t)len);
}

/* ========================================================================= */
/* scan / status / gc                                                        */
/* ========================================================================= */

/* Live-set lookup: returns the index into live[] or -1 (orphan). Linear -- the
 * live set is the full case corpus (a few hundred) scanned once per entry. */
static int live_index(const char *const *live, int nlive, const char *hash) {
    for (int i = 0; i < nlive; i++)
        if (strcmp(live[i], hash) == 0) return i;
    return -1;
}

/* Read "k" from a .meta (our fixed format), or 0 if unreadable. Leak-free (no
 * DOM): the recorder writes {"k":N,"actual_seconds":F}. */
static long meta_k(const char *meta_path) {
    FILE *f = fopen(meta_path, "r");
    if (!f) return 0;
    char buf[256];
    size_t got = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[got] = 0;
    const char *p = strstr(buf, "\"k\":");
    if (!p) return 0;
    return strtol(p + 4, NULL, 10);
}

cache_entry *cache_scan(const char *cache_dir,
                        const char *const *live, const long *live_k, int nlive,
                        int *n) {
    DIR *root = opendir(cache_dir);
    if (!root) { *n = 0; return NULL; }

    int cap = 256, cnt = 0;
    cache_entry *ents = malloc((size_t)cap * sizeof *ents);
    if (!ents) { closedir(root); *n = 0; return NULL; }

    struct dirent *rd;
    while ((rd = readdir(root))) {
        if (!is_hex_exact(rd->d_name, CACHE_HEXLEN)) continue;  /* a reference_md5 dir */
        char refdir[4096];
        snprintf(refdir, sizeof refdir, "%s/%s", cache_dir, rd->d_name);
        DIR *rdd = opendir(refdir);
        if (!rdd) continue;
        struct dirent *sd;
        while ((sd = readdir(rdd))) {
            if (strlen(sd->d_name) != 2 || !is_hex(sd->d_name, 2)) continue; /* AB shard */
            char shard[4096];
            /* assemble from cache_dir (unbounded param) so the fixed-array source
             * does not trip -Wformat-truncation */
            snprintf(shard, sizeof shard, "%s/%s/%s", cache_dir, rd->d_name, sd->d_name);
            DIR *sdd = opendir(shard);
            if (!sdd) continue;
            struct dirent *bd;
            while ((bd = readdir(sdd))) {
                size_t nl = strlen(bd->d_name);
                if (nl != CACHE_HEXLEN + 4 || strcmp(bd->d_name + CACHE_HEXLEN, ".bin"))
                    continue;
                char hash[CACHE_HEXLEN + 1];
                memcpy(hash, bd->d_name, CACHE_HEXLEN);
                hash[CACHE_HEXLEN] = 0;
                if (!is_hex(hash, CACHE_HEXLEN)) continue;

                char binp[4096];
                snprintf(binp, sizeof binp, "%s/%s/%s/%s",
                         cache_dir, rd->d_name, sd->d_name, bd->d_name);
                struct stat st;
                if (stat(binp, &st) != 0) continue;

                if (cnt == cap) { cap *= 2; ents = realloc(ents, (size_t)cap * sizeof *ents); }
                cache_entry *e = &ents[cnt++];
                memcpy(e->ref_md5, rd->d_name, CACHE_HEXLEN + 1);
                memcpy(e->args_hash, hash, CACHE_HEXLEN + 1);
                e->mtime = (long)st.st_mtime;
                e->size  = (long long)st.st_size;

                int li = live_index(live, nlive, hash);
                if (li >= 0) {
                    e->orphan = 0;
                    e->k = live_k ? live_k[li] : 0;
                } else {
                    e->orphan = 1;
                    char metap[4096];
                    snprintf(metap, sizeof metap, "%s/%s/%s/%.*s.meta",
                             cache_dir, rd->d_name, sd->d_name, CACHE_HEXLEN, hash);
                    e->k = meta_k(metap);
                }
            }
            closedir(sdd);
        }
        closedir(rdd);
    }
    closedir(root);
    *n = cnt;
    return ents;
}

void cache_compute_status(const cache_entry *e, int n, long budget_gb,
                          cache_status *out) {
    out->total_bytes = 0;
    out->n_entries = n;
    out->n_orphans = 0;
    out->budget_gb = budget_gb;
    out->budget_bytes = (long long)budget_gb << 30;
    for (int i = 0; i < n; i++) {
        out->total_bytes += e[i].size;
        if (e[i].orphan) out->n_orphans++;
    }
}

/* Cheapest-then-coldest: cost (K) ascending is primary so expensive references
 * are the last evicted ("live forever by cost rank", §9); mtime ascending
 * (coldest) breaks ties. */
static int cmp_evict(const void *a, const void *b) {
    const cache_entry *x = *(const cache_entry *const *)a;
    const cache_entry *y = *(const cache_entry *const *)b;
    if (x->k != y->k)         return (x->k < y->k) ? -1 : 1;
    if (x->mtime != y->mtime) return (x->mtime < y->mtime) ? -1 : 1;
    return strcmp(x->args_hash, y->args_hash);   /* stable, deterministic */
}

/* rmdir any emptied <AB> and <ref_md5> dir under cache_dir (rmdir refuses a
 * non-empty dir, so a shard still holding a surviving .meta is left intact). */
static void prune_empty_dirs(const char *cache_dir) {
    DIR *root = opendir(cache_dir);
    if (!root) return;
    struct dirent *rd;
    while ((rd = readdir(root))) {
        if (!is_hex_exact(rd->d_name, CACHE_HEXLEN)) continue;
        char refdir[4096];
        snprintf(refdir, sizeof refdir, "%s/%s", cache_dir, rd->d_name);
        DIR *rdd = opendir(refdir);
        if (rdd) {
            struct dirent *sd;
            while ((sd = readdir(rdd))) {
                if (strlen(sd->d_name) != 2 || !is_hex(sd->d_name, 2)) continue;
                char shard[4096];
                snprintf(shard, sizeof shard, "%s/%s/%s", cache_dir, rd->d_name, sd->d_name);
                rmdir(shard);   /* no-op if non-empty */
            }
            closedir(rdd);
        }
        rmdir(refdir);          /* no-op if any shard survived */
    }
    closedir(root);
}

int cache_gc(const char *cache_dir, cache_entry *e, int n, long long budget_bytes,
             long run_start, int do_apply, int *out_orphans, int *out_budget) {
    /* running total of live .bin bytes still on disk */
    long long total = 0;
    for (int i = 0; i < n; i++) total += e[i].size;

    int evicted_orphan = 0, evicted_budget = 0;
    char path[4096];

    /* Phase 1 -- orphans: unlink .bin AND .meta (the case is gone; no calibration
     * value). Never touch an entry younger than the run in progress. */
    for (int i = 0; i < n; i++) {
        if (!e[i].orphan || e[i].mtime >= run_start) continue;
        if (do_apply) {
            if (cache_entry_path(cache_dir, e[i].ref_md5, e[i].args_hash, ".bin",
                                 path, sizeof path)) unlink(path);
            if (cache_entry_path(cache_dir, e[i].ref_md5, e[i].args_hash, ".meta",
                                 path, sizeof path)) unlink(path);
        }
        total -= e[i].size;
        e[i].size = 0;          /* removed from the on-disk footprint */
        evicted_orphan++;
    }

    /* Phase 2 -- budget: if still over, evict live .bin (keep .meta for the
     * calibration dataset) cheapest-then-coldest until within budget. Protected
     * (young) entries and orphans (already handled) are excluded. */
    if (budget_bytes >= 0 && total > budget_bytes) {
        const cache_entry **victims = malloc((size_t)n * sizeof *victims);
        int nv = 0;
        for (int i = 0; i < n; i++)
            if (!e[i].orphan && e[i].size > 0 && e[i].mtime < run_start)
                victims[nv++] = &e[i];
        qsort(victims, (size_t)nv, sizeof *victims, cmp_evict);
        for (int i = 0; i < nv && total > budget_bytes; i++) {
            cache_entry *v = (cache_entry *)victims[i];
            if (do_apply &&
                cache_entry_path(cache_dir, v->ref_md5, v->args_hash, ".bin",
                                 path, sizeof path))
                unlink(path);
            total -= v->size;
            v->size = 0;
            evicted_budget++;
        }
        free(victims);
    }

    if (do_apply) prune_empty_dirs(cache_dir);

    if (out_orphans) *out_orphans = evicted_orphan;
    if (out_budget)  *out_budget  = evicted_budget;
    return evicted_orphan + evicted_budget;
}
