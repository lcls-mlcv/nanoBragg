/* cache_core.c -- image-cache path assembly, config, atomic writes, scan, gc.
 * See cache_core.h. */
#define _GNU_SOURCE   /* realpath / strdup under -std=c11; ST_NOATIME in statvfs */
#include "cache_core.h"
#include "argkey.h"
#include <json-c/json.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
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

/* <cache_dir>/settings.json as a json-c object, or NULL when it is absent or not
 * a JSON object. Checking access() first keeps an absent file from surfacing a
 * parser open error. Caller json_object_put()s the result. */
static struct json_object *settings_load(const char *cache_dir) {
    char path[4096];
    snprintf(path, sizeof path, "%s/settings.json", cache_dir);
    if (access(path, R_OK) != 0) return NULL;
    struct json_object *root = json_object_from_file(path);
    if (root && !json_object_is_type(root, json_type_object)) {
        json_object_put(root);
        return NULL;
    }
    return root;
}

/* One positive, finite number from settings.json, else `dflt` (§9: a bad value
 * falls back to its own default without disturbing the other keys). */
static double settings_pos_double(struct json_object *root, const char *key,
                                  double dflt) {
    struct json_object *v = NULL;
    if (!root || !json_object_object_get_ex(root, key, &v) || !v) return dflt;
    enum json_type t = json_object_get_type(v);
    if (t != json_type_int && t != json_type_double) return dflt;
    double d = json_object_get_double(v);
    if (!isfinite(d) || d <= 0.0) return dflt;
    return d;
}

long cache_settings_budget_gb(const char *cache_dir) {
    struct json_object *root = settings_load(cache_dir);
    if (!root) return -1;
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

void cache_weights_resolve(const char *cache_dir, cache_weights *out) {
    struct json_object *root = settings_load(cache_dir);
    out->cost_weight = settings_pos_double(root, "evict_cost_weight",
                                           CACHE_DEFAULT_COST_WEIGHT);
    out->decay       = settings_pos_double(root, "evict_decay",
                                           CACHE_DEFAULT_DECAY);
    out->grace_seconds = settings_pos_double(root, "evict_grace_days",
                                             CACHE_DEFAULT_GRACE_DAYS) * 86400.0;
    if (root) json_object_put(root);
}

int cache_set_budget_gb(const char *cache_dir, long budget_gb) {
    if (mkdirs(cache_dir) != 0) return -1;
    char path[4096];
    snprintf(path, sizeof path, "%s/settings.json", cache_dir);

    /* Read-merge-write: the hand-edited eviction weights live in this file and
     * must survive a budget update (§9). */
    struct json_object *root = settings_load(cache_dir);
    if (!root) root = json_object_new_object();
    json_object_object_add(root, "budget_gb", json_object_new_int64(budget_gb));

    const char *txt = json_object_to_json_string_ext(root, JSON_C_TO_STRING_PLAIN);
    char buf[8192];
    int len = snprintf(buf, sizeof buf, "%s\n", txt ? txt : "{}");
    json_object_put(root);
    if (len < 0 || (size_t)len >= sizeof buf) return -1;
    return cache_write_atomic(path, buf, (size_t)len);
}

double cache_score(double cost_actual, long last_use, long now,
                   const cache_weights *w) {
    double stale = (double)(now - last_use);
    if (!(stale > 0.0)) stale = 0.0;
    return pow(cost_actual, w->cost_weight) / pow(stale + w->grace_seconds, w->decay);
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
                     const char *args_hash, double cost_actual) {
    char path[4096];
    if (!cache_entry_path(cache_dir, ref_md5, args_hash, ".meta", path, sizeof path))
        return -1;
    /* Nested to mirror the suite schema's "cost":{"compute":...} shape (§9). */
    char json[128];
    int len = snprintf(json, sizeof json,
                       "{\"cost\":{\"actual\":%.6f}}\n", cost_actual);
    return cache_write_atomic(path, json, (size_t)len);
}

/* ========================================================================= */
/* scan / status / gc                                                        */
/* ========================================================================= */

/* cost.actual from a .meta, or 0 when the file is absent, truncated, or not the
 * {"cost":{"actual":S}} shape -- 0 ranks lowest and evicts first (§9). Scanned
 * once per entry, so it reads the fixed shape directly instead of building a DOM. */
static double meta_cost_actual(const char *meta_path) {
    FILE *f = fopen(meta_path, "r");
    if (!f) return 0.0;
    char buf[256];
    size_t got = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[got] = 0;
    const char *p = strstr(buf, "\"cost\"");
    if (p) p = strstr(p, "\"actual\"");
    if (p) p = strchr(p, ':');
    if (!p) return 0.0;
    char *end = NULL;
    double v = strtod(p + 1, &end);
    if (end == p + 1 || !isfinite(v) || v < 0.0) return 0.0;
    return v;
}

/* 1 when cache_dir sits on a noatime mount, where st_atime never advances. */
static int mount_is_noatime(const char *cache_dir) {
#ifdef ST_NOATIME
    struct statvfs vfs;
    if (statvfs(cache_dir, &vfs) == 0 && (vfs.f_flag & ST_NOATIME)) return 1;
#else
    (void)cache_dir;
#endif
    return 0;
}

cache_entry *cache_scan(const char *cache_dir, int *n, int *out_rank_by_mtime) {
    DIR *root = opendir(cache_dir);
    if (!root) { *n = 0; if (out_rank_by_mtime) *out_rank_by_mtime = 0; return NULL; }

    int rank_by_mtime = mount_is_noatime(cache_dir);
    if (rank_by_mtime)
        fprintf(stderr, "# cache: %s is mounted noatime -- eviction ranks by "
                        "age-since-written instead of last use\n", cache_dir);
    if (out_rank_by_mtime) *out_rank_by_mtime = rank_by_mtime;

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
                e->atime = (long)st.st_atime;
                e->mtime = (long)st.st_mtime;
                e->size  = (long long)st.st_size;

                char metap[4096];
                snprintf(metap, sizeof metap, "%s/%s/%s/%.*s.meta",
                         cache_dir, rd->d_name, sd->d_name, CACHE_HEXLEN, hash);
                e->cost_actual = meta_cost_actual(metap);
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
    out->budget_gb = budget_gb;
    out->budget_bytes = (long long)budget_gb << 30;
    for (int i = 0; i < n; i++) out->total_bytes += e[i].size;
}

/* An eviction candidate with its score resolved once, so the sort is a pure
 * numeric comparison. */
typedef struct { double score; cache_entry *e; } victim;

/* Ascending score -- cheap and long unread first (§9); args_hash then ref_md5
 * break ties so an eviction order is reproducible across runs. The same
 * args_hash can exist under two reference dirs, which qsort (unstable) would
 * otherwise order arbitrarily. */
static int cmp_victim(const void *a, const void *b) {
    const victim *x = (const victim *)a, *y = (const victim *)b;
    if (x->score != y->score) return (x->score < y->score) ? -1 : 1;
    int c = strcmp(x->e->args_hash, y->e->args_hash);
    if (c) return c;
    return strcmp(x->e->ref_md5, y->e->ref_md5);
}

/* args_hash is one of the run's declared (never-evicted) hashes. Linear -- the
 * protect set is one suite's case corpus, a few hundred, per entry. */
static int is_protected(const char *const *protect, int nprotect, const char *hash) {
    for (int i = 0; i < nprotect; i++)
        if (strcmp(protect[i], hash) == 0) return 1;
    return 0;
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
             const char *const *protect, int nprotect, const cache_weights *w,
             long now, int rank_by_mtime, int do_apply) {
    long long total = 0;
    for (int i = 0; i < n; i++) total += e[i].size;

    int evicted = 0;
    /* Under budget the cache is left alone; only the dir prune runs (§9). */
    if (budget_bytes >= 0 && total > budget_bytes && n > 0) {
        victim *v = malloc((size_t)n * sizeof *v);
        int nv = 0;
        for (int i = 0; i < n; i++) {
            if (e[i].size <= 0 || is_protected(protect, nprotect, e[i].args_hash))
                continue;
            long last_use = rank_by_mtime ? e[i].mtime : e[i].atime;
            v[nv].score = cache_score(e[i].cost_actual, last_use, now, w);
            v[nv].e = &e[i];
            nv++;
        }
        qsort(v, (size_t)nv, sizeof *v, cmp_victim);

        char path[4096];
        for (int i = 0; i < nv && total > budget_bytes; i++) {
            cache_entry *ve = v[i].e;
            /* The .meta survives: the measured render time outlives its image. */
            if (do_apply &&
                cache_entry_path(cache_dir, ve->ref_md5, ve->args_hash, ".bin",
                                 path, sizeof path))
                unlink(path);
            total -= ve->size;
            /* Only an apply rewrites the caller's array; a plan leaves it intact
             * so the same entries can be replayed into the applying call. */
            if (do_apply) ve->size = 0;   /* off the on-disk footprint */
            evicted++;
        }
        free(v);
    }

    if (do_apply) prune_empty_dirs(cache_dir);
    return evicted;
}
