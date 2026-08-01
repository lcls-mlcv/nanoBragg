/* case_core.c -- shared base.json + suites/<name>.jsonl parser. See case_core.h. */
#define _XOPEN_SOURCE 700   /* realpath(path, NULL), POSIX dirname(), getline() under -std=c11 */
#include "case_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <libgen.h>
#include <json-c/json.h>

/* ---- small helpers ------------------------------------------------------ */

/* Field of an object, or NULL if o is not an object or lacks the key. */
static struct json_object *jget(struct json_object *o, const char *key) {
    struct json_object *v = NULL;
    if (o && json_object_is_type(o, json_type_object) &&
        json_object_object_get_ex(o, key, &v))
        return v;
    return NULL;
}
static int is_num(const struct json_object *v) {
    enum json_type t = json_object_get_type((struct json_object *)v);
    return v && (t == json_type_int || t == json_type_double);
}
static double jnum(struct json_object *v, double dflt) {
    return is_num(v) ? json_object_get_double(v) : dflt;
}
static long jnum_l(struct json_object *v, long dflt) {
    return is_num(v) ? (long)json_object_get_int64(v) : dflt;
}
/* Pointer into the DOM; valid until the owning root is json_object_put(). */
static const char *jstr(struct json_object *v) {
    return (v && json_object_is_type(v, json_type_string)) ? json_object_get_string(v) : NULL;
}

/* Parse a .jsonl file into a json-c array (one object per non-blank line);
   json-c has no JSONL reader, so read line-by-line and parse each. Returns NULL
   if the file cannot be opened. A malformed line is fatal (exit 2). */
static struct json_object *parse_jsonl(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    struct json_object *arr = json_object_new_array();
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    while ((len = getline(&line, &cap, f)) != -1) {
        int only_ws = 1;
        for (ssize_t i = 0; i < len; i++)
            if (!isspace((unsigned char)line[i])) { only_ws = 0; break; }
        if (only_ws) continue;               /* skip blank / whitespace-only lines */
        struct json_object *o = json_tokener_parse(line);
        if (!o) { fprintf(stderr, "case_core: malformed JSONL line in %s\n", path); exit(2); }
        json_object_array_add(arr, o);
    }
    free(line);
    fclose(f);
    return arr;
}

/* Copy a JSON array-of-strings into a freshly malloc'd array of const char* that
   point into the DOM. Sets *out_n. Returns NULL (n=0) if absent/not an array. */
static const char **jstr_array(struct json_object *arr, int *out_n) {
    *out_n = 0;
    if (!arr || !json_object_is_type(arr, json_type_array)) return NULL;
    size_t n = json_object_array_length(arr);
    if (n == 0) return NULL;
    const char **out = (const char **)malloc(n * sizeof(char *));
    if (!out) { fprintf(stderr, "case_core: OOM\n"); exit(2); }
    int k = 0;
    for (size_t i = 0; i < n; i++) {
        struct json_object *e = json_object_array_get_idx(arr, i);
        if (json_object_is_type(e, json_type_string)) out[k++] = json_object_get_string(e);
    }
    *out_n = k;
    return out;
}

/* ---- base.json ---------------------------------------------------------- */

/* base.json lives at <harness_root>/spec/base.json; given its absolute path,
   return a freshly malloc'd absolute path to <harness_root> (dirname twice).
   POSIX dirname() may return a pointer into (or alias) its input buffer, so
   each level is copied out before the buffer is reused/freed. */
static char *harness_root_from_base_path(const char *base_json_abs) {
    char *buf = strdup(base_json_abs);
    if (!buf) { fprintf(stderr, "case_core: OOM\n"); exit(2); }
    char *spec_dir = strdup(dirname(buf));          /* .../cuda/test/spec */
    free(buf);
    if (!spec_dir) { fprintf(stderr, "case_core: OOM\n"); exit(2); }
    char *harness_dir = strdup(dirname(spec_dir));  /* .../cuda/test       */
    free(spec_dir);
    if (!harness_dir) { fprintf(stderr, "case_core: OOM\n"); exit(2); }
    return harness_dir;
}

cc_base *cc_load_base(const char *path) {
    struct json_object *root = json_object_from_file(path);
    if (!root || !json_object_is_type(root, json_type_object)) {
        if (root) json_object_put(root);
        return NULL;
    }

    struct json_object *ir = jget(root, "input_root");
    if (!jstr(ir)) { json_object_put(root); return NULL; }   /* input_root is required */

    /* Anchor the harness root at base.json's own on-disk location -- not the
       process CWD -- so input_root resolution works from any invocation dir. */
    char *base_abs = realpath(path, NULL);
    if (!base_abs) { json_object_put(root); return NULL; }
    char *harness_root = harness_root_from_base_path(base_abs);
    free(base_abs);

    cc_base *b = (cc_base *)calloc(1, sizeof(cc_base));
    if (!b) { fprintf(stderr, "case_core: OOM\n"); exit(2); }
    b->root = root;
    b->input_root = jstr(ir);
    b->harness_root_abs = harness_root;

    struct json_object *gate = jget(root, "gate");
    b->corr_min      = jnum(jget(gate, "corr_min"), 0.9999);
    b->sum_ratio_min = jnum(jget(gate, "sum_ratio_min"), 0.999);
    b->sum_ratio_max = jnum(jget(gate, "sum_ratio_max"), 1.001);

    b->cache_budget_gb = jnum_l(jget(root, "cache_budget_gb"), 20);

    b->reference_fix_branches =
        jstr_array(jget(root, "reference_fix_branches"), &b->n_reference_fix_branches);
    b->base_flags =
        jstr_array(jget(root, "base_flags"), &b->n_base_flags);
    b->base_geometry_dims =
        jstr_array(jget(root, "base_geometry_dims"), &b->n_base_geometry_dims);

    return b;
}

void cc_base_free(cc_base *b) {
    if (!b) return;
    free(b->harness_root_abs);
    free((void *)b->reference_fix_branches);
    free((void *)b->base_flags);
    free((void *)b->base_geometry_dims);
    /* Release the DOM last: every const char* field above pointed into it, so it
       must outlive them (json-c is refcounted; this drops the sole reference). */
    if (b->root) json_object_put(b->root);
    free(b);
}

/* ---- suites/<name>.jsonl ------------------------------------------------ */

/* Map a gate_type literal to the enum; returns dflt for NULL / unknown text. */
static cc_gate_type gate_type_of(const char *gt, cc_gate_type dflt) {
    if (!gt) return dflt;
    if (strcmp(gt, "reject") == 0)   return CC_GATE_REJECT;
    if (strcmp(gt, "perf") == 0)     return CC_GATE_PERF;
    if (strcmp(gt, "absolute") == 0) return CC_GATE_ABSOLUTE;
    return dflt;
}

/* The per-suite gate header (line 1 of a suites/<name>.jsonl, marked by the
   "gate_header" key). It states the suite gate type and any absolute params that
   differ from base; params it omits fall back to base. It is AUTHORITATIVE over
   the old suite-name inference. */
typedef struct {
    int present;
    cc_gate_type type;
    double corr_min, sum_ratio_min, sum_ratio_max;
} suite_header;

static suite_header read_gate_header(struct json_object *o, const cc_base *base) {
    suite_header h;
    h.present = 1;
    h.type = gate_type_of(jstr(jget(o, "gate_type")), CC_GATE_ABSOLUTE);
    h.corr_min      = jnum(jget(o, "corr_min"),      base ? base->corr_min : 0.9999);
    h.sum_ratio_min = jnum(jget(o, "sum_ratio_min"), base ? base->sum_ratio_min : 0.999);
    h.sum_ratio_max = jnum(jget(o, "sum_ratio_max"), base ? base->sum_ratio_max : 1.001);
    return h;
}

cc_suite *cc_load_suite(const char *path, const cc_base *base) {
    struct json_object *root = parse_jsonl(path);
    if (!root || !json_object_is_type(root, json_type_array) ||
        json_object_array_length(root) == 0) {
        if (root) json_object_put(root);
        return NULL;
    }
    int rn = (int)json_object_array_length(root);

    /* Suite gate header: the first line carrying a "gate_header" key. Absent =>
       fall back to base params and suite-name inference (legacy schema). */
    suite_header hdr;
    hdr.present = 0;
    hdr.type = CC_GATE_ABSOLUTE;
    hdr.corr_min      = base ? base->corr_min : 0.9999;
    hdr.sum_ratio_min = base ? base->sum_ratio_min : 0.999;
    hdr.sum_ratio_max = base ? base->sum_ratio_max : 1.001;
    for (int i = 0; i < rn; i++) {
        struct json_object *o = json_object_array_get_idx(root, i);
        if (jget(o, "gate_header")) { hdr = read_gate_header(o, base); break; }
    }

    cc_suite *s = (cc_suite *)calloc(1, sizeof(cc_suite));
    if (!s) { fprintf(stderr, "case_core: OOM\n"); exit(2); }
    s->root = root;
    s->cases = (cc_case *)calloc((size_t)rn, sizeof(cc_case));
    if (!s->cases) { fprintf(stderr, "case_core: OOM\n"); exit(2); }

    int nc = 0;
    for (int i = 0; i < rn; i++) {
        struct json_object *o = json_object_array_get_idx(root, i);
        if (jget(o, "gate_header")) continue;      /* the header is not a case */
        cc_case *c = &s->cases[nc++];

        c->id             = jstr(jget(o, "id"));
        c->suite          = jstr(jget(o, "suite"));
        c->candidate_args = jstr(jget(o, "candidate_args"));
        c->reference_args = jstr(jget(o, "reference_args"));
        c->args_hash      = jstr(jget(o, "args_hash"));

        struct json_object *cost = jget(o, "cost");
        c->K = jnum_l(jget(cost, "compute"), 0);

        /* gate type: per-case override > suite header > suite-name inference */
        const char *cgt = jstr(jget(o, "gate_type"));
        if (cgt)              c->gate_type = gate_type_of(cgt, CC_GATE_ABSOLUTE);
        else if (hdr.present) c->gate_type = hdr.type;
        else                  c->gate_type =
            (c->suite && strcmp(c->suite, "perf") == 0) ? CC_GATE_PERF : CC_GATE_ABSOLUTE;

        /* effective absolute params: per-case "gate" override > header > base */
        struct json_object *g = jget(o, "gate");
        c->has_gate_override = (g && json_object_is_type(g, json_type_object)) ? 1 : 0;
        c->corr_min      = jnum(jget(g, "corr_min"),      hdr.corr_min);
        c->sum_ratio_min = jnum(jget(g, "sum_ratio_min"), hdr.sum_ratio_min);
        c->sum_ratio_max = jnum(jget(g, "sum_ratio_max"), hdr.sum_ratio_max);
    }
    s->n = nc;
    return s;
}

void cc_suite_free(cc_suite *s) {
    if (!s) return;
    free(s->cases);
    /* Release the DOM last: cc_case string fields pointed into it (see
       cc_base_free). json_object_put frees the array and all its elements. */
    if (s->root) json_object_put(s->root);
    free(s);
}

/* ---- token resolution --------------------------------------------------- */

char *cc_resolve_input_root(const char *args, const char *input_root_abs) {
    if (!args) return NULL;
    const char *tok = "{input_root}";
    size_t toklen = strlen(tok);
    size_t rep = input_root_abs ? strlen(input_root_abs) : 0;

    /* count occurrences to size the output exactly */
    size_t count = 0;
    for (const char *p = strstr(args, tok); p; p = strstr(p + toklen, tok)) count++;

    size_t out_sz = strlen(args) + count * (rep > toklen ? rep - toklen : 0) + 1;
    char *out = (char *)malloc(out_sz);
    if (!out) { fprintf(stderr, "case_core: OOM\n"); exit(2); }

    char *w = out;
    const char *r = args;
    const char *p;
    while ((p = strstr(r, tok)) != NULL) {
        size_t lead = (size_t)(p - r);
        memcpy(w, r, lead); w += lead;
        if (rep) { memcpy(w, input_root_abs, rep); w += rep; }
        r = p + toklen;
    }
    strcpy(w, r);
    return out;
}

char *cc_input_root_abs(const cc_base *base) {
    if (!base || !base->input_root || !base->harness_root_abs) return NULL;

    /* Join harness_root_abs + "/" + input_root, then realpath() to canonicalize
       and confirm existence. Both operands are already fixed strings (not CWD),
       so the result no longer depends on the caller's working directory. */
    size_t hlen = strlen(base->harness_root_abs);
    size_t ilen = strlen(base->input_root);
    char *joined = (char *)malloc(hlen + 1 + ilen + 1);
    if (!joined) { fprintf(stderr, "case_core: OOM\n"); exit(2); }
    memcpy(joined, base->harness_root_abs, hlen);
    joined[hlen] = '/';
    memcpy(joined + hlen + 1, base->input_root, ilen);
    joined[hlen + 1 + ilen] = '\0';

    char *out = realpath(joined, NULL);   /* NULL => malloc'd result */
    free(joined);
    return out;
}
