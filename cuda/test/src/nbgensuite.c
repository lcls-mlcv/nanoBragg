/* nbgensuite.c -- the parity-suite compiler.
 *
 * Reads the JSON spec (base.json, dimensions.jsonl, groups.json, plan.json) and
 * emits one compiled case per line to <suite>.jsonl, preceded by a single suite
 * gate-header line. A case is a scenario: an exact candidate/reference CLI plus
 * id/axes/tags/cost metadata and a baked args_hash cache key.
 *
 *   Usage: nbgensuite <suite> <spec_dir> [out_file]
 *          out_file defaults to stdout.
 *
 * Output schema (per line):
 *   line 1  gate header  {"gate_header":1,"suite":..,"gate_type":..[,param overrides]}
 *   line 2+ one case     {"id","suite","axes","tags","cost","cpu_class",
 *                         "args_hash","candidate_args","reference_args"}
 *
 * args_hash = md5 (libmd) of argkey_canonicalize() over the reference_args in
 * their {input_root}-token form (NOT expanded), so the cache key is
 * machine-independent. It keys on reference_args because the image cache stores
 * reference images (reference_args differ from candidate_args, e.g. thickness
 * cases append -oversample_thick to the reference side only).
 *
 * The gate is stated once per suite in the header (type + any params that differ
 * from base.json), not per case. Suite gate types: guards -> reject, perf ->
 * perf, everything else -> absolute (base corr_min/sum_ratio range).
 *
 * Suites (from plan.json):
 *   grid320  -- Cartesian cross of crystal x crystal_size x grid320 x orientation
 *               (4x4x5x4 = 320).
 *   coverage -- main-effects: a baseline case, then one case per (dimension,value)
 *               that differs from the dimension baseline (only that dim changed),
 *               plus hand-authored interaction/feature scenarios.
 *   guards   -- hand-authored scenarios the candidate kernel MUST refuse (exit 9).
 *   perf     -- hand-authored parity cases tagged for timing (min-of-5, warn).
 *   pairwise -- hand-authored pairwise-coverage scenarios.
 *
 * No jq / no Python: JSON is parsed with the json-c library. A number valued
 * field that feeds a CLI arg string is read via json_object_get_string(), which
 * returns json-c's retained source text, so "231.27", "1e18", "1.0" round-trip
 * verbatim into the CLI (json-c 0.14 keeps the original literal). The output is
 * built and serialized with json-c too: each line is a json_object emitted with
 * json_object_to_json_string_ext(), one object per line (JSONL). Keys are added
 * in schema order (json-c preserves object insertion order). Slash escaping is
 * disabled (NOSLASHESCAPE) so the {input_root}/... path tokens stay unescaped.
 */
#define _XOPEN_SOURCE 700   /* getline() under -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include <json-c/json.h>
#include "argkey.h"
#include "case_core.h"
#include <md5.h>

static void die(const char *msg) { fprintf(stderr, "nbgensuite: %s\n", msg); exit(2); }

/* strdup of the first n bytes (fatal on OOM). */
static char *xstrndup(const char *p, size_t n) {
    char *r = (char *)malloc(n + 1);
    if (!r) die("OOM");
    memcpy(r, p, n); r[n] = 0; return r;
}

/* ---- json-c read helpers (thin analogs of the old DOM accessors) -------- */

/* Field of an object, or NULL if o is not an object or lacks the key. */
static struct json_object *jget(struct json_object *o, const char *k) {
    struct json_object *v = NULL;
    if (o && json_object_is_type(o, json_type_object) && json_object_object_get_ex(o, k, &v))
        return v;
    return NULL;
}
static struct json_object *jidx(struct json_object *a, int i) {
    return json_object_array_get_idx(a, (size_t)i);
}
static int jlen(struct json_object *a) {
    return a ? (int)json_object_array_length(a) : 0;
}
static int jis_arr(struct json_object *v) { return json_object_is_type(v, json_type_array); }
static int jis_str(struct json_object *v) { return json_object_is_type(v, json_type_string); }
/* Source text of a string OR number field (json-c retains the number literal),
   or NULL. This is the read path that keeps arg numbers byte-exact. */
static const char *js(struct json_object *v) { return v ? json_object_get_string(v) : NULL; }

/* Parse a whole JSON file (one value). Fatal if it cannot be read/parsed. */
static struct json_object *parse_json_file(const char *path) {
    struct json_object *o = json_object_from_file(path);
    if (!o) die("cannot read/parse JSON file");
    return o;
}

/* Parse a .jsonl file into a json-c array (one object per non-blank line):
   json-c has no JSONL reader, so read line-by-line and parse each line. */
static struct json_object *parse_jsonl_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open JSONL file");
    struct json_object *arr = json_object_new_array();
    char *line = NULL; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, f)) != -1) {
        int only_ws = 1;
        for (ssize_t i = 0; i < len; i++)
            if (!isspace((unsigned char)line[i])) { only_ws = 0; break; }
        if (only_ws) continue;
        struct json_object *o = json_tokener_parse(line);
        if (!o) die("malformed JSONL line");
        json_object_array_add(arr, o);
    }
    free(line);
    fclose(f);
    return arr;
}

/* ------------------------------------------------------------ token strings */

/* A growable token list (each token is one argv word). */
typedef struct { char **tok; int n, cap; } toklist;

static void tl_init(toklist *t) { t->n = 0; t->cap = 16; t->tok = (char **)malloc(t->cap * sizeof(char *)); if (!t->tok) die("OOM"); }
static void tl_push(toklist *t, const char *s) {
    if (t->n >= t->cap) { t->cap *= 2; t->tok = (char **)realloc(t->tok, t->cap * sizeof(char *)); if (!t->tok) die("OOM"); }
    t->tok[t->n++] = xstrndup(s, strlen(s));
}

/* A token is a FLAG if it is "-<alpha>..." (not a bare/negative number). */
static int is_flag_tok(const char *s) { return s && s[0] == '-' && isalpha((unsigned char)s[1]); }

/* Is this exact flag token already present in the list? */
static int tl_has(const toklist *t, const char *flag) {
    for (int i = 0; i < t->n; i++) if (strcmp(t->tok[i], flag) == 0) return 1;
    return 0;
}

/* Set an arity-1 flag: replace the value after an existing flag, else append
   "flag val". Lets a later override win over an earlier baseline/group value
   (e.g. a scenario's -pixel overriding the base-geometry -pixel) without the
   duplicate-flag collision that plain appending would produce. */
static void tl_set(toklist *t, const char *flag, const char *val) {
    for (int i = 0; i + 1 < t->n; i++)
        if (strcmp(t->tok[i], flag) == 0) { t->tok[i + 1] = xstrndup(val, strlen(val)); return; }
    tl_push(t, flag); tl_push(t, val);
}

/* Location-agnostic cases: emit the {input_root} token verbatim rather than
   baking a data directory into the case. The token is resolved to an absolute
   path at run time, so a case names no machine-specific location. (input_root is
   retained in the signature for the call sites; it is intentionally not consulted
   here -- the token is copied through unchanged.) */
static char *subst_data_root(const char *src, const char *data_root) {
    (void)data_root;
    size_t n = strlen(src) + 1;
    char *out = (char *)malloc(n); if (!out) die("OOM");
    memcpy(out, src, n);
    return out;
}

/* Append a group's args[] to a token list ({input_root} passed through verbatim). */
static void push_args(toklist *t, struct json_object *group, const char *data_root) {
    struct json_object *args = jget(group, "args");
    if (!jis_arr(args)) return;
    for (int i = 0; i < jlen(args); i++) {
        struct json_object *e = jidx(args, i);
        if (!jis_str(e)) die("args element not a string");
        char *sub = subst_data_root(js(e), data_root);
        tl_push(t, sub);
        free(sub);
    }
}

static void push_extra(toklist *t, struct json_object *group, const char *key) {
    struct json_object *ex = jget(group, key);
    if (!jis_arr(ex)) return;
    for (int i = 0; i < jlen(ex); i++) {
        struct json_object *e = jidx(ex, i);
        if (jis_str(e)) tl_push(t, js(e));
    }
}

/* ---------------------------------------------------------------- spec model */

typedef struct {
    struct json_object *base;     /* base.json object */
    struct json_object *dims;     /* array of dimension objects */
    struct json_object *groups;   /* array of group objects */
    struct json_object *plan;     /* array of plan objects */
    const char *data_root;
} spec;

static struct json_object *dim_by_name(spec *S, const char *name) {
    for (int i = 0; i < jlen(S->dims); i++) {
        struct json_object *d = jidx(S->dims, i);
        struct json_object *dn = jget(d, "dim");
        if (jis_str(dn) && strcmp(js(dn), name) == 0) return d;
    }
    return NULL;
}

/* All groups of a given class, in declaration order. Returns count via *out_n. */
static struct json_object **groups_of_class(spec *S, const char *cls, int *out_n) {
    int n = 0;
    struct json_object **res = (struct json_object **)malloc((size_t)jlen(S->groups) * sizeof(*res));
    if (!res) die("OOM");
    for (int i = 0; i < jlen(S->groups); i++) {
        struct json_object *g = jidx(S->groups, i);
        struct json_object *c = jget(g, "class");
        if (jis_str(c) && strcmp(js(c), cls) == 0) res[n++] = g;
    }
    *out_n = n;
    return res;
}

static const char *group_label(struct json_object *g) {
    struct json_object *l = jget(g, "label");
    return jis_str(l) ? js(l) : "?";
}
static const char *group_name(struct json_object *g) {
    struct json_object *l = jget(g, "group");
    return jis_str(l) ? js(l) : "?";
}

/* Assemble the invariant BASE geometry tokens. */
static void push_base_geometry(spec *S, toklist *t) {
    /* geometry dims: distance, lambda, pixel -> "-<flag> <baseline>" each */
    struct json_object *bgd = jget(S->base, "base_geometry_dims");
    const char *pixel_lit = "0.172";
    if (jis_arr(bgd)) {
        for (int i = 0; i < jlen(bgd); i++) {
            const char *dn = js(jidx(bgd, i));
            struct json_object *d = dim_by_name(S, dn);
            if (!d) die("base_geometry_dim not found in dimensions.jsonl");
            struct json_object *flag = jget(d, "flag");
            struct json_object *bl = jget(d, "baseline");
            if (!flag || !bl) die("geometry dim missing flag/baseline");
            tl_push(t, js(flag));
            tl_push(t, js(bl));
            if (strcmp(dn, "pixel") == 0) pixel_lit = js(bl);
        }
    }
    struct json_object *detp = jget(S->base, "detpixels");
    struct json_object *flux = jget(S->base, "flux");
    struct json_object *beamsz = jget(S->base, "beamsize_mm");
    if (!detp || !flux || !beamsz) die("base.json missing detpixels/flux/beamsize_mm");
    long detpixels = strtol(js(detp), NULL, 10);
    double pixel = strtod(pixel_lit, NULL);
    double beam = (double)detpixels * pixel / 2.0;
    char detbuf[32], beambuf[64];
    snprintf(detbuf, sizeof detbuf, "%ld", detpixels);
    snprintf(beambuf, sizeof beambuf, "%.6f", beam);

    tl_push(t, "-detpixels"); tl_push(t, detbuf);
    tl_push(t, "-Xbeam"); tl_push(t, beambuf);
    tl_push(t, "-Ybeam"); tl_push(t, beambuf);
    tl_push(t, "-flux"); tl_push(t, js(flux));
    tl_push(t, "-beamsize"); tl_push(t, js(beamsz));

    struct json_object *bf = jget(S->base, "base_flags");
    if (jis_arr(bf))
        for (int i = 0; i < jlen(bf); i++) {
            struct json_object *e = jidx(bf, i);
            if (jis_str(e)) tl_push(t, js(e));
        }
}

/* Long value of an int-valued flag in a token list, or dflt if absent. */
static long flag_ival(const toklist *t, const char *flag, long dflt) {
    for (int i = 0; i + 1 < t->n; i++)
        if (strcmp(t->tok[i], flag) == 0) return strtol(t->tok[i + 1], NULL, 10);
    return dflt;
}

/* Flag-collision check: no non-numeric flag token may appear twice. */
static void check_collisions(const toklist *t, const char *cell_id) {
    for (int i = 0; i < t->n; i++) {
        const char *a = t->tok[i];
        if (!(a[0] == '-' && isalpha((unsigned char)a[1]))) continue; /* value, not a flag */
        for (int k = i + 1; k < t->n; k++) {
            if (strcmp(a, t->tok[k]) == 0) {
                fprintf(stderr, "nbgensuite: FLAG COLLISION in case %s: '%s' set twice\n", cell_id, a);
                exit(3);
            }
        }
    }
}

/* --------------------------------------------------------------- serializing */

/* Serialize one json_object as a single JSONL line (object + '\n') and free it.
   NOSLASHESCAPE keeps the {input_root}/... path tokens unescaped, matching the
   argv text; PLAIN keeps the object compact (no spaces/newlines inside). */
static void emit_line(FILE *f, struct json_object *o) {
    fputs(json_object_to_json_string_ext(o, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE), f);
    fputc('\n', f);
    json_object_put(o);
}

/* Join a token list into a single space-separated argv string (malloc'd). No
   JSON escaping here: json-c escapes when the string is serialized. */
static char *argstr_join(const toklist *t) {
    size_t len = 1;   /* NUL */
    for (int i = 0; i < t->n; i++) len += strlen(t->tok[i]) + 1; /* token + space/NUL */
    char *s = (char *)malloc(len);
    if (!s) die("OOM");
    size_t pos = 0;
    for (int i = 0; i < t->n; i++) {
        if (i) s[pos++] = ' ';
        size_t l = strlen(t->tok[i]);
        memcpy(s + pos, t->tok[i], l);
        pos += l;
    }
    s[pos] = 0;
    return s;
}

/* The compute-K: the per-pixel step count
   oversample^2 x dispsteps x mosaic_domains x phisteps x thicksteps. It equals
   cost.compute (K, the wall-time proxy) and predicts CPU-oracle wall time nearly
   linearly. */
static long long compute_k(const toklist *t) {
    long oversample = flag_ival(t, "-oversample", 1);
    long dispsteps  = flag_ival(t, "-dispsteps", 1);
    long mos_dom    = flag_ival(t, "-mosaic_domains", 1);
    long phisteps   = flag_ival(t, "-phisteps", 1);
    long thicksteps = flag_ival(t, "-detector_thicksteps", 1);
    return (long long)oversample * oversample * dispsteps * mos_dom * phisteps * thicksteps;
}

/* K-budget classification into three CPU-oracle tiers. Measured anchors:
   K=6400 ~ 3 min, K=25600 ~ 11 min, K=102400 ~ 44 min, K=409600 ~ 3 h of
   CPU-oracle wall time. Two thresholds split the tiers:
     routine   K <= NB_K_BUDGET
     baked     NB_K_BUDGET < K <= NB_DEATHSTAR_BUDGET
     deathstar K > NB_DEATHSTAR_BUDGET */
#define NB_K_BUDGET 20000LL
#define NB_DEATHSTAR_BUDGET 150000LL
static const char *cpu_class_of(const toklist *t) {
    long long k = compute_k(t);
    if (k > NB_DEATHSTAR_BUDGET) return "deathstar";
    if (k > NB_K_BUDGET)         return "baked";
    return "routine";
}

/* Build the cost 3-vector {compute, precision, memory} from the assembled
   tokens. memory (h_range*k_range*l_range*4) is not derivable from the CLI, so
   it is json null. */
static struct json_object *build_cost(const toklist *t) {
    long Na = flag_ival(t, "-Na", 1), Nb = flag_ival(t, "-Nb", 1), Nc = flag_ival(t, "-Nc", 1);
    long long compute = compute_k(t);
    long long precision = (long long)Na * Nb * Nc;
    struct json_object *o = json_object_new_object();
    json_object_object_add(o, "compute", json_object_new_int64(compute));
    json_object_object_add(o, "precision", json_object_new_int64(precision));
    json_object_object_add(o, "memory", NULL);   /* -> json null */
    return o;
}

/* Bake the args_hash into buf[33]: md5 (libmd) of the canonical reference args
   in their {input_root}-token form. The canonicalization is order-independent,
   so the key is stable; the token form keeps it machine-independent. */
static void compute_args_hash(const toklist *ref, char buf[33]) {
    char *canon = argkey_canonicalize(ref->tok, ref->n);
    if (!canon) die("OOM (argkey)");
    MD5Data((const unsigned char *)canon, strlen(canon), buf);
    free(canon);
}

/* Emit the per-suite gate header (line 1). The gate lives here, not per case:
   base.json default -> this header (states only params that differ from base) ->
   per-case override (rare, absent in the current spec). gate_type is the suite's
   type: reject (guards), perf (perf), or absolute (everything else). Absolute
   suites match base corr_min/sum_ratio so no param keys are written; a suite that
   overrode them would add "corr_min"/"sum_ratio_min"/"sum_ratio_max" here. */
static void emit_gate_header(FILE *f, const char *suite, const char *gate_type) {
    struct json_object *o = json_object_new_object();
    json_object_object_add(o, "gate_header", json_object_new_int(1));
    json_object_object_add(o, "suite", json_object_new_string(suite));
    json_object_object_add(o, "gate_type", json_object_new_string(gate_type));
    emit_line(f, o);
}

/* Emit one full case line. axes_* are label strings; N is the crystal_size
   label (a small integer literal). Keys are added in schema order. The gate is
   not emitted per case (it lives in the suite header). */
static void emit_cell(FILE *f, spec *S, const char *id, const char *suite,
                      const char *ax_crystal, const char *ax_N, const char *ax_regime,
                      const char *ax_orient, const char *tags[], int ntags,
                      const toklist *cand, const toklist *ref) {
    (void)S;
    struct json_object *o = json_object_new_object();
    json_object_object_add(o, "id", json_object_new_string(id));
    json_object_object_add(o, "suite", json_object_new_string(suite));

    struct json_object *axes = json_object_new_object();
    json_object_object_add(axes, "crystal", json_object_new_string(ax_crystal));
    if (ax_N)      json_object_object_add(axes, "N", json_object_new_int((int)strtol(ax_N, NULL, 10)));
    if (ax_regime) json_object_object_add(axes, "regime", json_object_new_string(ax_regime));
    if (ax_orient) json_object_object_add(axes, "orientation", json_object_new_string(ax_orient));
    json_object_object_add(o, "axes", axes);

    struct json_object *jtags = json_object_new_array();
    for (int i = 0; i < ntags; i++) json_object_array_add(jtags, json_object_new_string(tags[i]));
    json_object_object_add(o, "tags", jtags);

    json_object_object_add(o, "cost", build_cost(cand));
    json_object_object_add(o, "cpu_class", json_object_new_string(cpu_class_of(cand)));

    char hash[33];
    compute_args_hash(ref, hash);   /* baked cache key over reference_args */
    json_object_object_add(o, "args_hash", json_object_new_string(hash));

    char *cand_s = argstr_join(cand);
    char *ref_s  = argstr_join(ref);
    json_object_object_add(o, "candidate_args", json_object_new_string(cand_s));
    json_object_object_add(o, "reference_args", json_object_new_string(ref_s));
    free(cand_s);
    free(ref_s);

    emit_line(f, o);
}

/* --------------------------------------------------------------- grid320 */

static int build_grid320(spec *S, FILE *out) {
    int ncr, nsz, nrg, nor;
    struct json_object **cr = groups_of_class(S, "crystal", &ncr);
    struct json_object **sz = groups_of_class(S, "crystal_size", &nsz);
    struct json_object **rg = groups_of_class(S, "grid320", &nrg);
    struct json_object **orr = groups_of_class(S, "orientation", &nor);
    if (ncr == 0 || nsz == 0 || nrg == 0 || nor == 0) die("grid320: a cross class is empty");

    int count = 0;
    /* nesting order crystal -> size(N) -> regime -> orient fixes the global
       index 1..320. */
    for (int a = 0; a < ncr; a++)
    for (int b = 0; b < nsz; b++)
    for (int c = 0; c < nrg; c++)
    for (int d = 0; d < nor; d++) {
        const char *lc = group_label(cr[a]);
        const char *ls = group_label(sz[b]);
        const char *lr = group_label(rg[c]);
        const char *lo = group_label(orr[d]);

        char id[128];
        snprintf(id, sizeof id, "%s_%s_N%s_%s", lc, lr, ls, lo);

        toklist cand; tl_init(&cand);
        push_args(&cand, cr[a], S->data_root);   /* crystal (-hkl -cell / -mat) */
        push_base_geometry(S, &cand);            /* invariant BASE geometry */
        push_args(&cand, sz[b], S->data_root);   /* -Na -Nb -Nc */
        push_args(&cand, rg[c], S->data_root);   /* regime */
        push_args(&cand, orr[d], S->data_root);  /* orientation */

        check_collisions(&cand, id);

        /* reference args = candidate args + any group's cpu_extra (none here) */
        toklist ref; tl_init(&ref);
        for (int i = 0; i < cand.n; i++) tl_push(&ref, cand.tok[i]);
        push_extra(&ref, cr[a], "cpu_extra");
        push_extra(&ref, sz[b], "cpu_extra");
        push_extra(&ref, rg[c], "cpu_extra");
        push_extra(&ref, orr[d], "cpu_extra");

        const char *tags[4] = { group_name(cr[a]), group_name(sz[b]), group_name(rg[c]), group_name(orr[d]) };
        emit_cell(out, S, id, "grid320", lc, ls, lr, lo, tags, 4, &cand, &ref);
        count++;
    }
    free(cr); free(sz); free(rg); free(orr);
    return count;
}

/* --------------------------------------------------------------- coverage */
/* main-effects: baseline case, then one case per (dimension,value != baseline). */

static const char *plan_field(spec *S, const char *suite, const char *field) {
    for (int i = 0; i < jlen(S->plan); i++) {
        struct json_object *p = jidx(S->plan, i);
        struct json_object *sn = jget(p, "suite");
        if (jis_str(sn) && strcmp(js(sn), suite) == 0) {
            struct json_object *fv = jget(p, field);
            return jis_str(fv) ? js(fv) : NULL;
        }
    }
    return NULL;
}

static struct json_object *group_by_name(spec *S, const char *name) {
    for (int i = 0; i < jlen(S->groups); i++) {
        struct json_object *g = jidx(S->groups, i);
        if (strcmp(group_name(g), name) == 0) return g;
    }
    return NULL;
}

/* Does dimension value literal equal the baseline literal? */
static int str_eq(const char *a, const char *b) { return a && b && strcmp(a, b) == 0; }

/* Apply a scenario's explicit args[] to an already-assembled token list.
   A "-flag value" pair whose flag is already present (only base-geometry /
   size flags ever are, and all such are arity-1) OVERRIDES that value in place;
   everything else (new flags, toggles like -curved_det, multi-value flags like
   -cell / -misset) is appended verbatim. This lets a scenario restate a few
   geometry knobs (short lambda, big pixel, off-center beam) on top of the
   invariant base without a duplicate-flag collision. */
static void apply_scenario_args(toklist *t, struct json_object *args, const char *data_root) {
    if (!jis_arr(args)) return;
    int n = jlen(args);
    for (int i = 0; i < n; i++) {
        struct json_object *ai = jidx(args, i);
        if (!jis_str(ai)) die("scenario args element not a string");
        char *cur = subst_data_root(js(ai), data_root);
        struct json_object *anext = (i + 1 < n) ? jidx(args, i + 1) : NULL;
        if (is_flag_tok(cur) && jis_str(anext)) {
            char *nxt = subst_data_root(js(anext), data_root);
            if (!is_flag_tok(nxt) && tl_has(t, cur)) {   /* arity-1 override */
                tl_set(t, cur, nxt);
                free(cur); free(nxt); i++; continue;
            }
            free(nxt);
        }
        tl_push(t, cur);
        free(cur);
    }
}

/* Hand-authored interaction scenarios (presets + overrides). Each plan line
   with a "scenario" field and matching suite compiles to one case: the named
   crystal + invariant base geometry + named size, then the scenario's args
   applied with override semantics. */
static int build_scenarios(spec *S, FILE *out, const char *suite) {
    int count = 0;
    for (int i = 0; i < jlen(S->plan); i++) {
        struct json_object *p = jidx(S->plan, i);
        struct json_object *sn = jget(p, "suite");
        struct json_object *scn = jget(p, "scenario");
        if (!jis_str(sn) || strcmp(js(sn), suite) != 0) continue;
        if (!jis_str(scn)) continue;
        struct json_object *jcr = jget(p, "crystal");
        struct json_object *jsz = jget(p, "size");
        struct json_object *jargs = jget(p, "args");
        if (!jis_str(jcr) || !jis_str(jsz))
            die("scenario missing crystal/size");
        struct json_object *gcr = group_by_name(S, js(jcr));
        struct json_object *gsz = group_by_name(S, js(jsz));
        if (!gcr || !gsz) die("scenario crystal/size group not found");

        toklist cand; tl_init(&cand);
        push_args(&cand, gcr, S->data_root);        /* crystal (may be empty for xtal_none) */
        push_base_geometry(S, &cand);               /* invariant BASE geometry */
        push_args(&cand, gsz, S->data_root);        /* -Na -Nb -Nc */
        apply_scenario_args(&cand, jargs, S->data_root);
        check_collisions(&cand, js(scn));

        toklist ref; tl_init(&ref);
        for (int k = 0; k < cand.n; k++) tl_push(&ref, cand.tok[k]);
        push_extra(&ref, gcr, "cpu_extra");
        push_extra(&ref, gsz, "cpu_extra");
        push_extra(&ref, p,   "cpu_extra");         /* scenario-level cpu_extra */

        struct json_object *jtags = jget(p, "tags");
        const char *tags[16]; int nt = 0;
        if (jis_arr(jtags))
            for (int k = 0; k < jlen(jtags) && nt < 16; k++) {
                struct json_object *te = jidx(jtags, k);
                if (jis_str(te)) tags[nt++] = js(te);
            }

        emit_cell(out, S, js(scn), suite, group_label(gcr), group_label(gsz),
                  NULL, NULL, tags, nt, &cand, &ref);
        count++;
    }
    return count;
}

static int build_coverage(spec *S, FILE *out) {
    const char *bc = plan_field(S, "coverage", "baseline_crystal");
    const char *bs = plan_field(S, "coverage", "baseline_size");
    if (!bc || !bs) die("coverage: plan missing baseline_crystal/baseline_size");
    struct json_object *gcr = group_by_name(S, bc);
    struct json_object *gsz = group_by_name(S, bs);
    if (!gcr || !gsz) die("coverage: baseline crystal/size group not found");
    const char *lc = group_label(gcr);
    const char *ls = group_label(gsz);

    int count = 0;

    /* baseline case */
    {
        toklist cand; tl_init(&cand);
        push_args(&cand, gcr, S->data_root);
        push_base_geometry(S, &cand);
        push_args(&cand, gsz, S->data_root);
        /* Pin oversample (explicit step counts). Without it nanoBragg
           auto-selects it from crystal size -- e.g. N=1000 -> 180x180 subpixels,
           making the CPU oracle take hours. Fixed sampling is common-mode (both
           reference and candidate use it) so parity is unaffected; the oversample
           DIMENSION cases override this to 4/8. */
        tl_push(&cand, "-oversample"); tl_push(&cand, "1");
        check_collisions(&cand, "baseline");
        toklist ref; tl_init(&ref);
        for (int i = 0; i < cand.n; i++) tl_push(&ref, cand.tok[i]);
        const char *tags[2] = { bc, bs };
        emit_cell(out, S, "baseline", "coverage", lc, ls, NULL, NULL, tags, 2, &cand, &ref);
        count++;
    }

    /* one case per (dimension, value != baseline) */
    for (int i = 0; i < jlen(S->dims); i++) {
        struct json_object *d = jidx(S->dims, i);
        struct json_object *dn = jget(d, "dim");
        struct json_object *kind = jget(d, "kind");
        struct json_object *flag = jget(d, "flag");
        struct json_object *baseline = jget(d, "baseline");
        struct json_object *values = jget(d, "values");
        if (!dn || !jis_arr(values)) continue;
        const char *baseline_lit = js(baseline);

        for (int v = 0; v < jlen(values); v++) {
            struct json_object *val = jidx(values, v);
            char idbuf[160];
            toklist cand; tl_init(&cand);
            push_args(&cand, gcr, S->data_root);
            push_base_geometry(S, &cand);
            push_args(&cand, gsz, S->data_root);
            /* pinned oversample (overridden by the oversample dimension via tl_set) */
            tl_push(&cand, "-oversample"); tl_push(&cand, "1");

            if (jis_str(kind) && strcmp(js(kind), "enum") == 0) {
                struct json_object *vv = jget(val, "val");
                struct json_object *vargs = jget(val, "args");
                /* baseline is the enum's default val string */
                if (baseline && vv && str_eq(js(baseline), js(vv))) continue; /* == baseline */
                if (jis_arr(vargs))
                    for (int k = 0; k < jlen(vargs); k++) {
                        struct json_object *e = jidx(vargs, k);
                        if (jis_str(e)) tl_push(&cand, js(e));
                    }
                snprintf(idbuf, sizeof idbuf, "cov_%s_%s", js(dn), vv ? js(vv) : "?");
            } else {
                /* scalar: skip the baseline value. tl_set OVERRIDES a value the
                   base geometry or size group already set (e.g. -Na from the size
                   preset, or -lambda/-pixel from the base geometry) instead of
                   appending a colliding duplicate. */
                if (baseline_lit && str_eq(baseline_lit, js(val))) continue;
                if (jis_str(flag)) {
                    tl_set(&cand, js(flag), js(val));
                    /* beam center follows pixel (base.json beam_center_rule). */
                    if (strcmp(js(flag), "-pixel") == 0) {
                        struct json_object *detp = jget(S->base, "detpixels");
                        long detpixels = detp ? strtol(js(detp), NULL, 10) : 2048;
                        double beam = (double)detpixels * strtod(js(val), NULL) / 2.0;
                        char bb[64]; snprintf(bb, sizeof bb, "%.6f", beam);
                        tl_set(&cand, "-Xbeam", bb); tl_set(&cand, "-Ybeam", bb);
                    }
                }
                snprintf(idbuf, sizeof idbuf, "cov_%s_%s", js(dn), js(val));
            }

            check_collisions(&cand, idbuf);
            toklist ref; tl_init(&ref);
            for (int k = 0; k < cand.n; k++) tl_push(&ref, cand.tok[k]);
            const char *tags[3] = { bc, bs, js(dn) };
            emit_cell(out, S, idbuf, "coverage", lc, ls, NULL, NULL, tags, 3, &cand, &ref);
            count++;
        }
    }

    /* hand-authored interaction + feature scenarios (after the main-effects) */
    count += build_scenarios(S, out, "coverage");
    return count;
}

/* --------------------------------------------------------------------- main */

/* The suite's gate type, keyed on suite name (v1: gate is a per-suite property).
   guards -> reject, perf -> perf, everything else -> the absolute base gate. */
static const char *gate_type_for_suite(const char *suite) {
    if (strcmp(suite, "guards") == 0) return "reject";
    if (strcmp(suite, "perf") == 0)   return "perf";
    return "absolute";
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <suite> <spec_dir> [out_file]\n", argv[0]);
        return 2;
    }
    const char *suite = argv[1];
    const char *specdir = argv[2];
    const char *outpath = (argc >= 4) ? argv[3] : NULL;

    char path[1024];
    spec S; memset(&S, 0, sizeof S);
    /* base.json is read through case_core (the one shared base/suite parser), so
       input_root and the gate defaults come from the same place every tool uses;
       the other base fields are read off the DOM it hands back. */
    snprintf(path, sizeof path, "%s/base.json", specdir);
    cc_base *cb = cc_load_base(path);
    if (!cb) die("base.json: cc_load_base failed (input_root required)");
    S.base = cb->root;
    S.data_root = cb->input_root;

    snprintf(path, sizeof path, "%s/dimensions.jsonl", specdir);S.dims   = parse_jsonl_file(path);
    snprintf(path, sizeof path, "%s/groups.json", specdir);
    { struct json_object *g = parse_json_file(path); struct json_object *garr = jget(g, "groups"); if (!jis_arr(garr)) die("groups.json: no groups array"); S.groups = garr; }
    snprintf(path, sizeof path, "%s/plan.json", specdir);       S.plan   = parse_jsonl_file(path);

    int known = (strcmp(suite, "grid320") == 0 || strcmp(suite, "coverage") == 0 ||
                 strcmp(suite, "guards") == 0  || strcmp(suite, "perf") == 0 ||
                 strcmp(suite, "pairwise") == 0);
    if (!known) { fprintf(stderr, "nbgensuite: unknown suite '%s'\n", suite); return 2; }

    FILE *out = outpath ? fopen(outpath, "wb") : stdout;
    if (!out) { fprintf(stderr, "nbgensuite: cannot write %s\n", outpath); return 2; }

    emit_gate_header(out, suite, gate_type_for_suite(suite));

    int count;
    if (strcmp(suite, "grid320") == 0)       count = build_grid320(&S, out);
    else if (strcmp(suite, "coverage") == 0) count = build_coverage(&S, out);
    else                                     count = build_scenarios(&S, out, suite);

    if (outpath) fclose(out);
    fprintf(stderr, "nbgensuite: suite=%s cases=%d -> %s\n", suite, count, outpath ? outpath : "(stdout)");
    return 0;
}
