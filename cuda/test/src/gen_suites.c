/* gen_suites.c -- the parity-suite compiler.
 *
 * Reads the JSON spec (base.json, dimensions.jsonl, groups.json, plan.json) and
 * emits one compiled cell per line to <suite>.jsonl. A cell is a scenario: an
 * exact GPU/CPU CLI plus id/axes/tags/gate/cost metadata.
 *
 *   Usage: gen_suites <suite> <spec_dir> [out_file]
 *          out_file defaults to stdout.
 *
 * Suites (from plan.json):
 *   grid320  -- Cartesian cross of crystal x crystal_size x grid320 x orientation
 *               (4x4x5x4 = 320). Reproduces run_parity_suite_5090.sh cell-for-cell.
 *   coverage -- main-effects: a baseline cell, then one cell per (dimension,value)
 *               that differs from the dimension baseline (only that dim changed).
 *   guards   -- hand-authored scenarios (build_scenarios) whose plan line carries
 *               "gate":"reject": cells the GPU kernel MUST refuse (exit 9) rather
 *               than silently misbehave. run.sh renders GPU-only, no CPU oracle.
 *   perf     -- hand-authored scenarios (build_scenarios), ordinary parity cells
 *               tagged for timing (min-of-5, warn-not-fail in run.sh).
 *
 * No jq / no Python: JSON is parsed here. Number tokens are preserved verbatim
 * (so "231.27", "1e18", "1.0" round-trip exactly into the CLI).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/* ------------------------------------------------------------------ JSON DOM */

typedef enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ } jtype;

typedef struct jval {
    jtype t;
    int b;                 /* JBOOL */
    char *s;               /* JSTR (unescaped) or JNUM (verbatim literal) */
    struct jval **arr;     /* JARR / JOBJ element values */
    char **keys;           /* JOBJ keys (parallel to arr) */
    int n;                 /* element count */
} jval;

static void die(const char *msg) { fprintf(stderr, "gen_suites: %s\n", msg); exit(2); }

static char *xstrndup(const char *p, size_t n) {
    char *r = (char *)malloc(n + 1);
    if (!r) die("OOM");
    memcpy(r, p, n); r[n] = 0; return r;
}

/* recursive-descent parser over a NUL-terminated buffer */
typedef struct { const char *p; } jparse;

static void jskip(jparse *j) {
    while (*j->p && (isspace((unsigned char)*j->p))) j->p++;
}

static jval *jnew(jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    if (!v) die("OOM");
    v->t = t; return v;
}

static jval *jparse_value(jparse *j);

static char *jparse_str_raw(jparse *j) {
    if (*j->p != '"') die("expected string");
    j->p++;
    /* build unescaped string */
    size_t cap = 16, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) die("OOM");
    while (*j->p && *j->p != '"') {
        char c = *j->p++;
        if (c == '\\') {
            char e = *j->p++;
            switch (e) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case '/': c = '/';  break;
                case '\\': c = '\\'; break;
                case '"': c = '"';  break;
                case 'u': {
                    /* minimal \uXXXX -> only handle ASCII range */
                    char hex[5] = {0};
                    for (int k = 0; k < 4 && *j->p; k++) hex[k] = *j->p++;
                    long cp = strtol(hex, NULL, 16);
                    c = (char)(cp & 0x7f);
                    break;
                }
                default: c = e; break;
            }
        }
        if (len + 1 >= cap) { cap *= 2; out = (char *)realloc(out, cap); if (!out) die("OOM"); }
        out[len++] = c;
    }
    if (*j->p != '"') die("unterminated string");
    j->p++;
    out[len] = 0;
    return out;
}

static jval *jparse_string(jparse *j) {
    jval *v = jnew(JSTR);
    v->s = jparse_str_raw(j);
    return v;
}

static jval *jparse_number(jparse *j) {
    const char *start = j->p;
    if (*j->p == '-' || *j->p == '+') j->p++;
    while (*j->p && (isdigit((unsigned char)*j->p) || *j->p == '.' ||
                     *j->p == 'e' || *j->p == 'E' || *j->p == '+' || *j->p == '-'))
        j->p++;
    jval *v = jnew(JNUM);
    v->s = xstrndup(start, (size_t)(j->p - start)); /* verbatim literal */
    return v;
}

static jval *jparse_array(jparse *j) {
    j->p++; /* [ */
    jval *v = jnew(JARR);
    size_t cap = 8; v->arr = (jval **)malloc(cap * sizeof(jval *));
    if (!v->arr) die("OOM");
    jskip(j);
    if (*j->p == ']') { j->p++; return v; }
    for (;;) {
        jskip(j);
        jval *e = jparse_value(j);
        if ((size_t)v->n >= cap) { cap *= 2; v->arr = (jval **)realloc(v->arr, cap * sizeof(jval *)); if (!v->arr) die("OOM"); }
        v->arr[v->n++] = e;
        jskip(j);
        if (*j->p == ',') { j->p++; continue; }
        if (*j->p == ']') { j->p++; break; }
        die("expected , or ] in array");
    }
    return v;
}

static jval *jparse_object(jparse *j) {
    j->p++; /* { */
    jval *v = jnew(JOBJ);
    size_t cap = 8;
    v->arr = (jval **)malloc(cap * sizeof(jval *));
    v->keys = (char **)malloc(cap * sizeof(char *));
    if (!v->arr || !v->keys) die("OOM");
    jskip(j);
    if (*j->p == '}') { j->p++; return v; }
    for (;;) {
        jskip(j);
        if (*j->p != '"') die("expected key string in object");
        char *key = jparse_str_raw(j);
        jskip(j);
        if (*j->p != ':') die("expected : in object");
        j->p++;
        jskip(j);
        jval *val = jparse_value(j);
        if ((size_t)v->n >= cap) {
            cap *= 2;
            v->arr = (jval **)realloc(v->arr, cap * sizeof(jval *));
            v->keys = (char **)realloc(v->keys, cap * sizeof(char *));
            if (!v->arr || !v->keys) die("OOM");
        }
        v->keys[v->n] = key;
        v->arr[v->n] = val;
        v->n++;
        jskip(j);
        if (*j->p == ',') { j->p++; continue; }
        if (*j->p == '}') { j->p++; break; }
        die("expected , or } in object");
    }
    return v;
}

static jval *jparse_value(jparse *j) {
    jskip(j);
    char c = *j->p;
    if (c == '"') return jparse_string(j);
    if (c == '{') return jparse_object(j);
    if (c == '[') return jparse_array(j);
    if (c == 't') { if (strncmp(j->p, "true", 4)) die("bad literal"); j->p += 4; jval *v = jnew(JBOOL); v->b = 1; return v; }
    if (c == 'f') { if (strncmp(j->p, "false", 5)) die("bad literal"); j->p += 5; jval *v = jnew(JBOOL); v->b = 0; return v; }
    if (c == 'n') { if (strncmp(j->p, "null", 4)) die("bad literal"); j->p += 4; return jnew(JNULL); }
    if (c == '-' || c == '+' || isdigit((unsigned char)c)) return jparse_number(j);
    die("unexpected token");
    return NULL;
}

static jval *jget(const jval *o, const char *key) {
    if (!o || o->t != JOBJ) return NULL;
    for (int i = 0; i < o->n; i++)
        if (strcmp(o->keys[i], key) == 0) return o->arr[i];
    return NULL;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gen_suites: cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) die("OOM");
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) die("short read");
    buf[sz] = 0; fclose(f);
    return buf;
}

static jval *parse_json_file(const char *path) {
    char *buf = read_file(path);
    jparse j = { buf };
    jval *v = jparse_value(&j);
    return v; /* buf intentionally leaked: lives for program duration */
}

/* Parse a .jsonl file into a JARR of objects (one per non-blank line). */
static jval *parse_jsonl_file(const char *path) {
    char *buf = read_file(path);
    jval *arr = jnew(JARR);
    size_t cap = 32; arr->arr = (jval **)malloc(cap * sizeof(jval *));
    if (!arr->arr) die("OOM");
    char *line = buf;
    while (*line) {
        char *nl = strchr(line, '\n');
        size_t len = nl ? (size_t)(nl - line) : strlen(line);
        /* skip blank / whitespace-only lines */
        size_t i = 0; while (i < len && isspace((unsigned char)line[i])) i++;
        if (i < len) {
            char *ln = xstrndup(line, len);
            jparse j = { ln };
            jval *o = jparse_value(&j);
            if ((size_t)arr->n >= cap) { cap *= 2; arr->arr = (jval **)realloc(arr->arr, cap * sizeof(jval *)); if (!arr->arr) die("OOM"); }
            arr->arr[arr->n++] = o;
        }
        if (!nl) break;
        line = nl + 1;
    }
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

/* Location-agnostic cells: emit the {data_root} token verbatim rather than baking
   a data directory into the cell. run.sh resolves {data_root} to an absolute path
   at run time, so a cell names no machine-specific location. (data_root is retained
   in the signature for the call sites; it is intentionally not consulted here.) */
static char *subst_data_root(const char *src, const char *data_root) {
    (void)data_root;
    size_t n = strlen(src) + 1;
    char *out = (char *)malloc(n); if (!out) die("OOM");
    memcpy(out, src, n);
    return out;
}

/* Append a group's args[] to a token list ({data_root} passed through verbatim). */
static void push_args(toklist *t, const jval *group, const char *data_root) {
    jval *args = jget(group, "args");
    if (!args || args->t != JARR) return;
    for (int i = 0; i < args->n; i++) {
        if (args->arr[i]->t != JSTR) die("args element not a string");
        char *sub = subst_data_root(args->arr[i]->s, data_root);
        tl_push(t, sub);
        free(sub);
    }
}

static void push_extra(toklist *t, const jval *group, const char *key) {
    jval *ex = jget(group, key);
    if (!ex || ex->t != JARR) return;
    for (int i = 0; i < ex->n; i++)
        if (ex->arr[i]->t == JSTR) tl_push(t, ex->arr[i]->s);
}

/* ---------------------------------------------------------------- spec model */

typedef struct {
    jval *base;     /* base.json object */
    jval *dims;     /* JARR of dimension objects */
    jval *groups;   /* JARR of group objects */
    jval *plan;     /* JARR of plan objects */
    const char *data_root;
} spec;

static jval *dim_by_name(spec *S, const char *name) {
    for (int i = 0; i < S->dims->n; i++) {
        jval *d = S->dims->arr[i];
        jval *dn = jget(d, "dim");
        if (dn && dn->t == JSTR && strcmp(dn->s, name) == 0) return d;
    }
    return NULL;
}

/* All groups of a given class, in declaration order. Returns count via *out_n. */
static jval **groups_of_class(spec *S, const char *cls, int *out_n) {
    int n = 0;
    jval **res = (jval **)malloc(S->groups->n * sizeof(jval *));
    if (!res) die("OOM");
    for (int i = 0; i < S->groups->n; i++) {
        jval *g = S->groups->arr[i];
        jval *c = jget(g, "class");
        if (c && c->t == JSTR && strcmp(c->s, cls) == 0) res[n++] = g;
    }
    *out_n = n;
    return res;
}

static const char *group_label(const jval *g) {
    jval *l = jget(g, "label");
    return (l && l->t == JSTR) ? l->s : "?";
}
static const char *group_name(const jval *g) {
    jval *l = jget(g, "group");
    return (l && l->t == JSTR) ? l->s : "?";
}

/* Assemble the invariant BASE geometry tokens (matches run_parity_suite_5090.sh). */
static void push_base_geometry(spec *S, toklist *t) {
    /* geometry dims: distance, lambda, pixel -> "-<flag> <baseline>" each */
    jval *bgd = jget(S->base, "base_geometry_dims");
    const char *pixel_lit = "0.172";
    if (bgd && bgd->t == JARR) {
        for (int i = 0; i < bgd->n; i++) {
            const char *dn = bgd->arr[i]->s;
            jval *d = dim_by_name(S, dn);
            if (!d) die("base_geometry_dim not found in dimensions.jsonl");
            jval *flag = jget(d, "flag");
            jval *bl = jget(d, "baseline");
            if (!flag || !bl) die("geometry dim missing flag/baseline");
            tl_push(t, flag->s);
            tl_push(t, bl->s);
            if (strcmp(dn, "pixel") == 0) pixel_lit = bl->s;
        }
    }
    jval *detp = jget(S->base, "detpixels");
    jval *flux = jget(S->base, "flux");
    jval *beamsz = jget(S->base, "beamsize_mm");
    if (!detp || !flux || !beamsz) die("base.json missing detpixels/flux/beamsize_mm");
    long detpixels = strtol(detp->s, NULL, 10);
    double pixel = strtod(pixel_lit, NULL);
    double beam = (double)detpixels * pixel / 2.0;
    char detbuf[32], beambuf[64];
    snprintf(detbuf, sizeof detbuf, "%ld", detpixels);
    snprintf(beambuf, sizeof beambuf, "%.6f", beam);

    tl_push(t, "-detpixels"); tl_push(t, detbuf);
    tl_push(t, "-Xbeam"); tl_push(t, beambuf);
    tl_push(t, "-Ybeam"); tl_push(t, beambuf);
    tl_push(t, "-flux"); tl_push(t, flux->s);
    tl_push(t, "-beamsize"); tl_push(t, beamsz->s);

    jval *bf = jget(S->base, "base_flags");
    if (bf && bf->t == JARR)
        for (int i = 0; i < bf->n; i++)
            if (bf->arr[i]->t == JSTR) tl_push(t, bf->arr[i]->s);
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
                fprintf(stderr, "gen_suites: FLAG COLLISION in cell %s: '%s' set twice\n", cell_id, a);
                exit(3);
            }
        }
    }
}

/* --------------------------------------------------------------- serializing */

static void json_puts_escaped(FILE *f, const char *s) {
    fputc('"', f);
    for (const char *p = s; *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p == '\n') fputs("\\n", f);
        else if (*p == '\t') fputs("\\t", f);
        else fputc(*p, f);
    }
    fputc('"', f);
}

/* Join a token list into a single space-separated string, escaped as JSON. */
static void json_put_argstr(FILE *f, const toklist *t) {
    fputc('"', f);
    for (int i = 0; i < t->n; i++) {
        if (i) fputc(' ', f);
        for (const char *p = t->tok[i]; *p; p++) {
            if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
            else fputc(*p, f);
        }
    }
    fputc('"', f);
}

/* Emit the gate object (verbatim numeric literals from base.json). */
static void json_put_gate(FILE *f, spec *S) {
    jval *g = jget(S->base, "gate");
    jval *cmin = jget(g, "corr_min");
    jval *smin = jget(g, "sum_ratio_min");
    jval *smax = jget(g, "sum_ratio_max");
    fprintf(f, "{\"corr_min\":%s,\"sum_ratio_min\":%s,\"sum_ratio_max\":%s}",
            cmin ? cmin->s : "0.9999", smin ? smin->s : "0.999", smax ? smax->s : "1.001");
}

/* The compute-K: the per-pixel step count
   oversample^2 x dispsteps x mosaic_domains x phisteps x thicksteps. It equals
   cost.compute and predicts CPU-oracle wall time nearly linearly. */
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
     routine   K <= NB_K_BUDGET                     -- oracle auto-generated on
                                                        demand (caps ~10 min/cell).
     baked     NB_K_BUDGET < K <= NB_DEATHSTAR_BUDGET
                                                    -- oracle generated once in bake
                                                       mode (NB_BAKE=1), frozen, its
                                                       verdict committed to golden;
                                                       never auto-generated routinely.
     deathstar K > NB_DEATHSTAR_BUDGET              -- the hours-long, box-pinning
                                                       extreme; generated ONLY under
                                                       an explicit NB_DEATHSTAR=1,
                                                       frozen forever. The ~150000
                                                       line sits near ~1 h of oracle
                                                       time (baked->deathstar). */
#define NB_K_BUDGET 20000LL
#define NB_DEATHSTAR_BUDGET 150000LL
static const char *cpu_class_of(const toklist *t) {
    long long k = compute_k(t);
    if (k > NB_DEATHSTAR_BUDGET) return "deathstar";
    if (k > NB_K_BUDGET)         return "baked";
    return "routine";
}

/* Compute + emit the cost 3-vector from the assembled tokens. */
static void json_put_cost(FILE *f, const toklist *t) {
    long Na = flag_ival(t, "-Na", 1), Nb = flag_ival(t, "-Nb", 1), Nc = flag_ival(t, "-Nc", 1);
    long long compute = compute_k(t);
    long long precision = (long long)Na * Nb * Nc;
    /* memory (h_range*k_range*l_range*4) is not derivable from the CLI -> null */
    fprintf(f, "{\"compute\":%lld,\"precision\":%lld,\"memory\":null}", compute, precision);
}

/* Emit one full cell line. axes_* are label strings; N is the crystal_size label.
   gate_type is NULL/"parity" for an ordinary parity cell (emits nothing extra,
   keeping grid320/coverage byte-identical) or "reject" for a guards cell (emits
   an explicit "gate_type":"reject" field; run.sh routes it to the reject-gate
   path instead of the CPU-oracle parity path). */
static void emit_cell(FILE *f, spec *S, const char *id, const char *suite,
                      const char *ax_crystal, const char *ax_N, const char *ax_regime,
                      const char *ax_orient, const char *tags[], int ntags,
                      const toklist *gpu, const toklist *cpu, const char *gate_type) {
    fputc('{', f);
    fprintf(f, "\"id\":"); json_puts_escaped(f, id);
    fprintf(f, ",\"suite\":"); json_puts_escaped(f, suite);
    fprintf(f, ",\"axes\":{");
    fprintf(f, "\"crystal\":"); json_puts_escaped(f, ax_crystal);
    if (ax_N)      { fprintf(f, ",\"N\":%s", ax_N); }
    if (ax_regime) { fprintf(f, ",\"regime\":");   json_puts_escaped(f, ax_regime); }
    if (ax_orient) { fprintf(f, ",\"orientation\":"); json_puts_escaped(f, ax_orient); }
    fputc('}', f);
    fprintf(f, ",\"tags\":[");
    for (int i = 0; i < ntags; i++) { if (i) fputc(',', f); json_puts_escaped(f, tags[i]); }
    fputc(']', f);
    fprintf(f, ",\"cost\":"); json_put_cost(f, gpu);
    fprintf(f, ",\"cpu_class\":"); json_puts_escaped(f, cpu_class_of(gpu));
    fprintf(f, ",\"gate\":"); json_put_gate(f, S);
    fprintf(f, ",\"gpu_args\":"); json_put_argstr(f, gpu);
    fprintf(f, ",\"cpu_args\":"); json_put_argstr(f, cpu);
    if (gate_type && strcmp(gate_type, "reject") == 0)
        fprintf(f, ",\"gate_type\":\"reject\"");
    fputc('}', f);
    fputc('\n', f);
}

/* --------------------------------------------------------------- grid320 */

static int build_grid320(spec *S, FILE *out) {
    int ncr, nsz, nrg, nor;
    jval **cr = groups_of_class(S, "crystal", &ncr);
    jval **sz = groups_of_class(S, "crystal_size", &nsz);
    jval **rg = groups_of_class(S, "grid320", &nrg);
    jval **orr = groups_of_class(S, "orientation", &nor);
    if (ncr == 0 || nsz == 0 || nrg == 0 || nor == 0) die("grid320: a cross class is empty");

    int count = 0;
    /* nesting order crystal -> size(N) -> regime -> orient reproduces the
       run_parity_suite_5090.sh global index 1..320. */
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

        toklist gpu; tl_init(&gpu);
        push_args(&gpu, cr[a], S->data_root);   /* crystal (-hkl -cell / -mat) */
        push_base_geometry(S, &gpu);            /* invariant BASE geometry */
        push_args(&gpu, sz[b], S->data_root);   /* -Na -Nb -Nc */
        push_args(&gpu, rg[c], S->data_root);   /* regime */
        push_args(&gpu, orr[d], S->data_root);  /* orientation */

        check_collisions(&gpu, id);

        /* cpu args = gpu args + any group's cpu_extra (none for grid320 classes) */
        toklist cpu; tl_init(&cpu);
        for (int i = 0; i < gpu.n; i++) tl_push(&cpu, gpu.tok[i]);
        push_extra(&cpu, cr[a], "cpu_extra");
        push_extra(&cpu, sz[b], "cpu_extra");
        push_extra(&cpu, rg[c], "cpu_extra");
        push_extra(&cpu, orr[d], "cpu_extra");

        const char *tags[4] = { group_name(cr[a]), group_name(sz[b]), group_name(rg[c]), group_name(orr[d]) };
        emit_cell(out, S, id, "grid320", lc, ls, lr, lo, tags, 4, &gpu, &cpu, NULL);
        count++;
    }
    free(cr); free(sz); free(rg); free(orr);
    return count;
}

/* --------------------------------------------------------------- coverage */
/* main-effects: baseline cell, then one cell per (dimension,value != baseline). */

static const char *plan_field(spec *S, const char *suite, const char *field) {
    for (int i = 0; i < S->plan->n; i++) {
        jval *p = S->plan->arr[i];
        jval *sn = jget(p, "suite");
        if (sn && sn->t == JSTR && strcmp(sn->s, suite) == 0) {
            jval *fv = jget(p, field);
            return (fv && fv->t == JSTR) ? fv->s : NULL;
        }
    }
    return NULL;
}

static jval *group_by_name(spec *S, const char *name) {
    for (int i = 0; i < S->groups->n; i++)
        if (strcmp(group_name(S->groups->arr[i]), name) == 0) return S->groups->arr[i];
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
static void apply_scenario_args(toklist *t, const jval *args, const char *data_root) {
    if (!args || args->t != JARR) return;
    for (int i = 0; i < args->n; i++) {
        if (args->arr[i]->t != JSTR) die("scenario args element not a string");
        char *cur = subst_data_root(args->arr[i]->s, data_root);
        if (is_flag_tok(cur) && i + 1 < args->n && args->arr[i + 1]->t == JSTR) {
            char *nxt = subst_data_root(args->arr[i + 1]->s, data_root);
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
   with a "scenario" field and matching suite compiles to one cell: the named
   crystal + invariant base geometry + named size, then the scenario's args
   applied with override semantics. Used for the coverage suite's stress cells
   (curved-detector probe, cancellation, extreme geometry) and single-axis
   feature cells (off-center beam, custom detector basis, powder mosaic) whose
   flags would collide with the base geometry under the plain main-effects path. */
static int build_scenarios(spec *S, FILE *out, const char *suite) {
    int count = 0;
    for (int i = 0; i < S->plan->n; i++) {
        jval *p = S->plan->arr[i];
        jval *sn = jget(p, "suite");
        jval *scn = jget(p, "scenario");
        if (!sn || sn->t != JSTR || strcmp(sn->s, suite) != 0) continue;
        if (!scn || scn->t != JSTR) continue;
        jval *jcr = jget(p, "crystal");
        jval *jsz = jget(p, "size");
        jval *jargs = jget(p, "args");
        jval *jgate = jget(p, "gate");           /* "reject" for a guards cell; absent = parity */
        const char *gate_type = (jgate && jgate->t == JSTR) ? jgate->s : NULL;
        if (!jcr || jcr->t != JSTR || !jsz || jsz->t != JSTR)
            die("scenario missing crystal/size");
        jval *gcr = group_by_name(S, jcr->s);
        jval *gsz = group_by_name(S, jsz->s);
        if (!gcr || !gsz) die("scenario crystal/size group not found");

        toklist gpu; tl_init(&gpu);
        push_args(&gpu, gcr, S->data_root);        /* crystal (may be empty for xtal_none) */
        push_base_geometry(S, &gpu);               /* invariant BASE geometry */
        push_args(&gpu, gsz, S->data_root);        /* -Na -Nb -Nc */
        apply_scenario_args(&gpu, jargs, S->data_root);
        check_collisions(&gpu, scn->s);

        toklist cpu; tl_init(&cpu);
        for (int k = 0; k < gpu.n; k++) tl_push(&cpu, gpu.tok[k]);
        push_extra(&cpu, gcr, "cpu_extra");
        push_extra(&cpu, gsz, "cpu_extra");
        push_extra(&cpu, p,   "cpu_extra");        /* scenario-level cpu_extra */

        jval *jtags = jget(p, "tags");
        const char *tags[16]; int nt = 0;
        if (jtags && jtags->t == JARR)
            for (int k = 0; k < jtags->n && nt < 16; k++)
                if (jtags->arr[k]->t == JSTR) tags[nt++] = jtags->arr[k]->s;

        emit_cell(out, S, scn->s, suite, group_label(gcr), group_label(gsz),
                  NULL, NULL, tags, nt, &gpu, &cpu, gate_type);
        count++;
    }
    return count;
}

static int build_coverage(spec *S, FILE *out) {
    const char *bc = plan_field(S, "coverage", "baseline_crystal");
    const char *bs = plan_field(S, "coverage", "baseline_size");
    if (!bc || !bs) die("coverage: plan missing baseline_crystal/baseline_size");
    jval *gcr = group_by_name(S, bc);
    jval *gsz = group_by_name(S, bs);
    if (!gcr || !gsz) die("coverage: baseline crystal/size group not found");
    const char *lc = group_label(gcr);
    const char *ls = group_label(gsz);

    int count = 0;

    /* baseline cell */
    {
        toklist gpu; tl_init(&gpu);
        push_args(&gpu, gcr, S->data_root);
        push_base_geometry(S, &gpu);
        push_args(&gpu, gsz, S->data_root);
        /* Pin oversample (spec sec.11: explicit step counts). Without it nanoBragg
           auto-selects it from crystal size -- e.g. N=1000 -> 180x180 subpixels,
           making the CPU oracle take hours. Fixed sampling is common-mode (CPU and
           GPU both use it) so parity is unaffected; the oversample DIMENSION cells
           override this to 4/8. */
        tl_push(&gpu, "-oversample"); tl_push(&gpu, "1");
        check_collisions(&gpu, "baseline");
        toklist cpu; tl_init(&cpu);
        for (int i = 0; i < gpu.n; i++) tl_push(&cpu, gpu.tok[i]);
        const char *tags[2] = { bc, bs };
        emit_cell(out, S, "baseline", "coverage", lc, ls, NULL, NULL, tags, 2, &gpu, &cpu, NULL);
        count++;
    }

    /* one cell per (dimension, value != baseline) */
    for (int i = 0; i < S->dims->n; i++) {
        jval *d = S->dims->arr[i];
        jval *dn = jget(d, "dim");
        jval *kind = jget(d, "kind");
        jval *flag = jget(d, "flag");
        jval *baseline = jget(d, "baseline");
        jval *values = jget(d, "values");
        if (!dn || !values || values->t != JARR) continue;
        const char *baseline_lit = baseline ? baseline->s : NULL;

        for (int v = 0; v < values->n; v++) {
            jval *val = values->arr[v];
            char idbuf[160];
            toklist gpu; tl_init(&gpu);
            push_args(&gpu, gcr, S->data_root);
            push_base_geometry(S, &gpu);
            push_args(&gpu, gsz, S->data_root);
            /* pinned oversample (overridden by the oversample dimension via tl_set) */
            tl_push(&gpu, "-oversample"); tl_push(&gpu, "1");

            if (kind && kind->t == JSTR && strcmp(kind->s, "enum") == 0) {
                jval *vv = jget(val, "val");
                jval *vargs = jget(val, "args");
                jval *base_enum = baseline; /* baseline is the enum's default val string */
                if (base_enum && vv && str_eq(base_enum->s, vv->s)) continue; /* == baseline */
                if (vargs && vargs->t == JARR)
                    for (int k = 0; k < vargs->n; k++)
                        if (vargs->arr[k]->t == JSTR) tl_push(&gpu, vargs->arr[k]->s);
                snprintf(idbuf, sizeof idbuf, "cov_%s_%s", dn->s, vv ? vv->s : "?");
            } else {
                /* scalar: skip the baseline value. tl_set OVERRIDES a value the
                   base geometry or size group already set (e.g. -Na from the size
                   preset, or -lambda/-pixel from the base geometry) instead of
                   appending a colliding duplicate. */
                if (baseline_lit && str_eq(baseline_lit, val->s)) continue;
                if (flag && flag->t == JSTR) {
                    tl_set(&gpu, flag->s, val->s);
                    /* beam center follows pixel (base.json beam_center_rule). */
                    if (strcmp(flag->s, "-pixel") == 0) {
                        jval *detp = jget(S->base, "detpixels");
                        long detpixels = detp ? strtol(detp->s, NULL, 10) : 2048;
                        double beam = (double)detpixels * strtod(val->s, NULL) / 2.0;
                        char bb[64]; snprintf(bb, sizeof bb, "%.6f", beam);
                        tl_set(&gpu, "-Xbeam", bb); tl_set(&gpu, "-Ybeam", bb);
                    }
                }
                snprintf(idbuf, sizeof idbuf, "cov_%s_%s", dn->s, val->s);
            }

            check_collisions(&gpu, idbuf);
            toklist cpu; tl_init(&cpu);
            for (int k = 0; k < gpu.n; k++) tl_push(&cpu, gpu.tok[k]);
            const char *tags[3] = { bc, bs, dn->s };
            emit_cell(out, S, idbuf, "coverage", lc, ls, NULL, NULL, tags, 3, &gpu, &cpu, NULL);
            count++;
        }
    }

    /* hand-authored interaction + feature scenarios (after the main-effects) */
    count += build_scenarios(S, out, "coverage");
    return count;
}

/* --------------------------------------------------------------------- main */

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
    snprintf(path, sizeof path, "%s/base.json", specdir);       S.base   = parse_json_file(path);
    snprintf(path, sizeof path, "%s/dimensions.jsonl", specdir);S.dims   = parse_jsonl_file(path);
    snprintf(path, sizeof path, "%s/groups.json", specdir);
    { jval *g = parse_json_file(path); jval *garr = jget(g, "groups"); if (!garr || garr->t != JARR) die("groups.json: no groups array"); S.groups = garr; }
    snprintf(path, sizeof path, "%s/plan.json", specdir);       S.plan   = parse_jsonl_file(path);

    jval *dr = jget(S.base, "data_root");
    if (!dr || dr->t != JSTR) die("base.json: data_root missing");
    S.data_root = dr->s;

    FILE *out = outpath ? fopen(outpath, "wb") : stdout;
    if (!out) { fprintf(stderr, "gen_suites: cannot write %s\n", outpath); return 2; }

    int count;
    if (strcmp(suite, "grid320") == 0)       count = build_grid320(&S, out);
    else if (strcmp(suite, "coverage") == 0) count = build_coverage(&S, out);
    else if (strcmp(suite, "guards") == 0)   count = build_scenarios(&S, out, "guards");
    else if (strcmp(suite, "perf") == 0)     count = build_scenarios(&S, out, "perf");
    else if (strcmp(suite, "pairwise") == 0) count = build_scenarios(&S, out, "pairwise");
    else { fprintf(stderr, "gen_suites: unknown suite '%s'\n", suite); if (outpath) fclose(out); return 2; }

    if (outpath) fclose(out);
    fprintf(stderr, "gen_suites: suite=%s cells=%d -> %s\n", suite, count, outpath ? outpath : "(stdout)");
    return 0;
}
