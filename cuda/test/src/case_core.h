/* case_core.h -- the ONE place base.json + suites/<name>.jsonl are parsed.
 *
 * Every consuming tool (nbgensuite / nbcache / nbrunsuite) links this module so
 * the spec schema is read in exactly one place. Built on the json-c library.
 *
 * Vocabulary (NBTOOLS-SPEC §3): candidate = binary under test (was "gpu"),
 * reference = trusted binary / truth (was "cpu"). K = cost.compute, the
 * wall-time proxy (NOT cost.precision).
 *
 * Args carry the literal {input_root} token; cc_resolve_input_root() expands it
 * to an absolute path (single source of truth for token resolution -- the job
 * the retired run.sh used to do). The tokenized form is what args_hash keys on,
 * so the cache key stays machine-independent. base.json's input_root is
 * harness-root-relative (harness root = dirname of base.json's own dir,
 * i.e. the dir containing spec/); cc_input_root_abs() anchors it there via
 * cc_load_base's own path argument, never the process CWD.
 *
 * Ownership: cc_load_base / cc_load_suite allocate a DOM plus the returned
 * struct; free with cc_base_free / cc_suite_free. All const char* fields point
 * into that DOM and are valid until the matching free. cc_resolve_input_root /
 * cc_input_root_abs return freshly malloc'd strings the caller frees.
 * Malformed JSON is fatal (the parser exits); a missing required key returns NULL.
 */
#ifndef NB_CASE_CORE_H
#define NB_CASE_CORE_H

/* The DOM handle is a json-c object; forward-declared so this header stays free
   of the json-c include (only case_core.c and the emitters touch its innards). */
struct json_object;

/* ---- base.json ---------------------------------------------------------- */

typedef struct {
    const char *input_root;               /* harness-root-relative render-input root */
    char *harness_root_abs;               /* owned: absolute harness root (dirname of
                                              base.json's dir), derived from the path
                                              cc_load_base was given -- NOT the CWD */

    /* default absolute gate (base.json "gate") */
    double corr_min;                      /* Pearson corr floor                */
    double sum_ratio_min;                 /* sum_ratio lower bound             */
    double sum_ratio_max;                 /* sum_ratio upper bound             */

    long cache_budget_gb;                 /* image-cache size budget           */

    const char **reference_fix_branches;  /* branches the oracle must carry    */
    int n_reference_fix_branches;

    const char **base_flags;              /* flags every case inherits         */
    int n_base_flags;

    const char **base_geometry_dims;      /* dims forming the invariant BASE   */
    int n_base_geometry_dims;

    struct json_object *root;             /* owns the base.json DOM (opaque)   */
} cc_base;

/* ---- suites/<name>.jsonl ------------------------------------------------ */

typedef enum {
    CC_GATE_ABSOLUTE,   /* corr_min + sum_ratio range on FP32 metrics         */
    CC_GATE_REJECT,     /* exit-code guard: kernel must refuse (exit 9)        */
    CC_GATE_PERF        /* timing, warn-not-fail; no expected baseline         */
} cc_gate_type;

typedef struct {
    const char *id;                 /* case id                                 */
    const char *suite;              /* suite name                              */
    const char *candidate_args;     /* binary under test                       */
    const char *reference_args;     /* trusted binary                          */
    const char *args_hash;          /* baked md5 cache key over reference_args */

    long K;                         /* cost.compute, the wall-time proxy       */

    cc_gate_type gate_type;         /* absolute / reject / perf                */
    int has_gate_override;          /* 1 if the case carried its own "gate"    */
    double corr_min;                /* effective (case override else base)     */
    double sum_ratio_min;
    double sum_ratio_max;
} cc_case;

typedef struct {
    cc_case *cases;
    int n;
    struct json_object *root;       /* owns the .jsonl DOM (opaque)            */
} cc_suite;

/* Load base.json into *cc_base (heap). Returns NULL if a required key is
   missing (input_root). Free with cc_base_free. */
cc_base *cc_load_base(const char *path);
void cc_base_free(cc_base *b);

/* Load a suites/<name>.jsonl file into an array of cases. `base` supplies the
   default gate values for cases that omit a "gate" object. Returns NULL on an
   empty/absent case list. Free with cc_suite_free. */
cc_suite *cc_load_suite(const char *path, const cc_base *base);
void cc_suite_free(cc_suite *s);

/* Expand every "{input_root}" token in `args` to `input_root_abs`. Returns a
   newly malloc'd string the caller frees. */
char *cc_resolve_input_root(const char *args, const char *input_root_abs);

/* Resolve base->input_root (harness-root-relative) to an absolute path via
   realpath, anchored at base->harness_root_abs (set by cc_load_base from the
   base.json path it was given, not the process CWD). Returns a newly
   malloc'd string the caller frees, or NULL if the path does not exist. */
char *cc_input_root_abs(const cc_base *base);

#endif /* NB_CASE_CORE_H */
