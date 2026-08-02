/* nbrunsuite.c -- run a parity suite (NBTOOLS-SPEC §5/§6/§10/§11).
 *
 * For each case in suites/<suite>.jsonl: acquire the reference image (image
 * cache hit, else render the trusted reference binary and store it), render the
 * candidate binary under test, compare the two float32 images with nbmetrics,
 * apply the suite's typed gate (absolute / reject / perf), and detect verdict
 * FLIPS vs the committed expected/<suite>.<precision>.tsv baseline. Writes
 * results.tsv (+ a provenance trailer) to the workdir.
 *
 *   nbrunsuite --suite NAME --candidate PATH --gpu "NAME|index|uuid" --workdir PATH
 *
 * Device selection (§5): GPUs are enumerated via NVML (the driver's management
 * library -- what nvidia-smi wraps; no nvidia-smi parsing, no cudart, no CUDA
 * context, no visibility pin), and --gpu is resolved by EXACT equality on index /
 * name / uuid (never strstr -- the desktop name is a strict prefix of the laptop's).
 * NVML reports each device's own UUID/name, which ARE the ground-truth identity, so
 * there is no pin-then-reprobe. The parent process keeps a clean environment; only
 * the render child (do_render) gets CUDA_DEVICE_ORDER=PCI_BUS_ID and
 * CUDA_VISIBLE_DEVICES=GPU-<uuid> so the candidate renders on exactly the resolved
 * card. Refuses (exit 4) on no-match / ambiguous match.
 *
 * Cache (§9): reference images only, keyed on the case's BAKED args_hash (read,
 * never recomputed). cache_gc() runs at suite start; cache_lookup is a .bin
 * existence test; a miss renders the reference and cache_store + cache_write_meta.
 *
 * --seed (§6): re-baseline expected/. Guarded by the reference canary -- a
 * from-source "gold" (main + base.json reference_fix_branches applied to
 * nanoBragg.c) is built and the three fix-sensitive canaries (phi0/subpixel/
 * curved, ported from the retired gen_cpu.sh) are rendered with gold and with
 * --reference; seeding is REFUSED unless --reference reproduces gold on all three.
 * It then refuses to change any existing verdict (prints flips, nonzero exit)
 * unless --force.
 *
 * nbmetrics is invoked as a subprocess (its FP32 stdout contract is stable);
 * cache_core / argkey / case_core are linked (plus the json-c system lib).
 *
 * TEST HOOKS (documented, harmless in production):
 *   --skip-device     skip NVML device selection, so the render->cache->metrics->
 *                     gate->flip pipeline can be exercised with a CPU stand-in
 *                     candidate on a box with no eligible GPU. --gpu is not
 *                     required under --skip-device.
 *   --expected-dir D  read/write the expected baseline under D instead of
 *                     <harness_root>/expected (the committed expected/ tree does
 *                     not exist until the golden migration, §13.7).
 *   --canary-fast     shrink the --seed canary geometry (detpixels/N/steps) so the
 *                     canary build+render+refuse PLUMBING can be verified in
 *                     seconds; the physics-faithful canary needs the full 2048^2
 *                     geometry and is deferred to the GPU validation step.
 */
#define _XOPEN_SOURCE 700
#include "cache_core.h"
#include "case_core.h"
#include "argkey.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <nvml.h>   /* GPU enumeration via the driver management lib; links -lnvidia-ml */

#ifndef NB_BUILD_COMMIT
#define NB_BUILD_COMMIT "unknown"
#endif

static void die(const char *m) { fprintf(stderr, "nbrunsuite: %s\n", m); exit(2); }

/* =========================================================================
 * small dynamic string / argv helpers
 * ========================================================================= */

typedef struct { char **v; int n, cap; } argv_t;

static void av_init(argv_t *a) {
    a->n = 0; a->cap = 16;
    a->v = (char **)malloc((size_t)a->cap * sizeof(char *));
    if (!a->v) die("OOM");
}
static void av_push(argv_t *a, const char *s) {
    if (a->n + 2 > a->cap) { a->cap *= 2; a->v = (char **)realloc(a->v, (size_t)a->cap * sizeof(char *)); if (!a->v) die("OOM"); }
    a->v[a->n++] = strdup(s);
    if (!a->v[a->n - 1]) die("OOM");
}
static void av_terminate(argv_t *a) {
    if (a->n + 1 > a->cap) { a->cap += 1; a->v = (char **)realloc(a->v, (size_t)a->cap * sizeof(char *)); if (!a->v) die("OOM"); }
    a->v[a->n] = NULL;
}
static void av_free(argv_t *a) {
    for (int i = 0; i < a->n; i++) free(a->v[i]);
    free(a->v);
    a->v = NULL; a->n = a->cap = 0;
}

/* Whitespace-split `s` and push each token onto `a`. Tokens are single argv
   words (case args carry no embedded spaces). */
static void av_push_split(argv_t *a, const char *s) {
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t') p++;
        size_t len = (size_t)(p - start);
        char *tok = (char *)malloc(len + 1);
        if (!tok) die("OOM");
        memcpy(tok, start, len); tok[len] = '\0';
        if (a->n + 2 > a->cap) { a->cap *= 2; a->v = (char **)realloc(a->v, (size_t)a->cap * sizeof(char *)); if (!a->v) die("OOM"); }
        a->v[a->n++] = tok;
    }
}

/* =========================================================================
 * device selection (§5)
 * ========================================================================= */

typedef struct {
    int index;
    char name[256];
    char uuid[128];             /* NVML "GPU-..." string; this IS the pin value */
    char pci[64];
    unsigned long long mem_total;   /* total device memory, bytes */
    int cc_major, cc_minor;         /* CUDA compute capability -> sm_XY */
} gpu_dev;

/* Resolved "GPU-<uuid>" pin for the render child's environment (§5). Empty under
   --skip-device; set once after resolve_gpu so each render fork inherits
   CUDA_VISIBLE_DEVICES while this parent process's own environment stays clean. */
static char g_child_cvd[128] = "";

/* Trim leading/trailing spaces of a field in place; returns the start. */
static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = '\0';
    return s;
}

/* Enumerate GPUs via NVML (the driver's management library -- what nvidia-smi
   wraps). No CUDA context and no CUDA_VISIBLE_DEVICES pin, so NVML sees every card.
   Returns count; *out receives a malloc'd array (caller frees). -1 on failure. */
static int query_gpus(gpu_dev **out) {
    if (nvmlInit_v2() != NVML_SUCCESS) return -1;
    unsigned int count = 0;
    if (nvmlDeviceGetCount_v2(&count) != NVML_SUCCESS) { nvmlShutdown(); return -1; }
    gpu_dev *d = (gpu_dev *)malloc((size_t)(count ? count : 1) * sizeof(gpu_dev));
    if (!d) { nvmlShutdown(); return -1; }
    int n = 0;
    for (unsigned int i = 0; i < count; i++) {
        nvmlDevice_t h;
        if (nvmlDeviceGetHandleByIndex_v2(i, &h) != NVML_SUCCESS) continue;
        memset(&d[n], 0, sizeof d[n]);
        d[n].index = (int)i;
        if (nvmlDeviceGetName(h, d[n].name, sizeof d[n].name) != NVML_SUCCESS)
            snprintf(d[n].name, sizeof d[n].name, "unknown");
        if (nvmlDeviceGetUUID(h, d[n].uuid, sizeof d[n].uuid) != NVML_SUCCESS)
            snprintf(d[n].uuid, sizeof d[n].uuid, "unknown");   /* "GPU-..." string */
        nvmlMemory_t mem;
        if (nvmlDeviceGetMemoryInfo(h, &mem) == NVML_SUCCESS) d[n].mem_total = mem.total;
        int major = 0, minor = 0;
        if (nvmlDeviceGetCudaComputeCapability(h, &major, &minor) == NVML_SUCCESS) {
            d[n].cc_major = major; d[n].cc_minor = minor;
        }
        nvmlPciInfo_t pci;
        if (nvmlDeviceGetPciInfo_v3(h, &pci) == NVML_SUCCESS)
            snprintf(d[n].pci, sizeof d[n].pci, "%s", pci.busId);
        n++;
    }
    nvmlShutdown();
    *out = d;
    return n;
}

static void list_gpus(void) {
    gpu_dev *d = NULL;
    int n = query_gpus(&d);
    if (n < 0) { fprintf(stderr, "nbrunsuite: NVML unavailable\n"); free(d); return; }
    printf("# %d GPU(s) via NVML\n", n);
    printf("%-6s %-34s %-7s %-9s %-42s %s\n", "index", "name", "sm", "mem", "uuid", "pci.bus_id");
    for (int i = 0; i < n; i++) {
        char sm[16];  snprintf(sm,  sizeof sm,  "sm_%d%d", d[i].cc_major, d[i].cc_minor);
        char mem[16]; snprintf(mem, sizeof mem, "%.1fGiB", (double)d[i].mem_total / (1024.0*1024.0*1024.0));
        printf("%-6d %-34s %-7s %-9s %-42s %s\n", d[i].index, d[i].name, sm, mem, d[i].uuid, d[i].pci);
    }
    free(d);
}

/* Resolve --gpu by EXACT equality on index/name/uuid. Never strstr. Fills
   *match on a unique hit; refuses (exit) on no-match / multi-match. */
static void resolve_gpu(const char *sel, gpu_dev *match) {
    gpu_dev *d = NULL;
    int n = query_gpus(&d);
    if (n < 0) die("NVML unavailable (need --gpu resolution; use --skip-device for a no-GPU test run)");
    if (n == 0) die("no GPUs reported by NVML");

    /* is sel an integer index? */
    int is_index = 1;
    for (const char *q = sel; *q; q++) if (!isdigit((unsigned char)*q)) { is_index = 0; break; }
    long want_idx = is_index ? strtol(sel, NULL, 10) : -1;

    int found = -1, nmatch = 0;
    for (int i = 0; i < n; i++) {
        int hit = 0;
        if (is_index && d[i].index == want_idx) hit = 1;
        if (strcmp(d[i].name, sel) == 0) hit = 1;   /* EXACT name */
        if (strcmp(d[i].uuid, sel) == 0) hit = 1;   /* EXACT uuid */
        if (hit) { nmatch++; found = i; }
    }
    if (nmatch == 0) {
        fprintf(stderr, "nbrunsuite: --gpu '%s' matched no device. Available:\n", sel);
        for (int i = 0; i < n; i++)
            fprintf(stderr, "  index=%d name=\"%s\" uuid=%s\n", d[i].index, d[i].name, d[i].uuid);
        free(d); exit(4);
    }
    if (nmatch > 1) {
        fprintf(stderr, "nbrunsuite: --gpu '%s' is ambiguous (%d matches); use the exact uuid\n", sel, nmatch);
        free(d); exit(4);
    }
    *match = d[found];
    free(d);
}

/* =========================================================================
 * render (fork + exec in an isolated cwd, timed)
 * ========================================================================= */

/* Run `bin` with the NULL-terminated argv `av->v` (av->v[0] must be bin), cwd =
   scratch (so Fdump.bin cannot leak between cases), stdout/stderr -> logpath.
   Returns the child exit status (WEXITSTATUS), or -1 if it could not run.
   *secs receives the wall time. */
static int do_render(const char *bin, argv_t *av, const char *scratch,
                     const char *logpath, double *secs) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    pid_t pid = fork();
    if (pid < 0) { *secs = 0.0; return -1; }
    if (pid == 0) {
        if (chdir(scratch) != 0) _exit(126);
        /* Pin the CUDA device in the CHILD's environment only (§5): the parent
           never sets these, so NVML enumeration stays context-free. g_child_cvd is
           empty under --skip-device, and the CPU reference oracle ignores it. */
        if (g_child_cvd[0]) {
            setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
            setenv("CUDA_VISIBLE_DEVICES", g_child_cvd, 1);
        }
        int fd = open(logpath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
        execv(bin, av->v);
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

/* Build the render argv: bin, -floatfile <path>, <args...>, [<precision...>]. */
static void build_render_argv(argv_t *av, const char *bin, const char *floatfile,
                              const char *args_resolved, const char *prec_args) {
    av_init(av);
    av_push(av, bin);
    av_push(av, "-floatfile");
    av_push(av, floatfile);
    av_push_split(av, args_resolved);
    if (prec_args && *prec_args) av_push_split(av, prec_args);
    av_terminate(av);
}

/* =========================================================================
 * nbmetrics subprocess (§7 stable stdout contract)
 * ========================================================================= */

typedef struct {
    int ok;
    double corr, sum_ratio, max_rel, wpf, peak_max_rel;
    char wis[16];
    double fp16_corr, fp16_sum_ratio;
    long long fp16_diff, fp16_excl;
} metrics_t;

/* Invoke `nbmetrics cand ref` and parse its one stable FP32+FP16 line. */
static metrics_t run_metrics(const char *nbmetrics, const char *cand, const char *ref) {
    metrics_t m; memset(&m, 0, sizeof m); m.ok = 0;
    size_t cmdlen = strlen(nbmetrics) + strlen(cand) + strlen(ref) + 64;
    char *cmd = (char *)malloc(cmdlen);
    if (!cmd) die("OOM");
    /* nbmetrics takes <candidate> <reference>; quote paths defensively */
    snprintf(cmd, cmdlen, "'%s' '%s' '%s' 2>/dev/null", nbmetrics, cand, ref);
    FILE *p = popen(cmd, "r");
    free(cmd);
    if (!p) return m;
    char line[512];
    if (fgets(line, sizeof line, p)) {
        int got = sscanf(line, "%lf %lf %lf %lf %lf %15s %lf %lf %lld %lld",
                         &m.corr, &m.sum_ratio, &m.max_rel, &m.wpf, &m.peak_max_rel,
                         m.wis, &m.fp16_corr, &m.fp16_sum_ratio, &m.fp16_diff, &m.fp16_excl);
        if (got >= 6) m.ok = 1;
    }
    pclose(p);
    return m;
}

/* =========================================================================
 * per-case scratch (holds the fields the per-case results.tsv row is built from;
 * the whole-suite tally/flips re-read results.tsv, §6 batching)
 * ========================================================================= */

typedef struct {
    char id[128];
    char verdict[16];
    double ms;
    char note[256];
} case_result;

typedef struct { case_result *r; int n, cap; } results_t;

static void res_init(results_t *R) { R->n = 0; R->cap = 64; R->r = malloc((size_t)R->cap * sizeof(case_result)); if (!R->r) die("OOM"); }
static case_result *res_add(results_t *R) {
    if (R->n == R->cap) { R->cap *= 2; R->r = realloc(R->r, (size_t)R->cap * sizeof(case_result)); if (!R->r) die("OOM"); }
    case_result *c = &R->r[R->n++];
    memset(c, 0, sizeof *c);
    return c;
}

/* =========================================================================
 * canary (--seed reference correctness gate, §6/§15)
 * ========================================================================= */

/* Run a plain command (argv NULL-terminated) with stdout/stderr -> logpath,
   cwd unchanged. Returns exit status or -1. */
static int run_cmd(char *const argv[], const char *logpath) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        if (logpath) {
            int fd = open(logpath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
        }
        execvp(argv[0], argv);
        _exit(127);
    }
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

/* md5 of a file's bytes -> out[33]. 0 ok, -1 err. (cache_reference_md5 already
   does exactly this for any file.) */
static int file_md5(const char *path, char out[33]) { return cache_reference_md5(path, out); }

/* Build the from-source gold binary: main:nanoBragg.c + each reference_fix_branch
   diff applied, compiled with gcc -O3 -fopenmp. Writes it to gold_out. Returns 0
   on success, nonzero on failure (message on stderr). Mirrors the retired
   gen_cpu.sh logic (not its script). */
static int build_gold(const cc_base *base, const char *harness_root,
                      const char *build_dir, const char *gold_out) {
    char main_src[4096], gold_src[4096], tmp_src[4096], patch_path[4096], log[4096];
    snprintf(main_src, sizeof main_src, "%s/main_nanoBragg.c", build_dir);
    snprintf(gold_src, sizeof gold_src, "%s/gold_nanoBragg.c", build_dir);
    snprintf(tmp_src,  sizeof tmp_src,  "%s/nanoBragg.c", build_dir);
    snprintf(log,      sizeof log,      "%s/canary_build.log", build_dir);
    mkdir(build_dir, 0755);

    /* repo root = harness_root/../.. (harness root is cuda/test) */
    char repo[4096];
    snprintf(repo, sizeof repo, "%s/../..", harness_root);

    /* git -C <repo> show main:nanoBragg.c > main_src */
    {
        char showexpr[64]; snprintf(showexpr, sizeof showexpr, "main:nanoBragg.c");
        char *argv[] = { "git", "-C", repo, "show", showexpr, NULL };
        int fd = open(main_src, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { fprintf(stderr, "nbrunsuite: canary: cannot create %s\n", main_src); return 1; }
        pid_t pid = fork();
        if (pid == 0) { dup2(fd, 1); execvp(argv[0], argv); _exit(127); }
        close(fd);
        int st = 0; while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) { fprintf(stderr, "nbrunsuite: canary: git show main:nanoBragg.c failed\n"); return 1; }
    }
    /* cp main -> gold (start point) */
    {
        char *argv[] = { "cp", main_src, gold_src, NULL };
        if (run_cmd(argv, log) != 0) { fprintf(stderr, "nbrunsuite: canary: cp failed\n"); return 1; }
    }
    /* apply each fix branch diff of nanoBragg.c to the gold copy */
    for (int i = 0; i < base->n_reference_fix_branches; i++) {
        const char *br = base->reference_fix_branches[i];
        snprintf(patch_path, sizeof patch_path, "%s/%d.patch", build_dir, i);
        /* git -C repo diff main..br -- nanoBragg.c > patch */
        char range[256]; snprintf(range, sizeof range, "main..%s", br);
        char *dargv[] = { "git", "-C", repo, "diff", range, "--", "nanoBragg.c", NULL };
        int fd = open(patch_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { fprintf(stderr, "nbrunsuite: canary: cannot create patch\n"); return 1; }
        pid_t pid = fork();
        if (pid == 0) { dup2(fd, 1); int e = open("/dev/null", O_WRONLY); if (e>=0) dup2(e,2); execvp(dargv[0], dargv); _exit(127); }
        close(fd);
        int st = 0; while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) { fprintf(stderr, "nbrunsuite: canary: git diff %s failed\n", range); return 1; }

        struct stat pst;
        if (stat(patch_path, &pst) != 0 || pst.st_size == 0) continue;  /* empty diff = already merged */

        /* cp gold -> build/nanoBragg.c ; patch -p1 <patch (in build_dir) ; cp back */
        char *cp1[] = { "cp", gold_src, tmp_src, NULL };
        if (run_cmd(cp1, log) != 0) { fprintf(stderr, "nbrunsuite: canary: cp gold->tmp failed\n"); return 1; }
        /* patch must run in build_dir so -p1 nanoBragg.c resolves */
        pid_t pp = fork();
        if (pp == 0) {
            if (chdir(build_dir) != 0) _exit(126);
            int in = open(patch_path, O_RDONLY);
            if (in >= 0) dup2(in, 0);
            int lg = open(log, O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (lg >= 0) { dup2(lg, 1); dup2(lg, 2); }
            char *pargv[] = { "patch", "-p1", NULL };
            execvp(pargv[0], pargv); _exit(127);
        }
        int pst2 = 0; while (waitpid(pp, &pst2, 0) < 0 && errno == EINTR) {}
        if (!WIFEXITED(pst2) || WEXITSTATUS(pst2) != 0) { fprintf(stderr, "nbrunsuite: canary: patch %s failed (see %s)\n", br, log); return 1; }
        char *cp2[] = { "cp", tmp_src, gold_src, NULL };
        if (run_cmd(cp2, log) != 0) { fprintf(stderr, "nbrunsuite: canary: cp tmp->gold failed\n"); return 1; }
    }
    /* gcc -O3 -fopenmp gold_src -o gold_out -lm */
    {
        char *gargv[] = { "gcc", "-O3", "-fopenmp", gold_src, "-o", (char *)gold_out, "-lm", NULL };
        if (run_cmd(gargv, log) != 0) { fprintf(stderr, "nbrunsuite: canary: gcc gold build failed (see %s)\n", log); return 1; }
    }
    return 0;
}

/* The three fix-sensitive canary CLIs (ported from gen_cpu.sh). `hkl` is the
   resolved 193L.hkl path; if fast!=0 the geometry is shrunk for plumbing tests. */
static void canary_cli(int which, const char *hkl, int fast, char *out, size_t cap) {
    long detpixels = fast ? 64 : 2048;
    const char *N = fast ? "4" : "100";
    const char *phisteps = fast ? "2" : "10";
    const char *thicksteps = fast ? "2" : "8";
    double beam = (double)detpixels * 0.172 / 2.0;
    char base[1536];
    snprintf(base, sizeof base,
        "-hkl %s -cell 78.540 78.540 37.770 90 90 90 "
        "-distance 231.27 -lambda 0.9768 -pixel 0.172 -detpixels %ld "
        "-Xbeam %.6f -Ybeam %.6f -flux 1e18 -beamsize 1.0 -nonoise -nointerpolate -nopgm "
        "-Na %s -Nb %s -Nc %s",
        hkl, detpixels, beam, beam, N, N, N);
    if (which == 0)        /* phi0 */
        snprintf(out, cap, "%s -oversample 1 -osc 0.1 -phisteps %s -misset 0 0 0", base, phisteps);
    else if (which == 1)   /* subpixel */
        snprintf(out, cap, "%s -oversample 2 -detector_thick 100 -detector_thicksteps %s -detector_abs 100 -oversample_thick -phisteps 1 -misset 0 0 0", base, thicksteps);
    else                   /* curved (trailing -curved_det) */
        snprintf(out, cap, "%s -oversample 1 -misset 0 0 0 -curved_det", base);
}

/* Render a canary CLI with `bin`, return md5 of the float image in out[33].
   Returns 0 ok, -1 on render/read failure. */
static int canary_render_md5(const char *bin, const char *cli, const char *scratch,
                             const char *tag, char out[33]) {
    char floatfile[4096], log[4096];
    snprintf(floatfile, sizeof floatfile, "%s/%s.bin", scratch, tag);
    snprintf(log, sizeof log, "%s/%s.log", scratch, tag);
    unlink(floatfile);
    argv_t av; build_render_argv(&av, bin, floatfile, cli, NULL);
    double secs;
    int rc = do_render(bin, &av, scratch, log, &secs);
    av_free(&av);
    struct stat st;
    if (rc != 0 || stat(floatfile, &st) != 0 || st.st_size == 0) return -1;
    int r = file_md5(floatfile, out);
    unlink(floatfile);
    return r;
}

/* Reference canary (§6): build gold, render the 3 canaries with gold and with
   `reference`, and REFUSE (return nonzero) unless reference reproduces gold on
   all three. Returns 0 if the reference is trustworthy. */
static int reference_canary(const cc_base *base, const char *harness_root,
                            const char *reference, const char *workdir, int fast) {
    char build_dir[1024], scratch[1024], gold_bin[2048], hkl_path[1024];
    snprintf(build_dir, sizeof build_dir, "%s/canary_build", workdir);
    snprintf(scratch,   sizeof scratch,   "%s/canary_scratch", workdir);
    snprintf(gold_bin,  sizeof gold_bin,  "%s/nb_gold", build_dir);
    mkdir(workdir, 0755); mkdir(build_dir, 0755); mkdir(scratch, 0755);

    char *input_root = cc_input_root_abs(base);
    if (!input_root) { fprintf(stderr, "nbrunsuite: canary: cannot resolve input_root\n"); return 1; }
    snprintf(hkl_path, sizeof hkl_path, "%s/crystals/193L.hkl", input_root);
    free(input_root);

    fprintf(stderr, "# canary: building gold (main + %d fix branch(es)) ...\n", base->n_reference_fix_branches);
    if (build_gold(base, harness_root, build_dir, gold_bin) != 0) return 1;

    const char *names[3] = { "phi0", "subpixel", "curved" };
    for (int i = 0; i < 3; i++) {
        char cli[2048], gtag[64], rtag[64], goldmd5[33], refmd5[33];
        canary_cli(i, hkl_path, fast, cli, sizeof cli);
        snprintf(gtag, sizeof gtag, "gold_%s", names[i]);
        snprintf(rtag, sizeof rtag, "ref_%s", names[i]);
        if (canary_render_md5(gold_bin, cli, scratch, gtag, goldmd5) != 0) {
            fprintf(stderr, "nbrunsuite: canary: gold render failed for %s\n", names[i]); return 1;
        }
        if (canary_render_md5(reference, cli, scratch, rtag, refmd5) != 0) {
            fprintf(stderr, "nbrunsuite: canary: reference render failed for %s\n", names[i]); return 1;
        }
        if (strcmp(goldmd5, refmd5) != 0) {
            fprintf(stderr, "nbrunsuite: canary REFUSED -- reference does not reproduce gold on '%s'\n", names[i]);
            fprintf(stderr, "            gold=%s reference=%s\n", goldmd5, refmd5);
            return 1;
        }
        fprintf(stderr, "# canary: %s OK (md5=%s)\n", names[i], goldmd5);
    }
    return 0;
}

/* =========================================================================
 * expected baseline (verdict) I/O
 * ========================================================================= */

typedef struct { char id[128]; char verdict[16]; } exp_row;
typedef struct { exp_row *r; int n, cap; int present; } expected_t;

static void expected_path(char *out, size_t cap, const char *expdir,
                          const char *suite, const char *precision) {
    snprintf(out, cap, "%s/%s.%s.tsv", expdir, suite, precision);
}

/* Load an expected/<suite>.<precision>.tsv (cell<TAB>verdict[...]). */
static expected_t load_expected(const char *path) {
    expected_t E; E.n = 0; E.cap = 64; E.present = 0;
    E.r = malloc((size_t)E.cap * sizeof(exp_row));
    if (!E.r) die("OOM");
    FILE *f = fopen(path, "r");
    if (!f) return E;
    E.present = 1;
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';
        char *v = tab + 1;
        char *tab2 = strchr(v, '\t'); if (tab2) *tab2 = '\0';
        char *nl = strchr(v, '\n'); if (nl) *nl = '\0';
        if (E.n == E.cap) { E.cap *= 2; E.r = realloc(E.r, (size_t)E.cap * sizeof(exp_row)); if (!E.r) die("OOM"); }
        snprintf(E.r[E.n].id, sizeof E.r[E.n].id, "%s", trim(line));
        snprintf(E.r[E.n].verdict, sizeof E.r[E.n].verdict, "%s", trim(v));
        E.n++;
    }
    fclose(f);
    return E;
}

static const char *expected_verdict(const expected_t *E, const char *id) {
    for (int i = 0; i < E->n; i++) if (strcmp(E->r[i].id, id) == 0) return E->r[i].verdict;
    return NULL;
}

/* ---- accumulated results.tsv (whole-file view across --cases batches) ---- */

typedef struct { char id[128]; char corr[32]; char sr[32]; char ms[32]; char verdict[16]; } res_row;
typedef struct { res_row *r; int n, cap; } loaded_t;

/* Load the DATA rows of results.tsv (cell corr sum_ratio ms verdict ...), so the
   final batch tallies and flip-checks the whole suite -- not just this
   invocation's cases. Comment/header rows are skipped. */
static loaded_t load_results(const char *path) {
    loaded_t L; L.n = 0; L.cap = 64; L.r = malloc((size_t)L.cap * sizeof(res_row));
    if (!L.r) die("OOM");
    FILE *f = fopen(path, "r");
    if (!f) return L;
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *nl = strchr(line, '\n'); if (nl) *nl = '\0';
        char *cols[6] = {0}; int nc = 0; char *s = line;
        while (nc < 6) { cols[nc++] = s; char *t = strchr(s, '\t'); if (!t) break; *t = '\0'; s = t + 1; }
        if (nc < 5) continue;
        if (strcmp(cols[0], "cell") == 0) continue;   /* header */
        if (L.n == L.cap) { L.cap *= 2; L.r = realloc(L.r, (size_t)L.cap * sizeof(res_row)); if (!L.r) die("OOM"); }
        snprintf(L.r[L.n].id,      sizeof L.r[L.n].id,      "%s", cols[0]);
        snprintf(L.r[L.n].corr,    sizeof L.r[L.n].corr,    "%s", cols[1]);
        snprintf(L.r[L.n].sr,      sizeof L.r[L.n].sr,      "%s", cols[2]);
        snprintf(L.r[L.n].ms,      sizeof L.r[L.n].ms,      "%s", cols[3]);
        snprintf(L.r[L.n].verdict, sizeof L.r[L.n].verdict, "%s", cols[4]);
        L.n++;
    }
    fclose(f);
    return L;
}

/* =========================================================================
 * main
 * ========================================================================= */

static void usage(void) {
    fprintf(stderr,
        "usage: nbrunsuite --suite NAME --candidate PATH --reference PATH --gpu \"NAME|index|uuid\" --workdir PATH\n"
        "  options:\n"
        "    --reference PATH        REQUIRED trusted CPU oracle binary, no default (see INPUTS.md)\n"
        "    --precision fp32|df64   candidate -precision single|double; selects expected baseline\n"
        "    --cases N-M             run only cases N..M (1-based inclusive)\n"
        "    --seed [--force]        re-baseline expected/ (canary-gated; refuses verdict flips)\n"
        "    --append-to-ledger [--tag NAME]\n"
        "    --list-gpus             print the GPU table and exit\n"
        "    --keep-candidate-images / --keep-reference-images\n"
        "    --refresh-cache / --no-cache\n"
        "    --cache-dir PATH / --budget-gb N\n"
        "    --build-commit          print the git HEAD this was built at and exit\n"
        "  test hooks: --skip-device  --expected-dir DIR  --ledger-dir DIR  --canary-fast\n");
}

int main(int argc, char **argv) {
    const char *suite = NULL, *candidate = NULL, *gpu = NULL, *workdir = NULL;
    const char *reference_flag = NULL, *precision = "fp32";
    const char *cache_dir_flag = NULL, *expected_dir_flag = NULL, *ledger_dir_flag = NULL, *tag = NULL;
    long flag_budget = -1;
    int range_lo = 1, range_hi = 1<<30;
    int do_seed = 0, force = 0, append_ledger = 0, list_only = 0;
    int keep_cand = 0, keep_ref = 0, refresh_cache = 0, no_cache = 0;
    int skip_device = 0, canary_fast = 0;

    enum { O_SUITE=1000, O_CAND, O_GPU, O_WORKDIR, O_REF, O_PREC, O_CASES, O_SEED,
           O_FORCE, O_LEDGER, O_TAG, O_LISTGPU, O_KEEPC, O_KEEPR, O_REFRESH, O_NOCACHE,
           O_CACHEDIR, O_BUDGET, O_BUILDCOMMIT, O_SKIPDEV, O_EXPDIR, O_LEDGERDIR, O_CANFAST, O_HELP };
    static struct option lo[] = {
        {"suite",required_argument,0,O_SUITE}, {"candidate",required_argument,0,O_CAND},
        {"gpu",required_argument,0,O_GPU}, {"workdir",required_argument,0,O_WORKDIR},
        {"reference",required_argument,0,O_REF}, {"precision",required_argument,0,O_PREC},
        {"cases",required_argument,0,O_CASES}, {"seed",no_argument,0,O_SEED},
        {"force",no_argument,0,O_FORCE}, {"append-to-ledger",no_argument,0,O_LEDGER},
        {"tag",required_argument,0,O_TAG}, {"list-gpus",no_argument,0,O_LISTGPU},
        {"keep-candidate-images",no_argument,0,O_KEEPC}, {"keep-reference-images",no_argument,0,O_KEEPR},
        {"refresh-cache",no_argument,0,O_REFRESH}, {"no-cache",no_argument,0,O_NOCACHE},
        {"cache-dir",required_argument,0,O_CACHEDIR}, {"budget-gb",required_argument,0,O_BUDGET},
        {"build-commit",no_argument,0,O_BUILDCOMMIT}, {"skip-device",no_argument,0,O_SKIPDEV},
        {"expected-dir",required_argument,0,O_EXPDIR}, {"ledger-dir",required_argument,0,O_LEDGERDIR},
        {"canary-fast",no_argument,0,O_CANFAST},
        {"help",no_argument,0,O_HELP}, {0,0,0,0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "c:", lo, NULL)) != -1) {
        switch (c) {
            case O_SUITE: suite = optarg; break;
            case 'c': case O_CAND: candidate = optarg; break;
            case O_GPU: gpu = optarg; break;
            case O_WORKDIR: workdir = optarg; break;
            case O_REF: reference_flag = optarg; break;
            case O_PREC: precision = optarg; break;
            case O_CASES: {
                char *dash = strchr(optarg, '-');
                if (dash) { *dash = '\0'; range_lo = (int)strtol(optarg, NULL, 10); range_hi = (int)strtol(dash+1, NULL, 10); }
                else { range_lo = range_hi = (int)strtol(optarg, NULL, 10); }
                break;
            }
            case O_SEED: do_seed = 1; break;
            case O_FORCE: force = 1; break;
            case O_LEDGER: append_ledger = 1; break;
            case O_TAG: tag = optarg; break;
            case O_LISTGPU: list_only = 1; break;
            case O_KEEPC: keep_cand = 1; break;
            case O_KEEPR: keep_ref = 1; break;
            case O_REFRESH: refresh_cache = 1; break;
            case O_NOCACHE: no_cache = 1; break;
            case O_CACHEDIR: cache_dir_flag = optarg; break;
            case O_BUDGET: flag_budget = strtol(optarg, NULL, 10); break;
            case O_BUILDCOMMIT: puts(NB_BUILD_COMMIT); return 0;
            case O_SKIPDEV: skip_device = 1; break;
            case O_EXPDIR: expected_dir_flag = optarg; break;
            case O_LEDGERDIR: ledger_dir_flag = optarg; break;
            case O_CANFAST: canary_fast = 1; break;
            case O_HELP: usage(); return 0;
            default: usage(); return 2;
        }
    }

    if (strcmp(precision, "fp32") != 0 && strcmp(precision, "df64") != 0)
        die("--precision must be fp32 or df64");

    /* --list-gpus: no other flags required. */
    if (list_only) { list_gpus(); return 0; }

    if (!suite)     die("--suite NAME required");
    if (!candidate) die("--candidate PATH (-c) required");
    if (!workdir)   die("--workdir PATH required");
    if (!reference_flag) die("--reference PATH required (see INPUTS.md)");
    if (!gpu && !skip_device) die("--gpu required (or --skip-device for a no-GPU test run)");

    /* ---- device selection (§5): resolved (and refused) before any suite work,
       so a bad --gpu never renders. NVML enumerates every card with no CUDA context
       and no visibility pin, so this parent process keeps a CLEAN environment; the
       resolved device's own NVML UUID/name ARE the ground-truth identity (no
       pin-then-reprobe). The "GPU-<uuid>" pin is applied only in each render child's
       environment (do_render), never on this process. */
    gpu_dev match; memset(&match, 0, sizeof match);
    int have_device = 0;
    if (!skip_device) {
        resolve_gpu(gpu, &match);
        snprintf(g_child_cvd, sizeof g_child_cvd, "%s", match.uuid);
        have_device = 1;
        fprintf(stderr, "# device: index=%d name=\"%s\" uuid=%s -> child CUDA_VISIBLE_DEVICES=%s\n",
                match.index, match.name, match.uuid, g_child_cvd);
    } else {
        fprintf(stderr, "# --skip-device: no GPU selected (test pipeline)\n");
    }

    /* base.json anchors the harness root and gate defaults (§11). Resolve it from
       this executable's location: <build>/../spec/base.json (build/ is a sibling
       of spec/ under the harness root). */
    char exe[4096]; ssize_t el = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (el <= 0) die("cannot resolve executable path");
    exe[el] = '\0';
    char *exe_dir = strdup(exe);
    { char *slash = strrchr(exe_dir, '/'); if (slash) *slash = '\0'; }   /* .../build */
    char base_json[4096], nbmetrics[4096];
    snprintf(base_json, sizeof base_json, "%s/../spec/base.json", exe_dir);
    snprintf(nbmetrics, sizeof nbmetrics, "%s/nbmetrics", exe_dir);

    cc_base *base = cc_load_base(base_json);
    if (!base) die("cannot load spec/base.json (expected beside the build dir)");
    const char *harness_root = base->harness_root_abs;

    /* reference: required, no default (validated above). See INPUTS.md for what
       the trusted CPU oracle must be. */
    char reference[4096];
    snprintf(reference, sizeof reference, "%s", reference_flag);

    /* Binaries must be ABSOLUTE: every render runs in an isolated cwd (do_render
       chdir's to the scratch dir), so a relative --candidate/--reference would not
       resolve from there. */
    static char candidate_abs[4096];
    { char *r = realpath(candidate, NULL);
      if (r) { snprintf(candidate_abs, sizeof candidate_abs, "%s", r); free(r); candidate = candidate_abs; } }
    { char *r = realpath(reference, NULL);
      if (r) { snprintf(reference, sizeof reference, "%s", r); free(r); } }

    /* expected dir default: <harness_root>/expected */
    char expdir[1024];
    if (expected_dir_flag) snprintf(expdir, sizeof expdir, "%s", expected_dir_flag);
    else snprintf(expdir, sizeof expdir, "%s/expected", harness_root);

    /* suite file: a path ending in .jsonl, else <harness_root>/suites/<name>.jsonl */
    char suite_file[4096]; const char *suite_name;
    struct stat sst;
    if (strstr(suite, ".jsonl") && stat(suite, &sst) == 0) {
        snprintf(suite_file, sizeof suite_file, "%s", suite);
        const char *bn = strrchr(suite, '/'); bn = bn ? bn + 1 : suite;
        static char sname[128]; snprintf(sname, sizeof sname, "%s", bn);
        char *dot = strstr(sname, ".jsonl"); if (dot) *dot = '\0';
        suite_name = sname;
    } else {
        snprintf(suite_file, sizeof suite_file, "%s/suites/%s.jsonl", harness_root, suite);
        suite_name = suite;
    }

    cc_suite *S = cc_load_suite(suite_file, base);
    if (!S) { fprintf(stderr, "nbrunsuite: cannot load suite %s\n", suite_file); return 2; }

    mkdir(workdir, 0755);
    /* Resolve workdir to absolute too: scratch/output paths under it are handed to
       children that have chdir'd away, and to nbmetrics from the parent cwd. */
    static char workdir_abs[512];
    { char *r = realpath(workdir, NULL);
      if (r) { snprintf(workdir_abs, sizeof workdir_abs, "%s", r); free(r); workdir = workdir_abs; } }
    char scratch[1024]; snprintf(scratch, sizeof scratch, "%s/scratch", workdir);
    mkdir(scratch, 0755);

    /* input_root for {input_root} token expansion */
    char *input_root = cc_input_root_abs(base);
    if (!input_root) die("cannot resolve input_root (does inputs/ exist?)");

    const char *prec_args = (strcmp(precision, "df64") == 0) ? "-precision double" : "-precision single";

    /* ---- --seed reference canary (before anything is recorded) ------------- */
    if (do_seed) {
        if (reference_canary(base, harness_root, reference, workdir, canary_fast) != 0) {
            fprintf(stderr, "nbrunsuite: --seed ABORTED (reference canary failed)\n");
            return 5;
        }
        fprintf(stderr, "# canary PASSED -- reference reproduces gold on all three canaries\n");
    }

    /* ---- cache setup (§9) -------------------------------------------------- */
    char *cache_dir = cache_dir_resolve(cache_dir_flag);
    if (!cache_dir) die("cannot resolve cache dir");
    long budget_gb = cache_budget_resolve(cache_dir, flag_budget >= 0 ? flag_budget : base->cache_budget_gb);

    char ref_md5[33];
    if (cache_reference_md5(reference, ref_md5) != 0) {
        fprintf(stderr, "nbrunsuite: cannot read reference binary %s\n", reference);
        return 2;
    }
    char cand_md5[33];
    if (cache_reference_md5(candidate, cand_md5) != 0) {
        fprintf(stderr, "nbrunsuite: cannot read candidate binary %s\n", candidate);
        return 2;
    }

    /* cache_gc at suite start over this suite's live set (§6). */
    if (!no_cache) {
        int nlive = S->n;
        const char **live = malloc((size_t)nlive * sizeof(char *));
        long *live_k = malloc((size_t)nlive * sizeof(long));
        int lc = 0;
        for (int i = 0; i < S->n; i++)
            if (S->cases[i].args_hash && strlen(S->cases[i].args_hash) == CACHE_HEXLEN) {
                live[lc] = S->cases[i].args_hash; live_k[lc] = S->cases[i].K; lc++;
            }
        int n;
        cache_entry *e = cache_scan(cache_dir, live, live_k, lc, &n);
        if (e) {
            int oe, be;
            cache_gc(cache_dir, e, n, (long long)budget_gb << 30, time(NULL), 1, &oe, &be);
            free(e);
        }
        free(live); free(live_k);
    }

    /* ---- results.tsv header ----------------------------------------------- */
    char results_path[4096]; snprintf(results_path, sizeof results_path, "%s/results.tsv", workdir);
    FILE *rf = fopen(results_path, (range_lo <= 1) ? "wb" : "ab");
    if (!rf) die("cannot open results.tsv");
    if (range_lo <= 1)
        fprintf(rf, "# cell\tcorr\tsum_ratio\tms\tverdict\tfp16_corr\tfp16_sum_ratio\tfp16_diff\tfp16_excl\tnote\n");

    printf("%-34s %-11s %-10s %-9s %-8s %s\n", "cell", "corr", "sum_ratio", "ms", "verdict", "note");

    results_t R; res_init(&R);

    /* ---- per-case loop ---------------------------------------------------- */
    for (int idx = 0; idx < S->n; idx++) {
        int caseno = idx + 1;
        if (caseno < range_lo || caseno > range_hi) continue;
        cc_case *cs = &S->cases[idx];
        case_result *out = res_add(&R);
        snprintf(out->id, sizeof out->id, "%s", cs->id ? cs->id : "?");
        strcpy(out->verdict, "BLOCKED");
        out->ms = 0.0;

        char *cand_args = cc_resolve_input_root(cs->candidate_args, input_root);
        char *ref_args  = cc_resolve_input_root(cs->reference_args, input_root);

        char candlog[4096]; snprintf(candlog, sizeof candlog, "%s/cand_%s.log", scratch, out->id);

        /* ---- reject gate: candidate render only, classify exit code ------- */
        if (cs->gate_type == CC_GATE_REJECT) {
            char gout[4096]; snprintf(gout, sizeof gout, "%s/%s.cand.bin", workdir, out->id);
            unlink(gout);
            if (!strstr(cand_args, "-hkl")) { char fd[1152]; snprintf(fd,sizeof fd,"%s/Fdump.bin",scratch); unlink(fd); }
            argv_t av; build_render_argv(&av, candidate, gout, cand_args, prec_args);
            double secs; int rc = do_render(candidate, &av, scratch, candlog, &secs);
            av_free(&av);
            out->ms = secs * 1000.0;
            if (rc == 9)      { strcpy(out->verdict, "REJECT"); snprintf(out->note, sizeof out->note, "rejected(exit9)"); }
            else if (rc == 0) { strcpy(out->verdict, "FAIL");   snprintf(out->note, sizeof out->note, "silent-no-op(exit0)"); }
            else              { strcpy(out->verdict, "BLOCKED");snprintf(out->note, sizeof out->note, "reject-unexpected(exit %d)", rc); }
            unlink(gout);
            fprintf(rf, "%s\t-\t-\t%.1f\t%s\t-\t-\t-\t-\t%s\n", out->id, out->ms, out->verdict, out->note);
            printf("%-34s %-11s %-10s %-9.1f %-8s %s\n", out->id, "-", "-", out->ms, out->verdict, out->note);
            free(cand_args); free(ref_args);
            continue;
        }

        /* ---- reference image: cache hit, else render + store -------------- */
        char cachebin[4096] = "";
        int have_ref = 0;
        char refimg[4096];
        if (!no_cache && cs->args_hash && strlen(cs->args_hash) == CACHE_HEXLEN) {
            cache_entry_path(cache_dir, ref_md5, cs->args_hash, ".bin", cachebin, sizeof cachebin);
            struct stat cst;
            if (!refresh_cache && stat(cachebin, &cst) == 0 && cst.st_size > 0) {
                snprintf(refimg, sizeof refimg, "%s", cachebin);   /* HIT */
                have_ref = 1;
                snprintf(out->note, sizeof out->note, "cache=hit");
            }
        }
        if (!have_ref) {
            char rout[4096]; snprintf(rout, sizeof rout, "%s/%s.ref.bin", workdir, out->id);
            unlink(rout);
            if (!strstr(ref_args, "-hkl")) { char fd[4096]; snprintf(fd,sizeof fd,"%s/Fdump.bin",scratch); unlink(fd); }
            char reflog[4096]; snprintf(reflog, sizeof reflog, "%s/ref_%s.log", scratch, out->id);
            argv_t av; build_render_argv(&av, reference, rout, ref_args, NULL);
            double secs; int rc = do_render(reference, &av, scratch, reflog, &secs);
            av_free(&av);
            struct stat rst;
            if (rc != 0 || stat(rout, &rst) != 0 || rst.st_size == 0) {
                strcpy(out->verdict, "BLOCKED");
                snprintf(out->note, sizeof out->note, "reference-render-fail(exit %d)", rc);
                fprintf(rf, "%s\t-\t-\t%.1f\t%s\t-\t-\t-\t-\t%s\n", out->id, 0.0, out->verdict, out->note);
                printf("%-34s %-11s %-10s %-9s %-8s %s\n", out->id, "-", "-", "-", out->verdict, out->note);
                free(cand_args); free(ref_args); continue;
            }
            /* store into the cache (unless bypassed) */
            if (!no_cache && cachebin[0]) {
                FILE *bf = fopen(rout, "rb");
                if (bf) {
                    fseek(bf, 0, SEEK_END); long sz = ftell(bf); rewind(bf);
                    void *buf = malloc((size_t)sz);
                    if (buf && fread(buf, 1, (size_t)sz, bf) == (size_t)sz) {
                        cache_write_atomic(cachebin, buf, (size_t)sz);
                        cache_write_meta(cache_dir, ref_md5, cs->args_hash, cs->K, secs);
                    }
                    free(buf); fclose(bf);
                }
            }
            snprintf(refimg, sizeof refimg, "%s", rout);
            snprintf(out->note, sizeof out->note, "cache=miss");
            have_ref = 1;
            if (!keep_ref && cachebin[0] && !no_cache) { /* keep the workdir copy only if asked; cache holds it */ unlink(rout); snprintf(refimg, sizeof refimg, "%s", cachebin); }
        }

        /* ---- candidate render (min-of-5 for perf, else once) -------------- */
        char gout[4096]; snprintf(gout, sizeof gout, "%s/%s.cand.bin", workdir, out->id);
        int nrep = (cs->gate_type == CC_GATE_PERF) ? 5 : 1;
        double best_ms = -1.0; int rc = -1;
        for (int rep = 0; rep < nrep; rep++) {
            unlink(gout);
            if (!strstr(cand_args, "-hkl")) { char fd[4096]; snprintf(fd,sizeof fd,"%s/Fdump.bin",scratch); unlink(fd); }
            argv_t av; build_render_argv(&av, candidate, gout, cand_args, prec_args);
            double secs; rc = do_render(candidate, &av, scratch, candlog, &secs);
            av_free(&av);
            struct stat gst;
            if (rc != 0 || stat(gout, &gst) != 0 || gst.st_size == 0) break;
            double ms = secs * 1000.0;
            if (best_ms < 0 || ms < best_ms) best_ms = ms;
        }
        out->ms = best_ms < 0 ? 0.0 : best_ms;

        struct stat gst;
        if (rc != 0 || stat(gout, &gst) != 0 || gst.st_size == 0) {
            strcpy(out->verdict, "BLOCKED");
            snprintf(out->note, sizeof out->note, "candidate-render-fail(exit %d)", rc);
            fprintf(rf, "%s\t-\t-\t%.1f\t%s\t-\t-\t-\t-\t%s\n", out->id, out->ms, out->verdict, out->note);
            printf("%-34s %-11s %-10s %-9.1f %-8s %s\n", out->id, "-", "-", out->ms, out->verdict, out->note);
            if (!keep_cand) unlink(gout);
            free(cand_args); free(ref_args); continue;
        }

        /* ---- compare + gate ---------------------------------------------- */
        metrics_t m = run_metrics(nbmetrics, gout, refimg);
        if (!m.ok) {
            strcpy(out->verdict, "BLOCKED");
            snprintf(out->note, sizeof out->note, "metric-fail");
            fprintf(rf, "%s\t-\t-\t%.1f\t%s\t-\t-\t-\t-\t%s\n", out->id, out->ms, out->verdict, out->note);
            printf("%-34s %-11s %-10s %-9.1f %-8s %s\n", out->id, "-", "-", out->ms, out->verdict, out->note);
            if (!keep_cand) unlink(gout);
            free(cand_args); free(ref_args); continue;
        }

        int pass = (m.corr >= cs->corr_min && m.sum_ratio >= cs->sum_ratio_min && m.sum_ratio <= cs->sum_ratio_max);
        if (cs->gate_type == CC_GATE_PERF) {
            /* warn-not-fail: a parity miss is diagnosed, the tier is forced PASS. */
            strcpy(out->verdict, "PASS");
            if (!pass) { fprintf(stderr, "# perf-warn: %s corr=%.7f sum_ratio=%.6f\n", out->id, m.corr, m.sum_ratio); }
            snprintf(out->note, sizeof out->note, "perf min_ms=%.1f%s", out->ms, pass ? "" : " (parity-warn)");
        } else {
            strcpy(out->verdict, pass ? "PASS" : "FAIL");
        }

        fprintf(rf, "%s\t%.7f\t%.6f\t%.1f\t%s\t%.7f\t%.6f\t%lld\t%lld\t%s\n",
                out->id, m.corr, m.sum_ratio, out->ms, out->verdict,
                m.fp16_corr, m.fp16_sum_ratio, m.fp16_diff, m.fp16_excl, out->note);
        printf("%-34s %-11.7f %-10.6f %-9.1f %-8s %s\n", out->id, m.corr, m.sum_ratio, out->ms, out->verdict, out->note);

        if (!keep_cand) unlink(gout);
        free(cand_args); free(ref_args);
    }

    int is_final = (range_hi >= S->n);
    fclose(rf);                       /* flush results.tsv before the whole-file tally */

    /* ---- whole-file tally + flip detection over the accumulated results.tsv
       (so --cases batches combine, and a SKIP row is classified, not counted). */
    loaded_t LR = load_results(results_path);
    char exp_file[4096]; expected_path(exp_file, sizeof exp_file, expdir, suite_name, precision);
    expected_t E = load_expected(exp_file);
    int is_perf = (S->n > 0 && S->cases[0].gate_type == CC_GATE_PERF);

    int flips_pf = 0, flips_fp = 0, flips_other = 0, gmatch = 0;
    char flipbuf[8192] = ""; size_t flen = 0;
    if (E.present && is_final && !is_perf) {
        for (int i = 0; i < LR.n; i++) {
            const char *v = LR.r[i].verdict;
            if (strcmp(v, "SKIP") == 0) continue;                       /* SKIP is not a flip */
            const char *ev = expected_verdict(&E, LR.r[i].id);
            if (!ev) { flips_other++; flen += (size_t)snprintf(flipbuf+flen, sizeof flipbuf-flen, "\n    %s : absent-from-expected actual=%s", LR.r[i].id, v); }
            else if (strcmp(v, ev) == 0) gmatch++;
            else if (strcmp(ev,"PASS")==0 && strcmp(v,"FAIL")==0) { flips_pf++; flen += (size_t)snprintf(flipbuf+flen, sizeof flipbuf-flen, "\n    %s : pass -> fail", LR.r[i].id); }
            else if (strcmp(ev,"FAIL")==0 && strcmp(v,"PASS")==0) { flips_fp++; flen += (size_t)snprintf(flipbuf+flen, sizeof flipbuf-flen, "\n    %s : fail -> pass", LR.r[i].id); }
            else { flips_other++; flen += (size_t)snprintf(flipbuf+flen, sizeof flipbuf-flen, "\n    %s : %s -> %s", LR.r[i].id, ev, v); }
        }
        for (int i = 0; i < E.n; i++) {                                 /* expected but not in results */
            int seen = 0;
            for (int j = 0; j < LR.n; j++) if (strcmp(E.r[i].id, LR.r[j].id) == 0) { seen = 1; break; }
            if (!seen) { flips_other++; flen += (size_t)snprintf(flipbuf+flen, sizeof flipbuf-flen, "\n    %s : missing-from-results (expected %s)", E.r[i].id, E.r[i].verdict); }
        }
    }
    int nflip = flips_pf + flips_fp + flips_other;

    int npass = 0, nfail = 0, nblocked = 0, nreject = 0, nskip = 0;
    for (int i = 0; i < LR.n; i++) {
        const char *v = LR.r[i].verdict;
        if (strcmp(v, "PASS") == 0) npass++;
        else if (strcmp(v, "FAIL") == 0) nfail++;
        else if (strcmp(v, "REJECT") == 0) nreject++;
        else if (strcmp(v, "SKIP") == 0) nskip++;
        else nblocked++;
    }
    const char *tier;
    if (is_perf) tier = "PASS";                          /* warn-not-fail */
    else if (E.present && is_final) tier = (nflip == 0) ? "PASS" : "FAIL";
    else if (!is_final) tier = "PARTIAL";
    else tier = (nfail == 0 && nblocked == 0) ? "PASS" : "FAIL";

    /* ---- provenance trailer (final batch only) ---------------------------- */
    if (is_final) {
        FILE *tf = fopen(results_path, "ab");
        if (tf) {
            time_t now = time(NULL); struct tm tmv; gmtime_r(&now, &tmv);
            char utc[32]; strftime(utc, sizeof utc, "%Y-%m-%dT%H:%M:%SZ", &tmv);
            char host[256]; if (gethostname(host, sizeof host) != 0) snprintf(host, sizeof host, "unknown");
            fprintf(tf, "# ==============================================================\n");
            fprintf(tf, "# TIER %s %s %d/%d   (PASS=%d FAIL=%d BLOCKED=%d REJECT=%d SKIP=%d)\n",
                    suite_name, tier, npass, S->n, npass, nfail, nblocked, nreject, nskip);
            if (E.present && !is_perf)
                fprintf(tf, "# expected=%s  flips=%d (pass->fail=%d fail->pass=%d other=%d)\n", exp_file, nflip, flips_pf, flips_fp, flips_other);
            else
                fprintf(tf, "# expected=<none for %s.%s>\n", suite_name, precision);
            fprintf(tf, "# candidate=%s md5=%s\n", candidate, cand_md5);
            fprintf(tf, "# reference=%s md5=%s\n", reference, ref_md5);
            fprintf(tf, "# gpu_name=%s\n", have_device ? match.name : "none(skip-device)");
            fprintf(tf, "# precision=%s\n", precision);
            fprintf(tf, "# commit=%s\n", NB_BUILD_COMMIT);
            fprintf(tf, "# utc=%s  host=%s\n", utc, host);
            fclose(tf);
        }
    }

    printf("# ==============================================================\n");
    printf("# TIER %s %s %d/%d  (PASS=%d FAIL=%d BLOCKED=%d REJECT=%d SKIP=%d)\n",
           suite_name, tier, npass, S->n, npass, nfail, nblocked, nreject, nskip);
    if (E.present && is_final && !is_perf)
        printf("# expected %s.%s: %d flips (%d cells match) -> suite %s%s\n",
               suite_name, precision, nflip, gmatch, tier, nflip ? flipbuf : "");

    /* ---- --seed: record verdicts as expected (flip-guarded) --------------- */
    int rc_final = 0;
    if (do_seed && is_final) {
        /* refuse to change any existing verdict unless --force */
        int changes = 0; char changebuf[8192] = ""; size_t cl = 0;
        for (int i = 0; i < LR.n; i++) {
            const char *v = LR.r[i].verdict;
            if (strcmp(v, "SKIP") == 0 || strcmp(v, "BLOCKED") == 0) continue;
            const char *ev = expected_verdict(&E, LR.r[i].id);
            if (ev && strcmp(ev, v) != 0) {
                changes++;
                cl += (size_t)snprintf(changebuf+cl, sizeof changebuf-cl, "\n    %s : %s -> %s", LR.r[i].id, ev, v);
            }
        }
        if (changes > 0 && !force) {
            fprintf(stderr, "nbrunsuite: --seed REFUSED -- %d verdict flip(s) vs existing %s (use --force):%s\n",
                    changes, exp_file, changebuf);
            rc_final = 6;
        } else {
            mkdir(expdir, 0755);
            FILE *ef = fopen(exp_file, "wb");
            if (!ef) die("cannot write expected file");
            fprintf(ef, "# expected verdicts  suite=%s  precision=%s  (seeded by nbrunsuite)\n", suite_name, precision);
            fprintf(ef, "# cell\tverdict\tcorr\tsum_ratio\n");
            for (int i = 0; i < LR.n; i++) {
                if (strcmp(LR.r[i].verdict, "SKIP") == 0) continue;
                fprintf(ef, "%s\t%s\t%s\t%s\n", LR.r[i].id, LR.r[i].verdict, LR.r[i].corr, LR.r[i].sr);
            }
            fclose(ef);
            fprintf(stderr, "# --seed: wrote %s (%d cells%s)\n", exp_file, LR.n, changes ? ", forced verdict changes" : "");
        }
    }

    /* ---- --append-to-ledger ----------------------------------------------- */
    if (append_ledger && is_final && rc_final == 0) {
        char runs_tsv[4096], res_tsv[4096];
        const char *ledger_dir = ledger_dir_flag ? ledger_dir_flag : NULL;
        if (ledger_dir) {
            snprintf(runs_tsv, sizeof runs_tsv, "%s/runs.tsv", ledger_dir);
            snprintf(res_tsv,  sizeof res_tsv,  "%s/results.tsv", ledger_dir);
            mkdir(ledger_dir, 0755);
        } else {
            snprintf(runs_tsv, sizeof runs_tsv, "%s/ledger/runs.tsv", harness_root);
            snprintf(res_tsv,  sizeof res_tsv,  "%s/ledger/results.tsv", harness_root);
        }
        time_t now = time(NULL); struct tm tmv; gmtime_r(&now, &tmv);
        char utc[32]; strftime(utc, sizeof utc, "%Y-%m-%dT%H:%M:%SZ", &tmv);
        char utc_compact[32]; int k=0; for (const char *p=utc; *p; p++) if (*p!='-'&&*p!=':') utc_compact[k++]=*p; utc_compact[k]='\0';
        char host[256]; if (gethostname(host,sizeof host)!=0) snprintf(host,sizeof host,"unknown");
        char run_id[512];
        snprintf(run_id, sizeof run_id, "%s-%.8s-%s%s%s", utc_compact, cand_md5, host, tag?"-":"", tag?tag:"");
        FILE *runs = fopen(runs_tsv, "ab");
        if (runs) {
            fprintf(runs, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                    run_id, utc, host, have_device?match.name:"none", "unknown", "unknown",
                    NB_BUILD_COMMIT, suite_name, precision, cand_md5, ref_md5);
            fclose(runs);
        }
        FILE *res = fopen(res_tsv, "ab");
        if (res) {
            for (int i = 0; i < LR.n; i++)
                fprintf(res, "%s\t%s\t%s\t%s\t%s\t%s\n", run_id, LR.r[i].id, LR.r[i].verdict, LR.r[i].corr, LR.r[i].sr, LR.r[i].ms);
            fclose(res);
        }
        fprintf(stderr, "# ledger: appended run %s\n", run_id);
    }

    free(input_root);
    free(cache_dir);
    free(exe_dir);
    if (rc_final) return rc_final;         /* --seed refused (verdict flip, no --force) */
    if (do_seed) return 0;                 /* a written/forced baseline is success, regardless
                                              of flips vs the OLD (now-replaced) baseline */
    if (is_final && !is_perf && E.present && nflip > 0) return 1;   /* flip = suite FAIL */
    return 0;
}
