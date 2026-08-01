# Out-of-source build for the nanoBragg parity harness nb* tool family
# (NBTOOLS-SPEC.md §13/§14: argkey, case_core, cache_core are shared objects;
# nbgensuite/nbcache/nbmetrics are the linked tools). Retired standalone tools
# (canon, gen_suites, metrics) are not built here; canon.c stays on disk only to
# build canon_ref, a test-only cross-check binary for test_argkey.
#
# Build prerequisites (system packages): gcc, json-c-devel (JSON parsing, via
# <json-c/json.h> + -ljson-c), libmd-devel (md5, via <md5.h> + -lmd). The
# harness may depend on system libraries; the nanoBragg binary itself does not.
#
# Objects-then-link, static, no .so. Build output goes to the gitignored
# cuda/test/build/; only src/ is committed. BINDIR is resolved from this
# Makefile's own path, so `make`, `make -C cuda/test`, and `cd cuda/test &&
# make` all land in the same place regardless of invocation CWD.

CC       ?= cc
CFLAGS   ?= -std=c11 -O2 -Wall -Wextra
DEPFLAGS  = -MMD -MP
LDLIBS   ?= -lm

HERE     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BINDIR   ?= $(abspath $(HERE)build)
ROOT     := $(abspath $(HERE)../..)

# nbcache/nbmetrics bake the commit HEAD was at when last compiled, queried
# at runtime via `<tool> --build-commit`.
BUILD_COMMIT := $(shell git -C $(HERE) rev-parse --short HEAD 2>/dev/null || echo unknown)

TOOLS := $(BINDIR)/nbgensuite $(BINDIR)/nbcache $(BINDIR)/nbmetrics $(BINDIR)/nbrunsuite
TESTS := $(BINDIR)/test_argkey $(BINDIR)/test_case_core $(BINDIR)/test_nbmetrics $(BINDIR)/test_cache_core

.PHONY: all clean test nbgensuite nbcache nbmetrics nbrunsuite

all: $(TOOLS)

# Convenience aliases so 'make nbcache' / 'make nbgensuite' work regardless
# of BINDIR.
nbgensuite: $(BINDIR)/nbgensuite
nbcache:    $(BINDIR)/nbcache
nbmetrics:  $(BINDIR)/nbmetrics
nbrunsuite: $(BINDIR)/nbrunsuite

# ---------------------------------------------------------------------------
# Objects (generic rule; -MMD/-MP emit per-object .d files for header deps)
# ---------------------------------------------------------------------------

$(BINDIR)/%.o: src/%.c | $(BINDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

# nbcache.o/nbmetrics.o need the commit bake, so they override the generic
# rule above (an explicit rule for a given target always wins over a pattern
# rule).
$(BINDIR)/nbcache.o: src/nbcache.c | $(BINDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) -DNB_BUILD_COMMIT='"$(BUILD_COMMIT)"' -c -o $@ $<

$(BINDIR)/nbmetrics.o: src/nbmetrics.c | $(BINDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) -DNB_BUILD_COMMIT='"$(BUILD_COMMIT)"' -c -o $@ $<

$(BINDIR)/nbrunsuite.o: src/nbrunsuite.c | $(BINDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) -DNB_BUILD_COMMIT='"$(BUILD_COMMIT)"' -c -o $@ $<

-include $(wildcard $(BINDIR)/*.d)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

$(BINDIR)/nbgensuite: $(BINDIR)/nbgensuite.o $(BINDIR)/argkey.o $(BINDIR)/case_core.o
	$(CC) $(CFLAGS) -o $@ $^ -lmd -ljson-c

$(BINDIR)/nbcache: $(BINDIR)/nbcache.o $(BINDIR)/cache_core.o $(BINDIR)/argkey.o $(BINDIR)/case_core.o
	$(CC) $(CFLAGS) -o $@ $^ -lmd -ljson-c

$(BINDIR)/nbmetrics: $(BINDIR)/nbmetrics.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# nbrunsuite links the shared parity core (cache_core needs libmd) and drives
# nbmetrics as a subprocess; -lm for the gate/metric arithmetic.
$(BINDIR)/nbrunsuite: $(BINDIR)/nbrunsuite.o $(BINDIR)/cache_core.o $(BINDIR)/argkey.o $(BINDIR)/case_core.o
	$(CC) $(CFLAGS) -o $@ $^ -lmd -ljson-c $(LDLIBS)

# ---------------------------------------------------------------------------
# Unit tests + the test-only canon_ref cross-check binary
# ---------------------------------------------------------------------------

$(BINDIR)/canon_ref: $(BINDIR)/canon.o
	$(CC) $(CFLAGS) -o $@ $^

$(BINDIR)/test_argkey: $(BINDIR)/test_argkey.o $(BINDIR)/argkey.o
	$(CC) $(CFLAGS) -o $@ $^

$(BINDIR)/test_case_core: $(BINDIR)/test_case_core.o $(BINDIR)/case_core.o
	$(CC) $(CFLAGS) -o $@ $^ -ljson-c

$(BINDIR)/test_nbmetrics: $(BINDIR)/test_nbmetrics.o
	$(CC) $(CFLAGS) -o $@ $^

$(BINDIR)/test_cache_core: $(BINDIR)/test_cache_core.o $(BINDIR)/cache_core.o $(BINDIR)/case_core.o $(BINDIR)/argkey.o
	$(CC) $(CFLAGS) -o $@ $^ -lmd -ljson-c

$(BINDIR):
	mkdir -p $(BINDIR)

# test_case_core and test_cache_core resolve "cuda/test/spec/base.json" as a
# path relative to the worktree root, so they (and, for uniformity,
# test_argkey) run with CWD = $(ROOT). test_nbmetrics takes the tool path as
# argv[1] and uses /tmp for its fixtures, so it is CWD-independent.
test: $(TESTS) $(BINDIR)/canon_ref $(BINDIR)/nbcache $(BINDIR)/nbmetrics
	@status=0; \
	echo "== test_argkey =="; (cd $(ROOT) && $(BINDIR)/test_argkey $(BINDIR)/canon_ref) || status=1; \
	echo "== test_case_core =="; (cd $(ROOT) && $(BINDIR)/test_case_core) || status=1; \
	echo "== test_nbmetrics =="; $(BINDIR)/test_nbmetrics $(BINDIR)/nbmetrics || status=1; \
	echo "== test_cache_core =="; (cd $(ROOT) && $(BINDIR)/test_cache_core $(BINDIR)/nbcache) || status=1; \
	exit $$status

clean:
	rm -rf $(BINDIR)
