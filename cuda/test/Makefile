# Out-of-source build for the nanoBragg parity harness nb* tool family
# (NBTOOLS-SPEC.md §13/§14: argkey, case_core, cache_core are shared objects;
# nbgensuite/nbcache/nbmetrics are the linked tools). Retired standalone tools
# (canon, gen_suites, metrics) are not built here; canon.c stays on disk only to
# build canon_ref, a test-only cross-check binary for test_argkey.
#
# Build prerequisites (system packages): gcc, json-c-devel (JSON parsing, via
# <json-c/json.h> + -ljson-c), libmd-devel (md5, via <md5.h> + -lmd). The
# harness may depend on system libraries; the nanoBragg binary itself does not.
# nbrunsuite ADDITIONALLY needs the CUDA toolkit dir for NVML: <nvml.h> at compile
# and the stub libnvidia-ml.so (-lnvidia-ml) at link; at runtime the NVIDIA driver's
# libnvidia-ml.so.1 loads (no cudart at all). The toolkit dir is auto-detected and,
# when absent, only nbrunsuite is skipped -- the CUDA-free core and `make test`
# build with just gcc/json-c/libmd regardless.
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

# CUDA detection (nbrunsuite only; the rest of the nb* family is CUDA-free). A
# toolkit dir is resolved robustly -- $CUDA_HOME, then $CUDA_PATH, then the parent
# of an nvcc on PATH, then /usr/local/cuda -- accepting the first that actually
# carries <cuda_runtime.h>. When none is found, nbrunsuite drops out of the default
# build and its alias errors, so `make` and `make test` still succeed on a CUDA-less
# box with only gcc/json-c/libmd.
NVCC_BIN := $(shell command -v nvcc 2>/dev/null)
CUDA_DIR := $(firstword $(foreach d,$(CUDA_HOME) $(CUDA_PATH) \
              $(if $(NVCC_BIN),$(abspath $(dir $(NVCC_BIN))/..)) /usr/local/cuda,\
              $(if $(wildcard $(d)/include/cuda_runtime.h),$(d))))
ifneq ($(CUDA_DIR),)
CUDA_CFLAGS := -I$(CUDA_DIR)/include
CUDA_LDLIBS := -L$(CUDA_DIR)/lib64/stubs -lnvidia-ml
endif

CORE_TOOLS := $(BINDIR)/nbgensuite $(BINDIR)/nbcache $(BINDIR)/nbmetrics
TOOLS      := $(CORE_TOOLS) $(if $(CUDA_DIR),$(BINDIR)/nbrunsuite)
TESTS := $(BINDIR)/test_argkey $(BINDIR)/test_case_core $(BINDIR)/test_nbmetrics $(BINDIR)/test_cache_core

.PHONY: all clean test nbgensuite nbcache nbmetrics nbrunsuite

all: $(TOOLS)

# Convenience aliases so 'make nbcache' / 'make nbgensuite' work regardless
# of BINDIR.
nbgensuite: $(BINDIR)/nbgensuite
nbcache:    $(BINDIR)/nbcache
nbmetrics:  $(BINDIR)/nbmetrics
ifneq ($(CUDA_DIR),)
nbrunsuite: $(BINDIR)/nbrunsuite
else
nbrunsuite:
	@echo "nbrunsuite needs the CUDA toolkit dir for NVML (nvml.h + stub libnvidia-ml.so); none found." >&2
	@echo "Set CUDA_HOME or put nvcc on PATH. The rest of the nb* family builds without it." >&2
	@exit 1
endif

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
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $(DEPFLAGS) -DNB_BUILD_COMMIT='"$(BUILD_COMMIT)"' -c -o $@ $<

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

# nbrunsuite links the shared parity core (cache_core needs libmd), NVML for GPU
# enumeration (-lnvidia-ml, driver's management lib -- no cudart), and drives
# nbmetrics as a subprocess; -lm for the gate/metric arithmetic.
$(BINDIR)/nbrunsuite: $(BINDIR)/nbrunsuite.o $(BINDIR)/cache_core.o $(BINDIR)/argkey.o $(BINDIR)/case_core.o
	$(CC) $(CFLAGS) -o $@ $^ -lmd -ljson-c $(CUDA_LDLIBS) $(LDLIBS)

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
