# Out-of-source build for the nanoBragg parity harness.
#
# Compiles src/*.c into the gitignored build dir cuda/test/build/:
#   cuda/test/src/gen_suites.c  -> $(BINDIR)/gen_suites
#   cuda/test/src/metrics.c    -> $(BINDIR)/metrics
#
# BINDIR defaults to cuda/test/build/ and may be overridden (run.sh points it
# there explicitly). Build output is gitignored; only src/ is committed.

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

# This Makefile lives in cuda/test/, so the default build dir is cuda/test/build.
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BINDIR  ?= $(abspath $(HERE)build)

# Build-time commit bake (Layer-2 ledger provenance): metrics is stamped with the
# commit HEAD was at when it was last compiled, queried at runtime via
# `metrics --build-commit`. gen_suites stays commit-agnostic -- its output is
# reviewed by diff, not stamped.
BUILD_COMMIT := $(shell git -C $(HERE) rev-parse HEAD 2>/dev/null || echo unknown)

BINS := $(BINDIR)/gen_suites $(BINDIR)/metrics

.PHONY: all clean gen_suites metrics
all: $(BINS)

# Convenience aliases so 'make metrics' / 'make gen_suites' work regardless of BINDIR.
gen_suites: $(BINDIR)/gen_suites
metrics:   $(BINDIR)/metrics

$(BINDIR)/gen_suites: src/gen_suites.c | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BINDIR)/metrics: src/metrics.c | $(BINDIR)
	$(CC) $(CFLAGS) -DNB_BUILD_COMMIT='"$(BUILD_COMMIT)"' -o $@ $< $(LDLIBS)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -f $(BINS)
