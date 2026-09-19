#!/usr/bin/env bash
# Build the C accuracy oracle for the parity matrix.
#
# The oracle is bl831/main plus the three fix branches that bl831's own GPU harness
# uses as its CPU reference. They are unmerged upstream, so the oracle only exists as
# a local merge; this script reproduces it from public refs and checks the result
# against a pinned checksum, so every parity number can be traced to a build.
#
#   scripts/oracle/build_oracle.sh [target_dir]      # default: ./oracle
#
# Prints the binary path on success. Set NB_C_BIN to it when running the parity matrix:
#   NB_C_BIN=$(scripts/oracle/build_oracle.sh | tail -1) pytest tests/test_parity_matrix.py
set -euo pipefail

TARGET=${1:-oracle}
BL831_URL=${BL831_URL:-https://github.com/bl831/nanoBragg.git}
BASE_REF=${BASE_REF:-main}
FIX_BRANCHES=(fix/phi0-stale-rotation fix/subpixel-oversampling fix/curved-det-flag-guard)
# md5 of the merged nanoBragg.c this repo's parity thresholds were measured against
# (bl831/main 0d24e6e, 2026-08-04, plus the three branches above).
EXPECTED_MD5=${EXPECTED_MD5:-0a8675a1ce4b19e0d0039542e1a2099d}
# Deterministic float behaviour: no fast-math, no FMA contraction.
CFLAGS_ORACLE=${CFLAGS_ORACLE:--O2 -fno-fast-math -ffp-contract=off}

mkdir -p "$TARGET"
cd "$TARGET"
if [ ! -d src/.git ]; then
  git clone -q "$BL831_URL" src
fi
cd src
git fetch -q origin
git checkout -q --detach "origin/$BASE_REF"
for b in "${FIX_BRANCHES[@]}"; do
  git merge -q --no-edit "origin/$b" >/dev/null
done
BASE_SHA=$(git rev-parse --short "origin/$BASE_REF")

if command -v md5sum >/dev/null; then ACTUAL_MD5=$(md5sum nanoBragg.c | cut -d' ' -f1); else ACTUAL_MD5=$(md5 -q nanoBragg.c); fi
if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
  echo "WARNING: merged nanoBragg.c md5 $ACTUAL_MD5 != pinned $EXPECTED_MD5" >&2
  echo "         bl831/$BASE_REF is now $BASE_SHA; re-measure the parity thresholds before trusting them." >&2
fi

# shellcheck disable=SC2086
${CC:-gcc} $CFLAGS_ORACLE -o ../nanoBragg nanoBragg.c -lm
cd ..
cat > ORACLE_PROVENANCE <<PROV
bl831/$BASE_REF $BASE_SHA + ${FIX_BRANCHES[*]}
nanoBragg.c md5 $ACTUAL_MD5
built $(date -u +%Y-%m-%dT%H:%M:%SZ) with ${CC:-gcc} $CFLAGS_ORACLE
PROV
echo "$PWD/nanoBragg"
