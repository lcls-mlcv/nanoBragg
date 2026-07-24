#!/bin/bash
# =============================================================================
# gen_cpu.sh -- establish + verify the CPU reference (cuda/testdata/nanoBragg_root).
#
# The CPU oracle MUST be main + fix/phi0-stale-rotation + fix/subpixel-oversampling.
# Provenance is BEHAVIORAL, not by branch name: two canary cells whose output only
# matches the both-fixes physics if BOTH fixes are present.
#
#   canary A (phi0)     : a phi-sweep cell   -- sensitive to fix/phi0-stale-rotation
#   canary B (subpixel) : an -oversample_thick cell with oversample>1
#                                              -- sensitive to fix/subpixel-oversampling
#
# The reference physics ("gold") is built FROM SOURCE every run: main:nanoBragg.c
# with the two branch diffs applied. That makes the check non-circular (it does not
# trust any cached image) and self-adapting: if the fixes are already merged into
# main, the diffs are empty and gold == main -- the check still passes, so the
# eventual PR merge needs ZERO edits here.
#
# Self-adapt / reuse:
#   - build gold (and main, for a merged-status diagnostic) into <workdir>/cpu_build
#   - if cuda/testdata/nanoBragg_root exists and reproduces gold on BOTH canaries
#     -> reuse it (do not overwrite)
#   - otherwise install the freshly built gold binary as nanoBragg_root
#
# A manifest (binary md5 + canary hashes + provenance) is written to
# <workdir>/cpu_manifest.txt. CPU reference images stay gitignored.
#
#   Usage: gen_cpu.sh [workdir]     (workdir defaults to cuda/testrun)
#          NB_FORCE_BUILD=1  install the freshly built gold even if root passes
# =============================================================================
set -u

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # cuda/test
REPO="$(cd "$TEST_DIR/../.." && pwd)"                         # repo root
WORKDIR="${1:-$REPO/cuda/testrun}"
BUILD="$WORKDIR/cpu_build"
RUNDIR="$BUILD/run"
MANIFEST="$WORKDIR/cpu_manifest.txt"
ROOT_BIN="$REPO/cuda/testdata/nanoBragg_root"
CRYST="$REPO/cuda/testdata/crystals"
SRC_FILE="nanoBragg.c"                                        # the root CPU oracle source
FIX_BRANCHES=(fix/phi0-stale-rotation fix/subpixel-oversampling fix/curved-det-flag-guard)

die(){ echo "ERROR: $*" >&2; exit 2; }
mkdir -p "$BUILD" "$RUNDIR"
cd "$REPO" || die "cannot cd repo"
command -v gcc >/dev/null 2>&1 || die "gcc not found"

# --- canary cell CLIs (2048^2, fix-sensitive) --------------------------------
BEAM="176.128000"
BASE="-distance 231.27 -lambda 0.9768 -pixel 0.172 -detpixels 2048 -Xbeam $BEAM -Ybeam $BEAM -flux 1e18 -beamsize 1.0 -nonoise -nointerpolate -nopgm"
C193L="-hkl $CRYST/193L.hkl -cell 78.540 78.540 37.770 90 90 90"
CAN_NAMES=(canary_phi0 canary_subpixel canary_curved)
CAN_phi0="$C193L $BASE -Na 100 -Nb 100 -Nc 100 -oversample 1 -osc 0.1 -phisteps 10 -misset 0 0 0"
CAN_subpixel="$C193L $BASE -Na 100 -Nb 100 -Nc 100 -oversample 2 -detector_thick 100 -detector_thicksteps 8 -detector_abs 100 -oversample_thick -phisteps 1 -misset 0 0 0"
# canary_curved: -curved_det is the LAST token, so it detects fix/curved-det-flag-guard.
# The pre-fix parser silently drops a trailing -curved_det (renders flat); the fixed
# parser honors it (curved). gold (fix applied) differs from main here iff the fix is
# missing on main, and nanoBragg_root must reproduce gold's curved image.
CAN_curved="$C193L $BASE -Na 100 -Nb 100 -Nc 100 -oversample 1 -misset 0 0 0 -curved_det"
canary_args(){ case "$1" in canary_phi0) echo "$CAN_phi0";; canary_subpixel) echo "$CAN_subpixel";; canary_curved) echo "$CAN_curved";; esac; }

# render a canary cell with a binary, return md5 of the float image
render_md5(){ # <binary> <canary_name>
  local bin="$1"
  local name="$2"
  local out="$RUNDIR/$(basename "$bin")_$name.bin"
  rm -f "$out"
  ( cd "$RUNDIR" && "$bin" -floatfile "$out" $(canary_args "$name") ) >"$RUNDIR/${name}_$(basename "$bin").log" 2>&1
  [ -s "$out" ] && md5sum "$out" | awk '{print $1}' || echo "RENDER_FAIL"
}

# --- build gold (main + both fixes) and main from source ---------------------
echo "== building CPU reference from source (main + fixes) =="
git rev-parse --verify -q main >/dev/null || die "'main' branch not found"
git show main:"$SRC_FILE" > "$BUILD/main_$SRC_FILE" || die "cannot extract main:$SRC_FILE"
cp "$BUILD/main_$SRC_FILE" "$BUILD/gold_$SRC_FILE"

merged_status=()
for br in "${FIX_BRANCHES[@]}"; do
  if git rev-parse --verify -q "$br" >/dev/null; then
    patch="$BUILD/$(basename "$br").patch"
    git diff "main..$br" -- "$SRC_FILE" > "$patch"
    if [ -s "$patch" ]; then
      # apply to the gold copy (rename gold file to the patch's target name temporarily)
      cp "$BUILD/gold_$SRC_FILE" "$BUILD/$SRC_FILE"
      if ( cd "$BUILD" && patch -p1 < "$patch" ) >>"$BUILD/patch.log" 2>&1; then
        cp "$BUILD/$SRC_FILE" "$BUILD/gold_$SRC_FILE"
        merged_status+=("$br=unmerged(diff applied)")
      else
        die "failed to apply $br diff to $SRC_FILE (see $BUILD/patch.log)"
      fi
      rm -f "$BUILD/$SRC_FILE"
    else
      merged_status+=("$br=merged(empty diff)")
    fi
  else
    merged_status+=("$br=absent(assumed merged)")
  fi
done

gcc -O3 -fopenmp "$BUILD/main_$SRC_FILE" -o "$BUILD/nb_main" -lm 2>"$BUILD/main_build.log" || { cat "$BUILD/main_build.log" >&2; die "main build failed"; }
gcc -O3 -fopenmp "$BUILD/gold_$SRC_FILE" -o "$BUILD/nb_gold" -lm 2>"$BUILD/gold_build.log" || { cat "$BUILD/gold_build.log" >&2; die "gold build failed"; }
echo "   built nb_main, nb_gold"

# --- canary hashes for gold + main ------------------------------------------
declare -A GOLD MAIN
sensitive_all=1
for name in "${CAN_NAMES[@]}"; do
  GOLD[$name]="$(render_md5 "$BUILD/nb_gold" "$name")"
  MAIN[$name]="$(render_md5 "$BUILD/nb_main" "$name")"
  [ "${GOLD[$name]}" != "RENDER_FAIL" ] || die "gold render failed for $name"
  if [ "${MAIN[$name]}" = "${GOLD[$name]}" ]; then
    echo "   $name: main==gold (fix already merged for this cell)"
  else
    echo "   $name: main!=gold (fix-sensitive: current main lacks the fix here)"
  fi
done

# --- decide: reuse existing nanoBragg_root, or install gold ------------------
install_reason=""
reuse=0
if [ "${NB_FORCE_BUILD:-0}" != 1 ] && [ -x "$ROOT_BIN" ]; then
  ok=1
  for name in "${CAN_NAMES[@]}"; do
    rh="$(render_md5 "$ROOT_BIN" "$name")"
    if [ "$rh" != "${GOLD[$name]}" ]; then ok=0; echo "   canary FAIL on existing nanoBragg_root: $name (root=$rh gold=${GOLD[$name]})"; fi
  done
  if [ "$ok" = 1 ]; then reuse=1; install_reason="existing nanoBragg_root reproduces gold on both canaries"; fi
fi

if [ "$reuse" = 1 ]; then
  echo "== REUSE: $install_reason =="
  FINAL_BIN="$ROOT_BIN"
else
  echo "== INSTALL: writing freshly built gold to $ROOT_BIN =="
  cp "$BUILD/nb_gold" "$ROOT_BIN" || die "cannot install $ROOT_BIN"
  install_reason="installed freshly built gold (main+fixes) from source"
  FINAL_BIN="$ROOT_BIN"
  # confirm the installed binary now passes
  for name in "${CAN_NAMES[@]}"; do
    rh="$(render_md5 "$FINAL_BIN" "$name")"
    [ "$rh" = "${GOLD[$name]}" ] || die "installed nanoBragg_root still fails canary $name"
  done
fi

# --- manifest ----------------------------------------------------------------
{
  echo "# nanoBragg CPU reference provenance -- generated by gen_cpu.sh"
  echo "timestamp        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cpu_reference    : $FINAL_BIN"
  echo "cpu_reference_md5: $(md5sum "$FINAL_BIN" | awk '{print $1}')"
  echo "decision         : $install_reason"
  echo "source_file      : $SRC_FILE (root CPU oracle)"
  echo "build_cmd        : gcc -O3 -fopenmp $SRC_FILE -o nanoBragg_root -lm"
  for s in "${merged_status[@]}"; do echo "fix_branch       : $s"; done
  echo "gate             : behavioral canary (fix-sensitive cells) vs from-source gold"
  for name in "${CAN_NAMES[@]}"; do
    echo "canary $name gold_md5=${GOLD[$name]} main_md5=${MAIN[$name]}"
  done
  echo "canary_phi0_cli    : $CAN_phi0"
  echo "canary_subpixel_cli: $CAN_subpixel"
  echo "canary_curved_cli  : $CAN_curved"
} > "$MANIFEST"

echo "== CPU reference READY: $FINAL_BIN =="
echo "   manifest -> $MANIFEST"
