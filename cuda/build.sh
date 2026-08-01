#!/bin/sh
# Provenance-stamped build entry point for nanoBraggCUDA. Resolves git commit/branch/tag,
# refuses a dirty tree unless explicitly allowed, then bakes the values via the Makefile.
set -u
HERE=$(CDPATH= cd "$(dirname "$0")" && pwd)

usage() {
  cat <<'EOF'
Usage: ./build.sh [options] [make-goals]

Provenance-stamped build for nanoBraggCUDA. In a git repository it bakes the
commit/branch/tag/date into the binary (see `nanoBraggCUDA -version`) and refuses
a dirty tree unless told otherwise; outside a git repository it just builds and the
binary reports "version: unknown (built without version info)".

Options:
  --allow-dirty       build even if tracked files are modified
  --allow-untracked   build even if untracked files are present
  -h, --help          show this help and exit

make-goals            targets passed to make (default: release), e.g.
                      release, debug, all, clean

Examples:
  ./build.sh
  ./build.sh --allow-dirty --allow-untracked
  ./build.sh debug
EOF
}

ALLOW_DIRTY=0; ALLOW_UNTRACKED=0; GOALS=""
for a in "$@"; do
  case "$a" in
    -h|--help|-help)   usage; exit 0 ;;
    --allow-dirty)     ALLOW_DIRTY=1 ;;
    --allow-untracked) ALLOW_UNTRACKED=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *)  GOALS="$GOALS $a" ;;
  esac
done
[ -n "$GOALS" ] || GOALS="release"

# build.sh is the universal entry point. In a git repo it resolves provenance
# and runs the dirty/untracked gate; outside one (e.g. an unzipped archive) it
# just builds, and the binary stamps "version: unknown (built without version info)".
if git -C "$HERE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  A=0; git -C "$HERE" diff --quiet HEAD || A=1
  B=0; [ -n "$(git -C "$HERE" ls-files --others --exclude-standard)" ] && B=1

  if { [ "$A" = 1 ] && [ "$ALLOW_DIRTY" = 0 ]; } || { [ "$B" = 1 ] && [ "$ALLOW_UNTRACKED" = 0 ]; }; then
    echo "ERROR: build blocked — working tree is not clean:" >&2
    if [ "$A" = 1 ]; then
      echo "  Modified tracked files:" >&2
      git -C "$HERE" status --porcelain -uno | sed 's/^/    /' >&2
    fi
    if [ "$B" = 1 ]; then
      n=$(git -C "$HERE" ls-files --others --exclude-standard | grep -c .)
      echo "  Untracked files ($n):" >&2
      git -C "$HERE" ls-files --others --exclude-standard | head -3 | sed 's/^/    /' >&2
      [ "$n" -gt 3 ] && echo "    …(+$((n-3)) more; run: git status)" >&2
    fi
    printf '  build anyway: ./build.sh' >&2
    [ "$A" = 1 ] && printf ' --allow-dirty' >&2
    [ "$B" = 1 ] && printf ' --allow-untracked' >&2
    printf '%s\n' "$GOALS" >&2
    exit 1
  fi

  GC=$(git -C "$HERE" rev-parse --short=12 HEAD)
  GB=$(git -C "$HERE" rev-parse --abbrev-ref HEAD)
  GT=$(git -C "$HERE" tag --points-at HEAD | head -1)
  GD=$(git -C "$HERE" show -s --format=%cd --date=short HEAD)
  exec make -C "$HERE" GIT_COMMIT="$GC" GIT_BRANCH="$GB" GIT_TAG="$GT" GIT_DATE="$GD" $GOALS
fi

# Not a git repository (e.g. an unzipped source archive): build with no version info.
echo "note: not a git repository — building without version info." >&2
exec make -C "$HERE" $GOALS
