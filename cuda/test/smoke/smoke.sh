#!/usr/bin/env bash
#
# Binary-only release smoke test for nanoBraggCUDA.
#
# This is a sanity check that a built (or deployed) binary runs end-to-end and
# emits a non-trivial diffraction image. It is deliberately independent of the
# parity harness in ../ : no -hkl, no -matrix, no external data files at all.
# The crystal is described entirely on the command line with an inline -cell
# and a flat -default_F structure factor, so the only input is the binary.
#
# Usage:  ./smoke.sh [path-to-nanoBraggCUDA]
#   $1 (optional) = path to the binary; default = the in-repo release build.
# Pick a GPU with CUDA_VISIBLE_DEVICES (e.g. a UUID); the script is device-agnostic.

set -u

# Resolve the binary: explicit $1, else the release build relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${1:-$SCRIPT_DIR/../../build/release/nanoBraggCUDA}"

fail() { echo "SMOKE FAIL: $*"; exit 1; }

[ -x "$BIN" ] || fail "binary not found or not executable: $BIN"

# Report which GPU will be used. The binary does not print a device name, so we
# resolve CUDA_VISIBLE_DEVICES (when it is a UUID) via nvidia-smi purely for
# visibility -- the render itself honors whatever CUDA_VISIBLE_DEVICES selects.
gpu_name=""
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader 2>/dev/null \
        | awk -F', ' -v want="${CUDA_VISIBLE_DEVICES:-}" '$1==want{print $2}')"
    [ -n "$gpu_name" ] || gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
fi
echo "GPU in use: ${gpu_name:-unknown}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
echo "binary: $BIN"

# Render inside a throwaway dir so no intimage.img / floatimage.bin / Fdump.bin
# is ever left in the repo. Clean it up on any exit.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# Small, fast render: 3x3x3 unit cells onto a 256x256 detector. A flat cubic
# cell plus -default_F needs no hkl/matrix file, which is the whole point.
DETPIXELS=256
LOG="$SCRATCH/render.log"
( cd "$SCRATCH" && "$BIN" \
    -cell 100 100 100 90 90 90 \
    -default_F 100 \
    -lambda 6.2 \
    -distance 100 \
    -detpixels "$DETPIXELS" \
    -pixel 0.1 \
    -N 3 ) >"$LOG" 2>&1
rc=$?

# Assert (i): clean exit.
[ "$rc" -eq 0 ] || { sed 's/^/  | /' "$LOG"; fail "binary exited $rc"; }

# Assert (ii): the printed peak intensity is a positive number (image has signal).
max_I="$(awk '/^max_I =/{print $3; exit}' "$LOG")"
[ -n "$max_I" ] || fail "no 'max_I' line in output (did the render run?)"
awk -v v="$max_I" 'BEGIN{exit !(v+0 > 0)}' || fail "max_I not > 0 (got '$max_I')"

# Assert (iii): the raw float image exists and is exactly detpixels^2 * 4 bytes
# (floatimage.bin is headerless 4-byte floats), i.e. a full, non-trivial frame.
IMG="$SCRATCH/floatimage.bin"
[ -f "$IMG" ] || fail "floatimage.bin was not written"
expect=$(( DETPIXELS * DETPIXELS * 4 ))
got=$(stat -c%s "$IMG")
[ "$got" -eq "$expect" ] || fail "floatimage.bin is $got bytes, expected $expect"

echo "SMOKE PASS: max_I=$max_I, floatimage.bin=$got bytes (${DETPIXELS}x${DETPIXELS})"
