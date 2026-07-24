#!/bin/bash
# =============================================================================
# run.sh -- nanoBragg parity harness orchestrator.
#
#   run.sh <suite> <kernel-binary> [workdir]
#
# Renders every cell of suites/<suite>.jsonl on the GPU kernel under test, renders
# (or reuses a cached) CPU reference from cuda/testdata/nanoBragg_root, compares
# with the metrics binary, applies the uniform gate, and writes a clean TSV plus a
# TIER <suite> PASS|FAIL n_pass/n_total summary line. Nonzero exit on any FAIL.
#
# DEVICE RULE (this box has two GPUs, CUDA index INVERTED vs nvidia-smi): the
# desktop RTX 5090 (170 SM) sorts first under CUDA_DEVICE_ORDER=FASTEST_FIRST, so
# CUDA_VISIBLE_DEVICES=0 selects it. This script PROVES the active device is the
# desktop 5090 before rendering a single pixel; a mismatch is a hard refusal.
#
# Layout: C tools compile to cuda/test/build/ (gitignored). CPU refs in
# <workdir>/cpu, GPU outputs in <workdir>/out, results in <workdir>/results.tsv
# (workdir defaults to cuda/testrun); data inputs in cuda/testdata. The committed
# cuda/test/ source is written only by a deliberate gen_suites regeneration of suites/.
# =============================================================================
set -u

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # cuda/test
REPO="$(cd "$TEST_DIR/../.." && pwd)"                         # repo root
SPEC="$TEST_DIR/spec"
SUITES="$TEST_DIR/suites"
BUILDDIR="$TEST_DIR/build"                                    # cuda/test/build (gitignored)

die(){ echo "ERROR: $*" >&2; exit 2; }

# --- provenance ---------------------------------------------------------------
# corr/sum_ratio/verdict are machine-independent but ms is not; the perf ledger
# collects results across machines, so every results file carries the full
# hardware+conditions profile. gather_provenance fills PROV_*; emit_provenance
# appends a clean, greppable "# key=value" block to the file named in $1.
gather_provenance(){
  PROV_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  PROV_CUDA="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+, V[0-9.]+' | head -1)"
  PROV_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  # Build-time commit bake: metrics is compiled with -DNB_BUILD_COMMIT=<HEAD at
  # compile time> (see Makefile), so this reports the commit `make` last compiled
  # it at rather than re-reading git HEAD now. run.sh runs `make` on every
  # invocation, so on this stable feature branch it equals HEAD-at-render; guard
  # against metrics being missing (e.g. a stale $BIN before the first make).
  if [ -x "$BIN/metrics" ]; then
    PROV_COMMIT="$("$BIN/metrics" --build-commit 2>/dev/null)"
  fi
  PROV_HOST="$(hostname 2>/dev/null || uname -n 2>/dev/null)"
  PROV_CPUMODEL="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"
  : "${PROV_DRIVER:=unknown}" "${PROV_CUDA:=unknown}" "${PROV_COMMIT:=unknown}" "${PROV_HOST:=unknown}" "${PROV_CPUMODEL:=unknown}"
}
emit_provenance(){   # $1 = output file (appended)
  {
    echo "# driver=$PROV_DRIVER  cuda=\"$PROV_CUDA\""
    echo "# utc=$PROV_UTC  host=$PROV_HOST"
    echo "# commit=$PROV_COMMIT"
    # Clean, greppable keys for ledger_append.sh (in addition to the free-form
    # lines above): gpu_name pulled out of $DEVICE_EVIDENCE, cpu_model from
    # gather_provenance, precision from the active $PRECISION run var.
    echo "# gpu_name=$(sed -n 's/.*name="\([^"]*\)".*/\1/p' <<<"${DEVICE_EVIDENCE:-}")"
    echo "# cpu_model=$PROV_CPUMODEL"
    echo "# precision=${PRECISION:-unknown}"
  } >> "$1"
}

# --- golden seeding (bake mode) ----------------------------------------------
# Upsert a baked cell's expected verdict into the committed golden via the
# EXISTING golden format ('# ...' header comments + 'cell verdict corr sum_ratio'
# rows, tab-separated). Existing (routine) rows are left untouched. corr/sum_ratio
# are recorded alongside verdict for the Layer-2 ledger's drift columns, but the
# gate below still reads/compares verdict only.
seed_golden(){   # $1=cell $2=verdict $3=corr $4=sum_ratio
  local cell="$1" verdict="$2" corr="${3:--}" sr="${4:--}" gf="$GOLDEN" tmp
  if [ ! -f "$gf" ]; then
    {
      printf '# golden expected-verdicts  suite=%s  precision=%s\n' "$SUITE_NAME" "$PRECISION"
      printf '# seeded by bake (NB_BAKE=1) / deathstar (NB_DEATHSTAR=1) mode: baked and\n'
      printf '# deathstar cells (compute-K over budget) whose CPU oracle was frozen once;\n'
      printf '# their expected verdict is recorded here.\n'
      printf '# cell\tverdict\tcorr\tsum_ratio\n'
    } > "$gf"
  fi
  if grep -q "^${cell}"$'\t' "$gf" 2>/dev/null; then
    tmp="$gf.bake.$$"
    awk -v c="$cell" -v v="$verdict" -v cr="$corr" -v s="$sr" \
        'BEGIN{FS=OFS="\t"} /^#/{print;next} $1==c{$2=v; $3=cr; $4=s} {print}' "$gf" > "$tmp" && mv "$tmp" "$gf"
  else
    printf '%s\t%s\t%s\t%s\n' "$cell" "$verdict" "$corr" "$sr" >> "$gf"
  fi
  echo "# bake: seeded golden row  $cell -> $verdict  ($gf)"
}

# NB_SCORE_ONLY=1 re-scores an EXISTING results.tsv against the golden with no
# device assertion and no GPU/CPU render (kernel arg optional; pass '-' as a
# placeholder). Used to re-verify the suite verdict without a fresh GPU run.
# NB_DIFF=1 renders each cell with TWO GPU kernels and compares the two GPU
# outputs to EACH OTHER -- no CPU reference, no golden, no gate. It is the
# accumulation probe (Phase B): df64 vs fp32 as a function of accumulation depth
# K, in the regime where the CPU oracle is impractical (hours). The <suite> arg
# may be a suite name (suites/<suite>.jsonl) or a path to any cells jsonl file.
DIFF_MODE="${NB_DIFF:-0}"
SCORE_ONLY="${NB_SCORE_ONLY:-0}"
# NB_BAKE=1 is the deliberate expensive run: for cells gen_suites tagged
# cpu_class="baked" (compute-K over the ~10-min budget) it GENERATES the CPU oracle
# once, freezes it (persistent), renders/compares, and seeds the cell's expected
# verdict into the committed golden. A routine run (NB_BAKE unset) NEVER generates a
# baked oracle: it compares against the frozen one if present, else SKIPs the cell.
BAKE_MODE="${NB_BAKE:-0}"
# NB_DEATHSTAR=1 is the third, extreme tier: for cells gen_suites tagged
# cpu_class="deathstar" (compute-K over NB_DEATHSTAR_BUDGET -- an hours-long,
# box-pinning oracle) it prints a LOUD banner, then GENERATES the CPU oracle once,
# freezing it with the same machinery as a baked cell and seeding its golden verdict.
# A deathstar cell is NEVER generated by a routine run OR by NB_BAKE=1 -- both SKIP
# it; only NB_DEATHSTAR=1 fires the generation. Once its frozen oracle is present, a
# routine run compares against it cheaply (the guard is on GENERATION, not compare).
DEATHSTAR_MODE="${NB_DEATHSTAR:-0}"
if [ "$DIFF_MODE" = "1" ]; then
  [ $# -ge 3 ] || die "usage (diff): NB_DIFF=1 run.sh <cellfile|suite> <kernel_a> <kernel_b> [workdir]"
  SUITE="$1"; KERNEL_A="$2"; KERNEL_B="$3"; WORKDIR="${4:-$REPO/cuda/testrun}"
  case "$KERNEL_A" in /*) ;; *) KERNEL_A="$PWD/$KERNEL_A";; esac
  case "$KERNEL_B" in /*) ;; *) KERNEL_B="$PWD/$KERNEL_B";; esac
  KERNEL="$KERNEL_A"
elif [ "$SCORE_ONLY" = "1" ]; then
  [ $# -ge 1 ] || die "usage (score-only): NB_SCORE_ONLY=1 run.sh <suite> [kernel|-] [workdir]"
  SUITE="$1"; KERNEL="${2:--}"; WORKDIR="${3:-$REPO/cuda/testrun}"
else
  [ $# -ge 2 ] || die "usage: run.sh <suite> <kernel-binary> [workdir]"
  SUITE="$1"
  KERNEL="$2"
  WORKDIR="${3:-$REPO/cuda/testrun}"
  case "$KERNEL" in /*) ;; *) KERNEL="$PWD/$KERNEL";; esac
fi

BIN="$BUILDDIR"
CPUD="$WORKDIR/cpu"
OUTD="$WORKDIR/out"
SCRATCH="$WORKDIR/scratch"
# Frozen baked oracles: persistent, never auto-deleted or auto-generated by a
# routine run. Kept separate from the ephemeral cpu/ cache so a cleanup
# of the cheap oracles cannot silently drop an expensive baked one.
FROZEN="$WORKDIR/frozen"
RESULTS="$WORKDIR/results.tsv"
CPU_BIN="${NB_CPU_BIN:-$REPO/cuda/testdata/nanoBragg_root}"
mkdir -p "$BIN" "$CPUD" "$OUTD" "$SCRATCH" "$FROZEN"

# --- device pinning (inverted-index rule) ------------------------------------
export CUDA_DEVICE_ORDER=FASTEST_FIRST
export CUDA_VISIBLE_DEVICES="${NB_CVD:-0}"
NB_EXPECT_NAME="${NB_EXPECT_NAME:-NVIDIA GeForce RTX 5090}"
NB_EXPECT_SM="${NB_EXPECT_SM:-170}"

# =============================================================================
# DEVICE ASSERTION -- refuse before any render if not on the desktop 5090
# =============================================================================
PROBE="$BUILDDIR/.devprobe"
PROBE_SRC="$BUILDDIR/.devprobe.cu"
assert_device(){
  if [ ! -x "$PROBE" ] || [ "$PROBE_SRC" -nt "$PROBE" ]; then
    command -v nvcc >/dev/null 2>&1 || die "nvcc not found; cannot build device probe"
    cat > "$PROBE_SRC" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
int main(){
  int dev=-1; cudaGetDevice(&dev);
  cudaDeviceProp p; cudaError_t e=cudaGetDeviceProperties(&p, dev);
  if(e!=cudaSuccess){ printf("PROBE_ERROR %s\n", cudaGetErrorString(e)); return 3; }
  printf("CUDA_DEVICE(active)=%d name=\"%s\" SMs=%d cc=%d.%d totalGiB=%.1f\n",
    dev, p.name, p.multiProcessorCount, p.major, p.minor, p.totalGlobalMem/1073741824.0);
  return 0;
}
CU
    nvcc -Wno-deprecated-gpu-targets -o "$PROBE" "$PROBE_SRC" >"$BUILDDIR/.devprobe_build.log" 2>&1 \
      || { cat "$BUILDDIR/.devprobe_build.log" >&2; die "device probe build failed"; }
  fi
  DEVICE_EVIDENCE="$("$PROBE")"; local prc=$?
  if [ $prc -ne 0 ] || [ -z "$DEVICE_EVIDENCE" ]; then
    echo "DEVICE ASSERTION REFUSED: probe failed (rc=$prc): ${DEVICE_EVIDENCE:-<none>}" >&2
    exit 4
  fi
  local got_name got_sm
  got_name="$(sed -n 's/.*name="\([^"]*\)".*/\1/p' <<<"$DEVICE_EVIDENCE")"
  got_sm="$(sed -n 's/.*SMs=\([0-9][0-9]*\).*/\1/p' <<<"$DEVICE_EVIDENCE")"
  if [ "$got_name" != "$NB_EXPECT_NAME" ] || [ "$got_sm" != "$NB_EXPECT_SM" ]; then
    echo "==============================================================" >&2
    echo "DEVICE ASSERTION REFUSED -- NOT rendering." >&2
    echo "  CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    echo "  active : $DEVICE_EVIDENCE" >&2
    echo "  expect : name=\"$NB_EXPECT_NAME\" SMs=$NB_EXPECT_SM" >&2
    exit 4
  fi
}
# =============================================================================
# DIFFERENTIAL MODE (Phase B accumulation probe): two GPU kernels, no CPU.
# =============================================================================
if [ "$DIFF_MODE" = "1" ]; then
  # cellfile may be an explicit path or a suite name under suites/
  if [ -s "$SUITE" ]; then DCELL="$SUITE"; else DCELL="$SUITES/$SUITE.jsonl"; fi
  [ -s "$DCELL" ] || die "diff: cells file missing: $DCELL"
  [ -x "$KERNEL_A" ] || die "diff: kernel_a missing/not-exec: $KERNEL_A"
  [ -x "$KERNEL_B" ] || die "diff: kernel_b missing/not-exec: $KERNEL_B"
  assert_device
  echo "DEVICE OK: $DEVICE_EVIDENCE"
  make -C "$TEST_DIR" BINDIR="$BIN" all >/dev/null || die "make failed"
  METRICS="$BIN/metrics"
  [ -x "$METRICS" ] || die "metrics binary missing after make: $METRICS"

  DATA_ROOT="$(sed -n 's/.*"data_root"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$SPEC/base.json")"
  DATA_ROOT_ABS="$REPO/$DATA_ROOT"
  abspath_args(){ printf '%s' "${1//\{data_root\}/$DATA_ROOT_ABS}"; }

  DRES="$WORKDIR/diff_results.tsv"
  KERNEL_A_MD5="$(md5sum "$KERNEL_A" | awk '{print $1}')"
  KERNEL_B_MD5="$(md5sum "$KERNEL_B" | awk '{print $1}')"
  gather_provenance
  {
    echo "# nanoBragg parity diff (A vs B GPU outputs; no CPU, no gate)"
    echo "# kernel_a=$KERNEL_A md5=$KERNEL_A_MD5"
    echo "# kernel_b=$KERNEL_B md5=$KERNEL_B_MD5"
    echo "# device=$DEVICE_EVIDENCE"
  } > "$DRES"
  emit_provenance "$DRES"
  printf '# cell\tcorr\tsum_ratio\tmax_rel\tworst_pixel_frac\tpeak_max_rel\tworst_is_peak\tms_a\tms_b\n' >> "$DRES"
  printf '%-18s %-11s %-10s %-10s %-11s %-11s %-5s %-8s %-8s\n' cell corr sum_ratio max_rel wpf peak_max_rel wis ms_a ms_b
  echo "# diff: A=$KERNEL_A"
  echo "# diff: B=$KERNEL_B"

  while IFS= read -r line; do
    [ -n "$line" ] || continue
    id="$(sed -n 's/.*"id":"\([^"]*\)".*/\1/p' <<<"$line")"
    gpu_args_rel="$(sed -n 's/.*"gpu_args":"\([^"]*\)".*/\1/p' <<<"$line")"
    [ -n "$id" ] || continue
    gpu_args="$(abspath_args "$gpu_args_rel")"
    case " $gpu_args " in *" -hkl "*) : ;; *) rm -f "$SCRATCH/Fdump.bin" ;; esac

    aout="$OUTD/$id.a.bin"; bout="$OUTD/$id.b.bin"; rm -f "$aout" "$bout"
    t0=$(date +%s.%N); ( cd "$SCRATCH" && "$KERNEL_A" -floatfile "$aout" $gpu_args ) >"$SCRATCH/diffA_$id.log" 2>&1; ra=$?; t1=$(date +%s.%N)
    ms_a=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f",(b-a)*1000.0}')
    t0=$(date +%s.%N); ( cd "$SCRATCH" && "$KERNEL_B" -floatfile "$bout" $gpu_args ) >"$SCRATCH/diffB_$id.log" 2>&1; rb=$?; t1=$(date +%s.%N)
    ms_b=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f",(b-a)*1000.0}')
    if [ $ra -ne 0 ] || [ ! -s "$aout" ] || [ $rb -ne 0 ] || [ ! -s "$bout" ]; then
      printf '%s\t-\t-\t-\t-\t-\t-\t%s\t%s\tRENDER_FAIL(a=%s b=%s)\n' "$id" "$ms_a" "$ms_b" "$ra" "$rb" >> "$DRES"
      printf '%-18s %-11s %-10s %-10s %-11s %-11s %-5s %-8s %-8s\n' "$id" - - - - - - "$ms_a" "$ms_b"
      rm -f "$aout" "$bout"; continue
    fi
    read -r corr sr mr wpf pmr wis < <("$METRICS" "$aout" "$bout" 2>/dev/null)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$corr" "$sr" "$mr" "$wpf" "$pmr" "$wis" "$ms_a" "$ms_b" >> "$DRES"
    printf '%-18s %-11s %-10s %-10s %-11s %-11s %-5s %-8s %-8s\n' "$id" "$corr" "$sr" "$mr" "$wpf" "$pmr" "$wis" "$ms_a" "$ms_b"
    rm -f "$aout" "$bout"
  done < "$DCELL"
  echo "# diff results -> $DRES   (A vs B GPU outputs; no CPU, no gate)"
  exit 0
fi

# --- config needed by BOTH render and score-only paths -----------------------
# <suite> may be a bare suite name (suites/<suite>.jsonl) or a path to a cells
# jsonl file (same convention diff mode already accepts). SUITE_NAME keys the
# golden and all display lines.
if [ -s "$SUITE" ] && [ "${SUITE##*.}" = "jsonl" ]; then
  CELLFILE="$SUITE"; SUITE_NAME="$(basename "${SUITE%.jsonl}")"
else
  CELLFILE="$SUITES/$SUITE.jsonl"; SUITE_NAME="$SUITE"
fi
[ -s "$CELLFILE" ] || die "cells file missing: $CELLFILE (compile with gen_suites)"
NTOTAL="$(grep -c . "$CELLFILE")"
PRECISION="${NB_PRECISION:-fp32}"
GOLDEN="$TEST_DIR/golden/$SUITE_NAME.$PRECISION.tsv"
# perf suite: min-of-5 timing, warn-not-fail (no perf golden is ever required).
PERF=0; [ "$SUITE_NAME" = "perf" ] && PERF=1
# The kernel's default precision is double; NB_PRECISION must be carried onto the
# GPU render explicitly so it drives BOTH the golden file AND the actual arithmetic.
# fp32 -> single, df64/fp64 -> double (the kernel's compiled double path). Without
# this the default-double kernel renders df64 while comparing against the fp32 golden.
case "$PRECISION" in
  fp32) GPU_PREC_ARGS="-precision single" ;;
  df64|fp64) GPU_PREC_ARGS="-precision double" ;;
  *) GPU_PREC_ARGS="" ;;
esac
RANGE_LO=1; RANGE_HI="$NTOTAL"
if [ "$SCORE_ONLY" != "1" ] && [ -n "${NB_RANGE:-}" ]; then RANGE_LO="${NB_RANGE%%-*}"; RANGE_HI="${NB_RANGE##*-}"; fi
npass=0; nfail=0; nother=0; nskip=0

if [ "$SCORE_ONLY" = "1" ]; then
  [ -s "$RESULTS" ] || die "score-only: nothing to score at $RESULTS"
  echo "# SCORE-ONLY: re-scoring existing results.tsv against golden (no device, no render)"
  echo "# suite=$SUITE_NAME precision=$PRECISION cells=$NTOTAL results=$RESULTS"
else
assert_device
echo "DEVICE OK: $DEVICE_EVIDENCE"

# --- preflight ---------------------------------------------------------------
[ -x "$KERNEL" ]  || die "GPU kernel missing/not-exec: $KERNEL"
[ -x "$CPU_BIN" ] || die "CPU reference missing/not-exec: $CPU_BIN (run gen_cpu.sh first)"

# build/refresh the C tools out-of-source into <workdir>/bin
make -C "$TEST_DIR" BINDIR="$BIN" all >/dev/null || die "make failed"
METRICS="$BIN/metrics"
[ -x "$METRICS" ] || die "metrics binary missing after make: $METRICS"

# --- spec-derived config: data_root + gate -----------------------------------
DATA_ROOT="$(sed -n 's/.*"data_root"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$SPEC/base.json")"
[ -n "$DATA_ROOT" ] || die "data_root not found in base.json"
DATA_ROOT_ABS="$REPO/$DATA_ROOT"
CORR_MIN="$(sed -n 's/.*"corr_min"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' "$SPEC/base.json")"
SR_MIN="$(sed -n 's/.*"sum_ratio_min"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' "$SPEC/base.json")"
SR_MAX="$(sed -n 's/.*"sum_ratio_max"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' "$SPEC/base.json")"
CORR_MIN="${CORR_MIN:-0.9999}"; SR_MIN="${SR_MIN:-0.999}"; SR_MAX="${SR_MAX:-1.001}"

KERNEL_MD5="$(md5sum "$KERNEL" | awk '{print $1}')"
CPU_MD5="$(md5sum "$CPU_BIN" | awk '{print $1}')"
NTOTAL="$(grep -c . "$CELLFILE")"

# Optional chunking: NB_RANGE="lo-hi" (1-based inclusive line index into the
# cells file). Lets a long suite run as several foreground/blocking invocations,
# each appending to results.tsv; the TIER tally is computed over the whole file.
RANGE_LO=1; RANGE_HI="$NTOTAL"
if [ -n "${NB_RANGE:-}" ]; then RANGE_LO="${NB_RANGE%%-*}"; RANGE_HI="${NB_RANGE##*-}"; fi

echo "# suite=$SUITE_NAME cells=$NTOTAL range=${RANGE_LO}-${RANGE_HI}$([ "$BAKE_MODE" = 1 ] && echo ' bake=1')$([ "$DEATHSTAR_MODE" = 1 ] && echo ' deathstar=1')"
echo "# kernel=$KERNEL md5=$KERNEL_MD5"
echo "# cpu_ref=$CPU_BIN md5=$CPU_MD5"
echo "# gate: corr>=$CORR_MIN AND sum_ratio in [$SR_MIN,$SR_MAX]"
echo "# device=$DEVICE_EVIDENCE"

# clean TSV (single-tab). Fresh header only when starting from cell 1.
if [ "$RANGE_LO" -le 1 ]; then
  printf '# cell\tcorr\tsum_ratio\tms\tverdict\tnote\n' > "$RESULTS"
fi
printf '%-34s %-11s %-10s %-8s %-8s %s\n' cell corr sum_ratio ms verdict note

# resolve {data_root} repo-relative paths to absolute for actual invocation
abspath_args(){ printf '%s' "${1//\{data_root\}/$DATA_ROOT_ABS}"; }

npass=0; nfail=0; nother=0; nskip=0; lineno=0
while IFS= read -r line; do
  [ -n "$line" ] || continue
  lineno=$((lineno+1))
  [ "$lineno" -lt "$RANGE_LO" ] && continue
  [ "$lineno" -gt "$RANGE_HI" ] && break
  id="$(sed -n 's/.*"id":"\([^"]*\)".*/\1/p' <<<"$line")"
  gpu_args_rel="$(sed -n 's/.*"gpu_args":"\([^"]*\)".*/\1/p' <<<"$line")"
  cpu_args_rel="$(sed -n 's/.*"cpu_args":"\([^"]*\)".*/\1/p' <<<"$line")"
  cpu_class="$(sed -n 's/.*"cpu_class":"\([^"]*\)".*/\1/p' <<<"$line")"
  gate_type="$(sed -n 's/.*"gate_type":"\([^"]*\)".*/\1/p' <<<"$line")"
  [ -n "$gate_type" ] || gate_type="parity"
  [ -n "$id" ] || { echo "WARN: unparseable cell line skipped" >&2; continue; }

  gpu_args="$(abspath_args "$gpu_args_rel")"
  cpu_args="$(abspath_args "$cpu_args_rel")"

  # Fdump.bin footgun: with no -hkl (e.g. -default_F cells), nanoBragg auto-LOADS
  # a stale scratch/Fdump.bin left by a previous -hkl cell. Remove it so the cell
  # uses its own structure factors. -hkl cells read their file directly (safe).
  case " $gpu_args " in *" -hkl "*) : ;; *) rm -f "$SCRATCH/Fdump.bin" ;; esac

  # gate_type==reject (guards suite): a cell the GPU kernel MUST refuse rather
  # than silently misbehave. No CPU oracle, no metrics -- render GPU-only and
  # classify by exit code: 9=REJECT (expected), 0=FAIL (silent no-op
  # regression, produced an image it should have refused), else BLOCKED.
  if [ "$gate_type" = "reject" ]; then
    gout="$OUTD/$id.bin"; rm -f "$gout"
    t0=$(date +%s.%N)
    ( cd "$SCRATCH" && "$KERNEL" -floatfile "$gout" $gpu_args $GPU_PREC_ARGS ) >"$SCRATCH/gpu_$id.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    ms=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f",(b-a)*1000.0}')
    case "$rc" in
      9) verdict="REJECT"; note="rejected(exit9)" ;;
      0) verdict="FAIL";   note="silent-no-op-regression(exit0)" ;;
      *) verdict="BLOCKED"; note="reject-unexpected(exit $rc)" ;;
    esac
    printf '%s\t-\t-\t%s\t%s\t%s\n' "$id" "$ms" "$verdict" "$note" >> "$RESULTS"
    printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" - - "$ms" "$verdict" "$note"
    case "$verdict" in REJECT) npass=$((npass+1));; FAIL) nfail=$((nfail+1));; *) nother=$((nother+1));; esac
    rm -f "$gout"
    continue
  fi

  # CPU reference. cpu_class (from gen_suites' compute-K budget) decides the policy:
  #   deathstar -- the hours-long, box-pinning extreme. Compare against the FROZEN
  #                oracle if present; if absent, SKIP on BOTH a routine run AND
  #                NB_BAKE=1. Only an explicit NB_DEATHSTAR=1 generates it, behind a
  #                loud banner. The guard is on GENERATION, not on comparison.
  #   baked   -- oracle too expensive to make on a routine run. Compare against the
  #              FROZEN oracle if present; if absent, SKIP (never auto-generate).
  #              Bake mode (NB_BAKE=1) is the one place a baked frozen oracle is made.
  #   routine -- (or unclassified) ephemeral cpu/ cache, auto-generated on demand.
  chash="$(printf '%s' "$cpu_args" | md5sum | awk '{print $1}')"
  if [ "$cpu_class" = "deathstar" ]; then
    cpuref="$FROZEN/$chash.bin"
    if [ ! -s "$cpuref" ]; then
      if [ "$DEATHSTAR_MODE" = "1" ]; then
        echo "*****************************************************************" >&2
        echo "*** DEATH STAR: generating hours-long CPU oracle(s);          ***" >&2
        echo "***             this WILL pin this box for hours.             ***" >&2
        echo "***   cell: $id" >&2
        echo "*****************************************************************" >&2
        ( cd "$SCRATCH" && "$CPU_BIN" -floatfile "$cpuref" $cpu_args ) >"$SCRATCH/deathstar_$id.log" 2>&1
        if [ ! -s "$cpuref" ]; then
          printf '%s\t-\t-\t-\tBLOCKED\tdeathstar-cpu-fail(%s)\n' "$id" "$SCRATCH/deathstar_$id.log" >> "$RESULTS"
          printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" - - - BLOCKED deathstar-cpu-fail
          nother=$((nother+1)); continue
        fi
      else
        printf '%s\t-\t-\t-\tSKIP\tdeathstar; requires NB_DEATHSTAR=1\n' "$id" >> "$RESULTS"
        printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" - - - SKIP "deathstar; requires NB_DEATHSTAR=1"
        nskip=$((nskip+1)); continue
      fi
    fi
  elif [ "$cpu_class" = "baked" ]; then
    cpuref="$FROZEN/$chash.bin"
    if [ ! -s "$cpuref" ]; then
      if [ "$BAKE_MODE" = "1" ]; then
        ( cd "$SCRATCH" && "$CPU_BIN" -floatfile "$cpuref" $cpu_args ) >"$SCRATCH/bake_$id.log" 2>&1
        if [ ! -s "$cpuref" ]; then
          printf '%s\t-\t-\t-\tBLOCKED\tbake-cpu-fail(%s)\n' "$id" "$SCRATCH/bake_$id.log" >> "$RESULTS"
          printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" - - - BLOCKED bake-cpu-fail
          nother=$((nother+1)); continue
        fi
      else
        printf '%s\t-\t-\t-\tSKIP\tbaked; frozen oracle absent; run bake mode\n' "$id" >> "$RESULTS"
        printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" - - - SKIP "baked; frozen oracle absent; run bake mode"
        nskip=$((nskip+1)); continue
      fi
    fi
  else
    cpuref="$CPUD/$chash.bin"
    if [ ! -s "$cpuref" ]; then
      ( cd "$SCRATCH" && "$CPU_BIN" -floatfile "$cpuref" $cpu_args ) >"$SCRATCH/cpu_$id.log" 2>&1
      if [ ! -s "$cpuref" ]; then
        printf '%s\t-\t-\t-\tBLOCKED\tcpu-fail(%s)\n' "$id" "$SCRATCH/cpu_$id.log" >> "$RESULTS"
        printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" - - - BLOCKED cpu-fail
        nother=$((nother+1)); continue
      fi
    fi
  fi

  # GPU render (timed). Perf suite: render N=5 and keep the MIN ms (warm/cold
  # jitter); every other suite renders once (nrep=1 is a 1-iteration loop,
  # identical to the old single render). corr/sum_ratio are still computed only
  # once below, from the last successful render's output.
  gout="$OUTD/$id.bin"
  nrep=1; [ "$PERF" = "1" ] && nrep=5
  ms=""; rc=0; rep=1
  while [ "$rep" -le "$nrep" ]; do
    rm -f "$gout"
    t0=$(date +%s.%N)
    ( cd "$SCRATCH" && "$KERNEL" -floatfile "$gout" $gpu_args $GPU_PREC_ARGS ) >"$SCRATCH/gpu_$id.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    ms_rep=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f",(b-a)*1000.0}')
    [ $rc -ne 0 ] || [ ! -s "$gout" ] && break
    if [ -z "$ms" ] || awk -v a="$ms_rep" -v b="$ms" 'BEGIN{exit !(a<b)}'; then ms="$ms_rep"; fi
    rep=$((rep+1))
  done
  [ -n "$ms" ] || ms="$ms_rep"
  if [ $rc -ne 0 ] || [ ! -s "$gout" ]; then
    printf '%s\t-\t-\t%s\tBLOCKED\tgpu-fail(exit %s; %s)\n' "$id" "$ms" "$rc" "$SCRATCH/gpu_$id.log" >> "$RESULTS"
    printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" - - "$ms" BLOCKED "gpu-fail(exit $rc)"
    rm -f "$gout"; nother=$((nother+1)); continue
  fi

  read -r corr sr mr wpf pmr wis < <("$METRICS" "$gout" "$cpuref" 2>/dev/null)
  if [ -z "${corr:-}" ]; then
    printf '%s\t-\t-\t%s\tBLOCKED\tmetric-fail\n' "$id" "$ms" >> "$RESULTS"
    printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" - - "$ms" BLOCKED metric-fail
    rm -f "$gout"; nother=$((nother+1)); continue
  fi

  verdict=$(awk -v c="$corr" -v s="$sr" -v cm="$CORR_MIN" -v sl="$SR_MIN" -v su="$SR_MAX" \
            'BEGIN{print (c>=cm && s>=sl && s<=su)?"PASS":"FAIL"}')
  # perf suite is warn-not-fail: a parity miss is diagnosed, not gated (see the
  # TIER/exit override below); no perf golden is ever required.
  [ "$PERF" = "1" ] && [ "$verdict" = "FAIL" ] && echo "# perf-warn: $id corr=$corr sr=$sr"
  note="mr=$mr wpf=$wpf pmr=$pmr $wis"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$corr" "$sr" "$ms" "$verdict" "$note" >> "$RESULTS"
  printf '%-34s %-11s %-10s %-8s %-8s %s\n' "$id" "$corr" "$sr" "$ms" "$verdict" "$note"
  case "$verdict" in PASS) npass=$((npass+1));; FAIL) nfail=$((nfail+1));; esac
  # bake / deathstar mode seeds each frozen cell's expected verdict into golden.
  [ "$BAKE_MODE" = "1" ] && [ "$cpu_class" = "baked" ] && seed_golden "$id" "$verdict" "$corr" "$sr"
  [ "$DEATHSTAR_MODE" = "1" ] && [ "$cpu_class" = "deathstar" ] && seed_golden "$id" "$verdict" "$corr" "$sr"
  rm -f "$gout"
done < "$CELLFILE"

fi   # end render (non-score-only) block

# --- raw tally over the WHOLE results.tsv (combines chunked/resumed runs) -----
# REJECT (guards suite: GPU correctly refused, exit 9) is its own bucket, NOT
# folded into BLOCKED -- it is the EXPECTED outcome for a reject-gate cell.
WP=0; WF=0; WB=0; WS=0; WR=0; WT=0
while IFS=$'\t' read -r nm _c _s _ms v _note; do
  case "$nm" in \#*|"") continue;; esac
  WT=$((WT+1))
  case "$v" in PASS) WP=$((WP+1));; FAIL) WF=$((WF+1));; BLOCKED) WB=$((WB+1));; SKIP) WS=$((WS+1));; REJECT) WR=$((WR+1));; esac
done < "$RESULTS"

FINAL=0; [ "$RANGE_HI" -ge "$NTOTAL" ] && FINAL=1
GOLDEN_EXISTS=0; [ -s "$GOLDEN" ] && GOLDEN_EXISTS=1

# =============================================================================
# SUITE VERDICT via GOLDEN (expected-verdict) comparison.
#
# The per-cell gate above is unchanged. Layered on top: each cell's ACTUAL
# PASS/FAIL is compared to its EXPECTED verdict in golden/<suite>.<precision>.tsv
# (committed, lockfile-style). The suite PASSES iff every cell matches its golden;
# it FAILS iff any cell FLIPPED (pass->fail or fail->pass) -- that flip IS the
# regression signal, and the flipped cells are named below. This lets a correct
# fp32 run (184/320, 136 legitimately-FAIL precision-heavy cells) read as PASS
# while still catching a single cell that changes verdict.
#
# Comparison is STRICT (no tolerance band): fp32's known near-gate jitter is
# absorbed by the golden itself -- a cell that legitimately sits either side of
# corr==0.9999 is recorded at its baseline verdict, so only a genuine change vs
# that baseline flips it. If a real GPU re-run ever shows spurious flips on cells
# whose corr sits within a tiny epsilon of 0.9999, a documented epsilon band may
# be added here -- deliberately NOT added preemptively.
#
# Only the final chunk (RANGE_HI >= NTOTAL) holds a complete results.tsv, so only
# it can decide the suite; earlier chunks defer the verdict and exit 0.
# =============================================================================
GOLD_USED=0; NFLIP=0; FLIP_PF=0; FLIP_FP=0; FLIP_OTHER=0; GMATCH=0; flips=""
if [ "$GOLDEN_EXISTS" -eq 1 ] && [ "$FINAL" -eq 1 ]; then
  GOLD_USED=1
  declare -A GOLD SEEN
  # 4-col golden (cell verdict corr sum_ratio), backward-compatible with the
  # older 2-col (cell verdict) format: a 2-col line leaves gcorr/gsr empty,
  # which is fine since the gate below reads gv only.
  while IFS=$'\t' read -r gc gv gcorr gsr; do
    case "$gc" in \#*|"") continue;; esac
    GOLD["$gc"]="$gv"
  done < "$GOLDEN"
  while IFS=$'\t' read -r nm _c _s _ms v _note; do
    case "$nm" in \#*|"") continue;; esac
    SEEN["$nm"]=1
    # A baked cell SKIPPED on a routine run was not evaluated -- not a flip.
    [ "$v" = "SKIP" ] && continue
    exp="${GOLD[$nm]:-}"
    if [ -z "$exp" ]; then
      FLIP_OTHER=$((FLIP_OTHER+1)); flips+=$'\n'"    $nm : absent-from-golden actual=$v"
    elif [ "$v" = "$exp" ]; then
      GMATCH=$((GMATCH+1))
    else
      # REJECT is handled by the generic string-equality match above (golden
      # REJECT == actual REJECT is a GMATCH) and by the catch-all "*" arm here
      # (golden REJECT vs actual FAIL/BLOCKED, or vice versa, IS a flip).
      case "$exp:$v" in
        PASS:FAIL) FLIP_PF=$((FLIP_PF+1)); flips+=$'\n'"    $nm : pass -> fail";;
        FAIL:PASS) FLIP_FP=$((FLIP_FP+1)); flips+=$'\n'"    $nm : fail -> pass";;
        *)         FLIP_OTHER=$((FLIP_OTHER+1)); flips+=$'\n'"    $nm : $exp -> $v";;
      esac
    fi
  done < "$RESULTS"
  for gc in "${!GOLD[@]}"; do
    [ -n "${SEEN[$gc]:-}" ] || { FLIP_OTHER=$((FLIP_OTHER+1)); flips+=$'\n'"    $gc : missing-from-results (expected ${GOLD[$gc]})"; }
  done
  NFLIP=$((FLIP_PF+FLIP_FP+FLIP_OTHER))
fi

# --- decide TIER -------------------------------------------------------------
if [ "$GOLD_USED" -eq 1 ]; then
  [ "$NFLIP" -eq 0 ] && TIER="PASS" || TIER="FAIL"
elif [ "$GOLDEN_EXISTS" -eq 1 ] && [ "$FINAL" -eq 0 ]; then
  TIER="PARTIAL"                       # partial chunk; golden verdict deferred
else
  if [ "$WF" -eq 0 ] && [ "$WB" -eq 0 ]; then TIER="PASS"; else TIER="FAIL"; fi
fi
# perf suite is warn-not-fail: correctness misses are diagnosed (perf-warn lines
# above), never gated. No perf golden is ever required, so TIER is forced PASS
# regardless of raw FAILs or (absent) golden flips.
[ "$PERF" = "1" ] && TIER="PASS"
SUMMARY="TIER $SUITE_NAME $TIER $WP/$NTOTAL"

# Append the summary block once the last cell has been run (never in score-only).
if [ "$FINAL" -eq 1 ] && [ "$SCORE_ONLY" != "1" ]; then
  gather_provenance
  {
    echo "# =============================================================="
    echo "# $SUMMARY   (PASS=$WP FAIL=$WF BLOCKED=$WB SKIP=$WS REJECT=$WR  of $WT scored)"
    if [ "$GOLD_USED" -eq 1 ]; then
      echo "# golden=$GOLDEN  flips=$NFLIP (pass->fail=$FLIP_PF fail->pass=$FLIP_FP other=$FLIP_OTHER)"
    else
      echo "# golden=<none for $SUITE_NAME.$PRECISION>  (verdict from raw counts)"
    fi
    echo "# kernel=$KERNEL md5=${KERNEL_MD5:-n/a}"
    echo "# cpu_ref=$CPU_BIN md5=${CPU_MD5:-n/a}"
    echo "# device=${DEVICE_EVIDENCE:-n/a}"
  } >> "$RESULTS"
  emit_provenance "$RESULTS"
fi

echo "# =============================================================="
[ "$SCORE_ONLY" = "1" ] || echo "# this invocation: PASS=$npass FAIL=$nfail BLOCKED=$nother SKIP=$nskip"
echo "$SUMMARY   (whole-file PASS=$WP FAIL=$WF BLOCKED=$WB SKIP=$WS REJECT=$WR of $WT scored)"
if [ "$GOLD_USED" -eq 1 ]; then
  if [ "$NFLIP" -eq 0 ]; then
    echo "# golden $SUITE_NAME.$PRECISION: 0 flips ($GMATCH/$WT cells match expected) -> suite PASS"
  else
    echo "# golden $SUITE_NAME.$PRECISION: $NFLIP FLIP(S) vs expected (pass->fail=$FLIP_PF fail->pass=$FLIP_FP other=$FLIP_OTHER):$flips"
  fi
elif [ "$GOLDEN_EXISTS" -eq 1 ] && [ "$FINAL" -eq 0 ]; then
  echo "# non-final chunk (range ${RANGE_LO}-${RANGE_HI} of $NTOTAL); golden verdict deferred to the final chunk"
else
  echo "# no golden for $SUITE_NAME.$PRECISION ($GOLDEN); reporting raw counts"
fi
echo "# results -> $RESULTS   (render: column -t < $RESULTS)"

# perf suite is warn-not-fail: always exit 0, regardless of FAIL/flips above.
[ "$PERF" = "1" ] && exit 0

if [ "$GOLD_USED" -eq 1 ]; then
  [ "$NFLIP" -eq 0 ] && exit 0 || exit 1
elif [ "$GOLDEN_EXISTS" -eq 1 ] && [ "$FINAL" -eq 0 ]; then
  exit 0                               # partial chunk; do not fail on incomplete data
else
  [ "$WF" -eq 0 ] && [ "$WB" -eq 0 ] && exit 0 || exit 1
fi
