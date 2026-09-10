#!/usr/bin/env bash
# Run C↔PyTorch parity tests (tests/test_parity_matrix.py) on a GPU cluster node.
#
# Usage (after interactive allocation):
#   salloc ... your usual args ...
#   cd /path/to/nanoBragg
#   bash scripts/cluster/run_parity_matrix.sh
#
# Or submit as a batch job (edit #SBATCH lines for your site):
#   sbatch scripts/cluster/run_parity_matrix.sh
#
# Environment (optional overrides):
#   REPO              — repo root (default: inferred from script location)
#   CONDA_SH          — path to conda.sh (default: $HOME/miniconda3/etc/profile.d/conda.sh)
#   PYTORCH_INDEX_URL — pip index for torch (default: cu124 wheels)
#   NB_EXTRA_PYTEST   — extra args for pytest (e.g. -k "AT-PARALLEL-002")
#   SKIP_GCC_BUILD    — if set to 1, skip gcc step and use existing $REPO/nanoBragg

# --- Slurm: edit for your account/partition/QoS ---
#SBATCH --job-name=nb-parity
#SBATCH --output=parity-%j.out
#SBATCH --error=parity-%j.err
#SBATCH --partition=ampere
#SBATCH --account=lcls:prjlumine22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

export KMP_DUPLICATE_LIB_OK=TRUE
export NB_SKIP_INFRA_GATE=1
export NB_RUN_PARALLEL=1

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda.sh not found: ${CONDA_SH} — set CONDA_SH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"

cd "${REPO}"

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi || true
fi

if [[ "${SKIP_GCC_BUILD:-0}" != "1" ]]; then
  conda activate nanobragg_c
  if ! gcc -O -O -o nanoBragg nanoBragg.c -lm -static 2>/dev/null; then
    gcc -O2 -o nanoBragg nanoBragg.c -lm
  fi
fi

if [[ ! -x "${REPO}/nanoBragg" ]]; then
  echo "Missing executable ${REPO}/nanoBragg — build failed or set SKIP_GCC_BUILD=1 with a prebuilt binary." >&2
  exit 1
fi

export NB_C_BIN="${REPO}/nanoBragg"

conda activate nanobragg_torch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible — run on a GPU node'" || {
  echo "Warning: torch.cuda.is_available() is False; parity may still run on CPU but is not the intended cluster use." >&2
}

if ! python -c "import nanobrag_torch" 2>/dev/null; then
  pip install --upgrade pip
  pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"
  pip install -e ".[test]"
fi

# shellcheck disable=SC2086
exec pytest -v tests/test_parity_matrix.py --tb=short ${NB_EXTRA_PYTEST:-}
