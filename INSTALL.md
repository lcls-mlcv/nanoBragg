# Installing and building nanoBragg

This document collects install and build steps. The canonical one-line compile for the C program remains in [README.md](README.md) (see **C reference implementation**); PyTorch CLI usage and pytest notes are under **PyTorch implementation** in the same file. Details below add reproducible conda environments for the C toolchain, native CUDA builds, and the Python/PyTorch package.

## C reference binary (`nanobragg_c` conda environment)

Use a minimal **conda-forge** toolchain so `gcc` matches across machines without relying on the system compiler.

### Create the environment

```bash
conda create -n nanobragg_c -c conda-forge c-compiler cxx-compiler
```

(or `mamba` instead of `conda`). Python is optional for this path; the metapackages pull `gcc_linux-64`, binutils, and the Linux sysroot used by the compiler.

### Build (from repository root)

```bash
conda activate nanobragg_c
cd /path/to/nanoBragg
gcc -O -O -o nanoBragg nanoBragg.c -lm -static
```

This matches the [README.md](README.md) compile section.

### Static linking

The `-static` flag links against static `libc` and produces a self-contained executable when the linker can find `libc.a`. Conda’s GCC uses its own sysroot; if linking fails with missing `libc` / `libc.a`, install your OS static C library (on RHEL-family systems, `glibc-static` via `dnf`/`yum`) or build without `-static` for a dynamically linked binary (convenient for local use, not identical to the README line).

### Verify the binary

- `file nanoBragg` should report a statically linked executable when `-static` succeeded.
- `ldd nanoBragg` should report `not a dynamic executable` for a fully static build.
- Running `./nanoBragg` without inputs prints usage and expected options (nonzero exit if no `-hkl`/dump file is expected).

---

## CUDA build (`nanobragg_cuda` conda environment)

GPU builds use [cuda/Makefile](cuda/Makefile) from the [`cuda/`](cuda/) directory. Upstream [cuda/README.md](cuda/README.md) assumes a system CUDA layout under `/usr/local/cuda-*`; **conda** installs libraries under `$CONDA_PREFIX/lib/` (not only `lib64/`). The Makefile adds `-L` paths for both `lib64` and `lib` so linking works for system and conda toolkits.

The Makefile’s default SASS targets include **through `sm_90` when using CUDA 12.2** (`nvcc` 12.2 does not support `compute_100` / `compute_120`; re-add those arches if you build with a newer toolkit).

### Driver vs toolkit (example: S3DF Ampere)

On **S3DF Ampere** nodes, `nvidia-smi` may report **CUDA Version: 12.2** (e.g. driver 535.x). Pin the **conda** compiler stack to **CUDA 12.2** so the built binary matches that driver. On other sites, check `nvidia-smi` and choose a matching toolkit minor version.

### Create the environment

Pin **`cuda-nvcc`** and **`cuda-cudart*`** to the **same 12.2.x** build so the solver does not upgrade **`cuda-nvcc`** to CUDA 13.x:

```bash
conda create -n nanobragg_cuda -c nvidia -c conda-forge \
  cuda-nvcc=12.2.140 cuda-cudart=12.2.140 cuda-cudart-dev=12.2.140 cuda-cudart-static=12.2.140 \
  gcc_linux-64=12 gxx_linux-64=12 c-compiler cxx-compiler make
```

**Host GCC:** CUDA **12.2** `nvcc` supports GNU **C++ up to gcc 12**. If `conda create` pulls **gcc 14**, `nvcc` will fail against libstdc++ headers—pin **`gcc_linux-64=12`** and **`gxx_linux-64=12`** as above, or run `conda install -n nanobragg_cuda -c conda-forge gcc_linux-64=12 gxx_linux-64=12` after the fact.

After creation, confirm:

```bash
conda activate nanobragg_cuda
nvcc --version    # should show Cuda compilation tools, release 12.2
ls "$CONDA_PREFIX/lib"/libcudart* "$CONDA_PREFIX/bin/nvcc"
```

### Build

Point **`CUDA_PATH`** at the env prefix (where `bin/nvcc` and `lib/` live):

```bash
conda activate nanobragg_cuda
export CUDA_PATH="$CONDA_PREFIX"
cd /path/to/nanoBragg/cuda
make clean
make all
```

Release binary: `cuda/build/release/nanoBraggCUDA`.

### PyPI CUDA wheels vs this env

Another conda env may install pip packages such as `nvidia-cuda-runtime-cu12` (e.g. 12.8.x) for PyTorch. Those are **runtime** wheels for Python; they do not replace **`nvcc`** and dev libraries for compiling `nanoBraggCUDA`. Keep **`nanobragg_cuda`** as the dedicated native build env when versions differ.

### Verify

- `nvcc --version` shows release **12.2** when using the pins above.
- `ldd build/release/nanoBraggCUDA` lists CUDA libraries resolved from `$CONDA_PREFIX/lib` or the system.
- Optional: `sh test/test_release.sh` from [`cuda/`](cuda/) per [cuda/README.md](cuda/README.md).

---

## PyTorch (`nanobragg_torch` conda environment)

The Python package is defined in [`pyproject.toml`](pyproject.toml) (name `nanobrag-torch`, import `nanobrag_torch`). Use a **dedicated conda env** so the PyTorch/CUDA stack does not fight the C compiler env (`nanobragg_c`).

### Create the environment

Use Python **3.10–3.12** (3.11 is a good default):

```bash
conda create -n nanobragg_torch python=3.11 pip
conda activate nanobragg_torch
```

(`mamba` works in place of `conda`.)

### Install PyTorch before the project

Install **one** PyTorch build first so `pip` does not pull a second, incompatible `torch` when you install the package:

- **GPU:** Pick a CUDA wheel whose major matches your **NVIDIA driver** (check `nvidia-smi`). Typical pattern for CUDA 12.x drivers:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

  Adjust the index URL (`cu121`, `cu124`, etc.) to match [PyTorch’s current wheels](https://pytorch.org/get-started/locally/) for your OS and CUDA.

- **CPU-only:** `pip install torch` (or the CPU wheel from the same page) is enough for many correctness checks.

### Install nanoBragg (editable)

From the **repository root**:

```bash
cd /path/to/nanoBragg
pip install -e .
```

Optional dev/test tooling:

```bash
pip install -e ".[dev,test]"
```

### Runtime environment

Avoid MKL/OpenMP duplicate-library issues when multiple BLAS stacks load:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

You can add that to `$CONDA_PREFIX/etc/conda/activate.d/` as a small shell snippet if you want it on every `conda activate nanobragg_torch`.

### Verify

```bash
python -c "import torch, nanobrag_torch; print(torch.__version__, torch.cuda.is_available())"
nanoBragg -h | head -20
```

On a GPU node, `torch.cuda.is_available()` should be `True` if you installed a CUDA build.

### Tests

PyTorch-focused testing is documented under **PyTorch implementation** in [README.md](README.md) (environment variables for pytest, infrastructure gate, and C↔PyTorch parity). This file only covers **environment creation**.

### Cluster / Slurm (C↔PyTorch parity on a GPU node)

GitHub Actions hosted runners do not provide NVIDIA GPUs. To run the **parity matrix** (`tests/test_parity_matrix.py`) on a Slurm cluster (e.g. S3DF Ampere), use the helper script [`scripts/cluster/run_parity_matrix.sh`](scripts/cluster/run_parity_matrix.sh).

It sets `KMP_DUPLICATE_LIB_OK`, `NB_SKIP_INFRA_GATE=1`, and `NB_RUN_PARALLEL=1`, builds the root **`nanoBragg`** binary with **`nanobragg_c`**, then runs pytest under **`nanobragg_torch`**. Edit the `#SBATCH` lines at the top of the script for your partition, account, and wall time.

**Interactive** (after `salloc` on a GPU node):

```bash
cd /path/to/nanoBragg
bash scripts/cluster/run_parity_matrix.sh
```

**Batch:**

```bash
cd /path/to/nanoBragg
sbatch scripts/cluster/run_parity_matrix.sh
```

Optional environment variables: `CONDA_SH`, `REPO`, `PYTORCH_INDEX_URL` (default PyTorch cu124 wheel index), `NB_EXTRA_PYTEST` (e.g. `-k "AT-PARALLEL-002"`), `SKIP_GCC_BUILD=1` if `./nanoBragg` is already built.

### Dependencies

Core Python dependencies are listed under `[project]` / `[project.optional-dependencies]` in [`pyproject.toml`](pyproject.toml) (`torch>=2.3`, `fabio`, `numpy`, etc.).
