# Installing and building nanoBragg

This document collects install and build steps. The canonical one-line compile for the C program remains in [README.md](README.md); details below add a reproducible compiler environment and room for GPU and Python builds.

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

## CUDA build

*To be expanded.* Prerequisites, Makefile usage, and CUDA toolkit notes are summarized in [cuda/README.md](cuda/README.md).

---

## PyTorch (`nanobragg_torch` and package install)

*To be expanded.* Editable install, dependencies, and CUDA wheel considerations are described in [README_PYTORCH.md](README_PYTORCH.md) and [pyproject.toml](pyproject.toml).
