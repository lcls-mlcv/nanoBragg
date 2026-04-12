# nanoBraggCUDA

## Overview

nanoBraggCUDA is a perfect-lattice nanocrystal diffraction simulator. This adaptation simulates diffraction patterns from nanocrystals using CPU and CUDA-accelerated computations.

The CUDA kernel is optimized for Kepler and Pascal architectures. nanoBraggCUDA is in the process of being updated to modern hardware and has been built and tested using CUDA 12.9 on the Rocky Linux 9 OS with an RTX 5090 GPU.

## Executing Pre-Built Binaries
After being built, the executable can be copied to other machines that have an NVIDIA GPU with NVIDIA drivers installed. While it is possible to install just the drivers, general practice is to follow the install instructions for the cuda-toolkit which include instructions for the nvidia-driver module. See NVIDIA CUDA Toolkit below.

## Prerequisites
Before building nanoBragg, ensure the following prerequisites are installed:

### NVIDIA CUDA Toolkit and Drivers
- Required for GPU acceleration.
- Download and installation instructions: [CUDA Downloads](https://developer.nvidia.com/cuda-downloads), or [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) for older versions. 
- Minimum version: CUDA 12.9.
- Ensure you have an NVIDIA GPU with CUDA support (Compute Capability 5.0 or higher)

### GCC/G++ Compiler
- Required for compiling C/C++ and CUDA code.
- Installation:
  - Ubuntu: `sudo apt update && sudo apt install build-essential`
  - Rocky Linux 9: `sudo dnf install gcc gcc-c++ make`

## Build Instructions

The project uses a Makefile for building debug and release versions.

### Common Steps
1. Clone or download the repository.
2. Ensure `CUDA_PATH` in the Makefile points to your CUDA installation (default: `/usr/local/cuda-12.9`). Edit if necessary.
3. Run `make` commands from the project root.
   - For both debug and release: `make all`
   - Debug only: `make debug` (outputs to `build/debug/nanoBraggCUDA`)
   - Release only: `make release` (outputs to `build/release/nanoBraggCUDA`)
   - Clean: `make clean`

## Run Tests
After building, run the executable (e.g., `./build/release/nanoBraggCUDA`).

Test the optimized release version: `sh ./test/test_release.sh`

## Troubleshooting
- If NVCC is not found, ensure CUDA bin path is in `$PATH`: `export PATH=/usr/local/cuda-12.9/bin:$PATH`
- For linker issues, ensure CUDA libs are in `$LD_LIBRARY_PATH`: `export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH`
- Verify CUDA installation: `nvcc --version`

## Compatibility
### Additional Hardware
nanoBraggCUDA can be built for different NVIDIA hardware, though most configurations have not been tested. Use the Hardware Compatibility Matrix below to identify the Compute Capability for the graphics card, then find a CUDA SDK version that supports it. Next determine if the CUDA SDK version is supported on the target Operating System by following the instructions above for installing the NVIDIA CUDA Toolkit. Update the CUDA_PATH and the ARCH_FLAGS in the Makefile accordingly.

The original CUDA kernel for nanoBraggCUDA was developed on and optimized for Kepler (Compute Capability 3.5) and Pascal (Compute Capability 6.1). Though the kernel should run on other hardware, it is not optimized for it. Ideally, a dedicated kernel should be written for each targeted Compute Capability.

### NVIDIA Hardware Compatibility Matrix
https://en.wikipedia.org/wiki/CUDA#GPUs_supported

## VSCode Support
Development for nanoBraggCUDA was done in VSCode. Note the .vscode folder in the repository. For IntelliSense to work properly, update the CUDA_PATH_LOCAL to point to your CUDA_PATH.

### VSCode Extensions
[C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)

[Nsight Visual Studio Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)

[Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) (optional)