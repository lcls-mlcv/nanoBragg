cmake_minimum_required(VERSION 3.21)
project(nanoBragg LANGUAGES C CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#Use static CUDA runtime
set (CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Set CUDA architectures (from your old Makefile; adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 61 70 80 86 90 103)

# Optimization flags (from your Makefile)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -maxrregcount=32")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#Sources
file(GLOB SOURCES 
    src/nanoBragg.c
    src/nanoBraggCPU.c
    src/nanoBraggCUDA.cu
)
#file(GLOB SOURCES "src/*.c" "src/*.cu")
add_executable(nanoBraggCUDA ${SOURCES})

# Compiler options (target-specific, generator expressions for language)
target_compile_options(nanoBraggCUDA PRIVATE -Wl,--verbose 
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3 -maxrregcount=32>
)

target_link_options(nanoBraggCUDA PRIVATE -static-libstdc++ -static-libgcc)

# Include directories
target_include_directories(nanoBraggCUDA PRIVATE 
    src
    /usr/local/cuda-12.9/include
)

# Output to build/ folder
set_target_properties(nanoBraggCUDA PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build
#    CUDA_SEPARABLE_COMPILATION ON
)