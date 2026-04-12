CUDA_PATH ?= /usr/local/cuda-12.9
NVCC = $(CUDA_PATH)/bin/nvcc
CC = gcc

SRC_DIR = src
BUILD_DIR = build

DEBUG_DIR = $(BUILD_DIR)/debug
RELEASE_DIR = $(BUILD_DIR)/release

COMMON_CFLAGS = -I$(SRC_DIR) -I$(CUDA_PATH)/include

COMMON_LDFLAGS = -L$(CUDA_PATH)/lib64/stubs -L$(CUDA_PATH)/lib64
CUDALIB_LDFLAGS = -lcudadevrt -lcudart_static
CLIB_LDFLAGS = -lrt -lpthread -ldl -lm

LDFLAGS = $(CUDALIB_LDFLAGS) $(CLIB_LDFLAGS) $(COMMON_LDFLAGS)

STATIC_LDFLAGS = -static-libstdc++ -static-libgcc

# Architectures
ARCH_FLAGS = \
	--generate-code=arch=compute_61,code=[compute_61,sm_61] \
	--generate-code=arch=compute_70,code=[compute_70,sm_70] \
	--generate-code=arch=compute_80,code=[compute_80,sm_80] \
	--generate-code=arch=compute_89,code=[compute_89,sm_89] \
	--generate-code=arch=compute_90,code=[compute_90,sm_90] \
	--generate-code=arch=compute_100,code=[compute_100,sm_100] \
	--generate-code=arch=compute_120,code=[compute_120,sm_120]
	
NVCC_FLAGS = -forward-unknown-to-host-compiler -maxrregcount=40 -std=c++17 $(ARCH_FLAGS) -x cu

# Always verbose: Add -v to all compiler flags
DEBUG_CFLAGS = $(COMMON_CFLAGS) -g -v
RELEASE_CFLAGS = $(COMMON_CFLAGS) -O3 -v

DEBUG_NVCC_FLAGS = -g -G -v $(NVCC_FLAGS) # Debug symbols for CUDA + verbose
RELEASE_NVCC_FLAGS = -lineinfo -O3 -v $(NVCC_FLAGS)

# Object files
OBJS = nanoBragg.o nanoBraggCPU.o nanoBraggCUDA.cu.o

# Default target: build both debug and release
all: debug release

debug: | $(DEBUG_DIR)
	$(CC) $(DEBUG_CFLAGS) -c $(SRC_DIR)/nanoBragg.c -o $(DEBUG_DIR)/nanoBragg.o
	$(CC) $(DEBUG_CFLAGS) -c $(SRC_DIR)/nanoBraggCPU.c -o $(DEBUG_DIR)/nanoBraggCPU.o
	$(NVCC) $(DEBUG_NVCC_FLAGS) -c $(SRC_DIR)/nanoBraggCUDA.cu -o $(DEBUG_DIR)/nanoBraggCUDA.cu.o $(COMMON_CFLAGS)
	g++ $(STATIC_LDFLAGS) $(DEBUG_DIR)/nanoBragg.o $(DEBUG_DIR)/nanoBraggCPU.o $(DEBUG_DIR)/nanoBraggCUDA.cu.o -o $(DEBUG_DIR)/nanoBraggCUDA $(LDFLAGS)

release: | $(RELEASE_DIR)
	$(CC) $(RELEASE_CFLAGS) -c $(SRC_DIR)/nanoBragg.c -o $(RELEASE_DIR)/nanoBragg.o
	$(CC) $(RELEASE_CFLAGS) -c $(SRC_DIR)/nanoBraggCPU.c -o $(RELEASE_DIR)/nanoBraggCPU.o
	$(NVCC) $(RELEASE_NVCC_FLAGS) -c $(SRC_DIR)/nanoBraggCUDA.cu -o $(RELEASE_DIR)/nanoBraggCUDA.cu.o $(COMMON_CFLAGS)
	g++ $(STATIC_LDFLAGS) $(RELEASE_DIR)/nanoBragg.o $(RELEASE_DIR)/nanoBraggCPU.o $(RELEASE_DIR)/nanoBraggCUDA.cu.o -o $(RELEASE_DIR)/nanoBraggCUDA $(LDFLAGS)

$(DEBUG_DIR):
	mkdir -p $(DEBUG_DIR)

$(RELEASE_DIR):
	mkdir -p $(RELEASE_DIR)

clean:
	rm -rf $(BUILD_DIR)