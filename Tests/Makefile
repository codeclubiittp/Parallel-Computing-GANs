# Compiler and build flags
CXX := g++
CXXFLAGS := -std=c++17 -g -Wall -Wextra

# Build directory
BUILDDIR := build

# CUDA paths and libraries
CUDA_PATH := /usr/local/cuda-12.6
CUDA_LIB_PATH := $(CUDA_PATH)/lib64
CUDA_LIBS := -L$(CUDA_LIB_PATH) -lcudart -lcudnn -lcublas

# Include paths
INCLUDES := -I./dependencies/include \
            -I/usr/include \
            -I/usr/local/include \
            -I$(CUDA_PATH)/include

# Linker flags including CUDA libraries
LDFLAGS := $(CUDA_LIBS)

# Test source files
TESTS := Test_convolution2D.cpp \

# Create a list of binary output files
TEST_BINARIES := $(patsubst %.cpp,$(BUILDDIR)/%,$(TESTS))

# Create the build directory if it doesn't exist
$(shell mkdir -p $(BUILDDIR))

# Default target
.PHONY: all
all: $(TEST_BINARIES)

# Rule to compile each test into a binary
$(BUILDDIR)/%: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LDFLAGS) -o $@

# Clean rule
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILDDIR)
	@echo "Clean complete"

# Run rule
.PHONY: run
run: all
	@echo "Running tests..."
	@for test in $(TEST_BINARIES); do \
		echo "\nRunning $$test:"; \
		./$$test; \
	done