# ================================
# Basic configurations
# ================================
CXX := g++
NVCC := nvcc
TARGET := mlp
SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build

# ================================
# Common flags
# ================================
COMMON_CXX_FLAGS := -std=c++17 -Wall -Wextra -pedantic -I$(INCLUDE_DIR)
OMP_CPU_FLAGS := -fopenmp
NVCCFLAGS := -std=c++17 -I$(INCLUDE_DIR) -arch=sm_86 -O3

# ================================
# Build mode specific
# ================================
ifeq ($(BUILD_MODE),debug)
    CXXFLAGS := $(COMMON_CXX_FLAGS) -O0 -g3 -DDEBUG
    NVCCFLAGS += -O0 -G -g
    BUILD_SUBDIR := debug
else ifeq ($(BUILD_MODE),omp)
    CXXFLAGS := $(COMMON_CXX_FLAGS) -O3 -march=native -DUSE_OMP $(OMP_CPU_FLAGS)
    BUILD_SUBDIR := omp
else ifeq ($(BUILD_MODE),cuda)
    CXXFLAGS := $(COMMON_CXX_FLAGS) -O3 -march=native -DUSE_CUDA
    NVCCFLAGS += -DUSE_CUDA
    BUILD_SUBDIR := cuda
else
    CXXFLAGS := $(COMMON_CXX_FLAGS) -O3 -march=native
    BUILD_SUBDIR := release
endif

# ================================
# Linker flags
# ================================
ifeq ($(BUILD_MODE),cuda)
    LINKER := $(NVCC)
    LDFLAGS := -lcublas -Xcompiler -fopenmp
else ifeq ($(BUILD_MODE),omp)
    LINKER := $(CXX)
    LDFLAGS := $(OMP_CPU_FLAGS)
else
    LINKER := $(CXX)
    LDFLAGS := -fopenmp
endif

# ================================
# File configurations
# ================================
BUILD_DIR_MODE := $(BUILD_DIR)/$(BUILD_SUBDIR)

CPP_SRC := $(filter-out $(SRC_DIR)/Toolkit_%.cpp, $(wildcard $(SRC_DIR)/*.cpp))
CU_SRC :=

ifeq ($(BUILD_MODE),omp)
    CPP_SRC += $(SRC_DIR)/Toolkit_omp.cpp
else ifeq ($(BUILD_MODE),cuda)
    CU_SRC := $(SRC_DIR)/Toolkit_cuda.cu
else
    CPP_SRC += $(SRC_DIR)/Toolkit_serial.cpp
endif

CPP_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR_MODE)/%.o,$(CPP_SRC))
CU_OBJ := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR_MODE)/%.o,$(CU_SRC))
DEPS := $(CPP_OBJ:.o=.d)

# ================================
# Main rules
# ================================
all: release

$(TARGET): $(CPP_OBJ) $(CU_OBJ)
	$(LINKER) $^ -o $@ $(LDFLAGS)

# Rule for .cpp files
$(BUILD_DIR_MODE)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

# Rule for .cu files
$(BUILD_DIR_MODE)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c "$<" -o "$@"

-include $(DEPS)

# ================================
# Build modes
# ================================
debug:
	$(MAKE) BUILD_MODE=debug $(TARGET)

release:
	$(MAKE) BUILD_MODE=release $(TARGET)

omp:
	$(MAKE) BUILD_MODE=omp $(TARGET)

cuda:
	$(MAKE) BUILD_MODE=cuda $(TARGET)

# ================================
# Cleaning
# ================================
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

clean-all:
	rm -rf build $(TARGET)

.PHONY: all clean clean-all debug release omp cuda
