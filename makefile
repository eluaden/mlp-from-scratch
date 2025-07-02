# Basic configurations
CXX := g++
TARGET := mlp
SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build

# Common flags
COMMON_CXX_FLAGS := -std=c++17 -Wall -Wextra -pedantic -I$(INCLUDE_DIR)
OMP_CPU_FLAGS := -fopenmp

# Build mode specific
ifeq ($(BUILD_MODE),debug)
    CXXFLAGS := $(COMMON_CXX_FLAGS) -O0 -g3 -DDEBUG
    BUILD_SUBDIR := debug
else ifeq ($(BUILD_MODE),omp)
    CXXFLAGS := $(COMMON_CXX_FLAGS) -O3 -march=native -DUSE_OMP $(OMP_CPU_FLAGS)
    BUILD_SUBDIR := omp
else
    CXXFLAGS := $(COMMON_CXX_FLAGS) -O3 -march=native
    BUILD_SUBDIR := release
endif

# Linker flags
LINKER := $(CXX)
ifeq ($(BUILD_MODE),omp)
    LDFLAGS := $(OMP_CPU_FLAGS)
else
    # Adicione a flag de OpenMP mesmo no modo serial
    LDFLAGS := -fopenmp
endif

# File configurations
BUILD_DIR_MODE := $(BUILD_DIR)/$(BUILD_SUBDIR)
SRC := $(filter-out $(SRC_DIR)/Toolkit_%.cpp, $(wildcard $(SRC_DIR)/*.cpp))

# Toolkit implementation selection
ifeq ($(BUILD_MODE),omp)
    SRC += $(SRC_DIR)/Toolkit_omp.cpp
else
    SRC += $(SRC_DIR)/Toolkit_serial.cpp
endif

# Object files
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR_MODE)/%.o,$(SRC))
DEPS := $(OBJ:.o=.d)

# Main rules
all: release

$(TARGET): $(OBJ)
	$(LINKER) $(OBJ) -o $@ $(LDFLAGS)

# Rule for .cpp files
$(BUILD_DIR_MODE)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -MP -c "$<" -o "$@"

-include $(DEPS)

# Build rules
debug:
	$(MAKE) BUILD_MODE=debug $(TARGET)

release:
	$(MAKE) BUILD_MODE=release $(TARGET)

omp:
	$(MAKE) BUILD_MODE=omp $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

clean-all:
	rm -rf build $(TARGET)

.PHONY: all clean clean-all debug release omp