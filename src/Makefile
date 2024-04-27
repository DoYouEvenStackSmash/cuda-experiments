CXX = g++
NVCC = nvcc
NVCCFLAGS = -ccbin g++ -I../Common -m64 --threads 0 -arch=compute_70 --compiler-options -fPIC -shared
CXXFLAGS = $(NVCCFLAGS) -std=c++11  `python3 -m pybind11 --includes`
LDFLAGS = `python3-config --ldflags`
SRC_DIR = .
OBJ_DIR = obj
BIN_DIR = bin

SRC_FILES := $(wildcard $(SRC_DIR)/numpy_multiply.cpp,$(SRC_DIR)/cuda_multiply_kernel.cu)
OBJ_FILES := $(SRC_DIR)/numpy_multiply.o $(SRC_DIR)/cuda_multiply_kernel.o
DEP_FILES := $(OBJ_FILES:.o=.d)
TARGET = numpy_multiply.so

all: $(TARGET)

cuda_multiply_kernel.o: cuda_multiply_kernel.cu
	$(NVCC) $(CXXFLAGS) -c $

numpy_multiply.o: numpy_multiply.cpp cuda_multiply_kernel.cu
	$(NVCC) $(CXXFLAGS) $(OBJ_FILES) -c $

-include $(DEP_FILES)

$(TARGET): cuda_multiply_kernel.o numpy_multiply.o
	$(NVCC) $(CXXFLAGS) $(OBJ_FILES) -o test



.PHONY: clean

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
