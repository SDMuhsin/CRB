# bench_ik - microbenchmark harness for ik_llama.cpp CPU kernels (iqk_mul_mat)
#
# Requires an existing CPU-only Release build of ik_llama.cpp at $(IK_BUILD)
# (cmake -B build -DGGML_CUDA=OFF -DGGML_NATIVE=ON; cmake --build build -t ggml).
# Links against the shared libggml.so which exports the iqk entry points.

IK       ?= ../../../temp/ik_llama.cpp
IK_BUILD ?= $(IK)/build

CXX      ?= g++
CXXFLAGS  = -O2 -g -march=native -std=c++17 -fopenmp \
            -I$(IK)/ggml/include -I$(IK)/ggml/src
LDFLAGS   = -L$(IK_BUILD)/ggml/src -lggml \
            -Wl,-rpath,$(abspath $(IK_BUILD)/ggml/src) -fopenmp -pthread

all: bench_ik

bench_ik: bench_ik.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f bench_ik

.PHONY: all clean
