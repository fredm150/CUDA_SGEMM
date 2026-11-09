# Set ARCH to your GPU's compute capability:
#   nvidia-smi --query-gpu=compute_cap --format=csv   (e.g. 8.0 -> sm_80)
ARCH ?= sm_80

bench: bench.cu naive.cu tiled.cu kernels.cuh
	nvcc -O3 -arch=$(ARCH) bench.cu naive.cu tiled.cu -o bench -lcublas

run: bench
	./bench

clean:
	rm -f bench results.csv

.PHONY: run clean
