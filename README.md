# CUDA SGEMM

Step-by-step comparison of a naive and a shared-memory tiled single-precision
matrix multiply (`C = alpha*A*B + beta*C`) in CUDA, benchmarked against cuBLAS.

## Overview

The kernels were run on an NVIDIA RTX 4090 (Ada, CC 8.9):

![Performance of the kernels](benchmark.png)

GFLOPs at matrix size 4096×4096:

| Kernel | GFLOPs/s | Performance relative to cuBLAS |
|---|---:|---:|
| 1: Naive | 1473.5 | 9.1% |
| 2: Tiled (SMEM) | 2199.2 | 13.6% |
| 0: cuBLAS | 16182.9 | 100.0% |

Full sweep (GFLOP/s), median of 5 runs from `./bench`:

| Size (N×N) | Naive | Tiled | Speedup | cuBLAS |
|---:|---:|---:|---:|---:|
| 512  | 1326.6 | 1680.9 | 1.27× | 8041.2  |
| 1024 | 1495.3 | 1902.9 | 1.27× | 13542.4 |
| 2048 | 1603.6 | 2622.6 | 1.64× | 21708.2 |
| 4096 | 1473.5 | 2199.2 | 1.49× | 16182.9 |

The tiled kernel is roughly ~1.3–1.6× faster than the naive kernel.

## Setup

Requires the CUDA toolkit (for `nvcc` and cuBLAS) and an NVIDIA GPU. Find your GPU's
compute capability with `nvidia-smi --query-gpu=compute_cap --format=csv`
(RTX 4090 → 8.9 → `sm_89`).

Files:
- `naive.cu` — naive kernel (one thread per output element)
- `tiled.cu` — shared-memory tiled kernel (`TILE_WIDTH = 32`)
- `kernels.cuh` — kernel declarations
- `bench.cu` — benchmark driver (size sweep, CUDA-event timing, cuBLAS)

## Build & run

```bash
make ARCH=sm_89 run        # builds ./bench and runs it
./bench > results.csv      # save CSV
python3 plot_results.py    # 
```

## Profiling (Nsight Compute)

```bash
nvcc -O3 -arch=sm_89 -lineinfo bench.cu naive.cu tiled.cu -o bench -lcublas
sudo ncu -k sgemm_tiled -c 1 ./bench         # one tiled launch (N=512)
sudo ncu -k naive_sgemm -c 1 ./bench         # one naive launch (N=512)
sudo ncu -k naive_sgemm -s 63 -c 1 ./bench   # skip to the N=4096 launch
```


Profile at N=512 (one launch of each kernel):

| Metric | Naive | Tiled | Reading |
|---|---:|---:|---|
| Duration | 265.95 µs | 213.92 µs | tiled is ~20% faster |
| Compute (SM) throughput | 84.76% | 72.33% | — |
| L1/TEX cache throughput | 96.49% | 82.39% | naive saturates the load pipe |
| L2 cache throughput | 8.68% | 8.53% | — |
| DRAM throughput | 4.14% | 3.41% | both low: 512² fits in cache |
| Achieved occupancy | 65.91% | 66.49% | capped by 1024-thread blocks |
| Static shared mem / block | 0 B | 8.19 KB | tiling's two 32×32 tiles |

At 512 both kernels are cache-resident, so the difference is the L1/TEX load pipe:
naive pins it at 96.5% (two global loads per multiply-add), while tiling serves most
reads from shared memory and drops it to 82%. Profile the 4096 launch to see DRAM
throughput actually diverge.

## Next steps

- **Register blocking (1D/2D)** — each thread computes many outputs, not one. Biggest single gain.
- **Vectorized loads** (`float4`) and coalesced access.
- **Double-buffering** — prefetch the next tile while computing the current one.

