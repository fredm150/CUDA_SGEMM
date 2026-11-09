#ifndef KERNELS_CUH
#define KERNELS_CUH

#define TILE_WIDTH 32

/* Both kernels compute C = alpha*A*B + beta*C for row-major M x K, K x N, M x N. */
__global__ void naive_sgemm(int M, int N, int K, float alpha,
                            const float* A, const float* B, float beta, float* C);

__global__ void sgemm_tiled(int M, int N, int K, float alpha,
                            const float* A, const float* B, float beta, float* C);

#endif
