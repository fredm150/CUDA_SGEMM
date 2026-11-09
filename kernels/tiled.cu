#include "kernels.cuh"

__global__ void sgemm_tiled(int M, int N, int K, float alpha,
                            const float* A, const float* B, float beta, float* C) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    float sum = 0.0f;
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_WIDTH + tx;
        s_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        int b_row = t * TILE_WIDTH + ty;
        s_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += s_A[ty][k] * s_B[k][tx];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
