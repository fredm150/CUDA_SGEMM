#include "kernels.cuh"

__global__ void naive_sgemm(int M, int N, int K, float alpha,
                            const float* A, const float* B, float beta, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
