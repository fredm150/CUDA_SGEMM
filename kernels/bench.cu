#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels.cuh"

/* Run STMT once to warm up, then time ITERS runs; store avg ms in OUT. */
#define TIME_MS(STMT, OUT, ITERS) do {                         \
    cudaEvent_t _s, _e;                                        \
    cudaEventCreate(&_s); cudaEventCreate(&_e);                \
    STMT; cudaDeviceSynchronize();                             \
    cudaEventRecord(_s);                                       \
    for (int _i = 0; _i < (ITERS); _i++) { STMT; }             \
    cudaEventRecord(_e); cudaEventSynchronize(_e);             \
    cudaEventElapsedTime(&(OUT), _s, _e); (OUT) /= (ITERS);    \
    cudaEventDestroy(_s); cudaEventDestroy(_e);                \
} while (0)

int main(void) {
    int sizes[] = {512, 1024, 2048, 4096};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    float alpha = 1.0f, beta = 0.0f;
    int iters = 20;

    cublasHandle_t handle;
    cublasCreate(&handle);
    printf("size,naive_GFLOPs,tiled_GFLOPs,cublas_GFLOPs,speedup\n");

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        size_t bytes = (size_t)n * n * sizeof(float);

        float* hA = (float*)malloc(bytes);
        float* hB = (float*)malloc(bytes);
        for (size_t i = 0; i < (size_t)n * n; i++) { hA[i] = 1.0f; hB[i] = 2.0f; }

        float *dA, *dB, *dC;
        cudaMalloc((void**)&dA, bytes);
        cudaMalloc((void**)&dB, bytes);
        cudaMalloc((void**)&dC, bytes);
        cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

        float nm, tm, cm;
        TIME_MS(naive_sgemm<<<grid, block>>>(n, n, n, alpha, dA, dB, beta, dC), nm, iters);
        TIME_MS(sgemm_tiled<<<grid, block>>>(n, n, n, alpha, dA, dB, beta, dC), tm, iters);
        /* cuBLAS is column-major, our data is row-major -> pass B,A swapped to get row-major C */
        TIME_MS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n), cm, iters);

        double flop = 2.0 * n * n * n;   /* 2*M*N*K */
        printf("%d,%.1f,%.1f,%.1f,%.2fx\n",
               n,
               flop / (nm / 1e3) / 1e9,
               flop / (tm / 1e3) / 1e9,
               flop / (cm / 1e3) / 1e9,
               nm / tm);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hA); free(hB);
    }

    cublasDestroy(handle);
    return 0;
}
