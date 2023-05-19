%%cu
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1024
#define N 512
#define K 768

// Utility function to check for CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
            __FILE__, __LINE__, result, cudaGetErrorString(result), #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void matrixMultiplication(float* mat1, float* mat2, float* product, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += mat1[row * k + i] * mat2[i * n + col];
        }
        product[row * n + col] = sum;
    }
}

int main()
{
    float* h_mat1 = (float*)malloc(M * K * sizeof(float));
    float* h_mat2 = (float*)malloc(K * N * sizeof(float));
    float* h_product = (float*)malloc(M * N * sizeof(float));

    // Matrix initialization

    float* d_mat1, *d_mat2, *d_product;
    CUDA_CHECK(cudaMalloc((void**)&d_mat1, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mat2, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_product, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_mat1, h_mat1, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat2, h_mat2, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Start timing
    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrixMultiplication<<<gridSize, blockSize>>>(d_mat1, d_mat2, d_product, M, N, K);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // End timing
    gettimeofday(&end, NULL);
    float elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_usec - start.tv_usec) / 1000.0f;
    printf("Basic Matrix Multiplication:\nMatrix Size: %dx%d\n", M, N);
    printf("Elapsed time: %.2f ms\n", elapsedTime);

    CUDA_CHECK(cudaMemcpy(h_product, d_product, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Perform verification or output the result as desired

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_product);

    free(h_mat1);
    free(h_mat2);
    free(h_product);

    return 0;
}
