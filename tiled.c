%%cu
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1024
#define N 512
#define K 768

#define TILE_SIZE 32

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
    __shared__ float tile1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile2[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < m && t * TILE_SIZE + threadIdx.x < k) {
            tile1[threadIdx.y][threadIdx.x] = mat1[row * k + t * TILE_SIZE + threadIdx.x];
        } else {
            tile1[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && t * TILE_SIZE + threadIdx.y < k) {
            tile2[threadIdx.y][threadIdx.x] = mat2[(t * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            tile2[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile1[threadIdx.y][i] * tile2[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
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

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
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
    printf("Tiled Matrix Multiplication:\nMatrix Size: %dx%d\nElapsed time: %.2f ms\n", M, N, elapsedTime);

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
