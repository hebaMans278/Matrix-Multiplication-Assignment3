#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1024
#define N 512
#define K 768

void matrixMultiplication(float* mat1, float* mat2, float* product, int m, int n, int k)
{
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += mat1[row * k + i] * mat2[i * n + col];
            }
            product[row * n + col] = sum;
        }
    }
}

int main()
{
    float* h_mat1 = (float*)malloc(M * K * sizeof(float));
    float* h_mat2 = (float*)malloc(K * N * sizeof(float));
    float* h_product = (float*)malloc(M * N * sizeof(float));

    // Matrix initialization

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrixMultiplication(h_mat1, h_mat2, h_product, M, N, K);

    gettimeofday(&end, NULL);
    float elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_usec - start.tv_usec) / 1000.0f;
    printf("Basic Matrix Multiplication:\nMatrix Size: %dx%d\nElapsed time: %.2f ms\n", M, N, elapsedTime);

    // Perform verification or output the result as desired

    free(h_mat1);
    free(h_mat2);
    free(h_product);

    return 0;
}
