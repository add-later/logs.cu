#include <cuda_runtime.h>
#include <stdio.h>
#include <ctime>

__global__ void matrixMultiply(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float value = 0;
        for (int i = 0; i < N; i++) {
            value += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = value;
    }
}

int main() {
    int M = 2;
    int N = 2;
    int K = 2;
    int size = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize matrices
    for (int i = 1; i <= 4; i++) {
        h_A[i-1] = i;
    }
    for (int i = 5; i <= 8; i++) {
        h_B[i-5] = i;
    }
    
    float *d_A, *d_B, *d_C;
    cudaError_t err;
    
    err = cudaMalloc(&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for A: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc(&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc(&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        exit(EXIT_FAILURE);
    }
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    clock_t start = clock();
    
    // Use 2D grid and block
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        exit(EXIT_FAILURE);
    }
    
    clock_t end = clock();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    printf("Matrix A:\n");
    printf("%.2f %.2f\n%.2f %.2f\n\n", h_A[0], h_A[1], h_A[2], h_A[3]);
    
    printf("Matrix B:\n");
    printf("%.2f %.2f\n%.2f %.2f\n\n", h_B[0], h_B[1], h_B[2], h_B[3]);
    
    printf("Matrix C (result):\n");
    printf("%.2f %.2f\n%.2f %.2f\n\n", h_C[0], h_C[1], h_C[2], h_C[3]);
    
    printf("GPU Matrix Multiply Time: %lf seconds\n", ((double)(end - start))/CLOCKS_PER_SEC);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}