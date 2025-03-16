#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixAddElem(float *A, float *B, float *C, int N, int M){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        int index = row * M + col;
        C[index] = A[index] + B[index];
    }
}

int main() {
   int M = 2;
    int N = 2;

    int size = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 1; i <= 4; i++) {
        h_A[i-1] = i;
    }
    for (int i = 5; i <= 8; i++) {
        h_B[i-5] = i;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAddElem<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Matrix A:\n");
    printf("%.2f %.2f\n%.2f %.2f\n\n", h_A[0], h_A[1], h_A[2], h_A[3]);
    
    printf("Matrix B:\n");
    printf("%.2f %.2f\n%.2f %.2f\n\n", h_B[0], h_B[1], h_B[2], h_B[3]);
    
    printf("Matrix C (result):\n");
    printf("%.2f %.2f\n%.2f %.2f\n\n", h_C[0], h_C[1], h_C[2], h_C[3]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}