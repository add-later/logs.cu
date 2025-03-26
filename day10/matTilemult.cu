#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

__global__ void tileMatMul(const float *A, const float *B, float *C, int width) {

    // shared memory array to store sub-matrices
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float tempVal = 0.0f;

    for (int m=0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; m++){
        if (row < width && (m * TILE_WIDTH + threadIdx.x) < width) {
            tileA[threadIdx.y][threadIdx.x] = A[row * width + m * TILE_WIDTH + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < width && (m * TILE_WIDTH + threadIdx.y) < width) {
            tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * width + col]; 
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // to ensure all threads finished loading
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            tempVal += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        // no thread will start to load next tile until everything is finished
        __syncthreads();
    }
    if (row < width && col < width) {
        C[row * width + col] = tempVal;
    }
}

void matInit(float *mat, int width) {
    for (int i=0; i<width * width; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}


int main() {
    int width = 1024;
    size_t size = width * width * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed");
        exit(EXIT_FAILURE);
    }

    float *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_B error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_C error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    tileMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch err %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Tile-based matrix multiplication completed!");
}

