#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloGPU() {
    if (threadIdx.x == 0) {
        printf("Hello world!\n");
    }
}

int main() {
    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("Hello from GPU!");
    return 0;
}