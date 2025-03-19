#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
       
        int pixel_idx = idx * 4;
        image[pixel_idx] = 255 - image[pixel_idx];         // R
        image[pixel_idx + 1] = 255 - image[pixel_idx + 1]; // G
        image[pixel_idx + 2] = 255 - image[pixel_idx + 2]; // B
       
    }
}

int main() {
    int width = 1;
    int height = 2;
    int size = width * height * 4;
    
    // host memory
    unsigned char *image = (unsigned char*)malloc(size);
    
    // image values
    image[0] = 255; image[1] = 0;   image[2] = 128; image[3] = 255; // First pixel
    image[4] = 0;   image[5] = 255; image[6] = 0;   image[7] = 255; // Second pixel
    
    printf("Original image:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", image[i]);
    }
    printf("\n");
    
    // device memory
    unsigned char *d_image;
    cudaError_t err = cudaMalloc(&d_image, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(image);
        exit(EXIT_FAILURE);
    }
    
    // host memory to device
    err = cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(image);
        exit(EXIT_FAILURE);
    }
    
    clock_t start = clock();
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(image);
        exit(EXIT_FAILURE);
    }
    clock_t end = clock();
    
    // device to host
    err = cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result back to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(image);
        exit(EXIT_FAILURE);
    }
    
    printf("Inverted image:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", image[i]);
    }
    printf("\n");
    
    printf("Processing time: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // free memory
    cudaFree(d_image);
    free(image);
    
    return 0;
}