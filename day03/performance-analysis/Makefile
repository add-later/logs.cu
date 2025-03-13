CC = gcc
NVCC = nvcc
CFLAGS = -O2
NVCCFLAGS = -O2 -arch=sm_75

all: cpu gpu

cpu: src/main_cpu.c
	$(CC) $(CFLAGS) src/main_cpu.c -o cpu

gpu: src/main_gpu.cu
	$(NVCC) $(NVCCFLAGS) src/main_gpu.cu -o gpu

clean:
	rm -f cpu gpu