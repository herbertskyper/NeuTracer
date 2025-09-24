#include "cuda_operations.h"
#include <cuda_runtime.h>

// CUDA核函数
__global__ void vectorAddKernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// 包装函数
void vectorAdd(const float *a, const float *b, float *c, int n) {
    float *d_a, *d_b, *d_c;
    
    // 分配设备内存
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动核函数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // 拷贝结果回主机
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}