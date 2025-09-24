#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unistd.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <signal.h>
// #include "NeuTracer.h"


using json = nlohmann::json;

// 全局变量用于信号处理
volatile bool running = true;

// 信号处理函数
void signalHandler(int signum) {
    std::cout << "收到信号 " << signum << "，准备退出..." << std::endl;
    running = false;
}

// CUDA 核函数（保持不变）
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// 初始化向量（保持不变）
void initializeVector(float *vec, int size, float value) {
    for (int i = 0; i < size; i++) {
        vec[i] = value + i;
    }
}

int main() {
    std::cout << "Main Process (PID: " << getpid() << "): Found " << std::endl;
    sleep(10);
    // 设置信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // 初始化 Tracer
    // NeuTracer::Tracer tracer(NeuTracer::UPROBE_CFG_PATH, "error");
    //                        
    // tracer.run();  // 开始追踪

    // 设置 CUDA 设备
    cudaSetDevice(0);

    // 打印设备信息
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Main Process (PID: " << getpid() << "): Found " 
              << deviceCount << " CUDA devices" << std::endl;

    int iteration = 0;
    const int maxIterations = 100;  // 设置最大迭代次数，也可以通过 running 标志无限循环
    
    // 主循环
    while (running && iteration < maxIterations) {
        std::cout << "迭代 " << iteration + 1 << "/" << maxIterations << std::endl;
        
        // 每次迭代调整分配的内存大小，使其更有变化
        const int numElements = 50000 + (iteration % 10) * 10000;
        const size_t size = numElements * sizeof(float);
        
        // 分配主机内存
        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C = (float *)malloc(size);
        
        // 分配设备内存
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_C, size);
        
        // 初始化数据
        initializeVector(h_A, numElements, 1.0f + iteration * 0.1f);
        initializeVector(h_B, numElements, 2.0f + iteration * 0.1f);
        
        // 拷贝数据到设备
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        
        // 启动核函数
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        
        // 等待核函数完成
        cudaDeviceSynchronize();
        
        // 拷贝结果回主机
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        
        // 打印部分结果
        std::cout << "CUDA result sample: " << h_C[0] << ", " << h_C[numElements-1] << std::endl;
        
        // 释放设备内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        // 释放主机内存
        free(h_A);
        free(h_B);
        free(h_C);
        
        // 增加迭代计数
        iteration++;
        
        // 每次迭代后休眠一定时间，避免过快分配/释放
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "完成 " << iteration << " 次分配/释放循环" << std::endl;

    // 停止追踪并退出
    // tracer.close();
    return 0;
}