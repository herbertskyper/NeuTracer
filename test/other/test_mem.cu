#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  // 用于 sleep() 函数

// 定义内存块大小和执行次数
#define SMALL_SIZE (10 * 1024 * 1024)  // 10MB
#define MEDIUM_SIZE (50 * 1024 * 1024)  // 50MB
#define LARGE_SIZE (100 * 1024 * 1024)  // 100MB
#define ITERATIONS 1000
#define SLEEP_SECONDS 1  // 每次迭代间隔

// 检查 CUDA 错误
#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA 错误: %s, 行: %d\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

// CUDA 内核函数 - 简单地修改数据
__global__ void modifyData(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size / sizeof(float)) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// 执行主机到设备的传输
void hostToDevice(float *h_data, float *d_data, size_t size) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
}

// 执行设备到主机的传输
void deviceToHost(float *d_data, float *h_data, size_t size) {
    CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
}

// 执行设备到设备的传输
void deviceToDevice(float *d_src, float *d_dst, size_t size) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
}

// 执行分段传输
void chunkedTransfer(float *h_data, float *d_data, size_t total_size, int num_chunks) {
    size_t chunk_size = total_size / num_chunks;
    for (int i = 0; i < num_chunks; i++) {
        size_t offset = i * chunk_size;
        CHECK_CUDA_ERROR(cudaMemcpy(
            d_data + offset / sizeof(float), 
            h_data + offset / sizeof(float), 
            chunk_size, 
            cudaMemcpyHostToDevice));
        
        printf("分段传输: 块 %d/%d 完成, 大小: %.2f MB\n", i+1, num_chunks, 
               (float)chunk_size / (1024 * 1024));
    }
}

// 执行异步传输
void asyncTransfer(float *h_data, float *d_data, size_t size, cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));
}

int main() {
    printf("当前进程 PID: %d\n", getpid());
    // 分配主机内存
    float *h_small = (float*)malloc(SMALL_SIZE);
    float *h_medium = (float*)malloc(MEDIUM_SIZE);
    float *h_large = (float*)malloc(LARGE_SIZE);
    float *h_result_small = (float*)malloc(SMALL_SIZE);
    float *h_result_medium = (float*)malloc(MEDIUM_SIZE);
    float *h_result_large = (float*)malloc(LARGE_SIZE);
    
    // 初始化主机内存
    for (int i = 0; i < SMALL_SIZE / sizeof(float); i++) h_small[i] = 1.0f;
    for (int i = 0; i < MEDIUM_SIZE / sizeof(float); i++) h_medium[i] = 2.0f;
    for (int i = 0; i < LARGE_SIZE / sizeof(float); i++) h_large[i] = 3.0f;
    
    // 分配固定内存 (pinned memory)
    float *h_pinned_medium;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_pinned_medium, MEDIUM_SIZE));
    for (int i = 0; i < MEDIUM_SIZE / sizeof(float); i++) h_pinned_medium[i] = 4.0f;
    
    // 分配设备内存
    float *d_small, *d_medium, *d_large, *d_temp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_small, SMALL_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&d_medium, MEDIUM_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&d_large, LARGE_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, LARGE_SIZE));
    
    // 创建 CUDA 流
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // 核心配置
    int blockSize = 256;
    int smallGridSize = (SMALL_SIZE / sizeof(float) + blockSize - 1) / blockSize;
    int mediumGridSize = (MEDIUM_SIZE / sizeof(float) + blockSize - 1) / blockSize;
    int largeGridSize = (LARGE_SIZE / sizeof(float) + blockSize - 1) / blockSize;
    
    printf("开始 CUDA 内存传输测试，将执行 %d 次迭代\n", ITERATIONS);
    printf("小数据块: %.2f MB, 中数据块: %.2f MB, 大数据块: %.2f MB\n", 
           (float)SMALL_SIZE / (1024 * 1024), 
           (float)MEDIUM_SIZE / (1024 * 1024),
           (float)LARGE_SIZE / (1024 * 1024));
    
    // 主循环 - 执行多种 cudaMemcpy 操作
    for (int iter = 0; iter < ITERATIONS; iter++) {
        printf("\n迭代 %d/%d\n", iter+1, ITERATIONS);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float elapsed_time;
        
        // === 1. 基本 H2D 传输 ===
        printf("执行 主机到设备(H2D) 传输...\n");
        cudaEventRecord(start);
        hostToDevice(h_small, d_small, SMALL_SIZE);
        hostToDevice(h_medium, d_medium, MEDIUM_SIZE);
        hostToDevice(h_large, d_large, LARGE_SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("H2D 传输完成: %.2f ms\n", elapsed_time);
        
        // === 2. 执行一些核心操作 ===
        printf("执行 CUDA 核心操作...\n");
        cudaEventRecord(start);
        modifyData<<<smallGridSize, blockSize>>>(d_small, SMALL_SIZE);
        modifyData<<<mediumGridSize, blockSize>>>(d_medium, MEDIUM_SIZE);
        modifyData<<<largeGridSize, blockSize>>>(d_large, LARGE_SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("核心操作完成: %.2f ms\n", elapsed_time);
        
        // === 3. D2H 传输 ===
        printf("执行 设备到主机(D2H) 传输...\n");
        cudaEventRecord(start);
        deviceToHost(d_small, h_result_small, SMALL_SIZE);
        deviceToHost(d_medium, h_result_medium, MEDIUM_SIZE);
        deviceToHost(d_large, h_result_large, LARGE_SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("D2H 传输完成: %.2f ms\n", elapsed_time);
        
        // === 4. D2D 传输 ===
        printf("执行 设备到设备(D2D) 传输...\n");
        cudaEventRecord(start);
        deviceToDevice(d_small, d_temp, SMALL_SIZE);
        deviceToDevice(d_medium, d_temp, MEDIUM_SIZE);
        deviceToDevice(d_large, d_temp, LARGE_SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("D2D 传输完成: %.2f ms\n", elapsed_time);
        
        // === 5. 分段传输 ===
        printf("执行分段传输...\n");
        cudaEventRecord(start);
        chunkedTransfer(h_large, d_large, LARGE_SIZE, 5);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("分段传输完成: %.2f ms\n", elapsed_time);
        
        // === 6. 固定内存传输 ===
        printf("执行固定内存传输...\n");
        cudaEventRecord(start);
        hostToDevice(h_pinned_medium, d_medium, MEDIUM_SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("固定内存传输完成: %.2f ms\n", elapsed_time);
        
        // === 7. 异步传输 ===
        printf("执行异步传输...\n");
        cudaEventRecord(start);
        asyncTransfer(h_medium, d_medium, MEDIUM_SIZE, stream);
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("异步传输完成: %.2f ms\n", elapsed_time);
        
        // === 8. 混合传输 - 交替 H2D 和 D2H ===
        printf("执行混合传输...\n");
        cudaEventRecord(start);
        for (int i = 0; i < 5; i++) {
            size_t offset = i * SMALL_SIZE / 5;
            hostToDevice(h_small + offset / sizeof(float), 
                         d_small + offset / sizeof(float), 
                         SMALL_SIZE / 5);
            deviceToHost(d_medium + offset / sizeof(float),
                         h_result_medium + offset / sizeof(float),
                         MEDIUM_SIZE / 5);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("混合传输完成: %.2f ms\n", elapsed_time);
        
        // 清理事件
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // 每次迭代后暂停
        printf("等待 %d 秒继续下一次迭代...\n", SLEEP_SECONDS);
        sleep(SLEEP_SECONDS);
    }
    
    printf("\n所有迭代完成，清理资源...\n");
    
    // 清理资源
    free(h_small);
    free(h_medium);
    free(h_large);
    free(h_result_small);
    free(h_result_medium);
    free(h_result_large);
    CHECK_CUDA_ERROR(cudaFreeHost(h_pinned_medium));
    
    CHECK_CUDA_ERROR(cudaFree(d_small));
    CHECK_CUDA_ERROR(cudaFree(d_medium));
    CHECK_CUDA_ERROR(cudaFree(d_large));
    CHECK_CUDA_ERROR(cudaFree(d_temp));
    
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    printf("测试完成!\n");
    return 0;
}