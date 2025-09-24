# CUDA API 异常与CUDA内存不足检测

NeuTracer通过追踪 CUDA API 的调用和返回结果，实现了对 CUDA API相关错误的检测。每当发生如 cudaMalloc 或 cudaFree 等内存操作时，代码会记录分配的地址、大小、返回值（ret），以及分配的时间戳等信息，并将这些数据存入进程的内存映射表。检测错误的关键在于：每次分配后，都会检查返回值 ret 是否等于 cudaSuccess，如果不是，则会通过 cudaGetErrorString(ret) 输出详细的错误信息，并在日志中进行警告提示。

内存不足检测算法通过分析系统内存使用情况和分配失败历史来判断是否存在内存不足问题，综合考虑当前内存使用率和历史分配失败情况。即使当前内存使用率不高，如果检测到近期有分配失败的情况，也会触发内存不足警告。对于内存使用情况的统计，我们会遍历维护的内存映射表，统计每个进程的内存分配情况，并计算当前使用的总内存量。同时，因为NeuTracer会自动清理已终止进程的内存记录，这样可以避免重复统计的问题。

```cpp
bool GPUSnoop::detectCudaMemoryShortage() {
    // 获取 GPU 硬件信息
    cudaGetDevice(&currentDevice);
    cudaMemGetInfo(&freeMem, &totalMem);
    
    // 分析分配失败和内存使用情况
    bool recentAllocationFailure = false;
    uint64_t use_mem = 0;
    
    for (const auto& [pid, memMap] : memory_map) {
        for (const auto& [addr, alloc] : memory_map[pid]) {
            if (alloc.ret != cudaSuccess) {
                recentAllocationFailure = true;
                continue;
            }
            use_mem += alloc.size;
        }
    }
    
    double memoryUsage = 100.0 * use_mem / totalMem;
    return (memoryUsage > MEMORY_THRESHOLD || recentAllocationFailure);
}
```