# GPU模块 用户态部分

GPUSnoop 是专门用于监控 CUDA 运行时 API 调用的核心组件，在 GPU 的 eBPF 程序收集到 GPU 内核相关数据并传到用户空间后，GPUSnoop 进行统一的数据处理以及整合，从而对 GPU 内存管理、内核启动和数据传输进行全方位跟踪。该模块不仅收集详细的 GPU 资源使用统计数据，还能实时分析内核调用模式和内存分配行为，具备 CUDA 内核启动监控、GPU 内存分配跟踪、数据传输分析、多进程支持、实时事件传输和调用栈捕获等核心能力。

数据结构设计涵盖了 GPU 监控的各个方面。CUDA 内核事件结构（gpukern_sample）记录内核启动的完整信息，包括进程 ID、进程名称、内核函数偏移地址、网格和块维度、CUDA 流 ID、内核参数和用户调用栈。内存事件结构（memleak_event）跟踪内存分配和释放操作，记录时间戳、设备内存地址、内存大小、进程 ID、返回值和事件类型。内存拷贝事件结构（cuda_memcpy）记录数据传输操作的详细信息，包括开始和结束时间、源和目标地址、传输字节数、进程 ID 和传输类型（H2D、D2H等）。

```cpp
struct gpukern_sample {
    int pid, ppid;                    // 进程 ID 和父进程 ID
    char comm[TASK_COMM_LEN];         // 进程名称
    uint64_t kern_func_off;           // 内核函数偏移地址
    int grid_x, grid_y, grid_z;       // 网格维度
    int block_x, block_y, block_z;    // 块维度
    uint64_t stream;                  // CUDA 流 ID
    uint64_t args[MAX_GPUKERN_ARGS];  // 内核参数
    size_t ustack_sz;                 // 用户栈深度
    stack_trace_t ustack;             // 用户调用栈
};

struct memleak_event {
    __u64 start;                      // 开始时间戳
    __u64 end;                        // 结束时间戳
    __u64 device_addr;                // 设备内存地址
    __u64 size;                       // 内存大小
    __u32 pid;                        // 进程 ID
    __s32 ret;                        // 返回值
    enum memleak_event_t event_type;  // 事件类型 (MALLOC/FREE)
};
```
核心方法涵盖初始化控制、事件处理和线程管理三个主要方面。初始化通过 `attach_bpf()` 加载 eBPF 程序并附加到 CUDA API 函数，`stop_trace()` 停止监控并清理所有资源，`setIdleTimeout()` 设置空闲超时时间优化资源使用。事件处理包括 `process_event()` 处理内核启动事件并解析符号信息，`process_memleak_event()` 处理内存分配/释放事件维护内存映射，`process_memcpy_event()` 处理数据传输事件计算带宽性能。线程管理通过 `ring_buffer_thread()` 主监控线程轮询三个环形缓冲区，`process_monitor_thread()` 进程监控线程定期清理已终止进程的数据。

```cpp
bool GPUSnoop::attach_bpf() {
    skel = gpuevent_snoop_bpf__open_and_load();
    if (!skel) return false;
    
    auto cuda_offsets = getCudaRuntimeOffsets();
    for (const auto& offset : cuda_offsets.launch_offsets) {
        auto link = bpf_program__attach_uprobe(skel->progs.handle_cuda_launch,
                                             false, env_.pid, 
                                             offset.lib_path.c_str(), offset.addr);
        if (link) launch_links.push_back(link);
    }
    
    setupRingBuffers();
    monitor_thread = std::thread(&GPUSnoop::ring_buffer_thread, this);
    return true;
}

int GPUSnoop::process_event(void *symUtils_ctx, void *data, size_t data_sz) {
    const struct gpukern_sample *e = (struct gpukern_sample *)data;
    SymUtils *symUtils = static_cast<SymUtils *>(symUtils_ctx);
    
    SymbolInfo symInfo = symUtils->getSymbolByAddr(e->kern_func_off, env_.args);
    {
        std::lock_guard<std::mutex> lock(kernel_func_stats_mutex_);
        kernel_func_stats_[e->kern_func_off].call_count++;
    }
    
    logger_.info("[GPU] KERNEL {} GRID ({},{},{}) BLOCK ({},{},{})",
                symInfo.name, e->grid_x, e->grid_y, e->grid_z,
                e->block_x, e->block_y, e->block_z);
    return 0;
}
```

GPUSnoop 采用三个独立的环形缓冲区处理不同类型的事件：内核启动事件、内存分配/释放事件和数据传输事件。这种多缓冲区设计分离了不同类型事件的处理，提高了并发性能和处理效率。主监控线程轮询所有环形缓冲区，确保所有类型的事件都能得到及时处理，同时通过合理的超时设置避免过度占用 CPU 资源。

```cpp
void GPUSnoop::ring_buffer_thread() {
    const int poll_timeout_ms = 100;
    while (running_) {
        int events = 0;
        events += ring_buffer__poll(ringBuffer, poll_timeout_ms);
        events += ring_buffer__poll(memleak_ringBuffer, poll_timeout_ms);
        events += ring_buffer__poll(memcpy_ringBuffer, poll_timeout_ms);
        
        if (events > 0) {
            last_activity_ = std::chrono::steady_clock::now();
        }
        checkIdleTimeout();
    }
}

// 数据传输带宽计算
double bandwidth_mbps = 0;
if (duration_ms > 0) {
    bandwidth_mbps = (e->count / (1024.0 * 1024.0)) / (duration_ms / 1000.0);
}
```
内核函数调用统计哈希表（kernel_func_stats）维护每个 CUDA 内核函数的调用统计信息，使用细粒度锁保护确保高并发场景下的数据一致性。多进程内存分配映射（memory_map）按进程 ID 组织，每个进程维护独立的内存分配记录列表。清理函数定期检查进程状态，移除已终止进程的所有相关数据，在清理前统计未释放的内存量为内存泄漏分析提供数据支持。

项目通过 eBPF uprobe 技术直接在用户空间函数入口/出口捕获事件实现低开销监控，通过 SymUtils 组件解析内核函数符号和调用栈信息，支持通过 RPC 将事件实时转发到 Grafana 等监控系统，使用互斥锁保护共享数据结构支持多线程并发访问，自动检测并清理已终止进程的数据防止内存泄漏。

该模块主要应用于 GPU 性能优化，可以识别频繁调用的内核函数和性能瓶颈，分析监控 GPU 内存分配模式和使用效率，分析 CPU-GPU 数据传输的带宽利用率，提供详细的 CUDA API 调用跟踪信息，从而实现适合生产环境的实时 GPU 资源监控。