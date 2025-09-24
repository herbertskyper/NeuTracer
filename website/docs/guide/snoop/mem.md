# 内存模块 用户态部分
在内存模块 eBPF 部分向用户态发送内存使用统计数据后，KmemSnoop 整合数据并能识别潜在的内存问题，如内存泄漏、碎片化和异常分配模式。核心数据结构包括内存事件结构（kmem_event:跟ebpf侧的数据结构保持一致）和内存统计结构（MemStats）。内存统计结构提供针对每个进程的完整内存统计和异常检测指标，包含基础信息、基础统计、内存操作历史、异常检测指标等数据。

```cpp
struct MemStats {
    uint32_t pid{0};            ///< 线程ID
    uint32_t tgid{0};           ///< 进程ID
    std::string comm;           ///< 进程名
    
    uint64_t total_allocs{0};   ///< 总分配次数
    uint64_t total_frees{0};    ///< 总释放次数
    uint64_t current_memory{0}; ///< 当前分配内存大小
    uint64_t peak_memory{0};    ///< 峰值内存使用
    
    // 内存操作历史
    std::deque<uint64_t> alloc_sizes;        ///< 分配大小历史
    std::deque<uint64_t> mem_usage_samples;  ///< 内存使用样本历史
    std::map<uint64_t, uint64_t> addr_to_size; ///< 地址到大小映射
    
    // 异常检测指标
    uint32_t churn_score{0};        ///< 内存周转分数
};
```
`attach_bpf()` 初始化 eBPF 程序并设置事件收集，将 eBPF 程序附加到系统调用点，`start_trace()` 启动监控线程和 ring buffer 处理，`stop_trace()` 停止监控并清理资源。`update_mem_stats()` 更新进程的内存统计信息，`formatMemorySize()` 将字节大小格式化为人类可读形式。`report_mem` 生成内存报告。
```cpp
void KmemSnoop::report_mem() {
    for (const auto& [pid, stats] : mem_stats_) {
        logger_.info("============ 内存报告 ============");
        logger_.info("[MEM]  分配次数: {}, 释放次数: {}", stats.total_allocs, stats.total_frees);
        logger_.info("[MEM]  当前内存: {} , 峰值内存: {} ", formatMemorySize(stats.current_memory), formatMemorySize(stats.peak_memory));
        logger_.info("===================================");
    }
}
```

为了增强易用性，代码提供可读的内存大小表示（KB/MB/GB）。
```c
// 格式化内存大小为可读形式
std::string KmemSnoop::formatMemorySize(uint64_t size) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    
    if (size < 1024) {
        ss << size << " B";
    } else if (size < 1024 * 1024) {
        ss << (double)size / 1024 << " KB";
    } else if (size < 1024 * 1024 * 1024) {
        ss << (double)size / (1024 * 1024) << " MB";
    } else {
        ss << (double)size / (1024 * 1024 * 1024) << " GB";
    }
    
    return ss.str();
}
```