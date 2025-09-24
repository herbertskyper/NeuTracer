# IO 模块用户态部分
通过 I/O 模块 eBPF 程序高效收集 I/O 情况后，IoSnoop 对数据进行整合，跟踪 I/O 操作，收集读写延迟、吞吐量等性能指标，并简单提供实时异常检测功能。该模块可以按进程、设备分别收集多维度 I/O 性能指标，并识别高延迟、不平衡 I/O、设备切换频繁等异常行为。

核心数据结构包括 I/O 事件结构（bio_event：跟ebpf侧的数据结构保持一致）和 I/O 统计结构（IOStats）。I/O 统计结构提供更全面的性能指标，包含进程标识、累计读写字节数、读写操作次数、平均延迟等关键数据。

```c
struct IOStats {
    uint32_t pid{0};                         ///< 进程ID
    std::string comm;                        ///< 进程名称
    
    // 基础统计
    uint64_t total_bytes{0};                 ///< 总IO字节数
    uint64_t read_bytes{0};                  ///< 读取的总字节数
    uint64_t write_bytes{0};                 ///< 写入的总字节数
    uint64_t read_ops{0};                    ///< 读操作次数
    uint64_t write_ops{0};                   ///< 写操作次数
    uint64_t total_latency_us{0};            ///< 总IO延迟(微秒)
    
    // 计算指标
    double avg_latency_ms{0.0};              ///< 平均延迟(毫秒)
    double avg_read_latency_us{0.0};         ///< 平均读延迟(毫秒)
    double avg_write_latency_us{0.0};        ///< 平均写延迟(毫秒)
    uint64_t max_latency_us{0};              ///< 最大延迟(微秒)
    
    // 历史数据样本
    std::deque<uint64_t> latency_samples;    ///< 延迟历史样本
    std::deque<uint64_t> size_samples;       ///< IO大小历史样本
    
    // 设备访问历史
    std::map<std::string, uint64_t> device_access_count;  ///< 设备访问计数
};
```

`attach_bpf()` 附加 eBPF 程序到相关内核函数实现初始化，`start_trace()` 启动监控线程和 ring buffer 处理，`stop_trace()` 停止监控并清理资源，`ring_buffer_thread()` 后台线程循环处理 I/O 事件。`process_event()` 处理从 eBPF 接收的 I/O 事件，`update_io_stats()` 更新特定进程的 I/O 统计信息，`detect_io_process_type()` 根据读写模式确定进程 I/O 类型，`load_device_names()` 加载块设备名称信息。`report_io_anomalies()` 生成 I/O 报告。此外，代码每隔一段时间会自动清理相关的统计结构，避免程序结构体过大导致内存占用过高。 

此外，代码通过简单的规则，识别可能的异常，比如 I/O 高延迟、超大 I/O 传输、延迟尖峰等。每个异常都会记录日志，帮助用户快速定位问题。

```c
// 高延迟异常
if (e->delta_us > IO_latency_threshold_us_) {
    logger_.warn("[IO异常] PID {} ({}): 高延迟 {:.2f}ms (平均: {:.2f}ms)",
                pid, stats.comm, e->delta_us / 1000.0, stats.avg_latency_ms);
}
// 超大IO异常
if (e->bytes > IO_large_size_ * 1024 * 1024) {
    logger_.warn("[IO异常] PID {} ({}): 超大IO传输 {:.2f}MB",
                pid, stats.comm, e->bytes / (1024.0 * 1024.0));
}
// 延迟尖峰异常
else if (detect_io_latency_spike(pid, e)) {
    logger_.warn("[IO异常] PID {} ({}): 延迟尖峰 {:.2f}ms",
                pid, stats.comm, e->delta_us / 1000.0);
}
```