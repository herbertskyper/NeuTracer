# GPUSnoop 监控进程与异常检测算法源代码分析

## 算法原理




## 简要概述
GPUSnoop 是一个基于eBPF技术的GPU性能监控和异常检测系统，通过多线程架构实现对CUDA应用的实时监控。系统采用生产者-消费者模式，其中eBPF程序作为内核空间的数据采集器，用户空间的监控线程作为数据处理器，协同完成GPU内存异常的检测与分析。

### 核心监控架构

系统通过三个独立的环形缓冲区分别处理不同类型的GPU事件。主要的环形缓冲区轮询线程 `ring_buffer_thread()` 负责从内核空间获取CUDA API调用数据，包括内核启动、内存分配/释放和内存拷贝事件。该线程采用非阻塞轮询机制，每100毫秒检查一次缓冲区状态：

```cpp
while (!exiting_ && !hasExceededProfilingLimit(duration, startTime)) {
    int err = ring_buffer__poll(ringBuffer, 100);
    int memleak_err = ring_buffer__poll(memleak_ringBuffer, 100);
    int memcpy_err = ring_buffer__poll(memcpy_ringBuffer, 100);
    
    if (memcpy_err > 0 || memleak_err > 0 || err > 0) {
        lastActivityTime_ = std::chrono::steady_clock::now();
    }
}
```

进程监控线程 `process_monitor_thread()` 采用分层检测策略，通过不同的时间间隔触发各种异常检测算法。该线程每5秒执行一次基础检查，包括已终止进程的清理和内存状态更新，同时根据预设的检测间隔分别触发内存不足检测、泄漏检测和碎片化分析。

### 内存泄漏检测算法

系统实现了基于拉普拉斯继承规则的内存泄漏检测算法。LeakDetector通过高水位标记机制识别内存分配模式的异常变化，当内存使用量超过历史最高值加上阈值时触发采样记录。泄漏概率计算采用贝叶斯统计方法：

```cpp
double LeakDetector::calculateLeakProbability(const LeakScore &score) {
    if (score.mallocs == 0) return 0.0;
    
    uint32_t unfreed = score.mallocs - score.frees;
    double probability = 1.0 - static_cast<double>(score.frees + 1) / (unfreed + 2);
    return std::max(0.0, std::min(1.0, probability));
}
```

该算法通过统计每个调用位置的分配次数、释放次数和内存增长率，计算出泄漏概率和内存增长趋势。当泄漏概率超过95%且内存增长率超过阈值时，系统会生成泄漏报告并记录调用堆栈信息。

### 内存碎片化检测算法

FragmentDetector实现了多维度的碎片化评估体系，通过外部碎片化率、内核不可用指数、分配模式和空间效率四个核心指标进行综合评分。外部碎片化率通过计算内存间隙与总地址空间的比例来衡量内存分割程度：

```cpp
double FragmentDetector::calculateFinalFragmentationScore(const FragmentationMetrics &metrics) {
    double score = 0.0;
    score += std::min(metrics.external_fragmentation_ratio * 2, 1.0) * 50.0;
    score += std::min(1.0, metrics.kernel_unusable_index / 10.0) * 15.0;
    
    double pattern_penalty = (metrics.small_allocation_ratio * 0.6) + 
                           (std::min(1.0, metrics.allocation_size_variance) * 0.4);
    score += pattern_penalty * 10.0;
    score += metrics.large_gap_ratio * 25.0;
    
    return std::min(100.0, std::max(0.0, score));
}
```

系统还集成了时间序列预测模型，通过TimeSeriesPredictor类实现基于多元线性回归的趋势预测。该模型通过滑动窗口机制捕捉历史数据模式，并采用梯度下降算法优化权重参数，预测未来的碎片化发展趋势。

### 实时异常监控

进程监控线程实现了分级的异常检测策略，通过调用 `detectCudaMemoryShortage()` 检测GPU内存不足情况，该函数通过CUDA运行时API获取设备内存使用情况，当内存使用率超过80%或检测到分配失败时触发警告。整个监控过程通过互斥锁保证线程安全，确保在高并发环境下数据的一致性和准确性。

这种设计使得GPUSnoop能够在运行时动态检测各种GPU内存异常，为CUDA应用的性能优化和故障诊断提供了强有力的技术支撑。