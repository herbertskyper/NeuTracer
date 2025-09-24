# CPU 用户态代码
通过 CPU 模块 eBPF 程序高效收集 CPU 使用情况后，CPUSnoop可以提供基本的 CPU 使用统计，还能检测异常行为模式。该模块具备实时 CPU 活动监控能力，跟踪进程的 CPU 运行时间、空闲时间和利用率，检测并记录进程在 CPU 核心间的迁移情况，识别资源争用、不合理调度、利用率突增等异常，分析跨 NUMA 节点迁移对性能的影响。

核心数据结构包括 CPU 事件结构（cpu_event: 跟ebpf侧的数据结构保持一致）和 CPU 统计结构（CPUStats）。CPU 统计结构提供更全面的性能指标，包含进程和 CPU 标识、运行和等待时间、利用率、迁移次数、NUMA 迁移计数、热点 CPU 信息、热点占比等关键数据。

```c
struct CPUStats {
    uint64_t total_oncpu_time{0};     ///< 总计CPU运行时间 (微秒)
    uint64_t total_offcpu_time{0};    ///< 总计CPU空闲时间 (微秒)
    uint64_t call_count{0};           ///< 记录次数
    double avg_utilization{0.0};      ///< 平均利用率 (0-100%)
    
    // 异常检测指标
    uint32_t migrations_count{0};     ///< CPU迁移总次数
    uint32_t numa_migrations{0};      ///< 跨NUMA节点迁移次数

    uint32_t hotspot_cpu{0};          ///< 最常使用的CPU核心ID
    double hotspot_percentage{0.0};   ///< 在热点CPU上执行的时间百分比
    // 记录不同CPU核心的时间分布
    std::unordered_map<uint32_t, uint64_t> cpu_distribution;
    
    // CPU迁移记录
    std::vector<CPUMigration> migrations;
    
    // 利用率历史样本
    std::deque<uint64_t> utilization_samples;
};
```
 
`init_bpf()` 初始化 eBPF 程序并设置事件收集，`attach_bpf()` 将 eBPF 程序附加到系统调用点，`start_trace()` 启动监控线程和 ring buffer 处理，`stop_trace()` 停止监控并清理资源。`process_event()` 处理从 eBPF 收到的 CPU 事件，`update_cpu_stats()` 更新特定进程的 CPU 统计信息，`update_hotspot_cpu()` 分析和更新 CPU 热点信息。`report_cpu` 生成 CPU 报告，`is_cross_numa()` 判断两个 CPU 是否位于不同的 NUMA 节点。

此外，代码每隔一段时间会自动清理相关的统计结构，避免程序结构体过大导致内存占用过高。 
```cpp
    // 初始化窗口开始时间
    if (stats.window_start.time_since_epoch().count() == 0) {
        stats.window_start = now;
    }
    
    // 检查是否需要重置窗口
    if (now - stats.window_start >= CLEAN_TIME_MIN * std::chrono::minutes(1)) {
        double avg_utilization = (stats.call_count > 0) ? stats.avg_utilization : 0.0;
        
        logger_.info("[CPU窗口] PID {} : {}分钟窗口重置 - 调用:{} 迁移:{} 平均利用率:{:.2f}% 热点CPU:{}",
                    pid, CLEAN_TIME_MIN,
                    stats.call_count, stats.migrations_count,
                    avg_utilization, stats.hotspot_cpu);
        
        // 重置窗口统计
        stats.window_start = now;
        //...
        // 清空容器
        stats.utilization_samples.clear();
        //....
    }
```
此外，代码通过简单的规则，识别可能的异常，比如 CPU 热点过高、NUMA 迁移次数异常、CPU 利用率突增等。每个异常都会记录日志，帮助用户快速定位问题。

```c
    if (stats.hotspot_percentage > CPU_HOTSPOT_THRESHOLD && stats.total_oncpu_time > 100000000000 && should_warn) { // 100秒以上的CPU时间) 
        logger_.warn("[CPU异常] PID {} ({}): CPU热点过高 - {:.1f}% 时间在核心{}上运行",
                    pid, pid_to_comm(pid), stats.hotspot_percentage * 100, stats.hotspot_cpu);
    }

    if(stats.numa_migrations >= CPU_NUMA_THRESHOLD && should_warn) {
                    logger_.warn("[CPU异常] PID {} ({}): 跨NUMA节点迁移:{}次",
                                    pid, pid_to_comm(pid), stats.numa_migrations);
    }

     if (e->utilization > avg * 2 && e->utilization / 100 > CPU_UTIL_THREHOLD && should_warn) { // 高于平均值2倍且超过50%
            logger_.warn("[CPU异常] PID {} ({}): 检测到突发型CPU使用 - 当前利用率{:.1f}% (平均{:.1f}%)",
                        pid, pid_to_comm(pid), e->utilization / 100.0, avg / 100.0);
    }

    if (switch_rate >  CPU_CONTEXT_CHANGE && should_warn) { 
            logger_.warn("[CPU异常] PID {} ({}): 上下文切换频率异常高 - {:.1f}% ({}次/{}次采样)",
                        pid, pid_to_comm(pid), switch_rate * 100, 
                        stats.migrations_count, stats.call_count);
    }

    if (stats.call_count > 20 && stats.avg_utilization < CPU_UTIL_LOW_THRESHOLD && stats.total_oncpu_time > 500000 && should_warn) {
            logger_.warn("[CPU异常] PID {} ({}): CPU利用率异常低 - 平均{:.2f}% (可能处于饥饿状态)",
                        pid, pid_to_comm(pid), stats.avg_utilization);
    }
```