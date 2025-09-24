# Trace Processor: 服务端数据处理框架

## 概述

Trace Processor 是系统中负责接收、处理、分析和存储来自客户端的跟踪数据的服务端组件。它提供了一套灵活、可扩展的处理管道，能够处理多种系统资源的性能数据，包括函数调用、CPU 使用情况、内存分配、网络活动和 I/O 操作，并支持数据可视化、异常检测和指标导出。

## 核心架构

### 基础处理器 (BaseTraceProcessor)

`BaseTraceProcessor` 是所有处理器的抽象基类，提供了通用的数据处理框架：

```python
class BaseTraceProcessor:
    def __init__(self, trace_type: str, file_manager: TraceFileManager):
        self.trace_type = trace_type
        self.file_manager = file_manager
        self.total_traces = 0
```

主要功能：
- **事件处理**：处理批量接收的跟踪事件
- **文件管理**：将处理后的数据写入 JSON 和 CSV 文件
- **异常导出**：支持将检测到的异常单独导出
- **统计汇总**：生成数据摘要信息

### 专业处理器

系统包含五个专业处理器，每个针对一种资源类型：

1. **函数调用处理器 (FunctionTraceProcessor)**
   - 处理函数调用的进入和退出事件
   - 计算调用次数、持续时间和统计信息
   - 识别慢函数调用和可疑错误

2. **CPU 处理器 (CPUTraceProcessor)**
   - 跟踪进程的 CPU 利用率和运行时间
   - 分析 CPU 迁移和 NUMA 行为
   - 进程分类与 CPU 热点检测

3. **内存处理器 (MemTraceProcessor)**
   - 监控内存分配和释放
   - 内存泄漏和碎片化检测
   - 内存使用模式分类

4. **网络处理器 (NetworkTraceProcessor)**
   - 跟踪网络发送和接收活动
   - 连接管理和协议分析
   - 网络速率和异常流量检测

5. **I/O 处理器 (IOTraceProcessor)**
   - 监控磁盘读写操作
   - I/O 延迟和吞吐量分析
   - 设备使用情况和 I/O 行为分类

## 数据流程

1. **数据接收**：通过 gRPC 服务接收来自客户端的跟踪数据批次
2. **预处理**：将原始 protobuf 消息转换为内部数据结构
3. **处理与分析**：对数据进行分类、统计和异常检测
4. **指标导出**：更新 Prometheus 指标，用于实时监控
5. **数据存储**：以 JSON 和 CSV 格式保存数据，便于后续分析
6. **异常报告**：生成详细的异常报告和性能洞察

## 关键特性

### 1. 异常检测

每种资源处理器都实现了特定的异常检测算法：

- **函数处理器**：识别执行时间异常长的函数调用
- **CPU 处理器**：检测资源争用、CPU 热点和过度调度
- **内存处理器**：识别内存泄漏、碎片化和异常分配模式
- **网络处理器**：检测突发流量和异常连接行为
- **I/O 处理器**：识别高延迟操作和不平衡的 I/O 模式

### 2. 数据可视化支持

处理器生成符合 Perfetto/Chrome 跟踪格式的 JSON 数据：

```python
event = {
    "name": function_name,
    "ph": "B",  # 开始事件
    "ts": timestamp_ms,
    "readable_time": format_timestamp(timestamp_ms),
    "pid": pid,
    "tid": tid,
    "args": { ... }
}
```

### 3. 增量数据导出

定期将统计数据和异常导出到 CSV 文件，支持实时分析：

```python
def _export_incremental_stats(self) -> None:
    stats_headers = ['Timestamp', 'PID', 'Process', ...]
    stats_rows = []
    # ... 填充数据 ...
    self.export_stats_to_csv(stats_headers, stats_rows)
```

### 4. 资源分类

根据使用特征对进程进行自动分类：

- **CPU 进程类型**：COMPUTE_BOUND, IO_BOUND, BALANCED, BURSTY, PERIODIC, IDLE
- **内存使用模式**：STABLE, GROWING, FLUCTUATING, LEAKING
- **网络使用模式**：TX_INTENSIVE, RX_INTENSIVE, BALANCED, BURSTY, IDLE
- **I/O 进程类型**：READ_INTENSIVE, WRITE_INTENSIVE, BALANCED_IO, SYNC_HEAVY

### 5. Prometheus 集成

所有处理器都将关键指标导出到 Prometheus，便于实时监控：

```python
# 更新 CPU 利用率指标
CPU_UTILIZATION.labels(pid=str(pid), process=comm).set(avg_utilization / 100.0)

# 更新内存使用指标
MEM_USAGE.labels(pid=str(pid), process=comm).set(stats['current_memory'])
```

### 6. 效率优化

处理器采用多种优化措施，确保高效处理大量事件：

- **批处理**：一次处理多个事件
- **增量导出**：定期导出数据，避免内存压力
- **事件过滤**：忽略非活跃进程的事件
- **优化数据结构**：使用高效的数据结构和索引

## 功能

1. **性能分析**：深入了解应用程序在 CPU、内存、网络和 I/O 方面的行为
2. **异常检测**：自动识别系统中的性能异常和潜在问题
3. **资源优化**：根据使用模式指导资源分配和系统调优
4. **故障诊断**：分析性能瓶颈和系统故障的原因

## 技术亮点

- **多维度分析**：同时分析多种系统资源，提供全面的性能视图
- **可扩展架构**：基类设计允许轻松添加新的处理器类型
- **智能异常检测**：基于统计和模式识别的异常检测算法
- **丰富的数据导出**：支持多种格式，便于后续分析和可视化
- **实时监控**：通过 Prometheus 集成支持实时指标监控和告警