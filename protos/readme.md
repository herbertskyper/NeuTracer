# TracerService: 分布式追踪数据传输协议

## 概述

tracer_service.proto 定义了 NeuTracer 框架中用于跨系统传输性能追踪数据的 gRPC 协议。该协议支持多种系统资源监控数据的批量传输，包括函数调用、CPU、内存、网络、IO 和 GPU 使用情况，为 NEU-Trace 的分布式架构提供了高效可靠的通信基础。

## 核心组件

### 数据结构

协议定义了六种主要的跟踪数据类型，每种类型对应系统的不同维度：

1. **基础函数调用追踪 (TraceData)**
   - 记录函数调用的入口和出口事件
   - 包含函数名、时间戳、调用计数和平均执行时间

2. **CPU 追踪 (CPUTraceData)**
   - 监控进程在 CPU 上的运行情况
   - 记录 CPU 利用率、运行/等待时间和异常行为

3. **内存追踪 (MemoryTraceData)**
   - 跟踪内存分配和释放活动
   - 识别内存泄露和碎片化等异常

4. **网络追踪 (NetworkTraceData)**
   - 监控网络连接和数据传输
   - 统计流量和识别突发行为

5. **I/O 追踪 (IOTraceData)**
   - 记录磁盘读写操作
   - 监测 I/O 延迟和吞吐量

6. **GPU 追踪 (GPUTraceData)**
   - 简洁的 GPU 活动记录
   - 包含时间戳和堆栈信息

### 异常表示

每种资源类型都包含专门的异常表示结构：

```protobuf
message Anomaly {
  string description = 1;  // 异常描述
  double severity = 2;     // 严重程度(0-1)
}
```

这允许在传输过程中保留异常检测的结果，使客户端和服务器都能了解系统异常状态。

### 批量传输

```protobuf
message TraceBatch {
  repeated TraceData trace_data = 1;
  repeated GPUTraceData gpu_trace_data = 2;
  repeated CPUTraceData cpu_trace_data = 3;
  repeated MemoryTraceData memory_trace_data = 4;
  repeated NetworkTraceData network_trace_data = 5;
  repeated IOTraceData io_trace_data = 6;
}
```

批量传输机制允许客户端一次性发送多个跟踪记录，显著提高网络效率。

## 服务定义

协议定义了两个主要的 RPC 服务：

```protobuf
service TracerService {
  // 发送跟踪数据批次
  rpc SendTraceBatch(TraceBatch) returns (TraceResponse);
  
  // 获取服务状态
  rpc GetStatus(StatusRequest) returns (StatusResponse);
}
```

1. **SendTraceBatch**：发送批量追踪数据到服务器
2. **GetStatus**：查询服务器状态，包括是否活跃和已接收的追踪数量

## 实现与使用

协议通过 Protocol Buffers 编译器生成了 Python 客户端/服务端代码：

- tracer_service_pb2.py：包含消息类型定义
- tracer_service_pb2_grpc.py：包含客户端存根和服务实现接口

## 技术特点

1. **高效的二进制协议**：使用 Protocol Buffers 进行数据序列化，比 JSON 更高效
2. **版本兼容性**：通过字段编号确保向前兼容和向后兼容
3. **双向流式通信**：gRPC 支持双向流式通信，适合实时监控场景
4. **类型安全**：强类型定义确保数据完整性和一致性
5. **多语言支持**：可以生成多种语言的客户端和服务端代码

## 应用场景

- **分布式系统监控**：跨多台机器收集性能数据
- **中央化分析**：将分散的监控数据集中处理和分析
- **实时警报系统**：快速传输异常事件信息
- **历史数据收集**：将追踪数据发送到存储系统进行长期分析
- **高级可视化**：为前端可视化工具提供结构化数据
