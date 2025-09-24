# 分布式追踪数据传输协议

tracer_service.proto 定义了 NeuTracer 框架中用于跨系统传输性能追踪数据的 gRPC 协议。该协议支持多种系统资源监控数据的批量传输，包括函数调用、CPU、内存、网络、IO 和 GPU 使用情况。协议采用 Protocol Buffers 进行数据序列化，比 JSON 更高效，通过字段编号确保版本兼容性，支持双向通信，适合实时监控场景，可以生成多种语言的客户端和服务端代码。

协议定义了六种主要的跟踪数据类型，每种类型对应系统的不同维度。TraceData 记录函数调用的入口和出口事件，包含函数名、时间戳、调用计数和平均执行时间。CPUTraceData 监控进程在 CPU 上的运行情况，记录 CPU 利用率、运行/等待时间。MemoryTraceData 跟踪内存分配和释放活动。NetworkTraceData 监控网络连接和数据传输，统计流量和识别突发行为。IOTraceData 记录磁盘读写操作，监测 I/O 延迟和吞吐量。GPUTraceData 提供简洁的 GPU 活动记录，包含时间戳和堆栈信息。

```protobuf
message TraceBatch {
  repeated TraceData trace_data = 1;
  repeated GPUTraceData gpu_trace_data = 2;
  repeated CPUTraceData cpu_trace_data = 3;
  repeated MemoryTraceData memory_trace_data = 4;
  repeated NetworkTraceData network_trace_data = 5;
  repeated IOTraceData io_trace_data = 6;
}

service TracerService {
  rpc SendTraceBatch(TraceBatch) returns (TraceResponse);
  rpc GetStatus(StatusRequest) returns (StatusResponse);
}
```
协议定义了两个主要的 RPC 服务：SendTraceBatch 用于发送批量追踪数据到服务器，GetStatus 用于查询服务器状态，包括是否活跃和已接收的追踪数量。

协议通过 Protocol Buffers 编译器生成了 Python 客户端/服务端代码，tracer_service_pb2.py 包含消息类型定义，tracer_service_pb2_grpc.py 包含客户端存根和服务实现接口。该协议主要应用于分布式系统监控跨多台机器收集性能数据，中央节点将分散的监控数据集中处理和分析，为前端可视化工具提供结构化数据。
这是python端与协议对应的数据结构TracerServicer的定义：
```python
# TracerServicer 核心实现
class TracerServicer(trace_pb2_grpc.TraceServiceServicer):
    def __init__(self):
        self.processors = {
            'function': FunctionTraceProcessor(),
            'cpu': CPUTraceProcessor(),
            'memory': MemTraceProcessor(),
            'network': NetworkTraceProcessor(),
            'io': IOTraceProcessor(),
            'gpu': GPUTraceProcessor()
        }
    
    def SendTraces(self, request, context):
        trace_type = self.detect_trace_type(request)
        processor = self.processors[trace_type]
        processor.process_traces(request.traces)
        return trace_pb2.TraceResponse(success=True)
```