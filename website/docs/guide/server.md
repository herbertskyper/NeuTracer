# 服务端

NeuTracer 服务器是一个高性能的跟踪数据收集、处理和分析系统，专为处理来自 NeuTracer 客户端的各种系统性能数据而设计。服务器采用 gRPC 通信协议，支持实时数据处理和 Prometheus 指标导出，提供了全方位的系统性能监控解决方案。该系统具备多维度性能数据收集能力，支持 CPU、内存、I/O、网络、函数调用和 GPU 活动的跟踪数据，事件处理管道支持批量数据处理。同时，服务端的 Prometheus 指标导出便于实时监控和告警，生成兼容 Perfetto/Chrome Tracing 的可视化文件。项目采用模块化设计易于添加新的数据处理器和分析功能。

服务器导出多种 Prometheus 指标便于实时监控和告警，包括函数调用次数与执行时间、CPU 利用率与上下文切换、内存分配释放和使用情况、网络流量连接和协议分布、I/O 操作数量延迟、服务器状态和性能指标。系统生成多种输出文件存储在 output 目录中，包括 Perfetto/Chrome Tracing 兼容的可视化数据文件、按资源类型分类的结构化 CSV 数据、检测到的异常记录和性能统计摘要。

系统具备良好的可扩展性，支持快速添加新的处理器和指标。添加新处理器只需创建继承自 BaseTraceProcessor 的新类，实现 process_traces() 方法，然后在 TracerServicer 中初始化和使用。添加新的 Prometheus 指标只需在 prometheus_metrics.py 中定义新指标，并在相应的处理器中更新这些指标。
```python
# 扩展示例：添加新处理器
class CustomTraceProcessor(BaseTraceProcessor):
    def __init__(self):
        self.custom_metrics = Counter('custom_events_total')
    
    def process_traces(self, traces):
        for trace in traces:
            self.custom_metrics.inc()
            self.handle_custom_logic(trace)

# Prometheus 指标定义示例
from prometheus_client import Counter, Histogram, Gauge

function_calls_total = Counter('function_calls_total', 'Total function calls', ['function_name'])
cpu_utilization = Gauge('cpu_utilization_percent', 'CPU utilization percentage', ['core_id'])
memory_allocated = Histogram('memory_allocated_bytes', 'Memory allocation size')
```
系统使用 gRPC 协议提供高性能的跨语言通信能力，通过 Prometheus 支持监控指标导出，采用模块化架构支持功能扩展。安全性考虑方面，默认配置使用不安全的 gRPC 连接，生产环境建议配置 TLS/SSL 证书启用安全连接，实现适当的认证机制，限制服务访问权限和网络暴露。

## 分布式追踪数据传输协议

[分布式追踪数据传输协议](./server/proto.md)

## 服务端数据处理框架
[服务端数据处理框架](./server/trace.md)