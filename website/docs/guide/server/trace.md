# 服务端数据处理框架

Trace Processor 是系统中负责接收、处理来自客户端的跟踪数据的服务端组件。它提供了一套灵活、可扩展的处理管道，能够处理多种系统资源的性能数据，包括函数调用、CPU 使用情况、内存分配、网络活动和 I/O 操作，并支持数据可视化、异常检测和指标导出。系统采用基于 BaseTraceProcessor 抽象基类的架构，提供通用的数据处理框架，包括事件处理、文件管理和统计汇总功能。

系统包含文件处理器、活动监视器等辅助工具。文件处理器负责跟踪数据的存储和管理，确保数据的高效写入和读取。活动监视器用于跟踪服务的运行状态，处理超时和异常情况，确保服务的稳定性和可靠性。

文件处理器
```py
class TraceFileManager:
    def __init__(self):
        """初始化跟踪文件管理器"""
        # 创建日志目录
        self.json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "json")
        os.makedirs(self.json_dir, exist_ok=True)
        # 创建CSV目录
        self.csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "csv")
        os.makedirs(self.csv_dir, exist_ok=True)
        
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.initialize_stack_message_file()
```
    活动监视器
```py
class ActivityMonitor:
    def __init__(self, inactivity_timeout: int, shutdown_event: Event, last_activity_time_getter: Callable[[], float]):
        """
        初始化活动监控器
        - inactivity_timeout: 不活动超时时间（秒）
        - shutdown_event: 关机事件，在超时时设置
        - last_activity_time_getter: 获取最后活动时间的回调函数
        """
        self.inactivity_timeout = inactivity_timeout
        self.shutdown_event = shutdown_event
        self.get_last_activity_time = last_activity_time_getter
        self.monitor_thread: Optional[threading.Thread] = None
        self.active = True
```



系统包含五个处理器，每个针对一种资源类型。FunctionTraceProcessor 处理函数调用的进入和退出事件，计算调用次数、持续时间和统计信息，识别慢函数调用。CPUTraceProcessor 跟踪进程的 CPU 利用率和运行时间，分析 CPU 迁移和 NUMA 行为。MemTraceProcessor 监控内存分配和释放。NetworkTraceProcessor 跟踪网络发送和接收活动，进行连接管理和协议分析，检测网络速率和异常流量。IOTraceProcessor 监控磁盘读写操作，分析 I/O 延迟和吞吐量。

```python
class BaseTraceProcessor:
    def __init__(self, trace_type: str, file_manager: TraceFileManager):
        self.trace_type = trace_type
        self.file_manager = file_manager
        self.total_traces = 0
    
    def process_traces(self, traces: List[Any], is_last_batch: bool = False) -> int:

        raise NotImplementedError("必须在子类中实现此方法")

class FunctionTraceProcessor(BaseTraceProcessor):
    def process_single_trace(self, trace):
    # 这里只给出实现逻辑
    # 统计批次大小，更新指标
    # 按时间戳排序
    # 遍历每个跟踪事件
    for trace in sorted_traces:
        # 获取跟踪事件的相关信息
        if is_entry:
            # 处理函数调用开始
            event = {
                "ts": timestamp_ms,
                "pid": pid,
                "tid": tid,
                "name": function_name,
                "ph": "B",
                "args": {"cookie": cookie}
            }
            events.append(event)
            # 更新活跃函数计数

    # 批量更新相关指标指标
    # 写入事件到JSON和CSV
    return len(traces)

```

数据流程采用六步处理管道：
1. 通过 gRPC 服务接收来自客户端的跟踪数据批次，
2. 将原始 protobuf 消息转换为内部数据结构，进行预处理，
3. 使用对应的处理器对数据进行分类、统计和异常检测，
4. 更新 Prometheus 指标用于实时监控，
5. 以 JSON 和 CSV 格式保存数据便于后续分析，

```python
# 数据可视化支持 - Perfetto/Chrome 跟踪格式
event = {
    "name": function_name,
    "ph": "B",  # 开始事件
    "ts": timestamp_ms,
    "readable_time": format_timestamp(timestamp_ms),
    "pid": pid,
    "tid": tid,
    "args": {"duration": duration_ms, "cpu_id": cpu_id}
}

# Prometheus 集成
CPU_UTILIZATION.labels(pid=str(pid), process=comm).set(avg_utilization / 100.0)
MEM_USAGE.labels(pid=str(pid), process=comm).set(stats['current_memory'])
FUNCTION_CALLS.labels(function=func_name).inc()
```