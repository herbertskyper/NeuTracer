# Python 栈回溯收集

本模块通过 eBPF 和 Python 联合实现高效的栈回溯（调用信息）采集与分析。该框架采用分层架构设计，包含 eBPF 采集层、Python 功能层、数据格式转换脚本和测试验证模块，实现了从内核态数据采集到用户态分析处理的完整链路。

eBPF 采集层负责底层数据收集，包含内核程序 pystack_time.bpf.c 用于采集栈帧和时间戳信息，用户态接口 pystack_time.c/h 定义了与 eBPF 程序的通信和数据结构， Makefile 用于编译构建。Python 功能层提供上层应用接口，core.py 作为核心模块调用 C 层接口获取和分析栈回溯数据，util.py 提供数据转换和辅助分析工具函数，neuron_error_models.py 和 weight_error_models.py 实现容错相关模型用于采集数据的进一步分析。

数据格式转换脚本模块支持多种输出格式，convert_to_json.py 将采集的调用信息转换为 JSON 格式便于存储和传输，convert_to_perfetto.py 转换为 Perfetto 性能分析工具支持的格式，perfetto_trace_pb2.py 提供 Perfetto 协议的 protobuf 定义和解析功能。对于测试验证模块，test_example_client.py 提供典型测试用例模拟实际采集场景，各种 test.py 文件针对不同功能模块进行单元测试，profiler_logs 目录存放采集到的原始和处理后的日志数据。

```python
# core.py 核心接口示例
class PyStackProfiler:
    def __init__(self):
        self.ebpf_interface = load_ebpf_program()
    
    def start_profiling(self, pid):
        """启动栈回溯采集"""
        return self.ebpf_interface.attach_to_process(pid)
    
    def get_stack_traces(self):
        """获取栈回溯数据"""
        raw_data = self.ebpf_interface.read_ring_buffer()
        return self.parse_stack_data(raw_data)
    
    def parse_stack_data(self, raw_data):
        """解析栈帧数据"""
        stack_traces = []
        for frame in raw_data:
            stack_traces.append({
                'timestamp': frame.timestamp,
                'pid': frame.pid,
                'function': frame.function_name,
                'filename': frame.filename,
                'line_number': frame.line_number
            })
        return stack_traces

# 数据转换示例
def convert_to_perfetto(stack_traces):
    """转换为 Perfetto 格式"""
    trace = perfetto_trace_pb2.Trace()
    for trace_data in stack_traces:
        event = trace.packet.add()
        event.timestamp = trace_data['timestamp']
        event.track_event.name = trace_data['function']
        event.track_event.categories.append("python")
    return trace.SerializeToString()
```
代码采用分层处理模式，eBPF 内核程序在内核态采集栈帧和时间戳信息，并通过 ring buffer 传递到用户态，用户态 C 接口读取 eBPF 数据并提供标准化接口供 Python 调用，Python 层调用 C 接口获取原始数据并进行解析分析和可视化处理，最终将处理结果导出为 JSON、Perfetto 等标准格式供外部工具使用。整个系统通过完善的测试用例确保各个模块的功能正确性和数据准确性。

```c
// pystack_time.bpf.c 关键结构
struct stack_event {
    __u32 pid;
    __u64 timestamp;
    __u64 user_stack_id;
    char comm[TASK_COMM_LEN];
    char filename[MAX_FILENAME_LEN];
    char function_name[MAX_FUNC_LEN];
    __u32 line_number;
};

// 用户态接口
int start_stack_tracing(int target_pid) {
    // 加载并附加 eBPF 程序
    struct bpf_object *obj = bpf_object__open_file("pystack_time.bpf.o", NULL);
    bpf_object__load(obj);
    
    // 附加到目标进程
    struct bpf_link *link = bpf_program__attach_uprobe(
        bpf_object__find_program_by_name(obj, "trace_python_call"),
        false, target_pid, "/usr/bin/python3", 0);
    
    return setup_ring_buffer(obj);
}
```