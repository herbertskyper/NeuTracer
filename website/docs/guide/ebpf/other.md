# 其他模块

主要包括 系统调用模块，总线模块，以及nvlink模块。这些模块通过 eBPF 技术实现对系统调用、总线通信和 NVLink 连接的实时监控和分析，可以对前面几个模块收集到的信息进行进一步的补充。这些模块的用户态程序实现逻辑基本一致，`attach_bpf()` 初始化 eBPF 程序并设置事件收集，将 eBPF 程序附加到系统调用点，`start_trace()` 启动监控线程和 ring buffer 处理，`stop_trace()` 停止监控并清理资源。`process_event()` 处理从 eBPF 接收的事件，`ring_buffer_thread()` 后台线程循环处理事件。因此不在赘述，下面给出各个模块ebpf部分的说明。

## 系统调用追踪

syscall_snoop.bpf.c 用于实时追踪和分析 Linux 内核中的系统调用活动。该程序能够捕获每个系统调用的进入与退出事件，详细记录调用进程的标识、系统调用号、参数、返回值、时间戳。该程序挂载到两个核心的原始系统调用跟踪点：raw_syscalls/sys_enter 用于捕获系统调用入口，raw_syscalls/sys_exit 用于捕获系统调用返回。程序采用 syscall_event 结构体向用户空间传递事件信息，内容包括线程ID、进程ID、系统调用号、唯一调用ID、纳秒级时间戳、参数、返回值、事件类型和进程名称。
```cpp
enum event_type {
    EVENT_SYSCALL_ENTER = 0,
    EVENT_SYSCALL_EXIT = 1,
};

struct syscall_event {
    enum event_type type;
    u32 pid;
    u32 tgid;
    u32 syscall_id;
    u64 call_id;
    char comm[TASK_COMM_LEN];
    u64 timestamp;
    u64 args[6];    // 参数，enter时有效
    u64 ret_val;    // 返回值，exit时有效
};

```
事件通过环形缓冲区（events）高效传递到用户空间。程序内部维护调用ID映射（call_id_map）以唯一标识每次系统调用，支持多进程并发追踪。进程过滤映射（snoop_proc）和系统调用过滤映射（traced_syscalls）可灵活配置追踪范围。
```cpp
struct call_id_key_t {
    u32 pid;
    u32 syscall_id;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, struct call_id_key_t);
    __type(value, u64); // call_id
} call_id_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 512);
    __type(key, u32);
    __type(value, u32);
} traced_syscalls SEC(".maps");
```

系统调用进入追踪流程：首先检查系统调用号和进程是否在追踪范围内，分配并递增唯一调用ID，采集参数、时间戳和进程信息，最后通过环形缓冲区提交进入事件。

系统调用退出追踪流程：同样进行过滤，查找对应的调用ID，采集返回值、时间戳和进程信息，提交退出事件，确保每次系统调用的完整闭环追踪。该机制为系统调用行为分析、异常检测和性能监控提供了坚实基础。

## 总线追踪

pcie_snoop.bpf.c 用于追踪和分析 Linux 内核中的 PCIe 配置空间访问与 DMA Fence 同步事件。该程序能够捕获每一次 PCIe 配置读写操作的详细参数、进程上下文和时间戳，同时对 DMA Fence 生命周期事件进行全流程监控。

本模块挂载于 PCIe 配置空间访问的关键内核探针（kprobe/kretprobe）和 DMA Fence 相关 tracepoint。PCIe 部分支持对 `pci_bus_read_config_{byte,word,dword}` 及 `pci_bus_write_config_{byte,word,dword}` 的入口和返回进行追踪，完整记录总线号、设备号、偏移、访问类型、数据值等参数。DMA Fence 部分则追踪 `dma_fence_destroy、dma_fence_enable_signal、dma_fence_signaled、dma_fence_emit、dma_fence_wait_start、dma_fence_wait_end` 等事件，记录同步对象的 context、seqno、进程信息及等待时长。
```cpp
struct pcie_event {
    enum event_type type;
    enum pcie_op_type op_type;
    u32 pid;
    u32 tgid;
    char comm[TASK_COMM_LEN];
    u64 timestamp;
    
    // PCIe 访问参数
    u32 bus;
    u32 devfn;
    u32 offset;
    u32 size;
    u32 value;  // 写入值或读取结果
};

struct dma_fence_event {
    enum dma_fence_event_type type;
    u32 pid;
    u32 tgid;
    char comm[TASK_COMM_LEN];
    u64 timestamp;
    
    // DMA Fence 信息
    u64 context;
    u32 seqno;
    char driver_name[32];
    char timeline_name[32];
    u64 duration_ns;  // 仅用于 wait_end 事件
};
```


事件通过环形缓冲区（events）高效传递到用户空间。进程过滤映射（snoop_proc）可灵活配置追踪目标，DMA Fence 等待时长通过 fence_wait_start 哈希映射实现精准统计。
```cpp
struct fence_wait_key {
    u32 pid;
    u32 tgid;
    u32 context;
    u32 seqno;
};

// 修改映射定义
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, struct fence_wait_key);  // 使用组合键
    __type(value, u64); // 时间戳
} fence_wait_start SEC(".maps");
```

PCIe 访问追踪流程：在每次配置空间读写入口，采集访问参数、进程上下文和时间戳，提交进入事件；在返回点采集操作结果，提交退出事件，确保 PCIe 配置访问的全流程可观测。
```cpp
// 通用辅助函数: 用于处理 PCI 读取操作的入口点
static inline void handle_pci_bus_read_enter(struct pcie_event *event, 
                                            struct pci_bus *bus,
                                            unsigned int devfn, 
                                            int where, 
                                            int size,
                                            enum pcie_op_type op_type) {
    u8 bus_num = 0;
    if (bus) {
        bus_num = BPF_CORE_READ(bus, number);
    }
    
    event->op_type = op_type;
    event->bus = bus_num;
    event->devfn = devfn;
    event->offset = where;
    event->size = size;
    event->value = 0;  // 读取时入口值未知
}

// 通用辅助函数: 用于处理 PCI 写入操作的入口点
static inline void handle_pci_bus_write_enter(struct pcie_event *event, 
                                             struct pci_bus *bus,
                                             unsigned int devfn, 
                                             int where, 
                                             int size,
                                             u32 val,
                                             enum pcie_op_type op_type) {
    u8 bus_num = 0;
    if (bus) {
        bus_num = BPF_CORE_READ(bus, number);
    }
    
    event->op_type = op_type;
    event->bus = bus_num;
    event->devfn = devfn;
    event->offset = where;
    event->size = size;
    event->value = val;  // 写入的值
}
```

DMA Fence 追踪流程：每次 Fence 相关事件发生时，采集同步对象标识、进程信息和时间戳，wait_start/wait_end 事件映射实现等待时长统计。
```cpp
SEC("tracepoint/dma_fence/dma_fence_signaled")
int trace_dma_fence_signaled(struct trace_event_raw_dma_fence *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct dma_fence_event *event = create_dma_fence_event(DMA_FENCE_SIGNALED, pid, tgid);
    if (!event)
        return 0;

    event->context = ctx->context;
    event->seqno = ctx->seqno;
    bpf_ringbuf_submit(event, 0);
    return 0;
}

```
## NVLink 追踪

nvlink_snoop.bpf.c 用于实时追踪和分析 Linux 内核中 NVLink 相关的内存操作函数调用。该程序能够捕获每一次 NVLink 设备上的字符串拷贝（strcpy）、内存拷贝（memcpy）和内存填充（memset）操作的详细信息，包括目标地址、源地址（或填充值）、操作大小、返回值、进程上下文和时间戳。
```cpp
struct nvlink_event {
    enum event_type type;
    enum nvlink_func_id func_id;
    u32 pid;
    u32 tgid;
    char comm[TASK_COMM_LEN];
    u64 timestamp;
    
    // 函数参数
    u64 dst_addr;     // 目标地址
    u64 src_addr;     // 源地址 (对于memset是值)
    u64 size;         // 大小/长度
    
    // 返回值 (对于strcpy和memcpy是目标指针，对于memset是目标指针)
    u64 ret_val;
};
```

本模块通过 kprobe/kretprobe 挂载于 `nvlink_strcpy、nvlink_memcpy` 和 `nvlink_memset`的入口和返回点，完整记录每次内存操作的参数和结果。
```cpp
// nvlink_strcpy 追踪
SEC("kprobe/nvlink_strcpy")
int BPF_KPROBE(trace_nvlink_strcpy_enter, char *dst, const char *src) {
    bpf_printk("enter the strcpy");
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct nvlink_event *event = create_nvlink_event(EVENT_NVLINK_ENTER, 
                                                NVLINK_FUNC_STRCPY, pid, tgid);
    if (!event)
        return 0;

    event->dst_addr = (u64)dst;
    event->src_addr = (u64)src;
    // 尝试获取字符串长度 (受限于BPF安全检查可能无法完全获取)
    event->size = 0;
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}
```

NVLink 操作追踪流程：每次内存操作函数被调用时，采集操作参数、进程信息和时间戳，提交进入事件；函数返回时，采集返回值并提交退出事件，实现对 NVLink 关键内存操作的全流程可观测，为高性能计算、GPU 互联调优和链路异常分析提供坚实基础。

由于服务器上的GPU走的是PCIe总线，所以NVLink的追踪模块目前并未启用。