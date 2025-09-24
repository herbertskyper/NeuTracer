# 函数模块 eBPF 程序

`func_snoop.bpf.c` 用于实现用户空间函数调用的精确跟踪和分析。该程序通过 uprobe/uretprobe 机制挂钩函数入口和出口点，捕获函数调用事件，并提供详细的性能分析数据。它可以具备精确捕获函数入口和退出事件、保存线程ID和进程ID等调用上下文信息，支持可执行文件和库中函数的通用挂钩。

程序定义了两个关键的BPF挂钩点来实现完整的函数调用跟踪。`generic_entry` 作为函数入口点挂钩（uprobe），在目标函数被调用时执行；`generic_exit` 作为函数退出点挂钩（uretprobe），在目标函数返回时执行。程序采用了func_trace_event作为记录和传递函数调用事件的核心，包含事件类型（EVENT_ENTRY或EVENT_EXIT）、唯一标识符cookie、线程ID、进程ID、函数名称、参数以及返回值等关键信息。

```c
struct func_trace_event {
    enum event_type type;
    u64 cookie;
    u32 pid;    // 线程 ID
    u32 tgid;   // 进程 ID
    u64 timestamp; // 时间戳
    char name[64];
    u64 args[6];    // 新增：最多6个参数
    u64 ret_val;    // 新增：返回值，仅在EXIT时有效
};
```
程序使用两个关键的BPF映射进行数据管理。函数名称映射（func_names）是一个哈希表，存储cookie ID到函数名称的映射，最大条目数为1024，确保能够支持大量函数的同时跟踪。事件环形缓冲区（events）负责向用户空间传输事件数据，缓冲区大小为256KB，在高频函数调用场景下仍能保证数据传输的可靠性。（这些数据结构也可以通过宏定义调整最大容量）
```c
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 3);
    __type(key, u32);
    __type(value, u32);
} snoop_proc SEC(".maps");

// 存储函数名称和ID的映射
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u64);        // cookie ID
    __type(value, char[64]); // 函数名称
} func_names SEC(".maps");

// 定义 ring buffer
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");
```
函数入口处理流程包括获取cookie ID和进程线程ID、分配环形缓冲区事件、查找并设置函数名称，最后提交事件到环形缓冲区。函数退出处理与入口处理流程类似，主要差异在于事件类型设置为EVENT_EXIT。这种设计确保了每个函数调用都能生成配对的入口和退出事件，为计算函数执行时间和分析调用关系提供了基础数据。整个跟踪过程采用低开销设计，最大程度减少对目标程序性能的影响。
```c
// 通用入口点处理函数
SEC("uprobe")
int generic_entry(struct pt_regs *ctx) {
    u64 cookie = bpf_get_attach_cookie(ctx);
    
    // 获取进程和线程 ID
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;       // 低 32 位为线程 ID
    u32 tgid = (u32)(pid_tgid >> 32); // 高 32 位为进程 ID

    if(bpf_map_lookup_elem(&snoop_proc, &pid) == NULL && bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }
    
    // 分配 ring buffer 事件
    struct func_trace_event *event = bpf_ringbuf_reserve(&events, sizeof(struct func_trace_event), 0);
    if (!event)
        return 0;

    // 设置事件类型、cookie 和进程信息
    event->type = EVENT_ENTRY;
    event->cookie = cookie;
    event->pid = pid;
    event->tgid = tgid;
    event->timestamp = bpf_ktime_get_ns(); // 获取时间戳
    
    // 获取函数名称
    char *func_name = bpf_map_lookup_elem(&func_names, &cookie);
    if (func_name) {
        // 复制函数名称到事件
        bpf_probe_read_str(event->name, sizeof(event->name), func_name);
    } else {
        // 生成默认函数名
        char unnamed[32];
        __builtin_memset(unnamed, 0, sizeof(unnamed));
        bpf_probe_read_str(event->name, sizeof(event->name), unnamed);
    }
    event->ret_val = 0; // 初始化返回值为0
    event->args[0] = PT_REGS_PARM1(ctx);
    event->args[1] = PT_REGS_PARM2(ctx);
    event->args[2] = PT_REGS_PARM3(ctx);
    event->args[3] = PT_REGS_PARM4(ctx);
    event->args[4] = PT_REGS_PARM5(ctx);
    event->args[5] = PT_REGS_PARM6(ctx); // 处理最多6个参数
    
    // 提交事件到 ring buffer
    bpf_ringbuf_submit(event, 0);
    return 0;
}

// 通用退出点处理函数
SEC("uretprobe")
int generic_exit(struct pt_regs *ctx) {
    //处理逻辑类似
}
```

