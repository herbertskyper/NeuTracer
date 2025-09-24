# 内存模块 eBPF 部分

`kmem_snoop.bpf.c` 用于跟踪和分析 Linux 内核中的内存分配和释放操作。该程序能够捕获每个内存分配/释放事件的详细信息，包括大小、地址、调用堆栈和时间戳，为内存使用分析和泄漏检测提供关键数据。该程序可以监控内核内存分配/释放函数如kmalloc、kfree、kmem_cache_alloc等，记录每个分配的大小、地址和时间戳。

该程序挂载到四个关键的内核内存管理跟踪点来实现完整的内存操作监控。`kmem/kmalloc`跟踪通用内存分配，`kmem/kfree`跟踪通用内存释放，`kmem/kmem_cache_alloc`跟踪内存缓存分配，`kmem/kmem_cache_free`跟踪内存缓存释放。

程序采用kmem_event向用户空间传递完整的内存事件信息，包含线程ID、进程ID、内存大小、内存地址、纳秒级时间戳、调用堆栈ID、事件类型和进程名称。

```c
struct kmem_event {
    __u32 pid;                  // 线程 ID
    __u32 tgid;                 // 进程 ID
    __u64 size;                 // 内存大小
    __u64 addr;                 // 内存地址
    __u64 timestamp_ns;         // 时间戳（纳秒）
    __u32 stack_id;             // 调用堆栈 ID
    __u32 event_type;           // 0=分配, 1=释放
    char comm[TASK_COMM_LEN];   // 进程名称
};
```
程序使用事件环形缓冲区（mem_events）向用户空间传输事件数据。临时分配大小映射（sizes）临时存储分配大小，分配记录映射（allocs）存储最近活跃的内存分配，调用栈跟踪映射（stack_traces）存储内存分配的完整调用堆栈，由于I/O事件触发主要由 pid = 0的 swapper 进程触发，因此不设置进程过滤。
```c
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 256);
} mem_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u64);       // pid_tgid
    __type(value, u64);     // size
} sizes SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);  
    __type(key, u64);       // address
    __type(value, struct alloc_info);
} allocs SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(max_entries, 1024);
    __type(key, u32);       // stack_id
    __type(value, u64[PERF_MAX_STACK_DEPTH]);
} stack_traces SEC(".maps");
```

内存分配跟踪流程：首先获取进程信息和分配细节、过滤无效分配、记录分配信息并获取调用栈、更新分配记录映射和统计信息，最后向环形缓冲区提交事件。

内存释放跟踪流程：获取释放地址和进程信息、查找匹配的分配记录、创建并提交释放事件，最后清理分配记录以防止内存泄漏。
