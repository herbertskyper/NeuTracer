# GPU eBPF 模块
`gpuevent_snoop.bpf.c` 是用于监控 CUDA 运行时 API 调用的 eBPF 程序，实现对 GPU 内存管理、内核启动和数据传输的全方位跟踪。该程序通过用户空间探针（uprobe）技术，以极低开销捕获 CUDA API 调用事件，提供细粒度的 GPU 资源使用情况分析。模块具备 CUDA 内核启动监控能力，能够捕获内核函数地址、网格/块配置、参数和调用栈信息，同时跟踪 GPU 内存分配和释放操作，监控 `cudaMalloc` 和 `cudaFree` 调用以检测内存泄漏，分析 `cudaMemcpy` 操作计算传输带宽和方向，支持多进程同时监控，通过环形缓冲区高效传输事件到用户空间，并能识别失败的 CUDA API 调用和异常模式。

程序挂载到多个关键的 CUDA 运行时 API 来实现全面的 GPU 活动监控。内核启动探针通过 `handle_cuda_launch` 捕获内核启动事件，内存分配/释放探针通过 `cudaMalloc` 和 `cudaFree` 的 uprobe/uretprobe 组合监控内存管理操作，数据传输探针通过 `cudaMemcpy` 的跟踪数据拷贝操作。

数据结构设计采用三个核心结构体来管理不同类型的 GPU 事件。CUDA 内核事件结构（gpukern_sample）记录内核启动的完整信息，包括进程信息、内核函数偏移地址、网格和块维度、CUDA 流 ID、内核参数数组和用户调用栈。内存事件结构（memleak_event）跟踪内存分配和释放操作，包含时间戳、设备内存地址、内存大小、进程 ID、返回值和事件类型。内存拷贝事件结构（cuda_memcpy）记录数据传输操作的详细信息，包括开始和结束时间、源和目标地址、传输字节数、进程 ID 和传输类型。

```c
struct gpukern_sample {
    int pid, ppid;                    // 进程 ID 和父进程 ID
    char comm[TASK_COMM_LEN];         // 进程名称
    uint64_t kern_func_off;           // 内核函数偏移地址
    int grid_x, grid_y, grid_z;       // 网格维度
    int block_x, block_y, block_z;    // 块维度
    uint64_t stream;                  // CUDA 流 ID
    uint64_t args[MAX_GPUKERN_ARGS];  // 内核参数
    size_t ustack_sz;                 // 用户栈深度
    stack_trace_t ustack;             // 用户调用栈
};
```

```c
struct memleak_event {
    __u64 start;                      // 开始时间戳
    __u64 end;                        // 结束时间戳
    __u64 device_addr;                // 设备内存地址
    __u64 size;                       // 内存大小
    __u32 pid;                        // 进程 ID
    __s32 ret;                        // 返回值
    enum memleak_event_t event_type;  // 事件类型 (MALLOC/FREE)
};
```

```c
struct cuda_memcpy {
    __u64 start_time;                 // 开始时间
    __u64 end_time;                   // 结束时间
    __u64 dst;                        // 目标地址
    __u64 src;                        // 源地址
    __u64 count;                      // 传输字节数
    __u32 pid;                        // 进程 ID
    enum memcpy_kind kind;            // 传输类型 (H2D/D2H/D2D)
};
```
程序使用多种 BPF 映射进行数据存储和事件传输。内核事件环形缓冲区用于高效传输事件到用户空间，设备指针映射跟踪进程与设备指针的关联，数据传输跟踪映射支持最多10240个并发的内存拷贝操作。(这些数据结构也可以通过宏定义调整最大容量)

核心算法涵盖了 CUDA 内核启动跟踪、内存管理跟踪和数据传输监控三个主要流程。内核启动跟踪包括从寄存器中提取网格和块配置、解析内核函数地址偏移、读取内核参数数组、捕获用户调用栈，最后将完整事件通过环形缓冲区发送到用户空间。内存管理跟踪通过 uprobe 捕获分配调用记录大小参数，uretprobe 获取返回的设备地址和状态码，计算分配耗时并检测失败情况，同时跟踪释放操作并通过匹配分配和释放事件识别内存泄漏。数据传输监控识别传输类型（主机到主机、主机到设备、设备到主机、设备到设备），记录传输开始和结束时间计算传输带宽，通过 PID 映射跟踪未完成的异步传输操作。