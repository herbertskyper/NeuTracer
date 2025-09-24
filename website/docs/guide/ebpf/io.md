# I/O模块 eBPF 部分

bio_snoop.bpf.c 用于跟踪和分析 Linux 内核中的块 I/O 操作，捕获磁盘读写操作的详细信息，包括延迟、吞吐量、读写模式和设备使用情况。该模块使用 eBPF 技术直接在内核空间捕获块 I/O 事件`block/block_rq_issue`和`block/block_rq_complete `，实现低开销的 I/O 监控。支持设备识别、读写分类、请求大小分析、进程关联和高效数据传输，通过环形缓冲区向用户态程序传输事件数据。

模块使用三个核心数据结构来管理和传输 I/O 事件信息：请求标识符（req_key_t）包含设备主次编号和扇区号用于唯一标识 I/O 请求，请求信息（start_req_t）存储请求发起时的时间戳、大小、类型标志和进程名，I/O 事件（bio_event）是传递到用户空间的完整事件结构，包含延迟、传输字节数、读写标志、设备编号和时间戳等信息。
```c
struct req_key_t {
    __u32 dev_major;
    __u32 dev_minor;
    __u64 sector;
};

struct start_req_t {
    __u64 ts;
    __u32 bytes;
    char rwbs[RWBS_LEN];
    char comm[TASK_COMM_LEN];
};

struct bio_event {
    __u32 pid;         // 进程 ID
    char comm[TASK_COMM_LEN]; // 进程名
    __u64 delta_us;    // I/O 延迟（微秒）
    __u64 bytes;       // 传输字节数
    __u8 rwflag;       // 读写标志：0=读，1=写
    __u8 major;        // 设备主编号
    __u8 minor;        // 设备次编号
    __u64 timestamp;   // 时间戳
    __u64 io_count;    // 当前设备的 I/O 计数
};
```

由于vmlinux.h 文件缺少部分结构体，我们翻阅了源码补充了这些结构体。以下是关键数据结构的定义：
```cpp
struct block_rq_issue_ctx {
    // 公共头部
    unsigned short common_type;
    unsigned char common_flags;
    unsigned char common_preempt_count;
    int common_pid;
    
    __u32 dev;           // offset: 8
    __u64 sector;        // offset: 16
    __u32 nr_sector;     // offset: 24
    __u32 bytes;         // offset: 28
     char rwbs[RWBS_LEN]; // offset: 32 
    char comm[TASK_COMM_LEN]; // offset: 40 
    __u32 cmd_loc;       // offset: 56 
};

struct block_rq_complete_ctx {
    // 公共头部
    unsigned short common_type;
    unsigned char common_flags;
    unsigned char common_preempt_count;
    int common_pid;
    
    __u32 dev;
    __u64 sector;
    unsigned int nr_sector;
    int error;
    char rwbs[RWBS_LEN]; 
    __u32 cmd_loc;
};
```
I/O 请求跟踪流程分为两个关键阶段。在请求发起阶段（block_rq_issue），代码捕获请求，包括设备、扇区、大小等信息，记录请求开始时间，并将相关信息存储到 start 映射中。在请求完成阶段（block_rq_complete），系统查找对应的请求记录，计算准确的延迟时间，更新设备 I/O 计数，最后创建完整事件并发送到环形缓冲区供用户空间程序处理。