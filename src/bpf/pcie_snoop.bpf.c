#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#define TASK_COMM_LEN 16

// #define SAFE_READ(dst, src_type, src, member) \
//     do { \
//         src_type *__tmp; \
//         bpf_probe_read(&__tmp, sizeof(__tmp), &(src)); \
//         if (__tmp) \
//             bpf_probe_read(&(dst), sizeof(dst), &(__tmp->member)); \
//     } while (0)

char LICENSE[] SEC("license") = "GPL";

enum event_type {
    EVENT_PCIE_ENTER = 0,
    EVENT_PCIE_EXIT = 1,
};

enum pcie_op_type {
    PCIE_CONFIG_READ_BYTE = 0,
    PCIE_CONFIG_READ_WORD = 1,
    PCIE_CONFIG_READ_DWORD = 2,
    PCIE_CONFIG_WRITE_BYTE = 3,
    PCIE_CONFIG_WRITE_WORD = 4,
    PCIE_CONFIG_WRITE_DWORD = 5,
};

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

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 3);
    __type(key, u32);
    __type(value, u32);
} snoop_proc SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

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

// 通用辅助函数: 创建并初始化事件
static inline struct pcie_event *create_pcie_event(enum event_type type, u32 pid, u32 tgid) {
    struct pcie_event *event = bpf_ringbuf_reserve(&events, sizeof(struct pcie_event), 0);
    if (!event)
        return NULL;
        
    event->type = type;
    event->pid = pid;
    event->tgid = tgid;
    event->timestamp = bpf_ktime_get_ns();
    bpf_get_current_comm(&event->comm, sizeof(event->comm));
    
    return event;
}



// DMA Fence 事件类型
enum dma_fence_event_type {
    DMA_FENCE_INIT = 0,
    DMA_FENCE_DESTROY = 1,
    DMA_FENCE_ENABLE_SIGNAL = 2,
    DMA_FENCE_SIGNALED = 3,
    DMA_FENCE_WAIT_START = 4,
    DMA_FENCE_WAIT_END = 5,
    DMA_FENCE_EMIT = 6
};

// 修改 dma_fence_init 的结构定义
// struct trace_event_raw_dma_fence_init {
//     unsigned short common_type;
//     unsigned char common_flags;
//     unsigned char common_preempt_count;
//     int common_pid;
    
//     // 根据 format 输出中的字段
//     char __data_loc_driver[4];   // offset:8; size:4
//     char __data_loc_timeline[4]; // offset:12; size:4
//     unsigned int context;        // offset:16; size:4
//     unsigned int seqno;          // offset:20; size:4
// };

struct trace_event_raw_dma_fence {
	struct trace_entry ent;
	u32 __data_loc_driver;
	u32 __data_loc_timeline;
	unsigned int context;
	unsigned int seqno;
	char __data[0];
};

// DMA Fence 事件结构
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

// 更新 fence_wait_start 映射使用组合键
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

// 创建 DMA Fence 事件
static inline struct dma_fence_event *create_dma_fence_event(enum dma_fence_event_type type, u32 pid, u32 tgid) {
    struct dma_fence_event *event = bpf_ringbuf_reserve(&events, sizeof(struct dma_fence_event), 0);
    if (!event)
        return NULL;
        
    event->type = type;
    event->pid = pid;
    event->tgid = tgid;
    event->timestamp = bpf_ktime_get_ns();
    bpf_get_current_comm(&event->comm, sizeof(event->comm));
    
    return event;
}

// 用于从 __data_loc 字段提取字符串的辅助函数
static __always_inline int 
read_str_from_data_loc(void *dst, int dst_size, const void *src, const void *base) {
    // 读取 __data_loc 值
    int data_loc;
    bpf_probe_read_kernel(&data_loc, sizeof(data_loc), src);
    
    // 提取偏移量和长度
    const void *ptr = base + (data_loc & 0xffff); // 低16位是偏移量
    int len = data_loc >> 16;                     // 高16位是长度
    
    // 确保不超过目标缓冲区大小
    if (len > dst_size - 1)
        len = dst_size - 1;
        
    // 读取字符串
    return bpf_probe_read_kernel_str(dst, len, ptr);
}

SEC("kprobe/pci_bus_read_config_byte")
int BPF_KPROBE(trace_pci_bus_read_byte_enter, struct pci_bus *bus, unsigned int devfn, int where, u8 *val) {
    bpf_printk("trace_pci_bus_read_byte_enter\n");
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    // if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
    //     bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
    //     return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_ENTER, pid, tgid);
    if (!event)
        return 0;

    handle_pci_bus_read_enter(event, bus, devfn, where, 1, PCIE_CONFIG_READ_BYTE);
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/pci_bus_read_config_byte")
int BPF_KRETPROBE(trace_pci_bus_read_byte_exit, int ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_EXIT, pid, tgid);
    if (!event)
        return 0;

    event->op_type = PCIE_CONFIG_READ_BYTE;
    event->value = ret;
    bpf_ringbuf_submit(event, 0);
    return 0;
}


SEC("kprobe/pci_bus_read_config_word")
int BPF_KPROBE(trace_pci_bus_read_word_enter, struct pci_bus *bus, unsigned int devfn, int where, u16 *val) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_ENTER, pid, tgid);
    if (!event)
        return 0;

    handle_pci_bus_read_enter(event, bus, devfn, where, 2, PCIE_CONFIG_READ_WORD);
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/pci_bus_read_config_word")
int BPF_KRETPROBE(trace_pci_bus_read_word_exit, int ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_EXIT, pid, tgid);
    if (!event)
        return 0;

    event->op_type = PCIE_CONFIG_READ_WORD;
    event->value = ret;
    bpf_ringbuf_submit(event, 0);
    return 0;
}


SEC("kprobe/pci_bus_read_config_dword")
int BPF_KPROBE(trace_pci_bus_read_dword_enter, struct pci_bus *bus, unsigned int devfn, int where, u32 *val) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_ENTER, pid, tgid);
    if (!event)
        return 0;

    handle_pci_bus_read_enter(event, bus, devfn, where, 4, PCIE_CONFIG_READ_DWORD);
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/pci_bus_read_config_dword")
int BPF_KRETPROBE(trace_pci_bus_read_dword_exit, int ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_EXIT, pid, tgid);
    if (!event)
        return 0;

    event->op_type = PCIE_CONFIG_READ_DWORD;
    event->value = ret;
    bpf_ringbuf_submit(event, 0);
    return 0;
}


SEC("kprobe/pci_bus_write_config_byte")
int BPF_KPROBE(trace_pci_bus_write_byte_enter, struct pci_bus *bus, unsigned int devfn, int where, u8 val) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_ENTER, pid, tgid);
    if (!event)
        return 0;

    handle_pci_bus_write_enter(event, bus, devfn, where, 1, val, PCIE_CONFIG_WRITE_BYTE);
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/pci_bus_write_config_byte")
int BPF_KRETPROBE(trace_pci_bus_write_byte_exit, int ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_EXIT, pid, tgid);
    if (!event)
        return 0;

    event->op_type = PCIE_CONFIG_WRITE_BYTE;
    event->value = ret;
    bpf_ringbuf_submit(event, 0);
    return 0;
}


SEC("kprobe/pci_bus_write_config_word")
int BPF_KPROBE(trace_pci_bus_write_word_enter, struct pci_bus *bus, unsigned int devfn, int where, u16 val) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_ENTER, pid, tgid);
    if (!event)
        return 0;

    handle_pci_bus_write_enter(event, bus, devfn, where, 2, val, PCIE_CONFIG_WRITE_WORD);
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/pci_bus_write_config_word")
int BPF_KRETPROBE(trace_pci_bus_write_word_exit, int ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_EXIT, pid, tgid);
    if (!event)
        return 0;

    event->op_type = PCIE_CONFIG_WRITE_WORD;
    event->value = ret;
    bpf_ringbuf_submit(event, 0);
    return 0;
}


SEC("kprobe/pci_bus_write_config_dword")
int BPF_KPROBE(trace_pci_bus_write_dword_enter, struct pci_bus *bus, unsigned int devfn, int where, u32 val) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_ENTER, pid, tgid);
    if (!event)
        return 0;

    handle_pci_bus_write_enter(event, bus, devfn, where, 4, val, PCIE_CONFIG_WRITE_DWORD);
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/pci_bus_write_config_dword")
int BPF_KRETPROBE(trace_pci_bus_write_dword_exit, int ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct pcie_event *event = create_pcie_event(EVENT_PCIE_EXIT, pid, tgid);
    if (!event)
        return 0;

    event->op_type = PCIE_CONFIG_WRITE_DWORD;
    event->value = ret;
    bpf_ringbuf_submit(event, 0);
    return 0;
}




// SEC("tracepoint/dma_fence/dma_fence_init")
// int trace_dma_fence_init(struct trace_event_raw_dma_fence_init *ctx) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     // PID 过滤
//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct dma_fence_event *event = create_dma_fence_event(DMA_FENCE_INIT, pid, tgid);
//     if (!event)
//         return 0;

//     event->context = ctx->context;
//     event->seqno = ctx->seqno;
    
//     read_str_from_data_loc(event->driver_name, sizeof(event->driver_name), 
//                           ctx->__data_loc_driver, ctx);
//     read_str_from_data_loc(event->timeline_name, sizeof(event->timeline_name), 
//                           ctx->__data_loc_timeline, ctx);
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// 跟踪 dma_fence_destroy
SEC("tracepoint/dma_fence/dma_fence_destroy")
int trace_dma_fence_destroy(struct trace_event_raw_dma_fence *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    // PID 过滤
    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct dma_fence_event *event = create_dma_fence_event(DMA_FENCE_DESTROY, pid, tgid);
    if (!event)
        return 0;

    event->context = ctx->context;
    event->seqno = ctx->seqno;
    
    // 读取驱动和timeline名称
    // read_str_from_data_loc(event->driver_name, sizeof(event->driver_name), 
    //                       ctx->__data_loc_driver, ctx);
    // read_str_from_data_loc(event->timeline_name, sizeof(event->timeline_name), 
    //                       ctx->__data_loc_timeline, ctx);
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("tracepoint/dma_fence/dma_fence_enable_signal")
int trace_dma_fence_enable_signal(struct trace_event_raw_dma_fence *ctx) {
    bpf_printk("trace_dma_fence_enable_signal\n");
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct dma_fence_event *event = create_dma_fence_event(DMA_FENCE_ENABLE_SIGNAL, pid, tgid);
    if (!event)
        return 0;

    event->context = ctx->context;
    event->seqno = ctx->seqno;
    
    // read_str_from_data_loc(event->driver_name, sizeof(event->driver_name), 
    //                       ctx->__data_loc_driver, ctx);
    // read_str_from_data_loc(event->timeline_name, sizeof(event->timeline_name), 
    //                       ctx->__data_loc_timeline, ctx);
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

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
    
    // read_str_from_data_loc(event->driver_name, sizeof(event->driver_name), 
    //                       ctx->__data_loc_driver, ctx);
    // read_str_from_data_loc(event->timeline_name, sizeof(event->timeline_name), 
    //                       ctx->__data_loc_timeline, ctx);
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("tracepoint/dma_fence/dma_fence_emit")
int trace_dma_fence_emit(struct trace_event_raw_dma_fence *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct dma_fence_event *event = create_dma_fence_event(DMA_FENCE_EMIT, pid, tgid);
    if (!event)
        return 0;

    event->context = ctx->context;
    event->seqno = ctx->seqno;
    
    // read_str_from_data_loc(event->driver_name, sizeof(event->driver_name), 
    //                       ctx->__data_loc_driver, ctx);
    // read_str_from_data_loc(event->timeline_name, sizeof(event->timeline_name), 
    //                       ctx->__data_loc_timeline, ctx);
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

// 修改 dma_fence_wait_start 和 dma_fence_wait_end
SEC("tracepoint/dma_fence/dma_fence_wait_start")
int trace_dma_fence_wait_start(struct trace_event_raw_dma_fence *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct dma_fence_event *event = create_dma_fence_event(DMA_FENCE_WAIT_START, pid, tgid);
    if (!event)
        return 0;

    event->context = ctx->context;
    event->seqno = ctx->seqno;
    
    // 读取驱动和timeline名称
    // read_str_from_data_loc(event->driver_name, sizeof(event->driver_name), 
    //                       ctx->__data_loc_driver, ctx);
    // read_str_from_data_loc(event->timeline_name, sizeof(event->timeline_name), 
    //                       ctx->__data_loc_timeline, ctx);
    
    // 使用组合键保存等待开始时间
    struct fence_wait_key key = {
        .pid = pid,
        .tgid = tgid,
        .context = ctx->context,
        .seqno = ctx->seqno
    };
    
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&fence_wait_start, &key, &ts, BPF_ANY);
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("tracepoint/dma_fence/dma_fence_wait_end")
int trace_dma_fence_wait_end(struct trace_event_raw_dma_fence *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct dma_fence_event *event = create_dma_fence_event(DMA_FENCE_WAIT_END, pid, tgid);
    if (!event)
        return 0;

    event->context = ctx->context;
    event->seqno = ctx->seqno;
    
    // 读取驱动和timeline名称
    // read_str_from_data_loc(event->driver_name, sizeof(event->driver_name), 
    //                       ctx->__data_loc_driver, ctx);
    // read_str_from_data_loc(event->timeline_name, sizeof(event->timeline_name), 
    //                       ctx->__data_loc_timeline, ctx);
    
    // 查找开始时间并计算持续时间
    struct fence_wait_key key = {
        .pid = pid,
        .tgid = tgid,
        .context = ctx->context,
        .seqno = ctx->seqno
    };
    
    u64 *start_ts = bpf_map_lookup_elem(&fence_wait_start, &key);
    if (start_ts) {
        event->duration_ns = bpf_ktime_get_ns() - *start_ts;
        bpf_map_delete_elem(&fence_wait_start, &key);
    }
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}