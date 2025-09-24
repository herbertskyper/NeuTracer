#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#define TASK_COMM_LEN 16

char LICENSE[] SEC("license") = "GPL";

enum event_type {
    EVENT_NVLINK_ENTER = 0,
    EVENT_NVLINK_EXIT = 1,
};

enum nvlink_func_id {
    NVLINK_FUNC_STRCPY = 0,
    NVLINK_FUNC_MEMCPY = 1,
    NVLINK_FUNC_MEMSET = 2,
};

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

static struct nvlink_event *create_nvlink_event(enum event_type type, 
                                           enum nvlink_func_id func_id, 
                                           u32 pid, u32 tgid) {
    struct nvlink_event *event = bpf_ringbuf_reserve(&events, sizeof(struct nvlink_event), 0);
    if (!event)
        return NULL;
        
    event->type = type;
    event->func_id = func_id;
    event->pid = pid;
    event->tgid = tgid;
    event->timestamp = bpf_ktime_get_ns();
    bpf_get_current_comm(&event->comm, sizeof(event->comm));
    
    return event;
}

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

SEC("kretprobe/nvlink_strcpy")
int BPF_KRETPROBE(trace_nvlink_strcpy_exit, char *ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct nvlink_event *event = create_nvlink_event(EVENT_NVLINK_EXIT, 
                                                NVLINK_FUNC_STRCPY, pid, tgid);
    if (!event)
        return 0;

    event->ret_val = (u64)ret;
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

// nvlink_memcpy 追踪
SEC("kprobe/nvlink_memcpy")
int BPF_KPROBE(trace_nvlink_memcpy_enter, void *dst, const void *src, size_t size) {
    bpf_printk("enter the memcpy");
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct nvlink_event *event = create_nvlink_event(EVENT_NVLINK_ENTER, 
                                                NVLINK_FUNC_MEMCPY, pid, tgid);
    if (!event)
        return 0;

    event->dst_addr = (u64)dst;
    event->src_addr = (u64)src;
    event->size = size;
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/nvlink_memcpy")
int BPF_KRETPROBE(trace_nvlink_memcpy_exit, void *ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct nvlink_event *event = create_nvlink_event(EVENT_NVLINK_EXIT, 
                                                NVLINK_FUNC_MEMCPY, pid, tgid);
    if (!event)
        return 0;

    event->ret_val = (u64)ret;
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

// nvlink_memset 追踪
SEC("kprobe/nvlink_memset")
int BPF_KPROBE(trace_nvlink_memset_enter, void *dst, int val, size_t size) {
    bpf_printk("enter the memset");
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct nvlink_event *event = create_nvlink_event(EVENT_NVLINK_ENTER, 
                                                NVLINK_FUNC_MEMSET, pid, tgid);
    if (!event)
        return 0;

    event->dst_addr = (u64)dst;
    event->src_addr = val;  // 对于memset，这是填充值
    event->size = size;
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/nvlink_memset")
int BPF_KRETPROBE(trace_nvlink_memset_exit, void *ret) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct nvlink_event *event = create_nvlink_event(EVENT_NVLINK_EXIT, 
                                                NVLINK_FUNC_MEMSET, pid, tgid);
    if (!event)
        return 0;

    event->ret_val = (u64)ret;
    
    bpf_ringbuf_submit(event, 0);
    return 0;
}