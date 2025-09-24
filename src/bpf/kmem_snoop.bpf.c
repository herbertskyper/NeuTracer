#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
// #include "kmem_snoop.h"

#define TASK_COMM_LEN 16
#define OFFSET_BYTES_ALLOC 0x20
#define OFFSET_PTR 0x10
#define PERF_MAX_STACK_DEPTH 127

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 3);
    __type(key, u32);
    __type(value, u32);
} snoop_proc SEC(".maps");

struct kmem_event {
    __u32 pid;
    __u32 tgid;
    __u64 size;
    __u64 addr;
    __u64 timestamp_ns;
    __u32 stack_id;
    __u32 event_type;  // 0: alloc, 1: free
    char comm[TASK_COMM_LEN];
};

struct alloc_info {
    __u64 size;
    __u64 timestamp_ns;
    __u32 stack_id;
    __u32 padding;  // Padding to align the structure size
};

// struct combined_alloc_info {
//     __u64 total_size;
//     __u64 number_of_allocs;
// };

// Ring buffer for events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 256);
} mem_events SEC(".maps");

// Record allocation sizes temporarily
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u64);       // pid_tgid
    __type(value, u64);     // size
} sizes SEC(".maps");

// Record memory allocation information
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);  // Increased for more allocations
    __type(key, u64);       // address
    __type(value, struct alloc_info);
} allocs SEC(".maps");

// Record call stacks
struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(max_entries, 1024);
    __type(key, u32);       // stack_id
    __type(value, u64[PERF_MAX_STACK_DEPTH]);
} stack_traces SEC(".maps");

// Record process-wide memory usage
// struct {
//     __uint(type, BPF_MAP_TYPE_HASH);
//     __uint(max_entries, 1024);
//     __type(key, u32);       // tgid
//     __type(value, struct combined_alloc_info);
// } combined_allocs SEC(".maps");

// Processes to monitor

// Update statistics for allocations
// static __always_inline void update_statistics_add(u32 tgid, u64 sz) {
//     struct combined_alloc_info *existing, cinfo = {0};
    
//     existing = bpf_map_lookup_elem(&combined_allocs, &tgid);
//     if (existing) {
//         cinfo = *existing;
//     }
    
//     cinfo.total_size += sz;
//     cinfo.number_of_allocs += 1;
//     bpf_map_update_elem(&combined_allocs, &tgid, &cinfo, BPF_ANY);
// }

// // Update statistics for deallocations
// static __always_inline void update_statistics_del(u32 tgid, u64 sz) {
//     struct combined_alloc_info *existing = bpf_map_lookup_elem(&combined_allocs, &tgid);
//     if (!existing) return;
    
//     if (sz >= existing->total_size) {
//         existing->total_size = 0;
//     } else {
//         existing->total_size -= sz;
//     }
    
//     if (existing->number_of_allocs > 0) {
//         existing->number_of_allocs -= 1;
//     }
// }

// Tracepoint handlers with direct field access based on struct definitions
SEC("tracepoint/kmem/kmalloc")
int handle_kmalloc(struct trace_event_raw_kmalloc *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    u32 pid = pid_tgid & 0xFFFFFFFF;

    if(bpf_map_lookup_elem(&snoop_proc, &pid) == NULL && bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }
    
    size_t bytes_alloc;
    bpf_probe_read(&bytes_alloc, sizeof(bytes_alloc), (void *)ctx + OFFSET_BYTES_ALLOC);
    const void *ptr;
    bpf_probe_read(&ptr, sizeof(ptr), (void *)ctx + OFFSET_PTR);
    u64 address = (u64)ptr;
    u64 size64 = bytes_alloc;


    // Skip if size is zero or process not monitored
    if (size64 == 0 
        //|| !bpf_map_lookup_elem(&snoop_proc, &tgid)
        ) {
        return 0;
    }
    
    // 存储分配大小
    bpf_map_update_elem(&sizes, &pid_tgid, &size64, BPF_ANY);
    
    // 处理分配结果
    if (address != 0) {
        // 获取栈跟踪ID
        u32 stack_id = bpf_get_stackid(ctx, &stack_traces, BPF_F_USER_STACK);
        u64 timestamp_ns = bpf_ktime_get_ns();
        
        // 创建分配信息记录
        struct alloc_info info = {
            .size = size64,
            .timestamp_ns = timestamp_ns,
            .stack_id = stack_id,
            .padding = 0
        };
        
        // 更新分配映射
        bpf_map_update_elem(&allocs, &address, &info, BPF_ANY);
        //update_statistics_add(tgid, info.size);
        
        // 向环形缓冲区提交事件
        struct kmem_event *event = bpf_ringbuf_reserve(&mem_events, sizeof(struct kmem_event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->size = size64;
            event->addr = address;
            event->timestamp_ns = timestamp_ns;
            event->stack_id = stack_id;
            event->event_type = 0; // 0 表示分配
            bpf_get_current_comm(&event->comm, sizeof(event->comm));
            
            bpf_ringbuf_submit(event, 0);
        }
    }
    
    bpf_map_delete_elem(&sizes, &pid_tgid);
    return 0;
}

SEC("tracepoint/kmem/kfree")
int handle_kfree(struct trace_event_raw_kfree *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    u32 pid = pid_tgid & 0xFFFFFFFF;

    if(bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL && bpf_map_lookup_elem(&snoop_proc, &pid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }
    
    const void *ptr;
    bpf_probe_read(&ptr, sizeof(ptr), (void *)ctx + OFFSET_PTR);
    u64 address = (u64)ptr;

    struct alloc_info *info = bpf_map_lookup_elem(&allocs, &address);
    if (info 
        //&& bpf_map_lookup_elem(&snoop_proc, &tgid)
        ) {
        u64 size = info->size;
        u32 stack_id = info->stack_id;
        //update_statistics_del(tgid, size);
        
        // 向环形缓冲区提交释放事件
        struct kmem_event *event = bpf_ringbuf_reserve(&mem_events, sizeof(struct kmem_event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->size = size;
            event->addr = address;
            event->timestamp_ns = bpf_ktime_get_ns();
            event->stack_id = stack_id;
            event->event_type = 1; // 1 表示释放
            bpf_get_current_comm(&event->comm, sizeof(event->comm));
            
            
            bpf_ringbuf_submit(event, 0);
        }
        
        bpf_map_delete_elem(&allocs, &address);
    }
    return 0;
}

SEC("tracepoint/kmem/kmem_cache_alloc")
int handle_kmem_cache_alloc(struct trace_event_raw_kmem_cache_alloc *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    u32 pid = pid_tgid & 0xFFFFFFFF;

    if(bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL && bpf_map_lookup_elem(&snoop_proc, &pid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }
    
    size_t bytes_alloc;
    bpf_probe_read(&bytes_alloc, sizeof(bytes_alloc), (void *)ctx + OFFSET_BYTES_ALLOC);
    const void *ptr;
    bpf_probe_read(&ptr, sizeof(ptr), (void *)ctx + OFFSET_PTR);
    u64 address = (u64)ptr;
    u64 size64 = bytes_alloc;

    // Skip if size is zero or process not monitored
    if (size64 == 0 
        //|| !bpf_map_lookup_elem(&snoop_proc, &tgid)
        ) {
        return 0;
    }
    
    // 存储分配大小
    bpf_map_update_elem(&sizes, &pid_tgid, &size64, BPF_ANY);
    
    // 处理分配结果
    if (address != 0) {
        // 获取栈跟踪ID
        u32 stack_id = bpf_get_stackid(ctx, &stack_traces, BPF_F_USER_STACK);
        u64 timestamp_ns = bpf_ktime_get_ns();
        
        // 创建分配信息记录
        struct alloc_info info = {
            .size = size64,
            .timestamp_ns = timestamp_ns,
            .stack_id = stack_id,
            .padding = 0
        };
        
        // 更新分配映射
        bpf_map_update_elem(&allocs, &address, &info, BPF_ANY);
        //update_statistics_add(tgid, info.size);
        
        // 向环形缓冲区提交事件
        struct kmem_event *event = bpf_ringbuf_reserve(&mem_events, sizeof(struct kmem_event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->size = size64;
            event->addr = address;
            event->timestamp_ns = timestamp_ns;
            event->stack_id = stack_id;
            event->event_type = 0; // 0 表示分配
            bpf_get_current_comm(&event->comm, sizeof(event->comm));
            
            
            bpf_ringbuf_submit(event, 0);
        }
    }
    
    bpf_map_delete_elem(&sizes, &pid_tgid);
    return 0;
}

SEC("tracepoint/kmem/kmem_cache_free")
int handle_kmem_cache_free(struct trace_event_raw_kmem_cache_free *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    u32 pid = pid_tgid & 0xFFFFFFFF;

    if(bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL && bpf_map_lookup_elem(&snoop_proc, &pid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }
    
    const void *ptr;
    bpf_probe_read(&ptr, sizeof(ptr), (void *)ctx + OFFSET_PTR);
    u64 address = (u64)ptr;

    struct alloc_info *info = bpf_map_lookup_elem(&allocs, &address);
    if (info 
        // && bpf_map_lookup_elem(&snoop_proc, &tgid)
        ) {
        u64 size = info->size;
        u32 stack_id = info->stack_id;
        //update_statistics_del(tgid, size);
        
        // 向环形缓冲区提交释放事件
        struct kmem_event *event = bpf_ringbuf_reserve(&mem_events, sizeof(struct kmem_event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->size = size;
            event->addr = address;
            event->timestamp_ns = bpf_ktime_get_ns();
            event->stack_id = stack_id;
            event->event_type = 1; // 1 表示释放
            bpf_get_current_comm(&event->comm, sizeof(event->comm));
            
            
            bpf_ringbuf_submit(event, 0);
        }
        
        bpf_map_delete_elem(&allocs, &address);
    }
    return 0;
}

char _license[] SEC("license") = "GPL";