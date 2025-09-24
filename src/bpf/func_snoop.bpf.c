#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

// 定义事件类型
enum event_type {
    EVENT_ENTRY = 0,
    EVENT_EXIT = 1,
};

// 修改事件结构，添加 pid 和 tgid 字段
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
    event->type = EVENT_EXIT;
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

    event->ret_val = PT_REGS_RC(ctx); // 获取返回值
    event->args[0] = 0;
    event->args[1] = 0;
    event->args[2] = 0;
    event->args[3] = 0; 
    event->args[4] = 0;
    event->args[5] = 0; // 退出点不需要参数，
    
    // 提交事件到 ring buffer
    bpf_ringbuf_submit(event, 0);
    return 0;
}