#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#define TASK_COMM_LEN 16

char LICENSE[] SEC("license") = "GPL";

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
    __uint(max_entries, 3);
    __type(key, u32);
    __type(value, u32);
} snoop_proc SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 512);
    __type(key, u32);
    __type(value, u32);
} traced_syscalls SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// 系统调用入口追踪
SEC("tp/raw_syscalls/sys_enter")
int trace_syscall_enter(struct trace_event_raw_sys_enter *ctx) {
    u32 syscall_id = ctx->id;
    u32 *should_trace = bpf_map_lookup_elem(&traced_syscalls, &syscall_id);
    if (!should_trace || *should_trace == 0)
        return 0;

    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct call_id_key_t key = {.pid = pid, .syscall_id = syscall_id};
    u64 *call_id = bpf_map_lookup_elem(&call_id_map, &key);
    u64 next_id = call_id ? (*call_id + 1) : 1;
    bpf_map_update_elem(&call_id_map, &key, &next_id, BPF_ANY);

    struct syscall_event *event = bpf_ringbuf_reserve(&events, sizeof(struct syscall_event), 0);
    if (!event)
        return 0;

    event->type = EVENT_SYSCALL_ENTER;
    event->pid = pid;
    event->tgid = tgid;
    event->syscall_id = syscall_id;
    event->call_id = next_id;
    event->timestamp = bpf_ktime_get_ns();
    bpf_get_current_comm(&event->comm, sizeof(event->comm));
    event->ret_val = 0;
    event->args[0] = ctx->args[0];
    event->args[1] = ctx->args[1];
    event->args[2] = ctx->args[2];
    event->args[3] = ctx->args[3];
    event->args[4] = ctx->args[4];
    event->args[5] = ctx->args[5];

    bpf_ringbuf_submit(event, 0);
    return 0;
}

// 系统调用退出追踪
SEC("tp/raw_syscalls/sys_exit")
int trace_syscall_exit(struct trace_event_raw_sys_exit *ctx) {
    u32 syscall_id = ctx->id;
    u32 *should_trace = bpf_map_lookup_elem(&traced_syscalls, &syscall_id);
    if (!should_trace || *should_trace == 0)
        return 0;

    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32)pid_tgid;
    u32 tgid = (u32)(pid_tgid >> 32);

    if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
        bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
        return 0;

    struct call_id_key_t key = {.pid = pid, .syscall_id = syscall_id};
    u64 *call_id = bpf_map_lookup_elem(&call_id_map, &key);

    struct syscall_event *event = bpf_ringbuf_reserve(&events, sizeof(struct syscall_event), 0);
    if (!event)
        return 0;

    event->type = EVENT_SYSCALL_EXIT;
    event->pid = pid;
    event->tgid = tgid;
    event->syscall_id = syscall_id;
    event->call_id = call_id ? *call_id : 0;
    event->timestamp = bpf_ktime_get_ns();
    bpf_get_current_comm(&event->comm, sizeof(event->comm));
    event->ret_val = ctx->ret;
    event->args[0] = 0;
    event->args[1] = 0;
    event->args[2] = 0;
    event->args[3] = 0;
    event->args[4] = 0;
    event->args[5] = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}