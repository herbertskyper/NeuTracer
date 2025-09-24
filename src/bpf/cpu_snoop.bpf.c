// SPDX-License-Identifier: GPL-2.0
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf_config/config.h"

#define TASK_COMM_LEN 16
#define MAX_PID (1 << 2)
#define IGNORE_PID 0  // swapper 的 PID
#define WINDOW_SIZE_NS (10000000000ULL)  // 改为10秒窗口
#define MAX_CPUS 64

struct cpu_event {
    u32 pid;
    u32 ppid;
    u32 cpu_id;
    u64 oncpu_time;
    u64 offcpu_time;
    u64 utilization;  // 0-10000 (100.00%)
    u64 timestamp;
    char comm[TASK_COMM_LEN];
};

struct task_data {
    u64 window_start;      // 窗口开始时间
    u64 last_switch_time;  // 最后切换时间  
    u64 window_oncpu;      // 窗口内累积on-cpu时间
    u32 last_cpu;          // 最后使用的CPU
    u32 is_oncpu;          // 当前是否在CPU上 
};


// 用于存储每个任务的 CPU 时间数据
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_PID);
    __type(key, u32);
    __type(value, struct task_data);
} task_stats SEC(".maps");

// 每个CPU的统计
// struct {
//     __uint(type, BPF_MAP_TYPE_ARRAY);
//     __uint(max_entries, MAX_CPUS);
//     __type(key, u32);
//     __type(value, struct cpu_stats);
// } cpu_stats_map SEC(".maps");

// // 用于存储进程信息
// struct {
//     __uint(type, BPF_MAP_TYPE_HASH);
//     __uint(max_entries, MAX_PID);
//     __type(key, u32);
//     __type(value, struct cpu_event);
// } proc_info SEC(".maps");

// 环形缓冲区输出
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} cpu_events SEC(".maps");

// 添加一个 BPF_MAP 来动态设置 target_pid
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 3);
    __type(key, u32);
    __type(value, u32);
} snoop_proc SEC(".maps");

static __always_inline bool should_track(u32 pid) {
    // 忽略 swapper 进程
    if (pid == IGNORE_PID)
        return false;
    
    // 查找进程是否在监控列表中
    u32 *found = bpf_map_lookup_elem(&snoop_proc, &pid);
    
    // 如果 snoop_proc 为空(没有设置监控的进程)，则监控所有进程
    // 如果设置了监控进程，则只监控列表中的进程
    return found != NULL;
}

// static __always_inline bool should_ignore_thread(const char *comm) {
//     // 检查是否是 kworker 或 KVM 线程
//     if (comm[0] == 'k' && comm[1] == 'w' && comm[2] == 'o' && comm[3] == 'r' &&
//         comm[4] == 'k' && comm[5] == 'e' && comm[6] == 'r') {  // 以 "kworker" 开头
//         return true;
//     }
//     if (comm[0] == 'C' && comm[1] == 'P' && comm[2] == 'U') {  // 以 "CPU" 开头 (KVM 线程)
//         return true;
//     }
//     return false;
// }

// static __always_inline u64 safe_div(u64 a, u64 b) {
//     if (b == 0) return 0;
//     u64 result = a * 10000 / b;  
//     if(result > 10000) {
//         return 9999;  // 限制最大值为9999 (99.99%)
//     }
//     return result;
// }


// static __always_inline void update_window_stats(struct task_data *data, u64 ts, u64 oncpu_delta, u32 cpu_id) {
//     // 检查是否需要重置窗口
//     if (ts - data->window_start >= WINDOW_SIZE_NS) {
//         data->window_start = ts;
//         data->window_oncpu = 0;
//         data->window_total = 0;
//     }
// }
static __always_inline void send_cpu_event(u32 pid, struct task_data *data, u64 ts, u32 cpu_id, struct task_struct *task) {
    struct cpu_event *event = bpf_ringbuf_reserve(&cpu_events, sizeof(*event), 0);
    if (!event) return;
    __builtin_memset(event, 0, sizeof(*event));
    
    event->pid = pid;
    event->ppid = BPF_CORE_READ(task, real_parent, pid);
    event->cpu_id = cpu_id;
    bpf_get_current_comm(&event->comm, TASK_COMM_LEN);
    
    // 计算窗口实际运行时间
    u64 window_duration = ts - data->window_start;
    
    event->oncpu_time = data->window_oncpu;
    event->offcpu_time = window_duration > data->window_oncpu ? 
                        window_duration - data->window_oncpu : 0;
    
    // 安全的利用率计算，确保不超过100%
    if (window_duration > 0) {
        u64 util_raw = (data->window_oncpu * 10000) / window_duration;
        event->utilization = util_raw > 10000 ? 9999 : util_raw;
    } else {
        event->utilization = 0;
    }
    
    event->timestamp = ts / 1000;  // 转换为微秒
    bpf_ringbuf_submit(event, 0);
}

SEC("tp_btf/sched_switch")
int BPF_PROG(sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next)
{
    u64 ts = bpf_ktime_get_ns();
    u32 cpu_id = bpf_get_smp_processor_id();

    u32 prev_pid = BPF_CORE_READ(prev, pid);
    u32 next_pid = BPF_CORE_READ(next, pid);

    // 处理离开CPU的进程
    if (should_track(prev_pid)) {
        struct task_data *data = bpf_map_lookup_elem(&task_stats, &prev_pid);
        if (!data) {
            struct task_data new_data = {};
            new_data.window_start = ts;
            new_data.last_cpu = cpu_id;
            new_data.is_oncpu = 0;  // 离开CPU
            new_data.last_switch_time = ts;
            new_data.window_oncpu = 0;
            bpf_map_update_elem(&task_stats, &prev_pid, &new_data, BPF_ANY);
        } else {
            // 如果当前在CPU上，累加这次的on-cpu时间
            if (data->is_oncpu && data->last_switch_time > 0) {
                u64 oncpu_delta = ts - data->last_switch_time;
                data->window_oncpu += oncpu_delta;
            }

            #ifdef INTERNAL
                if(ts % INTERVAL_NS) {
                    return 0; // 如果不在采样间隔内，则跳过
                }
            #endif

            send_cpu_event(prev_pid, data, ts, cpu_id, prev);
            
            // 检查是否需要重置窗口
            if (ts - data->window_start >= WINDOW_SIZE_NS) {
                
                
                // 重置窗口
                data->window_start = ts;
                data->window_oncpu = 0;
            }
            
            // 更新状态
            data->is_oncpu = 0;
            data->last_switch_time = ts;
            data->last_cpu = cpu_id;
            bpf_map_update_elem(&task_stats, &prev_pid, data, BPF_ANY);
        }
    }

    // 处理进入CPU的进程
    if (should_track(next_pid)) {
        struct task_data *data = bpf_map_lookup_elem(&task_stats, &next_pid);
        if (!data) {
            struct task_data new_data = {};
            new_data.window_start = ts;
            new_data.last_cpu = cpu_id;
            new_data.is_oncpu = 1;  // 进入CPU
            new_data.last_switch_time = ts;
            new_data.window_oncpu = 0;
            bpf_map_update_elem(&task_stats, &next_pid, &new_data, BPF_ANY);
        } else {
            // 更新状态
            data->is_oncpu = true;
            data->last_switch_time = ts;
            data->last_cpu = cpu_id;
            bpf_map_update_elem(&task_stats, &next_pid, data, BPF_ANY);
        }
    }

    return 0;
}

// SEC("tp_btf/sched_switch")
// int BPF_PROG(sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next)
// {
//     u64 ts = bpf_ktime_get_ns();
//     u32 cpu_id = bpf_get_smp_processor_id();  // 获取当前 CPU ID

// #ifdef INTERNAL
//     if(ts % INTERVAL_NS){
//         return 0; // 如果不在采样间隔内，则跳过
//     }
// #endif

//     u32 prev_pid = BPF_CORE_READ(prev, pid);
//     u32 next_pid = BPF_CORE_READ(next, pid);
    
//     char prev_comm[TASK_COMM_LEN];
//     char next_comm[TASK_COMM_LEN];
//     bpf_get_current_comm(&prev_comm, sizeof(prev_comm));
//     bpf_get_current_comm(&next_comm, sizeof(next_comm));

//     // 处理离开 CPU 的任务 (prev)
//     if (should_track(prev_pid)) {
//         struct task_data *data = bpf_map_lookup_elem(&task_stats, &prev_pid);
//         struct task_data new_data = {.window_start = ts, .last_cpu = cpu_id};
        
//         if (!data) {
//             data = &new_data;
//         }

//        // 更新 ON CPU 时间
//         if (data->last_oncpu > 0) {
//             u64 delta = ts - data->last_oncpu;
//             if (data->is_oncpu && data->last_switch_time > 0) {
//                 u64 oncpu_delta = ts - data->last_switch_time;
//                 data->window_oncpu += oncpu_delta;
//             }
            
//             // 准备事件数据
//             struct cpu_event *event = bpf_ringbuf_reserve(&cpu_events, sizeof(*event), 0);
//             if (event) {
//                 event->pid = prev_pid;
//                 event->ppid = BPF_CORE_READ(prev, real_parent, pid);
//                 event->cpu_id = cpu_id;
//                 bpf_get_current_comm(&event->comm, TASK_COMM_LEN);
//                 event->oncpu_time = data->window_oncpu;
//                 event->offcpu_time = data->window_total - data->window_oncpu;

//                 // 使用窗口计算利用率，防溢出
//                 event->utilization = safe_div(data->window_oncpu, data->window_total);
//                 event->timestamp = ts / 1000;  // 转换为微秒
//                 bpf_ringbuf_submit(event, 0);
//             }
//             update_window_stats(data, ts, delta, cpu_id);

//         }

//         // 记录 OFF CPU 开始时间
//         data->last_offcpu = ts;
//         data->last_oncpu = 0;
//         data->last_cpu = cpu_id;
//         bpf_map_update_elem(&task_stats, &prev_pid, data, BPF_ANY);
//     }

//     // 处理进入 CPU 的任务 (next)
//     if (should_track(next_pid)) {
//         struct task_data *data = bpf_map_lookup_elem(&task_stats, &next_pid);
//         struct task_data new_data = {.window_start = ts, .last_cpu = cpu_id};
//         if (!data) {
//             data = &new_data;
//         }

//         // 记录 ON CPU 开始时间
//         data->last_oncpu = ts;
//         data->last_offcpu = 0;
//         data->last_cpu = cpu_id;
//         bpf_map_update_elem(&task_stats, &next_pid, data, BPF_ANY);
//     }


//     return 0;
// }

SEC("tp_btf/sched_process_exit")
int BPF_PROG(sched_process_exit, struct task_struct *task)
{
    u32 pid = BPF_CORE_READ(task, pid);
    u64 ts = bpf_ktime_get_ns();
    u32 cpu_id = bpf_get_smp_processor_id();  // 获取当前 CPU ID

// #ifdef INTERNAL
//     if(ts % INTERVAL_NS){
//         return 0; // 如果不在采样间隔内，则跳过
//     }
// #endif

//     // 检查是否需要忽略线程
//     char comm[TASK_COMM_LEN];
//     bpf_get_current_comm(&comm, sizeof(comm));
//     // if (should_ignore_thread(comm)) {
//     //     return 0;  // 忽略这些线程
//     // }

//     if (!should_track(pid))
//         return 0;

//     // 1. 发送最终统计事件
//     struct task_data *data = bpf_map_lookup_elem(&task_stats, &pid);
//     if (data) {
//         // 更新 ON CPU 时间
//         if (data->last_oncpu > 0) {
//             u64 delta = ts - data->last_oncpu;
//             update_window_stats(data, ts, delta, cpu_id);

//             // 准备事件数据
//             struct cpu_event *event = bpf_ringbuf_reserve(&cpu_events, sizeof(*event), 0);
//             if (event) {
//                 event->pid =pid;
//                 event->ppid = BPF_CORE_READ(prev, real_parent, pid);
//                 event->cpu_id = cpu_id;
//                 bpf_get_current_comm(&event->comm, TASK_COMM_LEN);
//                 event->oncpu_time = data->window_oncpu;
//                 event->offcpu_time = data->window_total - data->window_oncpu;
                
//                 // 计算在当前CPU上的时间
//                 event->on_this_cpu_time = (cpu_id < MAX_CPUS) ? data->per_cpu_time[cpu_id] : 0;

//                 // 使用窗口计算利用率，防溢出
//                 event->utilization = safe_div(data->window_oncpu, data->window_total);
//                 event->timestamp = ts / 1000;  // 转换为微秒
//                 bpf_ringbuf_submit(event, 0);
//             }
//     }

    // 2. 清理map中的旧数据
    bpf_map_delete_elem(&task_stats, &pid);
    return 0;
}

char _license[] SEC("license") = "GPL";