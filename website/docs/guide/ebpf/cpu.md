# CPU模块 eBPF 部分

`CPUsnoop.bpf.c` 用于监控 Linux 内核的进程调度事件，通过 BTF (BPF Type Format) 增强型跟踪点，捕获调度器事件，提供细粒度的 CPU 使用情况分析。模块可以计算每个进程的 CPU 运行时间与等待时间，同时捕获进程上下文切换事件，分析调度行为。此外，该模块还能跟踪进程在不同 CPU 核心间的迁移，从创建到退出进行全周期监控。

该程序挂载到两个关键的内核调度跟踪点来实现完整的进程监控。`sched_switch` 跟踪点在 CPU 从一个进程切换到另一个进程时触发，而 `sched_process_exit` 跟踪点在进程退出时触发。
```cpp
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
            // ......
            // 初始化新进程数据
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
            // ......
            // 初始化新进程数据
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
```
cpu_event记录进程 CPU 使用情况的完整信息，包括进程 ID、父进程 ID、CPU 核心 ID、CPU 运行时间、等待时间、利用率、时间戳和进程名称。task_data则跟踪每个进程的 CPU 时间累计，包含上次在 CPU 上运行的开始时间、上次离开 CPU 的时间、累计 CPU 运行时间和累计 CPU 等待时间。
```c
struct cpu_event {
    u32 pid;           // 进程 ID
    u32 ppid;          // 父进程 ID
    u32 cpu_id;        // CPU 核心 ID
    u64 oncpu_time;    // CPU 运行时间（纳秒）
    u64 offcpu_time;   // CPU 等待时间（纳秒）
    u64 utilization;   // 利用率（0-10000，对应 0-100.00%）
    u64 timestamp;     // 时间戳（微秒）
    char comm[TASK_COMM_LEN]; // 进程名称
};

struct task_data {
    u64 last_oncpu;
    u64 last_offcpu;
    u64 total_oncpu;
    u64 total_offcpu;
};
```
