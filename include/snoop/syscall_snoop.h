#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <bpf/libbpf.h>
#include <unistd.h>
#include "config.h"
#include "syscall_snoop.skel.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"

#define TASK_COMM_LEN 16

namespace NeuTracer {

enum event_type {
    EVENT_SYSCALL_ENTER = 0,
    EVENT_SYSCALL_EXIT = 1,
};

// 支持参数个数字段的系统调用事件结构
struct syscall_event {
    event_type type;
    uint32_t pid;         // 线程 ID
    uint32_t tgid;        // 进程 ID
    uint32_t syscall_id;  // 系统调用号
    uint64_t call_id;      // 调用 ID，用于跟踪同一系统调用的多次调用
    char comm[TASK_COMM_LEN];
    uint64_t timestamp;   // 时间戳
    uint64_t args[6];     // 最多6个参数
    uint64_t ret_val;     // 返回值
};

struct call_id_key_t {
    uint64_t pid_tgid;
    uint32_t syscall_id;
};

struct SyscallStats {
    uint32_t pid;
    uint32_t tgid;
    std::map<uint32_t, uint64_t> avg_syscall_time;  // 系统调用耗时统计
    std::map<uint32_t, uint64_t> syscall_count;  // 调用次数
};

class SyscallSnoop {
public:
    SyscallSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler);
    ~SyscallSnoop() = default;

    bool attach_bpf();
    void stop_trace();
    void setIdleTimeout(int seconds) { idleTimeoutSec_ = seconds; }
    void ring_buffer_thread();
    int process_event(void *data, size_t data_sz);

    // 参数个数表，syscall_id -> 参数个数
    static uint32_t get_syscall_arg_count(uint32_t syscall_id);
    void clear_call_id_map();
    std::string format_args(const uint64_t* args, uint32_t arg_count);

    std::thread rb_thread_;

private:
    std::chrono::steady_clock::time_point lastActivityTime_;
    int idleTimeoutSec_{3};
    myenv env_;
    struct syscall_snoop_bpf *skel_{nullptr};
    std::vector<struct bpf_link *> links_;
    Logger &logger_;
    UprobeProfiler &profiler_;
    struct ring_buffer *rb_{nullptr};
    std::atomic<bool> exiting_{false};
    int64_t last_sample_time_{0};
     std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint64_t>, uint64_t> syscall_enter_ts;
    std::map<uint32_t,SyscallStats>syscall_stats_;

    static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args);
    static int handle_event(void *ctx, void *data, size_t data_sz);
    
};


} // namespace NeuTracer