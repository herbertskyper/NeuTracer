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
#include "nvlink_snoop.skel.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"

#define TASK_COMM_LEN 16

namespace NeuTracer {

enum nvlink_event_type {
    EVENT_NVLINK_ENTER = 0,
    EVENT_NVLINK_EXIT = 1,
};

enum nvlink_func_id {
    NVLINK_FUNC_STRCPY = 0,
    NVLINK_FUNC_MEMCPY = 1,
    NVLINK_FUNC_MEMSET = 2,
};

// NVLink事件结构
struct nvlink_event {
    nvlink_event_type type;
    nvlink_func_id func_id;
    uint32_t pid;
    uint32_t tgid;
    char comm[TASK_COMM_LEN];
    uint64_t timestamp;
    
    // 函数参数
    uint64_t dst_addr;     // 目标地址
    uint64_t src_addr;     // 源地址 (对于memset是值)
    uint64_t size;         // 大小/长度
    
    // 返回值
    uint64_t ret_val;
};

struct NVLinkStats {
    uint32_t pid;
    uint32_t tgid;
    std::map<nvlink_func_id, uint64_t> call_count;  // 函数调用次数
    std::map<nvlink_func_id, uint64_t> total_bytes; // 处理的总字节数
    std::map<nvlink_func_id, uint64_t> avg_time;    // 平均处理时间(ns)
};

class NVLinkSnoop {
public:
    NVLinkSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler);
    ~NVLinkSnoop() = default;

    bool attach_bpf();
    void stop_trace();
    void setIdleTimeout(int seconds) { idleTimeoutSec_ = seconds; }
    void ring_buffer_thread();
    int process_event(void *data, size_t data_sz);
    
    std::string get_func_name(nvlink_func_id func_id);

    std::thread rb_thread_;

private:
    std::chrono::steady_clock::time_point lastActivityTime_;
    int idleTimeoutSec_{3};
    myenv env_;
    struct nvlink_snoop_bpf *skel_{nullptr};
    std::vector<struct bpf_link *> links_;
    Logger &logger_;
    UprobeProfiler &profiler_;
    struct ring_buffer *rb_{nullptr};
    std::atomic<bool> exiting_{false};
    int64_t last_sample_time_{0};
    
    // 用于计算函数执行时间
    std::map<std::tuple<uint32_t, uint32_t, nvlink_func_id>, uint64_t> nvlink_enter_ts;
    std::map<uint32_t, NVLinkStats> nvlink_stats_;

    static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args);
    static int handle_event(void *ctx, void *data, size_t data_sz);
};

} // namespace NeuTracer