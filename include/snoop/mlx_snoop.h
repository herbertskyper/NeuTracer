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
#include "mlx_snoop.skel.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"

#define TASK_COMM_LEN 16

namespace NeuTracer {

enum mlx_event_type {
    EVENT_MLX_ENTER = 0,
    EVENT_MLX_EXIT = 1,
};

enum mlx_func_id {
    MLX_REG_DM_MR = 0,
    MLX_REG_USER_MR = 1,
    MLX_REG_USER_MR_DMABUF = 2,
    MLX_ALLOC_PD = 3,
    MLX_ALLOC_MR = 4,
    MLX_CREATE_CQ = 5,
    MLX_CREATE_QP = 6,
    MLX_CREATE_SRQ = 7,
};

struct mlx_event {
    mlx_event_type type;
    mlx_func_id func_id;
    uint32_t pid;
    uint32_t tgid;
    char comm[TASK_COMM_LEN];
    uint64_t timestamp;
    
    // 函数参数和返回值
    uint64_t arg1;
    uint64_t arg2;
    uint64_t arg3;
    uint64_t ret_val;
};

struct MLXStats {
    uint32_t pid;
    uint32_t tgid;
    std::map<mlx_func_id, uint64_t> call_count;    // 函数调用次数
};

class MLXSnoop {
public:
    MLXSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler);
    ~MLXSnoop() = default;

    bool attach_bpf();
    void stop_trace();
    void setIdleTimeout(int seconds) { idleTimeoutSec_ = seconds; }
    void ring_buffer_thread();
    int process_event(void *data, size_t data_sz);
    
    std::string get_func_name(mlx_func_id func_id);
    std::string get_func_desc(mlx_func_id func_id);

    std::thread rb_thread_;

private:
    std::chrono::steady_clock::time_point lastActivityTime_;
    int idleTimeoutSec_{3};
    myenv env_;
    struct mlx_snoop_bpf *skel_{nullptr};
    std::vector<struct bpf_link *> links_;
    Logger &logger_;
    UprobeProfiler &profiler_;
    struct ring_buffer *rb_{nullptr};
    std::atomic<bool> exiting_{false};
    int64_t last_sample_time_{0};
    
    // 性能统计相关
    std::map<uint32_t, MLXStats> mlx_stats_;

    static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args);
    static int handle_event(void *ctx, void *data, size_t data_sz);
};

} // namespace NeuTracer