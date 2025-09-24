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
#include "pcie_snoop.skel.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"

#define TASK_COMM_LEN 16

namespace NeuTracer {

enum pcie_event_type {
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

enum dma_fence_event_type {
    DMA_FENCE_INIT = 0,
    DMA_FENCE_DESTROY = 1,
    DMA_FENCE_ENABLE_SIGNAL = 2,
    DMA_FENCE_SIGNALED = 3,
    DMA_FENCE_WAIT_START = 4,
    DMA_FENCE_WAIT_END = 5,
    DMA_FENCE_EMIT = 6
};

// PCIe 访问事件结构
struct pcie_event {
    pcie_event_type type;
    pcie_op_type op_type;
    uint32_t pid;        // 线程 ID
    uint32_t tgid;       // 进程 ID
    char comm[TASK_COMM_LEN];
    uint64_t timestamp;  // 时间戳
    
    // PCIe 访问参数
    uint32_t bus;
    uint32_t devfn;
    uint32_t offset;
    uint32_t size;
    uint32_t value;     // 写入值或读取结果
};

struct dma_fence_event {
    dma_fence_event_type type;
    uint32_t pid;        // 线程 ID
    uint32_t tgid;       // 进程 ID
    char comm[TASK_COMM_LEN];
    uint64_t timestamp;  // 时间戳
    
    // DMA Fence 信息
    uint64_t context;        // fence context
    uint32_t seqno;          // fence 序列号
    char driver_name[32];    // 驱动程序名称
    char timeline_name[32];  // timeline 名称
    uint64_t duration_ns;    // 仅用于 wait_end 事件，表示等待时间
};

struct PcieStats {
    uint32_t pid;
    uint32_t tgid;
    std::map<uint32_t, uint64_t> avg_pcie_time;  // PCIe操作耗时统计
    std::map<uint32_t, uint64_t> pcie_count;     // 调用次数

    std::map<dma_fence_event_type, uint64_t> fence_count;
    std::map<uint64_t, uint64_t> fence_wait_start_ts; // 记录等待开始时间
    uint64_t total_fence_wait_time{0}; // 总等待时间
    uint64_t max_fence_wait_time{0};   // 最大等待时间
};

class PcieSnoop {
public:
    PcieSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler);
    ~PcieSnoop() = default;

    bool attach_bpf();
    void stop_trace();
    void setIdleTimeout(int seconds) { idleTimeoutSec_ = seconds; }
    void ring_buffer_thread();
    int process_event(void *data, size_t data_sz);
    int process_dma_fence_event(void *data, size_t data_sz); 

    std::string op_type_to_string(pcie_op_type op);
    std::string dma_fence_type_to_string(dma_fence_event_type type);

    std::thread rb_thread_;

private:
    std::chrono::steady_clock::time_point lastActivityTime_;
    int idleTimeoutSec_{3};
    myenv env_;
    struct pcie_snoop_bpf *skel_{nullptr};
    std::vector<struct bpf_link *> links_;
    Logger &logger_;
    UprobeProfiler &profiler_;
    struct ring_buffer *rb_{nullptr};
    std::atomic<bool> exiting_{false};
    int64_t last_sample_time_{0};
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint64_t> pcie_enter_ts;
    std::map<uint32_t, PcieStats> pcie_stats_;

    static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args);
    static int handle_event(void *ctx, void *data, size_t data_sz);
};

} // namespace NeuTracer