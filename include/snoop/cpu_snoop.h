#ifndef CPU_SNOOP_H
#define CPU_SNOOP_H

#pragma once

// 标准库头文件
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <deque>

// 系统库头文件
#include <argp.h>
#include <bpf/libbpf.h>
#include <unistd.h>

// 项目头文件
#include "config.h"
#include "utils/GetPid.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"
#include "cpu_snoop.skel.h"

#define TASK_COMM_LEN 16

namespace NeuTracer {

/**
 * @brief CPU事件结构体，与BPF端保持一致
 */
struct cpu_event {
    uint32_t pid;           ///< 进程ID
    uint32_t ppid;          ///< 父进程ID
    uint32_t cpu_id;        ///< CPU核心ID
    uint64_t oncpu_time;    ///< CPU运行时间
    uint64_t offcpu_time;   ///< CPU空闲时间
    uint64_t utilization;   ///< CPU利用率
    uint64_t timestamp;     ///< 时间戳
    char comm[TASK_COMM_LEN];  ///< 进程名
};

/**
 * @brief CPU迁移记录结构体
 */
struct CPUMigration {
    std::string timestamp;  ///< 发生时间
    uint32_t from_cpu;      ///< 源CPU核心ID
    uint32_t to_cpu;        ///< 目标CPU核心ID
};

/**
 * @brief CPU统计结构体
 */
struct CPUStats {
    uint64_t total_oncpu_time{0};     ///< 总计CPU运行时间 (微秒)
    uint64_t total_offcpu_time{0};    ///< 总计CPU空闲时间 (微秒)
    uint64_t call_count{0};           ///< 记录次数
    double avg_utilization{0.0};      ///< 平均利用率 (0-100%)
    
    // 异常检测指标
    uint32_t migrations_count{0};     ///< CPU迁移总次数
    uint32_t numa_migrations{0};      ///< 跨NUMA节点迁移次数

    uint32_t hotspot_cpu{0};          ///< 最常使用的CPU核心ID
    double hotspot_percentage{0.0};   ///< 在热点CPU上执行的时间百分比
    // 记录不同CPU核心的时间分布
    std::unordered_map<uint32_t, uint64_t> cpu_distribution;
    
    // CPU迁移记录
    std::vector<CPUMigration> migrations;
    
    // 利用率历史样本
    std::deque<uint64_t> utilization_samples;

    std::chrono::steady_clock::time_point window_start;  ///< 窗口开始时间
};

/**
 * @class CPUsnoop
 * @brief CPU监控类，用于跟踪和分析进程的CPU使用情况
 * 
 * 该类使用eBPF收集进程的CPU活动数据，包括CPU时间、迁移和利用率变化。
 * 它提供了检测CPU使用异常（如资源抢占、不合理的CPU迁移）的能力。
 */
class CPUsnoop {
public:
    /**
     * @brief 构造函数
     * 
     * @param env 环境配置
     * @param logger 日志管理器
     * @param profiler 性能分析器
     */
    //CPUsnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler);
    CPUsnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
        : env_(env), logger_(logger), profiler_(profiler) {
        lastActivityTime_ = std::chrono::steady_clock::now();
    }

    /**
     * @brief 析构函数
     */
    ~CPUsnoop() = default;

    /**
     * @brief 初始化BPF程序
     */
    void init_bpf();
    
    /**
     * @brief 停止跟踪
     */
    void stop_trace();
    
    /**
     * @brief 附加BPF程序
     * @return 是否成功附加
     */
    bool attach_bpf();
    
    /**
     * @brief 记录统计信息
     * 
     * @param os 输出流
     * @param cur_time 当前时间
     * @param period 统计周期
     * @param snoop_pid 要监控的进程ID
     */
    void record_stats(std::ostream& os, double cur_time, double period, uint32_t snoop_pid);
    
    /**
     * @brief 设置空闲超时时间
     * 
     * @param seconds 超时秒数
     */
    void setIdleTimeout(int seconds) { idleTimeoutSec_ = seconds; }
    
    /**
     * @brief Ring buffer处理线程
     */
    std::thread rb_thread_;

private:
    std::chrono::steady_clock::time_point lastActivityTime_;  ///< 最后活动时间
    int idleTimeoutSec_{3};  ///< 空闲超时时间(秒)

    myenv env_;  ///< 环境配置
    struct cpu_snoop_bpf *skel_{nullptr};  ///< BPF骨架
    std::vector<struct bpf_link *> links_;  ///< BPF链接
    Logger &logger_;  ///< 日志管理器
    UprobeProfiler &profiler_;  ///< 性能分析器
    struct ring_buffer *rb_{nullptr};  ///< ring buffer
    std::atomic<bool> exiting_{false};  ///< 退出标志

    long clock_ticks{sysconf(_SC_CLK_TCK)};  ///< 系统时钟周期
    std::map<uint32_t, std::map<std::string, uint64_t>> old_usage;  ///< 旧的使用情况

    // CPU统计数据
    std::unordered_map<uint32_t, CPUStats> cpu_stats_;  ///< 进程CPU统计
    std::unordered_map<uint32_t, uint32_t> last_cpu_;  ///< 记录每个进程最后运行的CPU
    size_t event_count_{0};  ///< 事件计数器
    int64_t last_sample_time_{0};  ///< 上次报告时间
    int64_t last_warning_time_{0};  ///< 上次报告时间


    static const int64_t RINGBUF_MAX_ENTRIES = 64 * 1024 * 1024;
    static const int CPU_SMAPLE_TIME = 1000;  ///< CPU采样时间间隔 (毫秒)


    /**
     * @brief libbpf打印回调函数
     */
    static int libbpf_print_fn(
        enum libbpf_print_level level,
        const char *format,
        va_list args);

    /**
     * @brief 事件处理回调函数
     */
    static int handle_event(void *ctx, void *data, size_t data_sz);
    
    /**
     * @brief Ring buffer处理线程函数
     */
    void ring_buffer_thread();
    
    /**
     * @brief 处理单个事件
     * 
     * @param ctx 上下文
     * @param data 事件数据
     * @param data_sz 数据大小
     * @return 处理结果
     */
    int process_event(void *ctx, void *data, size_t data_sz);
    
    /**
     * @brief 启动跟踪
     * @return 是否成功启动
     */
    bool start_trace();
    
    /**
     * @brief 检查是否超出分析时限
     * 
     * @param duration 持续时间
     * @param startTime 开始时间
     * @return 是否超时
     */
    bool hasExceededProfilingLimit(
        std::chrono::seconds duration,
        const std::chrono::steady_clock::time_point& startTime);
    
    /**
     * @brief 获取当前时间戳字符串
     * @return 格式化的时间戳
     */
    std::string get_timestamp();

    /**
     * @brief 更新CPU统计数据
     * 
     * @param pid 进程ID
     * @param e 事件数据
     * @param timestamp 时间戳
     */
    void update_cpu_stats(uint32_t pid, const cpu_event* e, std::string timestamp);
    
    /**
     * @brief 报告CPU情况
     */
    void report_cpu();
    
    /**
     * @brief 检查是否为跨NUMA节点迁移
     * 
     * @param cpu1 第一个CPU ID
     * @param cpu2 第二个CPU ID
     * @return 是否跨NUMA
     */
    bool is_cross_numa(uint32_t cpu1, uint32_t cpu2);
    
    /**
     * @brief 更新热点CPU信息
     * 
     * @param pid 进程ID
     * @param timestamp 时间戳
     */
    void update_hotspot_cpu(uint32_t pid, std::string timestamp);
    
};

} // namespace NeuTracer

#endif // CPU_SNOOP_H