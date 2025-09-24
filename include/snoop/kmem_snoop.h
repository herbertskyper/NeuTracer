#ifndef KMEM_SNOOP_H
#define KMEM_SNOOP_H

#pragma once

// 标准库
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <string>
#include <thread>
#include <vector>

// 系统库
#include <bpf/libbpf.h>

// 项目头文件
#include "config.h"
#include "kmem_snoop.skel.h"
#include "utils/Logger.h"
#include "utils/SymUtils.h"
#include "utils/UprobeProfiler.h"

#define TASK_COMM_LEN 16
#define PERF_MAX_STACK_DEPTH 127

namespace NeuTracer {

/**
 * @brief 内存事件结构体，与BPF端匹配
 */
struct kmem_event {
    uint32_t pid;          ///< 线程ID
    uint32_t tgid;         ///< 进程ID
    uint64_t size;         ///< 分配/释放的大小
    uint64_t addr;         ///< 内存地址
    uint64_t timestamp_ns; ///< 时间戳(纳秒)
    uint32_t stack_id;     ///< 堆栈ID
    uint32_t event_type;   ///< 事件类型: 0=分配, 1=释放
    char comm[TASK_COMM_LEN]; ///< 进程名
};

/**
 * @brief 分配信息结构体
 */
struct alloc_info {
    uint64_t size;         ///< 分配大小
    uint64_t timestamp_ns; ///< 分配时间戳
    uint32_t stack_id;     ///< 堆栈ID
};

/**
 * @brief 聚合分配信息结构体
 */
struct combined_alloc_info {
    uint64_t total_size;        ///< 总分配大小
    uint64_t number_of_allocs;  ///< 分配次数
};



/**
 * @brief 内存统计和异常检测结构体
 */
struct MemStats {
    uint32_t pid{0};            ///< 线程ID
    uint32_t tgid{0};           ///< 进程ID
    std::string comm;           ///< 进程名
    
    // 基础统计
    uint64_t total_allocs{0};   ///< 总分配次数
    uint64_t total_frees{0};    ///< 总释放次数
    uint64_t current_memory{0}; ///< 当前分配内存大小
    uint64_t peak_memory{0};    ///< 峰值内存使用
    
    // 内存操作历史
    std::deque<uint64_t> alloc_sizes;        ///< 分配大小历史
    std::deque<uint64_t> mem_usage_samples;  ///< 内存使用样本历史
    std::map<uint64_t, uint64_t> addr_to_size; ///< 地址到大小映射
    
    // 异常检测指标
    uint32_t churn_score{0};        ///< 内存周转分数

    std::chrono::steady_clock::time_point window_start;  ///< 窗口开始时间

};

/**
 * @class KmemSnoop
 * @brief 内核内存监控类，用于跟踪和分析进程的内存分配行为
 * 
 * 该类使用eBPF监控内存分配/释放操作，收集统计数据并检测异常模式，
 * 如内存泄漏、碎片化和不规则的分配模式等。
 */
class KmemSnoop {
public:
    /**
     * @brief 构造函数
     * 
     * @param env 环境配置
     * @param logger 日志管理器
     * @param profiler 性能分析器
     */
    //KmemSnoop(const myenv &env, Logger& logger, UprobeProfiler& profiler);
    KmemSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
        : env_(env), logger_(logger), profiler_(profiler) {
        lastActivityTime_ = std::chrono::steady_clock::now();
    }

    /**
     * @brief 析构函数
     */
    //~KmemSnoop();

    /**
     * @brief 附加BPF程序
     * @return 是否成功附加
     */
    bool attach_bpf();
    
    /**
     * @brief 停止跟踪
     */
    void stop_trace();
    
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
    // BPF对象
    struct kmem_snoop_bpf* skel_{nullptr};       ///< BPF骨架
    std::vector<struct bpf_link*> links_;        ///< BPF链接
    struct ring_buffer* rb_{nullptr};            ///< Ring buffer
    std::atomic<bool> exiting_{false};           ///< 退出标志
    
    // 活动跟踪
    std::chrono::steady_clock::time_point lastActivityTime_; ///< 最后活动时间
    int idleTimeoutSec_{3};                      ///< 空闲超时时间(秒)
    
    // 工具实例
    Logger& logger_;                             ///< 日志管理器
    UprobeProfiler& profiler_;                   ///< 性能分析器
    myenv env_;                                  ///< 环境配置
    
    // 全局内存统计
    uint64_t total_allocs_{0};                   ///< 总分配
    uint64_t total_frees_{0};                    ///< 总释放
    uint64_t current_memory_{0};                 ///< 当前已分配内存
    uint64_t peak_memory_{0};                    ///< 峰值内存使用
    
    std::map<uint32_t, MemStats> mem_stats_;     ///< 按进程分类的内存统计

    uint64_t event_count_{0};                    ///< 事件计数
    int64_t last_sample_time_{0};            ///< 上次报告时间(纳秒)
    int64_t last_warning_time_{0};           ///< 上次警告时间(纳秒)
    int64_t last_alloc_size{0};
    int64_t last_free_size{0}; ///< 上次分配/释放大小

    
    /**
     * @brief 获取当前时间戳字符串
     * @return 格式化的时间戳
     */
    std::string get_timestamp();

    /**
     * @brief libbpf打印回调函数
     */
    static int libbpf_print_fn(
        enum libbpf_print_level level,
        const char* format,
        va_list args);

    /**
     * @brief 事件处理回调函数
     */
    static int handle_event(void *ctx, void *data, size_t data_sz);
    
    /**
     * @brief 处理单个事件
     * 
     * @param data 事件数据
     * @param data_sz 数据大小
     * @return 处理结果
     */
    int process_event(void *data, size_t data_sz);
    
    /**
     * @brief Ring buffer处理线程函数
     */
    void ring_buffer_thread();
    
    /**
     * @brief 启动跟踪
     * @return 是否成功启动
     */
    bool start_trace();
    
    /**
     * @brief 格式化内存大小为人类可读形式
     * 
     * @param size 字节大小
     * @return 格式化后的字符串
     */
    std::string formatMemorySize(uint64_t size);
    
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
     * @brief 更新内存统计数据
     * 
     * @param pid 进程ID
     * @param e 事件数据
     * @param timestamp 时间戳
     */
    void update_mem_stats(uint32_t pid, const kmem_event* e, std::string timestamp);
    
    void detect_mem_anomalies(uint32_t pid, std::string timestamp);    

    void report_mem();
};

} // namespace NeuTracer

#endif // KMEM_SNOOP_H