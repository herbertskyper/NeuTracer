#ifndef IO_SNOOP_H
#define IO_SNOOP_H

#pragma once

// 标准库头文件
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

// 系统库头文件
#include <bpf/libbpf.h>
#include <unistd.h>

// 项目头文件
#include "bio_snoop.skel.h"
#include "config.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"
#include "utils/Format.h"

#define TASK_COMM_LEN 16

namespace NeuTracer {


/**
 * @brief I/O事件结构，与BPF端匹配
 */
struct bio_event {
    uint32_t tgid;
    uint32_t pid;                
    char comm[TASK_COMM_LEN];    ///< 进程名
    uint64_t delta_us;           ///< 操作耗时(微秒)
    uint64_t bytes;              ///< 操作字节数
    uint8_t rwflag;              ///< 读写标志: 0=读, 1=写
    uint8_t major;               ///< 主设备号
    uint8_t minor;               ///< 次设备号
    uint64_t timestamp;          ///< 时间戳
    uint64_t io_count;           ///< 累计IO计数
};

/**
 * @brief 设备I/O统计结构体
 */
struct io_stat {
    uint64_t read_bytes{0};       ///< 读取的总字节数
    uint64_t write_bytes{0};      ///< 写入的总字节数
    uint64_t read_ops{0};         ///< 读操作次数
    uint64_t write_ops{0};        ///< 写操作次数
    uint64_t avg_read_latency_us{0};  ///< 读操作总延迟(微秒)
    uint64_t avg_write_latency_us{0}; ///< 写操作总延迟(微秒)
};

/**
 * @brief 设备标识符结构体
 */
struct device_key {
    uint8_t major;   ///< 主设备号
    uint8_t minor;   ///< 次设备号
    
    /**
     * @brief 比较运算符重载，用于在map中作为键
     * @param other 另一个设备标识符
     * @return 是否小于other
     */
    bool operator<(const device_key& other) const {
        if (major != other.major)
            return major < other.major;
        return minor < other.minor;
    }
};


/**
 * @brief IO统计和异常检测结构体
 */
struct IOStats {
    uint32_t pid{0};                         ///< 进程ID
    std::string comm;                        ///< 进程名称
    
    // 基础统计
    uint64_t total_bytes{0};                 ///< 总IO字节数
    uint64_t read_bytes{0};                  ///< 读取的总字节数
    uint64_t write_bytes{0};                 ///< 写入的总字节数
    uint64_t read_ops{0};                    ///< 读操作次数
    uint64_t write_ops{0};                   ///< 写操作次数
    uint64_t total_latency_us{0};            ///< 总IO延迟(微秒)
    
    // 计算指标
    double avg_latency_ms{0.0};              ///< 平均延迟(毫秒)
    double avg_read_latency_us{0.0};         ///< 平均读延迟(毫秒)
    double avg_write_latency_us{0.0};        ///< 平均写延迟(毫秒)
    uint64_t max_latency_us{0};              ///< 最大延迟(微秒)
    
    // 历史数据样本
    std::deque<uint64_t> latency_samples;    ///< 延迟历史样本
    std::deque<uint64_t> size_samples;       ///< IO大小历史样本
    
    // 设备访问历史
    std::map<std::string, uint64_t> device_access_count;  ///< 设备访问计数
    std::string last_device;                 ///< 上次访问的设备

    std::chrono::steady_clock::time_point window_start;  ///< 窗口开始时间
};

/**
 * @class IoSnoop
 * @brief IO监控类，用于跟踪和分析进程的IO行为
 * 
 * 该类使用eBPF监控系统的块IO操作，收集读写延迟、吞吐量等数据，
 * 并提供异常检测功能，如识别异常高延迟、不合理的IO模式等。
 */
class IoSnoop {
public:
    /**
     * @brief 构造函数
     * 
     * @param env 环境配置
     * @param logger 日志管理器
     * @param profiler 性能分析器
     */
    //IoSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler);
    IoSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
        : env_(env), logger_(logger), profiler_(profiler) {
        lastActivityTime_ = std::chrono::steady_clock::now();
    }

    /**
     * @brief 析构函数
     */
    ~IoSnoop() = default;

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
     * @brief 获取设备IO统计信息
     * @return 设备IO统计映射表
     */
    std::map<device_key, io_stat> get_device_stats() const { return device_stats_; }
    
    
    /**
     * @brief 获取设备列表
     * @return 设备键列表
     */
    std::vector<device_key> get_device_list() const;
    
    /**
     * @brief 获取设备名称
     * 
     * @param major 主设备号
     * @param minor 次设备号
     * @return 设备名称
     */
    std::string get_device_name(uint8_t major, uint8_t minor) const;

    /**
     * @brief Ring buffer处理线程
     */
    std::thread rb_thread_;
    
    // /**
    //  * @brief 目标进程名称集合
    //  */
    // std::set<std::string> target_process_names;

private:
    std::chrono::steady_clock::time_point lastActivityTime_;  ///< 最后活动时间
    int idleTimeoutSec_{3};  ///< 空闲超时时间(秒)
    
    myenv env_;  ///< 环境配置
    struct bio_snoop_bpf *skel_{nullptr};  ///< BPF骨架
    std::vector<struct bpf_link *> links_;  ///< BPF链接
    Logger &logger_;  ///< 日志管理器
    UprobeProfiler &profiler_;  ///< 性能分析器
    struct ring_buffer *rb_{nullptr};  ///< Ring buffer
    std::atomic<bool> exiting_{false};  ///< 退出标志

    // 统计信息
    std::map<device_key, io_stat> device_stats_;  ///< 设备IO统计 (device -> io_stat)
    std::map<device_key, std::string> device_names_;  ///< 设备名称缓存
    
    uint64_t event_count_{0};  ///< 事件计数器
    std::map<uint32_t, IOStats> io_process_stats_;  ///< 进程IO详细统计
    
    // 统一的异常检测间隔时间（避免重复输出）
    int64_t last_sample_time_{0};  ///< 最后一次采样时间 (纳秒)
    int64_t last_warning_time_{0};  ///< 最后一次警告时间 (纳秒)
    int64_t last_read_byte{0};
    int64_t last_write_byte{0};


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
     * @brief 从/proc/partitions获取设备名称
     */
    void load_device_names();
    
    /**
     * @brief 更新IO统计数据
     * 
     * @param pid 进程ID
     * @param e 事件数据
     * @param timestamp 时间戳
     */
    void update_io_stats(uint32_t pid, const bio_event* e, std::string timestamp);
    
    /**
     * @brief 检测IO延迟峰值
     * 
     * @param pid 进程ID
     * @param e 事件数据
     * @return 是否检测到延迟峰值
     */
    bool detect_io_latency_spike(uint32_t pid, const bio_event* e);
    void report_io();
};

} // namespace NeuTracer

#endif // IO_SNOOP_H