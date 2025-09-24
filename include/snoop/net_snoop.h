#ifndef NET_SNOOP_H
#define NET_SNOOP_H

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
#include "config.h"
#include "net_snoop.skel.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"

#define TASK_COMM_LEN 16

namespace NeuTracer {

/**
 * @brief 网络事件结构体，与BPF端匹配
 */
struct net_event {
    uint32_t tgid;             
    uint32_t pid;              
    uint64_t bytes;            ///< 传输字节数
    uint64_t timestamp;        ///< 时间戳
    bool is_send;              ///< 传输方向: true=发送, false=接收
    char comm[TASK_COMM_LEN];  ///< 进程名
    uint32_t saddr;            ///< 源IP地址
    uint32_t daddr;            ///< 目标IP地址
    uint16_t sport;            ///< 源端口
    uint16_t dport;            ///< 目标端口
    uint8_t protocol;          ///< 协议类型
};


/**
 * @brief 网络统计和异常检测结构体
 */
struct NetStats {
    uint32_t pid{0};                  ///< 进程ID
    uint32_t tgid{0};                 ///< 线程组ID
    std::string comm;                 ///< 进程名
    
    // 基础统计
    uint64_t tx_bytes{0};             ///< 总发送字节数
    uint64_t rx_bytes{0};             ///< 总接收字节数
    uint64_t tx_packets{0};           ///< 总发送包数
    uint64_t rx_packets{0};           ///< 总接收包数
    
    // 流量历史(用于计算速率和检测突发)
    std::deque<uint64_t> tx_bytes_history;  ///< 发送字节历史
    std::deque<uint64_t> rx_bytes_history;  ///< 接收字节历史
    std::deque<double> tx_timestamps;       ///< 发送时间戳
    std::deque<double> rx_timestamps;       ///< 接收时间戳
    
    // 连接信息
    std::set<std::string> connections;      ///< 活跃连接集合
    std::set<uint16_t> listening_ports;     ///< 监听端口集合
    
    // 异常检测指标
    // uint32_t retransmits{0};                ///< TCP重传计数
    // uint32_t errors{0};                     ///< 错误计数
    double avg_latency_ms{0.0};             ///< 平均延迟(毫秒)

    std::chrono::steady_clock::time_point window_start;  ///< 窗口开始时间
    
    /**
     * @brief 默认构造函数
     */
    NetStats() = default;
};

/**
 * @class NetSnoop
 * @brief 网络监控类，用于跟踪和分析进程的网络行为
 * 
 * 该类使用eBPF监控网络活动，收集发送/接收字节数、连接信息等数据，
 * 并提供异常检测功能，如识别网络使用模式异常、突发流量等。
 */
class NetSnoop {
public:
    /**
     * @brief 构造函数
     * 
     * @param env 环境配置
     * @param logger 日志管理器
     * @param profiler 性能分析器
     */
    NetSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler);
    
    /**
     * @brief 析构函数
     */
    ~NetSnoop() = default;

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
    
    /**
     * @brief 目标进程名称集合
     */
    std::set<std::string> target_process_names;

private:
    std::chrono::steady_clock::time_point lastActivityTime_;  ///< 最后活动时间
    int idleTimeoutSec_{3};                 ///< 空闲超时时间(秒)
    
    myenv env_;                             ///< 环境配置
    struct net_snoop_bpf *skel_{nullptr};   ///< BPF骨架
    std::vector<struct bpf_link *> links_;  ///< BPF链接
    Logger &logger_;                        ///< 日志管理器
    UprobeProfiler &profiler_;              ///< 性能分析器
    struct ring_buffer *rb_{nullptr};       ///< Ring buffer
    std::atomic<bool> exiting_{false};      ///< 退出标志

    // 跟踪累计统计
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> traffic_stats_;  ///< pid -> (发送字节数, 接收字节数)
    
    // 网络统计数据
    std::map<uint32_t, NetStats> net_stats_;  ///< 进程网络统计
    uint64_t event_count_{0};                 ///< 事件计数器
    int64_t last_warning_time_{0};          ///< 上次处理时间戳
    int64_t last_sample_time_{0};          ///< 上次采样时间戳
    int64_t last_send_bytes_{0};         ///< 上次发送字节数
    int64_t last_receive_bytes_{0};     ///< 上次接收字节数
    

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
     * @param data 事件数据
     * @param data_sz 数据大小
     * @return 处理结果
     */
    int process_event(void *data, size_t data_sz);
    
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
     * @brief 更新网络统计数据
     * 
     * @param pid 进程ID
     * @param e 事件数据
     * @param timestamp 时间戳
     */
    void update_net_stats(uint32_t pid, const net_event* e, std::string timestamp);
    
   
    
    /**
     * @brief 检测网络异常
     * 
     * @param pid 进程ID
     * @param timestamp 时间戳
     */
    void detect_net_anomalies(uint32_t pid, std::string timestamp);
    
    /**
     * @brief 报告网络异常情况
     */
    void report_net();

    // 辅助方法
    /**
     * @brief 格式化IP地址
     * 
     * @param addr 32位IP地址
     * @return 格式化的IP地址字符串
     */
    std::string format_ip_address(uint32_t addr);
    
    /**
     * @brief 格式化协议类型
     * 
     * @param proto 协议号
     * @return 协议名称
     */
    std::string format_protocol(uint8_t proto);
    
    /**
     * @brief 格式化连接标识键
     * 
     * @param e 网络事件
     * @return 连接标识字符串
     */
    std::string format_connection_key(const net_event* e);
    
    /**
     * @brief 格式化吞吐量
     * 
     * @param bytes_per_sec 每秒字节数
     * @return 格式化的吞吐量字符串
     */
    std::string format_throughput(uint64_t bytes_per_sec);

};

/**
 * @brief NetSnoop类的构造函数实现
 */
inline NetSnoop::NetSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
    : env_(env), logger_(logger), profiler_(profiler) {
    lastActivityTime_ = std::chrono::steady_clock::now();
}

/**
 * @brief NetSnoop类的析构函数实现
 */
// inline NetSnoop::~NetSnoop() {
//     stop_trace();
// }
     

} // namespace NeuTracer

#endif // NET_SNOOP_H