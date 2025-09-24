// include/config.h
#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#define END 1000

namespace NeuTracer {
    // 常量定义

    const std::string UPROBE_CFG_PATH = "../config/uprobe_cfg.json";
    const std::string LOG_BASE_PATH = "../log/";  


    struct myenv {
        bool verbose = false;
        pid_t pid = 0;
        size_t rb_count = 0;
        size_t duration_sec = 0;
        bool args = false;
        bool stacks = false;
        std::string server_addr = "localhost:50051"; 
        bool grpc_enabled = false;
        struct {
            bool gpu = true;  
            bool cpu = true;  
            bool kmem = true; 
            bool net = true;   
            bool io = true;   
            bool func = true;  
            bool python = true; 
            bool syscall = true;
            bool pcie = true;  ///< PCIe监控
            bool nvlink = true;  ///< NVLink监控
        } trace_modules;
        uint32_t syscall_id[100] = {0,1,2,3,4,5,6,END};
    };

    const uint64_t CLEAN_TIME_MIN = 60;  ///< 清理时间间隔 (秒)
    const uint64_t SAMPLE_INTERVAL_NS = 10000; 
    const uint64_t WARN_INTERVAL = 1000000000;  ///< 警告间隔 (1秒)

    const uint64_t IO_latency_threshold_us_ = 100000;  ///< 延迟阈值(100ms)
    const int IO_max_history_samples_ = 100;  ///< 最大历史样本数
    const int IO_stats_report_interval_ = 20;  ///< 统计报告间隔
    const int IO_large_size_ = 10;  ///< 超大IO阈值 (MB)

    const int CPU_stats_report_interval_ = 1000;  ///< 统计报告间隔
     const int CPU_max_history_samples_ = 1000;  ///< 历史样本最大数量
     const int CPU_NUMA_THRESHOLD = 10;  ///< 跨NUMA迁移阈值
     const double CPU_UTIL_THREHOLD = 0.5;  ///< CPU利用率突发阈值 (50%), 超过这个值同时超过平均值两倍则认为有利用率徒增
    const double CPU_CONTEXT_CHANGE = 0.5;  ///< 上下文切换率阈值 (50%)
    const double CPU_UTIL_LOW_THRESHOLD = 0.05;  ///< CPU利用率过低阈值 (5%)
    const double CPU_HOTSPOT_THRESHOLD = 0.8;  ///< 热点CPU阈值 (80%)

    const int MEM_large_alloc_threshold_ = 1024 * 1024 * 500; ///< 大型分配阈值(500MB)
    const int MEM_stats_report_interval_ = 10000; ///< 统计报告间隔
    const double MEM_variation_threshold_ = 0.8; ///< 内存波动阈值 (80%)
    const double MEM_churn_threshold_ = 0.9; ///< 内存周转
    const int MEM_max_history_samples_ = 100 ; ///< 最大历史样本数量

    const int NET_stats_report_interval_ = 10000; ///< 统计报告间隔
     const int NET_max_history_samples_ = 100 ; ///< 最大历史样本数量
      const double NET_SEND_variation_threshold_ = 0.8;
       const double NET_RECV_variation_threshold_ = 0.8;
    const int NET_SEND_large_size_ = 10 * 1024 * 1024; ///< 大型发送阈值 (10MB)
    const int NET_RECV_large_size_ = 10 * 1024 * 1024; ///< 大型接收阈值 (10MB)
    const int NET_CONN_threshold_ = 100; ///< 连接数异常阈值 (100)
    }
#endif // CONFIG_H

#ifndef CONFIG_REPORT
#define CONFIG_REPORT

#define REPORT_IO

#define REPORT_KMEM

#define REPORT_CPU

#define REPORT_NET


#endif // CONFIG_REPORT


