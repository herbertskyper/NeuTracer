#ifndef func_snoop_H
#define func_snoop_H

#include <atomic>
#include <chrono>
#include <ctime>
#include <cxxabi.h>
#include <fcntl.h>
#include <gelf.h>
#include <iomanip>
#include <libelf.h>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

// 系统库
#include <bpf/libbpf.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <uuid/uuid.h>

// 第三方库
#include <nlohmann/json.hpp>

// 项目头文件
#include "config.h"
#include "func_snoop.skel.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"

// 前向声明
struct func_snoop_bpf;

namespace NeuTracer {

/**
 * @brief 使用nlohmann::json作为json类型
 */
using json = nlohmann::json;

/**
 * @brief 跟踪事件结构体
 */
struct func_trace_event {
    uint8_t type;       ///< 事件类型：0=入口，1=出口
    uint64_t cookie;    ///< 唯一标识符
    uint32_t pid;       ///< 线程ID
    uint32_t tgid;      ///< 进程ID
    uint64_t timestamp;        ///< 时间戳 (纳秒)
    char name[64];      ///< 函数名
    uint64_t args[6];   ///< 函数参数 (最多6个参数
    uint64_t ret_val;   ///< 返回值
};

/**
 * @class FuncSnoop
 * @brief BPF管理类，负责初始化、附加和管理BPF程序
 *
 * 该类使用libbpf骨架API管理BPF程序，提供函数附加、事件处理和统计功能。
 */
class FuncSnoop {
public:
    /**
     * @brief 函数调用统计结构
     */
    struct FunctionStats {
        uint64_t call_count = 0;            ///< 总调用次数
        uint64_t active_calls = 0;          ///< 当前活动调用数
        uint64_t total_duration_us = 0;     ///< 总耗时 (微秒)
        uint64_t max_duration_us = 0;       ///< 最长调用耗时
        uint64_t min_duration_us = 0;       ///< 最短调用耗时
        uint64_t avg_duration_us = 0;       ///< 平均调用耗时
        uint64_t slow_call_count = 0;       ///< 慢调用次数
        std::string first_call_time;        ///< 首次调用时间
        std::string last_call_time;         ///< 最后调用时间
        
        /**
         * @brief 慢调用记录结构
         */
        struct SlowCall {
            std::string timestamp;  ///< 调用时间戳
            uint64_t duration_us;   ///< 调用持续时间
            uint32_t pid;           ///< 线程ID
            uint32_t tgid;          ///< 进程ID
        };
        std::vector<SlowCall> slow_calls;   ///< 最慢调用列表
    };

    /**
     * @brief 函数计时结构
     */
    struct FunctionTiming {
        std::chrono::system_clock::time_point start_time;  ///< 开始时间
        std::string func_name;                            ///< 函数名
    };

    /**
     * @brief 构造函数
     * 
     * @param client_cfg 客户端配置JSON
     * @param uprobe_cfg uProbe配置JSON
     * @param env 环境配置
     * @param logger 日志管理器
     * @param profiler uProbe性能分析器
     */
    FuncSnoop( const json& uprobe_cfg, 
               const myenv& env, 
               Logger& logger,
               UprobeProfiler& profiler);
               

    /**
     * @brief 析构函数
     */
    ~FuncSnoop();

    /**
     * @brief 初始化BPF程序
     */
    void init_bpf();

    /**
     * @brief 附加所有函数
     */
    void attach_bpf();

    /**
     * @brief 附加单个函数
     * 
     * @param skel BPF骨架对象
     * @param attach_file_path 要附加的文件路径
     * @param func_name 函数名
     * @param sym 符号名
     * @param cookie 唯一标识符
     * @param links 链接向量引用
     * @return 是否附加成功
     */
    bool attach_function(struct func_snoop_bpf* skel,
                         const std::string& attach_file_path,
                         const std::string& func_name, 
                         const std::string& sym,
                         uint64_t cookie, 
                         std::vector<struct bpf_link*>& links);

    /**
     * @brief 读取跟踪管道
     * @return 包含跟踪信息的元组
     */
    std::tuple<std::string, int, int64_t, std::string> read_trace_pipe();

    /**
     * @brief 启动跟踪处理
     * @return 是否成功启动
     */
    bool start_trace();

    /**
     * @brief 停止跟踪处理
     */
    void stop_trace();

    /**
     * @brief Ring buffer处理线程
     */
    std::thread rb_thread_;

private:
    std::string log_path_;                   ///< 日志路径
    json uprobe_cfg_;                        ///< uProbe配置
    struct func_snoop_bpf* skel_;               ///< BPF骨架对象
    std::vector<struct bpf_link*> links_;    ///< BPF链接
    
    uint64_t next_func_id_ = 1;              ///< 函数ID计数器
    Logger& logger_;                         ///< 日志管理器
    UprobeProfiler& profiler_;               ///< uProbe性能分析器
    myenv env_;                              ///< 环境配置

    /**
     * @brief 函数ID到函数名的映射
     */
    std::map<uint64_t, std::string> func_id_map_;

    struct ring_buffer* rb_;                 ///< Ring buffer对象
    std::atomic<bool> exiting_;              ///< 退出标志

    /**
     * @brief Ring buffer回调函数
     * 
     * @param ctx 上下文
     * @param data 数据
     * @param data_sz 数据大小
     * @return 处理结果
     */
    static int handle_event(void* ctx, void* data, size_t data_sz);

    /**
     * @brief 获取修饰后的函数名
     * 
     * @param binary_path 二进制文件路径
     * @param func_name 函数名
     * @return 修饰后的函数名
     */
    static std::string get_mangled_name(const std::string& binary_path,
                                        const std::string& func_name);

    /**
     * @brief Ring buffer处理线程函数
     */
    void ring_buffer_thread();

    /**
     * @brief 处理回调事件
     * 
     * @param data 数据
     * @param data_sz 数据大小
     * @return 处理结果
     */
    int process_event(void* data, size_t data_sz);

    double log_sample_rate_ = 0.1;            ///< 日志采样率，默认10%
    std::mt19937 rng_{std::random_device{}()};  ///< 随机数生成器
    std::uniform_real_distribution<double> dist_{0.0, 1.0};  ///< 均匀分布

    /**
     * @brief 函数统计信息映射表
     */
    std::unordered_map<std::string, FunctionStats> func_stats_;

    /**
     * @brief 函数计时映射表 (tgid, pid, cookie) -> timing
     */
    std::map<std::tuple<uint32_t, uint32_t, uint64_t>, FunctionTiming> func_timing_;

    uint64_t event_count_ = 0;                  ///< 事件计数
    const uint64_t stats_report_interval_ = 1000;  ///< 统计报告间隔
    const uint64_t slow_call_threshold_us_ = 1000000;  ///< 慢调用阈值 (1s)
    const size_t max_slow_calls_to_record_ = 10;  ///< 每个函数记录的最慢调用数量
    const size_t output_slow_calls_count_ = 5;  ///< 输出的最慢调用数量
    const size_t output_freq_func_ = 5;  ///< 输出函数的数量

    int64_t last_sample_time_ = 0;  ///< 上次报告时间
    int64_t last_warning_time_ = 0;  ///< 上次警告时间
    /**
     * @brief 输出函数统计信息
     */
    void report_function_stats();
};

} // namespace NeuTracer

#endif // func_snoop_H