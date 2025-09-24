#ifndef TRACER_H
#define TRACER_H

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <tuple>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <uuid/uuid.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"
#include "utils/GetPid.h"
#include "snoop/func_snoop.h"
#include "snoop/cpu_snoop.h"
#include "snoop/gpu_snoop.h"
#include "snoop/kmem_snoop.h"
#include "snoop/net_snoop.h"
#include "snoop/io_snoop.h"
#include "snoop/syscall_snoop.h"
#include "snoop/pcie_snoop.h"
#include "snoop/nvlink_snoop.h"

namespace NeuTracer {

/**
 * @class Tracer
 * @brief 主要跟踪器类，协调所有监控组件
 * 
 * Tracer类负责初始化和管理不同的监控组件（CPU、内存、网络等），
 * 处理配置文件，并提供统一的接口来启动和停止跟踪过程。
 */
class Tracer {
public:
    /**
     * @brief 构造函数
     * @param uprobe_cfg_path uProbe配置文件路径
     * @param info_level 日志级别
     * @param env 环境配置
     */
    Tracer(const std::string uprobe_cfg_path = UPROBE_CFG_PATH,
           const std::string info_level = "info",
           const myenv& env = myenv());

    /**
     * @brief 析构函数，清理资源
     */
    ~Tracer();

    /**
     * @brief 启动跟踪过程
     */
    void run();
    
    /**
     * @brief 关闭跟踪过程
     */
    void close();
    
    /**
     * @brief 清理所有资源
     */
    void clean();
    
    /**
     * @brief 输出当前统计信息
     * @param cur_time 当前时间
     * @param prev_time 前一个时间点
     */
    void stat(double cur_time, double prev_time);

    /**
     * @brief 处理跟踪数据
     * @param work 工作类型
     * @param pid 进程ID
     * @param ts 时间戳
     * @param op 操作类型
     * @param func 函数名
     * @return 处理后的跟踪数据
     */
    std::vector<std::tuple<std::string, int, int, std::string>>
    processTraceData(const std::string& work, int pid, int ts,
                     const std::string& op, const std::string& func);

    /**
     * @brief 加载JSON配置文件
     * @param path 文件路径
     * @return 解析后的JSON对象
     */
    nlohmann::json loadJSON(std::string path);
    
    /**
     * @brief 转换为Perfetto格式
     * @return Perfetto格式的JSON数据
     */
    nlohmann::json toPerfetto();
    
    /**
     * @brief 转换为通用JSON格式
     * @return 表示跟踪数据的元组向量
     */
    std::vector<std::tuple<std::string, int, int, std::string>> toJson();

    /**
     * @brief 获取日志路径
     * @return 日志文件路径
     */
    std::string log_path_;

private:
    using json = nlohmann::json;
    
    json uprobe_cfg_;                               ///< uProbe配置
    
    Logger logger_;                                 ///< 日志管理器
    UprobeProfiler profiler_;                       ///< uProbe性能分析器
    
    FuncSnoop func_snoop_;                        ///< BPF管理器
    GPUSnoop gpu_snoop_;                            ///< GPU监控器
    CPUsnoop cpu_snoop_;                            ///< CPU监控器
    KmemSnoop kmem_snoop_;                          ///< 内核内存监控器
    NetSnoop net_snoop_;                            ///< 网络监控器
    IoSnoop io_snoop_;                              ///< IO监控器
    SyscallSnoop syscall_snoop_;                  ///< 系统调用监控器
    PcieSnoop pcie_snoop_;                          ///< PCIe监控器
    NVLinkSnoop nvlink_snoop_;
    
    std::vector<std::string> py_filter_;            ///< Python过滤器列表
    std::vector<std::string> cuda_func_;            ///< CUDA函数列表
    std::map<std::string, std::vector<std::string>> py2lib_; ///< Python到库的映射
    
    int pid_;                                       ///< 要监控的进程ID
    FILE* trace_file_;                              ///< 跟踪文件句柄
    std::mutex file_mutex_;                         ///< 文件互斥锁
    myenv env_;                                     ///< 环境配置
};

} // namespace NeuTracer

#endif // TRACER_H