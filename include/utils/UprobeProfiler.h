// include/UprobeProfiler.h
#ifndef UPROBE_PROFILER_H
#define UPROBE_PROFILER_H

#include "tracer_service.grpc.pb.h"
#include <cstdint>
#include <ctime>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <uuid/uuid.h>
#include <vector>

#include "Logger.h"
#include "config.h"

namespace NeuTracer {

class UprobeProfiler {
public:
  UprobeProfiler(Logger &logger,const myenv &env);
  ~UprobeProfiler();


  template <typename... Args>
  void add_trace_data(const std::string &message, Args &&...args) {
    if (trace_file_func_) {
      //fmt::print(trace_file_func_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_func_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_func_, "\n");
      fflush(trace_file_func_);
    }
  }

  template <typename... Args>
  void add_trace_data_gpu(const std::string &message, Args &&...args) {
    if (trace_file_gpu_) {
      fmt::print(trace_file_gpu_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_gpu_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_gpu_, "\n");
      fflush(trace_file_gpu_);
    }
  }

  template <typename... Args>
  void add_trace_data_gpu_memleak(const std::string &message, Args &&...args) {
    if (trace_file_gpu_memleak_) {
      fmt::print(trace_file_gpu_memleak_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_gpu_memleak_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_gpu_memleak_, "\n");
      fflush(trace_file_gpu_memleak_);
    }
  }

  template <typename... Args>
  void add_trace_data_gpu_memevent(const std::string &message, Args &&...args) {
    if (trace_file_gpu_memevent_) {
      fmt::print(trace_file_gpu_memevent_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_gpu_memevent_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_gpu_memevent_, "\n");
      fflush(trace_file_gpu_memevent_);
    }
  }

  template<typename... Args>
  void add_trace_data_gpu_memcpy(const std::string &message, Args &&...args) {
    if (trace_file_gpu_memcpy_) {
      fmt::print(trace_file_gpu_memcpy_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_gpu_memcpy_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_gpu_memcpy_, "\n");
      fflush(trace_file_gpu_memcpy_);
    }
  }

  template <typename... Args>
  void add_trace_data_gpu_kernlaunch(const std::string &message, Args &&...args) {
    if (trace_file_gpu_kernlaunch_) {
      fmt::print(trace_file_gpu_kernlaunch_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_gpu_kernlaunch_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_gpu_kernlaunch_, "\n");
      fflush(trace_file_gpu_kernlaunch_);
    }
  }

    template <typename... Args>
    void add_trace_data_gpu_memfragment(const std::string &message, Args &&...args) {
      if (trace_file_gpu_memfragment_) {
        fmt::print(trace_file_gpu_memfragment_, "[{}] [+] ", get_timestamp());
        fmt::print(trace_file_gpu_memfragment_, message, std::forward<Args>(args)...);
        fmt::print(trace_file_gpu_memfragment_, "\n");
        fflush(trace_file_gpu_memfragment_);
      }
    }

  

  template <typename... Args>
  void add_trace_data_cpu(const std::string &message, Args &&...args) {
    if (trace_file_cpu_) {
      //fmt::print(trace_file_cpu_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_cpu_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_cpu_, "\n");
      fflush(trace_file_cpu_);
    }
  }
  template <typename... Args>
  void add_trace_data_kmem(const std::string &message, Args &&...args) {
    if (trace_file_kmem_) {
      //fmt::print(trace_file_kmem_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_kmem_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_kmem_, "\n");
      fflush(trace_file_kmem_);
    }
  }
  template <typename... Args>
  void add_trace_data_net(const std::string &message, Args &&...args) {
    if (trace_file_net_) {
      //fmt::print(trace_file_net_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_net_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_net_, "\n");
      fflush(trace_file_net_);
    }
  }
  template <typename... Args>
  void add_trace_data_io(const std::string &message, Args &&...args) {
    if (trace_file_io_) {
      //fmt::print(trace_file_io_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_io_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_io_, "\n");
      fflush(trace_file_io_);
    }
  }
  template <typename... Args>
  void add_trace_data_syscall(const std::string &message, Args &&...args) {
    if (trace_file_syscall_) {
      //fmt::print(trace_file_syscall_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_syscall_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_syscall_, "\n");
      fflush(trace_file_syscall_);
    }
  }
  template <typename... Args>
  void add_trace_data_pcie(const std::string &message, Args &&...args) {
    if (trace_file_pcie_) {
      //fmt::print(trace_file_pcie_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_pcie_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_pcie_, "\n");
      fflush(trace_file_pcie_);
    }
  }
  template <typename... Args>
  void add_trace_data_nvlink(const std::string &message, Args &&...args) {
    if (trace_file_nvlink_) {
      //fmt::print(trace_file_nvlink_, "[{}] [+] ", get_timestamp());
      fmt::print(trace_file_nvlink_, message, std::forward<Args>(args)...);
      fmt::print(trace_file_nvlink_, "\n");
      fflush(trace_file_nvlink_);
    }
  }

  void close();
  struct CPUTraceItem {
    std::string timestamp;        // 时间戳
    uint32_t pid;                 // 进程ID
    uint32_t ppid;                // 父进程ID
    std::string comm;             // 进程名称
    uint32_t cpu_id;              // CPU ID
    uint64_t oncpu_time;          // 运行时间(微秒)
    uint64_t offcpu_time;         // 等待时间(微秒)
    double utilization;           // CPU利用率(0-100%)
    
    // 异常检测相关字段
    uint32_t migrations_count;    // CPU迁移次数
    uint32_t numa_migrations;     // NUMA迁移次数
    uint32_t hotspot_cpu;         // 热点CPU
    double hotspot_percentage;    // 热点占比
    
};

  struct TraceBufferItem {
    std::string timestamp;
    std::string function_name;
    uint32_t pid;
    uint32_t tgid;
    std::string event_type;  // "ENTER" 或 "EXIT"
    int64_t cookie;
    uint64_t call_count;
    uint64_t avg_duration_us;
};

 struct IOTraceItem {
    std::string timestamp;
    uint32_t pid;
    uint32_t tgid;           // 线程组ID
    std::string comm;
    std::string operation;   // 读/写操作
    uint64_t bytes;          // 传输字节数
    double latency_ms;       // 操作延迟(毫秒)
    std::string device;      // 设备名称
    uint8_t major;           // 主设备号
    uint8_t minor;           // 次设备号
    
    // 统计数据
    uint64_t read_bytes;     // 累计读取字节
    uint64_t write_bytes;    // 累计写入字节
    uint64_t read_ops;       // 读操作次数
    uint64_t write_ops;      // 写操作次数
    double avg_read_latency; // 平均读延迟
    double avg_write_latency;// 平均写延迟
};

struct MemTraceItem {
  std::string timestamp;       // 时间戳
  uint32_t pid;                // 进程ID
  uint32_t tgid;               // 线程组ID
  std::string comm;            // 进程名称
  std::string operation;       // 操作类型(alloc/free)
  uint64_t size;               // 分配/释放大小(字节)
  uint64_t addr;               // 内存地址
  uint64_t stack_id;           // 栈ID
  
  // 统计数据
  uint64_t total_allocs;       // 总分配次数
  uint64_t total_frees;        // 总释放次数
  uint64_t current_memory;     // 当前内存使用量
  uint64_t peak_memory;        // 峰值内存使用量
  
};

struct NetTraceItem {
  std::string timestamp;       // 时间戳
  uint32_t pid;                // 进程ID
  uint32_t tgid;               // 线程组ID
  std::string comm;            // 进程名称
  bool is_send;                // 发送或接收
  uint64_t bytes;              // 传输字节数
  
  // 网络地址信息
  std::string src_addr;        // 源IP地址
  std::string dst_addr;        // 目标IP地址
  uint16_t src_port;           // 源端口
  uint16_t dst_port;           // 目标端口
  std::string protocol;        // 协议
  
  // 统计数据
  uint64_t tx_bytes;           // 总发送字节数
  uint64_t rx_bytes;           // 总接收字节数
  uint64_t tx_packets;         // 总发送包数
  uint64_t rx_packets;         // 总接收包数
  
  // 异常检测相关字段
  uint32_t active_connections; // 活跃连接数
  
};

struct GPUTraceItem  {
  enum class EventType {
    CALLSTACK,  
    MEMEVENT,       
    CUDALAUNCHEVENT,  
    MEMTRANSEVENT,
    UNKNOWN       
  };
  EventType type;
  std::string timestamp;  // 时间戳
  union {
    struct {
      uint32_t pid;            // 进程ID
      char* stack_message; // 栈信息
    } callstack_event;

    struct {
      uint32_t pid;            // 进程ID
      uint64_t mem_size;      // 内存大小(字节)
    } mem_event;

    struct {
      char* kernel_name; // 内核函数名称
      uint32_t pid;            // 进程ID
      uint64_t num_calls;    
    } cuda_launch_event;

    struct {
      uint32_t pid;            // 进程ID
      uint64_t mem_trans_rate; // 内存传输速率(字节/秒)
      char* kind_str; // 传输方向描述(D2D, D2H, H2D, H2H)
    } mem_trans_event;

  };
  GPUTraceItem(EventType t = EventType::UNKNOWN) : type(t) {
    if (type == EventType::CALLSTACK) {
      callstack_event.stack_message = nullptr;
    } else if (type == EventType::MEMEVENT) {
      mem_event.pid = 0;
      mem_event.mem_size = 0;
    } else if (type == EventType::CUDALAUNCHEVENT) {
      cuda_launch_event.kernel_name = nullptr;
      cuda_launch_event.pid = 0;
      cuda_launch_event.num_calls = 0;
    }
    else if (type == EventType::MEMTRANSEVENT) {
      mem_trans_event.pid = 0;
      mem_trans_event.mem_trans_rate = 0;
    }
  }

};






  // 新增RPC相关方法
  bool initRPC();
  bool isRPCEnabled() const { return rpc_enabled_; }
  bool checkServerStatus(bool &is_active, int64_t &traces_count);
  std::string get_timestamp();
  // std::vector<UprobeMetaData> trace_data;
  // 新增RPC发送方法
  bool sendTraceBatch();
  void sendTrace(const std::string &timestamp, const std::string &Funcname,
    uint32_t pid, uint32_t tgid, const std::string &event_type,
    int64_t cookie, uint64_t call_count, uint64_t avg_duration_us);
  void sendGPUTrace(GPUTraceItem &gpu_data);
  void sendCPUTrace(const CPUTraceItem& cpu_data);
  void sendMemTrace(const MemTraceItem& kmem_data);
  void sendNetTrace(const NetTraceItem& net_data);
  void sendIOTrace(const IOTraceItem& io_data); 
  // UprobeProfiler& operator+(const UprobeProfiler& other);
  // void clean();
  // Json::Value to_json() const;
  // void add_json(const Json::Value& trace_data);
  // Json::Value to_perfetto() const;


private:
  Logger logger_;
  myenv env_;
  std::string log_path_func_;
  std::string log_path_gpu_;
  std::string log_path_gpu_memleak_;
  std::string log_path_gpu_memevent_;
  std::string log_path_gpu_memcpy_;
  std::string log_path_gpu_kernlaunch_;
  std::string log_path_gpu_memfragment_;
  std::string log_path_cpu_;
  std::string log_path_kmem_;
  std::string log_path_net_;
  std::string log_path_io_;
  std::string log_path_syscall_;
  std::string log_path_pcie;  
  std::string log_path_nvlink_;  
  FILE *trace_file_func_;
  FILE *trace_file_gpu_;
  FILE *trace_file_gpu_memleak_;
  FILE *trace_file_gpu_memevent_;
  FILE *trace_file_gpu_memcpy_;
  FILE *trace_file_gpu_kernlaunch_;
  FILE *trace_file_gpu_memfragment_;
  FILE *trace_file_cpu_;
  FILE *trace_file_kmem_;
  FILE *trace_file_net_;
  FILE *trace_file_io_;
  FILE *trace_file_syscall_;
  FILE *trace_file_pcie_;
  FILE *trace_file_nvlink_;
  // 新增RPC相关成员变量
  bool rpc_enabled_ = false;
  std::shared_ptr<grpc::Channel> grpc_channel_;
  std::unique_ptr<TracerService::Stub> stub_;
  std::mutex rpc_mutex_;

  // 缓冲区
  std::vector<TraceBufferItem> trace_buffer_;
  std::vector<GPUTraceItem> gpu_buffer_;
  std::vector<CPUTraceItem> cpu_buffer_;
  std::vector<MemTraceItem> kmem_buffer_;
  std::vector<IOTraceItem> io_buffer_;
  std::vector<NetTraceItem> net_buffer_;

  // 批处理阈值
  static constexpr int BATCH_SIZE = 100;
};

} // namespace NeuTracer

#endif // UPROBE_PROFILER_H