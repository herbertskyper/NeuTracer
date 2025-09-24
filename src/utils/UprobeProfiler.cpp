#include "utils/UprobeProfiler.h"



#include "tracer_service.pb.h"
#include "utils/Logger.h"
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <unistd.h>
#include <uuid/uuid.h>

namespace NeuTracer {
// 初始化RPC客户端
bool UprobeProfiler::initRPC() {
  try {
    if(env_.grpc_enabled) {
      grpc::ChannelArguments args;
      args.SetMaxSendMessageSize(100 * 1024 * 1024);    // 100MB
      args.SetMaxReceiveMessageSize(100 * 1024 * 1024); // 100MB

      grpc_channel_ = grpc::CreateCustomChannel(
          env_.server_addr, grpc::InsecureChannelCredentials(), args);
      stub_ = TracerService::NewStub(grpc_channel_);

      rpc_enabled_ = true;
      logger_.info("RPC client initialized, server address: " + env_.server_addr);
    }
    else {

      rpc_enabled_ = false;
      logger_.info("RPC client is disabled, please launch program with -g/--grpc to use RPC features");    
    }
      
    return true;
  } catch (const std::exception &e) {
    logger_.error("Failed to initialize RPC client: " + std::string(e.what()));
    rpc_enabled_ = false;
    return false;
  }
}

// 添加sendTraceBatch方法
bool UprobeProfiler::sendTraceBatch() {
  if (!rpc_enabled_ || !stub_) {
    return false;
  }

  try {
    TraceBatch batch; // 使用NeuTracer::TraceBatch

  // 添加跟踪数据
  for (const auto &trace : trace_buffer_) {
    TraceData *trace_data = batch.add_trace_data(); // 使用NeuTracer::TraceData
    
    // 根据TraceData消息定义填充所有字段
    trace_data->set_timestamp(trace.timestamp);
    trace_data->set_event_type(trace.event_type == "ENTER" ? 0 : 1);
    trace_data->set_cookie(trace.cookie);
    trace_data->set_function_name(trace.function_name);
    trace_data->set_pid(trace.pid);
    trace_data->set_call_count(trace.call_count);
    trace_data->set_avg_duration_us(trace.avg_duration_us);
    trace_data->set_tgid(trace.tgid);
}

    // 在 sendTraceBatch() 方法中更新 GPU 事件处理部分

    for (const auto &gpu_event : gpu_buffer_) {
      GPUTraceData *trace = batch.add_gpu_trace_data();

      // 根据事件类型设置不同的字段
      switch (gpu_event.type) {
      case GPUTraceItem::EventType::CALLSTACK:
      {
        trace->set_timestamp(gpu_event.timestamp);
        auto* callstack = trace->mutable_callstack_event();
        callstack->set_pid(gpu_event.callstack_event.pid);
        callstack->set_stack_message(gpu_event.callstack_event.stack_message);
        trace->set_event_type(GPUTraceData_EventType_CALLSTACK);
        break;
      }
      case GPUTraceItem::EventType::MEMEVENT:
      {
        trace->set_timestamp(gpu_event.timestamp);
        auto* mem_event = trace->mutable_mem_event();
        mem_event->set_pid(gpu_event.mem_event.pid);
        mem_event->set_mem_size(gpu_event.mem_event.mem_size);
        trace->set_event_type(GPUTraceData_EventType_MEMEVENT);
        break;
      }
      case GPUTraceItem::EventType::CUDALAUNCHEVENT:
      {
        trace->set_timestamp(gpu_event.timestamp);
        auto* cuda_event = trace->mutable_cuda_launch_event();
        cuda_event->set_kernel_name(gpu_event.cuda_launch_event.kernel_name);
        cuda_event->set_pid(gpu_event.cuda_launch_event.pid);
        cuda_event->set_num_calls(gpu_event.cuda_launch_event.num_calls);
        trace->set_event_type(GPUTraceData_EventType_CUDALAUNCHEVENT);
        break;
      }
      case GPUTraceItem::EventType::MEMTRANSEVENT:
      {
        trace->set_timestamp(gpu_event.timestamp);
        auto* mem_trans_event = trace->mutable_mem_trans_event();
        mem_trans_event->set_pid(gpu_event.mem_trans_event.pid);
        mem_trans_event->set_mem_trans_rate(gpu_event.mem_trans_event.mem_trans_rate);
        mem_trans_event->set_kind_str(gpu_event.mem_trans_event.kind_str);
        trace->set_event_type(GPUTraceData_EventType_MEMTRANSEVENT);
        break;
      }
      
      case GPUTraceItem::EventType::UNKNOWN:
      default:
        logger_.error("Unknown GPU event type encountered, skipping.");
        break;
      }
    }

    // 添加CPU跟踪数据
    for (const auto &cpu_data : cpu_buffer_) {

      try {
        // 获取新的 CPUTraceData protobuf 消息对象
        CPUTraceData *trace = batch.add_cpu_trace_data();

        // 设置基本信息
        trace->set_timestamp(cpu_data.timestamp);
        trace->set_pid(cpu_data.pid);
        trace->set_ppid(cpu_data.ppid);
        trace->set_comm(cpu_data.comm);
        trace->set_cpu_id(cpu_data.cpu_id);
        trace->set_oncpu_time(cpu_data.oncpu_time);
        trace->set_offcpu_time(cpu_data.offcpu_time);
        trace->set_utilization(cpu_data.utilization);

        // 设置其他相关字段
        trace->set_migrations_count(cpu_data.migrations_count);
        trace->set_numa_migrations(cpu_data.numa_migrations);
        trace->set_hotspot_cpu(cpu_data.hotspot_cpu);
        trace->set_hotspot_percentage(cpu_data.hotspot_percentage);
      } catch (const std::exception &e) {
        logger_.error("Failed to add CPU trace data: " + std::string(e.what()));
        return false;
      }
    }

    for (const auto &io_data : io_buffer_) {
      IOTraceData *trace = batch.add_io_trace_data();

      // 设置基本信息
      trace->set_timestamp(io_data.timestamp);
      trace->set_pid(io_data.pid);
      trace->set_comm(io_data.comm);
      trace->set_operation(io_data.operation);
      trace->set_bytes(io_data.bytes);
      trace->set_latency_ms(io_data.latency_ms);
      trace->set_device(io_data.device);
      trace->set_major(io_data.major);
      trace->set_minor(io_data.minor);

      // 设置统计数据
      trace->set_read_bytes(io_data.read_bytes);
      trace->set_write_bytes(io_data.write_bytes);
      trace->set_read_ops(io_data.read_ops);
      trace->set_write_ops(io_data.write_ops);
      trace->set_avg_read_latency(io_data.avg_read_latency);
      trace->set_avg_write_latency(io_data.avg_write_latency);
    }

  for (const auto &mem_data : kmem_buffer_) {
    MemoryTraceData *trace = batch.add_memory_trace_data();
    
    // 设置基本信息
    trace->set_timestamp(mem_data.timestamp);
    trace->set_pid(mem_data.pid);
    trace->set_tgid(mem_data.tgid);
    trace->set_comm(mem_data.comm);
    trace->set_operation(mem_data.operation);
    trace->set_size(mem_data.size);
    trace->set_addr(mem_data.addr);
    trace->set_stack_id(mem_data.stack_id);
    
    // 设置统计数据
    trace->set_total_allocs(mem_data.total_allocs);
    trace->set_total_frees(mem_data.total_frees);
    trace->set_current_memory(mem_data.current_memory);
    trace->set_peak_memory(mem_data.peak_memory);
    }

for (const auto &net_data : net_buffer_) {
    NetworkTraceData *trace = batch.add_network_trace_data();
    
    // 设置基本信息
    trace->set_timestamp(net_data.timestamp);
    trace->set_pid(net_data.pid);
    trace->set_tgid(net_data.tgid);
    trace->set_comm(net_data.comm);
    trace->set_is_send(net_data.is_send);
    trace->set_bytes(net_data.bytes);
    
    // 设置网络信息
    trace->set_src_addr(net_data.src_addr);
    trace->set_dst_addr(net_data.dst_addr);
    trace->set_src_port(net_data.src_port);
    trace->set_dst_port(net_data.dst_port);
    trace->set_protocol(net_data.protocol);
    
    // 设置统计数据
    trace->set_tx_bytes(net_data.tx_bytes);
    trace->set_rx_bytes(net_data.rx_bytes);
    trace->set_tx_packets(net_data.tx_packets);
    trace->set_rx_packets(net_data.rx_packets);
    
    trace->set_active_connections(net_data.active_connections);
    }
    // RPC调用
    grpc::ClientContext context;
    TraceResponse response; // 使用NeuTracer::TraceResponse
    grpc::Status status = stub_->SendTraceBatch(&context, batch, &response);

  // 清空缓冲区
  if (status.ok() && response.success()) {
    trace_buffer_.clear();
    gpu_buffer_.clear();
    cpu_buffer_.clear();
    kmem_buffer_.clear();
    net_buffer_.clear();
    io_buffer_.clear();
    return true;
  } else {
    //logger_.error("RPC failed: " + status.error_message());
    logger_.error("RPC failed: " + status.error_message() + " (code: " + 
                   std::to_string(status.error_code()) + ", details: " + status.error_details() + ")");
    if (status.error_code() == grpc::StatusCode::UNAVAILABLE) {
      logger_.warn("RPC server unavailable, disabling RPC temporarily");
      // 可选: 暂时禁用RPC
      rpc_enabled_ = false;
    }
    return false;
  }
   } catch (const std::exception& e) {
    logger_.error("Exception in sendTraceBatch: " + std::string(e.what()));
    return false;
  }
}

// 添加检查服务器状态方法
bool UprobeProfiler::checkServerStatus(bool &is_active, int64_t &traces_count) {
  if (!rpc_enabled_ || !stub_) {
    logger_.error("RPC not enabled");
    return false;
  }

  grpc::ClientContext context;
  StatusRequest request;   // 使用NeuTracer::StatusRequest
  StatusResponse response; // 使用NeuTracer::StatusResponse

  // 设置一个唯一的客户端标识符
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  std::string client_id =
      std::string(hostname) + "_" + std::to_string(getpid());
  request.set_client_id(client_id);

  grpc::Status status = stub_->GetStatus(&context, request, &response);

  if (status.ok()) {
    is_active = response.active();
    traces_count = response.received_traces();
    return true;
  } else {
    logger_.error("GetStatus RPC failed: " + status.error_message());
    return false;
  }
}

// 添加其他发送方法的实现
void UprobeProfiler::sendTrace(const std::string &timestamp,
                               const std::string &Funcname, uint32_t pid,
                               uint32_t tgid, const std::string &event_type,
                               int64_t cookie, uint64_t call_count,
                               uint64_t avg_duration_us) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  auto tmp = UprobeProfiler::TraceBufferItem{timestamp,  Funcname,       pid,
                                             tgid,       event_type,     cookie,
                                             call_count, avg_duration_us};
  trace_buffer_.push_back(tmp);

  if (trace_buffer_.size() >= BATCH_SIZE) {
    sendTraceBatch();
  }
}

void UprobeProfiler::sendGPUTrace(GPUTraceItem &gpu_data) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);

  gpu_buffer_.push_back(gpu_data);

  if (gpu_buffer_.size() >= BATCH_SIZE) {
    sendTraceBatch();
  }
}

void UprobeProfiler::sendCPUTrace(const CPUTraceItem &cpu_data) {
    // logger_.info("CPU trace added to buffer, current size: " + 
    //              std::to_string(cpu_buffer_.size()) + 
    //              " (batch size: " + std::to_string(BATCH_SIZE) + ")");
                 
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  cpu_buffer_.push_back(cpu_data);

  if (cpu_buffer_.size() >= BATCH_SIZE) {
    sendTraceBatch();
  }
}

void UprobeProfiler::sendMemTrace(const MemTraceItem &mem_data) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  kmem_buffer_.push_back(mem_data);

  if (kmem_buffer_.size() >= BATCH_SIZE) {
    sendTraceBatch();
  }
}

void UprobeProfiler::sendNetTrace(const NetTraceItem &net_data) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  net_buffer_.push_back(net_data);

  if (net_buffer_.size() >= BATCH_SIZE) {
    sendTraceBatch();
  }
}

void UprobeProfiler::sendIOTrace(const IOTraceItem &io_data) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  io_buffer_.push_back(io_data);

  if (io_buffer_.size() >= BATCH_SIZE) {
    sendTraceBatch();
  }
}

// 在 UprobeProfiler 构造函数中添加
UprobeProfiler::UprobeProfiler(Logger &logger,const myenv &env) : logger_(logger), env_(env) {

  // 生成唯一的日志文件路径
  uuid_t uuid;
  char uuid_str[37];
  uuid_generate(uuid);
  uuid_unparse_lower(uuid, uuid_str);
  std::string log_path_prefix_func = LOG_BASE_PATH+"NeuTracer_log_func_";
  std::string log_path_prefix_gpu = LOG_BASE_PATH+"NeuTracer_log_gpu_";
  std::string log_path_prefix_cpu = LOG_BASE_PATH+"NeuTracer_log_cpu_";
  std::string log_path_prefix_kmem = LOG_BASE_PATH+"NeuTracer_log_kmem_";
  std::string log_path_prefix_net = LOG_BASE_PATH+"NeuTracer_log_net_";
  std::string log_path_prefix_io = LOG_BASE_PATH+"NeuTracer_log_io_";
  std::string log_path_prefix_syscall = LOG_BASE_PATH+"NeuTracer_log_syscall_";
  std::string log_path_prefix_pcie = LOG_BASE_PATH+"NeuTracer_log_pcie_";
  std::string log_path_prefix_nvlink = LOG_BASE_PATH+"NeuTracer_log_nvlink_";
  // 获取当前时间
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_now;
  localtime_r(&now_time_t, &tm_now);

  // 格式化日期时间
  char date_time_str[64];
  std::strftime(date_time_str, sizeof(date_time_str), "%Y%m%d_%H%M%S", &tm_now);

  // 创建包含日期时间的日志文件名
  log_path_func_ = log_path_prefix_func + date_time_str + "_" + ".log";
  log_path_gpu_ = log_path_prefix_gpu + date_time_str + "_" + ".log";
  log_path_gpu_memleak_ =
      log_path_prefix_gpu + "memleak_" + date_time_str + "_" + ".log";
  log_path_gpu_memevent_ = log_path_prefix_gpu + "memevent_" + date_time_str + "_" + ".log";
  log_path_gpu_memcpy_ =
      log_path_prefix_gpu + "memcpy_" + date_time_str + "_" + ".log";
  log_path_gpu_kernlaunch_ =
      log_path_prefix_gpu + "kernelaunch_" + date_time_str + "_" + ".log";
  log_path_gpu_memfragment_ =
      log_path_prefix_gpu + "memfragment_" + date_time_str + "_" + ".log";

  log_path_cpu_ = log_path_prefix_cpu + date_time_str + "_" + ".log";
  log_path_kmem_ =
      log_path_prefix_kmem + date_time_str + "_" + ".log";
  log_path_net_ = log_path_prefix_net + date_time_str + "_" + ".log";
  log_path_io_ = log_path_prefix_io + date_time_str + "_" + ".log";
  log_path_syscall_ = log_path_prefix_syscall + date_time_str + "_" + ".log";
  log_path_pcie = log_path_prefix_pcie + date_time_str + "_" + ".log";
  log_path_nvlink_ = log_path_prefix_nvlink + date_time_str + "_" + ".log";
  // 打印当前日期时间
  char full_date_time[100];
  std::strftime(full_date_time, sizeof(full_date_time),
                "Trace started on %Y-%m-%d %H:%M:%S", &tm_now);

  if(env_.trace_modules.func) {
    // 创建跟踪文件
    trace_file_func_ = fopen(log_path_func_.c_str(), "w");
    if (!trace_file_func_) {
      logger_.error("Failed to create trace file: " + log_path_func_);
    }
    // 打印表头和日期时间信息
    if (trace_file_func_) {
      fprintf(trace_file_func_, "%s\n\n", full_date_time);
      fprintf(trace_file_func_, "func_log\n");
      // // 打印表头
      // fprintf(trace_file_func_, "%-12s %-5s %-6s %s\n", "TIME", "EVENT", "COOKIE",
      //         "FUNCTION");
      // fprintf(trace_file_func_, "%-12s %-5s %-6s %s\n", "------------", "-----",
      //         "------", "----------");
      fflush(trace_file_func_);
    } else {
      logger_.error("Trace file not available, data only printed to console");
    }
  } else {
    trace_file_func_ = nullptr;
  }

  if(env_.trace_modules.gpu) {
    trace_file_gpu_ = fopen(log_path_gpu_.c_str(), "w");
    if (!trace_file_gpu_) {
      logger_.error("Failed to create log file: " + log_path_gpu_);
    }

    if (trace_file_gpu_) {
      fprintf(trace_file_gpu_, "%s\n\n", full_date_time);
      fprintf(trace_file_gpu_, "gpu_log\n");
      fflush(trace_file_gpu_);
    } else {
      logger_.error("Trace file gpu not available, data only printed to console");
    }

    trace_file_gpu_memleak_ = fopen(
        log_path_gpu_memleak_.c_str(), "w");
    if (!trace_file_gpu_memleak_) {
      logger_.error("Failed to create log file: " + log_path_gpu_memleak_);
    }
    if (trace_file_gpu_memleak_) {
      fprintf(trace_file_gpu_memleak_, "%s\n\n", full_date_time);
      fprintf(trace_file_gpu_memleak_, "gpu_memleak_log\n");
      fflush(trace_file_gpu_memleak_);
    } else {
      logger_.error("Trace file gpu memleak not available, data only printed to console");
    }

    trace_file_gpu_memevent_ = fopen(
        log_path_gpu_memevent_.c_str(), "w");
    if (!trace_file_gpu_memevent_) {
      logger_.error("Failed to create log file: " + log_path_gpu_memevent_);
    }
    if (trace_file_gpu_memevent_) {
      fprintf(trace_file_gpu_memevent_, "%s\n\n", full_date_time);
      fprintf(trace_file_gpu_memevent_, "gpu_memevent_log\n");
      fflush(trace_file_gpu_memevent_);
    } else {
      logger_.error("Trace file gpu memevent not available, data only printed to console");
    }

    trace_file_gpu_memcpy_ = fopen(
        log_path_gpu_memcpy_.c_str(), "w");
    if (!trace_file_gpu_memcpy_) {
      logger_.error("Failed to create log file: " + log_path_gpu_memcpy_);
    }
    if (trace_file_gpu_memcpy_) {
      fprintf(trace_file_gpu_memcpy_, "%s\n\n", full_date_time);
      fprintf(trace_file_gpu_memcpy_, "gpu_memcpy_log\n");
      fflush(trace_file_gpu_memcpy_); 
    } else {
      logger_.error("Trace file gpu memcpy not available, data only printed to console");
    }

    trace_file_gpu_kernlaunch_ = fopen(
        log_path_gpu_kernlaunch_.c_str(), "w");
    if (!trace_file_gpu_kernlaunch_) {
      logger_.error("Failed to create log file: " + log_path_gpu_kernlaunch_);
    } 
    if (trace_file_gpu_kernlaunch_) {
      fprintf(trace_file_gpu_kernlaunch_, "%s\n\n", full_date_time);
      fprintf(trace_file_gpu_kernlaunch_, "gpu_kernlaunch_log\n");
      fflush(trace_file_gpu_kernlaunch_);
    } else {
      logger_.error("Trace file gpu kernlaunch not available, data only printed to console");
    }

    trace_file_gpu_memfragment_ = fopen(
        log_path_gpu_memfragment_.c_str(), "w");
    if (!trace_file_gpu_memfragment_) {
      logger_.error("Failed to create log file: " + log_path_gpu_memfragment_);
    } 
    if (trace_file_gpu_memfragment_) {
      fprintf(trace_file_gpu_memfragment_, "%s\n\n", full_date_time);
      fprintf(trace_file_gpu_memfragment_, "gpu_memfragment_log\n");
      fflush(trace_file_gpu_memfragment_);
    } else {
      logger_.error("Trace file gpu memfragment not available, data only printed to console");
    }


  } else {
    trace_file_gpu_ = nullptr;
    trace_file_gpu_memleak_ = nullptr;
    trace_file_gpu_memevent_ = nullptr;
    trace_file_gpu_memcpy_ = nullptr;
    trace_file_gpu_kernlaunch_ = nullptr;
    trace_file_gpu_memfragment_ = nullptr;
  }

  if(env_.trace_modules.cpu) {
    // 创建 CPU 跟踪文件
    trace_file_cpu_ = fopen(log_path_cpu_.c_str(), "w");
    if (trace_file_cpu_) {
      fprintf(trace_file_cpu_, "%s\n\n", full_date_time);
      fprintf(trace_file_cpu_, "cpu_log\n");
      // fprintf(trace_file_cpu_, "%-26s %-10s %-10s %-16s %-12s %-12s %-8s\n",
      //         "TIME", "PPID", "PID", "COMM", "ON CPU", "OFF CPU", "CPU%");
      // fprintf(trace_file_cpu_, "%-26s %-10s %-10s %-16s %-12s %-12s %-8s\n",
      //         "--------------------------", "----------", "----------",
      //         "----------------", "----------", "----------", "------");
      fflush(trace_file_cpu_);
    } else {
      logger_.error("Failed to create CPU log file: " + log_path_cpu_);
    }
  }

  if(env_.trace_modules.kmem) {
    trace_file_kmem_ = fopen(log_path_kmem_.c_str(), "w");
    if (trace_file_kmem_) {
      fprintf(trace_file_kmem_, "%s\n\n", full_date_time);
      fprintf(trace_file_kmem_, "kmem_log\n");
      fflush(trace_file_kmem_);
    } else {
      logger_.error("Failed to create kmem log file: " + log_path_kmem_);
    }
  } else {
    trace_file_kmem_ = nullptr;
  }

  if(env_.trace_modules.net) {
    trace_file_net_ = fopen(log_path_net_.c_str(), "w");
    if (trace_file_net_) {
      fprintf(trace_file_net_, "%s\n\n", full_date_time);
      fprintf(trace_file_net_, "net_log\n");
      fflush(trace_file_net_);
    } else {
      logger_.error("Failed to create net log file: " + log_path_net_);
    }
  } else {
    trace_file_net_ = nullptr;
  }

  if(env_.trace_modules.io) {
    trace_file_io_ = fopen(log_path_io_.c_str(), "w");
    if (trace_file_io_) {
      fprintf(trace_file_io_, "%s\n\n", full_date_time);
      fprintf(trace_file_io_, "io_log\n");
      fflush(trace_file_io_);
    } else {
      logger_.error("Failed to create io log file: " + log_path_io_);
    }
  } else {
    trace_file_io_ = nullptr;
  }
  if(env_.trace_modules.syscall) {
    trace_file_syscall_ = fopen(log_path_syscall_.c_str(), "w");
    if (trace_file_syscall_) {
      fprintf(trace_file_syscall_, "%s\n\n", full_date_time);
      fprintf(trace_file_syscall_, "syscall_log\n");
      fflush(trace_file_syscall_);
    } else {
      logger_.error("Failed to create syscall log file: " + log_path_syscall_);
    }
  } else {
    trace_file_syscall_ = nullptr;
  }

  if(env_.trace_modules.pcie) {
    trace_file_pcie_ = fopen(log_path_pcie.c_str(), "w");
    if (trace_file_pcie_) {
      fprintf(trace_file_pcie_, "%s\n\n", full_date_time);
      fprintf(trace_file_pcie_, "pcie_log\n");
      fflush(trace_file_pcie_);
    } else {
      logger_.error("Failed to create pcie log file: " + log_path_pcie);
    }
  } else {
    trace_file_pcie_ = nullptr;
  }
  if(env_.trace_modules.nvlink) {
    trace_file_nvlink_ = fopen(log_path_nvlink_.c_str(), "w");
    if (trace_file_nvlink_) {
      fprintf(trace_file_nvlink_, "%s\n\n", full_date_time);
      fprintf(trace_file_nvlink_, "nvlink_log\n");
      fflush(trace_file_nvlink_);
    } else {
      logger_.error("Failed to create nvlink log file: " + log_path_nvlink_);
    }
  } else {
    trace_file_nvlink_ = nullptr;
  }

  initRPC();
  
}

UprobeProfiler::~UprobeProfiler() {
  // 关闭文件
}

// void UprobeProfiler::add_trace_data(
//     const std::tuple<std::string, int, int64_t, std::string> &trace_data) {
//   // 写入日志文件
//   if (trace_file_func_) {
//     fprintf(trace_file_func_, "%-12s %-5s %-6s %s\n",
//             std::get<0>(trace_data).c_str(),
//             (std::get<1>(trace_data) == 0) ? "ENTRY" : "EXIT",
//             std::to_string(std::get<2>(trace_data)).c_str(),
//             std::get<3>(trace_data).c_str());

//     // 立即刷新缓冲区，确保数据写入文件
//     fflush(trace_file_func_);
//   } else {
//     logger_.error("Trace file not available, data only printed to console");
//   }
// }

void UprobeProfiler::close() {
  // 发送所有缓冲的数据
  if (rpc_enabled_) {
    std::lock_guard<std::mutex> lock(rpc_mutex_);
    if (!trace_buffer_.empty() || !gpu_buffer_.empty() ||
        !cpu_buffer_.empty() || !kmem_buffer_.empty() || !net_buffer_.empty() ||
        !io_buffer_.empty()) {
      sendTraceBatch();
    }
  }
  // 关闭文件
  if(env_.trace_modules.func) {
    logger_.info("Closing trace file: " + log_path_func_);
    if (trace_file_func_) {
      fclose(trace_file_func_);
      trace_file_func_ = nullptr;
    }
  }
  if(env_.trace_modules.gpu) {
    logger_.info("Closing trace file: " + log_path_gpu_);
    if (trace_file_gpu_) {
      fclose(trace_file_gpu_);
      trace_file_gpu_ = nullptr;
    }

    if (trace_file_gpu_memleak_) {
      logger_.info("Closing trace file: " + log_path_gpu_memleak_);
      fclose(trace_file_gpu_memleak_);
      trace_file_gpu_memleak_ = nullptr;
    } 

    if (trace_file_gpu_memevent_) {
      logger_.info("Closing trace file: " + log_path_gpu_memevent_);
      fclose(trace_file_gpu_memevent_);
      trace_file_gpu_memevent_ = nullptr;
    }

    if (trace_file_gpu_memcpy_) {
      logger_.info("Closing trace file: " + log_path_gpu_memcpy_);
      fclose(trace_file_gpu_memcpy_);
      trace_file_gpu_memcpy_ = nullptr;
    }
    if (trace_file_gpu_kernlaunch_) {
      logger_.info("Closing trace file: " + log_path_gpu_kernlaunch_);
      fclose(trace_file_gpu_kernlaunch_);
      trace_file_gpu_kernlaunch_ = nullptr;
    }

      if (trace_file_gpu_memfragment_) {
        logger_.info("Closing trace file: " + log_path_gpu_memfragment_);
        fclose(trace_file_gpu_memfragment_);
        trace_file_gpu_memfragment_ = nullptr;
      }


  }

  if(env_.trace_modules.cpu) {
    logger_.info("Closing trace file: " + log_path_cpu_);
    if (trace_file_cpu_) {
      fclose(trace_file_cpu_);
      trace_file_cpu_ = nullptr;
    }
  }

  if(env_.trace_modules.kmem) {
    logger_.info("Closing trace file: " + log_path_kmem_);
    if (trace_file_kmem_) {
      fclose(trace_file_kmem_);
      trace_file_kmem_ = nullptr;
    }
  }

  if(env_.trace_modules.net) {
    logger_.info("Closing trace file: " + log_path_net_);
    if (trace_file_net_) {
      fclose(trace_file_net_);
      trace_file_net_ = nullptr;
    }
  }

  if(env_.trace_modules.io) {
    logger_.info("Closing trace file: " + log_path_io_);
    if (trace_file_io_) {
      fclose(trace_file_io_);
      trace_file_io_ = nullptr;
    }
  }
  if(env_.trace_modules.syscall) {
    logger_.info("Closing trace file: " + log_path_syscall_);
    if (trace_file_syscall_) {
      fclose(trace_file_syscall_);
      trace_file_syscall_ = nullptr;
    }
  }

  if(env_.trace_modules.pcie) {
    logger_.info("Closing trace file: " + log_path_pcie);
    if (trace_file_pcie_) {
      fclose(trace_file_pcie_);
      trace_file_pcie_ = nullptr;
    }
  }
  if(env_.trace_modules.nvlink){
    logger_.info("Closing trace file: " + log_path_nvlink_);
    if (trace_file_nvlink_) {
      fclose(trace_file_nvlink_);
      trace_file_nvlink_ = nullptr;
    }
  }
}

std::string UprobeProfiler::get_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;

  // 使用 std::strftime 和 std::stringstream 代替 fmt::format
  char buffer[20]; // YYYY-MM-DD HH:MM:SS
  std::tm *timeinfo = std::localtime(&now_time_t);
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);

  // 使用 stringstream 添加毫秒部分
  std::stringstream ss;
  ss << buffer << '.' << std::setfill('0') << std::setw(3) << ms.count();

  return ss.str();
}

} // namespace NeuTracer