#include "client/tracer.h"
#include <fstream>
#include <mutex>
#include <future>
#include <signal.h>

namespace NeuTracer {

Tracer::Tracer(const std::string uprobe_cfg_path, const std::string info_level,const myenv& env)
               
    : uprobe_cfg_(loadJSON(uprobe_cfg_path)),
      pid_(0), logger_(info_level),env_(env), profiler_(logger_,env),
      func_snoop_(uprobe_cfg_, env, logger_, profiler_),
      gpu_snoop_(env,logger_,profiler_),cpu_snoop_(env,logger_,profiler_),
      kmem_snoop_(env,logger_,profiler_),net_snoop_(env,logger_,profiler_),
      io_snoop_(env,logger_,profiler_),syscall_snoop_(env,logger_,profiler_),
      pcie_snoop_(env, logger_, profiler_),nvlink_snoop_(env, logger_, profiler_) { 

        if (info_level != "none" && info_level != "info" && info_level != "debug" &&
            info_level != "error") {
          logger_.error("Invalid log level: " + info_level);
          exit(0);
        }
        // auto py_filter_cfg_ = uprobe_cfg_.at("py_filter");
        // auto py2lib_cfg_ = uprobe_cfg_.at("py2lib");

        // if (py_filter_cfg_.is_object()) {
        //   for (const auto &item : py_filter_cfg_.items()) {
        //     py_filter_.push_back(item.value().get<std::string>());
        //   }
        // } else {
        //   logger_.error("Invalid py_filter configuration.");
        //   exit(0);
        // }
        
        auto cuda_cfg_ = uprobe_cfg_.at("cuda_func_sym");

        if(cuda_cfg_.is_object()){
          for(const auto &item : cuda_cfg_.items()){
            cuda_func_.push_back(item.value().get<std::string>());
          }
        }else{
                logger_.error("Invalid cuda_func_sym configuration.");
                exit(0);
        }

}

Tracer::~Tracer() {
}

json Tracer::loadJSON(std::string path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    logger_.error("Failed to open config file: " + path);
    exit(0);
  }
  json config = json::parse(file);
  return config;
}
void Tracer::run() {
  if(env_.trace_modules.io)
    io_snoop_.attach_bpf();
  if(env_.trace_modules.net)
    net_snoop_.attach_bpf();
  if(env_.trace_modules.gpu)
    gpu_snoop_.attach_bpf();
  if(env_.trace_modules.kmem)
    kmem_snoop_.attach_bpf();
  if(env_.trace_modules.cpu)
    cpu_snoop_.attach_bpf();
  if(env_.trace_modules.func)
    func_snoop_.attach_bpf();
  if (env_.trace_modules.python) {
    // TODO: Implement Python tracing start logic
  }
  if(env_.trace_modules.syscall){
    syscall_snoop_.attach_bpf();
  }
  if(env_.trace_modules.pcie){
    pcie_snoop_.attach_bpf();
  }
  if(env_.trace_modules.nvlink){
    nvlink_snoop_.attach_bpf();
  }
  logger_.info("NeuTracer start tracing...");
  pid_ = getpid();
}
void Tracer::stat(double cur_time, double prev_time) {
    cpu_snoop_.record_stats(std::cout, cur_time, cur_time - prev_time, env_.pid);
}

void Tracer::close() {
  logger_.info("Stopping all components...");
  if(env_.trace_modules.net) {
    net_snoop_.stop_trace();
    logger_.info("net_snoop stop.");
  } 
  if(env_.trace_modules.io) {
    io_snoop_.stop_trace();
    logger_.info("io_snoop stop.");
  }
  if(env_.trace_modules.kmem) {
    kmem_snoop_.stop_trace();
    logger_.info("kmem_snoop stop.");
  }
  if(env_.trace_modules.gpu) {
    gpu_snoop_.stop_trace();
    logger_.info("gpu_snoop stop.");
  }
  if(env_.trace_modules.cpu) {
    cpu_snoop_.stop_trace();
    logger_.info("cpu_snoop stop.");
  }
  if(env_.trace_modules.func) {
    func_snoop_.stop_trace();
    logger_.info("func_snoop stop.");
  }
  if (env_.trace_modules.python) {
    // TODO: Implement Python tracing stop logic
    logger_.info("Python tracing stop.");
  }
  if(env_.trace_modules.syscall){
    syscall_snoop_.stop_trace();
    logger_.info("syscall_snoop stop");
  }
  if(env_.trace_modules.pcie){
    pcie_snoop_.stop_trace();
    logger_.info("pcie_snoop stop");
  }
  if(env_.trace_modules.nvlink){
    nvlink_snoop_.stop_trace();
    logger_.info("nvlink_snoop stop");
  }
  profiler_.close();
  logger_.info("NeuTracer stop tracing.");
}

void Tracer::clean() {
  std::lock_guard<std::mutex> lock(file_mutex_);
    
    // 先安全关闭现有文件
    if (trace_file_ && trace_file_ != stdout && trace_file_ != stderr) {
        fclose(trace_file_);
    }
    
    // 打开新文件（使用"a"模式避免覆盖已有数据）
    trace_file_ = fopen(log_path_.c_str(), "a");  // 改为追加模式
    if (!trace_file_) {
        logger_.error("Failed to reopen log file: " + log_path_);
        trace_file_ = stderr;  // 降级到标准错误输出
    }
}

std::vector<std::tuple<std::string, int, int, std::string>>
Tracer::processTraceData(const std::string &work, int pid, int ts,
                         const std::string &op, const std::string &func) {

  std::vector<std::tuple<std::string, int, int, std::string>> result;

  // 移除换行符
  std::string func_name = func;
  if (!func_name.empty() && func_name.back() == '\n') {
    func_name.pop_back();
  }

  // 添加基本跟踪数据
  result.push_back(std::make_tuple(work, pid, ts, op + " " + func_name));

  // 检查是否有对应的库函数
  // auto it = py2lib_.find(func_name);
  // if (it != py2lib_.end()) {
  //   // 添加libtorch版本
  //   std::string libtorch_name = func_name;
  //   size_t pos = libtorch_name.find("pytorch");
  //   if (pos != std::string::npos) {
  //     libtorch_name.replace(pos, 7, "libtorch");
  //     result.push_back(
  //         std::make_tuple(work, pid, ts, op + " " + libtorch_name));
  //   }

  //   // 添加其他关联的库函数
  //   for (const auto &lib_func : it->second) {
  //     result.push_back(std::make_tuple(work, pid, ts, op + " " + lib_func));
  //   }
  // }

  return result;
}


} // namespace NeuTracer