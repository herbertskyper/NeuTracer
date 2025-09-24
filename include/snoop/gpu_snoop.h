#ifndef GPUSNOOP_H
#define GPUSNOOP_H

#include <bpf/libbpf.h>
#include <cstdint>
#include <thread>
#include <cupti.h>

#include "config.h"
#include "detector/leak_detector.h"
#include "detector/fragment_detector.h"
#include "utils/Logger.h"
#include "utils/SymUtils.h"
#include "utils/UprobeProfiler.h"

#include "gpuevent_snoop.skel.h" // 添加这行，包含自动生成的骨架头文件

// #include "utils/Guard.h"
// #include "utils/SymUtils.h"

namespace NeuTracer {
// 内核函数调用信息结构体
struct KernelFuncInfo {
  std::string name;    // 函数名称
  uint64_t call_count; // 调用次数

  // 构造函数
  KernelFuncInfo(const std::string &n = "", uint64_t count = 1)
      : name(n), call_count(count) {}
};

// struct CudaMemoryAlloc {
//   uint64_t address;
//   uint64_t size;
//   double duration_ms;
//   cudaError_t ret;
//   std::chrono::steady_clock::time_point timestamp; // 添加时间戳
//   uint64_t caller_func_off;                        // 调用者函数偏移地址

//   CudaMemoryAlloc()
//       : address(0), size(0), duration_ms(0), ret(cudaSuccess),
//         timestamp(std::chrono::steady_clock::now()), caller_func_off(0) {}

//   CudaMemoryAlloc(uint64_t addr, uint64_t sz, double dur, cudaError_t r,
//                   std::chrono::steady_clock::time_point ts, uint64_t caller_off)
//       : address(addr), size(sz), duration_ms(dur), ret(r), timestamp(ts),
//         caller_func_off(caller_off) {}
// };

class GPUSnoop {
public:
  std::thread rb_thread_;
  std::thread process_monitor_thread_;

  GPUSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
      : env_(env), logger_(logger), profiler_(profiler) {
    lastActivityTime_ = std::chrono::steady_clock::now();
  };
  bool attach_bpf();
  void stop_trace();
  void setIdleTimeout(int seconds) { idleTimeoutSec_ = seconds; }
  // 新增：初始化泄漏检测器
  void initLeakDetector(Logger &logger, UprobeProfiler &profiler,
                           uint64_t threshold = 10 * 1024 * 1024);
  void initFragmentDetector(Logger &logger, UprobeProfiler &profiler);
                           
  void updateMemoryUsageHistory(uint64_t current_usage);

private:
  // 内核函数调用统计哈希表
  std::unordered_map<uint64_t, KernelFuncInfo> kernel_func_stats_;
  // 互斥锁，保护哈希表的并发访问
  std::mutex kernel_func_stats_mutex_;

  std::unordered_map<uint32_t, std::map<uint64_t, CudaMemoryAlloc>> memory_map;
  std::mutex memory_map_mutex_;

  std::chrono::steady_clock::time_point lastActivityTime_;
  int idleTimeoutSec_ = 30; // 默认空闲超时时间为60秒，可根据需要调整
  myenv env_;
  Logger logger_;
  UprobeProfiler &profiler_;
  SymUtils *symUtils_;
  uint64_t total_size_ = 0;
  int process_event(void *ctx, void *data, size_t data_sz);
  int process_memleak_event(void *ctx, void *data, size_t data_sz);
  int process_memcpy_event(void *ctx, size_t data_sz);
  static int handle_event(void *symUtils_ctx, void *data, size_t data_sz);
  static int handle_memleak_event(void *symUtils_ctx, void *data,
                                  size_t data_sz);
  static int handle_memcpy_event(void *symUtils_ctx, void *data,
                                 size_t data_sz);
  // static int libbpf_print_fn(enum libbpf_print_level level, const char
  // *format,
  //                     va_list args);

  bool start_trace();
  void ring_buffer_thread();
  bool hasExceededProfilingLimit(
      std::chrono::seconds duration,
      const std::chrono::steady_clock::time_point &startTime);
  std::string get_timestamp();
  void cleanUpTerminatedProcesses();
  void process_monitor_thread();
  bool detectCudaMemoryShortage();
  void detectCudaMemoryLeaks();


  // CUPTI回调函数
  static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
  static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
  
  // 内存活动处理函数
  void processMemoryActivity(CUpti_ActivityMemory *memory);
  void processMemcpyActivity(CUpti_ActivityMemcpy *memcpy);
  void processKernelActivity(CUpti_ActivityKernel *kernel);
  void setupCuptiCallbacks();
  bool enableCUPTIActivities();

  struct ring_buffer *ringBuffer;
  struct ring_buffer *memleak_ringBuffer;
  struct ring_buffer *memcpy_ringBuffer;
  bool exiting_ = false;
  std::vector<bpf_link *> links;
  struct gpuevent_snoop_bpf *skel;
  std::unique_ptr<LeakDetector> leak_detector_;
  std::unique_ptr<FragmentDetector> fragment_detector_;
};

} // namespace NeuTracer

#endif // GPUSNOOP_H
