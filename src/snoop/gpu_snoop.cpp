// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <argp.h>
#include <bpf/libbpf.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>


#include "snoop/gpu_event.h"
#include "snoop/gpu_snoop.h"
#include "utils/Logger.h"
#include "utils/SymUtils.h"
#include "utils/UprobeProfiler.h"

#ifdef FBCODE_STROBELIGHT
#include "strobelight/src/profilers/gpuevent_snoop/gpuevent_snoop.skel.h"
#else
#include "gpuevent_snoop.skel.h"
#endif

namespace NeuTracer {

#define MAX_FUNC_DISPLAY_LEN 32

static const int64_t RINGBUF_MAX_ENTRIES = 64 * 1024 * 1024;
static const std::string kCudaLaunchSymName = "cudaLaunchKernel";
static const std::string kCudaMallocSymName = "cudaMalloc";
static const std::string kCudaFreeSymName = "cudaFree";
static const std::string kCudaMemcpySymName1 = "cudaMemcpy";
static const std::string kCudaMemcpySymName2 =
    "cudaMemcpyAsync"; // 一般不会直接用cudamamcpy，而是用这个api

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
                           va_list args) {
  if (level == LIBBPF_DEBUG) {
    return 0;
  }
  return vfprintf(stderr, format, args);
}

int GPUSnoop::handle_memcpy_event(void *ctx, void *data, size_t data_sz) {
  GPUSnoop *self = static_cast<GPUSnoop *>(ctx);
  return self->process_memcpy_event(data, data_sz);
}

// 实现正确的 cudaMemcpy 事件处理函数
int GPUSnoop::process_memcpy_event(void *data, size_t data_sz) {
  const struct cuda_memcpy *e = (struct cuda_memcpy *)data;

  // 获取当前时间点
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);

  // 获取毫秒部分
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;

  // 格式化时间
  std::stringstream time_ss;
  time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
          << '.' << std::setfill('0') << std::setw(3) << ms.count();

  // 计算持续时间(毫秒)和带宽
  double duration_ms = (e->end_time - e->start_time) / 1000000.0;
  double bandwidth_mbps = 0;

  if (duration_ms > 0) {
    // 计算MB/s (将字节转换为MB,再将纳秒转换为秒)
    bandwidth_mbps = (e->count / (1024.0 * 1024.0)) / (duration_ms / 1000.0);
  }

  // 确定传输方向描述
  std::string kind_str;
  switch (e->kind) {
  case 0:
    kind_str = "H2H";
    break; // 设备到设备
  case 1:
    kind_str = "H2D";
    break; // 设备到主机
  case 2:
    kind_str = "D2H";
    break; // 主机到设备
  case 3:
    kind_str = "D2D";
    break; // 主机到主机
  default:
    kind_str = "DEFAULT";
    break;
  }

  // // 记录日志
  // logger_.info(
  //     "{} CUDA MEMCPY {} dst=0x{:<16x} src=0x{:<16x} size={} bytes,
  //     duration={:.3f}ms, bandwidth={:.2f}MB/s", time_ss.str(), kind_str,
  //     e->dst,
  //     e->src,
  //     e->count,
  //     duration_ms,
  //     bandwidth_mbps);
  if (profiler_.isRPCEnabled()) {
    UprobeProfiler::GPUTraceItem gpu_data;
    gpu_data.type = UprobeProfiler::GPUTraceItem::EventType::MEMTRANSEVENT;
    gpu_data.timestamp = profiler_.get_timestamp();
    gpu_data.mem_trans_event.pid = e->pid;
    gpu_data.mem_trans_event.mem_trans_rate = bandwidth_mbps;
    gpu_data.mem_trans_event.kind_str = strdup(kind_str.c_str());
    profiler_.sendGPUTrace(gpu_data);
  }
  // 添加到分析器
  profiler_.add_trace_data_gpu_memcpy(
      "CUDA MEMCPY {} dst=0x{:<16x} src=0x{:<16x} size={} bytes, "
      "duration={:.3f}ms, bandwidth={:.2f}MB/s\n",
      kind_str, e->dst, e->src, e->count, duration_ms, bandwidth_mbps);

  return 0;
}

int GPUSnoop::handle_memleak_event(void *ctx, void *data, size_t data_sz) {
  GPUSnoop *self = static_cast<GPUSnoop *>(ctx);
  return self->process_memleak_event(self->symUtils_, data, data_sz);
}

int GPUSnoop::process_memleak_event(void *symUtils_ctx, void *data,
                                    size_t data_sz) {
  const struct memleak_event *e = (struct memleak_event *)data;

  SymUtils *symUtils = static_cast<SymUtils *>(symUtils_ctx);
  // Logger& logger = symUtils->logger_;
  // SymbolInfo symInfo = symUtils->getSymbolByAddr(e->caller_func_off, env_.args);
  auto stack = symUtils->getStackByAddrs((uint64_t *)e->ustack, e->ustack_sz);
  // 获取当前时间点
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);

  // 获取毫秒部分
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;

  // 格式化时间
  std::stringstream time_ss;
  time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
          << '.' << std::setfill('0') << std::setw(3) << ms.count();

  // 操作类型（分配或释放）
  std::string op_type = e->event_type == CUDA_MALLOC ? "MALLOC" : "FREE";
  double duration_ms = (e->end - e->start) / 1000000.0;
  cudaError_t ret = static_cast<cudaError_t>(e->ret);
  // logger_.info(
  //     "{} CUDA {} pid={} addr=0x{:<16x} size={} bytes, ret={},
  //     duration={:.3f}ms", time_ss.str(), op_type, e->pid, e->device_addr,
  //     e->size, e->ret, duration_ms);
  if (e->ret != 0) {
    if (ret != cudaSuccess) {
      logger_.error("CUDA error: {}", cudaGetErrorString(ret));
    }
  }

  if (e->event_type == CUDA_MALLOC) {
    auto address = e->device_addr;
    auto size = e->size;
    auto pid = e->pid;
    // logger_.info("[GPU] Stack: ");

    // if(!stack.empty()) {
    // auto frame = stack[1];
    // profiler_.add_trace_data_gpu_memleak(
    // "调用函数: {:016x}: {} @ 0x{:x}+0x{:x}",
    //  frame.address, frame.name.c_str(), frame.address, frame.offset);
    // }
    // if(stack.empty()) {
    //   logger_.warn("No stack trace available for CUDA malloc event.");
    // }
    //? 
    // SymbolInfo symInfo = symUtils->getSymbolByAddr(e->caller_func_off, env_.args);
    // auto stack = symUtils->getStackByAddrs((uint64_t *)e->ustack, e->ustack_sz);
    StackFrame caller_frame = stack.size() > 1 ? stack[1] : StackFrame();
    bool is_frame = stack.size() > 1;
    // if(is_frame) logger_.info("调用函数: {:016x}: {} @ 0x{:x}+0x{:x}",
    //                            caller_frame.address, caller_frame.name.c_str(),
    //                            caller_frame.address, caller_frame.offset);
    // StackFrame caller_frame = StackFrame();
    // bool is_frame = false;
    {
      std::lock_guard<std::mutex> lock(memory_map_mutex_);
      CudaMemoryAlloc alloc(address, size, duration_ms, ret,
                            std::chrono::steady_clock::now(),
                            e->caller_func_off);
      memory_map[pid][address] = alloc;

      total_size_ += e->size;

      if (leak_detector_ && ret == cudaSuccess) {
        // 记录分配到SCALENE检测器
        leak_detector_->recordAllocation(address, size, e->caller_func_off,
                                         total_size_, caller_frame,is_frame);
      }
      if (profiler_.isRPCEnabled()) {
        UprobeProfiler::GPUTraceItem memevent;
        memevent.type = UprobeProfiler::GPUTraceItem::EventType::MEMEVENT;
        memevent.timestamp = profiler_.get_timestamp();
        memevent.mem_event.pid = e->pid;
        memevent.mem_event.mem_size = total_size_;
        profiler_.sendGPUTrace(memevent);
      }
    }

    // logger_.info("内存分配：PID={}, 地址=0x{:x}, 大小={} 字节", pid, address,
    //              size);
    // logger_.info(
    //     "{} CUDA {} pid={} addr=0x{:<16x} size={} bytes, ret={},
    //     duration={:.3f}ms", time_ss.str(), op_type, e->pid, e->device_addr,
    //     e->size,
    //     e->ret,
    //     duration_ms);

    profiler_.add_trace_data_gpu_memevent(
        "CUDA {} pid={} addr=0x{:<16x} caller_offset=0x{:<16x} size={} bytes, "
        "ret={}, duration={:.3f}ms\n",
        op_type, e->pid, e->device_addr, e->caller_func_off, e->size, e->ret,
        duration_ms);
  } else {
    auto address = e->device_addr;
    auto pid = e->pid;
    {
      std::lock_guard<std::mutex> lock(memory_map_mutex_);
      auto caller_func_off = memory_map[pid][address].caller_func_off;
      if (leak_detector_) {
        leak_detector_->recordDeallocation(address, caller_func_off,
                                           memory_map[pid][address].size);
      }

      // memory_map[pid][address].size = 0;
      if (total_size_ > 0)
        total_size_ -= memory_map[pid][address].size;
      memory_map[pid].erase(address);

      if (profiler_.isRPCEnabled()) {
        UprobeProfiler::GPUTraceItem memevent;
        memevent.type = UprobeProfiler::GPUTraceItem::EventType::MEMEVENT;
        memevent.timestamp = profiler_.get_timestamp();
        memevent.mem_event.pid = e->pid;
        memevent.mem_event.mem_size = total_size_;
        profiler_.sendGPUTrace(memevent);
        // logger_.info("当前进程总cuda占用内存：PID={}, 大小={} 字节", pid,
        //              total_size);
      }
    }

    // 记录到内存映射中
    // memory_map[pid][address].size = 0;
    // memory_map[pid][address].duration_ms = duration_ms;
    // memory_map[pid][address].ret = ret;

    // CUDA FREE 事件
    // logger_.info("{} CUDA {} pid={} addr=0x{:<16x} ret={},
    // duration={:.3f}ms",
    //              time_ss.str(), op_type, e->pid, e->device_addr, e->ret,
    //              duration_ms);
    profiler_.add_trace_data_gpu_memevent(
        "CUDA {} pid={} addr=0x{:<16x} caller_offset=0x{:<16x} ret={}, "
        "duration={:.3f}ms\n",
        op_type, e->pid, e->device_addr, e->caller_func_off, e->ret,
        duration_ms);
  }

  return 0;
}

// Ring buffer 事件处理回调
int GPUSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
  GPUSnoop *self = static_cast<GPUSnoop *>(ctx);
  return self->process_event(self->symUtils_, data, data_sz);
}

int GPUSnoop::process_event(void *symUtils_ctx, void *data, size_t data_sz) {

  // printf("hello printf");
  const struct gpukern_sample *e = (struct gpukern_sample *)data;

  SymUtils *symUtils = static_cast<SymUtils *>(symUtils_ctx);
  // Logger& logger = symUtils->logger_;

  SymbolInfo symInfo = symUtils->getSymbolByAddr(e->kern_func_off, env_.args);

  {
    // 使用锁保护哈希表的并发访问
    std::lock_guard<std::mutex> lock(kernel_func_stats_mutex_);

    // 检查函数偏移是否已存在于哈希表中
    auto it = kernel_func_stats_.find(e->kern_func_off);
    if (it != kernel_func_stats_.end()) {
      // 已存在，增加调用计数
      it->second.call_count++;
    } else {
      // 不存在，添加新条目
      kernel_func_stats_[e->kern_func_off] = KernelFuncInfo(symInfo.name, 1);
    }

    if (profiler_.isRPCEnabled()) {
      UprobeProfiler::GPUTraceItem cuda_launch_event;
      cuda_launch_event.type =
          UprobeProfiler::GPUTraceItem::EventType::CUDALAUNCHEVENT;
      cuda_launch_event.timestamp = profiler_.get_timestamp();
      cuda_launch_event.cuda_launch_event.pid = e->pid;
      cuda_launch_event.cuda_launch_event.kernel_name =
          strdup(symInfo.name.c_str());
      cuda_launch_event.cuda_launch_event.num_calls =
          kernel_func_stats_[e->kern_func_off].call_count;
      profiler_.sendGPUTrace(cuda_launch_event);
      // logger_.info("当前进程总cuda占用内存：PID={}, 大小={} 字节", pid,
      //              total_size);
    }
    // logger_.info("kernel_addr:0x{}, name:{}, num:{}\n", e->kern_func_off,
    // symInfo.name.c_str(), kernel_func_stats_[e->kern_func_off].call_count);
    // 使用临时作用域的目的是确保在离开作用域时自动释放锁
  }
  // 获取当前时间点
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);

  // 获取毫秒部分
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;

  // 格式化时间
  std::stringstream time_ss;
  time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
          << '.' << std::setfill('0') << std::setw(3) << ms.count();

  // logger_.info(
  //     "[{}] {} [{}] KERNEL [0x{:x}] STREAM 0x{:<16x} GRID ({},{},{}) BLOCK
  //     ({},{},{}) {}\n", time_ss.str(), e->comm, e->pid, e->kern_func_off,
  //     e->stream,
  //     e->grid_x,
  //     e->grid_y,
  //     e->grid_z,
  //     e->block_x,
  //     e->block_y,
  //     e->block_z,
  //     symInfo.name.substr(0, MAX_FUNC_DISPLAY_LEN) +
  //         (symInfo.name.length() > MAX_FUNC_DISPLAY_LEN ? "..." : ""));
  profiler_.add_trace_data_gpu_kernlaunch(
      "[{}] {} [{}] KERNEL [0x{:x}] STREAM 0x{:<16x} GRID ({},{},{}) BLOCK "
      "({},{},{}) {}\n",
      time_ss.str(), e->comm, e->pid, e->kern_func_off, e->stream, e->grid_x,
      e->grid_y, e->grid_z, e->block_x, e->block_y, e->block_z,
      symInfo.name.substr(0, MAX_FUNC_DISPLAY_LEN) +
          (symInfo.name.length() > MAX_FUNC_DISPLAY_LEN ? "..." : ""));

  if (env_.args) {
    // logger_.info("[GPU] Args: ");
    profiler_.add_trace_data_gpu_kernlaunch("Args: ");
    for (size_t i = 0; i < symInfo.args.size() && i < MAX_GPUKERN_ARGS; i++) {
      // logger_.info("{} arg{}=0x{:x}\n ", symInfo.args[i], i, e->args[i]);
      profiler_.add_trace_data_gpu_kernlaunch("{} arg{}=0x{:x}\n ",
                                              symInfo.args[i], i, e->args[i]);
    }
  }

  if (env_.stacks) {
    // logger_.info("[GPU] Stack: ");
    profiler_.add_trace_data_gpu_kernlaunch("Stack: ");
    auto stack = symUtils->getStackByAddrs((uint64_t *)e->ustack, e->ustack_sz);
    for (auto &frame : stack) {
      // frame.print();
      // logger_.info("{:016x}: {} @ 0x{:x}+0x{:x}", frame.address,
      //              frame.name.c_str(), frame.address, frame.offset);
      profiler_.add_trace_data_gpu_kernlaunch("{:016x}: {} @ 0x{:x}+0x{:x}",
                                              frame.address, frame.name.c_str(),
                                              frame.address, frame.offset);
    }
  }
  if (profiler_.isRPCEnabled()) {
    auto stack = symUtils->getStackByAddrs((uint64_t *)e->ustack, e->ustack_sz);
    std::string stackmessage;
    for (auto &frame : stack) {
      if (!stackmessage.empty())
        stackmessage += "\n"; // 在非第一个函数前添加空格
      stackmessage += frame.name;
    }
    UprobeProfiler::GPUTraceItem stackevent;
    stackevent.type = UprobeProfiler::GPUTraceItem::EventType::CALLSTACK;
    stackevent.timestamp = profiler_.get_timestamp();
    stackevent.callstack_event.pid = e->pid;
    stackevent.callstack_event.stack_message = strdup(stackmessage.c_str());
    profiler_.sendGPUTrace(stackevent);
  }
  // logger_.info("{:-<40}\n", '-');
  profiler_.add_trace_data_gpu_kernlaunch("{:-<40}\n", '-');
  return 0;
}

bool GPUSnoop::hasExceededProfilingLimit(
    std::chrono::seconds duration,
    const std::chrono::steady_clock::time_point &startTime) {
  if (duration.count() == 0) { // 0 = profle forever
    return false;
  }

  if (std::chrono::steady_clock::now() - startTime >= duration) {
    logger_.info("[GPU] Done Profiling: exceeded duration of {}s.\n",
                 duration.count());
    return true;
  }
  return false;
}

bool GPUSnoop::attach_bpf() {

  /* Set up libbpf errors and debug info callback */
  libbpf_set_print(libbpf_print_fn);

  /* Load and verify BPF application */
  skel = gpuevent_snoop_bpf__open();
  if (!skel) {
    logger_.error("Failed to open and load [GPU] BPF skeleton - err: {}",
                  errno);
    return -1;
  }

  // SymUtils symUtils(env_.pid,logger_,profiler_,env_);
  symUtils_ = new SymUtils(env_.pid, logger_, profiler_, env_);

  logger_.info("GPU pid: {}", env_.pid);

  /* Init Read only variables and maps */
  bpf_map__set_max_entries(
      skel->maps.rb, env_.rb_count > 0 ? env_.rb_count : RINGBUF_MAX_ENTRIES);
  bpf_map__set_max_entries(skel->maps.memleak_events_rb,
                           env_.rb_count > 0 ? env_.rb_count
                                             : RINGBUF_MAX_ENTRIES);
  bpf_map__set_max_entries(skel->maps.cuda_memcpy_rb,
                           env_.rb_count > 0 ? env_.rb_count
                                             : RINGBUF_MAX_ENTRIES);
  skel->rodata->prog_cfg.capture_args = env_.args;

  /* Load & verify BPF programs */
  int err = gpuevent_snoop_bpf__load(skel);
  if (err) {
    logger_.error("Failed to load and verify GPU BPF skeleton");
    return -1;
  }

  // Guard guard([&] {
  //   gpuevent_snoop_bpf__destroy(skel);
  //   for (auto link : links) {
  //     bpf_link__destroy(link);
  //   }
  //   ring_buffer__free(ringBuffer);
  // });

  auto offsets = symUtils_->findSymbolOffsets(kCudaLaunchSymName);
  if (offsets.empty()) {
    logger_.error("Failed to find symbol {}", kCudaLaunchSymName);
    // return -1;
  }
  for (auto &offset : offsets) {
    auto link = bpf_program__attach_uprobe(skel->progs.handle_cuda_launch,
                                           false /* retprobe */, env_.pid,
                                           offset.first.c_str(), offset.second);
    if (link) {
      links.emplace_back(link);
    }
  }

  offsets = symUtils_->findSymbolOffsets(kCudaMallocSymName);
  if (offsets.empty()) {
    logger_.error("Failed to find symbol {}", kCudaMallocSymName);
    // return -1;
  }
  for (auto &offset : offsets) {
    auto link = bpf_program__attach_uprobe(skel->progs.handle_cuda_malloc,
                                           false /* retprobe */, -1,
                                           offset.first.c_str(), offset.second);
    if (link) {
      links.emplace_back(link);
    }

    auto ret_link = bpf_program__attach_uprobe(
        skel->progs.handle_cuda_malloc_ret, true /* retprobe */, -1,
        offset.first.c_str(), offset.second);
    if (ret_link) {
      links.emplace_back(ret_link);
    }
  }

  offsets = symUtils_->findSymbolOffsets(kCudaFreeSymName);
  if (offsets.empty()) {
    logger_.error("Failed to find symbol {}", kCudaFreeSymName);
    // return -1;
  }
  for (auto &offset : offsets) {
    auto link = bpf_program__attach_uprobe(skel->progs.handle_cuda_free,
                                           false /* retprobe */, -1,
                                           offset.first.c_str(), offset.second);
    if (link) {
      links.emplace_back(link);
    }

    auto ret_link = bpf_program__attach_uprobe(
        skel->progs.handle_cuda_free_ret, true /* retprobe */, -1,
        offset.first.c_str(), offset.second);
    if (ret_link) {
      links.emplace_back(ret_link);
    }
  }

  offsets = symUtils_->findSymbolOffsets(kCudaMemcpySymName1);
  if (offsets.empty()) {
    logger_.error("Failed to find symbol {}", kCudaMemcpySymName1);
    // return -1;
  }
  for (auto &offset : offsets) {
    auto link = bpf_program__attach_uprobe(skel->progs.handle_cuda_memcpy,
                                           false /* retprobe */, env_.pid,
                                           offset.first.c_str(), offset.second);
    if (link) {
      links.emplace_back(link);
    }

    auto ret_link = bpf_program__attach_uprobe(
        skel->progs.handle_cuda_memcpy_ret, true /* retprobe */, env_.pid,
        offset.first.c_str(), offset.second);
    if (ret_link) {
      links.emplace_back(ret_link);
    }
  }

  offsets = symUtils_->findSymbolOffsets(kCudaMemcpySymName2);
  if (offsets.empty()) {
    logger_.error("Failed to find symbol {}", kCudaMemcpySymName2);
    // return -1;
  }
  for (auto &offset : offsets) {
    auto link = bpf_program__attach_uprobe(skel->progs.handle_cuda_memcpy,
                                           false /* retprobe */, env_.pid,
                                           offset.first.c_str(), offset.second);
    if (link) {
      links.emplace_back(link);
    }

    auto ret_link = bpf_program__attach_uprobe(
        skel->progs.handle_cuda_memcpy_ret, true /* retprobe */, env_.pid,
        offset.first.c_str(), offset.second);
    if (ret_link) {
      links.emplace_back(ret_link);
    }
  }
  /* Set up ring buffer polling */
  ringBuffer = ring_buffer__new(bpf_map__fd(skel->maps.rb), handle_event,
                                (void *)this, nullptr);
  if (!ringBuffer) {
    logger_.error("Failed to create ring buffer");
    return -1;
  }

  /* Set up ring buffer polling for memory events */
  memleak_ringBuffer =
      ring_buffer__new(bpf_map__fd(skel->maps.memleak_events_rb),
                       handle_memleak_event, (void *)this, nullptr);
  if (!memleak_ringBuffer) {
    logger_.error("Failed to create memory events ring buffer");
    return -1;
  }

  /* Set up ring buffer polling for memory events */
  memcpy_ringBuffer =
      ring_buffer__new(bpf_map__fd(skel->maps.cuda_memcpy_rb),
                       handle_memcpy_event, (void *)this, nullptr);
  if (!memcpy_ringBuffer) {
    logger_.error("Failed to create memory events ring buffer");
    return -1;
  }

  // 设置退出标志并启动线程
  exiting_ = false;
  rb_thread_ = std::thread(&GPUSnoop::ring_buffer_thread, this);
  process_monitor_thread_ =
      std::thread(&GPUSnoop::process_monitor_thread, this);
  // ring_buffer_thread();

  logger_.info("GPU Ring buffer polling started");
  initLeakDetector(logger_, profiler_, 10 * 1024 * 1024);
  initFragmentDetector(logger_, profiler_);
  return 1;
};

void GPUSnoop::ring_buffer_thread() {

  auto startTime = std::chrono::steady_clock::now();
  auto duration = std::chrono::seconds(env_.duration_sec);

  std::time_t ttp =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  logger_.info("Started profiling at {}", std::ctime(&ttp));

  while (!exiting_ && !hasExceededProfilingLimit(duration, startTime)) {
    // 轮询前记录当前时间
    auto pollStartTime = std::chrono::steady_clock::now();

    // 轮询环形缓冲区
    int err = ring_buffer__poll(ringBuffer, 100 /* timeout, ms */);

    if (err == -EINTR) {
      err = 0;
      break;
    }
    if (err < 0) {
      logger_.error("Error polling GPU perf buffer: {}", err);
      continue;
    }

    // int memleak_err = 0;
    // 轮询内存事件环形缓冲区
    int memleak_err =
        ring_buffer__poll(memleak_ringBuffer, 100 /* timeout, ms */);
    if (memleak_err == -EINTR) {
      memleak_err = 0;
      break;
    }
    if (memleak_err < 0) {
      logger_.error("Error polling GPU memory events buffer: {}", memleak_err);
      continue;
      // 继续下一次轮询
    }

    // 轮询内存事件环形缓冲区
    int memcpy_err =
        ring_buffer__poll(memcpy_ringBuffer, 100 /* timeout, ms */);
    if (memcpy_err == -EINTR) {
      memcpy_err = 0;
      break;
    }
    if (memcpy_err < 0) {
      logger_.error("Error polling GPU memory events buffer: {}", memcpy_err);
      continue;
      // 继续下一次轮询
    }

    // 检查是否收到了数据
    if (memcpy_err > 0 || memleak_err > 0 || err > 0) {
      // 收到数据，更新最后一次活动时间
      lastActivityTime_ = std::chrono::steady_clock::now();
    } else {
      // 未收到数据，检查是否超过了空闲超时时间
      auto currentTime = std::chrono::steady_clock::now();
      auto idleTime = std::chrono::duration_cast<std::chrono::seconds>(
                          currentTime - lastActivityTime_)
                          .count();

      // 如果空闲时间超过阈值，退出循环
      // if (idleTimeoutSec_ > 0 && idleTime >= idleTimeoutSec_) {
      //   logger_.info("No activity for {} seconds, stopping GPU profiling",
      //                idleTime);
      //   break;
      // }
    }
  }
  ring_buffer__consume(ringBuffer);
  ring_buffer__consume(memleak_ringBuffer);
  ring_buffer__consume(memcpy_ringBuffer);

  ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  // logger_.info("Stopped GPU profiling at {}", std::ctime(&ttp));

  logger_.info("GPU Ring buffer polling thread stopped");
}

bool isPIDActive(pid_t pid) {
  std::string procPath = "/proc/" + std::to_string(pid);
  struct stat statBuf;

  // 检查 /proc/[pid] 目录是否存在
  if (stat(procPath.c_str(), &statBuf) != 0) {
    // 目录不存在，进程已结束
    return false;
  }

  return true;
}

// 清理已结束进程的内存映射
void GPUSnoop::cleanUpTerminatedProcesses() {
  std::lock_guard<std::mutex> lock(memory_map_mutex_);
  std::vector<pid_t> terminatedPIDs;

  // 找出所有已结束的进程
  for (const auto &[pid, memMap] : memory_map) {
    if (!isPIDActive(pid)) {
      terminatedPIDs.push_back(pid);
    }
  }

  // 清理已结束进程的内存映射
  for (pid_t pid : terminatedPIDs) {
    // 计算未释放的内存总量
    uint64_t leakedSize = 0;
    size_t allocationCount = memory_map[pid].size();

    if (allocationCount > 0) {
      for (const auto &[addr, alloc] : memory_map[pid]) {
        leakedSize += alloc.size;
      }

      logger_.info("[GPU] 进程 {} 已终止但有 {} 个未释放的内存分配，总计 {} "
                   "字节可能泄漏",
                   pid, allocationCount, leakedSize);
    }

    // 移除该进程的所有内存记录
    memory_map.erase(pid);
    logger_.info("已清理已终止进程 PID={} 的内存映射记录", pid);
  }
}

// 在 GPUSnoop 类中添加新的成员函数和变量
// 检测 CUDA 内存不足
bool GPUSnoop::detectCudaMemoryShortage() {
  // logger_.info("检测 CUDA 内存不足...");
  // 获取当前设备
  int currentDevice = 0;
  cudaGetDevice(&currentDevice);

  // 获取设备属性
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, currentDevice);

  // 获取当前设备的内存使用情况
  size_t freeMem = 0;
  size_t totalMem = 0;
  cudaError_t memError = cudaMemGetInfo(&freeMem, &totalMem);

  if (memError != cudaSuccess) {
    logger_.error("获取 CUDA 内存信息失败: {}", cudaGetErrorString(memError));
    return false;
  }

  // 检查是否存在近期分配失败的情况
  bool recentAllocationFailure = false;
  uint64_t use_mem = 0;
  {
    std::lock_guard<std::mutex> lock(memory_map_mutex_);

    // 检查最近30秒内的分配错误
    auto now = std::chrono::steady_clock::now();

    for (const auto &[pid, memMap] : memory_map) {
      for (const auto &[addr, alloc] : memory_map[pid]) {
        // 检查是否存在最近的分配错误
        if (alloc.ret != cudaSuccess) {
          recentAllocationFailure = true;
          logger_.warn("检测到最近的 CUDA 内存分配失败: PID={}, 错误={}", pid,
                       cudaGetErrorString(alloc.ret));
          logger_.warn("分配地址: 0x{:x}, 大小: {} bytes, 时间: {}", addr,
                       alloc.size,
                       std::chrono::duration_cast<std::chrono::seconds>(
                           now - alloc.timestamp)
                           .count());
          continue; // 跳过此分配
        }
        use_mem += alloc.size;
      }
      if (recentAllocationFailure)
        break;
    }
  }
  // 计算内存使用率
  double memoryUsage = 100.0 * use_mem / totalMem;
  // 内存使用率高且有近期分配失败，认为可能出现内存不足
  constexpr double MEMORY_THRESHOLD = 80; // 80% 内存使用率为警告阈值

  if (memoryUsage > MEMORY_THRESHOLD || recentAllocationFailure) {
    logger_.warn("CUDA 内存不足警告: 设备={}, 总内存={}MB, 该项目已使用={}MB "
                 "({}%), 可用={}MB",
                 currentDevice, totalMem / (1024 * 1024),
                 use_mem / (1024 * 1024), memoryUsage, freeMem / (1024 * 1024));

    return true;
  }

  return false;
}

// 检测 CUDA 内存泄漏
void GPUSnoop::detectCudaMemoryLeaks() {
  const int LEAK_THRESHOLD_SEC = 300; // 5分钟未释放视为可能泄漏
  auto now = std::chrono::steady_clock::now();

  std::lock_guard<std::mutex> lock(memory_map_mutex_);

  // 记录可能的泄漏
  std::unordered_map<pid_t, std::vector<std::pair<uint64_t, uint64_t>>>
      potentialLeaks;

  // 按进程检查所有分配
  for (const auto &[pid, memMap] : memory_map) {
    if (!isPIDActive(pid)) {
      continue; // 进程已终止，由 cleanUpTerminatedProcesses 处理
    }

    // 检查此进程的所有分配
    for (const auto &[addr, alloc] : memMap) {
      // 计算分配存在的时间（秒）
      auto allocAge = std::chrono::duration_cast<std::chrono::seconds>(
                          now - alloc.timestamp)
                          .count();

      // 如果分配存在时间超过阈值，可能存在泄漏
      if (allocAge > LEAK_THRESHOLD_SEC) {
        potentialLeaks[pid].push_back({addr, alloc.size});
      }
    }
  }

  // 报告潜在泄漏
  for (const auto &[pid, leaks] : potentialLeaks) {
    if (leaks.empty()) {
      continue;
    }

    uint64_t totalLeakSize = 0;
    for (const auto &[addr, size] : leaks) {
      totalLeakSize += size;
    }

    logger_.warn("检测到潜在 CUDA 内存泄漏: PID={}, 分配数量={}, 总大小={}MB",
                 pid, leaks.size(), totalLeakSize / (1024 * 1024));

    // 详细记录大的泄漏（超过10MB）
    for (const auto &[addr, size] : leaks) {
      if (size > 10 * 1024 * 1024) { // 10MB
        logger_.warn("大型内存泄漏: PID={}, 地址=0x{:x}, 大小={}MB", pid, addr,
                     size / (1024 * 1024));
      }
    }
  }
}

// 进程监控线程
void GPUSnoop::process_monitor_thread() {
  const int CHECK_INTERVAL_SEC = 5;
  const int MEMORY_CHECK_INTERVAL = 30;        // 每30秒检查一次内存不足
  const int LEAK_CHECK_INTERVAL = 60;          // 每60秒检查一次内存泄漏
  const int FRAGMENTATION_CHECK_INTERVAL = 10; // 每10秒检查一次碎片化
  const int LEAKDETECTOR_CHECK_INTERVAL = 20;

  int memoryCheckCounter = 0;
  int leakCheckCounter = 0;
  int fragmentationCheckCounter = 0;
  int leakdetectorCheckCounter = 0;

  logger_.info("进程监控线程已启动");

  while (!exiting_) {
    // 检查并清理已终止的进程
    cleanUpTerminatedProcesses();

    // 周期性检查内存不足
    if (++memoryCheckCounter >= MEMORY_CHECK_INTERVAL / CHECK_INTERVAL_SEC) {
      memoryCheckCounter = 0;
      try {
        detectCudaMemoryShortage();
      } catch (const std::exception &e) {
        logger_.error("内存不足检测异常: {}", e.what());
      }
    }

    // 周期性检查内存泄漏
    if (++leakCheckCounter >= LEAK_CHECK_INTERVAL / CHECK_INTERVAL_SEC) {
      leakCheckCounter = 0;
      try {
        detectCudaMemoryLeaks();
      } catch (const std::exception &e) {
        logger_.error("内存泄漏检测异常: {}", e.what());
      }
    }

    // 周期性检查内存碎片化
    if (++fragmentationCheckCounter >=
        FRAGMENTATION_CHECK_INTERVAL / CHECK_INTERVAL_SEC) {
      fragmentationCheckCounter = 0;
      try {
          if (!fragment_detector_) {
            logger_.error("fragment_detector_ is nullptr, skipping fragmentation check");
            continue;
          }
        fragment_detector_->detectCudaMemoryFragmentation(memory_map);
        
      } catch (const std::exception &e) {
        logger_.error("内存碎片化检测异常: {}", e.what());
      }
    }

    // if (++leakdetectorCheckCounter >=
    //     LEAKDETECTOR_CHECK_INTERVAL / CHECK_INTERVAL_SEC) {
    //   leakdetectorCheckCounter = 0;
    //   try {
    //     leak_detector_->generateLeakReport();
    //   } catch (const std::exception &e) {
    //     logger_.error("内存泄漏plus检测异常: {}", e.what());
    //   }
    // }

    // 休眠一段时间
    std::this_thread::sleep_for(std::chrono::seconds(CHECK_INTERVAL_SEC));
  }

  logger_.info("进程监控线程已停止");
}

// 停止 ring buffer 处理
void GPUSnoop::stop_trace() {
  // 1. 首先标记退出并等待线程结束
  exiting_ = true;
  if (rb_thread_.joinable()) {
    rb_thread_.join();
  }

  if (process_monitor_thread_.joinable()) {
    process_monitor_thread_.join();
  }

  leak_detector_->generateLeakReport();
  cleanUpTerminatedProcesses();

  // 2. 清理 ring buffer
  if (ringBuffer && ringBuffer != nullptr && ringBuffer != (void *)0x11) {
    ring_buffer__free(ringBuffer);
    ringBuffer = nullptr;
  }
  logger_.info("GPU Ring buffer polling thread stopped");

  // 新增：清理内存事件 ring buffer
  if (memleak_ringBuffer) {
    ring_buffer__free(memleak_ringBuffer);
    memleak_ringBuffer = nullptr;
  }

  logger_.info("GPU memleak Ring buffer polling stopped");

  if (memcpy_ringBuffer) {
    ring_buffer__free(memcpy_ringBuffer);
    memcpy_ringBuffer = nullptr;
  }

  logger_.info("GPU memcpy Ring buffer polling stopped");
  // 3. 清理 links
  for (auto link : links) {
    if (link) {
      bpf_link__destroy(link);
      link = nullptr;
    }
  }
  links.clear();

  // 4. 最后清理 BPF skeleton 和 symUtils
  if (skel) {
    gpuevent_snoop_bpf__destroy(skel);
    skel = nullptr;
  }
  logger_.info("GPU BPF skeleton destroyed");
  if (symUtils_) {
    delete symUtils_;
    symUtils_ = nullptr;
  }

  // logger_.info("GPU Ring buffer polling stopped");
}

// 初始化SCALENE检测器
void GPUSnoop::initLeakDetector(Logger &logger, UprobeProfiler &profiler,
                                uint64_t threshold) {
  leak_detector_ = std::make_unique<LeakDetector>(logger, profiler, threshold);
  logger_.info("内存泄漏检测器已初始化，阈值: {}MB", threshold / (1024 * 1024));
}

void GPUSnoop::initFragmentDetector(Logger &logger,
                                    UprobeProfiler &profiler ) {
  fragment_detector_ = std::make_unique<FragmentDetector>(logger, profiler);
}

// // 在GPUSnoop类中添加
// bool GPUSnoop::enableCUPTIActivities() {
//     // 启用内存活动追踪
//     CUptiResult status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY);
//     if (status != CUPTI_SUCCESS) {
//         logger_.error("Failed to enable CUPTI memory activity");
//         return false;
//     }
    
//     // 启用内存拷贝活动追踪
//     status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
//     if (status != CUPTI_SUCCESS) {
//         logger_.error("Failed to enable CUPTI memcpy activity");
//         return false;
//     }
    
//     // 启用内核活动追踪（用于分析内核中的内存访问）
//     status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
//     if (status != CUPTI_SUCCESS) {
//         logger_.error("Failed to enable CUPTI kernel activity");
//         return false;
//     }
    
//     return true;
// }

// // 在GPUSnoop类中添加
// void GPUSnoop::setupCuptiCallbacks() {
//     CUptiResult status = cuptiActivityRegisterCallbacks(
//         bufferRequested,    // 缓冲区请求回调
//         bufferCompleted     // 缓冲区完成回调
//     );
//     if (status != CUPTI_SUCCESS) {
//         logger_.error("Failed to register CUPTI callbacks");
//     }
// }

// // 在GPUSnoop类中添加
// void CUPTIAPI GPUSnoop::bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
//     *size = 16 * 1024;  // 设置缓冲区大小为16KB
//     *buffer = (uint8_t*)malloc(*size);
//     *maxNumRecords = 0;
// }

// void CUPTIAPI GPUSnoop::bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
//     CUpti_Activity *record = nullptr;
//     CUptiResult status;
    
//     do {
//         status = cuptiActivityGetNextRecord(buffer, validSize, &record);
//         if (status == CUPTI_SUCCESS) {
//             switch (record->kind) {
//                 case CUPTI_ACTIVITY_KIND_MEMORY: {
//                     auto* memory = (CUpti_ActivityMemory*)record;
//                     processMemoryActivity(memory);
//                     break;
//                 }
//                 case CUPTI_ACTIVITY_KIND_MEMCPY: {
//                     auto* memcpy = (CUpti_ActivityMemcpy*)record;
//                     processMemcpyActivity(memcpy);
//                     break;
//                 }
//                 case CUPTI_ACTIVITY_KIND_KERNEL: {
//                     auto* kernel = (CUpti_ActivityKernel*)record;
//                     processKernelActivity(kernel);
//                     break;
//                 }
//                 default:
//                     // 其他类型的活动可以在这里处理
//                     break;
//             }
//         }
//     } while (status == CUPTI_SUCCESS);
    
//     free(buffer);
// }

// // 在GPUSnoop类中添加
// void GPUSnoop::processMemoryActivity(CUpti_ActivityMemory *memory) {
//     // 创建内存访问事件
//     FragmentEvent event;
//     event.address = memory->address;
//     event.size = memory->bytes;
//     event.timestamp = memory->start;
//     event.access_type = memory->memoryKind;
    
//     // 添加到fragment进行分析
//     if (fragment_detector_) {
//         fragment_detector_->addEvent(event);
//     }
// }

// void GPUSnoop::processMemcpyActivity(CUpti_ActivityMemcpy *memcpy) {
//     FragmentEvent event;
//     event.address = memcpy->dstAddress;  // 或 srcAddress
//     event.size = memcpy->bytes;
//     event.timestamp = memcpy->start;
//     event.access_type = 2;  // MEMCPY
    
//     if (fragment_detector_) {
//         fragment_detector_->addEvent(event);
//     }
// }

// void GPUSnoop::processKernelActivity(CUpti_ActivityKernel *kernel) {
//     // 获取内核执行信息
//     FragmentEvent event;
//     event.address = (uint64_t)kernel->deviceId;
//     event.size = kernel->gridX * kernel->gridY * kernel->gridZ * 
//                 kernel->blockX * kernel->blockY * kernel->blockZ;
//     event.timestamp = kernel->start;
//     event.access_type = 4;  // KERNEL
    
//     if (fragment_detector_) {
//         fragment_detector_->addEvent(event);
//     }
// }



} // namespace NeuTracer