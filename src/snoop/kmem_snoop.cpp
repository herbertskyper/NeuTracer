#include "snoop/kmem_snoop.h"
#include "utils/SymUtils.h"
#include "config.h"
#include "utils/Format.h"
#include <argp.h>
#include <bpf/libbpf.h>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <signal.h>
#include <pthread.h>
#include <future>

// static void sig_handler(int sig) {
//     pthread_exit(nullptr);
// }

namespace NeuTracer {

static const int64_t RINGBUF_MAX_ENTRIES = 64 * 1024;

int KmemSnoop::libbpf_print_fn(enum libbpf_print_level level, const char* format, va_list args) {
    if (level == LIBBPF_DEBUG) {
        return 0;
    }
    return vfprintf(stderr, format, args);
}

int KmemSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    KmemSnoop* self = static_cast<KmemSnoop*>(ctx);
    if (data_sz != sizeof(kmem_event)) {
        std::cerr << "Invalid KMEM event size: " << data_sz << std::endl;
        return -1;
    }
    return self->process_event(data, data_sz);
}

int KmemSnoop::process_event(void *data, size_t data_sz) {
    const struct kmem_event* e = static_cast<struct kmem_event*>(data);
    
    if(!e)
        return 0;

    bool should_trace = e->pid == env_.pid || e->tgid == env_.pid;

    if (!should_trace) {
        // 如果不满足条件，则直接返回
        return 0;
    }
    bool should_sample = false;
    if((e->timestamp_ns - last_sample_time_) >= SAMPLE_INTERVAL_NS * 10) {
        should_sample = true;
    } 
    // bool should_sample = (e->timestamp_ns - last_sample_time_) >= SAMPLE_INTERVAL_NS * 10;
    // logger_.info(
    //     "[{}] PID {} TGID {} {} size={} addr=0x{:x}  stack_id={}\n",
    //     time_ss.str(),
    //     e->pid,
    //     e->tgid,
    //     e->event_type == 0 ? "alloc" : "free",
    //     e->size,
    //     e->addr,
    //     e->stack_id
    // );
    std::string size_str = formatMemorySize(e->size);

    // // 更新内存统计信息
    // if (e->event_type == 0) { // 分配
    //     total_allocs_++;
    //     current_memory_ += e->size;
    //     if (current_memory_ > peak_memory_) {
    //         peak_memory_ = current_memory_;
    //     }
    // } else { // 释放
    //     total_frees_++;
    //     if (e->size <= current_memory_) {
    //         current_memory_ -= e->size;
    //     } else {
    //         current_memory_ = 0; // 防止负数
    //     }
    // }
    event_count_++;
    bool show_summary = (event_count_ % MEM_stats_report_interval_ == 0);

    // auto now = std::chrono::system_clock::now();
    // auto now_time_t = std::chrono::system_clock::to_time_t(now);
    // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     now.time_since_epoch()) % 1000;
    
    // std::stringstream time_ss;
    // time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
    //         << '.' << std::setfill('0') << std::setw(3) << ms.count();
    std::string time_ss = logger_.format_event_timestamp(e->timestamp_ns);

    // 更新内存异常检测统计
    update_mem_stats(e->tgid, e, time_ss);

#ifdef REPORT_KMEM
    if (show_summary) {
        report_mem();
    }
#endif // REPORT_KMEM
    if(!should_sample) return 0;
    last_sample_time_ = e->timestamp_ns;

    auto& stats = mem_stats_[e->tgid];



    profiler_.add_trace_data_kmem(
        "[KMEM] [{}] PID {} TID {} {} size={} addr=0x{:x}  stack_id={} current memory={} free={} alloc={}\n",
        time_ss,
        e->tgid,
        e->pid,
        e->event_type == 0 ? "alloc" : "free",
        size_str,
        e->addr,
        e->stack_id,
        formatMemorySize(stats.current_memory * 100),
        formatMemorySize(stats.total_allocs - last_alloc_size),
        formatMemorySize(stats.total_frees - last_free_size)
    );

// #ifdef REPORT_KMEM
//     if (show_summary) {
//         // 格式化当前和峰值内存使用量
//         std::string current_mem_str = formatMemorySize(current_memory_);
//         std::string peak_mem_str = formatMemorySize(peak_memory_);
        
//         // 构建并显示汇总信息
//         std::stringstream summary_ss;
//         summary_ss << "[KERNEL MEMORY SUMMARY] "  << "\n"
//                    << "  Total Allocations: " << total_allocs_ << "\n"
//                    << "  Total Frees: " << total_frees_ << "\n"
//                    << "  Current Memory: " << current_mem_str << "\n"
//                    << "  Peak Memory: " << peak_mem_str << "\n"
//                    << "  Potential Leaks: " << (total_allocs_ - total_frees_) << " allocations";
        
//         logger_.info("{}", summary_ss.str());
        
//     }
// #endif // REPORT_MEM_STATS



    // 如果启用了RPC，发送内存跟踪数据
    if (profiler_.isRPCEnabled()) {
        
        
        UprobeProfiler::MemTraceItem mem_data;
        mem_data.timestamp = time_ss;
        mem_data.pid = e->pid;
        mem_data.tgid = e->tgid;
        mem_data.comm = std::string(e->comm);
        mem_data.operation = e->event_type == 0 ? "alloc" : "free";
        mem_data.size = e->size;
        mem_data.addr = e->addr;
        mem_data.stack_id = e->stack_id;
        
        // 添加统计和异常检测字段
        mem_data.total_allocs = stats.total_allocs - last_alloc_size;
        mem_data.total_frees = stats.total_frees - last_free_size;
        mem_data.current_memory = stats.current_memory;
        mem_data.peak_memory = stats.peak_memory;
        
        profiler_.sendMemTrace(mem_data);
        }
    last_alloc_size = stats.total_allocs;
    last_free_size = stats.total_frees;
    return 0;
}

// 格式化内存大小为人类可读形式
std::string KmemSnoop::formatMemorySize(uint64_t size) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    
    if (size < 1024) {
        ss << size << " B";
    } else if (size < 1024 * 1024) {
        ss << (double)size / 1024 << " KB";
    } else if (size < 1024 * 1024 * 1024) {
        ss << (double)size / (1024 * 1024) << " MB";
    } else {
        ss << (double)size / (1024 * 1024 * 1024) << " GB";
    }
    
    return ss.str();
}

bool KmemSnoop::hasExceededProfilingLimit(
    std::chrono::seconds duration,
    const std::chrono::steady_clock::time_point& startTime) {
    if (duration.count() == 0) { // 0 = profle forever
        return false;
    }

  if (std::chrono::steady_clock::now() - startTime >= duration) {
    //logger_.info("[KMEM] Done Profiling: exceeded duration of {}s.\n", duration.count());
    return true;
  }
  return false;
}

bool KmemSnoop::attach_bpf() {
    // struct sigaction sa = {};
    // sa.sa_handler = sig_handler;  // 使用全局函数
    // sigemptyset(&sa.sa_mask);
    // sa.sa_flags = 0; 
    // sigaction(SIGUSR1, &sa, nullptr);

    libbpf_set_print(libbpf_print_fn);

    skel_ = kmem_snoop_bpf__open();
    if (!skel_) {
        logger_.error("Failed to open KMEM BPF skeleton");
        return false;
    }

    bpf_map__set_max_entries(
        skel_->maps.mem_events, env_.rb_count > 0 ? env_.rb_count : RINGBUF_MAX_ENTRIES);


    if (kmem_snoop_bpf__load(skel_)) {
        logger_.error("Failed to load KMEM BPF skeleton");
        return false;
    }

    if (kmem_snoop_bpf__attach(skel_)) {
        logger_.error("Failed to attach [KMEM] BPF programs");
        return false;
    }

    rb_ = ring_buffer__new(
        bpf_map__fd(skel_->maps.mem_events),
        handle_event,
        this,
        nullptr);

    if (!rb_) {
        logger_.error("Failed to create ring buffer");
        return false;
    }

    uint32_t key = env_.pid;
    uint32_t value = env_.pid;
    
    if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
                        &value, sizeof(value), BPF_ANY)) {
        logger_.error("Failed to update kmem snoop_proc map: {}", strerror(errno));
        return false;
    }
    logger_.info("Tracking network traffic for PID: {}", env_.pid);
    

    exiting_ = false;
    rb_thread_ = std::thread(&KmemSnoop::ring_buffer_thread, this);

    logger_.info("Kmem ring buffer polling started");
    return true;
}

void KmemSnoop::ring_buffer_thread() {
    auto startTime = std::chrono::steady_clock::now();

    auto duration = std::chrono::seconds(env_.duration_sec);

    std::time_t ttp =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logger_.info("Started KMEM profiling at {}", std::ctime(&ttp));

    // while (!exiting_ && !hasExceededProfilingLimit(duration, startTime)) {
    //     auto pollStartTime = std::chrono::steady_clock::now();

    //     int err = ring_buffer__poll(rb_, 20);

    //     if (exiting_) break;
        
    //     if (err == -EINTR ) {
    //         break;
    //     }
        
    //     if (err < 0) {
    //         logger_.error("Error polling kernel memory buffer: {}", err);
    //         continue;  
    //     }
        
    //     if (err > 0) {
    //         lastActivityTime_ = std::chrono::steady_clock::now();
    //     } else {
    //         auto currentTime = std::chrono::steady_clock::now();
    //         auto idleTime = std::chrono::duration_cast<std::chrono::seconds>(
    //             currentTime - lastActivityTime_).count();
            
    //         if (idleTimeoutSec_ > 0 && idleTime >= idleTimeoutSec_) {
    //             logger_.info("No activity for {} seconds, stopping kmem profiling", idleTime);
    //             break;
    //         }
    //     }
    // }  
    int err;
    while (!exiting_) {
        err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            break;
        }
        if (err < 0) {
            logger_.error("Error polling KMEM ring buffer: " + std::to_string(err));
            break;
        }
    }
    
    ring_buffer__consume(rb_);

    ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    logger_.info("Kmem ring buffer polling thread stopped");
}

void KmemSnoop::stop_trace() {
    exiting_ = true;

    if (rb_thread_.joinable()) {
        ring_buffer__consume(rb_);  // 唤醒Ring buffer线程
        rb_thread_.join();
    }
    
    if (rb_) {
        ring_buffer__free(rb_);
        rb_ = nullptr;
    }

    if (skel_) {
        kmem_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }

    logger_.info("Kmem profiling stopped");
}

// 添加内存异常检测相关方法
void KmemSnoop::update_mem_stats(uint32_t pid, const kmem_event* e, std::string timestamp) {
    auto& stats = mem_stats_[pid];

    auto now = std::chrono::steady_clock::now();
    
    // 初始化窗口开始时间
    if (stats.window_start.time_since_epoch().count() == 0) {
        stats.window_start = now;
    }
    
    // 检查是否需要重置2小时窗口
    if (now - stats.window_start >= CLEAN_TIME_MIN * std::chrono::minutes(1)) {
        
       logger_.info("[KMEM] PID {}: {}分钟窗口重置", 
                pid, CLEAN_TIME_MIN);
        stats.total_allocs = 0;  // 重置分配计数
        stats.total_frees = 0;   // 重置释放计数
        stats.current_memory = 0; // 重置当前内存使用
        stats.peak_memory = 0;    // 重置峰值内存使用
        stats.alloc_sizes.clear(); // 清空分配大小记录
        stats.addr_to_size.clear(); // 清空地址到大小的映射
        stats.mem_usage_samples.clear(); // 清空内存使用样本
        stats.churn_score = 0;   // 重置内存周转分数
        stats.window_start = now;  // 更新窗口开始时间
        return;  // 不需要进一步处理
    }
    
    // 更新基本信息
    stats.pid = e->pid;
    stats.tgid = e->tgid;
    stats.comm = std::string(e->comm);
    bool should_warn = (e->timestamp_ns - last_warning_time_) >= WARN_INTERVAL;
    
    // 更新操作统计
    if (e->event_type == 0) { // 分配
        stats.total_allocs += e->size;
        stats.current_memory += e->size;
        
        // 追踪分配大小
        if (stats.alloc_sizes.size() >= MEM_max_history_samples_) {
            stats.alloc_sizes.pop_front();
        }
        stats.alloc_sizes.push_back(e->size);
        
        // 记录地址和大小的映射
        stats.addr_to_size[e->addr] = e->size;
        
        // 检测大型分配
        if (e->size >= MEM_large_alloc_threshold_) {
            logger_.warn("[KMEM] 大型内存分配: {} bytes at addr=0x{:x} in PID {}", 
                    e->size, e->addr, pid);
        }
    } else { // 释放
        stats.total_frees += e->size;
        
        // 更新当前内存使用
        auto it = stats.addr_to_size.find(e->addr);
        if (it != stats.addr_to_size.end()) {
            // 找到了精确的大小记录
            uint64_t freed_size = it->second;
            if (freed_size <= stats.current_memory) {
                stats.current_memory -= freed_size;
            } else {
                stats.current_memory = 0;  // 防止负数
            }
            stats.addr_to_size.erase(it);
        } else {
            // 没有找到精确记录，使用事件中的大小
            if (e->size <= stats.current_memory) {
                stats.current_memory -= e->size;
            } else {
                stats.current_memory = 0;  // 防止负数
            }
        }
    }
    
    // 更新峰值
    if (stats.current_memory > stats.peak_memory) {
        stats.peak_memory = stats.current_memory;
    }
    
    // 追踪内存使用样本
    if (stats.mem_usage_samples.size() >= MEM_max_history_samples_) {
        stats.mem_usage_samples.pop_front();
    }
    stats.mem_usage_samples.push_back(stats.current_memory);
    if(!should_warn) return;
    detect_mem_anomalies(pid, timestamp);
    last_warning_time_ = e->timestamp_ns;  // 更新上次警告时间
}
        
void KmemSnoop::detect_mem_anomalies(uint32_t pid, std::string timestamp) {
    auto& stats = mem_stats_[pid];
    auto now = std::chrono::steady_clock::now();
    if (stats.mem_usage_samples.size() < 10) {
        return;
    }
    // 分析内存使用趋势
    int increasing = 0;
    int decreasing = 0;
    int stable = 0;
    uint64_t max_value = 0;
    uint64_t min_value = UINT64_MAX;
    uint64_t total = 0;
    
    // for (size_t i = 1; i < stats.mem_usage_samples.size(); i++) {
    for (size_t i = 1; i < 20; i++) {
        uint64_t prev = stats.mem_usage_samples[i-1];
        uint64_t curr = stats.mem_usage_samples[i];
        if (curr > prev * 1.05) {  // 增加超过5%
            increasing++;
        } else if (curr < prev * 0.95) {  // 减少超过5%
            decreasing++;
        } else {
            stable++;
        }
        
        max_value = std::max(max_value, curr);
        min_value = std::min(min_value, curr);
        total += curr;
    }
    
    // uint64_t avg_value = total / stats.mem_usage_samples.size();
    // double variation_ratio = (max_value > 0) ? 
    //                         (double)(max_value - min_value) / max_value : 0;
    
    // if (variation_ratio > MEM_variation_threshold_) {  
    //     // 波动模式
    //     logger_.warn("[KMEM] 内存使用波动较大，波动系数: {}% in PID {}", 
    //             static_cast<int>(variation_ratio * 100), pid);
        
    // }

    // if (stats.total_allocs > 0) {
    //     stats.churn_score = (100 * std::min(stats.total_frees, stats.total_allocs)) / 
    //                         stats.total_allocs;
    // }
    // //     if (stats.churn_score > MEM_churn_threshold_ * 100) { 
    // //         logger_.warn("[KMEM] 高内存周转率: {}% in PID {}", stats.churn_score, pid);
    // // }
    
    
}

void KmemSnoop::report_mem() {
    for (const auto& [pid, stats] : mem_stats_) {
        logger_.info("============ 内存报告 ============");
        logger_.info("[MEM]  分配次数: {}, 释放次数: {}", stats.total_allocs, stats.total_frees);
        logger_.info("[MEM]  当前内存: {} ", formatMemorySize(stats.current_memory * 100));
        // logger_.info("[MEM]  周转率: {}%", stats.churn_score);
        logger_.info("===================================");
    }
}
    

} // namespace NeuTracer