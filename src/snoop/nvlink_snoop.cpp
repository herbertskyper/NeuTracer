#include "snoop/nvlink_snoop.h"
#include "utils/Format.h"
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace NeuTracer {

NVLinkSnoop::NVLinkSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
    : env_(env), logger_(logger), profiler_(profiler) {
}

int NVLinkSnoop::libbpf_print_fn(enum libbpf_print_level level, const char* format, va_list args) {
    if (level == LIBBPF_DEBUG) {
        return 0;
    }
    return vfprintf(stderr, format, args);
}

int NVLinkSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    NVLinkSnoop* self = static_cast<NVLinkSnoop*>(ctx);
    if (data_sz != sizeof(nvlink_event)) {
        std::cerr << "Invalid NVLINK event size: " << data_sz << std::endl;
        return -1;
    }
    return self->process_event(data, data_sz);
}

int NVLinkSnoop::process_event(void *data, size_t data_sz) {
    const struct nvlink_event* e = static_cast<const nvlink_event*>(data);
    auto key = std::make_tuple(e->tgid, e->pid, e->func_id);
    bool should_sample = (e->timestamp - last_sample_time_) >= SAMPLE_INTERVAL_NS;
    
    if (e->type == EVENT_NVLINK_ENTER) {
        nvlink_enter_ts[key] = e->timestamp;
    } else if (e->type == EVENT_NVLINK_EXIT) {
        auto it = nvlink_enter_ts.find(key);
        if (it != nvlink_enter_ts.end()) {
            uint64_t duration_ns = e->timestamp - it->second;
            nvlink_enter_ts.erase(it);

            auto& stats = nvlink_stats_[e->tgid];
            stats.pid = e->pid;
            stats.tgid = e->tgid;
            
            // 更新统计信息
            stats.call_count[e->func_id]++;
            
            // 更新平均处理时间
            auto& avg_time = stats.avg_time[e->func_id];
            auto call_count = stats.call_count[e->func_id];
            if (call_count == 1) {
                avg_time = duration_ns;
            } else {
                avg_time = (avg_time * (call_count - 1) + duration_ns) / call_count;
            }
            
            // 更新总处理字节数
            if (e->func_id != NVLINK_FUNC_STRCPY) { // 对于memcpy和memset，size是已知的
                stats.total_bytes[e->func_id] += e->size;
            }
        }
    }
    
    // 不需要每次事件都打印，可以根据采样间隔处理
    if (!should_sample) return 0;
    
    // 格式化时间戳
    std::string time_ss = logger_.format_event_timestamp(e->timestamp);
    
    // 记录事件
    if (e->type == EVENT_NVLINK_ENTER) {
        if (e->func_id == NVLINK_FUNC_STRCPY) {
            profiler_.add_trace_data_nvlink(
                "[NVLINK] [{}] PID:{} TID:{} ({}) {} dst=0x{:x}, src=0x{:x}\n",
                time_ss,
                e->tgid,
                e->pid,
                e->comm,
                get_func_name(static_cast<nvlink_func_id>(e->func_id)),
                e->dst_addr,
                e->src_addr
            );
        } else if (e->func_id == NVLINK_FUNC_MEMCPY) {
             profiler_.add_trace_data_nvlink(
            "[NVLINK] [{}] PID:{} TID:{} ({}) {}() dst=0x{:x}, src=0x{:x}, size={}\n",
            time_ss,
            e->tgid,
            e->pid,
            e->comm,
            get_func_name(static_cast<nvlink_func_id>(e->func_id)),
            e->dst_addr, e->src_addr, e->size
        );
        } else if (e->func_id == NVLINK_FUNC_MEMSET) {
             profiler_.add_trace_data_nvlink(
            "[NVLINK] [{}] PID:{} TID:{} ({}) {}() {} dst=0x{:x}, val={}, size={}\n",
            time_ss,
            e->tgid,
            e->pid,
            e->comm,
            get_func_name(static_cast<nvlink_func_id>(e->func_id)),
            e->dst_addr, static_cast<int>(e->src_addr), e->size
            );
        }
        
        
    } else {
        auto& stats = nvlink_stats_[e->tgid];
        profiler_.add_trace_data_nvlink(
            "[NVLINK] [{}] PID:{} TID:{} ({}) {}() returns=0x{:x} calls={} avg_time={}us total_bytes={}\n",
            time_ss,
            e->tgid,
            e->pid,
            e->comm,
            get_func_name(static_cast<nvlink_func_id>(e->func_id)),
            e->ret_val,
            stats.call_count[e->func_id],
            stats.avg_time[e->func_id] / 1000, // 转换为微秒
            stats.total_bytes[e->func_id]
        );
    }
    
    return 0;
}

std::string NVLinkSnoop::get_func_name(nvlink_func_id func_id) {
    switch (func_id) {
        case NVLINK_FUNC_STRCPY: return "nvlink_strcpy";
        case NVLINK_FUNC_MEMCPY: return "nvlink_memcpy";
        case NVLINK_FUNC_MEMSET: return "nvlink_memset";
        default: return "unknown";
    }
}

bool NVLinkSnoop::attach_bpf() {
    libbpf_set_print(libbpf_print_fn);

    skel_ = nvlink_snoop_bpf__open();
    if (!skel_) {
        logger_.error("Failed to open NVLink BPF skeleton - err: {}", errno);
        return false;
    }

    int err = nvlink_snoop_bpf__load(skel_);
    if (err) {
        logger_.error("Failed to load NVLink BPF skeleton: {}", err);
        return false;
    }
    
    // 附加所有kprobe
    links_.push_back(bpf_program__attach(skel_->progs.trace_nvlink_strcpy_enter));
    links_.push_back(bpf_program__attach(skel_->progs.trace_nvlink_strcpy_exit));
    links_.push_back(bpf_program__attach(skel_->progs.trace_nvlink_memcpy_enter));
    links_.push_back(bpf_program__attach(skel_->progs.trace_nvlink_memcpy_exit));
    links_.push_back(bpf_program__attach(skel_->progs.trace_nvlink_memset_enter));
    links_.push_back(bpf_program__attach(skel_->progs.trace_nvlink_memset_exit));
    
    for (auto link : links_) {
        if (!link) {
            logger_.error("Failed to attach NVLink BPF program");
            return false;
        }
    }

    rb_ = ring_buffer__new(
        bpf_map__fd(skel_->maps.events),
        handle_event,
        this,
        nullptr);

    if (!rb_) {
        logger_.error("Failed to create NVLink ring buffer");
        return false;
    }

    // 进程过滤
    uint32_t key = env_.pid;
    uint32_t value = env_.pid;
    if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
                        &value, sizeof(value), BPF_ANY)) {
        logger_.error("Failed to update NVLink snoop_proc map: {}", strerror(errno));
        return false;
    }

    exiting_ = false;
    rb_thread_ = std::thread(&NVLinkSnoop::ring_buffer_thread, this);

    logger_.info("NVLink模块启动");
    return true;
}

void NVLinkSnoop::ring_buffer_thread() {
    std::time_t ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // logger_.info("Started NVLink profiling at {}", std::ctime(&ttp));

    int err;
    while(!exiting_) {
        err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            break;
        }
        if (err < 0) {
            logger_.error("Error polling NVLink ring buffer: " + std::to_string(err));
            break;
        }
        
        // 更新最后活动时间
        lastActivityTime_ = std::chrono::steady_clock::now();
    }
    
    ring_buffer__consume(rb_);

    ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logger_.info("Stopped NVLink profiling at {}", std::ctime(&ttp));
}

void NVLinkSnoop::stop_trace() {
    exiting_ = true;
    if (rb_thread_.joinable()) {
        rb_thread_.join();
    }
    
    // 清理资源
    for (auto link : links_) {
        bpf_link__destroy(link);
    }
    links_.clear();
    
    if (rb_) {
        ring_buffer__free(rb_);
        rb_ = nullptr;
    }
    
    if (skel_) {
        nvlink_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }
    
    logger_.info("NVLink monitoring thread stopped");
}

} // namespace NeuTracer