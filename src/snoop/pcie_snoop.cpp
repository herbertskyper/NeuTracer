#include "snoop/pcie_snoop.h"
#include "utils/Format.h"
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace NeuTracer {

PcieSnoop::PcieSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
    : env_(env), logger_(logger), profiler_(profiler) {
}

int PcieSnoop::libbpf_print_fn(enum libbpf_print_level level, const char* format, va_list args) {
    if (level == LIBBPF_DEBUG) {
        return 0;
    }
    return vfprintf(stderr, format, args);
}

int PcieSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    PcieSnoop* self = static_cast<PcieSnoop*>(ctx);
    
    // 检查事件类型
    if (data_sz == sizeof(pcie_event)) {
        return self->process_event(data, data_sz);
    } else if (data_sz == sizeof(dma_fence_event)) {
        return self->process_dma_fence_event(data, data_sz);
    } else {
        std::cerr << "Unknown pcie event size: " << data_sz << std::endl;
        return -1;
    }
}

std::string PcieSnoop::op_type_to_string(pcie_op_type op) {
    switch (op) {
        case PCIE_CONFIG_READ_BYTE: return "pci_read_byte";
        case PCIE_CONFIG_READ_WORD: return "pci_read_word";
        case PCIE_CONFIG_READ_DWORD: return "pci_read_dword";
        case PCIE_CONFIG_WRITE_BYTE: return "pci_write_byte";
        case PCIE_CONFIG_WRITE_WORD: return "pci_write_word";
        case PCIE_CONFIG_WRITE_DWORD: return "pci_write_dword";
        default: return "unknown";
    }
}

std::string PcieSnoop::dma_fence_type_to_string(dma_fence_event_type type) {
    switch (type) {
        case DMA_FENCE_INIT: return "init";
        case DMA_FENCE_DESTROY: return "destroy";
        case DMA_FENCE_ENABLE_SIGNAL: return "enable_signal";
        case DMA_FENCE_SIGNALED: return "signaled";
        case DMA_FENCE_WAIT_START: return "wait_start";
        case DMA_FENCE_WAIT_END: return "wait_end";
        case DMA_FENCE_EMIT: return "emit";
        default: return "unknown";
    }
}

int PcieSnoop::process_event(void *data, size_t data_sz) {
    logger_.info("enter the pciesnoop\n");
    const struct pcie_event* e = static_cast<const pcie_event*>(data);
    auto key = std::make_tuple(e->tgid, e->pid, e->op_type);
    bool should_sample = (e->timestamp - last_sample_time_) >= SAMPLE_INTERVAL_NS;
    
    if (e->type == EVENT_PCIE_ENTER) {
        pcie_enter_ts[key] = e->timestamp;
    } else if (e->type == EVENT_PCIE_EXIT) {
        auto it = pcie_enter_ts.find(key);
        if (it != pcie_enter_ts.end()) {
            uint64_t duration_ns = e->timestamp - it->second;
            pcie_enter_ts.erase(it);

            auto& stats = pcie_stats_[e->tgid];
            auto& avg_time = stats.avg_pcie_time[e->op_type];
            auto& count = stats.pcie_count[e->op_type];
            
            if (count == 0) {
                avg_time = duration_ns;
            } else {
                avg_time = (avg_time * count + duration_ns) / (count + 1);
            }
            count += 1;
        }
    }
    
    // 格式化时间戳
    std::string time_ss = logger_.format_event_timestamp(e->timestamp);

    // 统计
    auto& stats = pcie_stats_[e->tgid];
    stats.pid = e->pid;
    stats.tgid = e->tgid;
    
    if(!should_sample) return 0;
    
    // 记录事件
    if (e->type == EVENT_PCIE_ENTER) {
        profiler_.add_trace_data_pcie(
            "[PCIE] [{}] PID:{} TID:{} ({}) op:{} bus:0x{:02x} dev:0x{:02x} off:0x{:x} size:{} val:0x{:x}\n",
            time_ss,
            e->tgid,
            e->pid,
            e->comm,
            op_type_to_string(static_cast<pcie_op_type>(e->op_type)),
            e->bus,
            e->devfn,
            e->offset,
            e->size,
            e->value
        );
    } else {
        profiler_.add_trace_data_pcie(
            "[PCIE] [{}] PID:{} TID:{} ({}) op:{} result:0x{:x} count:{} avg_time:{}us\n",
            time_ss,
            e->tgid,
            e->pid,
            e->comm,
            op_type_to_string(static_cast<pcie_op_type>(e->op_type)),
            e->value,
            stats.pcie_count[e->op_type],
            stats.avg_pcie_time[e->op_type] / 1000 // 转换为微秒
        );
    }

    return 0;
}

// 添加 DMA Fence 事件处理函数
int PcieSnoop::process_dma_fence_event(void *data, size_t data_sz) {
    const struct dma_fence_event* e = static_cast<const dma_fence_event*>(data);
    bool should_sample = (e->timestamp - last_sample_time_) >= SAMPLE_INTERVAL_NS;
    
    // 处理 DMA Fence 等待统计
    auto& stats = pcie_stats_[e->tgid];
    stats.pid = e->pid;
    stats.tgid = e->tgid;
    
    // 更新统计信息
    stats.fence_count[e->type]++;
    
    if (e->type == DMA_FENCE_WAIT_END && e->duration_ns > 0) {
        stats.total_fence_wait_time += e->duration_ns;
        if (e->duration_ns > stats.max_fence_wait_time) {
            stats.max_fence_wait_time = e->duration_ns;
        }
    }
    
    if (!should_sample) return 0;
    
    // 格式化时间戳
    std::string time_ss = logger_.format_event_timestamp(e->timestamp);
    
    // 构建事件记录
    std::string event_str;
    switch (e->type) {
        case DMA_FENCE_WAIT_END:
            profiler_.add_trace_data_pcie(
                "[DMA_FENCE] [{}] PID:{} TID:{} ({}) type:{} ctx:{} seqno:{} wait:{:.3f}ms\n",
                time_ss, e->tgid, e->pid, e->comm,
                dma_fence_type_to_string(static_cast<dma_fence_event_type>(e->type)),
                e->context, e->seqno,
                e->duration_ns / 1000000.0  // 转换为毫秒
            );
            break;
        default:
            profiler_.add_trace_data_pcie(
                "[DMA_FENCE] [{}] PID:{} TID:{} ({}) type:{} ctx:{} seqno:{} driver:{}\n",
                time_ss, e->tgid, e->pid, e->comm,
                dma_fence_type_to_string(static_cast<dma_fence_event_type>(e->type)),
                e->context, e->seqno,
                e->driver_name[0] ? e->driver_name : "unknown"
            );
    }
    return 0;
}


bool PcieSnoop::attach_bpf() {
    libbpf_set_print(libbpf_print_fn);

    //debug
    libbpf_print_level(LIBBPF_INFO);

    skel_ = pcie_snoop_bpf__open();
    if (!skel_) {
        logger_.error("Failed to open PCIe BPF skeleton - err: {}", errno);
        return false;
    }

    int err = pcie_snoop_bpf__load(skel_);
    if (err) {
        logger_.error("Failed to load PCIe BPF skeleton: {}", err);
        return false;
    }
    
    err = pcie_snoop_bpf__attach(skel_);
    if (err) {
        logger_.error("Failed to attach [PCIE] BPF programs");
        return false;
    }
    
    rb_ = ring_buffer__new(
        bpf_map__fd(skel_->maps.events),
        handle_event,
        this,
        nullptr);

    if (!rb_) {
        logger_.error("Failed to create [PCIE] ring buffer");
        return false;
    }

    // 进程过滤
    uint32_t key = env_.pid;
    uint32_t value = env_.pid;
    logger_.info("Setting up PCIe snoop filter for PID: {}", key);

    if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
                        &value, sizeof(value), BPF_ANY)) {
        logger_.error("Failed to update PCIe snoop_proc map: {}", strerror(errno));
        return false;
    }

    exiting_ = false;
    rb_thread_ = std::thread(&PcieSnoop::ring_buffer_thread, this);

    logger_.info("PCIe监控模块启动");
    return true;
}

void PcieSnoop::ring_buffer_thread() {
    auto last_cleanup = std::chrono::steady_clock::now();
    std::time_t ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logger_.info("Started PCIe profiling at {}", std::ctime(&ttp));

    int err;
    while(!exiting_) {
        err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            break;
        }
        if (err < 0) {
            logger_.error("Error polling [PCIE] ring buffer: " + std::to_string(err));
            break;
        }
    }
    ring_buffer__consume(rb_);
}

void PcieSnoop::stop_trace() {
    exiting_ = true;
    if (rb_thread_.joinable()) {
        rb_thread_.join();
    }
    if (skel_) {
        pcie_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }
    logger_.info("PCIe monitoring thread stopped");
}

} // namespace NeuTracer