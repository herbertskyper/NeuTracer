#include "snoop/io_snoop.h"
#include "config.h"
#include <bpf/bpf.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <fstream>
#include <sstream>
#include <iomanip>



namespace NeuTracer {

int IoSnoop::libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args) {
    if (level == LIBBPF_DEBUG)
        return 0;
    return vfprintf(stderr, format, args);
}

bool IoSnoop::start_trace() {
    // logger_.info("[IO] Starting I/O monitoring with pid: {}", env_.pid);
    libbpf_set_print(libbpf_print_fn);

    // 打开并加载BPF程序
    skel_ = bio_snoop_bpf__open();
    if (!skel_) {
        logger_.error("[IO] Failed to open BPF skeleton");
        return false;
    }

    // 加载并验证BPF程序
    int err = bio_snoop_bpf__load(skel_);
    if (err) {
        logger_.error("[IO] Failed to load BPF skeleton: {}", err);
        bio_snoop_bpf__destroy(skel_);
        return false;
    }

    // 挂载钩子
    err = bio_snoop_bpf__attach(skel_);
    if (err) {
        logger_.error("[IO] Failed to attach BPF skeleton: {}", err);
        bio_snoop_bpf__destroy(skel_);
        return false;
    }

    // 加载设备名称映射
    load_device_names();

    uint32_t key = env_.pid;
    uint32_t value = env_.pid;
    
    if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
                        &value, sizeof(value), BPF_ANY)) {
        logger_.error("Failed to update IO snoop_proc map: {}", strerror(errno));
        return false;
    }

    // logger_.info("[IO] I/O snoop_proc map updated with PID: {}", env_.pid);

    io_process_stats_.emplace(env_.pid, IOStats{key});
    

    // 设置环形缓冲区回调
    rb_ = ring_buffer__new(bpf_map__fd(skel_->maps.bio_events), handle_event, this, nullptr);
    if (!rb_) {
        logger_.error("[IO] Failed to create ring buffer");
        bio_snoop_bpf__destroy(skel_);
        return false;
    }

    // 启动环形缓冲区处理线程
    exiting_ = false;
    rb_thread_ = std::thread(&IoSnoop::ring_buffer_thread, this);


    logger_.info("[IO] I/O monitoring started");
    return true;
}

void IoSnoop::ring_buffer_thread() {
    while (!exiting_) {
        int err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            break;
        }
        if (err < 0) {
            logger_.error("[IO] Error polling ring buffer: {}", err);
            break;
        }
        
        // // 检查是否超时
        // auto now = std::chrono::steady_clock::now();
        // if (hasExceededProfilingLimit(std::chrono::seconds(idleTimeoutSec_), lastActivityTime_)) {
        //     logger_.info("[IO] I/O monitoring idle timeout");
        //     break;
        // }
    }
    ring_buffer__consume(rb_);
    // ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // logger_.info("[IO] I/O Ring buffer polling thread stopped");
}

bool IoSnoop::hasExceededProfilingLimit(std::chrono::seconds duration,
                                      const std::chrono::steady_clock::time_point &startTime) {
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
    return elapsedTime > duration;
}

int IoSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    auto io_snoop = static_cast<IoSnoop *>(ctx);
    if (data_sz != sizeof(struct bio_event)) {
        std::cerr << "Invalid IO event size: " << data_sz << std::endl;
        return -1;  // 错误处理
    }
    return io_snoop->process_event(ctx, data, data_sz);
}

int IoSnoop::process_event(void *ctx, void *data, size_t data_sz) {
    lastActivityTime_ = std::chrono::steady_clock::now();
    
    const struct bio_event *event = static_cast<const struct bio_event *>(data);
    if (!event)
        return 0;
    bool should_sample = (event->timestamp - last_sample_time_) >= SAMPLE_INTERVAL_NS;

    // 1. 首先决定是否处理此事件
    // bool should_process = env_.pid == 0 || env_.pid == event->pid || env_.pid == event->tgid;
    // bool is_kernel_thread = (event->pid == 0);
    // if(!should_process && !is_kernel_thread) {
    //     return 0;
    // }
    // if (!should_process) {
    //     //logger_.info("[IO] 跳过PID {}的IO事件 (目标PID: {})", event->pid, env_.pid);
    //     return 0;  // 如果设置了 PID 过滤且不匹配，则完全跳过
    // }
    //logger_.info("[IO] 处理PID {}的IO事件 (目标PID: {})", event->pid, env_.pid);
    
    // 2. 更新统计信息
    device_key dev_key = {event->major, event->minor};
    auto &dev_stat = device_stats_[dev_key];
    
    
    // 根据读写标志更新统计
    if (event->rwflag) {  // 写操作
        dev_stat.write_bytes += event->bytes;
        dev_stat.write_ops++;
        auto tmp = dev_stat.avg_write_latency_us;
        dev_stat.avg_write_latency_us = tmp * (dev_stat.write_ops - 1) / dev_stat.write_ops + event->delta_us / dev_stat.write_ops;
    } else {  // 读操作
        dev_stat.read_bytes += event->bytes;
        dev_stat.read_ops++;
        auto tmp = dev_stat.avg_read_latency_us;
        dev_stat.avg_read_latency_us = tmp * (dev_stat.read_ops - 1) / dev_stat.read_ops + event->delta_us / dev_stat.read_ops;
    }
    
    
    // 3. 记录单个 I/O 事件日志
    std::string rw_str = event->rwflag ? "WRITE" : "READ";
    std::string dev_name = get_device_name(event->major, event->minor);
    
    // 获取当前时间点作为时间戳
    // auto now = std::chrono::system_clock::now();
    // auto now_time_t = std::chrono::system_clock::to_time_t(now);
    // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     now.time_since_epoch()) % 1000;
    
    // std::stringstream time_now;
    // time_now << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
    //         << '.' << std::setfill('0') << std::setw(3) << ms.count();

    std::string time_ss = logger_.format_event_timestamp(event->timestamp);
    
    // 4. 更新 IO 统计
    update_io_stats(event->tgid, event, time_ss);
    auto &proc_stat = io_process_stats_[event->tgid];
    
    // logger_.info("[IO] PID {} ({}) {} {}B on {} ({},{}), latency: {:.3f}ms",
    //     event->pid, event->comm, rw_str, event->bytes,
    //     dev_name, (int)event->major, (int)event->minor,
    //     event->delta_us / 1000.0);

#ifdef REPORT_IO    
    // 定期报告
    event_count_++;
    if (event_count_ % IO_stats_report_interval_ == 0) {
        report_io();
    }
#endif
    if(should_sample){
        last_sample_time_ = event->timestamp;  // 更新最后一次采样时间
    }else {
        return 0;  // 如果不满足采样条件，则直接返回
    }
    auto& stat = io_process_stats_[event->tgid];
    auto read_size = stat.read_bytes > last_read_byte ? 
        stat.read_bytes - last_read_byte : 0;
    auto write_size = stat.write_bytes > last_write_byte ? 
        stat.write_bytes - last_write_byte : 0;
    profiler_.add_trace_data_io(
        "[IO] [{}] PID {} ({}) {} {:.2f}KB on {} ({},{}), latency:{}  recent: read_size:{}KB write_size:{}KB  read_latency: {:.3f}ms  write_latency: {:.3f}ms\n",
        time_ss, event->tgid, event->comm, rw_str, event->bytes / 1024.0, event->delta_us,
        dev_name, (int)event->major, (int)event->minor, read_size / 1024.0, write_size / 1024.0,
        stat.avg_read_latency_us, stat.avg_write_latency_us);
   
    // 4. 定期打印累积统计信息
    // static auto last_log_time = std::chrono::steady_clock::now();
    // auto current_time = std::chrono::steady_clock::now();
    // auto time_since_last_log = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_log_time);



    if (profiler_.isRPCEnabled()) {
        UprobeProfiler::IOTraceItem io_data;
        io_data.timestamp = time_ss;
        io_data.pid = event->tgid;
        io_data.comm = std::string(event->comm);
        io_data.operation = event->rwflag ? "WRITE" : "READ";
        io_data.bytes = event->bytes;
        io_data.latency_ms = event->delta_us / 1000.0;
        io_data.device = get_device_name(event->major, event->minor);
        io_data.major = event->major;
        io_data.minor = event->minor;
        
        // 添加进程统计数据
        io_data.read_bytes = read_size;
        io_data.write_bytes = write_size;
        io_data.read_ops = proc_stat.read_ops;
        io_data.write_ops = proc_stat.write_ops;
        io_data.avg_read_latency = 
            proc_stat.avg_read_latency_us;
        io_data.avg_write_latency = 
            proc_stat.avg_write_latency_us;
        
        // 发送给Profiler
        profiler_.sendIOTrace(io_data);
    }
    last_read_byte = stat.read_bytes;
    last_write_byte = stat.write_bytes;

    return 0;
}

bool IoSnoop::attach_bpf() {
    return start_trace();
}

void IoSnoop::stop_trace() {
    exiting_ = true;
    
    if (rb_thread_.joinable()){
        rb_thread_.join();
    }
        
    
    if (rb_) {
        ring_buffer__free(rb_);
        rb_ = nullptr;
    }
    
    if (skel_) {
        bio_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }
    
    logger_.info("[IO] I/O monitoring stopped");
}

void IoSnoop::load_device_names() {
    std::ifstream partitions("/proc/partitions");
    std::string line;
    
    // Skip header lines
    std::getline(partitions, line); // skip "major minor #blocks name"
    std::getline(partitions, line); // skip empty line
    
    while (std::getline(partitions, line)) {
        std::istringstream iss(line);
        int major, minor;
        long long blocks;
        std::string name;
        
        if (iss >> major >> minor >> blocks >> name) {
            device_key key = {static_cast<uint8_t>(major), static_cast<uint8_t>(minor)};
            device_names_[key] = name;
        }
    }
}

std::string IoSnoop::get_device_name(uint8_t major, uint8_t minor) const {
    device_key key = {major, minor};
    auto it = device_names_.find(key);
    if (it != device_names_.end()) {
        return it->second;
    }
    
    // 未找到设备名，返回格式化的设备号
    std::stringstream ss;
    ss << "dev(" << (int)major << ":" << (int)minor << ")";
    return ss.str();
}

std::vector<device_key> IoSnoop::get_device_list() const {
    std::vector<device_key> devices;
    for (const auto &pair : device_stats_) {
        devices.push_back(pair.first);
    }
    return devices;
}

void IoSnoop::update_io_stats(uint32_t pid, const bio_event* e, std::string timestamp) {
    auto& stats = io_process_stats_[pid];
    auto now = std::chrono::steady_clock::now();

    // 初始化窗口开始时间
    if (stats.window_start.time_since_epoch().count() == 0) {
        stats.window_start = now;
    }

    // 检查是否需要重置2小时窗口
    if (now - stats.window_start >= CLEAN_TIME_MIN * std::chrono::minutes(1)) {
        logger_.info("[IO窗口] PID {} ({}): {}分钟窗口重置 - 读:{} 写:{} 平均读延迟:{:.2f}ms 平均写延迟:{:.2f}ms",
                    pid, stats.comm, CLEAN_TIME_MIN,
                    stats.read_ops, stats.write_ops,
                    stats.avg_read_latency_us, stats.avg_write_latency_us);
        
        stats.window_start = now;
        stats.total_bytes = 0;
        stats.read_bytes = 0;
        stats.write_bytes = 0;
        stats.read_ops = 0;
        stats.write_ops = 0;
        stats.total_latency_us = 0;
        stats.max_latency_us = 0;
        stats.avg_latency_ms = 0.0;
        stats.avg_read_latency_us = 0.0;
        stats.avg_write_latency_us = 0.0;
        
        // // 清空容器
        stats.latency_samples.clear();
        stats.size_samples.clear();
        stats.device_access_count.clear();
        stats.last_device.clear();

        // 重新初始化设备访问统计
        for (const auto& [dev_key, dev_stat] : device_stats_) {
            device_stats_[dev_key] = io_stat{
                0, 0, 0, 0, 0, 0
            }; // 重置设备统计
        }
    }
    
    // 1. 基础数据更新
    stats.pid = pid;
    stats.comm = std::string(e->comm);
    stats.total_bytes += e->bytes;
    
    if (e->rwflag) {
        stats.write_bytes += e->bytes;
        stats.write_ops++;
        auto tmp = stats.avg_write_latency_us;
        stats.avg_write_latency_us = (tmp * (stats.write_ops - 1) + e->delta_us) / stats.write_ops;
    } else {
        stats.read_bytes += e->bytes;
        stats.read_ops++;
        auto tmp = stats.avg_read_latency_us;
        stats.avg_read_latency_us = (tmp * (stats.read_ops - 1) + e->delta_us)/ stats.read_ops;
    }
    
    // stats.total_latency_us += e->delta_us;
    uint64_t total_ops = stats.read_ops + stats.write_ops;

    if (total_ops > 0) {
        stats.avg_latency_ms = stats.total_latency_us / 1000.0 / total_ops;
    }
    
    if (e->delta_us > stats.max_latency_us) {
        stats.max_latency_us = e->delta_us;
    }
    
    // 2. 记录样本历史
    if (stats.latency_samples.size() >= IO_max_history_samples_) {
        stats.latency_samples.pop_front();
    }
    stats.latency_samples.push_back(e->delta_us);
    
    if (stats.size_samples.size() >= IO_max_history_samples_) {
        stats.size_samples.pop_front();
    }
    stats.size_samples.push_back(e->bytes);
    
    // 设备访问统计
    std::string device = get_device_name(e->major, e->minor);
    stats.device_access_count[device]++;
    
    bool should_warn = (e->timestamp - last_warning_time_) >= WARN_INTERVAL;
    if (should_warn && total_ops > 10) {
        bool anomaly_detected = false;
        
        // 高延迟异常
        if (e->delta_us > IO_latency_threshold_us_) {
            logger_.warn("[IO异常] PID {} ({}): 高延迟 {:.2f}ms (平均: {:.2f}ms)",
                        pid, stats.comm, e->delta_us / 1000.0, stats.avg_latency_ms);
        }
        // 超大IO异常
        if (e->bytes > IO_large_size_ * 1024 * 1024) {
            logger_.warn("[IO异常] PID {} ({}): 超大IO传输 {:.2f}MB",
                        pid, stats.comm, e->bytes / (1024.0 * 1024.0));
        }
        // 延迟尖峰异常
        else if (detect_io_latency_spike(pid, e)) {
            logger_.warn("[IO异常] PID {} ({}): 延迟尖峰 {:.2f}ms",
                        pid, stats.comm, e->delta_us / 1000.0);
        }
        
         last_warning_time_ = e->timestamp;  // 更新最后一次警告时间
    }
}

bool IoSnoop::detect_io_latency_spike(uint32_t pid, const bio_event* e) {
    auto& stats = io_process_stats_[pid];
    
    // 如果样本数量不足，无法判断
    if (stats.latency_samples.size() < 10) {
        return false;
    }
    
    // 计算最近样本的平均延迟
    uint64_t avg_latency = 0;
    for (size_t i = stats.latency_samples.size() - 10; i < stats.latency_samples.size(); i++) {
        avg_latency += stats.latency_samples[i];
    }
    avg_latency /= 10;
    
    // 如果当前延迟远高于平均值，视为异常尖峰
    return (e->delta_us > avg_latency * 3 && e->delta_us > 10000); // 3倍平均值且至少10ms
}

void IoSnoop::report_io() {
    for (const auto& [pid, stats] : io_process_stats_) {
        logger_.info("============ IO报告:{} ============", pid);
        // 基本IO统计
        logger_.info("[IO]  IO操作: {} (读:{}, 写:{})",
                stats.read_ops + stats.write_ops, stats.read_ops, stats.write_ops);
        logger_.info("[IO]  传输量: {}MB (读:{}MB, 写:{}MB)",
                stats.total_bytes / (1024*1024), 
                stats.read_bytes / (1024*1024),
                stats.write_bytes / (1024*1024));
        logger_.info("[IO]  平均延迟: {:.2f}ms (最大:{:.2f}ms)",
                stats.avg_latency_ms, stats.max_latency_us / 1000.0);
        
        // 设备分布
        if (stats.device_access_count.size() > 1) {
            logger_.info("[IO]  访问设备数: {}", stats.device_access_count.size());
        }
    }
    for (const auto& [dev_key, dev_stat] : device_stats_) {
        std::string dev_name = get_device_name(dev_key.major, dev_key.minor);
        logger_.info("[IO设备] {} ({}:{}) - 读:{}MB, 写:{}MB, 读延迟:{:.2f}ms, 写延迟:{:.2f}ms",
                dev_name, (int)dev_key.major, (int)dev_key.minor,
                dev_stat.read_bytes / (1024*1024),
                dev_stat.write_bytes  / (1024*1024),
                dev_stat.avg_read_latency_us / 1000.0,
                dev_stat.avg_write_latency_us / 1000.0);
    }
    logger_.info("===========================================");
}

}// namespace NeuTracer