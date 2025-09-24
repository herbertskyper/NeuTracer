#include "snoop/net_snoop.h"
#include "utils/Format.h"
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <arpa/inet.h>

namespace NeuTracer {

int NetSnoop::libbpf_print_fn(
    enum libbpf_print_level level,
    const char* format,
    va_list args) {
    if (level == LIBBPF_DEBUG) {
        return 0;
    }
    return vfprintf(stderr, format, args);
}

int NetSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    NetSnoop* self = static_cast<NetSnoop*>(ctx);
    if (data_sz != sizeof(net_event)) {
        std::cerr << "Invalid NET event size: " << data_sz << std::endl;
        return -1;
    }
    return self->process_event(data, data_sz);
}

int NetSnoop::process_event(void *data, size_t data_sz) {
    const struct net_event* e = static_cast<struct net_event*>(data);
    bool should_trace = e->pid == env_.pid;
    

    if (!should_trace) {
        // 如果不满足条件，则直接返回
        return 0;
    }
    bool should_sample = (e->timestamp - last_sample_time_) >= SAMPLE_INTERVAL_NS;
    // auto now = std::chrono::system_clock::now();
    // auto now_time_t = std::chrono::system_clock::to_time_t(now);
    // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     now.time_since_epoch()) % 1000;
    
    // std::stringstream time_ss;
    // time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
    //         << '.' << std::setfill('0') << std::setw(3) << ms.count();
    std::string time_ss = logger_.format_event_timestamp(e->timestamp);
    
    bool is_target_by_pid = (e->pid == env_.pid);
    bool is_target_by_name = false;

    std::string comm_name(e->comm);
    if (target_process_names.find(comm_name) != target_process_names.end()) {
        is_target_by_name = true;
    }
    
    // 更新网络异常检测统计
    update_net_stats(e->pid, e, time_ss);
    
    // 只记录目标进程的事件
    if (is_target_by_pid || is_target_by_name) {
        std::string protocol_str = format_protocol(e->protocol);
        std::string source_ip = format_ip_address(e->saddr);
        std::string dest_ip = format_ip_address(e->daddr);

#ifdef REPORT_NET
        event_count_++;
        if (event_count_ % NET_stats_report_interval_ == 0) {
            report_net();
        }
#endif
        // 记录事件
        if(!should_sample) return 0;
        last_sample_time_ = e->timestamp;
        auto& stats = net_stats_[e->pid];
        auto send_byte = stats.tx_bytes - last_send_bytes_;
        auto receive_byte = stats.rx_bytes - last_receive_bytes_; 

        profiler_.add_trace_data_net(
            "[NET] [{}] PID:{} TID:{} ({}) {} {}:{} -> {}:{} {} bytes recent send:{} receive:{}\n",
            time_ss,
            e->tgid,
            e->pid,
            e->comm,
            e->is_send ? "TX" : "RX",
            source_ip,
            e->sport,
            dest_ip,
            e->dport,
            e->bytes,
            send_byte,
            receive_byte
        );
        

        // 如果启用了RPC，发送网络跟踪数据
        if (profiler_.isRPCEnabled()) {
            
            
            // 创建NetworkTraceItem
            UprobeProfiler::NetTraceItem net_item;
            net_item.timestamp = time_ss;
            net_item.pid = e->pid;
            net_item.tgid = e->tgid;
            net_item.comm = e->comm;
            net_item.is_send = e->is_send;
            net_item.bytes = e->bytes;
            net_item.src_addr = source_ip;
            net_item.dst_addr = dest_ip;
            net_item.src_port = e->sport;
            net_item.dst_port = e->dport;
            net_item.protocol = protocol_str;
            net_item.tx_bytes = send_byte;
            net_item.rx_bytes = receive_byte;
            net_item.tx_packets = stats.tx_packets;
            net_item.rx_packets = stats.rx_packets;
            net_item.active_connections = stats.connections.size();

            profiler_.sendNetTrace(net_item);
        }
        last_receive_bytes_ = stats.rx_bytes;
        last_send_bytes_ = stats.tx_bytes;
    }
    
    return 0;
}

void NetSnoop::update_net_stats(uint32_t pid, const net_event* e, std::string timestamp) {
    auto& stats = net_stats_[pid];
    auto now = std::chrono::steady_clock::now();
    
    // 初始化窗口开始时间
    if (stats.window_start.time_since_epoch().count() == 0) {
        stats.window_start = now;
    }
    
    // 检查是否需要重置2小时窗口
    if (now - stats.window_start >= CLEAN_TIME_MIN * std::chrono::minutes(1)) {
        logger_.info("[NET] PID {}: {}分钟窗口重置 ", pid, CLEAN_TIME_MIN);
        stats.tx_bytes = 0;
        stats.rx_bytes = 0;
        stats.tx_packets = 0;
        stats.rx_packets = 0;
        stats.tx_timestamps.clear();
        stats.rx_timestamps.clear();
        stats.tx_bytes_history.clear();
        stats.rx_bytes_history.clear();
        stats.connections.clear();
        stats.listening_ports.clear();
        stats.avg_latency_ms = 0.0;
        
        // 重置窗口开始时间
        stats.window_start = now;
    }
    
    stats.pid = e->pid;
    stats.tgid = e->tgid; 
    stats.comm = std::string(e->comm);
    bool should_warn = (e->timestamp - last_warning_time_) >= WARN_INTERVAL;
    
    // 当前时间
    double current_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()) / 1000.0;
    
    // 根据是否发送更新统计
    if (e->is_send) {
        stats.tx_bytes += e->bytes;
        stats.tx_packets++;
        
        // 追踪时间戳和字节数
        if (stats.tx_timestamps.size() >= NET_max_history_samples_) {
            stats.tx_timestamps.pop_front();
            stats.tx_bytes_history.pop_front();
        }
        stats.tx_timestamps.push_back(current_time);
        stats.tx_bytes_history.push_back(e->bytes);
    } else {
        stats.rx_bytes += e->bytes;
        stats.rx_packets++;
        
        // 追踪时间戳和字节数
        if (stats.rx_timestamps.size() >= NET_max_history_samples_) {
            stats.rx_timestamps.pop_front();
            stats.rx_bytes_history.pop_front();
        }
        stats.rx_timestamps.push_back(current_time);
        stats.rx_bytes_history.push_back(e->bytes);
    }
    
    // 记录连接信息
    std::string conn_key = format_connection_key(e);
    stats.connections.insert(conn_key);
    
    // 如果是监听端口，记录
    if (e->dport == 0 && e->sport > 0) {
        stats.listening_ports.insert(e->sport);
    }
    if(!should_warn) return ;
    detect_net_anomalies(pid, timestamp);
    last_warning_time_ = e->timestamp;
}



    


void NetSnoop::detect_net_anomalies(uint32_t pid, std::string timestamp) {
    auto& stats = net_stats_[pid];
    auto now = std::chrono::steady_clock::now();
    
    if (stats.tx_timestamps.size() > 10 && stats.rx_timestamps.size() > 10) {
        double tx_variance = 0.0;
        double rx_variance = 0.0;
        
        // 计算发送间隔的方差
        if (stats.tx_timestamps.size() > 10) {
            std::vector<double> tx_intervals;
            for (size_t i = 1; i < stats.tx_timestamps.size(); i++) {
                tx_intervals.push_back(stats.tx_timestamps[i] - stats.tx_timestamps[i-1]);
            }
            
            double tx_mean = 0.0;
            for (double interval : tx_intervals) {
                tx_mean += interval;
            }
            tx_mean /= tx_intervals.size();
            
            for (double interval : tx_intervals) {
                tx_variance += (interval - tx_mean) * (interval - tx_mean);
            }
            tx_variance /= tx_intervals.size();
        }
        
        // 计算接收间隔的方差
        if (stats.rx_timestamps.size() > 10) {
            std::vector<double> rx_intervals;
            for (size_t i = 1; i < stats.rx_timestamps.size(); i++) {
                rx_intervals.push_back(stats.rx_timestamps[i] - stats.rx_timestamps[i-1]);
            }
            
            double rx_mean = 0.0;
            for (double interval : rx_intervals) {
                rx_mean += interval;
            }
            rx_mean /= rx_intervals.size();
            
            for (double interval : rx_intervals) {
                rx_variance += (interval - rx_mean) * (interval - rx_mean);
            }
            rx_variance /= rx_intervals.size();
        }
        
        // 如果方差大于阈值，判定为突发型
        if (tx_variance > NET_SEND_variation_threshold_ ){
            logger_.info("[NET] PID {}: 发送流量突发检测 (方差: {:.2f})", pid, tx_variance);
        }
        if(rx_variance > NET_RECV_variation_threshold_) {
            logger_.info("[NET] PID {}: 接收流量突发检测 (方差: {:.2f})", pid, rx_variance);
        }
    
    if(stats.tx_bytes > NET_SEND_large_size_) {
        logger_.info("[NET] PID {}: 发送大数据包检测 (大小: {} bytes)", pid, stats.tx_bytes);
    }
    if(stats.rx_bytes > NET_RECV_large_size_) {
        logger_.info("[NET] PID {}: 接收大数据包检测 (大小: {} bytes)", pid, stats.rx_bytes);
    }
    
    // 检测连接数异常
    static bool add_connection_anomaly = false;
    if (stats.connections.size() > NET_CONN_threshold_ ){
        logger_.info("[NET] PID {}: 连接数异常检测 (连接数: {})", pid, stats.connections.size());
    }
}
    
}


// 辅助方法实现
std::string NetSnoop::format_ip_address(uint32_t addr) {
    struct in_addr ip_addr;
    ip_addr.s_addr = addr;
    return std::string(inet_ntoa(ip_addr));
}

std::string NetSnoop::format_protocol(uint8_t proto) {
    switch(proto) {
        case 1: return "ICMP";
        case 6: return "TCP";
        case 17: return "UDP";
        default: return "Proto(" + std::to_string(proto) + ")";
    }
}

std::string NetSnoop::format_connection_key(const net_event* e) {
    std::stringstream ss;
    ss << format_ip_address(e->saddr) << ":" << e->sport << "->"
       << format_ip_address(e->daddr) << ":" << e->dport
       << " (" << format_protocol(e->protocol) << ")";
    return ss.str();
}

std::string NetSnoop::format_throughput(uint64_t bytes_per_sec) {
    if (bytes_per_sec < 1024) {
        return std::to_string(bytes_per_sec) + " B";
    } else if (bytes_per_sec < 1024 * 1024) {
        return std::to_string(bytes_per_sec / 1024) + " KB";
    } else if (bytes_per_sec < 1024 * 1024 * 1024) {
        return std::to_string(bytes_per_sec / (1024 * 1024)) + " MB";
    } else {
        return std::to_string(bytes_per_sec / (1024 * 1024 * 1024)) + " GB";
    }
}

bool NetSnoop::attach_bpf() {
    libbpf_set_print(libbpf_print_fn);

    skel_ = net_snoop_bpf__open();
    if (!skel_) {
        logger_.error("Failed to open network BPF skeleton - err: {}", errno);
        return false;
    }

    int err = net_snoop_bpf__load(skel_);
    if (err) {
        logger_.error("Failed to load network BPF skeleton: {}", err);
        return false;
    }
    if (net_snoop_bpf__attach(skel_)) {
        logger_.error("Failed to attach [NET] BPF programs");
        return false;
    }
    rb_ = ring_buffer__new(
        bpf_map__fd(skel_->maps.net_events),
        handle_event,
        this,
        nullptr);
    
    if (!rb_) {
        logger_.error("Failed to create [NET] ring buffer");
        return false;
    }

    // Attach tracepoints
    // skel_->links.net_dev_queue = bpf_program__attach(skel_->progs.net_dev_queue);
    // if (!skel_->links.net_dev_queue) {
    //     logger_.error("Failed to attach net_dev_queue: {}", strerror(errno));
    //     return false;
    // }
    // links_.push_back(skel_->links.net_dev_queue);

    // skel_->links.netif_receive_skb = bpf_program__attach(skel_->progs.netif_receive_skb);
    // if (!skel_->links.netif_receive_skb) {
    //     logger_.error("Failed to attach netif_receive_skb: {}", strerror(errno));
    //     return false;
    // }
    // links_.push_back(skel_->links.netif_receive_skb);

    uint32_t key = env_.pid;
    uint32_t value = env_.pid;
    
    if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
                        &value, sizeof(value), BPF_ANY)) {
        logger_.error("Failed to update net snoop_proc map: {}", strerror(errno));
        return false;
    }
    // logger_.info("[NET] 网络模块跟踪进程 PID: {}", env_.pid);

    // Start polling thread

    target_process_names.insert("python");
    target_process_names.insert("python3");
    target_process_names.insert("requests");
    target_process_names.insert("urllib3");

    exiting_ = false;
    rb_thread_ = std::thread(&NetSnoop::ring_buffer_thread, this);
    
    logger_.info("网络模块启动");
    return true;
}

void NetSnoop::ring_buffer_thread() {
    auto startTime = std::chrono::steady_clock::now();
    // auto duration = std::chrono::seconds(env_.duration_sec);
    auto duration = std::chrono::seconds(60);  // 固定60秒

    std::time_t ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logger_.info("Started network profiling at {}", std::ctime(&ttp));
    
    // Maps to keep track of previous stats to calculate rates
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> previous_stats; // pid -> (bytes_sent, bytes_recv)
    
    // while (!exiting_ && !hasExceededProfilingLimit(duration, startTime)) {
    //     auto pollStartTime = std::chrono::steady_clock::now();

    //     int err = ring_buffer__poll(rb_, 100);

    //     if (exiting_) break;
        
    //     if (err == -EINTR ) {
    //         break;
    //     }
        
    //     if (err < 0) {
    //         logger_.error("Error polling [NET] buffer: {}", err);
    //         continue;  
    //     }
        
    //     if (err > 0)  {
    //         lastActivityTime_ = std::chrono::steady_clock::now();
    //     } else {
    //         // Check for idle timeout
    //         auto currentTime = std::chrono::steady_clock::now();
    //         auto idleTime = std::chrono::duration_cast<std::chrono::seconds>(
    //             currentTime - lastActivityTime_).count();
            
    //         if (idleTimeoutSec_ > 0 && idleTime >= idleTimeoutSec_) {
    //             logger_.info("No network activity for {} seconds, stopping profiling", idleTime);
    //             break;
    //         }
    //     }
    // }

    int err;
    while(!exiting_) {
        err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            break;
        }
        if (err < 0) {
            logger_.error("Error polling [NET] ring buffer: " + std::to_string(err));
            break;
        }
    }
    ring_buffer__consume(rb_);

    ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logger_.info("Stopped network profiling at {}", std::ctime(&ttp));
}


void NetSnoop::stop_trace() {
    exiting_ = true;
    if (rb_thread_.joinable()) {
        rb_thread_.join();
    }
    
    if (skel_) {
        net_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }
    logger_.info("Network monitoring thread stopped");
    // logger_.info("Network BPF skeleton destroyed");
}

bool NetSnoop::hasExceededProfilingLimit(
    std::chrono::seconds duration,
    const std::chrono::steady_clock::time_point& startTime) {
    if (duration.count() == 0) { // 0 = profile forever
        return false;
    }
    if (std::chrono::steady_clock::now() - startTime >= duration) {
        logger_.info("Done Network Profiling: exceeded duration of {}s.\n", duration.count());
        return true;
    }
    return false;
}

void NetSnoop::report_net() {
    for (const auto& [pid, stats] : net_stats_) {
        logger_.info("============ 网络报告 ============");
        logger_.info("[NET]  发送: {} bytes, 接收: {} bytes", stats.tx_bytes, stats.rx_bytes);
        logger_.info("[NET]  发送包数: {}, 接收包数: {}", stats.tx_packets, stats.rx_packets);
        logger_.info("[NET]  活跃连接数: {}", stats.connections.size());
        logger_.info("[NET]  监听端口数: {}", stats.listening_ports.size());
        logger_.info("[NET]  平均延迟: {:.2f}ms", stats.avg_latency_ms);
    }
    logger_.info("===================================");
}

} // namespace NeuTracer