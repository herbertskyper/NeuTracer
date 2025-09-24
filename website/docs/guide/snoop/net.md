# 网络模块 用户态部分
网络部分 eBPF 模块收集到网络相关数据后，NetSnoop 对收集到流量统计数据进行分析，识别网络连接模式、检测异常行为，并提供网络行为特征。

核心数据结构包括网络事件结构（net_event：跟epbf侧结构体保持一致）和网络统计结构（NetStats）。网络统计结构提供每个进程的详细网络使用统计和异常检测数据，包含基础信息、流量统计、流量历史、连接信息和分类异常检测等全面数据。

```cpp
struct NetStats {
    uint32_t pid{0};                  ///< 进程ID
    uint32_t tgid{0};                 ///< 线程组ID
    std::string comm;                 ///< 进程名
    
    uint64_t tx_bytes{0};             ///< 总发送字节数
    uint64_t rx_bytes{0};             ///< 总接收字节数
    uint64_t tx_packets{0};           ///< 总发送包数
    uint64_t rx_packets{0};           ///< 总接收包数
    
    // 流量历史(用于计算速率和检测突发)
    std::deque<uint64_t> tx_bytes_history;  ///< 发送字节历史
    std::deque<uint64_t> rx_bytes_history;  ///< 接收字节历史
    std::deque<double> tx_timestamps;       ///< 发送时间戳
    std::deque<double> rx_timestamps;       ///< 接收时间戳
    
    // 连接信息
    std::set<std::string> connections;      ///< 活跃连接集合
    std::set<uint16_t> listening_ports;     ///< 监听端口集合
    
    double avg_latency_ms{0.0};             ///< 平均延迟(毫秒)
};
```

`attach_bpf()` 初始化 eBPF 程序并设置事件收集，将 eBPF 程序附加到系统调用点，`start_trace()` 启动监控线程和 ring buffer 处理，`stop_trace()` 停止监控并清理资源。`process_event()` 处理从 eBPF 接收的网络事件，`update_net_stats()` 更新特定进程的网络统计信息，`ring_buffer_thread()` 后台线程循环处理网络事件。`report_net()` 生成网络报告。辅助方法包括 `format_ip_address()` 将32位IP地址转换为可读格式，`format_protocol()` 将协议号转换为协议名称，`format_throughput()` 格式化网络吞吐量为可读形式。
```cpp
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
```

NetSnoop 可以识别多种网络相关异常，比如发送和接收流量突发、连接数异常、超大数据包传输等。每个异常都会记录日志，帮助用户快速定位问题。

```cpp
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
```