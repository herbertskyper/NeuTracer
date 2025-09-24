# 网络模块 eBPF 部分

`net_snoop.bpf.c` 用于实时跟踪和分析系统中的网络流量，能够捕获发送和接收的网络数据包，提取详细的连接信息，并按进程提供精确的网络使用统计。该模块可以同时跟踪网络发送和接收操作，按进程聚合网络使用数据，识别TCP、UDP等不同协议的流量，提取源/目标IP地址和端口信息，持续更新吞吐量统计。程序挂载到两个核心网络跟踪点来实现完整的网络流量监控。`net_dev_queue`在网络数据包发送时触发，`netif_receive_skb`在网络数据包接收时触发。

代码采用网络事件结构（net_event）传递完整的网络事件信息到用户空间，包含进程ID、数据包大小、时间戳、发送/接收标志、进程名称、源/目标IP地址、源/目标端口和协议类型。吞吐量结构体（throughput_key）用于在哈希表中标识特定进程，包含进程ID和进程名称，便于进行进程级别的流量统计和分析。

```c
struct net_event {
    u32 pid;                 // 进程 ID
    u64 bytes;               // 数据包大小
    u64 timestamp;           // 时间戳
    bool is_send;            // true=发送, false=接收
    char comm[TASK_COMM_LEN]; // 进程名称
    u32 saddr;               // 源 IP 地址
    u32 daddr;               // 目标 IP 地址
    __u16 sport;             // 源端口
    __u16 dport;             // 目标端口
    __u8 protocol;           // 协议类型
};
struct throughput_key {
    u32 pid;
    char name[TASK_COMM_LEN];
};
```

程序使用发送字节统计映射（send_bytes）和接收字节统计映射（recv_bytes）分别跟踪每个进程发送和接收的总流量。进程过滤器映射（snoop_proc）提供可选的进程过滤功能，网络事件环形缓冲区（net_events）向用户空间传输事件。
```c
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, struct throughput_key);
    __type(value, u64);
} send_bytes SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, struct throughput_key);
    __type(value, u64);
} recv_bytes SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} net_events SEC(".maps");
```

网络数据包处理流程：首先获取进程信息并准备键值、更新发送或接收字节统计、创建并填充网络事件、解析IP头部信息，最后提交事件到环形缓冲区。IP头部解析功能检查网络头部有效性，仅处理IP协议数据包，安全读取IP头部信息，提取IP地址和协议信息，并根据协议类型（TCP、UDP等）提取相应的端口信息。
```c
// 提取 IP header 信息的辅助函数
static __always_inline bool parse_ip_header(struct sk_buff *skb, struct net_event *event)
{
    // 确保有一个有效的网络头
    if (skb->network_header == 0)
        return false;
    // 使用 bpf_probe_read 安全地读取数据
    struct iphdr iph;
    u16 proto = bpf_ntohs(skb->protocol);
    // 仅处理 IP 数据包
    if (proto == ETH_P_IP) {
        // 安全地读取 IP 头
        void *ip_ptr = (void *)(skb->head + skb->network_header);
        if (bpf_probe_read(&iph, sizeof(iph), ip_ptr) < 0)
            return false;
        event->saddr = iph.saddr;
        event->daddr = iph.daddr;
        event->protocol = iph.protocol;
        // 如果是 TCP 或 UDP，获取端口信息
        if (iph.protocol == IPPROTO_TCP) {
            struct tcphdr tcph;
            void *tcp_ptr = ip_ptr + (iph.ihl * 4);
            if (bpf_probe_read(&tcph, sizeof(tcph), tcp_ptr) == 0) {
                event->sport = bpf_ntohs(tcph.source);
                event->dport = bpf_ntohs(tcph.dest);
            }
        } 
        else if (iph.protocol == IPPROTO_UDP) {
            struct udphdr udph;
            void *udp_ptr = ip_ptr + (iph.ihl * 4);
            if (bpf_probe_read(&udph, sizeof(udph), udp_ptr) == 0) {
                event->sport = bpf_ntohs(udph.source);
                event->dport = bpf_ntohs(udph.dest);
            }
        }
        return true;
    }
    return false;
}
```