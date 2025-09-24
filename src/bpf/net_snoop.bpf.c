// SPDX-License-Identifier: GPL-2.0
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_endian.h>
#define TASK_COMM_LEN 16
#define ETH_P_IP 0x0800    // IP 协议
#define ETH_HLEN 14        // 以太网头部长度
#define IP_OFFSET (ETH_HLEN)

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 3);
    __type(key, u32);
    __type(value, u32);
} snoop_proc SEC(".maps");
struct throughput_key {
    u32 pid;
    char name[TASK_COMM_LEN];
};

struct net_event {
    u32 tgid;
    u32 pid;
    u64 bytes;
    u64 timestamp;
    bool is_send;         // true for send, false for receive
    char comm[TASK_COMM_LEN];
    u32 saddr;            // 源 IP 地址
    u32 daddr;            // 目标 IP 地址
    __u16 sport;          // 源端口
    __u16 dport;          // 目标端口
    __u8 protocol;        // 协议
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, struct throughput_key);
    __type(value, u64);
} send_bytes SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, struct throughput_key);
    __type(value, u64);
} recv_bytes SEC(".maps");


struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} net_events SEC(".maps");

// 提取 IP header 信息的辅助函数
static __always_inline bool parse_ip_header(struct sk_buff *skb, struct net_event *event)
{
    // 确保我们有一个有效的网络头
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
        
        // 现在我们可以安全地使用 iph 结构体
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

SEC("tp_btf/net_dev_queue")
int BPF_PROG(net_dev_queue, struct sk_buff *skb)
{
    u32 tgid = bpf_get_current_pid_tgid() >> 32;
    u32 pid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
        
    struct throughput_key key = {
        .pid = tgid
    };
    if(bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL && bpf_map_lookup_elem(&snoop_proc, &pid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }
    bpf_get_current_comm(&key.name, sizeof(key.name));
    
    u64 *val = bpf_map_lookup_elem(&send_bytes, &key);
    u64 new_val = skb->len;
    
    if (val) {
        new_val += *val;
    }
    
    bpf_map_update_elem(&send_bytes, &key, &new_val, BPF_ANY);
    
    struct net_event *event;
    event = bpf_ringbuf_reserve(&net_events, sizeof(*event), 0);
    if (event) {
        event->tgid = tgid;
        event->pid = pid;
        event->bytes = skb->len;
        event->timestamp = bpf_ktime_get_ns();
        event->is_send = true;
        event->saddr = 0;
        event->daddr = 0;
        event->sport = 0;
        event->dport = 0;
        event->protocol = 0;
        bpf_get_current_comm(&event->comm, sizeof(event->comm));
        
        // 提取 IP 信息
        parse_ip_header(skb, event);
        
        bpf_ringbuf_submit(event, 0);
    }
    
    return 0;
}

SEC("tp_btf/netif_receive_skb")
int BPF_PROG(netif_receive_skb, struct sk_buff *skb)
{
    u32 tgid = bpf_get_current_pid_tgid() >> 32;
    u32 pid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    if(bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL && bpf_map_lookup_elem(&snoop_proc, &pid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }
        
    struct throughput_key key = {
        .pid = tgid
    };
    bpf_get_current_comm(&key.name, sizeof(key.name));
    
    u64 *val = bpf_map_lookup_elem(&recv_bytes, &key);
    u64 new_val = skb->len;
    
    if (val) {
        new_val += *val;
    }
    
    bpf_map_update_elem(&recv_bytes, &key, &new_val, BPF_ANY);
    
    // Send event to ring buffer
    struct net_event *event;
    event = bpf_ringbuf_reserve(&net_events, sizeof(*event), 0);
    if (event) {
        event->tgid = tgid;
        event->pid = pid;   
        event->bytes = skb->len;
        event->timestamp = bpf_ktime_get_ns();
        event->is_send = false;
        event->saddr = 0;
        event->daddr = 0;
        event->sport = 0;
        event->dport = 0;
        event->protocol = 0;
        bpf_get_current_comm(&event->comm, sizeof(event->comm));
        
        // 提取 IP 信息
        parse_ip_header(skb, event);
        
        bpf_ringbuf_submit(event, 0);
    }
    
    return 0;
}

char _license[] SEC("license") = "GPL";