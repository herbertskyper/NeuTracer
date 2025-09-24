// SPDX-License-Identifier: GPL-2.0
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_tracing.h>
#include "bpf_config/config.h"
#define TASK_COMM_LEN 16
#define RWBS_LEN 8


struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 3);
    __type(key, u32);
    __type(value, u32);
} snoop_proc SEC(".maps");

struct block_rq_issue_ctx {
    // 公共头部
    unsigned short common_type;
    unsigned char common_flags;
    unsigned char common_preempt_count;
    int common_pid;
    
    // 特定字段
    __u32 dev;           // offset: 8
    __u64 sector;        // offset: 16
    __u32 nr_sector;     // offset: 24
    __u32 bytes;         // offset: 28
    // char rwbs[RWBS_LEN];
    // char comm[TASK_COMM_LEN];
    // __u32 rwbs_loc;
    // __u64 comm_loc;
     char rwbs[RWBS_LEN]; // offset: 32 - 直接是数组，不是 __data_loc
    char comm[TASK_COMM_LEN]; // offset: 40 - 直接是数组，不是 __data_loc
    __u32 cmd_loc;       // offset: 56 (注意这是 __data_loc 字段)
};

struct block_rq_complete_ctx {
    // 公共头部
    unsigned short common_type;
    unsigned char common_flags;
    unsigned char common_preempt_count;
    int common_pid;
    
    // 特定字段
    __u32 dev;
    __u64 sector;
    unsigned int nr_sector;
    int error;
    // __u32 rwbs_loc;
    char rwbs[RWBS_LEN]; // offset: 32 - 直接是数组，不是 __data_loc
    __u32 cmd_loc;
};

// 基本数据结构
struct req_key_t {
    __u32 dev_major;
    __u32 dev_minor;
    __u64 sector;
};

struct start_req_t {
    __u64 ts;
    __u32 bytes;
    char rwbs[RWBS_LEN];
    char comm[TASK_COMM_LEN];
};

// 环形缓冲区事件结构
struct bio_event {
    __u32 pid;
    __u32 tgid; // 真实进程 ID
    char comm[TASK_COMM_LEN];
    __u64 delta_us;
    __u64 bytes;
    __u8 rwflag;
    __u8 major;
    __u8 minor;
    __u64 timestamp;
    __u64 io_count;
};


// 映射定义
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, struct req_key_t);
    __type(value, struct start_req_t);
} start SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u8);
    __type(value, __u64);
} io_count SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} bio_events SEC(".maps");

// 从 rwbs 字段判断是否为写操作
static __always_inline int is_write(char *rwbs) {
    bpf_printk("rwbs: %s\n", rwbs);  // 调试输出
    for (int i = 0; i < RWBS_LEN && i < 3; i++) {  // 通常只需检查前几个字符
        if (rwbs[i] == 0)
            break;
        if (rwbs[i] == 'W' || rwbs[i] == 'w' )  
            return 1;
    }
    
    return 0;
}
// 安全读取 tracepoint 上下文字段
static __always_inline int read_ctx_field_int(void *ctx, size_t offset) {
    int value = 0;
    bpf_probe_read(&value, sizeof(value), (void *)ctx + offset);
    return value;
}

// 安全读取 tracepoint __data_loc 字符串 - 修复版本
static __always_inline 
void read_str_at_loc(void *ctx, __u32 data_loc, char *dest, size_t size) {
    // 确保合法的偏移量
    if (data_loc == 0) {
        dest[0] = '\0';  // 设置为空字符串
        return;
    }
    
    // 计算字符串在事件缓冲区中的位置
    unsigned long str_ptr = (unsigned long)ctx + (data_loc & 0xffff);
    
    // 确保目标缓冲区以 null 终止
    bpf_probe_read_str(dest, size, (void *)str_ptr);
}

// 安全读取数据
static __always_inline int safe_bpf_probe_read(void *dst, size_t size, const void *src) {
    int ret = bpf_probe_read(dst, size, src);
    if (ret < 0) {
        // 如果读取失败，确保目标内存是干净的
        __builtin_memset(dst, 0, size);
    }
    return ret;
}

SEC("tp/block/block_rq_issue")
int handle_block_rq_issue(struct block_rq_issue_ctx *ctx) {
    // 预先从上下文读取所有需要的数据，减少对 ctx 的多次访问
    __u32 dev;
    __u64 sector;
    __u32 bytes;
    char rwbs[RWBS_LEN] = {0}; 
    char comm[TASK_COMM_LEN] = {0};

// #ifdef INTERNAL
//     if(ts % INTERVAL_NS){
//         return 0; // 如果不在采样间隔内，则跳过
//     }
// #endif

     // 获取目标PID
    __u32 kkey = 0;
    
    // 获取当前进程ID
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;  // 高32位是进程ID
    u32 pid = pid_tgid & 0xFFFFFFFF; // 低32位是线程ID (PID)
    
    // // 如果设置了目标PID且不匹配，则跳过
   if(tgid != 0 && pid != 0 && bpf_map_lookup_elem(&snoop_proc, &pid) == NULL && bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }
    
    // 一次性读取所有字段
    // if (bpf_probe_read(&dev, sizeof(dev), &ctx->dev) ||
    //     bpf_probe_read(&sector, sizeof(sector), &ctx->sector) ||
    //     bpf_probe_read(&bytes, sizeof(bytes), &ctx->bytes) ||
    //     // bpf_probe_read(&rwbs_loc, sizeof(char) * RWBS_LEN, &ctx->rwbs) ||
    //     // bpf_probe_read(&comm_loc, sizeof(char) * TASK_COMM_LEN, &ctx->comm) ||
    //     bpf_probe_read(&rwbs_loc, sizeof(rwbs_loc), &ctx->rwbs_loc) ||
    //     bpf_probe_read(&comm_loc, sizeof(comm_loc), &ctx->comm_loc) ||
    //     // bpf_probe_read(&rwbs_loc, RWBS_LEN, &ctx->rwbs) ||       // 直接读取数组
    //     // bpf_probe_read(&comm_loc, TASK_COMM_LEN, &ctx->comm) ||     
    //     bpf_probe_read(&pid, sizeof(pid), &ctx->common_pid)) {
    //     return 0; // 读取失败，提前返回
    // }
    if (bpf_probe_read(&dev, sizeof(dev), &ctx->dev) ||
        bpf_probe_read(&sector, sizeof(sector), &ctx->sector) ||
        bpf_probe_read(&bytes, sizeof(bytes), &ctx->bytes) ||
        bpf_probe_read(&rwbs, sizeof(rwbs), &ctx->rwbs) ||        // 直接读取嵌入的数组
        bpf_probe_read(&comm, sizeof(comm), &ctx->comm) ||       // 直接读取嵌入的数组
        bpf_probe_read(&pid, sizeof(pid), &ctx->common_pid)) {
        return 0; // 读取失败，提前返回
    }
    
    // 继续使用本地变量而不是直接访问 ctx
    struct req_key_t key = {0};
    struct start_req_t start_data = {0};
    
    // 解析设备号
    key.dev_major = dev >> 20;
    key.dev_minor = dev & ((1U << 20) - 1);
    key.sector = sector;
    
    // 记录请求开始时间和细节
    start_data.ts = bpf_ktime_get_ns();
    start_data.bytes = bytes;

    // 复制 rwbs 和 comm 字段
    for (int i = 0; i < RWBS_LEN; i++) {
        start_data.rwbs[i] = rwbs[i];
    }
    
    for (int i = 0; i < TASK_COMM_LEN; i++) {
        start_data.comm[i] = comm[i];
    }
    
    // 更新映射
    bpf_map_update_elem(&start, &key, &start_data, 0);
    
    return 0;
}

SEC("tp/block/block_rq_complete")
int handle_block_rq_complete(struct block_rq_complete_ctx *ctx) {
    // 获取目标PID
    __u32 kkey = 0;
    
    // 获取当前进程ID
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;  // 高32位是进程ID
    u32 pid = pid_tgid & 0xFFFFFFFF; // 低32位是线程ID (PID)
// #ifdef INTERNAL
//     if(ts % INTERVAL_NS){
//         return 0; // 如果不在采样间隔内，则跳过
//     }
// #endif
    // 如果设置了目标PID且不匹配，则跳过
    if(tgid != 0 && pid != 0 && bpf_map_lookup_elem(&snoop_proc, &pid) == NULL && bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL) {
        return 0; // 如果进程不在 snoop_proc 中，则不处理
    }

    struct req_key_t key = {0};
    struct start_req_t *startp;
    __u64 delta_us;
    __u8 major, minor;
    
    // 安全读取设备字段
    __u32 dev = 0;
    bpf_probe_read(&dev, sizeof(dev), &ctx->dev);
    
    // 解析设备号
    major = dev >> 20;
    minor = dev & ((1U << 20) - 1);
    key.dev_major = major;
    key.dev_minor = minor;
    
    // 安全读取扇区
    bpf_probe_read(&key.sector, sizeof(key.sector), &ctx->sector);
    
    // 获取开始时间
    startp = bpf_map_lookup_elem(&start, &key);
    if (!startp) {
        return 0;  // 找不到开始记录
    }
    
    // 计算时延
    delta_us = (bpf_ktime_get_ns() - startp->ts) / 1000;
    
    // 更新 IO 计数
    __u64 zero = 0;
    __u64 *count;
    __u8 dev_id = major;
    
    count = bpf_map_lookup_elem(&io_count, &dev_id);
    if (!count) {
        bpf_map_update_elem(&io_count, &dev_id, &zero, 0);
        count = bpf_map_lookup_elem(&io_count, &dev_id);
        if (!count)
            goto cleanup;
    }
    
    // 增加 IO 计数 - 使用原子操作
    __sync_fetch_and_add(count, 1);
    
    // 创建事件并发送到环形缓冲区
    struct bio_event *event;
    event = bpf_ringbuf_reserve(&bio_events, sizeof(*event), 0);
    if (event) {
        // 正确获取 PID 和 TGID
        u64 pid_tgid = bpf_get_current_pid_tgid();
        u32 pid = pid_tgid & 0xFFFFFFFF;  // 低 32 位是线程 ID (PID)
        u32 tgid = pid_tgid >> 32;        // 高 32 位是进程 ID (TGID)
        
        // 存储正确的 PID（使用 TGID，即真实进程 ID）
        event->pid = pid;
        event->tgid = tgid;  // 存储 TGID
        
        // 获取当前进程名称 - 使用 bpf_get_current_comm 
        char comm[TASK_COMM_LEN];
        __builtin_memset(comm, 0, sizeof(comm));
        bpf_get_current_comm(&comm, sizeof(comm));
        
        // 安全地复制进程名称
        __builtin_memset(event->comm, 0, sizeof(event->comm));  // 先清空
        for (int i = 0; i < TASK_COMM_LEN-1 && comm[i]; i++) {
            event->comm[i] = comm[i];
        }

        event->delta_us = delta_us;
        event->bytes = startp->bytes;
        event->rwflag = is_write(startp->rwbs);
        event->major = major;
        event->minor = minor;
        event->timestamp = bpf_ktime_get_ns();
        event->io_count = *count;
        
        bpf_ringbuf_submit(event, 0);
    }
    
cleanup:
    bpf_map_delete_elem(&start, &key);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";