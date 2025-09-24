// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifdef FBCODE_STROBELIGHT
#include <bpf/vmlinux/vmlinux.h>
#else
#include "vmlinux.h"
#endif

#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

//! for cuda kernel launch 
#include "gpu_event.h"
struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
} rb SEC(".maps");

// // 跟踪未完成的异步操作
// struct {
//   __uint(type, BPF_MAP_TYPE_HASH);
//   __uint(max_entries, 10240);
//   __type(key, u64);    // 操作ID (地址+时间戳)
//   __type(value, struct mem_access_event);
// } async_mem_ops SEC(".maps");

const volatile struct {
  bool debug;
  bool capture_args;
  bool capture_stack;
} prog_cfg = {
    // These defaults will be overridden from user space
    .debug = true,
    .capture_args = true,
    .capture_stack = true,
};

//! for cudaMalloc and cudaFree



/**
 * Several required data and metadata fields of a memleak event can only be 
 * read from the initial uprobe, but are needed in order to emit events from
 * the uretprobe on return. We map pid to the started event, which is then
 * read and cleared from the uretprobe. This works under the assumption that
 * only one instance of either `cudaMalloc` or `cudaFree` is being executed at
 * a time per process.
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, __u32);
	__type(value, struct memleak_event);
	__uint(max_entries, 1024);
} memleak_pid_to_event SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, __u32);
	__type(value, __u64);
	__uint(max_entries, 1024);
} memleak_pid_to_dev_ptr SEC(".maps");

/**
 * Queue of memleak events that are updated from eBPF space, then dequeued
 * and processed from userspace by the GPUprobe daemon.
 */
// struct {
// 	__uint(type, BPF_MAP_TYPE_QUEUE);
// 	__uint(key_size, 0);
// 	__type(value, struct memleak_event);
// 	__uint(max_entries, 1024);
// } memleak_events_queue SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
} memleak_events_rb SEC(".maps");


//! for cudaMemcpy


/**
 * Maps a pid to an information on an incomplete cudaMemcpy call. This is 
 * needed because we cannot access the input arguments inside of the uretprobe.
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, __u32);
	__type(value, struct cuda_memcpy);
	__uint(max_entries, 10240);
} pid_to_memcpy SEC(".maps");

/**
 * Queue of successful cudaMemcpy calls to be processed from userspace.
 */
// struct {
// 	__uint(type, BPF_MAP_TYPE_QUEUE);
// 	__uint(key_size, 0);
// 	__uint(value_size, sizeof(struct cuda_memcpy));
// 	__uint(max_entries, 10240);
// } successful_cuda_memcpy_q SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
} cuda_memcpy_rb SEC(".maps");


//! for utils 
#define bpf_printk_debug(fmt, ...)    \
  ({                                  \
    if (prog_cfg.debug)               \
      bpf_printk(fmt, ##__VA_ARGS__); \
  })

// The caller uses registers to pass the first 6 arguments to the callee.  Given
// the arguments in left-to-right order, the order of registers used is: %rdi,
// %rsi, %rdx, %rcx, %r8, and %r9. Any remaining arguments are passed on the
// stack in reverse order so that they can be popped off the stack in order.
#define SP_OFFSET(offset) (void*)PT_REGS_SP(ctx) + offset * 8

SEC("uprobe")
int BPF_KPROBE(
    handle_cuda_launch,
    u64 func_off,
    u64 grid_xy,
    u64 grid_z,
    u64 block_xy,
    u64 block_z,
    uintptr_t argv) {
  struct gpukern_sample* e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
  if (!e) {
    // bpf_printk_debug("Failed to allocate ringbuf entry");
    return 0;
  }

  struct task_struct* task = (struct task_struct*)bpf_get_current_task();

  e->pid = bpf_get_current_pid_tgid() >> 32;
  e->ppid = BPF_CORE_READ(task, real_parent, tgid);
  bpf_get_current_comm(&e->comm, sizeof(e->comm));

  e->kern_func_off = func_off;
  e->grid_x = (u32)grid_xy;
  e->grid_y = (u32)(grid_xy >> 32);
  e->grid_z = (u32)grid_z;
  e->block_x = (u32)block_xy;
  e->block_y = (u32)(block_xy >> 32);
  e->block_z = (u32)block_z;

  bpf_probe_read_user(&e->stream, sizeof(uintptr_t), SP_OFFSET(2));

  if (prog_cfg.capture_args) {
    // Read the Cuda Kernel Launch Arguments
    for (int i = 0; i < MAX_GPUKERN_ARGS; i++) {
      const void* arg_addr;
      // We don't know how many argument this kernel has until we parse the
      // signature, so we always attemps to read the maximum number of args,
      // even if some of these arg values are not valid.
      bpf_probe_read_user(
          &arg_addr, sizeof(u64), (const void*)(argv + i * sizeof(u64)));

      bpf_probe_read_user(&e->args[i], sizeof(arg_addr), arg_addr);
    }
  }

  if (prog_cfg.capture_stack) {
    // Read the Cuda Kernel Launch Stack
    e->ustack_sz =
        bpf_get_stack(ctx, e->ustack, sizeof(e->ustack), BPF_F_USER_STACK) /
        sizeof(uint64_t);
  }

  bpf_ringbuf_submit(e, 0);
  return 0;
}

/// uprobe triggered by a call to `cudaMalloc`
SEC("uprobe/cudaMalloc")
int handle_cuda_malloc(struct pt_regs *ctx)
{
	struct memleak_event e = { 0 };
	__u64 dev_ptr;
	__u32 pid, key0 = 0;

	e.size = (__u64)PT_REGS_PARM2(ctx);
	dev_ptr = (__u64) PT_REGS_PARM1(ctx);
	pid = (__u32)bpf_get_current_pid_tgid();

	e.event_type = CUDA_MALLOC;
	e.start = bpf_ktime_get_ns();
	e.pid = pid;

  e.caller_func_off = PT_REGS_IP(ctx);
  e.ustack_sz =
      bpf_get_stack(ctx, e.ustack, sizeof(e.ustack), BPF_F_USER_STACK) /
      sizeof(uint64_t);


	if (bpf_map_update_elem(&memleak_pid_to_event, &pid, &e, 0)) {
		return -1;
	}

	return bpf_map_update_elem(&memleak_pid_to_dev_ptr, &pid, &dev_ptr, 0);
}

/// uretprobe triggered when `cudaMalloc` returns
SEC("uretprobe/cudaMalloc")
int handle_cuda_malloc_ret(struct pt_regs *ctx)
{
	__s32 cuda_malloc_ret;
	__u32 pid;
	struct memleak_event *e;
	__u64 dev_ptr, map_ptr;

	cuda_malloc_ret = (__s32)PT_REGS_RC(ctx);
	pid = (__u32)bpf_get_current_pid_tgid();

	e = bpf_map_lookup_elem(&memleak_pid_to_event, &pid);
	if (!e) {
		return -1;
	}

	e->ret = cuda_malloc_ret;

	// lookup the value of `devPtr` passed to `cudaMalloc` by this process
	map_ptr = (__u64)bpf_map_lookup_elem(&memleak_pid_to_dev_ptr, &pid);
	if (!map_ptr) {
		return -1;
	}
	dev_ptr = *(__u64*)map_ptr;

	// read the value copied into `*devPtr` by `cudaMalloc` from userspace
	if (bpf_probe_read_user(&e->device_addr, sizeof(void *), (void*)dev_ptr)) {
		return -1;
	}

	e->end = bpf_ktime_get_ns();

	// return bpf_map_push_elem(&memleak_events_queue, e, 0);
  struct memleak_event *rb_event = bpf_ringbuf_reserve(&memleak_events_rb, sizeof(*rb_event), 0);
  if (!rb_event) {
      bpf_printk_debug("Failed to reserve ringbuffer for cudaMalloc event");
      return -1;
  }

  // 复制事件数据
  __builtin_memcpy(rb_event, e, sizeof(*rb_event));
  
  // 提交事件
  bpf_ringbuf_submit(rb_event, 0);
  
  // // 清理临时数据
  // bpf_map_delete_elem(&memleak_pid_to_event, &pid);
  // bpf_map_delete_elem(&memleak_pid_to_dev_ptr, &pid);
  
  return 0;
}

/// uprobe triggered by a call to `cudaFree`
SEC("uprobe/cudaFree")
int handle_cuda_free(struct pt_regs *ctx)
{
	struct memleak_event e = { 0 };

	e.event_type = CUDA_FREE;
	e.pid = (u32)bpf_get_current_pid_tgid();
	e.start = bpf_ktime_get_ns();
	e.device_addr = (__u64)PT_REGS_PARM1(ctx);

   e.caller_func_off = PT_REGS_IP(ctx);

	return bpf_map_update_elem(&memleak_pid_to_event, &e.pid, &e, 0);
}

/// uretprobe triggered when `cudaFree` returns
SEC("uretprobe/cudaFree")
int handle_cuda_free_ret(struct pt_regs *ctx)
{
	__s32 cuda_free_ret;
	__u32 pid;
	struct memleak_event *e;

	pid = (__u32)bpf_get_current_pid_tgid();

	e = (struct memleak_event *)bpf_map_lookup_elem(&memleak_pid_to_event,
							&pid);
	if (!e) {
		return -1;
	}

	e->end = bpf_ktime_get_ns();
	e->ret = PT_REGS_RC(ctx);

	// return bpf_map_push_elem(&memleak_events_queue, e, 0);
  // 修改这里：使用 ringbuffer 替代 queue
  struct memleak_event *rb_event = bpf_ringbuf_reserve(&memleak_events_rb, sizeof(*rb_event), 0);
  if (!rb_event) {
      bpf_printk_debug("Failed to reserve ringbuffer for cudaFree event");
      return -1;
  }

  // 复制事件数据
  __builtin_memcpy(rb_event, e, sizeof(*rb_event));
  
  // 提交事件
  bpf_ringbuf_submit(rb_event, 0);
  
  // // 清理临时数据
  // bpf_map_delete_elem(&memleak_pid_to_event, &pid);
  
  return 0;
}


/**
 * This function exhibits synchronous behavior in MOST cases as specified by
 * Nvidia documentation. It is under the assumption that this call is 
 * synchronous that we compute the average memory bandwidth of a transfer as:
 *		avg_throughput = count /  (end - start)
 */
// 修复 handle_cuda_memcpy 函数
SEC("uprobe/cudaMemcpy")
int handle_cuda_memcpy(struct pt_regs *ctx)
{
    __u64 dst = PT_REGS_PARM1(ctx);
    __u64 src = PT_REGS_PARM2(ctx);
    __u64 count = PT_REGS_PARM3(ctx);
    enum memcpy_kind kind = PT_REGS_PARM4(ctx);
    __u32 pid = (__u32)bpf_get_current_pid_tgid();

    /* no host-side synchronization is performed in the D2D case - as a result,
     * we cannot compute average throughput using information available from
     * this uprobe. If the DEFAULT argument is passed, we cannot make any 
     * assumption on the direction of the transfer */
    if (kind == D2D || kind == DEFAULT)
        return 0;

    struct cuda_memcpy in_progress_memcpy = {0};  // 初始化为0
    
    // 单独设置每个字段，避免一次性复杂赋值
    in_progress_memcpy.start_time = bpf_ktime_get_ns();
    in_progress_memcpy.dst = dst;
    in_progress_memcpy.src = src;
    in_progress_memcpy.count = count;
    in_progress_memcpy.kind = kind;

    if (bpf_map_update_elem(&pid_to_memcpy, &pid, &in_progress_memcpy, 0)) {
        return -1;
    }

    return 0;
}

// 修复 handle_cuda_memcpy_ret 函数
SEC("uretprobe/cudaMemcpy")
int handle_cuda_memcpy_ret(struct pt_regs *ctx)
{
    __u32 ret = PT_REGS_RC(ctx);
    __u32 pid = (__u32)bpf_get_current_pid_tgid();
    
    // 如果返回值不为0，表示失败
    if (ret) {
        return -1;
    }

    // 查找之前存储的memcpy信息
    struct cuda_memcpy *exited_memcpy = bpf_map_lookup_elem(&pid_to_memcpy, &pid);
    if (!exited_memcpy) {
        return -1;
    }

    // 记录结束时间
    __u64 end_time = bpf_ktime_get_ns();
    
    // 准备要提交到ringbuffer的数据
    struct cuda_memcpy rb_memcpy = {0};
    
    // 复制数据到本地变量
    rb_memcpy.start_time = exited_memcpy->start_time;
    rb_memcpy.end_time = end_time;
    rb_memcpy.dst = exited_memcpy->dst;
    rb_memcpy.src = exited_memcpy->src;
    rb_memcpy.count = exited_memcpy->count;
    rb_memcpy.kind = exited_memcpy->kind;
    rb_memcpy.pid = pid;
    
    // 清理临时数据
    bpf_map_delete_elem(&pid_to_memcpy, &pid);
    
    // 提交到ringbuffer
    struct cuda_memcpy *rb_event = bpf_ringbuf_reserve(&cuda_memcpy_rb, sizeof(*rb_event), 0);
    if (!rb_event) {
        bpf_printk_debug("Failed to reserve ringbuffer for cudaMemcpy event");
        return -1;
    }

    // 复制数据到ringbuffer
    __builtin_memcpy(rb_event, &rb_memcpy, sizeof(rb_memcpy));
    
    // 提交事件
    bpf_ringbuf_submit(rb_event, 0);
    
    return 0;
}

// // 跟踪cudaMalloc调用
// SEC("uprobe/cudaMalloc")
// int BPF_KPROBE(handle_cuda_malloc, void **devPtr, size_t size) {
//     // 创建一个暂存结构体在栈上
//     struct mem_access_event event = {0};
    
//     event.timestamp = bpf_ktime_get_ns();
//     event.size = size;
//     event.pid = bpf_get_current_pid_tgid() >> 32;
//     event.tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
//     event.access_type = 1; // 分配操作
//     event.is_async = 0;
//     bpf_get_current_comm(&event.comm, sizeof(event.comm));
    
//     // 测试: 直接提交一个事件到ringbuffer，不等待返回探针
//     struct mem_access_event *test_event = bpf_ringbuf_reserve(&mem_events, sizeof(*test_event), 0);
//     if (test_event) {
//         // 复制本地事件到ringbuffer
//         __builtin_memcpy(test_event, &event, sizeof(event));
//         // 设置一些特殊值便于识别这是测试事件
//         test_event->address = (u64)devPtr;  // 保存指针地址，不是实际分配的地址
//         test_event->host_address = 0xdeadbeef;  // 特殊标记
        
        
//         // 提交事件
//         bpf_ringbuf_submit(test_event, 0);
//     }
    
//     // 我们需要在返回探针中获取实际分配的地址
//     u64 key = (u64)devPtr;
//     // 从栈中复制到 map
//     return bpf_map_update_elem(&async_mem_ops, &key, &event, BPF_ANY);
// }

// // 跟踪cudaMalloc返回，获取分配的地址
// SEC("uretprobe/cudaMalloc")
// int BPF_KRETPROBE(handle_cuda_malloc_ret, int ret) {
//     if (ret != 0) {
//         // 分配失败，不处理
//         return 0;
//     }
    
//     u64 key = (u64)PT_REGS_PARM1(ctx); // 获取devPtr参数
//     struct mem_access_event *map_event = bpf_map_lookup_elem(&async_mem_ops, &key);
//     if (!map_event) {
//         return 0;
//     }
    
//     // 从 map 获取事件，读取设备地址
//     void *dev_addr;
//     bpf_probe_read_user(&dev_addr, sizeof(dev_addr), (void*)key);
    
//     // 为 ringbuffer 创建新的事件
//     struct mem_access_event *rb_event = bpf_ringbuf_reserve(&mem_events, sizeof(*rb_event), 0);
//     if (!rb_event) {
//         bpf_map_delete_elem(&async_mem_ops, &key);
//         return 0;
//     }
    
//     // 复制 map 中的事件到 ringbuffer 事件
//     __builtin_memcpy(rb_event, map_event, sizeof(*rb_event));
//     rb_event->address = (u64)dev_addr;
    
//     // 从 ringbuffer 提交事件并从 map 中删除
//     bpf_ringbuf_submit(rb_event, 0);
//     bpf_map_delete_elem(&async_mem_ops, &key);
    
//     return 0;
// }

// // 跟踪cudaMemcpy调用
// SEC("uprobe/cudaMemcpy")
// int BPF_KPROBE(handle_cuda_memcpy, void *dst, const void *src, size_t count, int kind) {
//     struct mem_access_event *event = bpf_ringbuf_reserve(&mem_events, sizeof(*event), 0);
//     if (!event) {
//         return 0;
//     }
    
//     event->timestamp = bpf_ktime_get_ns();
//     event->size = count;
//     event->pid = bpf_get_current_pid_tgid() >> 32;
//     event->tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
//     event->is_async = 0;
//     bpf_get_current_comm(&event->comm, sizeof(event->comm));
    
//     // 解析cudaMemcpyKind
//     // cudaMemcpyHostToDevice = 1
//     // cudaMemcpyDeviceToHost = 2
//     // cudaMemcpyDeviceToDevice = 3
//     if (kind == 1) {
//         event->access_type = 2; // 主机到设备
//         event->address = (u64)dst;
//         event->host_address = (u64)src;
//     } else if (kind == 2) {
//         event->access_type = 3; // 设备到主机
//         event->address = (u64)src;
//         event->host_address = (u64)dst;
//     } else {
//         event->access_type = 5; // 其他
//         event->address = (u64)dst;
//         event->host_address = (u64)src;
//     }
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

char LICENSE[] SEC("license") = "Dual MIT/GPL";
