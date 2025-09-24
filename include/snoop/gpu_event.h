// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once


#define TASK_COMM_LEN 16
#define MAX_GPUKERN_ARGS 16

#ifndef MAX_STACK_DEPTH
#define MAX_STACK_DEPTH 128
#endif

typedef uint64_t stack_trace_t[MAX_STACK_DEPTH];

struct gpukern_sample {
  int pid, ppid;
  char comm[TASK_COMM_LEN];
  uint64_t kern_func_off;
  int grid_x, grid_y, grid_z;
  int block_x, block_y, block_z;
  uint64_t stream;
  uint64_t args[MAX_GPUKERN_ARGS];
  size_t ustack_sz;
  stack_trace_t ustack;
};


enum memleak_event_t {
	CUDA_MALLOC = 0,
	CUDA_FREE,
};

/**
 * Wraps the arguments passed to `cudaMalloc` or `cudaFree`, and return code,
 * and some metadata
 */
struct memleak_event {
	__u64 start;
	__u64 end;
	__u64 device_addr;
	__u64 size;
	__u32 pid;
	__s32 ret;
	enum memleak_event_t event_type;

	uint64_t caller_func_off;       // 调用者函数偏移地址
	size_t ustack_sz;
  	uint64_t ustack[8];
};

/**
 * redefinition of `enum cudaMemcpyKind` in driver_types.h.
 */
enum memcpy_kind {
	H2H = 0, // host to host
	H2D = 1, // host to device
	D2H = 2, // device to host
	D2D = 3, // device to device
	DEFAULT = 4, // inferred from pointer type at runtime
};

struct cuda_memcpy {
	__u64 start_time;
	__u64 end_time;
	__u64 dst;
	__u64 src;
	__u64 count;
	__u32 pid;
	enum memcpy_kind kind;
};
