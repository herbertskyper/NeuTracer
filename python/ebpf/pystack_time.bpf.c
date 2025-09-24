#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <linux/ptrace.h>

// Python 3.10类型定义
struct PyTypeObject {
	char _[24]; // 未使用的填充字段
	char *tp_name; // 类型名称
};

struct PyObject {
	char _[8]; // 未使用的填充字段
	struct PyTypeObject *ob_type;
};

struct PyVarObject {
	struct PyObject ob_base;
	char _[8]; // 未使用的填充字段
};

struct _PyStr {
	char _[48]; // 未使用的填充字段
	char buf[100]; // 字符串内容
};

struct PyCodeObject {
	char _[104]; // 未使用的填充字段
	struct _PyStr *co_filename;
	struct _PyStr *co_name;
};

struct PyFrameObject {
	struct PyVarObject ob_base;
	struct PyFrameObject *f_back;
	struct PyCodeObject *f_code;
};

// 栈帧信息结构
struct stack_frame_info {
	char filename[100];
	char funcname[100];
};

// 添加时间戳字段，单位纳秒
struct stack_trace {
    __u32 pid;
    __u32 num_frames;
    __u64 timestamp_ns;   // 新增：内核采样时刻
    struct stack_frame_info frames[20];
};

// 存储栈信息的BPF映射
struct {
	__uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
	__uint(key_size, sizeof(int));
	__uint(value_size, sizeof(int));
} events SEC(".maps");

// 过滤的PID映射
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(key_size, sizeof(int));
	__uint(value_size, sizeof(int));
	__uint(max_entries, 1);
} filter_pid SEC(".maps");

// 用于存储栈追踪数据的每CPU映射
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(key_size, sizeof(int));
	__uint(value_size, sizeof(struct stack_trace));
	__uint(max_entries, 1);
} stack_traces SEC(".maps");

SEC("perf_event")
int python_stack_trace(struct pt_regs *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 pid = pid_tgid >> 32;
	__u32 tgid = (__u32)pid_tgid;

	// 获取过滤的PID
	int key = 0;
	int *target_pid = bpf_map_lookup_elem(&filter_pid, &key);
	if (!target_pid || *target_pid != tgid)
		return 0;

	// 获取栈指针
	void *sp = (void *)PT_REGS_SP(ctx);
	struct PyFrameObject *frame = NULL;
	unsigned long i = 0;

	// 搜索栈中的PyFrameObject指针
	for (i = 0; i < 200 && !frame; i++) {
		void **addr = (void **)(sp + i * sizeof(void *));
		void *potential_frame = NULL;

		// 安全地读取内存
		bpf_probe_read(&potential_frame, sizeof(potential_frame), addr);
		if (!potential_frame)
			continue;

		// 尝试验证是否为PyFrameObject
		struct PyObject *obj = (struct PyObject *)potential_frame;
		struct PyTypeObject *type = NULL;

		// 读取对象类型
		if (bpf_probe_read(&type, sizeof(type), &obj->ob_type))
			continue;
		if (!type)
			continue;

		// 读取类型名称
		char *tp_name = NULL;
		if (bpf_probe_read(&tp_name, sizeof(tp_name), &type->tp_name))
			continue;
		if (!tp_name)
			continue;

		// 读取类型名称的前5个字符
		char t0, t1, t2, t3, t4;

		bpf_probe_read(&t0, sizeof(t0), tp_name);
		bpf_probe_read(&t1, sizeof(t1), tp_name + 1);
		bpf_probe_read(&t2, sizeof(t2), tp_name + 2);
		bpf_probe_read(&t3, sizeof(t3), tp_name + 3);
		bpf_probe_read(&t4, sizeof(t4), tp_name + 4);

		if (t0 == 'f' && t1 == 'r' && t2 == 'a' && t3 == 'm' && t4 == 'e') {
			frame = (struct PyFrameObject *)potential_frame;
		}
	}

	// 如果没找到PyFrameObject则返回
	if (!frame)
		return 0;

	// 获取per-CPU映射中的栈追踪结构
	int zero = 0;
	struct stack_trace *trace = bpf_map_lookup_elem(&stack_traces, &zero);
	if (!trace)
		return 0;

	// 采集时间戳
    trace->timestamp_ns = bpf_ktime_get_ns();

	// 手动初始化结构体
	trace->pid = tgid;
	trace->num_frames = 0;

// 由于无法使用memset，我们需要确保至少将使用到的帧初始化为0
#pragma unroll
	for (i = 0; i < 20; i++) {
		trace->frames[i].filename[0] = 0;
		trace->frames[i].funcname[0] = 0;
	}

	// 收集栈信息，最多20个栈帧
	struct PyFrameObject *current_frame = frame;
	for (i = 0; i < 20 && current_frame; i++) {
		struct PyCodeObject *code = NULL;

		// 读取代码对象
		if (bpf_probe_read(&code, sizeof(code), &current_frame->f_code))
			break;
		if (!code)
			break;

		// 读取文件名
		struct _PyStr *filename = NULL;
		if (bpf_probe_read(&filename, sizeof(filename), &code->co_filename) == 0 &&
		    filename) {
			bpf_probe_read_str(trace->frames[i].filename,
					   sizeof(trace->frames[i].filename), filename->buf);
		}

		// 读取函数名
		struct _PyStr *funcname = NULL;
		if (bpf_probe_read(&funcname, sizeof(funcname), &code->co_name) == 0 && funcname) {
			bpf_probe_read_str(trace->frames[i].funcname,
					   sizeof(trace->frames[i].funcname), funcname->buf);
		}

		// 更新栈帧数量
		trace->num_frames++;

		// 获取上一个栈帧
		struct PyFrameObject *prev_frame = NULL;
		if (bpf_probe_read(&prev_frame, sizeof(prev_frame), &current_frame->f_back))
			break;

		current_frame = prev_frame;
		if (!current_frame)
			break;
	}

	// 发送数据到用户空间
	bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, trace, sizeof(*trace));

	return 0;
}

char LICENSE[] SEC("license") = "GPL";