#include "snoop/syscall_snoop.h"
#include "utils/Format.h"
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace NeuTracer {

SyscallSnoop::SyscallSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
    : env_(env), logger_(logger), profiler_(profiler) {
}

int SyscallSnoop::libbpf_print_fn(enum libbpf_print_level level, const char* format, va_list args) {
    if (level == LIBBPF_DEBUG) {
        return 0;
    }
    return vfprintf(stderr, format, args);
}


void SyscallSnoop::clear_call_id_map() {
    int map_fd = bpf_map__fd(skel_->maps.call_id_map);
    struct call_id_key_t key, next_key;
    while (bpf_map_get_next_key(map_fd, &key, &next_key) == 0) {
        bpf_map_delete_elem(map_fd, &next_key);
        key = next_key;
    }
}

int SyscallSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    SyscallSnoop* self = static_cast<SyscallSnoop*>(ctx);
    if (data_sz != sizeof(syscall_event)) {
        std::cerr << "Invalid SYSCALL event size: " << data_sz << std::endl;
        return -1;
    }
    return self->process_event(data, data_sz);
}

int SyscallSnoop::process_event(void *data, size_t data_sz) {
    const struct syscall_event* e = static_cast<const syscall_event*>(data);
    auto key = std::make_tuple(e->tgid, e->pid, e->syscall_id, e->call_id);
    bool should_sample = (e->timestamp - last_sample_time_) >= SAMPLE_INTERVAL_NS;
    
    if (e->type == EVENT_SYSCALL_ENTER) {
        syscall_enter_ts[key] = e->timestamp;
    } else if (e->type == EVENT_SYSCALL_EXIT) {
        auto it = syscall_enter_ts.find(key);
        if (it != syscall_enter_ts.end()) {
            uint64_t duration_ns = e->timestamp - it->second;
            syscall_enter_ts.erase(it);

            auto& stats = syscall_stats_[e->tgid];
            auto tmp = stats.avg_syscall_time[e->syscall_id];
            auto tmp_count = stats.syscall_count[e->syscall_id];
            if (tmp_count == 0) {
                tmp = duration_ns;
            } else {
                tmp = (tmp * tmp_count + duration_ns) / (tmp_count + 1);
            }
            stats.avg_syscall_time[e->syscall_id] = tmp;
            stats.syscall_count[e->syscall_id] += 1;
        }
    }
    // 格式化时间戳
    std::string time_ss = logger_.format_event_timestamp(e->timestamp); // 你可以用ts字段

    // 获取参数个数
    uint32_t arg_count = get_syscall_arg_count(e->syscall_id);

    // 统计
    auto& stats = syscall_stats_[e->tgid];
    stats.pid = e->pid;
    stats.tgid = e->tgid;
    if(!should_sample) return 0;
    // 记录事件
    if (e->type == EVENT_SYSCALL_ENTER) {
        profiler_.add_trace_data_syscall(
        "[SYSCALL] [{}] PID:{} TID:{} ({}) syscall:{} args:{} \n",
        time_ss,
        e->tgid,
        e->pid,
        e->comm,
        e->syscall_id,
        format_args(e->args, arg_count)
        );
    } else {
    profiler_.add_trace_data_syscall(
        "[SYSCALL] [{}] PID:{} TID:{} ({}) syscall:{} ret:{} call_nums:{} avg_time:{}us\n",
        time_ss,
        e->tgid,
        e->pid,
        e->comm,
        e->syscall_id,
        e->ret_val,
        stats.syscall_count[e->syscall_id],
        stats.avg_syscall_time[e->syscall_id] / 1000 // 转换为微秒
    );
}

    return 0;
}

std::string SyscallSnoop::format_args(const uint64_t* args, uint32_t arg_count) {
    std::ostringstream oss;
    oss << "[";
    for (uint32_t i = 0; i < arg_count; ++i) {
        if (i > 0) oss << ", ";
        oss << "0x" << std::hex << args[i];
    }
    oss << "]";
    return oss.str();
}

bool SyscallSnoop::attach_bpf() {
    libbpf_set_print(libbpf_print_fn);

    skel_ = syscall_snoop_bpf__open();
    if (!skel_) {
        logger_.error("Failed to open syscall BPF skeleton - err: {}", errno);
        return false;
    }

    int err = syscall_snoop_bpf__load(skel_);
    if (err) {
        logger_.error("Failed to load syscall BPF skeleton: {}", err);
        return false;
    }
    if (syscall_snoop_bpf__attach(skel_)) {
        logger_.error("Failed to attach [SYSCALL] BPF programs");
        return false;
    }
    rb_ = ring_buffer__new(
        bpf_map__fd(skel_->maps.events),
        handle_event,
        this,
        nullptr);

    if (!rb_) {
        logger_.error("Failed to create [SYSCALL] ring buffer");
        return false;
    }

    // 进程过滤
    uint32_t key = env_.pid;
    uint32_t value = env_.pid;
    if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
                        &value, sizeof(value), BPF_ANY)) {
        logger_.error("Failed to update syscall snoop_proc map: {}", strerror(errno));
        return false;
    }

    for(int i=0; i<100; i++){
        auto key = env_.syscall_id[i];
        if(key == END) break;
        auto value = key;
        bpf_map__update_elem(skel_->maps.traced_syscalls,&key,sizeof(key), &value, sizeof(value), BPF_ANY) ;
    }

    exiting_ = false;
    rb_thread_ = std::thread(&SyscallSnoop::ring_buffer_thread, this);

    logger_.info("系统调用模块启动");
    return true;
}

void SyscallSnoop::ring_buffer_thread() {
    auto startTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::seconds(60);  // 固定60秒
    auto last_cleanup = std::chrono::steady_clock::now();

    std::time_t ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logger_.info("Started syscall profiling at {}", std::ctime(&ttp));

    int err;
    while(!exiting_) {
        err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            break;
        }
        if (err < 0) {
            logger_.error("Error polling [SYSCALL] ring buffer: " + std::to_string(err));
            break;
        }
        if( std::chrono::steady_clock::now() - last_cleanup > std::chrono::hours(2) ) {
            clear_call_id_map();
            last_cleanup = std::chrono::steady_clock::now();
            logger_.info("Cleared call_id_map to free memory");
        }
    }
    ring_buffer__consume(rb_);

    ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
}

void SyscallSnoop::stop_trace() {
    exiting_ = true;
    if (rb_thread_.joinable()) {
        rb_thread_.join();
    }
    if (skel_) {
        syscall_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }
    logger_.info("Syscall monitoring thread stopped");
}

inline uint32_t SyscallSnoop::get_syscall_arg_count(uint32_t syscall_id) {
    static const std::map<uint32_t, uint32_t> syscall_arg_counts = {
        {0, 3},    // read(int fd, void *buf, size_t count)
        {1, 3},    // write(int fd, const void *buf, size_t count)
        {2, 3},    // open(const char *pathname, int flags, mode_t mode)
        {3, 1},    // close(int fd)
        {4, 2},    // stat(const char *pathname, struct stat *statbuf)
        {5, 2},    // fstat(int fd, struct stat *statbuf)
        {6, 2},    // lstat(const char *pathname, struct stat *statbuf)
        {9, 3},    // mmap(void *addr, size_t length, int prot)
        {10, 1},   // mprotect(void *addr, size_t len, int prot)
        {11, 2},   // munmap(void *addr, size_t length)
        {12, 1},   // brk(void *end_data_segment)
        {16, 2},   // ioctl(int fd, unsigned long request)
        {21, 2},   // access(const char *pathname, int mode)
        {32, 2},   // dup2(int oldfd, int newfd)
        {39, 0},   // getpid(void)
        {41, 3},   // socket(int domain, int type, int protocol)
        {42, 3},   // connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
        {43, 3},   // accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
        {44, 3},   // sendto(int sockfd, const void *buf, size_t len)
        {45, 3},   // recvfrom(int sockfd, void *buf, size_t len)
        {56, 2},   // clone(unsigned long flags, void *child_stack)
        {57, 1},   // fork(void)
        {59, 3},   // execve(const char *filename, char *const argv[], char *const envp[])
        {60, 1},   // exit(int status)
        {61, 1},   // wait4(pid_t pid, int *wstatus, int options, struct rusage *rusage)
        {63, 3},   // uname(struct utsname *buf)
        {89, 2},   // readlink(const char *pathname, char *buf, size_t bufsiz)
        {97, 3},   // getrlimit(int resource, struct rlimit *rlim)
        {102, 2},  // getuid(void)
        {104, 2},  // getgid(void)
        {110, 2},  // getppid(void)
        {202, 2},  // futex(int *uaddr, int futex_op)
        {217, 1},  // prctl(int option, unsigned long arg2, ...)
        {257, 4},  // openat(int dirfd, const char *pathname, int flags, mode_t mode)
        // ... 可继续补充
    };
    auto it = syscall_arg_counts.find(syscall_id);
    return (it != syscall_arg_counts.end()) ? it->second : 6;
}

} // namespace NeuTracer