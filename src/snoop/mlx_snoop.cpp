#include "snoop/mlx_snoop.h"
#include "utils/Format.h"
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace NeuTracer {

MLXSnoop::MLXSnoop(const myenv &env, Logger &logger, UprobeProfiler &profiler)
    : env_(env), logger_(logger), profiler_(profiler) {
}

int MLXSnoop::libbpf_print_fn(enum libbpf_print_level level, const char* format, va_list args) {
    if (level == LIBBPF_DEBUG) {
        return 0;
    }
    return vfprintf(stderr, format, args);
}

int MLXSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    MLXSnoop* self = static_cast<MLXSnoop*>(ctx);
    if (data_sz != sizeof(mlx_event)) {
        std::cerr << "Invalid MLX event size: " << data_sz << std::endl;
        return -1;
    }
    return self->process_event(data, data_sz);
}

int MLXSnoop::process_event(void *data, size_t data_sz) {
    const struct mlx_event* e = static_cast<const mlx_event*>(data);
    bool should_sample = (e->timestamp - last_sample_time_) >= SAMPLE_INTERVAL_NS;
    
    // 更新统计
    if (e->type == EVENT_MLX_EXIT) {
        auto& stats = mlx_stats_[e->tgid];
        stats.pid = e->pid;
        stats.tgid = e->tgid;
        
        // 更新调用次数
        stats.call_count[e->func_id]++;
    }
    
    // 不需要每次事件都打印，可以根据采样间隔处理
    if (!should_sample) return 0;
    
    // 格式化时间戳
    std::string time_ss = logger_.format_event_timestamp(e->timestamp);
    
    // // 记录事件
    // if (e->type == EVENT_MLX_ENTER) {
    //     std::string args_str;
    //     switch (e->func_id) {
    //         case MLX_REG_DM_MR:
    //             args_str = profiler_.format_string("dm=0x{:x}, attr=0x{:x}", e->arg1, e->arg2);
    //             break;
    //         case MLX_REG_USER_MR:
    //             args_str = profiler_.format_string("pd=0x{:x}, attr=0x{:x}", e->arg1, e->arg2);
    //             break;
    //         case MLX_REG_USER_MR_DMABUF:
    //             args_str = profiler_.format_string("pd=0x{:x}, attr=0x{:x}, dma_buf=0x{:x}", 
    //                        e->arg1, e->arg2, e->arg3);
    //             break;
    //         case MLX_ALLOC_PD:
    //             args_str = profiler_.format_string("pd=0x{:x}", e->arg1);
    //             break;
    //         case MLX_ALLOC_MR:
    //             args_str = profiler_.format_string("pd=0x{:x}, mr_type={}, max_sg={}", 
    //                        e->arg1, e->arg2, e->arg3);
    //             break;
    //         case MLX_CREATE_CQ:
    //         case MLX_CREATE_QP:
    //         case MLX_CREATE_SRQ:
    //             args_str = profiler_.format_string("obj=0x{:x}, attr=0x{:x}", e->arg1, e->arg2);
    //             break;
    //         default:
    //             args_str = profiler_.format_string("arg1=0x{:x}, arg2=0x{:x}, arg3=0x{:x}", 
    //                        e->arg1, e->arg2, e->arg3);`
    //             break;
    //     }
        
    //     profiler_.add_trace_data_mlx(profiler_.format_string(
    //         "[MLX5] [{}] PID:{} TID:{} ({}) {}() {}\n",
    //         time_ss,
    //         e->tgid,
    //         e->pid,
    //         e->comm,
    //         get_func_name(static_cast<mlx_func_id>(e->func_id)),
    //         args_str
    //     ));
    // } else {
    //     auto& stats = mlx_stats_[e->tgid];
    //     std::string ret_str;
        
    //     // 处理不同类型的返回值
    //     if (e->func_id == MLX_REG_DM_MR || e->func_id == MLX_REG_USER_MR || 
    //         e->func_id == MLX_REG_USER_MR_DMABUF || e->func_id == MLX_ALLOC_MR) {
    //         // 这些函数返回指针
    //         ret_str = profiler_.format_string("mr=0x{:x}", e->ret_val);
    //     } else {
    //         // 其他返回状态码
    //         ret_str = profiler_.format_string("ret={}", static_cast<int>(e->ret_val));
    //     }
        
    //     profiler_.add_trace_data_mlx(profiler_.format_string(
    //         "[MLX5] [{}] PID:{} TID:{} ({}) {}() {} [calls={}, desc={}]\n",
    //         time_ss,
    //         e->tgid,
    //         e->pid,
    //         e->comm,
    //         get_func_name(static_cast<mlx_func_id>(e->func_id)),
    //         ret_str,
    //         stats.call_count[e->func_id],
    //         get_func_desc(static_cast<mlx_func_id>(e->func_id))
    //     ));
    // }
    
    last_sample_time_ = e->timestamp;
    return 0;
}

std::string MLXSnoop::get_func_name(mlx_func_id func_id) {
    switch (func_id) {
        case MLX_REG_DM_MR: return "mlx5_ib_reg_dm_mr";
        case MLX_REG_USER_MR: return "mlx5_ib_reg_user_mr";
        case MLX_REG_USER_MR_DMABUF: return "mlx5_ib_reg_user_mr_dmabuf";
        case MLX_ALLOC_PD: return "mlx5_ib_alloc_pd";
        case MLX_ALLOC_MR: return "mlx5_ib_alloc_mr";
        case MLX_CREATE_CQ: return "mlx5_ib_create_cq";
        case MLX_CREATE_QP: return "mlx5_ib_create_qp";
        case MLX_CREATE_SRQ: return "mlx5_ib_create_srq";
        default: return "unknown_mlx_func";
    }
}

std::string MLXSnoop::get_func_desc(mlx_func_id func_id) {
    switch (func_id) {
        case MLX_REG_DM_MR: return "Register Device Memory MR";
        case MLX_REG_USER_MR: return "Register User Memory Region";
        case MLX_REG_USER_MR_DMABUF: return "Register User MR with DMABUF";
        case MLX_ALLOC_PD: return "Allocate Protection Domain";
        case MLX_ALLOC_MR: return "Allocate Memory Region";
        case MLX_CREATE_CQ: return "Create Completion Queue";
        case MLX_CREATE_QP: return "Create Queue Pair";
        case MLX_CREATE_SRQ: return "Create Shared Receive Queue";
        default: return "Unknown MLX5 function";
    }
}

bool MLXSnoop::attach_bpf() {
    // libbpf_set_print(libbpf_print_fn);

    // skel_ = mlx_snoop_bpf__open();
    // if (!skel_) {
    //     logger_.error("Failed to open MLX5 BPF skeleton - err: {}", errno);
    //     return false;
    // }

    // int err = mlx_snoop_bpf__load(skel_);
    // if (err) {
    //     logger_.error("Failed to load MLX5 BPF skeleton: {}", err);
    //     return false;
    // }
    
    // // 使用 bpf_program__attach_kprobe_opts 为每个函数单独 attach，并指定 mlx5_ib 模块
    // struct bpf_kprobe_opts opts = {};
    // opts.retprobe = false;
    // opts.module = "mlx5_ib";
    // struct bpf_link *link;
    
    // // mlx5_ib_reg_dm_mr
    // opts.func_name = "mlx5_ib_reg_dm_mr";
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_reg_dm_mr_enter, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kprobe to mlx5_ib_reg_dm_mr");
    // } else {
    //     links_.push_back(link);
    // }
    
    // opts.retprobe = true;
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_reg_dm_mr_exit, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kretprobe to mlx5_ib_reg_dm_mr");
    // } else {
    //     links_.push_back(link);
    // }
    
    // // mlx5_ib_reg_user_mr
    // opts.retprobe = false;
    // opts.func_name = "mlx5_ib_reg_user_mr";
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_reg_user_mr_enter, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kprobe to mlx5_ib_reg_user_mr");
    // } else {
    //     links_.push_back(link);
    // }
    
    // opts.retprobe = true;
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_reg_user_mr_exit, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kretprobe to mlx5_ib_reg_user_mr");
    // } else {
    //     links_.push_back(link);
    // }
    
    // // mlx5_ib_reg_user_mr_dmabuf
    // opts.retprobe = false;
    // opts.func_name = "mlx5_ib_reg_user_mr_dmabuf";
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_reg_user_mr_dmabuf_enter, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kprobe to mlx5_ib_reg_user_mr_dmabuf");
    // } else {
    //     links_.push_back(link);
    // }
    
    // opts.retprobe = true;
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_reg_user_mr_dmabuf_exit, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kretprobe to mlx5_ib_reg_user_mr_dmabuf");
    // } else {
    //     links_.push_back(link);
    // }
    
    // // mlx5_ib_alloc_pd
    // opts.retprobe = false;
    // opts.func_name = "mlx5_ib_alloc_pd";
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_alloc_pd_enter, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kprobe to mlx5_ib_alloc_pd");
    // } else {
    //     links_.push_back(link);
    // }
    
    // opts.retprobe = true;
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_alloc_pd_exit, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kretprobe to mlx5_ib_alloc_pd");
    // } else {
    //     links_.push_back(link);
    // }
    
    // // mlx5_ib_alloc_mr
    // opts.retprobe = false;
    // opts.func_name = "mlx5_ib_alloc_mr";
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_alloc_mr_enter, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kprobe to mlx5_ib_alloc_mr");
    // } else {
    //     links_.push_back(link);
    // }
    
    // opts.retprobe = true;
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_alloc_mr_exit, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kretprobe to mlx5_ib_alloc_mr");
    // } else {
    //     links_.push_back(link);
    // }
    
    // // mlx5_ib_create_cq
    // opts.retprobe = false;
    // opts.func_name = "mlx5_ib_create_cq";
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_create_cq_enter, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kprobe to mlx5_ib_create_cq");
    // } else {
    //     links_.push_back(link);
    // }
    
    // opts.retprobe = true;
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_create_cq_exit, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kretprobe to mlx5_ib_create_cq");
    // } else {
    //     links_.push_back(link);
    // }
    
    // // mlx5_ib_create_qp
    // opts.retprobe = false;
    // opts.func_name = "mlx5_ib_create_qp";
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_create_qp_enter, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kprobe to mlx5_ib_create_qp");
    // } else {
    //     links_.push_back(link);
    // }
    
    // opts.retprobe = true;
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_create_qp_exit, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kretprobe to mlx5_ib_create_qp");
    // } else {
    //     links_.push_back(link);
    // }
    
    // // mlx5_ib_create_srq
    // opts.retprobe = false;
    // opts.func_name = "mlx5_ib_create_srq";
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_create_srq_enter, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kprobe to mlx5_ib_create_srq");
    // } else {
    //     links_.push_back(link);
    // }
    
    // opts.retprobe = true;
    // link = bpf_program__attach_kprobe_opts(skel_->progs.mlx_create_srq_exit, &opts);
    // if (!link) {
    //     logger_.error("Failed to attach kretprobe to mlx5_ib_create_srq");
    // } else {
    //     links_.push_back(link);
    // }

    // rb_ = ring_buffer__new(
    //     bpf_map__fd(skel_->maps.events),
    //     handle_event,
    //     this,
    //     nullptr);

    // if (!rb_) {
    //     logger_.error("Failed to create MLX5 ring buffer");
    //     return false;
    // }

    // // 进程过滤
    // uint32_t key = env_.pid;
    // uint32_t value = env_.pid;
    // if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
    //                     &value, sizeof(value), BPF_ANY)) {
    //     logger_.error("Failed to update MLX5 snoop_proc map: {}", strerror(errno));
    //     return false;
    // }

    // exiting_ = false;
    // rb_thread_ = std::thread(&MLXSnoop::ring_buffer_thread, this);

    // logger_.info("MLX5 InfiniBand 模块启动");
    return true;
}

void MLXSnoop::ring_buffer_thread() {
    std::time_t ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logger_.info("Started MLX5 profiling at {}", std::ctime(&ttp));

    int err;
    while(!exiting_) {
        err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            break;
        }
        if (err < 0) {
            logger_.error("Error polling MLX5 ring buffer: " + std::to_string(err));
            break;
        }
        
        // 更新最后活动时间
        lastActivityTime_ = std::chrono::steady_clock::now();
    }
    
    ring_buffer__consume(rb_);

    ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logger_.info("Stopped MLX5 profiling at {}", std::ctime(&ttp));
}

void MLXSnoop::stop_trace() {
    exiting_ = true;
    if (rb_thread_.joinable()) {
        rb_thread_.join();
    }
    
    // 清理资源
    for (auto link : links_) {
        bpf_link__destroy(link);
    }
    links_.clear();
    
    if (rb_) {
        ring_buffer__free(rb_);
        rb_ = nullptr;
    }
    
    if (skel_) {
        mlx_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }
    
    logger_.info("MLX5 monitoring thread stopped");
}

} // namespace NeuTracer