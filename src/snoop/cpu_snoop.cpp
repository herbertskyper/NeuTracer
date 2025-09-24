#include "snoop/cpu_snoop.h"
#include "config.h"
#include "utils/Format.h"

#include <bpf/libbpf.h>
#include <bpf/bpf.h>


namespace NeuTracer {



int CPUsnoop::libbpf_print_fn(
    enum libbpf_print_level level,
    const char* format,
    va_list args) {
  if (level == LIBBPF_DEBUG ) {
    return 0;
  }
  return vfprintf(stderr, format, args);
};

int CPUsnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    CPUsnoop* self = static_cast<CPUsnoop*>(ctx);
    if (data_sz != sizeof(cpu_event)) {
        std::cerr << "Invalid CPU event size: " << data_sz << std::endl;
        return -1;
    }
    return self->process_event(ctx, data, data_sz);
    
}

int CPUsnoop::process_event(void *ctx, void *data, size_t data_sz) {
    lastActivityTime_ = std::chrono::steady_clock::now();
    auto *e = static_cast<cpu_event*>(data);

    bool should_trace = e->pid == env_.pid || e->ppid == env_.pid;
    bool should_sample = (e->timestamp - last_sample_time_ > SAMPLE_INTERVAL_NS);
    

    if (!should_trace) {
        // 如果不满足条件，则直接返回
        return 0;
    }
    
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    
    // 获取毫秒部分
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    // 格式化时间
    std::stringstream time_s;
    time_s << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
    std::string time_ss = time_s.str();



//    logger_.info(
//         "[{}] PPID:{} PID:{} {} ON:{} OFF:{} CPU:{:.2f}%\n",
//         time_ss.str(),
//         e->ppid,
//         e->pid,
//         e->comm,
//         e->oncpu_time,
//         e->offcpu_time,
//         e->utilization / 100.0
//     );

    // std::string time_ss = logger_.format_event_timestamp(e->timestamp);
    update_cpu_stats(e->pid, e, time_ss);
    
#ifdef REPORT_CPU
   event_count_++;
    if (event_count_ % CPU_stats_report_interval_ == 0) {
        report_cpu();
    }
#endif

    if(!should_sample) {
        // 如果不满足采样条件，则直接返回
        return 0;
    }
    last_sample_time_ = e->timestamp;

//     profiler_.add_trace_data_cpu(
//          "[CPU] [{}] PPID:{} PID:{} {} CPU_ID:{} ON:{} OFF:{} CPU:{:.2f}%\n",
//         time_ss,
//         e->ppid,
//         e->pid,
//         e->comm,
//         e->cpu_id,
//         e->oncpu_time,
//         e->offcpu_time,
//         e->utilization / 100.0
//    );
    auto& stats = cpu_stats_[e->pid];

    profiler_.add_trace_data_cpu(
         "[CPU] [{}] PPID:{} PID:{} {} CPU_ID:{} ON:{}ms OFF:{}ms CPU:{:.2f}%\n",
        time_ss,
        e->ppid,
        e->pid,
        e->comm,
        e->cpu_id,
        stats.total_oncpu_time / 1000,  // 转换为毫秒
        stats.total_offcpu_time / 1000,  // 转换为毫秒
        // stats.total_oncpu_time > 0 ? 
        //     (stats.total_oncpu_time * 100.0 / (stats.total_oncpu_time + stats.total_offcpu_time)) : 0.0
        e->utilization / 100.0
    );




    if(profiler_.isRPCEnabled()) {
        
        // 创建CPU数据结构体
        UprobeProfiler::CPUTraceItem cpu_data;
        cpu_data.timestamp = time_ss;
        cpu_data.pid = e->pid;
        cpu_data.ppid = e->ppid;
        cpu_data.comm = std::string(e->comm);
        cpu_data.cpu_id = e->cpu_id;
        cpu_data.oncpu_time = stats.total_oncpu_time / 1000;  // 转换为毫秒
        cpu_data.offcpu_time = stats.total_offcpu_time / 1000;  //
        cpu_data.utilization = e->utilization / 100.0;  // 转换为百分比
        // cpu_data.utilization = stats.total_oncpu_time > 0 ? 
        //     (stats.total_oncpu_time * 100.0 / (stats.total_oncpu_time + stats.total_offcpu_time)) : 0.0;
        
        // 添加其他指标
        cpu_data.migrations_count = stats.migrations_count;
        cpu_data.numa_migrations = stats.numa_migrations;
        cpu_data.hotspot_cpu = stats.hotspot_cpu;
        cpu_data.hotspot_percentage = stats.hotspot_percentage;

        profiler_.sendCPUTrace(cpu_data);
    }


    return 0;
}

//直接从proc文件系统获取CPU使用情况
    void CPUsnoop::record_stats(std::ostream& os, double cur_time, double period, uint32_t snoop_pid) {
        static bool first_time = true;
        static long clock_ticks = sysconf(_SC_CLK_TCK);
        static std::chrono::time_point<std::chrono::steady_clock> last_sample_time;
        
        // 控制采样间隔，避免过于频繁的采样
        auto current_time = std::chrono::steady_clock::now();
        if (!first_time) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_sample_time).count();
            if (elapsed < CPU_SMAPLE_TIME) {  // 至少间隔0.4秒才采样
                return;
            }
        }
        last_sample_time = current_time;
        
        std::vector<uint32_t> pid_list = bfs_get_procs(snoop_pid);
        std::map<uint32_t, std::map<std::string, uint64_t>> proc_cpu_usages;
        
        for (auto pid : pid_list) {
            if(pid != env_.pid) continue;
            if (first_time) {
                std::cout << "Monitoring PID: " << pid << std::endl;
            }
    
            std::string stat_file = "/proc/" + std::to_string(pid) + "/stat";
            std::ifstream stat(stat_file);
            if (!stat) continue;
    
            std::string line;
            std::getline(stat, line);
            std::istringstream iss(line);
    
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
    
            if (tokens.size() < 17) continue;
    
            // 将时间转换为毫秒，注意溢出检查
            uint64_t utime = 0, stime = 0, cutime = 0, cstime = 0;
            try {
                utime = std::stoull(tokens[13]) * 1000 / clock_ticks;
                stime = std::stoull(tokens[14]) * 1000 / clock_ticks;
                cutime = std::stoull(tokens[15]) * 1000 / clock_ticks;
                cstime = std::stoull(tokens[16]) * 1000 / clock_ticks;
            } catch (const std::exception& e) {
                logger_.error("Error parsing CPU times: {}", e.what());
                continue;
            }
    
            uint64_t total_time = utime + stime + cutime + cstime;
            if (total_time > 0) {  // 只记录有CPU使用的数据
                proc_cpu_usages[pid]["total_time"] = total_time;
            }
    
            if (!old_usage.empty() && old_usage.count(pid)) {
                int64_t time_diff = total_time - old_usage[pid]["total_time"];
                
                uint64_t period_ms = period * 1000;
                float utilization = (period_ms > 0) ? (time_diff * 100.0f) / period_ms : 0.0f;
    
                // 只输出有意义的数据
                if (time_diff > 0 || utilization > 0) {
                    os << std::fixed << std::setprecision(2)
                       << cur_time << ","
                       << std::setw(12) << pid << ","
                       << std::setw(20) << pid_to_comm(pid) << ","
                       << time_diff << ","
                       << (period_ms - time_diff) << ","
                       << utilization << std::endl;
                }
            }
        }
    
        old_usage = proc_cpu_usages;
        first_time = false;
    }

bool CPUsnoop::hasExceededProfilingLimit(std::chrono::seconds duration,
                                      const std::chrono::steady_clock::time_point &startTime) {
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
    return elapsedTime > duration;
}

bool CPUsnoop::attach_bpf(){

    libbpf_set_print(libbpf_print_fn);

    skel_= cpu_snoop_bpf__open();
    if(!skel_) {
        logger_.error("Failed to open and load CPU BPF skeleton - err: {}",errno);
        return false;
    }

    // bpf_map__set_max_entries(
    //     skel_->maps.cpu_events, env_.rb_count > 0 ? env_.rb_count : RINGBUF_MAX_ENTRIES);

    int err = cpu_snoop_bpf__load(skel_);
    if (err)
    {
        logger_.error("Failed to load and verify CPU BPF skeleton");
        return -1;
    }

    skel_->links.sched_switch = bpf_program__attach(skel_->progs.sched_switch);
    if (!skel_->links.sched_switch) {
        fprintf(stderr, "Failed to attach sched_switch: %s\n", strerror(errno));
    }
    links_.push_back(skel_->links.sched_switch);

    skel_->links.sched_process_exit = bpf_program__attach(skel_->progs.sched_process_exit);
    if (!skel_->links.sched_process_exit) {
        fprintf(stderr, "Failed to attach sched_process_exit: %s\n", strerror(errno));
    }
    links_.push_back(skel_->links.sched_process_exit);

    rb_ = ring_buffer__new(bpf_map__fd(skel_->maps.cpu_events), handle_event, (void*)this, NULL);
    if (!rb_) {
        logger_.error("Failed to create cpu ring buffer");
        return false;
    }

    exiting_ = false;
    rb_thread_ = std::thread(&CPUsnoop::ring_buffer_thread, this);

    uint32_t key = env_.pid;
    uint32_t value = env_.pid;

    cpu_stats_.emplace(env_.pid, CPUStats());
    
    if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
                        &value, sizeof(value), BPF_ANY)) {
        logger_.error("Failed to update cpu snoop_proc map: {}", strerror(errno));
        return false;
    }
    logger_.info("Tracking CPU events for PID: {}", env_.pid);
    // printf("Tracking PID: %u\n",  env_.pid);


    return true;
    
    
}

void CPUsnoop::ring_buffer_thread() {
    logger_.info("Starting cpu ring buffer polling thread\n");
    // auto startTime = std::chrono::steady_clock::now();
    // auto duration = std::chrono::seconds(env_.duration_sec);

    // std::time_t ttp =
    //     std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // // logger_.info("Started profiling at {}", std::ctime(&ttp));
    
    // while (!exiting_ && !hasExceededProfilingLimit(duration,startTime)) {
    //     // 轮询前记录当前时间
    //     auto pollStartTime = std::chrono::steady_clock::now();
        
    //     // 轮询环形缓冲区
    //     int err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        
    //     /* Ctrl-C will cause -EINTR */
    //     if (err == -EINTR) {
    //         err = 0;
    //         break;
    //     }
    //     if (err < 0) {
    //         logger_.error("Error polling CPU ring buffer: {}", err);
    //         continue;
    //     }
        
    //     // 检查是否收到了数据
    //     if (err > 0) {
    //         // 收到数据，更新最后一次活动时间
    //         lastActivityTime_ = std::chrono::steady_clock::now();
    //     } else {
    //         // 未收到数据，检查是否超过了空闲超时时间
    //         auto currentTime = std::chrono::steady_clock::now();
    //         auto idleTime = std::chrono::duration_cast<std::chrono::seconds>(
    //             currentTime - lastActivityTime_).count();
            
    //         // 如果空闲时间超过阈值，退出循环
    //         if (idleTimeoutSec_ > 0 && idleTime >= idleTimeoutSec_) {
    //             logger_.info("No activity for {} seconds, stopping CPU profiling", idleTime);
    //             break;
    //         }
    //     }
    // }
    while(!exiting_) {
        int err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            err = 0;
            break;
        }
        if (err < 0) {
            logger_.error("Error polling CPU ring buffer: {}", err);
            continue;
        }
    }
    ring_buffer__consume(rb_);

    // ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // logger_.info("Stopped CPU profiling at {}", std::ctime(&ttp));
    // logger_.info("CPU Ring buffer polling thread stopped");
}

void CPUsnoop::stop_trace() {
    // 1. 首先标记退出并等待线程结束
    exiting_ = true;
    if (rb_thread_.joinable()) {
        rb_thread_.join();
    }
    logger_.info("CPU Ring buffer polling thread stopped");

    // 2. 清理 ring buffer
    if (rb_) {
        ring_buffer__free(rb_);
        rb_ = nullptr;
    }
    logger_.info("CPU Ring buffer polling stopped");

    links_.clear();

    // 3. 最后清理 BPF skeleton
    if (skel_) {
        cpu_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }
    logger_.info("CPU BPF skeleton destroyed");
}


void CPUsnoop::update_cpu_stats(uint32_t pid, const cpu_event* e, std::string timestamp) {
    auto& stats = cpu_stats_[pid];
    auto now = std::chrono::steady_clock::now();
    
    // 初始化窗口开始时间
    if (stats.window_start.time_since_epoch().count() == 0) {
        stats.window_start = now;
    }
    
    // 检查是否需要重置2小时窗口
    if (now - stats.window_start >= CLEAN_TIME_MIN * std::chrono::minutes(1)) {
        double avg_utilization = (stats.call_count > 0) ? stats.avg_utilization : 0.0;
        
        logger_.info("[CPU窗口] PID {} : {}分钟窗口重置 - 调用:{} 迁移:{} 平均利用率:{:.2f}% 热点CPU:{}",
                    pid, CLEAN_TIME_MIN,
                    stats.call_count, stats.migrations_count,
                    avg_utilization, stats.hotspot_cpu);
        
        // 重置窗口统计
        stats.window_start = now;
        stats.total_oncpu_time = 0;
        stats.total_offcpu_time = 0;
        stats.call_count = 0;
        stats.avg_utilization = 0.0;
        stats.migrations_count = 0;
        stats.numa_migrations = 0;
        stats.hotspot_cpu = 0;
        stats.hotspot_percentage = 0.0;
        
        // 清空容器
        stats.utilization_samples.clear();
        stats.cpu_distribution.clear();
        stats.migrations.clear();
    }
    

    
    // 1. 基础数据更新
    stats.total_oncpu_time += e->oncpu_time;
    stats.total_offcpu_time += e->offcpu_time;
    stats.call_count++;
    
    // 2. 计算利用率
    stats.avg_utilization = stats.total_oncpu_time > 0 ? 
        (stats.total_oncpu_time * 100.0 / (stats.total_oncpu_time + stats.total_offcpu_time)) : 0.0;
    
    // 3. 热点CPU追踪
    stats.cpu_distribution[e->cpu_id] += e->oncpu_time;
    update_hotspot_cpu(pid, timestamp);
    bool should_warn = (e->timestamp - last_warning_time_ > WARN_INTERVAL);

    if (stats.hotspot_percentage > CPU_HOTSPOT_THRESHOLD && stats.total_oncpu_time > 100000000000 && should_warn) { // 100秒以上的CPU时间) 
        logger_.warn("[CPU异常] PID {} ({}): CPU热点过高 - {:.1f}% 时间在核心{}上运行",
                    pid, pid_to_comm(pid), stats.hotspot_percentage * 100, stats.hotspot_cpu);
    }
    
    // 检测CPU迁移
    if (last_cpu_.find(pid) != last_cpu_.end() && 
        last_cpu_[pid] != e->cpu_id) {
        // 记录CPU迁移统计
        stats.migrations_count++;
        
        // 记录迁移事件
        stats.migrations.push_back({
            timestamp,
            last_cpu_[pid],
            e->cpu_id
        });
        
        // 检测是否是跨NUMA迁移
        if (is_cross_numa(last_cpu_[pid], e->cpu_id)) {
            stats.numa_migrations++;
            
            if(stats.numa_migrations >= CPU_NUMA_THRESHOLD && should_warn) {
                    logger_.warn("[CPU异常] PID {} ({}): 跨NUMA节点迁移:{}次",
                                    pid, pid_to_comm(pid), stats.numa_migrations);
                        
                }
        }
    }
    last_cpu_[pid] = e->cpu_id;
    
    // 追踪利用率样本
    if (stats.utilization_samples.size() >= CPU_max_history_samples_) {
        stats.utilization_samples.pop_front();
    }
    stats.utilization_samples.push_back(e->utilization);
    
    // 检测利用率突增
    if (stats.utilization_samples.size() >= 10) {
        double avg = 0.0;
        for (size_t i = stats.utilization_samples.size() - 10; i < stats.utilization_samples.size() - 1; i++) {
            avg += stats.utilization_samples[i];
        }
        avg /= 10.0; // 计算最近10个样本的平均值
        
        if (e->utilization > avg * 2 && e->utilization / 100 > CPU_UTIL_THREHOLD && should_warn) { // 高于平均值2倍且超过50%
            logger_.warn("[CPU异常] PID {} ({}): 检测到突发型CPU使用 - 当前利用率{:.1f}% (平均{:.1f}%)",
                        pid, pid_to_comm(pid), e->utilization / 100.0, avg / 100.0);
            }
        }
    
    
    // 检测资源抢占(thrashing)
    // bool is_thrashing = detect_resource_thrashing(pid, e);
    // if (is_thrashing) {
    //         logger_.warn("[CPU异常] PID {} ({}): 检测到CPU资源争用 - 可能与其他进程竞争",
    //                     pid, pid_to_comm(pid));
    // }
    
    
    // 检测上下文切换率异常
    double switch_rate = (double)stats.migrations_count / stats.call_count;
    
    if (switch_rate >  CPU_CONTEXT_CHANGE && should_warn) { 
            logger_.warn("[CPU异常] PID {} ({}): 上下文切换频率异常高 - {:.1f}% ({}次/{}次采样)",
                        pid, pid_to_comm(pid), switch_rate * 100, 
                        stats.migrations_count, stats.call_count);
    }
    
    // 检测CPU利用率过低（可能的饥饿状态）
    if (stats.call_count > 20 && stats.avg_utilization < CPU_UTIL_LOW_THRESHOLD && stats.total_oncpu_time > 500000 && should_warn) {
            logger_.warn("[CPU异常] PID {} ({}): CPU利用率异常低 - 平均{:.2f}% (可能处于饥饿状态)",
                        pid, pid_to_comm(pid), stats.avg_utilization);
    }
    if(should_warn) {
        last_warning_time_ = e->timestamp;  // 更新最后一次警告时间
    }
}

// 检测资源抢占
// bool CPUsnoop::detect_resource_thrashing(uint32_t pid, const cpu_event* e) {
//     auto& stats = cpu_stats_[pid];
    
//     // 至少需要10个样本
//     if (stats.utilization_samples.size() < 10) {
//         return false;
//     }
    
//     // 检查最近几个样本是否有频繁的高低利用率交替
//     int switches = 0;
//     bool was_high = stats.utilization_samples[stats.utilization_samples.size() - 1] > 5000;
    
//     for (int i = stats.utilization_samples.size() - 2; i >= 0 && i >= (int)stats.utilization_samples.size() - 10; i--) {
//         bool is_high = stats.utilization_samples[i] > 5000;
//         if (is_high != was_high) {
//             switches++;
//             was_high = is_high;
//         }
//     }
    
//     // 频繁在高低CPU使用率间切换可能表示资源抢占
//     return switches >= 4;  // 在10个样本中有4次或更多切换
// }

void CPUsnoop::update_hotspot_cpu(uint32_t pid, std::string timestamp) {
    auto& stats = cpu_stats_[pid];
    uint32_t hotspot = 0;
    uint64_t max_time = 0;
    uint64_t total_time = 0;
    
    for (const auto& [cpu_id, time] : stats.cpu_distribution) {
        total_time += time;
        if (total_time > max_time) {
            max_time = time;
            hotspot = cpu_id;
        }
    }
    
    stats.hotspot_cpu = hotspot;
    stats.hotspot_percentage = (total_time > 0) ? 
        (double)max_time / total_time : 0;
    
}

// 判断是否跨NUMA节点迁移
bool CPUsnoop::is_cross_numa(uint32_t cpu1, uint32_t cpu2) {
    //lscpu | grep NUMA
    // NUMA node0 CPU(s):                    0-31
    // NUMA node1 CPU(s):                    32-63
    int numa1 = cpu1 / 32;
    int numa2 = cpu2 / 32;
    return numa1 != numa2;
}

void CPUsnoop::report_cpu() {
    for (const auto& [pid, stats] : cpu_stats_) {
        logger_.info("============ CPU报告 ============");
        logger_.info("[CPU]  调用次数: {}", stats.call_count);
        logger_.info("[CPU]  总ON时间: {}ms, 总OFF时间: {}ms",
            stats.total_oncpu_time / 1000, stats.total_offcpu_time / 1000);
        logger_.info("[CPU]  平均利用率: {:.2f}%", stats.avg_utilization);
        logger_.info("[CPU]  迁移次数: {}, NUMA迁移: {}", stats.migrations_count, stats.numa_migrations);
        logger_.info("[CPU]  热点CPU: {} (占比:{:.2f}%)", stats.hotspot_cpu, stats.hotspot_percentage * 100);
        if (stats.migrations.size() > 0) {
            logger_.info("[CPU]  迁移事件数: {}", stats.migrations.size());
        }
    }
    logger_.info("===================================");
}

} // namespace NeuTracer