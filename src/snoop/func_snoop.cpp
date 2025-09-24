#include "snoop/func_snoop.h"
#define _PRINTF_FUNCTION_

namespace NeuTracer {

using json = nlohmann::json;

// FuncSnoop实现
FuncSnoop::FuncSnoop(const json &uprobe_cfg, const myenv &env, Logger &logger, UprobeProfiler &profiler) 
    : uprobe_cfg_(uprobe_cfg), env_(env), next_func_id_(1), skel_(nullptr),logger_(logger),rb_(nullptr), exiting_(false),profiler_(profiler) {
    init_bpf();
}

FuncSnoop::~FuncSnoop() {
    // stop_trace();
    // // 清理链接和骨架
    // for (auto link : links_) {
    //     bpf_link__destroy(link);
    // }
    // links_.clear();
    
    // if (skel_) {
    //     func_snoop_bpf__destroy(skel_);
    //     skel_ = nullptr;
    // }
}

void FuncSnoop::init_bpf() {
    // 打开BPF骨架
    skel_ = func_snoop_bpf__open();
    if (!skel_) {
      
        throw std::runtime_error("Failed to open BPF skeleton");
    }
    // 加载BPF程序
    if (func_snoop_bpf__load(skel_)) {
        throw std::runtime_error("Failed to load BPF skeleton");
    }
    // logger_.info("Loaded FUNC BPF skeleton");
}

// Ring buffer 事件处理回调
int FuncSnoop::handle_event(void *ctx, void *data, size_t data_sz) {
    FuncSnoop* self = static_cast<FuncSnoop*>(ctx);
    //printf("enter handle_event\n");
    if(data_sz != sizeof(struct func_trace_event)) {
        printf("Error: Invalid event size: %zu\n", data_sz);
        return 0;
    }
    return self->process_event(data, data_sz);
}

int FuncSnoop::process_event(void *data, size_t data_sz) {
    struct func_trace_event *e = static_cast<struct func_trace_event*>(data);
    

    // 检查数据大小
    if (data_sz < sizeof(struct func_trace_event)) {
        printf("Error: Invalid event size: %zu\n", data_sz);
        return 0;
    }
    // bool should_log = (log_sample_rate_ >= 1.0) || 
    //                   (dist_(rng_) <= log_sample_rate_);
    bool should_trace = e->pid == env_.pid || e->tgid == env_.pid;
    if (!should_trace) {
        // 如果不满足条件，则直接返回
        return 0;
    }
    bool should_sample = false;
    if((e->timestamp - last_sample_time_) >= SAMPLE_INTERVAL_NS * 10) {
        should_sample = true;
    } 
    
    // 获取当前时间，包含毫秒
    char time_str[32];
    char time_with_ms[64];  // 增加缓冲区大小以避免截断警告
    
    // 使用 chrono 获取毫秒级精度的时间
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    

  // 格式化时间
  std::stringstream time_ss;
  time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
          << '.' << std::setfill('0') << std::setw(3) << ms.count();
  std::string timestamp = time_ss.str();
    
    // 根据事件类型输出信息
    // if (event->type == 0) { // EVENT_ENTRY
    //     logger_.info("{:<12} {:<5} {:<16} [cookie: {}]", 
    //             time_with_ms, "ENTER", event->name, event->cookie);
    // } else if (event->type == 1) { // EVENT_EXIT
    //     logger_.info("{:<12} {:<5} {:<16} [cookie: {}]", 
    //             time_with_ms, "EXIT", event->name, event->cookie);
    // }
    std::string func_key = e->name;
    std::string event_type_str = (e->type == 0) ? "ENTER" : "EXIT";
    // 更新函数统计信息
    if (e->type == 0) { //EVENT_ENTRY
        // 入口点：记录开始时间并更新调用计数
        auto& stats = func_stats_[func_key];
        stats.call_count++;
        stats.active_calls++;
            
        // 记录函数首次调用时间
        if (stats.first_call_time.empty()) {
            stats.first_call_time = time_str;
        }

        // 记录开始时间用于计算持续时间
        auto& timing = func_timing_[std::make_tuple(e->tgid, e->pid, e->cookie)];
        timing.start_time = now;
        timing.func_name = func_key;
        
    }
    else { // EVENT_EXIT
        // 查找对应的函数入口记录
        auto key = std::make_tuple(e->tgid, e->pid, e->cookie);
        auto timing_it = func_timing_.find(key);
        
        if (timing_it != func_timing_.end()) {
            // 计算函数执行时间
            auto& timing = timing_it->second;
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                now - timing.start_time).count();
            
            // 更新函数统计信息
            auto& stats = func_stats_[timing.func_name];
            stats.active_calls--;
            stats.total_duration_us += duration;
            stats.last_call_time = timestamp;
            
            // 更新最长/最短调用时间
            if (duration > stats.max_duration_us || stats.max_duration_us == 0) {
                stats.max_duration_us = duration;
            }
            if (duration < stats.min_duration_us || stats.min_duration_us == 0) {
                stats.min_duration_us = duration;
            }
            
            // 计算平均调用时间
            stats.avg_duration_us = stats.total_duration_us * 1.0 / stats.call_count;
            
            // 检测慢调用
            if (duration > slow_call_threshold_us_) {
                stats.slow_call_count++;
                auto temp = FuncSnoop::FunctionStats::SlowCall{
                    timestamp,
                    static_cast<unsigned long>(duration),
                    e->pid,
                    e->tgid
                };
                // 记录慢调用信息
                if (stats.slow_calls.size() < max_slow_calls_to_record_) {
                    stats.slow_calls.push_back(temp);
                }
                
                //记录慢调用警告
                logger_.warn("[FUNC_SLOW]  {} | PID: {}, TGID: {} | Duration: {}",
                          timing.func_name, 
                          e->pid, e->tgid, logger_.format_duration(duration));
            }
            
            // 移除计时记录
            func_timing_.erase(timing_it);
        }

        
    }

    event_count_++;

#ifdef _PRINTF_FUNCTION_
    if (event_count_ % stats_report_interval_ == 0) {
        report_function_stats();
    }
#endif
    if(should_sample) {
        last_sample_time_ = e->timestamp;  // 更新最后一次采样时间
    }else {
        // 如果不满足采样条件，则直接返回
        // logger_.info("Skipping event due to sampling condition not met");
        return 0;
    }

    std::ostringstream args_ss;
    args_ss << "[";
    for (int i = 0; i < 6; ++i) {
        if (i > 0) args_ss << ", ";
        args_ss << "0x" << std::hex << e->args[i];
    }
    args_ss << "]";

    // 传递带毫秒的时间字符串
    if(e->type == 0) { // EVENT_ENTRY
        profiler_.add_trace_data(
        "[FUNC] [{}] PID:{} TGID:{} TYPE:{} COOKIE:{} NAME:{} COUNT:{} AVG_TIME:{} ARGS:{}\n",
        time_ss.str(),
        e->pid,
        e->tgid,
        e->type == 0 ? "ENTER" : "EXIT",
        e->cookie,
        e->name,
        func_stats_[func_key].call_count,
        logger_.format_duration(func_stats_[func_key].avg_duration_us),
        args_ss.str() 
    );
    } else { // EVENT_EXIT
        profiler_.add_trace_data(
        "[FUNC] [{}] PID:{} TGID:{} TYPE:{} COOKIE:{} NAME:{} COUNT:{} AVG_TIME:{} RET:{}\n",
        time_ss.str(),
        e->pid,
        e->tgid,
        e->type == 0 ? "ENTER" : "EXIT",
        e->cookie,
        e->name,
        func_stats_[func_key].call_count,
        logger_.format_duration(func_stats_[func_key].avg_duration_us),
        std::to_string(e->ret_val) 
    );
    }

  if(profiler_.isRPCEnabled()){
    profiler_.sendTrace(profiler_.get_timestamp(), e->name, e->pid, e->tgid, e->type == 0 ? "ENTER" : "EXIT", e->cookie,
        func_stats_[func_key].call_count, func_stats_[func_key].avg_duration_us);
  }

    return 0;
}
// Ring buffer 处理线程
void FuncSnoop::ring_buffer_thread() {
    logger_.info("Starting func ring buffer polling thread");
    
    int err;
    while (!exiting_) {
        err = ring_buffer__poll(rb_, 100 /* timeout, ms */);
        if (err == -EINTR) {
            break;
        }
        if (err < 0) {
            logger_.error("Error polling func ring buffer: " + std::to_string(err));
            break;
        }
    }
    // 处理剩余事件
    ring_buffer__consume(rb_);
    logger_.info("Func Ring buffer polling thread stopped");
}

// 启动 ring buffer 处理
bool FuncSnoop::start_trace() {
    if (!skel_) {
        logger_.error("Func BPF skeleton not initialized");
        return false;
    }
    
    // 确保不会重复启动
    if (rb_) {
        logger_.error("Func Ring buffer already started");
        return true;
    }
    
    // 创建 ring buffer
    rb_ = ring_buffer__new(bpf_map__fd(skel_->maps.events), handle_event, this, NULL);
    if (!rb_) {
        logger_.error("Failed to create Func ring buffer");
        return false;
    }

    uint32_t key = env_.pid;
    uint32_t value = env_.pid;
    
    if (bpf_map__update_elem(skel_->maps.snoop_proc, &key, sizeof(key), 
                        &value, sizeof(value), BPF_ANY)) {
        logger_.error("Failed to update func snoop_proc map: {}", strerror(errno));
        return false;
    }

    logger_.info("Tracking func for PID: {}", env_.pid);
    
    // 设置退出标志并启动线程
    exiting_ = false;
    rb_thread_ = std::thread(&FuncSnoop::ring_buffer_thread, this);
        
    return true;
}

// 停止 ring buffer 处理
void FuncSnoop::stop_trace() {
    // 设置退出标志
    exiting_ = true;
    
    // 等待线程完成
    if (rb_thread_.joinable()) {
        rb_thread_.join();
    }
    
    // 释放 ring buffer
    if (rb_) {
        ring_buffer__free(rb_);
        rb_ = nullptr;
    }
    
    logger_.info("Ring buffer polling stopped");

    // 清理链接和骨架
    for (auto link : links_) {
        bpf_link__destroy(link);
    }
    links_.clear();
    
    if (skel_) {
        func_snoop_bpf__destroy(skel_);
        skel_ = nullptr;
    }
    logger_.info("BPF skeleton destroyed");
}

void FuncSnoop::attach_bpf() {
    if (!skel_) {
        logger_.error("BPF skeleton not initialized");
        return;
    }

   
    std::string proc_self_path = "/proc/self/exe";
    uint64_t cookie = next_func_id_;
    int map_fd = bpf_map__fd(skel_->maps.func_names);
    
    //! 处理uprobe_cfg中的cuda函数
    std::string cuda_lib_path = ""; 
    if (uprobe_cfg_.contains("cuda_lib_path")) {
        cuda_lib_path = uprobe_cfg_["cuda_lib_path"];
    }

    std::string trace_file_path = ""; 
    if (uprobe_cfg_.contains("trace_file_path")) {
        trace_file_path = uprobe_cfg_["trace_file_path"];
    }

    if (uprobe_cfg_.contains("cuda_func_sym")) {
        const auto &func_sym = uprobe_cfg_["cuda_func_sym"];
        
        for (const auto &[func, sym] : func_sym.items()) {
            // 生成唯一cookie
            // uint64_t cookie = next_func_id_++;
            cookie = ++cookie;
            
            // 保存函数名到映射
            func_id_map_[cookie] = func;
            
            // 将函数名存入BPF映射
            map_fd = bpf_map__fd(skel_->maps.func_names);
            char func_name[64] = {0};
            strncpy(func_name, func.c_str(), sizeof(func_name) - 1);
            
            int err = bpf_map__update_elem(skel_->maps.func_names, &cookie, sizeof(cookie), 
                                            func_name, sizeof(func_name), BPF_ANY);
            if (err) {
                logger_.error("Failed to update func_names map: " + func + ", err=" + std::to_string(err));
                continue;
            }
            
            bool success = attach_function(
                skel_, cuda_lib_path, func, sym, cookie,  links_);

            if (!success) {
                logger_.error("Failed to attach function: " + func);
            }
        }
    }

        //! 处理libtorch中的函数
    std::string libtorch_lib_path = ""; 
    if (uprobe_cfg_.contains("libtorch_lib_path")) {
        libtorch_lib_path = uprobe_cfg_["libtorch_lib_path"];
        logger_.info("libtorch_lib_path: " + libtorch_lib_path);
    }

    if (uprobe_cfg_.contains("libtorch_func_sym")) {
        const auto &func_sym = uprobe_cfg_["libtorch_func_sym"];
        
        for (const auto &[func, sym] : func_sym.items()) {
            cookie = ++cookie;
            
            // 保存函数名到映射
            func_id_map_[cookie] = func;
            
            // 将函数名存入BPF映射
            map_fd = bpf_map__fd(skel_->maps.func_names);
            char func_name[64] = {0};
            strncpy(func_name, func.c_str(), sizeof(func_name) - 1);
            
            int err = bpf_map__update_elem(skel_->maps.func_names, &cookie, sizeof(cookie), 
                                            func_name, sizeof(func_name), BPF_ANY);
            if (err) {
                logger_.error("Failed to update func_names map: " + func + ", err=" + std::to_string(err));
                continue;
            }
            
            bool success = attach_function(
                skel_, libtorch_lib_path, func, sym, cookie,  links_);

            if (!success) {
                logger_.error("Failed to attach function: " + func);
            }
        }
    }

    if (uprobe_cfg_.contains("cublas_lib_path") && uprobe_cfg_.contains("cublas_func_sym")) {
        std::string cublas_lib_path = uprobe_cfg_["cublas_lib_path"];
        const auto &cublas_funcs = uprobe_cfg_["cublas_func_sym"];
        
        for (const auto &[func, sym] : cublas_funcs.items()) {
            cookie = ++cookie;
            func_id_map_[cookie] = func;
            
            char func_name[64] = {0};
            strncpy(func_name, func.c_str(), sizeof(func_name) - 1);
            
            int err = bpf_map__update_elem(skel_->maps.func_names, &cookie, sizeof(cookie), 
                                            func_name, sizeof(func_name), BPF_ANY);
            if (err) {
                logger_.error("Failed to update func_names map: " + func);
                continue;
            }
            
            bool success = attach_function(
                skel_, cublas_lib_path, func, sym, cookie, links_);

            if (!success) {
                logger_.error("Failed to attach cuBLAS function: " + func);
            }
        }
    }

    if (uprobe_cfg_.contains("cudnn_lib_path") && uprobe_cfg_.contains("cudnn_func_sym")) {
        std::string cudnn_lib_path = uprobe_cfg_["cudnn_lib_path"];
        const auto &cudnn_funcs = uprobe_cfg_["cudnn_func_sym"];
        
        for (const auto &[func, sym] : cudnn_funcs.items()) {
            cookie = ++cookie;
            func_id_map_[cookie] = func;
            
            char func_name[64] = {0};
            strncpy(func_name, func.c_str(), sizeof(func_name) - 1);
            
            int err = bpf_map__update_elem(skel_->maps.func_names, &cookie, sizeof(cookie), 
                                            func_name, sizeof(func_name), BPF_ANY);
            if (err) {
                logger_.error("Failed to update func_names map: " + func);
                continue;
            }
            
            bool success = attach_function(
                skel_, cudnn_lib_path, func, sym, cookie, links_);

            if (!success) {
                logger_.error("Failed to attach cuBLAS function: " + func);
            }
        }
    }

    if (uprobe_cfg_.contains("nccl_lib_path") && uprobe_cfg_.contains("nccl_func_sym")) {
        std::string nccl_lib_path = uprobe_cfg_["nccl_lib_path"];
        const auto &nccl_funcs = uprobe_cfg_["nccl_func_sym"];
        
        for (const auto &[func, sym] : nccl_funcs.items()) {
            cookie = ++cookie;
            func_id_map_[cookie] = func;
            
            char func_name[64] = {0};
            strncpy(func_name, func.c_str(), sizeof(func_name) - 1);
            
            int err = bpf_map__update_elem(skel_->maps.func_names, &cookie, sizeof(cookie), 
                                            func_name, sizeof(func_name), BPF_ANY);
            if (err) {
                logger_.error("Failed to update func_names map: " + func);
                continue;
            }
            
            bool success = attach_function(
                skel_, nccl_lib_path, func, sym, cookie, links_);

            if (!success) {
                logger_.error("Failed to attach cuBLAS function: " + func);
            }
        }
    }

    // cookie = ++cookie;

    // // 保存函数名到映射
    // func_id_map_[cookie] = "linear_forward";
    
    // // 将函数名存入BPF映射
    // map_fd = bpf_map__fd(skel_->maps.func_names);
    // char func_name[64] = {0};
    // strncpy(func_name, "linear_forward", sizeof(func_name) - 1);
    
    // int err = bpf_map__update_elem(skel_->maps.func_names, &cookie, sizeof(cookie), 
    // func_name, sizeof(func_name), BPF_ANY);
    // if (err) {
    // logger_.error("Failed to update func_names map: linear_forward, err=" + std::to_string(err));
    // }

    // int success = attach_function(
    //     skel_, libtorch_lib_path, func_name, "_ZN5torch2nn10LinearImpl7forwardERKN2at6TensorE", cookie,  links_);
    
    // if (!success) {
    //     logger_.error("Failed to attach function:linear_forward");
    // }

    //自定义的库函数
    // if (uprobe_cfg_.contains("custom_sym")) {
    //     const auto &custom_sym = uprobe_cfg_["custom_sym"];
    

    //     for (const auto &[func, sym] : custom_sym.items()) {
    //         // 生成唯一cookie
    //         uint64_t cookie = next_func_id_++;
            
    //         // 保存函数名到映射
    //         func_id_map_[cookie] = func;
            
    //         // 将函数名存入BPF映射
    //         map_fd = bpf_map__fd(skel_->maps.func_names);
    //         char func_name[64] = {0};
    //         strncpy(func_name, func.c_str(), sizeof(func_name) - 1);
            
    //         int err = bpf_map__update_elem(skel_->maps.func_names, &cookie, sizeof(cookie), 
    //         func_name, sizeof(func_name), BPF_ANY);
    //         if (err) {
    //             logger_.error("Failed to update func_names map: " + func + ", err=" + std::to_string(err));
    //             continue;
    //         }
    //         std::string custom_lib = ""; // 这里需要提供实际的库路径
    //         if (uprobe_cfg_.contains("my_custom_path")) {
    //             custom_lib = uprobe_cfg_["my_custom_path"];
    //         }
                        
    //         bool success = attach_function(
    //             skel_, custom_lib, func, sym, cookie,  links_);

    //         if (!success) {
    //             logger_.error("Failed to attach function: " + func);
    //         }
    //     }
    // }

    if (!start_trace()) {
        logger_.error("Failed to start ring buffer polling");
    }
    
}


// 读取 trace_pipe
std::tuple<std::string, int, int64_t, std::string> FuncSnoop::read_trace_pipe() {

    static int trace_fd = -1;
    
    // 如果文件描述符未打开，则打开
    if (trace_fd < 0) {
        trace_fd = open("/sys/kernel/debug/tracing/trace_pipe", O_RDONLY | O_NONBLOCK);
        if (trace_fd < 0) {
            return {"", 0, 0, ""};
        }
    }
    
    char buf[4096];
    ssize_t nread = read(trace_fd, buf, sizeof(buf) - 1);
    if (nread <= 0) {
        return {"", 0, 0, ""};
    }
    
    buf[nread] = '\0';
    
    // 解析数据
    // 格式通常是: <进程名>-<PID> [CPU] ... : <消息>
    // 我们需要分离出进程名、PID、时间戳和消息
    std::string line(buf);
    size_t pos;
    
    // 获取进程名和PID
    std::string task = "";
    int pid = 0;
    
    pos = line.find('-');
    if (pos != std::string::npos) {
        task = line.substr(0, pos);
        line = line.substr(pos + 1);
        
        pos = line.find(' ');
        if (pos != std::string::npos) {
            pid = std::stoi(line.substr(0, pos));
        }
    }
    
    // 获取消息（通常是最后一部分）
    std::string msg = "";
    pos = line.rfind(": ");
    if (pos != std::string::npos) {
        msg = line.substr(pos + 2);
    }
    
    // 从消息中提取时间戳
    int64_t ts = 0;
    pos = msg.find(' ');
    if (pos != std::string::npos) {
        ts = std::stoll(msg.substr(0, pos));
        msg = msg.substr(pos + 1);
    }
    
    return {task, pid, ts, msg};
}
// 附加单个函数
bool FuncSnoop::attach_function(struct func_snoop_bpf *skel,
                    const std::string &attach_file_path,
                    const std::string &func_name,
                    const std::string &sym,
                    uint64_t cookie,
                    std::vector<struct bpf_link *> &links) {
    try {
        // 准备uprobe选项
        struct bpf_uprobe_opts entry_opts = {};
        entry_opts.retprobe = false;
        entry_opts.bpf_cookie = cookie;
        entry_opts.func_name = sym.c_str();
        entry_opts.sz = sizeof(entry_opts);
        // 附加入口点
        struct bpf_link *entry_link = bpf_program__attach_uprobe_opts(
            skel->progs.generic_entry,
            -1,  
            attach_file_path.c_str(),
            0,   // 偏移量0表示使用符号
            &entry_opts
        );
        
        if (!entry_link) {
            logger_.error("Failed to attach entry point: " + func_name);
            return false;
        }
        
        // logger_.info("Attached generic_entry to " + attach_file_path + ":" + sym + 
        //            " with cookie " + std::to_string(cookie));
        
        // 准备uretprobe选项
        struct bpf_uprobe_opts exit_opts = {};
        exit_opts.retprobe = true;
        exit_opts.bpf_cookie = cookie;
        exit_opts.func_name = sym.c_str();
        exit_opts.sz = sizeof(exit_opts);
        
        // 附加出口点
        struct bpf_link *exit_link = bpf_program__attach_uprobe_opts(
            skel->progs.generic_exit,
            -1,  
            attach_file_path.c_str(),
            0,   // 偏移量0表示使用符号
            &exit_opts
        );
        
        if (!exit_link) {
            logger_.error("Failed to attach exit point: " + func_name);
            bpf_link__destroy(entry_link);
            // links.pop_back();
            return false;
        }
        
        // links.push_back(exit_link);
        // logger_.info("Attached generic_exit to " + attach_file_path + ":" + sym +
        //            " with cookie " + std::to_string(cookie));
        
        links.push_back(entry_link);
        links.push_back(exit_link);
            
        return true;
    } catch (const std::exception &e) {
        logger_.error(std::string(e.what()) + " in " + attach_file_path);
        return false;
    }
}

// 将此函数修改为类的静态成员函数
std::string FuncSnoop::get_mangled_name(const std::string& binary_path, const std::string& func_name) {
    int fd = open(binary_path.c_str(), O_RDONLY);
    if (fd < 0) return "";
    
    if (elf_version(EV_CURRENT) == EV_NONE) {
        close(fd);
        return "";
    }
    
    Elf* elf = elf_begin(fd, ELF_C_READ, NULL);
    if (!elf) {
        close(fd);
        return "";
    }
    
    Elf_Scn* scn = NULL;
    GElf_Shdr shdr;
    
    // 遍历所有节（section）
    while ((scn = elf_nextscn(elf, scn)) != NULL) {
        gelf_getshdr(scn, &shdr);
        
        // 查找符号表
        if (shdr.sh_type == SHT_SYMTAB || shdr.sh_type == SHT_DYNSYM) {
            Elf_Data* data = elf_getdata(scn, NULL);
            int count = shdr.sh_size / shdr.sh_entsize;
            
            // 遍历符号表中的所有条目
            for (int i = 0; i < count; i++) {
                GElf_Sym sym;
                gelf_getsym(data, i, &sym);
                
                char* name = elf_strptr(elf, shdr.sh_link, sym.st_name);
                char* demangled = NULL;
                int status;
                
                // 使用abi::__cxa_demangle解码符号名
                if (name) {
                    demangled = abi::__cxa_demangle(name, NULL, NULL, &status);
                    
                    // 如果解码成功并且包含函数名
                    if (status == 0 && demangled && strstr(demangled, func_name.c_str()) != NULL) {
                        std::string result(name);
                        free(demangled);
                        elf_end(elf);
                        close(fd);
                        return result;
                    }
                    
                    if (demangled) {
                        free(demangled);
                    }
                }
            }
        }
    }
    
    elf_end(elf);
    close(fd);
    return "";
}

void FuncSnoop::report_function_stats() {
    if (func_stats_.empty()) {
        return;
    }
    
    // 按调用次数排序的函数列表
    std::vector<std::pair<std::string, FunctionStats>> sorted_by_calls;
    for (const auto& [func, stats] : func_stats_) {
        if (stats.call_count > 0) {
            sorted_by_calls.push_back({func, stats});
        }
    }
    
    if (sorted_by_calls.empty()) {
        return;
    }
    
    // 按调用次数排序
    std::sort(sorted_by_calls.begin(), sorted_by_calls.end(),
            [](const auto& a, const auto& b) {
                return a.second.call_count > b.second.call_count;
            });
    
    // 输出函数统计摘要
    //logger_.info("============== Function Statistics ==============");
    // logger_.info("{:<30} {:>10} {:>12} {:>12} {:>12}",
    //         "Function", "Calls", "Avg", "Max", "Slow Calls");
    
    // 只输出最常调用的函数
    size_t count = std::min(sorted_by_calls.size(), output_freq_func_);
    for (size_t i = 0; i < count; i++) {
        const auto& [func, stats] = sorted_by_calls[i];
        // logger_.info("[FUNC] func:{} calls:{} avg:{} max:{} slow calls:{}",
        //         func,
        //         stats.call_count,
        //         logger_.format_duration(stats.avg_duration_us),
        //         logger_.format_duration(stats.max_duration_us),
        //         stats.slow_call_count);
    }
    
    // 按最大耗时排序
    std::vector<std::pair<std::string, FunctionStats>> sorted_by_time;
    for (const auto& [func, stats] : func_stats_) {
        if (stats.max_duration_us > 0) {
            sorted_by_time.push_back({func, stats});
        }
    }
    
    if (!sorted_by_time.empty()) {
        std::sort(sorted_by_time.begin(), sorted_by_time.end(),
                [](const auto& a, const auto& b) {
                    return a.second.max_duration_us > b.second.max_duration_us;
                });
        
        //logger_.info("------------- Slowest Functions --------------");
        count = std::min(sorted_by_time.size(), output_slow_calls_count_);
        for (size_t i = 0; i < count; i++) {
            const auto& [func, stats] = sorted_by_time[i];
            // logger_.info("[FUNC] {:<30} Max: {}µs, Avg: {}µs, Calls: {}",
            //         func,
            //         logger_.format_duration(stats.max_duration_us),
            //         logger_.format_duration(stats.avg_duration_us),
            //         stats.call_count);
        }
    }
    
    //logger_.info("================================================");
}

} // namespace NeuTracer