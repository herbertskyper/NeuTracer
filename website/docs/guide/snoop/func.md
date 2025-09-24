# 函数模块 用户态程序

FuncSnoop 通过 uprobes 实现对用户空间函数的低开销跟踪和分析。可以统计函数调用次数、耗时、入口/出口事件的功能，能够识别慢调用、计算平均/最大/最小响应时间，支持运行时对特定二进制文件的函数进行动态挂钩，通过 ring buffer 高效收集和处理跟踪数据，并定期生成函数调用统计和性能报告。
该模块不仅支持对 CUDA、cuBLAS、cuDNN、NCCL、libtorch 等多种库函数的统一监控，还能灵活扩展自定义函数，具备高性能、低开销和多进程支持等核心能力。
```json
// uprobe 配置示例
{
    "cuda_func_sym": {
        "cudaMalloc": "cudaMalloc",
        "cudaFree": "cudaFree",
        "cudaDeviceSynchronize": "cudaDeviceSynchronize",
        "cudaGetLastError": "cudaGetLastError",
        "cudaGetDevice": "cudaGetDevice",
        ......
    },
    "custom_sym":{
        "vectorAdd": "vectorAdd"
        ......
    },
    "libtorch_func_sym": {
        "linear_forward": "_ZN5torch2nn10LinearImpl7forwardERKN2at6TensorE"
        ......
    },
    "cudnn_func_sym": {
        "cudnnCreate": "cudnnCreate",
        "cudnnDestroy": "cudnnDestroy",
        "cudnnConvolutionForward": "cudnnConvolutionForward",
        "cudnnConvolutionBackwardData": "cudnnConvolutionBackwardData"
        ......

    },

    "nccl_func_sym": {
        "ncclCommInitRank": "ncclCommInitRank",
        "ncclAllReduce": "ncclAllReduce",
        "ncclBroadcast": "ncclBroadcast",
        ......
    },
}
```

核心事件结构（func_trace_event）记录函数名、PID、TGID、调用类型（ENTRY/EXIT）、参数、返回值、唯一 cookie 及时间戳。统计结构（FunctionStats）维护每个函数的调用次数、平均/最大/最小耗时、慢调用次数及详细慢调用记录，支持高频函数和慢函数的实时分析。

核心方法涵盖初始化、事件处理和线程管理三大部分。初始化时 attach_bpf() 加载 eBPF 程序并自动解析配置，批量附加到目标库的指定函数。start_trace() 启动 ring buffer 事件轮询线程，stop_trace() 停止监控并清理所有资源。事件处理包括 process_event() 解析函数调用事件，统计调用次数、耗时、慢调用等信息，并通过 report_function_stats() 定期输出函数统计摘要。多线程管理通过 ring_buffer_thread() 实现高效事件轮询，确保数据实时采集和处理。
```cpp
// 函数挂载示例
if (uprobe_cfg_.contains("cuda_func_sym")) {
        const auto &func_sym = uprobe_cfg_["cuda_func_sym"];
        
        for (const auto &[func, sym] : func_sym.items()) {
            // 生成唯一cookie
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
```
代码采用采样机制和高效哈希表维护函数统计信息，支持高并发场景下的数据一致性。通过互斥锁保护关键数据结构，确保多线程安全访问。同时代码支持将采集到的函数调用事件实时转发到外部系统（如 RPC/Grafana），并对慢调用自动输出告警日志，便于快速定位性能瓶颈。