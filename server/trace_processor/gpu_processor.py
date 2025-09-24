import logging
from typing import Dict, List, Any, Tuple
import time
import hashlib

from trace_processor.base_processor import BaseTraceProcessor
from utils.formatters import parse_timestamp
from utils.file_manager import TraceFileManager
from prometheus_metrics import GPU_MEM_SIZE,GPU_KERNEL_FUNCTION_COUNT, GPU_MEM_TRANS_RATE
from tracer_service_pb2 import GPUTraceData

logger = logging.getLogger('tracer-service')

class GPUTraceProcessor(BaseTraceProcessor):
    def __init__(self, file_manager: TraceFileManager):
        super().__init__("gpu", file_manager)
        
    def process_traces(self, traces: List[Any]) -> int:
        """处理GPU跟踪数据"""
        # print(f"Processing {len(traces)} GPU traces")
        for trace in traces:
            # 检查事件类型
            if hasattr(trace, 'event_type'):
                # 处理内存事件
                if trace.event_type == GPUTraceData.EventType.MEMEVENT:
                    if hasattr(trace, 'mem_event') and trace.mem_event:
                        # 获取内存事件数据
                        pid = trace.mem_event.pid
                        mem_size = trace.mem_event.mem_size
                        
                        # 设置 Prometheus 指标
                        GPU_MEM_SIZE.labels(pid=str(pid)).set(mem_size)
                        
                        # 记录日志
                        # logger.info(f"GPU Memory Event: PID={pid}, Size={mem_size} bytes")
                        
                
                # 处理调用栈事件
                elif trace.event_type == GPUTraceData.EventType.CALLSTACK:
                    from datetime import datetime
            
                    # 获取调用栈和时间戳
                    stack_message = trace.callstack_event.stack_message
                    timestamp = trace.timestamp

                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    
                    # 计算自 Unix 纪元以来的秒数
                    epoch_seconds = dt.timestamp()
                    
                    # 转换为纳秒 (1 秒 = 1,000,000,000 纳秒)
                    timestamp = int(epoch_seconds * 1_000_000_000)
                    if self.file_manager.firstime is None:
                        self.file_manager.firstime = timestamp
                    timestamp -= self.file_manager.firstime
                    
                    # 解析调用栈 - 按行分割并移除 cudaLaunchKernel 行
                    stack_frames = []
                    if stack_message:
                        stack_frames = stack_message.split('\n')
                        stack_frames = [frame for frame in stack_frames if frame.strip() and not frame.strip().startswith('cudaLaunchKernel')]
                    
                    # 获取 PID (从事件中获取或使用当前进程的 PID)
                    pid = trace.callstack_event.pid
                    
                    # 计算帧数量
                    frame_count = len(stack_frames)
                    
                    if frame_count > 0:
                        # 构建格式化的输出行
                        event_line = f"EVENT|{timestamp}|{pid}|{frame_count}|{'|'.join(stack_frames)}"
                        
                        # 将输出写入文件
                        stack_trace_file = self.file_manager.stack_file
                        
                        stack_trace_file.write(event_line + "\n")
                        
                        # logger.info(f"已将调用栈写入文件 (event行: {event_line})")
                    
                
                # 处理CUDA启动事件
                elif trace.event_type == GPUTraceData.EventType.CUDALAUNCHEVENT:
                    if hasattr(trace, 'cuda_launch_event') and trace.cuda_launch_event:
                        # 获取内存事件数据
                        pid = trace.cuda_launch_event.pid
                        kernel_name = trace.cuda_launch_event.kernel_name
                        num_calls = trace.cuda_launch_event.num_calls
                        
                        # 设置 Prometheus 指标
                        GPU_KERNEL_FUNCTION_COUNT.labels(pid=str(pid), kernel_name=kernel_name).set(num_calls)
                        
                        # 记录日志
                        # logger.info(f"GPU Kernel Launch Event: PID={pid}, Kernel={kernel_name}, Count={num_calls}")
                elif trace.event_type == GPUTraceData.EventType.MEMTRANSEVENT:
                    if hasattr(trace, 'mem_trans_event') and trace.mem_trans_event:
                        # 获取内存事件数据
                        pid = trace.mem_trans_event.pid
                        mem_trans_rate = trace.mem_trans_event.mem_trans_rate
                        kind_str = trace.mem_trans_event.kind_str
                        
                        # 设置 Prometheus 指标
                        GPU_MEM_TRANS_RATE.labels(pid=str(pid), kind_str=str(kind_str)).set(mem_trans_rate)
                        
                        # 记录日志
                        # logger.info(f"GPU Memory Transfer Event: PID={pid}, Kind={kind_str}, Transfer Rate={mem_trans_rate} Mbytes/s")
        
        # count = len(traces)
        # BATCH_SIZE.labels(type="gpu").observe(count)
        # TRACE_COUNTER.labels(type="gpu").inc(count)
        # self.total_traces += count
        
        # # 创建事件字典
        # events = []
        
        # # 按照时间戳排序输入的跟踪数据
        # sorted_traces = sorted(traces, 
        #                     key=lambda t: float(t.timestamp) if isinstance(t.timestamp, (int, float)) or 
        #                     (isinstance(t.timestamp, str) and t.timestamp.replace('.', '', 1).isdigit()) else 0)
        
        # # 记录每个调用栈的最后出现时间戳，用于计算持续时间
        # stack_last_time = {}
        
        # # 第一步：收集所有事件的时间戳和调用栈
        # all_trace_info = []
        
        # for trace in sorted_traces:
        #     # 获取时间戳和调用栈
        #     timestamp_str = trace.timestamp
        #     stack_message = trace.stackmessage
            
        #     # 处理时间戳 - 确保为微秒级整数
        #     try:
        #         timestamp = int(parse_timestamp(timestamp_str) * 1000)  # 转为微秒
        #     except (ValueError, TypeError):
        #         timestamp = int(time.time() * 1000000)
        #         logger.warning(f"无法解析时间戳 '{timestamp_str}'，使用当前时间代替")
            
        #     # 处理调用栈
        #     stack_parts = stack_message.split('\n')
        #     if stack_parts and stack_parts[0] == 'cudaLaunchKernel':
        #         kernel_name = stack_parts[0]  # 保留CUDA核心名称
        #         stack_parts = stack_parts[1:]  # 移除cudaLaunchKernel
        #     else:
        #         kernel_name = "unknown"
            
        #     # 过滤空行
        #     stack_parts = [part for part in stack_parts if part.strip()]
            
        #     if stack_parts:
        #         # 将调用栈反转，使最底层的调用在前面
        #         stack_parts = stack_parts[::-1]
                
        #         # 为每一层调用栈生成唯一标识
        #         for i in range(len(stack_parts)):
        #             # 当前层的调用路径
        #             current_path = tuple(stack_parts[:i+1])
        #             stack_key = " > ".join(current_path)
                    
        #             # 记录这个调用栈路径的最后出现时间
        #             stack_last_time[stack_key] = timestamp
                
        #         # 保存完整信息供后续处理
        #         all_trace_info.append({
        #             "timestamp": timestamp,
        #             "stack_parts": stack_parts,
        #             "kernel_name": kernel_name
        #         })
        
        # # 第二步：基于收集的信息创建事件
        # for trace_info in all_trace_info:
        #     timestamp = trace_info["timestamp"]
        #     stack_parts = trace_info["stack_parts"]
        #     kernel_name = trace_info["kernel_name"]
            
        #     # 为调用栈的每一层创建事件
        #     for i in range(len(stack_parts)):
        #         # 当前层的调用路径和函数名
        #         current_path = tuple(stack_parts[:i+1])
        #         func_name = stack_parts[i]
        #         stack_key = " > ".join(current_path)
        #         parent_key = " > ".join(current_path[:-1]) if i > 0 else ""
                
        #         # 计算这一层的持续时间
        #         # 默认持续时间为10ms (10000μs)
        #         default_duration = 10000
                
        #         # 如果能够计算实际持续时间，则使用实际值
        #         if i < len(stack_parts) - 1:
        #             # 对于非叶子节点，持续时间是固定的小值，因为它们的宽度由子节点决定
        #             duration = 10  # 微秒
        #         else:
        #             # 对于叶子节点（最深层调用），使用默认持续时间
        #             duration = default_duration
                    
        #         # 为该调用层次生成一致的线程ID
        #         # 不同调用路径应有不同的tid，相同路径的不同层次应有相同的tid
        #         path_hash = int(hashlib.md5(parent_key.encode()).hexdigest(), 16) % 10000 if parent_key else 1
                
        #         # 创建事件
        #         event = {
        #             "name": func_name,
        #             "cat": "gpu",
        #             "ph": "B",  # 开始事件
        #             "ts": timestamp / 1000,  # 转换为毫秒
        #             "pid": 1,  # 进程ID
        #             "tid": path_hash,  # 线程ID基于调用路径
        #             "args": {
        #                 "kernel": kernel_name if i == len(stack_parts) - 1 else "",
        #                 "stack_level": i,
        #                 "full_path": stack_key
        #             }
        #         }
        #         events.append(event)
                
        #         # 添加对应的结束事件
        #         end_event = {
        #             "name": func_name,
        #             "cat": "gpu",
        #             "ph": "E",  # 结束事件
        #             "ts": (timestamp + duration) / 1000,  # 转换为毫秒
        #             "pid": 1,
        #             "tid": path_hash
        #         }
        #         events.append(end_event)
        
        # # 写入事件到JSON文件
        # self.write_events(events, is_last_batch)
        
        # return count