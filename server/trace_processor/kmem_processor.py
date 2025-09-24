import logging
import time
from typing import Dict, List, Any
from collections import defaultdict
import datetime

from trace_processor.base_processor import BaseTraceProcessor
from utils.formatters import parse_timestamp, format_bytes
from utils.file_manager import TraceFileManager
from prometheus_metrics import (
    MEM_USAGE, MEM_PEAK, MEM_ALLOC_RATE, MEM_FREE_RATE,
    MEM_TOTAL_ALLOCS, MEM_TOTAL_FREES, MEM_ALLOCATION_SIZE,
)

logger = logging.getLogger('tracer-service')

class MemTraceProcessor(BaseTraceProcessor):
    """内存跟踪处理器，负责从C++端接收已分析的内存数据，转换为Prometheus指标"""
    
    def __init__(self, file_manager: TraceFileManager):
        """初始化内存跟踪处理器"""
        super().__init__("memory", file_manager)
        
        # 进程内存统计数据
        self.process_stats: Dict[int, Dict[str, Any]] = {}
        
        # 跟踪开始时间
        self.start_time = time.time()
        
        # 上次更新时间，用于计算速率
        self.last_update_time = self.start_time
        
        # logger.info("CPU内存数据处理器初始化完成")
    
    def process_traces(self, traces: List[Any], is_last_batch: bool = False) -> int:
        """处理内存跟踪数据"""
        if not traces:
            return 0
            
        count = len(traces)
        self.total_traces += count

        sorted_traces = sorted(traces, key=lambda e: getattr(e, "timestamp", 0))
        
        events = []
        csv_rows = []
        csv_headers = [
            'Timestamp', 'PID', 
            'Alloc_size', 'Free_Size', 
            'Current Usage'
        ]

        for trace in sorted_traces:
            # 2.1 更新进程结构体
            event = self._process_single_trace(trace)
            if not event:
                continue

            pid = trace.tgid
            args = event["args"]
            timestamp_ms = event["ts"]

            # 2.2 构造JSON数据
            events.append(event)

            stats = self.process_stats[pid]

            labels = {'pid': str(pid)}

            MEM_USAGE.labels(**labels).set(event["args"]["current_usage"])
            MEM_TOTAL_ALLOCS.labels(**labels).set(event["args"]["alloc_size"])
            MEM_TOTAL_FREES.labels(**labels).set(event["args"]["free_size"])

            # 2.3 构造CSV数据
            csv_rows.append([
                timestamp_ms,
                pid,
                event["args"]["alloc_size"],
                event["args"]["free_size"],
                event["args"]["current_usage"]
            ])

        # 3. 写入json和csv
        self.write_events_to_json(events, is_last_batch)
        if csv_rows:
            self.write_rows_to_csv(csv_headers, csv_rows)

        # 4. 更新Prometheus指标
        # self._update_prometheus_metrics()

        return len(events)
    
    def _process_single_trace(self, trace) -> Dict[str, Any]:
        """处理单个内存跟踪事件"""
        try:
            # 提取基本信息
            pid = trace.tgid
            tid = trace.pid
            timestamp = trace.timestamp
            comm = trace.comm
            operation = trace.operation
            size = trace.size
            addr = trace.addr
            stack_id = trace.stack_id
            
            # 如果进程不存在于缓存中，初始化它
            if pid not in self.process_stats:
                self.process_stats[pid] = {
                    'comm': comm,
                    'pid': pid,
                    'tid': tid,
                    'time': timestamp,
                    'allocs_size': getattr(trace, 'total_allocs', 0),
                    'frees_size': getattr(trace, 'total_frees', 0),
                    'current_memory': getattr(trace, 'current_memory', 0),
                    'peak_memory': getattr(trace, 'peak_memory', 0),  
                    'total_allocs': 0,
                    'total_frees': 0,               
                }
            self.process_stats[pid]['total_allocs'] += getattr(trace, 'total_allocs', 0)
            self.process_stats[pid]['total_frees'] += getattr(trace, 'total_frees', 0)
            
            # 更新进程统计
            stats = self.process_stats[pid]
            
            # 跟踪内存分配和释放时间，用于计算速率
            # current_time = time.time()
            
            # # 根据操作类型更新统计
            # if operation.lower() == "alloc":
            #     # 记录分配时间
            #     stats['allocation_times'].append(current_time)
            #     if len(stats['alloc_sizes']) >= 30:
            #         stats['alloc_sizes'].pop(0)
            #     stats['alloc_sizes'].append(size)
                
            #     stats['addresses'][addr] = size
                
            # elif operation.lower() == "free":
            #     # 记录释放时间
            #     stats['deallocation_times'].append(current_time)
                
            #     if addr in stats['addresses']:
            #         del stats['addresses'][addr]
            
            # 创建JSON事件对象
            timestamp_ms = parse_timestamp(timestamp) if timestamp else time.time() * 1000
            
            event_type = "M" if operation.lower() == "alloc" else "F"
            
            # 创建事件
            event = {
                "name": f"{comm}",
                "ph": operation, 
                "ts": timestamp_ms,
                "pid": pid,
                "tid": tid,
                "args": {
                    "size": size,
                    "address": f"0x{addr:x}",
                    "stack_id": stack_id,
                    "current_usage": trace.current_memory,
                    "peak_usage": trace.peak_memory,
                    "alloc_size": trace.total_allocs,
                    "free_size": trace.total_frees
                }
            }
            
            return event
            
        except Exception as e:
            logger.error(f"Error processing memory trace: {str(e)}", exc_info=True)
            return None
    
    def _update_prometheus_metrics(self) -> None:
        """更新Prometheus指标"""
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # 对于每个进程更新指标
        for pid, stats in self.process_stats.items():
            comm = stats['comm']
            labels = {'pid': str(pid), 'process': comm}
            
            # 更新基本内存使用指标
            MEM_USAGE.labels(**labels).set(stats['current_memory'])
            MEM_PEAK.labels(**labels).set(stats['peak_memory'])
            MEM_TOTAL_ALLOCS.labels(**labels).set(stats['total_allocs'])
            MEM_TOTAL_FREES.labels(**labels).set(stats['total_frees'])
            
            # # 计算近期分配/释放速率
            # recent_cutoff = current_time - 60  # 最近60秒
            # recent_allocs = sum(1 for t in stats['allocation_times'] if t >= recent_cutoff)
            # recent_frees = sum(1 for t in stats['deallocation_times'] if t >= recent_cutoff)
            
            # # 删除旧记录以保持列表较小
            # stats['allocation_times'] = [t for t in stats['allocation_times'] if t >= recent_cutoff]
            # stats['deallocation_times'] = [t for t in stats['deallocation_times'] if t >= recent_cutoff]
            
            # # 设置速率指标
            # alloc_rate = recent_allocs / 60.0
            # free_rate = recent_frees / 60.0
            # MEM_ALLOC_RATE.labels(**labels).set(alloc_rate)
            # MEM_FREE_RATE.labels(**labels).set(free_rate)

            # # 更新分配大小分布
            # for size in stats['alloc_sizes']:
            #     MEM_ALLOCATION_SIZE.labels(**labels).observe(size)
    
    # def _export_incremental_stats_to_csv(self, events: List[Dict[str,Any]]) -> None:
    #     """增量导出内存统计数据到CSV"""
    #     # 准备表头和数据
    #     events_headers = [
    #         'Timestamp', 'PID', 'Size', 
    #         'Current Memory', 
    #     ]
        
    #     events_rows = []
    #     current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    #     # 使用最近一段时间内收集的数据
    #     # elapsed = time.time() - getattr(self, '_last_exported_stats_time', 0)
    #     # if elapsed < 10:  # 每10秒导出一次
    #     #     return
        
    #     self._last_exported_stats_time = time.time()
        
    #     for i, event in enumerate(events):
    #         # pending = events['total_allocs'] - events['total_frees']
            
    #         row = [
    #             event['ts'],
    #             event['pid'],
    #             event['args']['size'],
    #             event['args']['current_usage']
    #         ]
    #         events_rows.append(row)
        
    #     # 导出统计数据
    #     if events_rows:
    #         self.write_rows_to_csv(events_headers, events_rows)
        
        # # 导出内存分配大小分布
        # alloc_headers = [
        #     'Timestamp', 'PID', 'Process', 'Alloc Size', 'Count'
        # ]
        # alloc_rows = []
        
        # # 处理每个进程的分配大小分布
        # for pid, stats in self.process_stats.items():
        #     if not stats['alloc_sizes']:
        #         continue
                
        #     # 统计分配大小分布
        #     size_counts = {}
        #     for size in stats['alloc_sizes']:
        #         # 将大小按范围分组，如0-16B, 16-64B, 64-256B等
        #         size_range = self._get_size_range(size)
        #         size_counts[size_range] = size_counts.get(size_range, 0) + 1
            
        #     # 添加每个大小范围的记录
        #     for size_range, count in size_counts.items():
        #         row = [
        #             current_time,
        #             pid,
        #             stats['comm'],
        #             size_range,
        #             count
        #         ]
        #         alloc_rows.append(row)
        
        # 导出分配大小分布
        # if alloc_rows:
        #     self.file_manager.export_to_csv(
        #         trace_type="memory",
        #         data_type="allocation_sizes",
        #         headers=alloc_headers,
        #         rows=alloc_rows,
        #         append=True
        #     )

    def _get_size_range(self, size: int) -> str:
        """根据分配大小返回范围字符串"""
        ranges = [
            (0, 16, "0-16B"),
            (16, 64, "16-64B"),
            (64, 256, "64-256B"),
            (256, 1024, "256B-1KB"),
            (1024, 4096, "1-4KB"),
            (4096, 16384, "4-16KB"),
            (16384, 65536, "16-64KB"),
            (65536, 262144, "64-256KB"),
            (262144, 1048576, "256KB-1MB"),
            (1048576, 4194304, "1-4MB"),
            (4194304, 16777216, "4-16MB"),
            (16777216, 67108864, "16-64MB"),
            (67108864, float('inf'), ">64MB")
        ]
        
        for min_size, max_size, range_str in ranges:
            if min_size <= size < max_size:
                return range_str
        
        return "未知大小"


    def collect_stats(self) -> None:
        """收集并记录内存统计信息"""
        # 记录总体内存使用
        total_current = sum(stats['current_memory'] for stats in self.process_stats.values())
        total_peak = sum(stats['peak_memory'] for stats in self.process_stats.values())
        total_allocs = sum(stats['total_allocs'] for stats in self.process_stats.values())
        total_frees = sum(stats['total_frees'] for stats in self.process_stats.values())
        
        logger.info(f"Total memory usage: current={format_bytes(total_current)}, "
                   f"peak={format_bytes(total_peak)}")
        logger.info(f"Total operations: allocations={total_allocs}, "
                   f"frees={total_frees}, "
                   f"pending={total_allocs - total_frees}")
        
        # 记录内存最高的进程
        top_memory = sorted(
            self.process_stats.items(), 
            key=lambda x: x[1]['current_memory'], 
            reverse=True
        )[:5]
        
        logger.info("Top memory consumers:")
        for pid, stats in top_memory:
            logger.info(f"  {stats['comm']} (PID {pid}): {format_bytes(stats['current_memory'])} "
                      f"current, {format_bytes(stats['peak_memory'])} peak"
        )
    

    def finalize(self) -> None:
        """完成处理，执行清理工作"""
        return  # 目前不需要清理工作
        self.collect_stats()
        