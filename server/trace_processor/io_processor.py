import logging
import time
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime

from trace_processor.base_processor import BaseTraceProcessor
from utils.formatters import parse_timestamp, format_bytes
from utils.file_manager import TraceFileManager
from prometheus_metrics import (
    IO_THROUGHPUT, IO_LATENCY, IO_OPS_RATE, IO_TOTAL_BYTES,
)

logger = logging.getLogger('tracer-service')

class IOTraceProcessor(BaseTraceProcessor):
    """IO跟踪处理器，负责从C++端接收已分析的IO数据，转换为Prometheus指标"""
    
    def __init__(self, file_manager: TraceFileManager):
        """初始化IO跟踪处理器"""
        super().__init__("io", file_manager)
        
        # 进程IO统计数据 - 保存从C++端接收的数据
        self.process_stats: Dict[int, Dict[str, Any]] = {}
        
        # 设备统计数据
        self.device_stats: Dict[str, Dict[str, Any]] = {}
        
        # 跟踪开始时间
        self.start_time = time.time()
        
        # 上次更新时间，用于计算速率
        self.last_update_time = self.start_time
        
        # logger.info("IO数据收集器初始化成功")
    
    def process_traces(self, traces: List[Any], is_last_batch: bool = False) -> int:
        """处理IO跟踪数据"""
        if not traces:
            return 0
        
        sorted_traces = sorted(traces, key=lambda t: getattr(t, 'timestamp', 0))
        count = len(traces)
        self.total_traces += count
        events = []
        csv_rows = []
        csv_headers = [
            'Timestamp', 'PID', 
            'Read Bytes', 'Write Bytes', 'READ Latency (ms)','WRITE Latency (ms)',
            'Read Ops', 'Write Ops', 
        ]

        for trace in sorted_traces:
            event = self._process_single_trace(trace)
            if not event:
                continue
            pid = trace.pid
            args = event["args"]
            time = event["ts"]

            IO_THROUGHPUT.labels(pid=str(pid), operation="read").set(args["read_bytes"])
            IO_LATENCY.labels(pid=str(pid), operation="read").set(args["read_latency_ms"])
            IO_THROUGHPUT.labels(pid=str(pid), operation="write").set(args["write_bytes"])
            IO_LATENCY.labels(pid=str(pid), operation="write").set(args["write_latency_ms"])

            events.append(event)

            csv_rows.append([
                time,
                pid,
                args["read_bytes"],
                args["write_bytes"],
                args["read_latency_ms"],
                args["write_latency_ms"],
                args["read_ops"],
                args["write_ops"]
            ])
        
        self.write_events_to_json(events, is_last_batch)
        if csv_rows:
            self.write_rows_to_csv(csv_headers, csv_rows)

        # 4. 更新Prometheus指标
        # self._update_prometheus_metrics()

        return len(events)
    
    def _process_single_trace(self, trace) -> Dict[str, Any]:
        # print("Processing IO trace:")
        """处理单个IO跟踪事件"""
        try:
            # 提取基本信息
            pid = trace.pid
            timestamp = trace.timestamp
            comm = trace.comm
            operation = trace.operation
            bytes_count = trace.bytes
            device = trace.device
            read_latency_ms = trace.avg_read_latency
            write_latency_ms = trace.avg_write_latency
            read_bytes = trace.read_bytes
            write_bytes = trace.write_bytes
            
            # 如果进程不存在于缓存中，初始化它
            if pid not in self.process_stats:
                self.process_stats[pid] = {
                    'comm': comm,
                    'read_bytes': 0,
                    'write_bytes': 0,
                    'read_ops': 0,
                    'write_ops': 0,
                    'read_latency_ms': 0,
                    'write_latency_ms': 0,
                    'devices': device,
                    'last_seen': timestamp
                }
            
            # 更新进程统计
            stats = self.process_stats[pid]
            stats['read_bytes'] = read_bytes
            stats['write_bytes'] = write_bytes
            stats['read_ops'] = trace.read_ops
            stats['write_ops'] = trace.write_ops
            stats['read_latency_ms'] = read_latency_ms
            stats['write_latency_ms'] = write_latency_ms
            
            # # 根据操作类型更新统计
            # if operation.upper() == "READ":
            #     stats['read_bytes'] += bytes_count
            #     stats['read_ops'] += 1
            #     stats['read_latency_sum_ms'] += latency_ms
            #     # stats['devices'][device]['read_bytes'] += bytes_count
            # elif operation.upper() == "WRITE":
            #     stats['write_bytes'] += bytes_count
            #     stats['write_ops'] += 1
            #     stats['write_latency_sum_ms'] += latency_ms
            #     # stats['devices'][device]['write_bytes'] += bytes_count
            
            # 更新设备操作计数
            # stats['devices'][device]['ops'] += 1
            
            # # 记录最大延迟
            # if latency_ms > stats['max_latency_ms']:
            #     stats['max_latency_ms'] = latency_ms
                
            # 更新设备统计
            # if device not in self.device_stats:
            #     self.device_stats[device] = {
            #         'read_bytes': 0,
            #         'write_bytes': 0,
            #         'read_ops': 0,
            #         'write_ops': 0,
            #         'processes': set()
            #     }
            
            # dev_stats = self.device_stats[device]
            # dev_stats['processes'].add(pid)
            # if operation.upper() == "READ":
            #     dev_stats['read_bytes'] += bytes_count
            #     dev_stats['read_ops'] += 1
            # elif operation.upper() == "WRITE":
            #     dev_stats['write_bytes'] += bytes_count
            #     dev_stats['write_ops'] += 1

            # 创建JSON事件对象
            timestamp_ms = parse_timestamp(timestamp) if timestamp else time.time() * 1000
            
            # 创建IO事件
            event = {
                "name": f"{comm}",
                "ts": timestamp_ms,
                "pid": pid,
                "args": {
                    "operation": operation,
                    "device": device,
                    "read_bytes": stats['read_bytes'],
                    "write_bytes": stats['write_bytes'],
                    "read_ops": stats['read_ops'],
                    "write_ops": stats['write_ops'],
                    "read_latency_ms": stats['read_latency_ms'],
                    "write_latency_ms": stats['write_latency_ms'],
                }
            }
            
            return event
            
        except Exception as e:
            logger.error(f"Error processing IO trace: {str(e)}", exc_info=True)
            return None
    
    # def _update_prometheus_metrics(self) -> None:
    #     """直接用trace传来的数据更新Prometheus指标，无需本地聚合和速率计算"""
    #     for pid, stats in self.process_stats.items():
    #         # 直接用最新trace数据更新Prometheus指标

            
    # def _update_prometheus_metrics(self) -> None:
    #     """更新Prometheus指标"""
    #     current_time = time.time()
    #     elapsed_time = current_time - self.last_update_time
    #     self.last_update_time = current_time
        
    #     # 对于每个进程更新指标
    #     for pid, stats in self.process_stats.items():
    #         comm = stats['comm']
    #         labels = {'pid': str(pid), 'process': comm}
            
    #         # 更新基本吞吐量和延迟指标
    #         total_bytes = stats['read_bytes'] + stats['write_bytes']
    #         total_ops = stats['read_ops'] + stats['write_ops']
            
    #         # 计算速率
    #         throughput_bps = total_bytes / (current_time - self.start_time) if (current_time > self.start_time) else 0
    #         ops_per_sec = total_ops / (current_time - self.start_time) if (current_time > self.start_time) else 0
            
    #         # 读操作指标
    #         if stats['read_ops'] > 0:
    #             avg_read_latency = stats['read_latency_sum_ms'] / stats['read_ops']
    #             IO_LATENCY.labels(pid=str(pid), operation='read').set(avg_read_latency)
    #             IO_THROUGHPUT.labels(pid=str(pid), operation='read').set(
    #                 stats['read_bytes'] / (current_time - self.start_time) if (current_time > self.start_time) else 0
    #             )
    #             IO_OPS_RATE.labels(pid=str(pid), operation='read').set(
    #                 stats['read_ops'] / (current_time - self.start_time) if (current_time > self.start_time) else 0
    #             )
    #             IO_TOTAL_BYTES.labels(pid=str(pid), operation='read').set(stats['read_bytes'])
            
    #         # 写操作指标
    #         if stats['write_ops'] > 0:
    #             avg_write_latency = stats['write_latency_sum_ms'] / stats['write_ops']
    #             IO_LATENCY.labels(pid=str(pid), operation='write').set(avg_write_latency)
    #             IO_THROUGHPUT.labels(pid=str(pid), operation='write').set(
    #                 stats['write_bytes'] / (current_time - self.start_time) if (current_time > self.start_time) else 0
    #             )
    #             IO_OPS_RATE.labels(pid=str(pid), operation='write').set(
    #                 stats['write_ops'] / (current_time - self.start_time) if (current_time > self.start_time) else 0
    #             )
    #             IO_TOTAL_BYTES.labels(pid=str(pid), operation='write').set(stats['write_bytes'])
            
    #         # 总体IO指标
    #         IO_THROUGHPUT.labels(pid=str(pid), operation='total').set(throughput_bps)
    #         IO_OPS_RATE.labels(pid=str(pid), operation='total').set(ops_per_sec)
    #         IO_TOTAL_BYTES.labels(pid=str(pid), operation='total').set(total_bytes)
            
            # # 更新设备使用指标
            # for device, dev_stats in stats['devices'].items():
            #     IO_DEVICE_USAGE.labels(pid=str(pid), device=device, operation='read').set(
            #         dev_stats['read_bytes']
            #     )
            #     IO_DEVICE_USAGE.labels(pid=str(pid), device=device, operation='write').set(
            #         dev_stats['write_bytes']
            #     )
            
    # def export_incremental_stats_to_csv(self, events: List[Dict[str,Any]]) -> None:
    #     """增量导出IO统计数据到CSV"""
    #     # 准备表头和数据
    #     events_headers = [
    #         'Timestamp', 'PID', 
    #         'Read Bytes', 'Write Bytes', 'Latency (ms)',
    #         'Read Ops', 'Write Ops', 
    #     ]
        
    #     events_rows = []
    #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    #     # # 使用最近一段时间内收集的数据
    #     # elapsed = time.time() - getattr(self, '_last_exported_events_time', 0)
    #     # if elapsed < 10:  # 每10秒导出一次
    #     #     return
        
    #     self._last_exported_stats_time = time.time()
        
    #     for i, event in enumerate(events):
    #         # total_bytes = stats['read_bytes'] + stats['write_bytes']
    #         # devices_str = ", ".join(list(stats['devices'].keys()))
            
    #         row = [
    #             event['ts'],  
    #             event['pid'],
    #             event['args']['read_bytes'],
    #             event['args']['write_bytes'],
    #             event['args']['latency_ms'],
    #             event['args']['read_ops'],
    #             event['args']['write_ops']
    #         ]
    #         events_rows.append(row)
        
    #     # 导出统计数据
    #     if events_rows:
    #         self.write_rows_to_csv(events_headers, events_rows)
        
        
        # # 导出设备统计数据
        # device_headers = [
        #     'Timestamp', 'Device', 'Read Bytes', 'Write Bytes',
        #     'Read Ops', 'Write Ops', 'Process Count'
        # ]
        # device_rows = []
        
        # for device, stats in self.device_stats.items():
        #     row = [
        #         current_time,
        #         device,
        #         stats['read_bytes'],
        #         stats['write_bytes'],
        #         stats['read_ops'],
        #         stats['write_ops'],
        #         len(stats['processes'])
        #     ]
        #     device_rows.append(row)
        

    
    def collect_stats(self) -> None:
        """收集并记录IO统计信息"""
        
        # 记录设备统计
        logger.info(f"IO Devices: Monitoring {len(self.device_stats)} devices")
        for device, stats in self.device_stats.items():
            logger.info(f"  Device {device}: {format_bytes(stats['read_bytes'])} read, "
                       f"{format_bytes(stats['write_bytes'])} written, "
                       f"{len(stats['processes'])} processes")
        
        
        # 记录IO最活跃的进程
        top_io = sorted(
            self.process_stats.items(), 
            key=lambda x: x[1]['read_bytes'] + x[1]['write_bytes'], 
            reverse=True
        )[:5]
        
        logger.info("Top IO consumers:")
        for pid, stats in top_io:
            total_bytes = stats['read_bytes'] + stats['write_bytes']
            total_ops = stats['read_ops'] + stats['write_ops']
            logger.info(f"  {stats['comm']} (PID {pid}): {format_bytes(total_bytes)} total, "
                       f"{total_ops} operations, "
                       f"{len(stats['devices'])} devices")
            

    def finalize(self) -> None:
        """完成处理，执行清理工作"""
        return
        self.collect_stats()
        
