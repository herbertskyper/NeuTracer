import logging
from typing import Dict, List, Tuple, Any, Set
import time
import datetime as datetime

from trace_processor.base_processor import BaseTraceProcessor
from utils.formatters import format_timestamp, format_duration, parse_timestamp
from utils.file_manager import TraceFileManager
from prometheus_metrics import (
    TRACE_COUNTER, BATCH_SIZE, FUNCTION_CALLS, 
    ACTIVE_FUNCTIONS,
    FUNCTION_STATS_SUMMARY
)

logger = logging.getLogger('tracer-service')

class FunctionTraceProcessor(BaseTraceProcessor):
    def __init__(self, file_manager: TraceFileManager):
        super().__init__("function", file_manager)
        
        # 函数执行持续时间 - 记录每个函数调用的开始时间
        # 键格式: (pid, tgid, cookie)
        self.function_starts: Dict[Tuple[int, int, int], float] = {}
        
        # 跟踪活跃函数计数
        self.active_functions: Dict[str, int] = {}  # 按进程分组的活跃函数计数

        # 函数统计
        self.function_stats: Dict[str, Dict[str, Any]] = {}
        
    def process_traces(self, traces: List[Any], is_last_batch: bool = False) -> int:
        """处理函数调用跟踪数据"""
        if not traces:
            return 0
        
        count = len(traces)
        BATCH_SIZE.labels(type="function").observe(count)
        TRACE_COUNTER.labels(type="function").inc(count)
        self.total_traces += count
        
        # 按时间戳排序输入的跟踪数据
        sorted_traces = sorted(traces, 
                            key=lambda t: t.timestamp if t.timestamp else "")
        
        events = []
        csv_rows = []
        
        # 处理所有跟踪事件
        for trace in sorted_traces:
            # 解析时间戳为毫秒级浮点数
            timestamp_ms = parse_timestamp(trace.timestamp)
            
            # 获取进程和线程ID
            pid = trace.tgid   # 进程ID (默认为1)
            tid = trace.pid    # 线程ID (默认为1)
            
            # 获取函数名称和调用ID
            function_name = trace.function_name or "unknown"
            cookie = trace.cookie
            
            # 创建调用标识符
            call_id = (pid, tid, cookie)
            
            # 事件类型: 0=ENTRY, 1=EXIT
            is_entry = trace.event_type == 0
            
            # 获取进程名称 (使用 TGID 作为标识符)
            process_name = f"pid-{pid}"
            
            # 处理函数调用开始
            if is_entry:
                # 记录该函数的开始时间
                # self.function_starts[call_id] = timestamp_ms
                
                # 创建函数调用开始事件
                event = {
                    "ts": timestamp_ms,
                    # "readable_time": format_timestamp(timestamp_ms),  # 添加可读时间戳
                    "pid": pid,
                    "tid": tid,
                    "name": function_name,
                    "ph": "B",  # 开始事件
                    "args": {
                        "cookie": cookie,
                    }
                }
                events.append(event)
                
                # 更新活跃函数计数
                self.active_functions[process_name] = self.active_functions.get(process_name, 0) + 1
                
                # 记录函数调用次数指标
                FUNCTION_CALLS.labels(function=function_name, process=process_name).set(trace.call_count)
                
                # # 如果有函数统计数据，记录在摘要指标中
                # if trace.call_count > 0:
                #     FUNCTION_STATS_SUMMARY.labels(
                #         function=function_name, 
                #         process=process_name, 
                #         metric="call_count"
                #     ).set(trace.call_count)
                    
                if trace.avg_duration_us > 0:
                    FUNCTION_STATS_SUMMARY.labels(
                        function=function_name, 
                        process=process_name, 
                        metric="avg_duration_us"
                    ).set(trace.avg_duration_us)
            
            # 处理函数调用结束
            else:  # EXIT
                
                # 确保持续时间为正值且合理
                # if duration_ms <= 0:
                #     duration_ms = 0.001  # 最小显示为1μs
                    
                # 创建函数调用结束事件
                event = {
                    "ts": timestamp_ms,
                    # "readable_time": format_timestamp(timestamp_ms),
                    "pid": pid,
                    "tid": tid,
                    "name": function_name,
                    "ph": "E",  # 结束事件
                    "args": {
                        "cookie": cookie,
                        "call_count": trace.call_count,
                        "avg_duration_us": trace.avg_duration_us 
                    }
                }
                events.append(event)
                csv_rows.append([
                    timestamp_ms,
                    pid,
                    cookie,
                    trace.call_count,
                    trace.avg_duration_us
                ])
                
                # 更新活跃函数计数
                self.active_functions[process_name] = max(0, self.active_functions.get(process_name, 0) - 1)
                
                # # 记录函数持续时间直方图 (毫秒转换为秒)
                # FUNCTION_DURATION.labels(
                #     function=function_name, 
                #     process=process_name
                # ).observe(duration_ms / 1000)
                
                # 检测慢函数 (超过100ms的函数)
                # if duration_ms > 100:
                #     SLOW_FUNCTION_CALLS.labels(
                #         function=function_name, 
                #         process=process_name,
                #         threshold="100ms"
                #     ).inc()
                
                # 更新函数统计
                # self._update_function_stats(function_name, process_name,cookie)
                
                # 清理不再需要的映射
                # self.function_starts.pop(call_id, None)
        
        # 完成后更新活跃函数计数指标
        for process, count in self.active_functions.items():
            ACTIVE_FUNCTIONS.labels(process=process).set(count)
        
        # 写入事件到JSON文件
        self.write_events_to_json(events, is_last_batch)
        csv_headers = [
            'Timestamp', 'PID',
            'Cookie', 'Call Count',
             'AVG Duration(us)'
        ]
        self.write_rows_to_csv(csv_headers, csv_rows)
        
        return len(traces)
    
    # def _export_incremental_stats_to_csv(self, events:List[Dict[str,Any]]) -> None:
    #     """增量导出函数统计数据到CSV"""
    #     # 准备表头和数据
    #     events_headers = [
    #         'Timestamp', 'Process', 
    #         'Cookie', 'Call Count', 
    #         'Duration(us)', 'AVG Duration(us)'
    #     ]
        
    #     events_rows = []
    #     current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    #     for i, event in enumerate(events):
    #         if(event['ph'] == 'B'):
    #             continue
    #         row = [
    #             event['ts'],
    #             event['pid'],
    #             event['args']['cookie'],
    #             event['args']['call_count'],
    #             event['args']['duration_ms']*1000,
    #             event['args']['avg_duration_us']
    #         ]
    #         events_rows.append(row)
        
    #     # 导出统计数据
    #     if events_rows:
    #         self.write_rows_to_csv(events_headers, events_rows)
        
    def _update_function_stats(self, func_name: str, process_name: str, duration_ms: float, cookie: int) -> None:
        """更新函数调用统计"""
        key = func_name
        if key not in self.function_stats:
            self.function_stats[key] = {
                'process': process_name,
                'call_count': 0,
                'total_duration_ms': 0,
                'max_duration_ms': 0,
                'avg_duration_ms': 0
            }
        
        stats = self.function_stats[key]
        stats['cookie'] = cookie
        stats['call_count'] += 1
        stats['total_duration_ms'] += duration_ms
        stats['avg_duration_ms'] = stats['total_duration_ms'] / stats['call_count']
        
        if duration_ms > stats['max_duration_ms']:
            stats['max_duration_ms'] = duration_ms
            
            # 更新最大持续时间指标
            FUNCTION_STATS_SUMMARY.labels(
                function=func_name, 
                process=process_name, 
                metric="max_duration_seconds"
            ).set(duration_ms / 1000)  # 转换为秒
    
    def collect_stats(self) -> None:
        """汇总并记录函数统计数据"""
        # 按总执行时间排序找出前10个函数
        top_functions = {}
        
        for func_name, stats in sorted(
            self.function_stats.items(), 
            key=lambda x: x[1].get('total_duration_ms', 0), 
            reverse=True
        )[:10]:
            # 记录这些函数的更详细统计
            process = stats.get('process', 'unknown')
            call_count = stats.get('call_count', 0)
            avg_duration = stats.get('avg_duration_ms', 0) / 1000  # 转换为秒
            max_duration = stats.get('max_duration_ms', 0) / 1000  # 转换为秒
            
            # 保存以便日志和报告
            top_functions[func_name] = {
                'call_count': call_count,
                'avg_duration': avg_duration,
                'max_duration': max_duration
            }
        
        # 记录日志中的前5个最慢函数
        for i, (func, stats) in enumerate(list(top_functions.items())[:5]):
            logger.info(f"Top {i+1} function: {func} - calls: {stats['call_count']}, "
                    f"avg: {format_duration(stats['avg_duration']*1000000)}, "
                    f"max: {format_duration(stats['max_duration']*1000000)}")
    
    def finalize(self) -> None:
        """完成处理，执行清理工作"""
        self.collect_stats()

 