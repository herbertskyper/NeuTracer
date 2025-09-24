import logging
import time
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
from utils.cpu_utils import bfs_get_procs, pid_to_comm

from trace_processor.base_processor import BaseTraceProcessor
from utils.formatters import parse_timestamp, format_duration
from utils.file_manager import TraceFileManager
from prometheus_metrics import (
    CPU_UTILIZATION, CPU_ONCPU_TIME, CPU_OFFCPU_TIME, 
    CPU_MIGRATION_COUNT,
    CPU_NUMA_MIGRATION_COUNT
)

logger = logging.getLogger('tracer-service')


class CPUTraceProcessor(BaseTraceProcessor):
    """CPU跟踪处理器，主要负责从C++端接收已分析的CPU数据，转换为Prometheus指标"""
    
    def __init__(self, file_manager: TraceFileManager):
        """初始化CPU跟踪处理器"""
        super().__init__("cpu", file_manager)
        
        # 进程统计数据 - 保存从C++端接收的数据
        self.process_stats: Dict[int, Dict[str, Any]] = {}
        
        # 跟踪开始时间
        self.start_time = time.time()

        # logger.info("CPU追踪器初始化成功")
    
    def process_traces(self, traces: List[Any], is_last_batch: bool = False) -> int:
        """处理CPU跟踪数据（重构版）"""
        if not traces:
            return 0

        # 1. 按时间排序
        sorted_traces = sorted(traces, key=lambda e: getattr(e, "timestamp", 0))

        events = []
        csv_rows = []
        csv_headers = [
            'Timestamp', 'PID', 'CPU Utilization(%)', 'OnCPU Time(μs)', 'OffCPU Time(μs)',
            'Migrations', 'NUMA Migrations', 'Hotspot CPU Percentage(%)'
        ]

        for trace in sorted_traces:
            # 2.1 更新进程结构体
            event = self._process_single_trace(trace)
            if not event:
                continue

            pid = trace.pid
            args = event["args"]
            timestamp_ms = event["ts"]

            # 2.2 构造JSON数据
            events.append(event)

            # 2.3 构造CSV数据
            csv_rows.append([
                timestamp_ms,
                pid,
                args["utilization"],
                args["oncpu_time_us"],
                args["offcpu_time_us"],
                args["migrations"],
                args["numa_migrations"],
                args["hotspot_percentage"],
            ])

        # 3. 写入json和csv
        self.write_events_to_json(events, is_last_batch)
        if csv_rows:
            self.write_rows_to_csv(csv_headers, csv_rows)

        # 4. 更新Prometheus指标
        self._update_prometheus_metrics()

        return len(events)
    
    def _process_single_trace(self, trace) -> None:
        """处理单个CPU跟踪事件，不再进行异常检测，仅存储和转换C++已处理的数据"""
        try:
            # 提取基本信息
            pid = trace.pid
            ppid = trace.ppid
            timestamp = trace.timestamp
            comm = trace.comm
            cpu_id = trace.cpu_id
            oncpu_time = trace.oncpu_time
            offcpu_time = trace.offcpu_time
            utilization = trace.utilization
            migrations_count = trace.migrations_count
            numa_migrations = trace.numa_migrations
            hotspot_cpu = trace.hotspot_cpu
            hotspot_percentage = trace.hotspot_percentage
            
            # 如果进程不存在于缓存中，初始化它
            if pid not in self.process_stats:
                self.process_stats[pid] = {
                    'comm': comm,
                    'oncpu_time_total': 0,
                    'offcpu_time_total': 0,
                    'samples_count': 0,
                    'utilization': 0,
                    'migrations_count': migrations_count,
                    'numa_migrations': numa_migrations,
                    'hotspot_cpu': hotspot_cpu,
                    'hotspot_percentage': hotspot_percentage,
                    'last_seen': timestamp
                }
            
            # 更新进程统计 - 主要用于绘制时间序列图表
            stats = self.process_stats[pid]
            stats['oncpu_time_total'] = oncpu_time
            stats['offcpu_time_total'] = offcpu_time
            stats['samples_count'] += 1
            stats['utilization'] = utilization
            stats['last_seen'] = timestamp
            
            stats['migrations_count'] = migrations_count
            stats['numa_migrations'] = numa_migrations
            stats['hotspot_cpu'] = hotspot_cpu
            stats['hotspot_percentage'] = hotspot_percentage
            
            # 创建JSON事件对象
            timestamp_ms = parse_timestamp(timestamp) if timestamp else time.time() * 1000
            
            # 创建CPU使用事件
            event = {
                "name": f"{comm}",
                "ts": timestamp_ms,
                "pid": pid,
                "args": {
                    "cpu_id": cpu_id,
                    "utilization": f"{utilization:.2f}",
                    "oncpu_time_us": oncpu_time,
                    "offcpu_time_us": offcpu_time,
                    "migrations": stats['migrations_count'],
                    "numa_migrations": stats['numa_migrations'],
                    "hotspot_cpu": stats['hotspot_cpu'],
                    "hotspot_percentage": f"{stats['hotspot_percentage']:.1f}",
                }
            }
            
            return event
            
        except Exception as e:
            logger.error(f"Error processing CPU trace: {str(e)}", exc_info=True)
    
    def _update_prometheus_metrics(self) -> None:
        """更新Prometheus指标"""
        # 对于每个进程更新指标
        for pid, stats in self.process_stats.items():
            comm = stats['comm']
            labels = {'pid': str(pid), 'process': comm}
            
            # 更新基本指标
            CPU_UTILIZATION.labels(**labels).set(stats['utilization'] / 100.0)  # 转换为0-100%
            CPU_ONCPU_TIME.labels(**labels).set(stats['oncpu_time_total'])
            CPU_OFFCPU_TIME.labels(**labels).set(stats['offcpu_time_total'])
            
            # 更新迁移和调度指标
            CPU_MIGRATION_COUNT.labels(**labels).set(stats['migrations_count'])
            CPU_NUMA_MIGRATION_COUNT.labels(**labels).set(stats['numa_migrations'])
            
    # def _export_incremental_stats_to_csv(self, events: List[Dict[str,Any]]) -> None:
    #     """增量导出CPU统计数据到CSV（调用 write_rows_to_csv）"""
    #     # 准备表头和数据
    #     events_headers = [
    #         'Timestamp', 'PID', 
    #         'CPU Utilization(%)', 'OnCPU Time(μs)', 'OffCPU Time(μs)',
    #         'Migrations', 'NUMA Migrations','Hotspot CPU Percentage(%)'
    #     ]

    #     events_rows = []
    #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #     for i, event in enumerate(events):

    #         row = [
    #             event['ts'],
    #             event['pid'],
    #             event['args']['utilization'],
    #             event['args']['oncpu_time_us'],
    #             event['args']['offcpu_time_us'],
    #             event['args']['migrations'],
    #             event['args']['numa_migrations'],
    #             event['args']['hotspot_percentage'],
    #         ]
    #         events_rows.append(row)

    #     # 批量写入到CSV
    #     if events_rows:
    #         self.write_rows_to_csv(events_headers, events_rows)
    
    def cpu_stat_record(self, pid_list:List) -> None:
        for pid in pid_list:
            self.cpu_stat_record_to_csv(pid)

    def cpu_stat_record_to_csv(self, snoop_pid: int, period: float = 60.0, old_usage: Dict[int, Dict[str, Any]] = None) -> Dict[int, Dict[str, Any]]:
        """
        读取进程CPU信息并写入CSV
        :param file_manager: TraceFileManager实例
        :param snoop_pid: 监控的主进程PID
        :param old_usage: 上一次采样的CPU使用情况
        :param period: 采样周期（秒）
        :return: 当前采样的CPU使用情况
        """
        pid_list = bfs_get_procs(snoop_pid)
        proc_cpu_usages = {}
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        headers = ['Timestamp', 'PID', 'Process','User Time(s)', 'System Time(s)','Child User Time(s)', 'Child System Time(s)','OnCPU Time(s)', 'OffCPU Time(s)', 'Utilization(%)']
        rows = []
        import os
        clk_tck = os.sysconf(os.sysconf_names['SC_CLK_TCK'])

        for pid in pid_list:
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    data = f.read().split(" ")
                    utime = int(data[13])
                    stime = int(data[14])
                    cutime = int(data[15])
                    cstime = int(data[16])
                    run_time = utime + stime + cutime + cstime
            except Exception:
                continue

            cpu_usage = {
                "run_time": run_time / clk_tck,
                "total_time": period,
            }

            if old_usage is not None and pid in old_usage:
                cpu_usage["oncpu_time"] = cpu_usage["run_time"] - old_usage[pid]["run_time"]
                cpu_usage["offcpu_time"] = period - cpu_usage["oncpu_time"]
                cpu_usage["utilization"] = (
                    cpu_usage["oncpu_time"] / (cpu_usage["oncpu_time"] + cpu_usage["offcpu_time"])
                    if (cpu_usage["oncpu_time"] + cpu_usage["offcpu_time"]) > 0 else 0
                )
                rows.append([
                    cur_time,
                    pid,
                    pid_to_comm(pid),
                    utime,
                    stime,
                    cutime,
                    cstime,
                    f"{cpu_usage['oncpu_time']:.2f}",
                    f"{cpu_usage['offcpu_time']:.2f}",
                    f"{cpu_usage['utilization'] * 100:.2f}"
                ])
            proc_cpu_usages[pid] = cpu_usage

        # 写入CSV
        if rows:
            self.file_manager.write_rows_to_csv('cpu_top',headers, rows)

        return proc_cpu_usages
    
    def collect_stats(self) -> None:
        """收集并记录CPU统计信息"""
        
        # 记录最高CPU使用率的进程
        top_cpu = sorted(
            self.process_stats.items(), 
            key=lambda x: x[1]['oncpu_time_total'], 
            reverse=True
        )[:5]
        
        logger.info("Top CPU consumers:")
        for pid, stats in top_cpu:
            avg_util = stats['utilization_sum'] / stats['samples_count'] if stats['samples_count'] > 0 else 0
            logger.info(f"  {stats['comm']} (PID {pid}): {avg_util/100:.2f}% CPU, "
                       f"{format_duration(stats['oncpu_time_total'])} total, "
                       f"{stats['migrations_count']} migrations")

    
    def finalize(self) -> None:
        """完成处理，执行清理工作"""
        # self.collect_stats()

