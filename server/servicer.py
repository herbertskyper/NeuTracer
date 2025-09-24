import os
import time
import logging
import threading
from typing import Dict, Set, Optional, IO

import grpc
import tracer_service_pb2
import tracer_service_pb2_grpc

from utils.file_manager import TraceFileManager
from utils.monitoring import ActivityMonitor, StatsReporter
from trace_processor.function_processor import FunctionTraceProcessor
from trace_processor.gpu_processor import GPUTraceProcessor
from trace_processor.cpu_processor import CPUTraceProcessor
from trace_processor.io_processor import IOTraceProcessor
from trace_processor.kmem_processor import MemTraceProcessor
from trace_processor.net_process import NetworkTraceProcessor
from prometheus_metrics import ACTIVE_CLIENTS, PROCESSING_TIME

logger = logging.getLogger('tracer-service')


class TracerServicer(tracer_service_pb2_grpc.TracerServiceServicer):
    def __init__(self, inactivity_timeout: int = 3600) -> None:
        # 初始化TracerService对象
        self.active: bool = True  # 设置服务为活跃状态
        self.total_traces: int = 0
        self.clients: Set[str] = set()
        self.version: str = "1.0.0"
        self.last_activity_time: float = time.time()  # 记录最后活动时间
        self.inactivity_timeout: int = inactivity_timeout  # 默认超时时间（秒）
        self.shutdown_event = threading.Event()  # 用于通知主线程关闭服务器
        self.pid_list = {}  
        
        # 初始化文件管理器
        self.file_manager = TraceFileManager()
        
        # 初始化各种处理器
        self.function_processor = FunctionTraceProcessor(self.file_manager)
        self.gpu_processor = GPUTraceProcessor(self.file_manager)
        self.cpu_processor = CPUTraceProcessor(self.file_manager)
        self.io_processor = IOTraceProcessor(self.file_manager)  
        self.kmem_processor = MemTraceProcessor(self.file_manager) 
        self.net_processor = NetworkTraceProcessor(self.file_manager)
        # 启动监控线程
        self.activity_monitor = ActivityMonitor(
            inactivity_timeout, 
            self.shutdown_event,
            lambda: self.last_activity_time
        )
        self.activity_monitor.start()
        
        # 启动统计报告定时器
        self.stats_reporter = StatsReporter(self._collect_stats)
        self.stats_reporter.start()
        global CPU_STATS_ENABLED
        CPU_STATS_ENABLED = False  # 默认不启用CPU采样
        if(CPU_STATS_ENABLED == True):
            # 启动CPU采样定时器
            self._cpu_sampling_thread = threading.Thread(target=self._cpu_sampling_loop, daemon=True)
            self._cpu_sampling_thread.start()


        logger.info(f"TracerService initialized with inactivity timeout: {self.inactivity_timeout} seconds")
    
    def __del__(self) -> None:
        """析构函数 - 在对象被垃圾回收时调用"""
        self.close_all_files()
        self.activity_monitor.stop()
        self.stats_reporter.stop()

    
    def close_all_files(self) -> None:
        """关闭所有日志文件"""
        self.file_manager.close_all()
        if(CPU_STATS_ENABLED == True):
            self.active = False  # 通知CPU采样线程退出
            if hasattr(self, "_cpu_sampling_thread") and self._cpu_sampling_thread.is_alive():
                self._cpu_sampling_thread.join(timeout=5)
    
    def _update_activity_time(self) -> None:
        """更新最后活动时间"""
        self.last_activity_time = time.time()
    
    def _collect_stats(self) -> None:
        """收集所有处理器的统计数据"""
        pass
        # try:
        #     self.function_processor.collect_stats()
        #     self.cpu_processor.collect_stats()
        #     # 其他处理器的统计收集
        # except Exception as e:
        #     logger.error(f"收集统计数据时出错: {str(e)}")
            
    def _cpu_sampling_loop(self):
        """每分钟调用一次CPU采样函数"""
        while self.active:
            try:
                pid_set = set()
                pid_set.update(self.kmem_processor.process_stats.keys())
                self.cpu_processor.cpu_stat_record(list(pid_set))
            except Exception as e:
                logger.error(f"CPU采样定时器出错: {str(e)}")
            time.sleep(60)  # 每60秒采样一次
    
    def SendTraceBatch(self, request: tracer_service_pb2.TraceBatch, context: grpc.ServicerContext) -> tracer_service_pb2.TraceResponse:
        # 更新最后活动时间
        self._update_activity_time()
        
        start_time: float = time.time()
        try:
            # 更新客户端列表
            client_id: str = context.peer()
            # logger.info(f"Current clients: {self.clients}")
            if client_id not in self.clients:
                self.clients.add(client_id)
                ACTIVE_CLIENTS.set(len(self.clients))
            
            # 处理函数调用跟踪数据
            if request.trace_data:
                self.function_processor.process_traces(request.trace_data)
                
            # 处理CPU跟踪数据 
            if request.cpu_trace_data:
                global CPU_STATS_ENABLED
                if(CPU_STATS_ENABLED == False):
                    self.cpu_processor.process_traces(request.cpu_trace_data)
            
            if request.io_trace_data:
                self.io_processor.process_traces(request.io_trace_data)
            
            if request.memory_trace_data:
                self.kmem_processor.process_traces(request.memory_trace_data)
            
            if request.network_trace_data:
                self.net_processor.process_traces(request.network_trace_data)
            
            # 处理GPU跟踪数据
            if request.gpu_trace_data:
                self.gpu_processor.process_traces(request.gpu_trace_data)
            
            # 记录处理时间
            processing_time: float = time.time() - start_time
            PROCESSING_TIME.observe(processing_time)
            
            return tracer_service_pb2.TraceResponse(success=True, message="Batch processed successfully")
        
        except Exception as e:
            logger.error(f"处理批次时出错: {str(e)}", exc_info=True)
            return tracer_service_pb2.TraceResponse(success=False, message=f"Error: {str(e)}")
    
    def GetStatus(self, request: tracer_service_pb2.StatusRequest, context: grpc.ServicerContext) -> tracer_service_pb2.StatusResponse:
        # 更新最后活动时间
        self._update_activity_time()
        
        # 更新客户端列表
        client_id: str = request.client_id if request.client_id else context.peer()
        if client_id not in self.clients:
            self.clients.add(client_id)
            ACTIVE_CLIENTS.set(len(self.clients))
        
        # 计算不活动时间
        inactive_time = time.time() - self.last_activity_time
        
        return tracer_service_pb2.StatusResponse(
            active=self.active,
            received_traces=self.total_traces,
            server_version=self.version,
            message=f"服务正常，不活动时间: {inactive_time:.1f}秒，超时设置: {self.inactivity_timeout}秒"
        )