import logging
import json
from typing import Dict, List, Any, Optional

from utils.file_manager import TraceFileManager

logger = logging.getLogger('tracer-service')

class BaseTraceProcessor:
    def __init__(self, trace_type: str, file_manager: TraceFileManager):
        """
        初始化基础跟踪处理器
        
        参数:
        - trace_type: 跟踪类型标识符（如 'function', 'gpu'）
        - file_manager: 文件管理器实例
        """
        self.trace_type = trace_type
        self.file_manager = file_manager
        self.total_traces = 0
        self.file_manager.initialize_json(trace_type)
        logger.info(f"初始化了 {trace_type} 处理器")
        
    def process_traces(self, traces: List[Any], is_last_batch: bool = False) -> int:
        """
        处理跟踪数据
        
        参数:
        - traces: 跟踪数据列表
        - is_last_batch: 是否为最后一批数据
        
        返回:
        - 处理的跟踪数据数量
        """
        raise NotImplementedError("必须在子类中实现此方法")
    
    def finalize(self) -> None:
        """完成处理，执行任何必要的清理工作"""
        pass
    
    def write_events_to_json(self, events: List[Dict[str, Any]], is_last_batch: bool = False) -> None:
        """
        将事件写入JSON文件
        
        参数:
        - events: 事件字典列表
        - is_last_batch: 是否为最后一批数据
        """
        if not events:
            return
            
        # 按时间戳排序事件
        # sorted_events = sorted(events, key=lambda e: e.get("ts", 0))
        
        # 写入事件到JSON文件
        for i, event in enumerate(events):
            is_last = is_last_batch and i == len(events) - 1
            self.file_manager.write_event_to_json(self.trace_type, event, is_last)

    def write_rows_to_csv(self, headers: List[str], rows: List[List[Any]]) -> None:
        """
        批量写入事件到CSV文件（仿照JSON处理方式）

        参数:
        - headers: CSV表头
        - rows: 数据行列表
        """
        if not rows:
            return

        # 按时间戳排序（假设每行第一个字段为 ts）
        # sorted_rows = sorted(rows, key=lambda r: r[0] if r else 0)
        self.file_manager.write_rows_to_csv(self.trace_type, headers, rows)
        
