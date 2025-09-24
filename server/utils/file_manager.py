import os
import time
import logging
import csv
from datetime import datetime
from typing import Any, List
from typing import Dict, IO, Optional

logger = logging.getLogger('tracer-service')

class TraceFileManager:
    def __init__(self):
        """初始化跟踪文件管理器"""
        # 创建日志目录
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.json_dir = os.path.join(self.output_dir, "json")
        os.makedirs(self.json_dir, exist_ok=True)

        self.csv_dir = os.path.join(self.output_dir, "csv")
        os.makedirs(self.csv_dir, exist_ok=True)

        self.txt_dir = os.path.join(self.output_dir, "txt")
        os.makedirs(self.txt_dir, exist_ok=True)
        
        
        self.initialize_stack_message_file()
        # 文件句柄字典
        self.json_files: Dict[str, IO[str]] = {}
        # CSV会话字典
        self.csv_files: Dict[str, IO[str]] = {}
    
        
        # 初始化标志
        self._initialized_json = set()
        self._initialized_csv = set()
        
    def initialize_stack_message_file(self) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        hostname = os.uname().nodename
        self.firstime = None
        # 构建文件路径
        file_name = f"stackmessage_{timestamp}_{hostname}.txt"
        file_path = os.path.join(self.txt_dir, file_name)
        
        # 打开文件准备写入
        self.stack_file = open(file_path, "w")
        logger.info(f"创建了调用栈文件: {file_path}")
        
        # 写入标题头
        self.stack_file.write("FORMAT|1.0\n")
        self.stack_file.write("METADATA|\n")
        self.stack_file.write("HEADER|timestamp|pid|frame_count|[frames...]\n")
    
    def get_json_file(self, trace_type: str) -> Optional[IO[str]]:
        """获取特定类型的跟踪文件，如果不存在则创建"""
        if trace_type not in self.json_files:
            timestamp = time.strftime("%Y%m%d_%H%M%S")            
            file_path = os.path.join(self.json_dir, f"{trace_type}_trace_{timestamp}.json")
            self.json_files[trace_type] = open(file_path, "w")
            logger.info(f"创建了 {trace_type} 跟踪文件: {file_path}")
            
        return self.json_files.get(trace_type)
    
    def initialize_json(self, trace_type: str) -> None:
        """初始化JSON文件头部"""
        if trace_type in self._initialized_json:
            return
            
        file = self.get_json_file(trace_type)
        if file and file.tell() == 0:
            file.write('{\n  "traceEvents": [\n')
            self._initialized_json.add(trace_type)
    
    def write_event_to_json(self, trace_type: str, event: dict, is_last: bool = False) -> None:
        """向JSON文件写入事件"""
        file = self.get_json_file(trace_type)
        if not file:
            return
            
        # 确保文件已初始化
        self.initialize_json(trace_type)
        
        # 写入事件
        import json
        json_line = json.dumps(event)
        file.write(json_line)
        
        if not is_last:
            file.write(",\n")
        else:
            file.write("\n")
        
        file.flush()
    
    # def close_all(self) -> None:
    #     """关闭所有文件"""
    #     for name, file in list(self.json_files.items()):
    #         try:
    #             if name in self._initialized_json:
    #                 # 添加JSON结束标记
    #                 file.write("\n]}\n")
    #             file.close()
    #             logger.info(f"关闭了 {name} 跟踪文件")
    #             self.json_files.pop(name, None)
    #         except Exception as e:
    #             logger.error(f"关闭 {name} 跟踪文件时出错: {str(e)}")

    #     self.stack_file.close()
    #     logger.info("关闭了调用栈文件")

    def get_csv_file(self, trace_type: str, headers: List[str]) -> Optional[IO[str]]:
        """获取或创建CSV文件，如果不存在则创建并写入表头"""
        if trace_type not in self.csv_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.csv_dir, f"{trace_type}_trace_{timestamp}.csv")
            file = open(file_path, "w", newline='')

            import csv
            writer = csv.writer(file)
            writer.writerow(headers)
            self.csv_files[trace_type] = file
            self._initialized_csv.add(trace_type)
            logger.info(f"创建了 {trace_type} CSV文件: {file_path}")

        return self.csv_files.get(trace_type)

    def write_row_to_csv(self, trace_type: str, headers: List[str], row: List[Any]) -> None:
        """写入一行数据到CSV文件，自动初始化表头"""
        file = self.get_csv_file(trace_type, headers)
        if not file:
            return
        import csv
        writer = csv.writer(file)
        writer.writerow(row)
        file.flush()

    def write_rows_to_csv(self, trace_type: str, headers: List[str], rows: List[List[Any]]) -> None:
        """批量写入多行数据到CSV文件，自动初始化表头"""
        file = self.get_csv_file(trace_type, headers)
        if not file:
            return
        import csv
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)
        file.flush()
            
    def close_all(self) -> None:
        """
        关闭所有文件，并完成所有CSV会话（可选写入摘要信息）
        """
        for name, file in list(self.csv_files.items()):
            try:
                file.close()
                logger.info(f"关闭了 {name} CSV文件")
                self.csv_files.pop(name, None)
            except Exception as e:
                logger.error(f"关闭 {name} CSV文件时出错: {str(e)}")
        self._initialized_csv.clear()


        # 关闭所有JSON文件
        for name, file in list(self.json_files.items()):
            try:
                if name in self._initialized_json:
                    # 添加JSON结束标记
                    file.write("\n]}\n")
                file.close()
                logger.info(f"关闭了 {name} 跟踪文件")
                self.json_files.pop(name, None)
            except Exception as e:
                logger.error(f"关闭 {name} 跟踪文件时出错: {str(e)}")

        # 关闭调用栈
        self.stack_file.close()
        logger.info("关闭了调用栈文件")         

