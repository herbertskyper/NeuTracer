#!/usr/bin/env python3
import sys
import argparse
import random
import os
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union

# 直接导入编译好的 protobuf 模块
import protos.perfetto.trace.perfetto_trace_pb2 as perfetto_trace_pb2  # 确保此文件在当前目录中


def parse_standard_format(
    file_path: str,
) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    """
    解析标准格式的输出文件

    Args:
        file_path: 输入文件路径

    Returns:
        元组，包含 (pid, events, metadata)
    """
    events: List[Dict[str, Any]] = []
    pid: Optional[int] = None
    metadata: Dict[str, Any] = {}

    with open(file_path, "r") as f:
        lines = f.readlines()

    # 解析元数据
    for line in lines[:10]:  # 只检查前几行
        if line.startswith("# Target PID:"):
            pid = int(line.split(":")[1].strip())
            metadata["pid"] = pid
        elif line.startswith("# Sampling Rate:"):
            rate = int(line.split(":")[1].split("Hz")[0].strip())
            metadata["sampling_rate"] = rate

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "EVENT":
            event: Dict[str, Any] = {}
            i += 1
            frames: List[Dict[str, str]] = []

            while i < len(lines) and lines[i].strip() != "END":
                line = lines[i].strip()
                if line.startswith("timestamp="):
                    event["timestamp_ns"] = int(line.split("=")[1])
                elif line.startswith("pid="):
                    event["pid"] = int(line.split("=")[1])
                    if pid is None:
                        pid = event["pid"]
                elif line.startswith("frames="):
                    event["frames_count"] = int(line.split("=")[1])
                elif line.startswith("frame"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        frame_data = parts[1].split(":", 1)
                        if len(frame_data) == 2:
                            filename, funcname = frame_data
                            frames.append({"filename": filename, "function": funcname})
                i += 1

            event["frames"] = frames
            events.append(event)
        i += 1

    if pid is None:
        raise ValueError("未能解析到PID")

    return pid, events, metadata


def parse_structured_format(
    file_path: str,
) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    """
    解析结构化格式的输出文件

    Args:
        file_path: 输入文件路径

    Returns:
        元组，包含 (pid, events, metadata)
    """
    events: List[Dict[str, Any]] = []
    pid: Optional[int] = None
    metadata: Dict[str, Any] = {}

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")

        if parts[0] == "METADATA":
            for item in parts[1:]:
                if "=" in item:
                    key, value = item.split("=")
                    if key == "pid":
                        pid = int(value)
                        metadata["pid"] = pid
                    elif key == "rate":
                        metadata["sampling_rate"] = int(value)

        elif parts[0] == "EVENT" and len(parts) >= 4:
            timestamp = int(parts[1])
            current_pid = int(parts[2])
            frames_count = int(parts[3])

            if pid is None:
                pid = current_pid

            frames: List[Dict[str, str]] = []
            for i in range(4, len(parts)):
                if ":" in parts[i]:
                    filename, funcname = parts[i].split(":", 1)
                    frames.append({"filename": filename, "function": funcname})

            # 反转frames列表，使调用栈从下到上排列（调用的在低索引，被调用的在高索引）
            frames.reverse()

            events.append(
                {
                    "timestamp_ns": timestamp,
                    "pid": current_pid,
                    "frames_count": frames_count,
                    "frames": frames,
                }
            )

    if pid is None:
        raise ValueError("未能解析到PID")

    return pid, events, metadata


class FunctionCall:
    """表示一个函数调用，包含时间戳和持续时间信息"""

    def __init__(self, name: str, start_time: int):
        self.name = name
        self.start_time = start_time
        self.end_time: Optional[int] = None
        self.duration: Optional[int] = None

    def end(self, time: int):
        """设置函数调用结束时间"""
        self.end_time = time
        self.duration = self.end_time - self.start_time

    def __str__(self) -> str:
        return f"{self.name} [{self.start_time} - {self.end_time}] ({self.duration}ns)"


def generate_binary_perfetto_trace(
    pid: int, events: List[Dict[str, Any]], metadata: Dict[str, Any], output_file: str
) -> None:
    """
    生成二进制格式的Perfetto trace文件，实时生成事件保证顺序
    
    Args:
        pid: 进程ID
        events: 事件列表（必须已按时间戳排序）
        metadata: 元数据字典
        output_file: 输出文件路径
    """
    # 创建根跟踪消息
    trace = perfetto_trace_pb2.Trace()

    # 生成唯一ID（全trace保持一致）
    process_uuid: int = random.randint(1, 1<<63)
    sequence_id: int = random.randint(1, 1<<31)
    process_name: str = f"Python Process {pid}"

    # 添加进程描述符（必须第一个packet）
    packet = trace.packet.add()
    track_desc = packet.track_descriptor
    track_desc.uuid = process_uuid
    track_desc.process.pid = pid
    track_desc.process.process_name = process_name

    # 线程跟踪上下文
    threads: Dict[int, Dict[str, Any]] = {}

    # 必须确保事件已按时间戳排序
    events.sort(key=lambda e: e["timestamp_ns"])

    # 从protobuf获取事件类型常量
    TYPE_SLICE_BEGIN = perfetto_trace_pb2.TrackEvent.TYPE_SLICE_BEGIN
    TYPE_SLICE_END = perfetto_trace_pb2.TrackEvent.TYPE_SLICE_END

    # 预注册所有线程（确保描述符在事件前发出）
    seen_tids = set()
    for event in events:
        tid = event.get("tid", pid)
        if tid not in seen_tids:
            seen_tids.add(tid)
            thread_uuid = random.randint(1, 1<<63)
            threads[tid] = {
                "uuid": thread_uuid,
                "name": f"Thread {tid}",
                "call_stack": [],  # 当前调用栈（存储函数名）
            }
            
            # 发出线程描述符
            packet = trace.packet.add()
            track_desc = packet.track_descriptor
            track_desc.uuid = thread_uuid
            track_desc.thread.pid = pid
            track_desc.thread.tid = tid
            track_desc.thread.thread_name = threads[tid]["name"]

    # 处理每个事件（必须严格按时间顺序）
    for event in events:
        tid = event.get("tid", pid)
        thread_info = threads[tid]
        call_stack = thread_info["call_stack"]
        thread_uuid = thread_info["uuid"]
        base_ts = event["timestamp_ns"]
        current_ts = base_ts  # 用于同一事件内的时间递增
        
        # 解析当前调用栈
        current_frames = [f"{f['filename']}:{f['function']}" for f in event["frames"]]
        
        # 计算与当前调用栈的共同前缀长度
        common_len = 0
        while (common_len < len(call_stack) and 
               common_len < len(current_frames) and 
               call_stack[common_len] == current_frames[common_len]):
            common_len += 1

        # 生成END事件（后进先出）
        for i in reversed(range(common_len, len(call_stack))):
            # 生成SLICE_END
            packet = trace.packet.add()
            packet.timestamp = current_ts
            packet.trusted_packet_sequence_id = sequence_id
            te = packet.track_event
            te.type = TYPE_SLICE_END
            te.track_uuid = thread_uuid
            current_ts += 1  # 确保时间戳严格递增
        
        # 更新调用栈状态
        del call_stack[common_len:]
        
        # 生成BEGIN事件（顺序添加）
        for i in range(common_len, len(current_frames)):
            func_name = current_frames[i]
            # 生成SLICE_BEGIN
            packet = trace.packet.add()
            packet.timestamp = current_ts
            packet.trusted_packet_sequence_id = sequence_id
            te = packet.track_event
            te.type = TYPE_SLICE_BEGIN
            te.name = func_name
            te.track_uuid = thread_uuid
            current_ts += 1  # 递增时间戳
        
            call_stack.append(func_name)

    # 关闭所有剩余调用（使用最终时间戳+增量）
    final_ts = events[-1]["timestamp_ns"] + 1 if events else 0
    for tid, thread_info in threads.items():
        call_stack = thread_info["call_stack"]
        while call_stack:
            packet = trace.packet.add()
            packet.timestamp = final_ts
            packet.trusted_packet_sequence_id = sequence_id
            te = packet.track_event
            te.type = TYPE_SLICE_END
            te.track_uuid = thread_info["uuid"]
            final_ts += 1  # 保持递增
            call_stack.pop()

    # 写入二进制文件
    with open(output_file, "wb") as f:
        f.write(trace.SerializeToString())

    print(f"Perfetto trace generated: {output_file}")

def main() -> None:
    """主函数，处理命令行参数并调用相应的函数"""
    parser = argparse.ArgumentParser(description="将栈跟踪输出转换为二进制Perfetto格式")
    parser.add_argument("input_file", help="包含栈跟踪数据的输入文件")
    parser.add_argument(
        "-o", "--output", required=True, help="输出Perfetto二进制trace文件"
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        default=-1,
        help="输入格式 (0=标准, 1=结构化, 默认: 自动检测)",
    )
    args = parser.parse_args()

    # 自动检测格式
    if args.format == -1:
        with open(args.input_file, "r") as f:
            first_line = f.readline().strip()
            if first_line.startswith("FORMAT|"):
                args.format = 1
            else:
                args.format = 0

    # 根据格式解析文件
    if args.format == 0:
        pid, events, metadata = parse_standard_format(args.input_file)
    else:
        pid, events, metadata = parse_structured_format(args.input_file)

    # 生成二进制Perfetto trace
    generate_binary_perfetto_trace(pid, events, metadata, args.output)


if __name__ == "__main__":
    main()
