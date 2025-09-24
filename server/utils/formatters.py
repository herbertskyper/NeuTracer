import datetime
from typing import Union, Optional

def format_timestamp(timestamp_ms: float) -> str:
    """将毫秒时间戳转换为人类可读的时间格式"""
    dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0)
    return dt.strftime("%H:%M:%S.%f")[:-3]  # 格式化为 HH:MM:SS.mmm

def format_duration(duration_us: Optional[Union[float, int]]) -> str:
    """将微秒值格式化为人类可读的时间格式"""
    if duration_us is None:
        return "N/A"
    
    try:
        duration_us = float(duration_us)
        if duration_us < 1000:
            return f"{duration_us:.2f}us"
        elif duration_us < 1000000:
            return f"{duration_us/1000:.2f}ms"
        else:
            return f"{duration_us/1000000:.3f}s"
    except (ValueError, TypeError):
        return str(duration_us)

def parse_timestamp(timestamp_str: str) -> float:
    """解析各种格式的时间戳字符串为毫秒级时间戳"""
    try:
        if isinstance(timestamp_str, str) and not timestamp_str.replace('.', '', 1).isdigit():
            time_obj = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            return time_obj.timestamp() * 1000
        else:
            return float(timestamp_str) * 1000
    except (ValueError, TypeError):
        return datetime.datetime.now().timestamp() * 1000

def format_bytes(bytes_value: Optional[Union[float, int]]) -> str:
    """将字节数格式化为人类可读的形式（KB, MB, GB等）"""
    if bytes_value is None:
        return "N/A"
    
    try:
        bytes_value = float(bytes_value)
        if bytes_value < 1024:
            return f"{bytes_value:.0f}B"
        elif bytes_value < 1024 * 1024:
            return f"{bytes_value/1024:.2f}KB"
        elif bytes_value < 1024 * 1024 * 1024:
            return f"{bytes_value/(1024*1024):.2f}MB"
        elif bytes_value < 1024 * 1024 * 1024 * 1024:
            return f"{bytes_value/(1024*1024*1024):.2f}GB"
        else:
            return f"{bytes_value/(1024*1024*1024*1024):.2f}TB"
    except (ValueError, TypeError):
        return str(bytes_value)