import threading
import time
import logging
from threading import Event
from typing import Callable, Optional

from prometheus_metrics import INACTIVITY_TIME

logger = logging.getLogger('tracer-service')

class ActivityMonitor:
    def __init__(self, inactivity_timeout: int, shutdown_event: Event, last_activity_time_getter: Callable[[], float]):
        """
        初始化活动监控器
        
        参数:
        - inactivity_timeout: 不活动超时时间（秒）
        - shutdown_event: 关机事件，在超时时设置
        - last_activity_time_getter: 获取最后活动时间的回调函数
        """
        self.inactivity_timeout = inactivity_timeout
        self.shutdown_event = shutdown_event
        self.get_last_activity_time = last_activity_time_getter
        self.monitor_thread: Optional[threading.Thread] = None
        self.active = True
        
    def start(self) -> None:
        """启动监控线程"""
        if self.monitor_thread is not None:
            return
            
        self.monitor_thread = threading.Thread(target=self._monitor_activity, daemon=True)
        self.monitor_thread.name = "InactivityMonitor"
        self.monitor_thread.start()
        logger.info(f"不活动监控线程已启动，超时时间: {self.inactivity_timeout} 秒")
        
    def stop(self) -> None:
        """停止监控线程"""
        self.active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        
    def _monitor_activity(self) -> None:
        """监控活动的线程函数"""
        while self.active:
            time.sleep(30)  # 每30秒检查一次
            
            if not self.active:  # 如果服务已被标记为不活动，退出循环
                break
                
            current_time = time.time()
            last_activity_time = self.get_last_activity_time()
            elapsed = current_time - last_activity_time
            
            # 更新不活动时间指标
            INACTIVITY_TIME.set(elapsed)
            
            if elapsed > self.inactivity_timeout:
                logger.warning(f"服务已 {elapsed:.1f} 秒未收到消息，超过 {self.inactivity_timeout} 秒限制，准备关闭")
                self.active = False
                self.shutdown_event.set()  # 通知主线程关闭服务器
                break


class StatsReporter:
    def __init__(self, report_func: Callable[[], None], interval: int = 60):
        """
        初始化统计报告器
        
        参数:
        - report_func: 报告统计信息的回调函数
        - interval: 报告间隔（秒）
        """
        self.report_func = report_func
        self.interval = interval
        self.reporter_thread: Optional[threading.Thread] = None
        self.active = True
        
    def start(self) -> None:
        """启动报告线程"""
        if self.reporter_thread is not None:
            return
            
        self.reporter_thread = threading.Thread(target=self._report_stats, daemon=True)
        self.reporter_thread.name = "StatsReporter"
        self.reporter_thread.start()
        
    def stop(self) -> None:
        """停止报告线程"""
        self.active = False
        if self.reporter_thread:
            self.reporter_thread.join(timeout=1.0)
            self.reporter_thread = None
        
    def _report_stats(self) -> None:
        """报告统计信息的线程函数"""
        while self.active:
            time.sleep(self.interval * 60)  # 默认每60分钟报告一次
            if self.active:  # 再次检查，以防在睡眠期间被停止
                self.report_func()