import logging
import time
from typing import Dict, List, Any, Set
from collections import defaultdict
from datetime import datetime

from trace_processor.base_processor import BaseTraceProcessor
from utils.formatters import parse_timestamp, format_bytes
from utils.file_manager import TraceFileManager
from prometheus_metrics import (
    NET_TX_BYTES_RATE, NET_RX_BYTES_RATE, NET_TX_PACKETS_RATE, NET_RX_PACKETS_RATE,
    NET_TOTAL_TX_BYTES, NET_TOTAL_RX_BYTES, NET_TOTAL_TX_PACKETS, NET_TOTAL_RX_PACKETS,
    NET_ACTIVE_CONNECTIONS, NET_LISTENING_PORTS
)
## sum(rate(neutracer_batch_size_bucket[5m])) by (le) 服务端处理数据批次的时间直方图
#

logger = logging.getLogger('tracer-service')


class NetworkTraceProcessor(BaseTraceProcessor):
    """网络跟踪处理器，负责从C++端接收已分析的网络数据，转换为Prometheus指标"""
    
    def __init__(self, file_manager: TraceFileManager):
        """初始化网络跟踪处理器"""
        super().__init__("network", file_manager)
        
        # 进程网络统计数据
        self.process_stats: Dict[int, Dict[str, Any]] = {}
        
        # 连接跟踪
        self.connections: Dict[str, Dict[str, Any]] = {}  # 键: ip:port -> dest_ip:dest_port
        
        # 跟踪开始时间
        self.start_time = time.time()
        
        # 上次更新时间，用于计算速率
        self.last_update_time = self.start_time
        
        # 特殊应用名称集合(常见网络应用)
        self.network_apps = {
            'python', 'python3', 'curl', 'wget', 'netcat', 'nc', 'ssh', 'scp',
            'rsync', 'ftp', 'sftp', 'nmap', 'ping', 'traceroute', 'dig', 'nslookup'
        }
        
        # logger.info("网络数据收集器初始化成功")
    
    def process_traces(self, traces: List[Any], is_last_batch: bool = False) -> int:
        """处理网络跟踪数据"""
        if not traces:
            return 0
            
        count = len(traces)
        self.total_traces += count

        sorted_traces = sorted(traces, key=lambda e: getattr(e, "timestamp", 0))
        
        events = []
        csv_rows = []
        csv_headers = [
            'Timestamp', 'PID', 
            'tx_bytes', 'rx_bytes', 'total_tx_packets', 'total_rx_packets',
            'active_connections'
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
                event['args']['tx_bytes'],
                event['args']['rx_bytes'],
                event['args']['total_tx_packets'],
                event['args']['total_rx_packets'],
                event['args']['active_connect']
            ])

        # 3. 写入json和csv
        self.write_events_to_json(events, is_last_batch)
        if csv_rows:
            self.write_rows_to_csv(csv_headers, csv_rows)

        # 4. 更新Prometheus指标
        # self._update_prometheus_metrics()

        return len(events)
    
    def _process_single_trace(self, trace) -> Dict[str, Any]:
        """处理单个网络跟踪事件"""
        try:
            # 提取基本信息
            pid = trace.tgid
            tid = trace.pid  # 线程ID (默认为1)
            timestamp = trace.timestamp
            comm = trace.comm
            is_send = trace.is_send
            bytes_count = trace.bytes
            
            # 网络信息
            src_addr = trace.src_addr
            dst_addr = trace.dst_addr
            src_port = trace.src_port
            dst_port = trace.dst_port
            protocol = trace.protocol
            
            # 统计信息
            tx_bytes = trace.tx_bytes
            rx_bytes = trace.rx_bytes
            tx_packets = trace.tx_packets
            rx_packets = trace.rx_packets
            
            active_connections = trace.active_connections
            
            # 如果进程不存在于缓存中，初始化它
            if pid not in self.process_stats:
                self.process_stats[pid] = {
                    'comm': comm,
                    'time':timestamp,
                    'tx_bytes': tx_bytes,
                    'rx_bytes': rx_bytes,
                    'tx_packets': tx_packets,
                    'rx_packets': rx_packets,
                    'tx_bytes_history': [],
                    'rx_bytes_history': [],
                    'connections': set(),
                    'connection_key': f"{src_addr}:{src_port}->{dst_addr}:{dst_port}",
                    'listening_ports': set(),
                    'active_connections': active_connections,
                }
            else:
                # 更新现有进程的统计信息
                stats = self.process_stats[pid]
                stats['comm'] = comm  # 更新进程名，以防更改
                stats['tx_bytes'] += tx_bytes
                stats['rx_bytes'] += rx_bytes
                stats['tx_packets'] = tx_packets
                stats['rx_packets'] = rx_packets
                stats['active_connections'] = active_connections
                stats['last_activity'] = time.time()
                
                # 添加连接信息
                connection_key = f"{src_addr}:{src_port}->{dst_addr}:{dst_port}"
                stats['connections'].add(connection_key)
                
                # 如果是监听端口，记录
                if dst_port == 0 and src_port > 0:
                    stats['listening_ports'].add(src_port)
            
            stats = self.process_stats[pid]
            
            # # 计算速率
            # current_time = time.time()
            # elapsed = current_time - self.last_update_time
            
            # if elapsed > 0:
            #     # 计算发送和接收速率
            #     if is_send:
            #         # 只计算这次事件的字节数除以时间间隔
            #         tx_rate = bytes_count / elapsed
            #         stats['tx_rate'] = tx_rate
                    
            #         # 保持最近的历史记录
            #         if len(stats['tx_bytes_history']) >= 30:
            #             stats['tx_bytes_history'].pop(0)
            #         stats['tx_bytes_history'].append(tx_rate)
            #     else:
            #         # 只计算这次事件的字节数除以时间间隔
            #         rx_rate = bytes_count / elapsed
            #         stats['rx_rate'] = rx_rate
                    
            #         # 保持最近的历史记录
            #         if len(stats['rx_bytes_history']) >= 30:
            #             stats['rx_bytes_history'].pop(0)
            #         stats['rx_bytes_history'].append(rx_rate)
            stats = self.process_stats[pid]

            labels = {'pid': str(pid),'comm':comm}

            NET_TOTAL_TX_BYTES.labels(**labels).set(stats['tx_bytes'])
            NET_TOTAL_RX_BYTES.labels(**labels).set(stats['rx_bytes'])
            NET_TOTAL_TX_PACKETS.labels(**labels).set(stats['tx_packets'])
            NET_TOTAL_RX_PACKETS.labels(**labels).set(stats['rx_packets'])
            
            # 更新连接指标
            NET_ACTIVE_CONNECTIONS.labels(**labels).set(stats['active_connections'])
            NET_LISTENING_PORTS.labels(**labels).set(len(stats['listening_ports']))
            
            # 创建用于可视化的事件对象
            timestamp_ms = parse_timestamp(timestamp) if timestamp else int(time.time() * 1000)
            
            operation = "send" if is_send else "receive"
            
            event = {
                "name": f"{comm}",
                "ts": timestamp_ms,
                "pid": pid,
                "tid": tid,  # 使用pid作为tid
                "args": {
                    "op": operation,
                    "bytes": bytes_count,
                    "src": f"{src_addr}:{src_port}",
                    "dst": f"{dst_addr}:{dst_port}",
                    "protocol": protocol,
                    "total_tx_packets": tx_packets,
                    "total_rx_packets": rx_packets,
                    "tx_bytes": tx_bytes,
                    "rx_bytes": rx_bytes,
                    "active_connect": active_connections
                }
            }
            
            return event
            
        except Exception as e:
            logger.error(f"Error processing network trace: {str(e)}", exc_info=True)
            return None
    
    def _update_prometheus_metrics(self) -> None:
        """更新Prometheus指标"""
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        
        # 处理每个进程的统计信息
        for pid, stats in self.process_stats.items():
            # if current_time - stats['last_activity'] > 60:  # 60秒无活动则跳过
            #     continue
                
            comm = stats['comm']
            labels = {'pid': str(pid), 'process': comm}
            
            # # 更新基本网络速率指标
            # NET_TX_BYTES_RATE.labels(**labels).set(stats['tx_rate'])
            # NET_RX_BYTES_RATE.labels(**labels).set(stats['rx_rate'])
            
            # 更新累计网络指标
            NET_TOTAL_TX_BYTES.labels(**labels).set(stats['tx_bytes'])
            NET_TOTAL_RX_BYTES.labels(**labels).set(stats['rx_bytes'])
            NET_TOTAL_TX_PACKETS.labels(**labels).set(stats['tx_packets'])
            NET_TOTAL_RX_PACKETS.labels(**labels).set(stats['rx_packets'])
            
            # 更新连接指标
            NET_ACTIVE_CONNECTIONS.labels(**labels).set(stats['active_connections'])
            NET_LISTENING_PORTS.labels(**labels).set(len(stats['listening_ports']))
            
            # # 计算包速率
            # tx_packet_rate = 0
            # rx_packet_rate = 0
            
            # # 使用简单的平均值估算
            # if elapsed_time > 0 and hasattr(stats, 'prev_tx_packets'):
            #     tx_packet_delta = stats['tx_packets'] - getattr(stats, 'prev_tx_packets', stats['tx_packets'])
            #     rx_packet_delta = stats['rx_packets'] - getattr(stats, 'prev_rx_packets', stats['rx_packets'])
            #     tx_packet_rate = tx_packet_delta / elapsed_time
            #     rx_packet_rate = rx_packet_delta / elapsed_time
            
            # # 保存当前值以便下次计算
            # stats['prev_tx_packets'] = stats['tx_packets']
            # stats['prev_rx_packets'] = stats['rx_packets']
            
            # # 设置包速率指标
            # NET_TX_PACKETS_RATE.labels(**labels).set(tx_packet_rate)
            # NET_RX_PACKETS_RATE.labels(**labels).set(rx_packet_rate)
    
    def _export_incremental_stats_to_csv(self, events: List[Dict[str,Any]]) -> None:
        """增量导出网络统计数据到CSV"""
        # 准备表头和数据
        events_headers = [
            'Timestamp', 'PID', 
            'TX Bytes', 'RX Bytes', 'TX Packets', 'RX Packets',
            'active_connections'
        ]
        
        events_rows = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 使用最近一段时间内收集的数据
        # elapsed = time.time() - getattr(self, '_last_exported_events_time', 0)
        # if elapsed < 10:  # 每10秒导出一次
        #     return
        
        self._last_exported_events_time = time.time()
        
        for i, event in enumerate(events):
                
            row = [
                event['time'],  # 使用事件时间戳
                event['pid'],
                event['args']['tx_bytes'],
                event['args']['rx_bytes'],
                event['args']['tx_packets'],
                event['args']['rx_packets'],
                event['args']['active_connections'],
            ]
            events_rows.append(row)
        
        # 导出统计数据
        if events_rows:
            self.write_rows_to_csv(events_headers, events_rows)
        
        # 导出连接数据
        conn_headers = [
            'Timestamp', 'PID', 'Process', 'Connection', 
            'Source', 'Destination', 'Protocol'
        ]
        conn_rows = []
        
        # # 获取活跃的进程连接
        # for pid, stats in self.process_stats.items():
        #     if not stats['connections']:
        #         continue
                
        #     # 最多导出每个进程的前10个连接
        #     for i, conn in enumerate(list(stats['connections'])[:10]):
        #         parts = conn.split('->')
        #         if len(parts) != 2:
        #             continue
                    
        #         src, dst = parts
                
        #         row = [
        #             current_time,
        #             pid,
        #             stats['comm'],
        #             conn,
        #             src,
        #             dst,
        #             "TCP"  # 假设为TCP，实际应从trace中读取
        #         ]
        #         conn_rows.append(row)
        
        # # 导出连接数据
        # if conn_rows:
        #     self.file_manager.export_to_csv(
        #         trace_type="network",
        #         data_type="connections",
        #         headers=conn_headers,
        #         rows=conn_rows,
        #         append=True
        #     )



    def collect_stats(self) -> None:
        """收集并返回网络统计信息"""
        # result = {
        #     'total_traces': self.total_traces,
        #     'processes': len(self.process_stats),
        #     'anomalies': sum(len(stats['anomalies']) for stats in self.process_stats.values()),
        #     'top_processes': []
        # }
        top_processes = []
        
        # 按网络流量排序进程
        sorted_processes = sorted(
            self.process_stats.items(),
            key=lambda x: x[1]['tx_bytes'] + x[1]['rx_bytes'],
            reverse=True
        )

        # 获取前5个最活跃的进程
        for pid, stats in sorted_processes[:5]:
            top_processes.append({
                'pid': pid,
                'comm': stats['comm'],
                'tx_bytes': stats['tx_bytes'],
                'rx_bytes': stats['rx_bytes'],
                'tx_packets': stats['tx_packets'],
                'rx_packets': stats['rx_packets'],
                'active_connections': stats['active_connections'],
            })
        logger.info("Total network traces processed: %d", self.total_traces)
        logger.info("Total processes monitored: %d", len(self.process_stats))
        logger.info("Top 5 active processes: %s", top_processes)

        for proc in top_processes:
            logger.info(
                "Process %s (PID %d): TX %s (%d packets), RX %s (%d packets), Active Connections: %d, Pattern: %s, Anomalies: %d",
                proc['comm'], proc['pid'],
                format_bytes(proc['tx_bytes']), proc['tx_packets'],
                format_bytes(proc['rx_bytes']), proc['rx_packets'],
                proc['active_connections'], proc['pattern'], proc['anomaly_count']
            )
        
        for pid, stats in self.process_stats.items():
            if stats['anomalies']:
                logger.info(
                    "Process %s (PID %d) has anomalies: %s",
                    stats['comm'], pid,
                    ", ".join(f"{a['description']} (Severity: {a['severity']:.2f})" for a in stats['anomalies'])
                )
        # return result
        

    def finalize(self) -> None:
        """完成处理，执行清理工作"""
        return
        self.collect_stats()
        