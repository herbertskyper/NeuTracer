import os
import time
import grpc
import logging
import argparse
import signal
from concurrent import futures
from prometheus_client import start_http_server
from py_grpc_prometheus.prometheus_server_interceptor import PromServerInterceptor
import sys

# 添加包含protos/python目录的路径到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protos", "python"))
import tracer_service_pb2_grpc

from servicer import TracerServicer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tracer-service')


def serve(port: int = 50051, metrics_port: int = 9091, inactivity_timeout: int = 60) -> None:
    """启动跟踪服务并监听请求"""
    # 启动Prometheus HTTP服务器
    start_http_server(metrics_port)
    logger.info(f"Prometheus指标服务器已启动，端口: {metrics_port}")
    
    # 创建并配置服务
    servicer = TracerServicer(inactivity_timeout=inactivity_timeout)
    
    # 创建gRPC服务器
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=(PromServerInterceptor(),)
    )
    tracer_service_pb2_grpc.add_TracerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"服务器已启动，端口: {port}，不活动超时: {inactivity_timeout}秒")
    
    try:
        # 等待服务结束信号
        while not servicer.shutdown_event.is_set():
            time.sleep(5)
        logger.info("检测到超时，正在关闭服务器...")
        server.stop(2)
    except KeyboardInterrupt:
        logger.info("收到键盘中断，正在关闭服务器...")
        server.stop(0)
    finally:
        try:
            servicer.close_all_files()
        except Exception:
            pass  # 如果文件已关闭则忽略异常
        logger.info("服务器已关闭")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='NeuTracer 服务端')
    parser.add_argument('--port', type=int, default=int(os.environ.get('SERVICE_PORT', '50051')),
                      help='gRPC服务端口')
    parser.add_argument('--metrics-port', type=int, default=int(os.environ.get('METRICS_PORT', '9091')),
                      help='Prometheus指标端口')
    parser.add_argument('--timeout', type=int, default=int(os.environ.get('INACTIVITY_TIMEOUT', '60')),
                      help='无活动自动关闭超时时间(秒)')
    
    args = parser.parse_args()
    
    # 启动服务
    serve(port=args.port, metrics_port=args.metrics_port, inactivity_timeout=args.timeout)