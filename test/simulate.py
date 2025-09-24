"""
NeuTracer 单次异常模拟脚本
模拟AI/ML系统中常见的各种异常情况，用于测试NeuTracer的异常检测能力
"""

import torch
import os
import time
import traceback
import threading
import socket
import requests
import numpy as np
import multiprocessing
import tempfile
import random
import datetime
import concurrent.futures
from pathlib import Path
import psutil
import gc

print(f"Python 进程 PID: {os.getpid()}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 设备数量: {torch.cuda.device_count()}")
print(f"CPU 核心数: {multiprocessing.cpu_count()}")

# 等待跟踪器附加
print("等待 10 秒以便附加 NeuTracer...")
time.sleep(10)

class AnomalySimulator:
    def __init__(self):
        self.temp_files = []
        
    def log_with_timestamp(self, message):
        """带时间戳的日志输出"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")

    def print_stack(self):
        """打印当前调用栈"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        stack = traceback.format_stack()[:-1]
        print(f"\n[{timestamp}] 当前 Python 调用栈:")
        for line in stack:
            print(f"  {line.strip()}")
        print("----------")

    def simulate_high_cpu_usage(self):
        """模拟高CPU占用异常"""
        self.log_with_timestamp("开始模拟高CPU占用异常...")
        
        # 短时间高强度CPU运算
        for i in range(10):
            a = np.random.random((1000, 1000))
            b = np.random.random((1000, 1000))
            c = np.dot(a, b)
            result = np.sum(np.exp(np.sin(c)) * np.cos(c))
            self.log_with_timestamp(f"CPU密集运算 {i+1}/10 完成")
        
        self.log_with_timestamp("高CPU占用模拟完成")

    def simulate_network_connections(self):
        """模拟网络连接"""
        self.log_with_timestamp("开始模拟网络连接...")
        
        urls = [
                "http://httpbin.org/delay/1",
                "http://httpbin.org/status/200",
                "http://httpbin.org/json",
                "http://httpbin.org/uuid",
                "http://httpbin.org/user-agent",
                "http://httpbin.org/delay/1",
                "http://httpbin.org/status/200",
                "http://httpbin.org/json",
                "http://httpbin.org/uuid",
                "http://httpbin.org/user-agent",
        ]
        for i, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=5)
                self.log_with_timestamp(f"网络连接 {i+1}: {url} -> {response.status_code}")
            except Exception as e:
                self.log_with_timestamp(f"网络连接 {i+1} 失败: {e}")
            time.sleep(1)
        
        self.log_with_timestamp("网络连接模拟完成")

    def simulate_io_operations(self):
        """模拟IO操作"""
        self.log_with_timestamp("开始模拟IO操作...")
        
        temp_dir = tempfile.mkdtemp()
        self.log_with_timestamp(f"临时目录: {temp_dir}")
        
        # 创建几个大文件
        for i in range(10):
            file_path = os.path.join(temp_dir, f"test_file_{i}.dat")
            with open(file_path, 'wb') as f:
                # 写入100MB数据
                data = os.urandom(1024 * 1024 * 10)  # 10MB随机数据
                for _ in range(10):
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            
            self.temp_files.append(file_path)
            self.log_with_timestamp(f"创建文件 {i+1}/10: {file_path}")
        
        # 读取文件
        # for file_path in self.temp_files:
        #     with open(file_path, 'rb') as f:
        #         data = f.read(1024 * 1024)  # 读取1MB
        #     self.log_with_timestamp(f"读取文件: {file_path}")
        
        self.log_with_timestamp("IO操作模拟完成")

    def simulate_memory_allocation(self):
        """模拟内存分配"""
        self.log_with_timestamp("开始模拟内存分配...")
        
        tensors = []
        
        # 分配一些大内存块
        for i in range(10):
            size_mb = 100 + i * 100  # 100MB, 150MB, 200MB...
            tensor = torch.randn(size_mb * 1024 * 256, dtype=torch.float32)
            tensors.append(tensor)
            current_memory = psutil.virtual_memory().used / (1024**3)
            self.log_with_timestamp(f"分配内存块 {i+1}: {size_mb}MB, 当前系统内存: {current_memory:.2f}GB")
            time.sleep(1)
        
        # 释放内存
        tensors.clear()
        gc.collect()
        final_memory = psutil.virtual_memory().used / (1024**3)
        self.log_with_timestamp(f"内存释放完成，当前系统内存: {final_memory:.2f}GB")

    def simulate_gpu_operations(self):
        """模拟GPU操作"""
        if not torch.cuda.is_available():
            self.log_with_timestamp("CUDA不可用，跳过GPU操作模拟")
            return
        
        self.log_with_timestamp("开始模拟GPU操作...")
        
        initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        self.log_with_timestamp(f"初始GPU显存: {initial_memory:.2f} MB")
        
        gpu_tensors = []
        
        # 分配不同大小的GPU张量
        sizes = [(5000, 5000), (2000, 10000), (1500, 10000), (8000, 2000), (3000, 10000)]
        
        for i, (h, w) in enumerate(sizes):
            tensor = torch.randn(h, w, device='cuda')
            gpu_tensors.append(tensor)
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            self.log_with_timestamp(f"分配GPU张量 {i+1}: {h}x{w}, GPU显存: {current_memory:.2f} MB")
            time.sleep(1)
        
        # 进行一些GPU运算
        self.log_with_timestamp("执行GPU矩阵运算...")
        for i in range(3):
            a = torch.randn(1000, 1000, device='cuda')
            b = torch.randn(1000, 1000, device='cuda')
            c = torch.matmul(a, b)
            result = torch.sum(c)
            self.log_with_timestamp(f"GPU运算 {i+1}/3: 结果 = {result.item():.4f}")
        
        # 释放部分张量，制造碎片
        self.log_with_timestamp("释放部分GPU张量...")
        for i in [1, 3]:  # 释放第2和第4个张量
            gpu_tensors[i] = None
        
        torch.cuda.empty_cache()
        fragmented_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        self.log_with_timestamp(f"碎片化后GPU显存: {fragmented_memory:.2f} MB")
        
        # 尝试分配大张量
        try:
            large_tensor = torch.randn(4000, 4000, device='cuda')
            self.log_with_timestamp("大张量分配成功")
            gpu_tensors.append(large_tensor)
        except RuntimeError as e:
            self.log_with_timestamp(f"GPU显存碎片化异常: {e}")
        
        # 清理
        gpu_tensors.clear()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        self.log_with_timestamp(f"GPU显存清理完成: {final_memory:.2f} MB")

    def simulate_neural_network(self):
        """模拟神经网络训练"""
        if not torch.cuda.is_available():
            self.log_with_timestamp("CUDA不可用，跳过神经网络模拟")
            return
        
        self.log_with_timestamp("开始模拟神经网络训练...")
        
        # 简单的CNN模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.fc1 = torch.nn.Linear(64 * 32 * 32, 512)
                self.fc2 = torch.nn.Linear(512, 10)
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(0.5)
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        try:
            model = SimpleModel().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # 训练几个批次
            for epoch in range(3):
                batch_size = 32
                inputs = torch.randn(batch_size, 3, 128, 128, device='cuda')
                targets = torch.randint(0, 10, (batch_size,), device='cuda')
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 模拟梯度异常
                if epoch == 1:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad *= 100  # 制造梯度爆炸
                    self.log_with_timestamp("模拟梯度爆炸异常")
                
                optimizer.step()
                
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                self.log_with_timestamp(f"训练轮次 {epoch+1}: 损失 {loss.item():.4f}, GPU显存 {gpu_memory:.2f} MB")
                
                time.sleep(2)
            
        except Exception as e:
            self.log_with_timestamp(f"神经网络训练错误: {e}")
        
        self.log_with_timestamp("神经网络训练完成")

    def run_single_anomaly_simulation(self):
        """运行单次异常模拟"""
        self.log_with_timestamp("=== 开始 NeuTracer 单次异常模拟测试 ===")
        
        try:
            # 打印初始调用栈
            # self.print_stack()
            
            # 按顺序执行各种异常模拟
            # self.simulate_gpu_operations()
            # time.sleep(3)
            
            self.simulate_network_connections()
            time.sleep(3)

            self.simulate_high_cpu_usage()
            time.sleep(3)
            
            self.simulate_io_operations()
            time.sleep(3)
            
            self.simulate_memory_allocation()
            time.sleep(3)
            
            
            self.simulate_neural_network()
            time.sleep(3)
            
            # # 打印最终状态
            # cpu_percent = psutil.cpu_percent()
            # memory_info = psutil.virtual_memory()
            # status_msg = f"最终系统状态 - CPU: {cpu_percent:.1f}%, 内存: {memory_info.percent:.1f}%"
            
            # if torch.cuda.is_available():
            #     gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            #     status_msg += f", GPU显存: {gpu_memory:.2f} GB"
            
            self.log_with_timestamp(status_msg)
            # self.print_stack()
            
        except KeyboardInterrupt:
            self.log_with_timestamp("收到中断信号...")
        except Exception as e:
            self.log_with_timestamp(f"模拟测试出错: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.log_with_timestamp("开始清理资源...")
        
        # 清理临时文件
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.log_with_timestamp(f"删除临时文件: {file_path}")
            except Exception as e:
                self.log_with_timestamp(f"删除文件失败 {file_path}: {e}")
        
        # 清理GPU显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        self.log_with_timestamp("资源清理完成")
        self.log_with_timestamp("=== NeuTracer 单次异常模拟测试结束 ===")

if __name__ == "__main__":
    simulator = AnomalySimulator()
    simulator.run_single_anomaly_simulation()