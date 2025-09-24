#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np
import argparse
from multiprocessing import Pool

"""
机器学习IO模式测试工具

此脚本模拟不同机器学习场景下的IO模式，用于测试NeuTracer的IO异常检测功能。
"""

class MLIOSimulator:
    def __init__(self, output_dir="/data/AI-tracking-SYSU/temp_test/ml_data", file_size_mb=100):
        """
        初始化ML IO模拟器
        
        参数:
            output_dir: 输出目录
            file_size_mb: 基础文件大小(MB)
        """
        self.output_dir = output_dir
        self.file_size_mb = file_size_mb
        os.makedirs(output_dir, exist_ok=True)
        
    def _create_large_file(self, filename, size_mb):
        """创建指定大小的文件"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(os.urandom(size_mb * 1024 * 1024))
        return filepath
    
    def simulate_data_loading(self, batch_size=32, num_batches=100, num_files=10):
        """
        模拟数据加载过程
        
        特点：频繁的小批量随机读取，模拟数据加载器行为
        可能触发的异常：频繁的小IO读取
        """
        print(f"[测试1] 模拟数据加载过程 - 频繁小批量读取")
        
        # 创建测试数据文件
        data_files = []
        for i in range(num_files):
            filename = f"dataset_shard_{i}.bin"
            filepath = self._create_large_file(filename, self.file_size_mb)
            data_files.append(filepath)
        
        # 模拟数据加载
        for batch in range(num_batches):
            for _ in range(batch_size):
                # 随机选择文件
                file_path = random.choice(data_files)
                with open(file_path, 'rb') as f:
                    # 随机读取位置
                    offset = random.randint(0, self.file_size_mb * 1024 * 1024 - 4096)
                    f.seek(offset)
                    # 读取小块数据（模拟单个样本）
                    _ = f.read(4096)
                    time.sleep(0.001)  # 短暂停顿
            
            if batch % 10 == 0:
                print(f"  完成批次 {batch}/{num_batches}")
                
        print("[测试1] 完成")
        
    def simulate_model_checkpoint(self, num_epochs=5, large_checkpoint=True):
        """
        模拟模型训练过程中的检查点保存
        
        特点：周期性大规模写操作，模拟保存模型权重
        可能触发的异常：IO写入延迟峰值
        """
        print(f"[测试2] 模拟模型检查点保存 - 大规模写操作")
        
        checkpoint_size = 2048 if large_checkpoint else 512  # MB
        
        for epoch in range(num_epochs):
            # 模拟训练过程
            print(f"  训练周期 {epoch+1}/{num_epochs}...")
            time.sleep(5)  # 模拟训练时间
            
            # 模拟检查点保存（大文件写入）
            print(f"  保存检查点 (大小: {checkpoint_size}MB)...")
            checkpoint_file = f"model_epoch_{epoch+1}.checkpoint"
            self._create_large_file(checkpoint_file, checkpoint_size)
            
            # 可选：模拟优化器状态保存
            self._create_large_file(f"optimizer_epoch_{epoch+1}.state", checkpoint_size // 4)
            
        print("[测试2] 完成")
        
    def simulate_distributed_training(self, num_nodes=4):
        """
        模拟分布式训练
        
        特点：多进程同时读写，设备负载不均衡
        可能触发的异常：设备切换、进程间IO冲突
        """
        print(f"[测试3] 模拟分布式训练 - {num_nodes}节点并发IO")
        
        def node_worker(node_id):
            # 每个节点有独立的数据分片
            local_dir = os.path.join(self.output_dir, f"node_{node_id}")
            os.makedirs(local_dir, exist_ok=True)
            
            # 模拟梯度同步与参数更新
            for step in range(20):
                # 读取模型参数
                with open(os.path.join(self.output_dir, "shared_model.bin"), "rb") as f:
                    f.seek(random.randint(0, self.file_size_mb * 1024 * 1024 - 8192))
                    _ = f.read(8192)
                
                # 本地梯度计算
                time.sleep(0.5)
                
                # 将本地梯度写入共享存储
                with open(os.path.join(local_dir, f"gradients_{step}.bin"), "wb") as f:
                    f.write(os.urandom(8 * 1024 * 1024))  # 8MB梯度
                
                # 随机IO密集操作 (模拟节点2特别IO密集)
                if node_id == 2:
                    for _ in range(5):
                        with open(os.path.join(local_dir, f"temp_{random.randint(0,1000)}.tmp"), "wb") as f:
                            f.write(os.urandom(16 * 1024 * 1024))
        
        # 创建共享模型文件
        self._create_large_file("shared_model.bin", self.file_size_mb * 2)
        
        # 使用多进程模拟多节点
        with Pool(processes=num_nodes) as pool:
            pool.map(node_worker, range(num_nodes))
            
        print("[测试3] 完成")
        
    def simulate_inference_workload(self, num_requests=500):
        """
        模拟推理工作负载
        
        特点：频繁随机读取不同设备中的模型分片
        可能触发的异常：设备切换频繁
        """
        print(f"[测试4] 模拟模型推理工作负载 - 跨设备随机访问")
        
        # 创建不同的"设备"目录
        devices = ["ssd1", "ssd2", "hdd1"]
        for device in devices:
            device_dir = os.path.join(self.output_dir, device)
            os.makedirs(device_dir, exist_ok=True)
            
            # 每个设备创建模型分片
            for i in range(3):
                self._create_large_file(os.path.join(device, f"model_shard_{i}.bin"), 
                                       self.file_size_mb // 2)
        
        # 模拟推理请求
        for req in range(num_requests):
            # 读取每个设备中的模型分片
            for device in devices:
                shard_id = random.randint(0, 2)
                shard_path = os.path.join(self.output_dir, device, f"model_shard_{shard_id}.bin")
                
                with open(shard_path, 'rb') as f:
                    # 随机访问模型权重
                    offset = random.randint(0, (self.file_size_mb // 2) * 1024 * 1024 - 1024)
                    f.seek(offset)
                    _ = f.read(1024)
            
            if req % 50 == 0:
                print(f"  完成请求 {req}/{num_requests}")
                
        print("[测试4] 完成")
        
    def simulate_data_preprocessing(self):
        """
        模拟数据预处理流水线
        
        特点：读写交替，IO大小不均匀
        可能触发的异常：读写不平衡
        """
        print(f"[测试5] 模拟数据预处理流水线 - 读写交替不平衡")
        
        # 创建原始数据
        raw_data_size = self.file_size_mb * 4
        raw_data_path = self._create_large_file("raw_data.bin", raw_data_size)
        
        # 模拟数据预处理流水线
        chunk_sizes = [64*1024*1024, 32*1024*1024, 128*1024*1024, 8*1024*1024]
        
        with open(raw_data_path, 'rb') as src:
            # 分块处理
            offset = 0
            chunk_id = 0
            
            while offset < raw_data_size * 1024 * 1024:
                # 读取一个不规则大小的块
                chunk_size = min(random.choice(chunk_sizes), 
                                raw_data_size * 1024 * 1024 - offset)
                src.seek(offset)
                chunk_data = src.read(chunk_size)
                
                # 写入处理后的数据（通常比原始数据小）
                with open(os.path.join(self.output_dir, f"processed_{chunk_id}.bin"), 'wb') as dst:
                    # 模拟50%压缩率
                    dst.write(chunk_data[:chunk_size//2])
                
                offset += chunk_size
                chunk_id += 1
                
                # 产生一些IO延迟变化
                if chunk_id % 3 == 0:
                    time.sleep(0.5)  # 正常延迟
                elif chunk_id % 5 == 0:
                    time.sleep(2.0)  # 异常高延迟
                else:
                    time.sleep(0.1)  # 低延迟
                    
        print("[测试5] 完成")

def main():
    parser = argparse.ArgumentParser(description='机器学习IO模式测试工具')
    parser.add_argument('--test', type=int, choices=[0, 1, 2, 3, 4, 5], default=0,
                        help='要运行的测试: 0=全部, 1=数据加载, 2=检查点, 3=分布式训练, 4=推理, 5=预处理')
    parser.add_argument('--output-dir', type=str, default='/data/AI-tracking-SYSU/temp_test/ml_io_test_data',
                        help='测试数据输出目录')
    parser.add_argument('--file-size', type=int, default=100,
                        help='基本测试文件大小(MB)')
    args = parser.parse_args()
    
    simulator = MLIOSimulator(args.output_dir, args.file_size)
    
    print("=" * 60)
    print("机器学习IO模式测试工具")
    print(f"\n{os.getpid()}")
    print("=" * 60)
    
    if args.test == 0 or args.test == 1:
        simulator.simulate_data_loading()
    if args.test == 0 or args.test == 2:
        simulator.simulate_model_checkpoint()
    # if args.test == 0 or args.test == 3:
    #     simulator.simulate_distributed_training()
    if args.test == 0 or args.test == 4:
        simulator.simulate_inference_workload()
    if args.test == 0 or args.test == 5:
        simulator.simulate_data_preprocessing()
    
    print("\n所有测试完成!")
    print(f"测试数据保存在: {args.output_dir}")
    print("可以使用NeuTracer分析这些IO模式并查找异常")

if __name__ == "__main__":
    main()