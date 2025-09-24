#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import random
import psutil

"""
机器学习训练过程CPU尖峰异常模拟工具

该脚本模拟机器学习训练过程中可能出现的CPU使用异常模式：
1. 训练过程中的突发性负载尖峰
2. 批处理导致的CPU周期性波动
3. 模型评估时的CPU抢占
4. 跨NUMA节点的任务迁移
"""

class MLCPUAnomalySimulator:
    def __init__(self, duration=60, workers=4):
        """初始化CPU异常模拟器
        
        Args:
            duration: 测试持续时间(秒)
            workers: 工作进程数量
        """
        self.duration = duration
        # self.workers = min(workers, mp.cpu_count())
        # self.stop_flag = mp.Value('i', 0)
        print(f"进程ID: {os.getpid()}")
        time.sleep(4)  # 确保进程ID打印完成后再继续
        
    def _simulate_matrix_operations(self, intensity=0.7, duration=1.0, worker_id=0):
        """模拟矩阵计算，占用CPU资源
        
        Args:
            intensity: CPU使用强度(0-1)
            duration: 持续时间(秒)
            worker_id: 工作进程ID
        """
        size = int(1000 * intensity)  # 矩阵大小随强度变化
        start_time = time.time()
        
        # 打印开始消息
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: "
              f"开始矩阵运算 (强度={intensity:.2f}, 时长={duration:.2f}s)")
        
        while time.time() - start_time < duration:
            # 创建随机矩阵
            a = np.random.random((size, size))
            b = np.random.random((size, size))
            
            # 执行矩阵乘法
            c = np.dot(a, b)
            
            # 额外运算增加CPU负载
            for _ in range(3):
                c = np.linalg.matrix_power(c, 2)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: "
              f"矩阵运算完成")

    def _create_cpu_spikes(self, worker_id=0, num_spikes=5):
        """创建明显的CPU使用率尖峰
        
        通过睡眠和高负载交替产生明显的尖峰
        
        Args:
            worker_id: 工作进程ID
            num_spikes: 尖峰次数
        """
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: 开始创建CPU尖峰模式")
        
        for i in range(num_spikes):
            # 首先睡眠3-5秒，让CPU使用率降到很低
            sleep_time = random.uniform(3, 5)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: 休眠{sleep_time:.2f}秒")
            time.sleep(sleep_time)
            
            # 然后突然100%使用CPU 1-2秒
            spike_time = random.uniform(1, 2)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: 突发CPU尖峰 {i+1}/{num_spikes}")
            self._simulate_matrix_operations(intensity=1.0, duration=spike_time, worker_id=worker_id)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: CPU尖峰模式完成")
        
    def _simulate_gradient_descent(self, batch_size=100, num_batches=20, worker_id=0):
        """模拟梯度下降训练过程
        
        Args:
            batch_size: 批次大小
            num_batches: 批次数量
            worker_id: 工作进程ID
        """
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: "
              f"开始模拟梯度下降 ({num_batches}个批次)")
        
        # 创建一个简单的线性模型
        weights = np.random.random(100)
        
        for batch in range(num_batches):
            # if self.stop_flag.value:
            #     break
                
            # 模拟批次开始时的CPU高负载
            self._simulate_matrix_operations(intensity=0.9, duration=0.3, worker_id=worker_id)
            
            # 模拟每个批次的训练
            for _ in range(batch_size):
                # 创建随机数据点
                x = np.random.random(100)
                y = np.dot(x, weights) + np.random.normal(0, 0.1)
                
                # 计算梯度并更新权重
                gradient = 2 * (np.dot(x, weights) - y) * x
                weights -= 0.01 * gradient
            
            # 模拟批次间的低CPU阶段
            time.sleep(0.2)
            
            if batch % 5 == 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: "
                      f"完成{batch}/{num_batches}个批次")

    """
    def _worker_process(self, worker_id):
        #工作进程模拟不同机器学习阶段的CPU负载
        
        Args:
            worker_id: 工作进程ID
        #
        try:
            process = psutil.Process()
            
            # 尝试设置CPU亲和性，强制在特定核心上运行
            if worker_id % 2 == 0:
                try:
                    # 每次在不同CPU核心上运行，增加CPU迁移
                    target_cpu = worker_id % mp.cpu_count()
                    process.cpu_affinity([target_cpu])
                    print(f"Worker {worker_id}: 设置CPU亲和性为核心 {target_cpu}")
                except:
                    print(f"Worker {worker_id}: 无法设置CPU亲和性")
            
            start_time = time.time()
            
            # 开始时添加一个长时间的睡眠，然后突然使用100%CPU资源，制造尖峰
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: 初始休眠5秒")
            time.sleep(5)  # 初始睡眠，确保CPU利用率从低开始
            
            # 添加明显的CPU尖峰模式
            self._create_cpu_spikes(worker_id=worker_id, num_spikes=3)
            
            while time.time() - start_time < self.duration and not self.stop_flag.value:
                # 模式1: 梯度下降批处理
                self._simulate_gradient_descent(
                    batch_size=random.randint(50, 150),
                    num_batches=random.randint(10, 30),
                    worker_id=worker_id
                )
                
                # 再次创建CPU尖峰模式
                if random.random() < 0.6:  # 60%的概率
                    self._create_cpu_spikes(worker_id=worker_id, num_spikes=random.randint(2, 4))
                
                # 模式2: 模型评估(单次高CPU使用)
                if random.random() < 0.7:  # 70%概率执行
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: "
                          f"开始模型评估")
                    self._simulate_matrix_operations(
                        intensity=0.95,
                        duration=random.uniform(1.0, 3.0),
                        worker_id=worker_id
                    )
                
                # 模式3: 数据预处理(中等CPU使用)
                if random.random() < 0.5:  # 50%概率执行
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: "
                          f"开始数据预处理")
                    self._simulate_matrix_operations(
                        intensity=0.6,
                        duration=random.uniform(0.5, 1.5),
                        worker_id=worker_id
                    )
                
                # 模拟线程切换到其他CPU核心
                if worker_id % 2 == 0 and random.random() < 0.2:
                    try:
                        old_affinity = process.cpu_affinity()
                        new_cpu = (old_affinity[0] + mp.cpu_count()//2) % mp.cpu_count()
                        process.cpu_affinity([new_cpu])
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Worker {worker_id}: "
                              f"CPU亲和性从{old_affinity[0]}迁移到{new_cpu} (模拟跨NUMA迁移)")
                    except:
                        pass
                
        except KeyboardInterrupt:
            print(f"Worker {worker_id}: 收到中断信号，退出")
        except Exception as e:
            print(f"Worker {worker_id} 发生错误: {e}")
            """
    
    def run(self):
        """运行CPU异常测试"""
        print("=" * 60)
        print(f"机器学习训练CPU异常模拟工具 (单进程版)")
        print(f"持续时间: {self.duration}秒")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # 使用单进程直接执行工作流程
            process = psutil.Process()
            # 记录开始时间
            start_time = time.time()
            
            # 首先休眠几秒
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始休眠5秒")
            time.sleep(5)
            
            # 创建CPU尖峰
            self._create_cpu_spikes(worker_id=0, num_spikes=3)
            
            # 使用tqdm显示进度
            with tqdm(total=self.duration, desc="测试进度") as pbar:
                while time.time() - start_time < self.duration:
                    # 执行梯度下降
                    self._simulate_gradient_descent(
                        batch_size=random.randint(50, 150),
                        num_batches=5,  # 减少批次数量以适应单进程
                        worker_id=0
                    )
                    
                    # 定期创建CPU尖峰
                    self._create_cpu_spikes(worker_id=0, num_spikes=2)
                    
                    # 模型评估
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始模型评估")
                    self._simulate_matrix_operations(
                        intensity=0.95,
                        duration=random.uniform(1.0, 2.0),
                        worker_id=0
                    )
                    
                    # 睡眠制造尖峰
                    sleep_time = random.uniform(2, 4)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 休眠{sleep_time:.2f}秒")
                    time.sleep(sleep_time)
                    
                    # 尝试切换CPU亲和性
                    try:
                        old_affinity = process.cpu_affinity()
                        cpu_count = len(old_affinity) if isinstance(old_affinity, list) else mp.cpu_count()
                        new_cpu = [(old_affinity[0] + cpu_count//2) % cpu_count]
                        process.cpu_affinity(new_cpu)
                        time.sleep(1)
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"CPU亲和性从{old_affinity}迁移到{new_cpu} (模拟跨NUMA迁移)")
                    except:
                        pass
                    
                    # 更新进度条
                    elapsed = min(int(time.time() - start_time), self.duration)
                    pbar.update(elapsed - pbar.n)
                    
        except KeyboardInterrupt:
            print("\n收到中断信号，退出...")
        
        print("=" * 60)
        print(f"测试完成，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("可以使用NeuTracer分析CPU使用异常")


def main():
    parser = argparse.ArgumentParser(description='机器学习训练CPU异常模拟工具')
    parser.add_argument('--duration', type=int, default=30,
                        help='测试持续时间(秒)')
    # parser.add_argument('--workers', type=int, default=4,
    #                     help='工作进程数量')
    args = parser.parse_args()
    
    simulator = MLCPUAnomalySimulator(
        duration=args.duration,
        # workers=args.workers
    )
    simulator.run()

if __name__ == "__main__":
    main()