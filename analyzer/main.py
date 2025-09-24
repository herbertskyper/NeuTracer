#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess
import glob
import time
import shlex
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

class AnalyzerManager:
    """分析器管理器 - 负责启动不同的异常检测/性能分析服务"""
    
    def __init__(self):
        self.services = {
            'stack': {
                'name': '调用栈分析与可视化服务',
                'script': 'python analyzer/visualizer/convert_to_perfetto.py', 
                'description': '分析python和cuda调用栈数据',
                'env': {'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'}
            },
            'gpu': {
                'name': 'GPU碎片化分析与可视化',
                'script': 'python analyzer/visualizer/draw_cuda_frag.py',
                'description': '专门分析GPU内存相关性能指标'
            },
            'anomaly': {
                'name': '数据异常检测服务',
                'script': 'python anomaly_detector.py',
                'description': '利用压缩感知算法检测数据异常'
            },
            'nn': {
                'name': '神经网络静默检测服务',
                'script': 'python nn_detect.py',
                'description': '检测神经网络模型的静默异常'
            }
        }
    
    def list_available_services(self):
        """列出所有可用的分析服务"""
        print("\n可用的分析服务:")
        print("=" * 60)
        for key, service in self.services.items():
            # 检查脚本文件是否存在
            script_parts = service['script'].split()
            if len(script_parts) >= 2:
                script_file = script_parts[1]
                status = "✓" if os.path.exists(script_file) else "✗"
            else:
                status = "?"
                
            print(f"{status} {key:12} - {service['name']}")
            print(f"   {'':12}   {service['description']}")
            print(f"   {'':12}   命令: {service['script']}")
            
            # 显示环境变量（如果有）
            if 'env' in service:
                env_str = ', '.join([f"{k}={v}" for k, v in service['env'].items()])
                print(f"   {'':12}   环境变量: {env_str}")
            print()
    
    def start_service(self, service_name: str, extra_args: List[str] = None) -> bool:
        """
        启动指定的分析服务
        
        Args:
            service_name: 服务名称
            extra_args: 额外的命令行参数
        
        Returns:
            True if successful, False otherwise
        """
        if service_name not in self.services:
            print(f"错误: 未知的服务 '{service_name}'")
            print("使用 --list 查看可用服务")
            return False
        
        service = self.services[service_name]
        script = service['script']
        
        # 分割命令字符串
        cmd = script.split()
        
        # 处理额外参数
        if extra_args:
            # 如果extra_args是单个字符串（从命令行-a传入的整个字符串），需要分割
            processed_args = []
            for arg in extra_args:
                if isinstance(arg, str) and ' ' in arg:
                    # 使用shlex.split来正确处理包含空格的参数
                    processed_args.extend(shlex.split(arg))
                else:
                    processed_args.append(arg)
            cmd.extend(processed_args)
        
        # 准备环境变量
        env = os.environ.copy()
        if 'env' in service:
            env.update(service['env'])
        
        print(f"\n启动服务: {service['name']}")
        print(f"执行命令: {' '.join(cmd)}")
        
        # 显示环境变量（如果有）
        if 'env' in service:
            env_str = ', '.join([f"{k}={v}" for k, v in service['env'].items()])
            print(f"环境变量: {env_str}")
        
        # 显示调试信息
        print(f"命令参数详情: {cmd}")
        print("=" * 60)
        
        # 检查脚本文件是否存在
        if len(cmd) >= 2:
            script_file = cmd[1]
            if not os.path.exists(script_file):
                print(f"错误: 脚本文件 {script_file} 不存在")
                print(f"当前工作目录: {os.getcwd()}")
                return False
        
        try:
            # 启动服务
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            print(f"服务已启动，PID: {process.pid}")
            
            # 实时输出日志
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()
                
                if output == '' and error == '' and process.poll() is not None:
                    break
                    
                if output:
                    print(f"[{service_name}] {output.strip()}")
                if error:
                    print(f"[{service_name}] ERROR: {error.strip()}")
            
            # 检查返回码
            return_code = process.poll()
            if return_code == 0:
                print(f"服务 {service['name']} 执行完成")
                return True
            else:
                # 读取剩余的错误输出
                remaining_stderr = process.stderr.read()
                print(f"服务执行失败，返回码: {return_code}")
                if remaining_stderr:
                    print(f"错误信息: {remaining_stderr}")
                return False
                
        except KeyboardInterrupt:
            print(f"\n用户中断，正在停止服务...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
            return False
        except FileNotFoundError as e:
            print(f"命令未找到: {e}")
            print("请检查：")
            print("1. Python 是否已安装并在 PATH 中")
            print("2. 脚本文件路径是否正确")
            print(f"3. 当前工作目录: {os.getcwd()}")
            return False
        except Exception as e:
            print(f"启动服务时出错: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description='NeuTracer 分析器启动管理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python analyzer/main.py --list                           # 列出所有可用服务
  python analyzer/main.py --service gpu                    # 启动GPU分析服务  
  python analyzer/main.py -s stack -a "-o analyzer/test06101.pftrace  output/txt/stackmessage_20250610_183233_node1.txt" # 启动调用栈分析服务
        '''
    )
    
    # 基本选项
    parser.add_argument('--list', '-l', action='store_true', 
                       help='列出所有可用的分析服务')
    
    parser.add_argument('--service', '-s', 
                       help='要启动的服务名称')
    
    # 修改args参数处理方式
    parser.add_argument('--args', '-a', nargs='*', default=[], 
                       help='传递给服务的额外参数（可以是多个参数或包含空格的单个字符串）')
    
    # 添加调试选项
    parser.add_argument('--debug', action='store_true',
                       help='显示详细的调试信息')
    
    args = parser.parse_args()
    
    # 创建管理器实例
    manager = AnalyzerManager()
    
    # 显示调试信息
    if args.debug:
        print(f"当前工作目录: {os.getcwd()}")
        print(f"Python 路径: {sys.executable}")
        print(f"原始命令行参数: {sys.argv}")
        print(f"解析后的args.args: {args.args}")
        print()
    
    # 列出服务
    if args.list:
        manager.list_available_services()
        return
    
    # 检查是否指定了服务
    if not args.service:
        print("错误: 请指定要启动的服务 (使用 --service)")
        print("使用 --list 查看可用服务")
        parser.print_help()
        return
    
    # 启动服务
    success = manager.start_service(
        service_name=args.service,
        extra_args=args.args or []
    )
    
    if success:
        print(f"\n✓ 服务 '{args.service}' 执行成功")
    else:
        print(f"\n✗ 服务 '{args.service}' 执行失败")
        sys.exit(1)

if __name__ == '__main__':
    main()