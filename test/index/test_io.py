#!/usr/bin/env python3
# filepath: test_io_module.py

import os
import time
import random
import shutil
import threading
import argparse

# 配置变量
TEST_DIR = "./temp/"
LARGE_FILE_SIZE_MB = 10
SMALL_FILE_SIZE_MB = 1
NUM_SMALL_FILES = 5
IO_PATTERNS = ["sequential", "random", "mixed"]

def print_with_pid(message):
    """打印消息并附带进程 ID"""
    print(f"[PID {os.getpid()}] {message}")

def create_test_directory():
    """创建测试目录"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    print_with_pid(f"创建测试目录: {TEST_DIR}")

def clean_test_directory():
    """清理测试目录"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
        print_with_pid(f"清理测试目录: {TEST_DIR}")

def write_large_file():
    """写入大文件 (顺序写)"""
    filename = os.path.join(TEST_DIR, "large_file.dat")
    size_bytes = LARGE_FILE_SIZE_MB * 1024 * 1024
    
    print_with_pid(f"开始写入 {LARGE_FILE_SIZE_MB}MB 的大文件...")
    
    with open(filename, 'wb') as f:
        # 写入大块数据
        chunk_size = 1024 * 1024  # 1MB 块
        for _ in range(LARGE_FILE_SIZE_MB):
            f.write(os.urandom(chunk_size))
            f.flush()
            os.fsync(f.fileno())  # 强制写入磁盘
    
    print_with_pid(f"大文件写入完成: {filename}")
    return filename

def read_large_file_sequential(filename):
    """顺序读取大文件"""
    print_with_pid("开始顺序读取大文件...")
    
    with open(filename, 'rb') as f:
        chunk_size = 1024 * 1024  # 1MB 块
        while True:
            data = f.read(chunk_size)
            if not data:
                break
    
    print_with_pid("顺序读取完成")

def read_large_file_random(filename):
    """随机读取大文件"""
    print_with_pid("开始随机读取大文件...")
    
    file_size = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        # 执行100次随机读取
        for _ in range(10):
            # 随机选择文件位置
            position = random.randint(0, max(0, file_size - 8192))
            f.seek(position)
            # 读取8KB数据
            data = f.read(8192)
    
    print_with_pid("随机读取完成")

def write_small_files():
    """写入多个小文件 (模拟多文件操作)"""
    print_with_pid(f"开始写入 {NUM_SMALL_FILES} 个小文件 (每个 {SMALL_FILE_SIZE_MB}MB)...")
    
    filenames = []
    for i in range(NUM_SMALL_FILES):
        filename = os.path.join(TEST_DIR, f"small_file_{i}.dat")
        size_bytes = SMALL_FILE_SIZE_MB * 1024 * 1024
        
        with open(filename, 'wb') as f:
            f.write(os.urandom(size_bytes))
            f.flush()
            os.fsync(f.fileno())
        
        filenames.append(filename)
    
    print_with_pid("小文件写入完成")
    return filenames

def read_small_files(filenames):
    """读取多个小文件"""
    print_with_pid("开始读取小文件...")
    
    for filename in filenames:
        with open(filename, 'rb') as f:
            data = f.read()
    
    print_with_pid("小文件读取完成")

def mixed_io_workload():
    """混合 I/O 工作负载 (读写混合)"""
    print_with_pid("开始执行混合 I/O 工作负载...")
    
    # 创建一些文件
    files = []
    for i in range(10):
        filename = os.path.join(TEST_DIR, f"mixed_file_{i}.dat")
        with open(filename, 'wb') as f:
            f.write(os.urandom(10 * 1024 * 1024))  # 10MB
            f.flush()
        files.append(filename)
    
    # 同时进行读写操作
    for _ in range(5):
        # 随机选择一个文件进行读取
        read_file = random.choice(files)
        with open(read_file, 'rb') as f:
            f.seek(random.randint(0, 9 * 1024 * 1024))
            data = f.read(1024 * 1024)
        
        # 随机选择一个文件进行写入
        write_file = random.choice(files)
        with open(write_file, 'wb') as f:
            f.seek(random.randint(0, 8 * 1024 * 1024))
            f.write(os.urandom(2 * 1024 * 1024))
            f.flush()
            os.fsync(f.fileno())
    
    print_with_pid("混合 I/O 工作负载完成")

def parallel_io_operations():
    """并行 I/O 操作 (多线程)"""
    print_with_pid("开始执行并行 I/O 操作...")
    
    # 创建测试文件
    base_file = os.path.join(TEST_DIR, "parallel_base.dat")
    with open(base_file, 'wb') as f:
        f.write(os.urandom(20 * 1024 * 1024))  # 20MB
    
    # 定义线程函数
    def thread_io(thread_id):
        print_with_pid(f"线程 {thread_id} 启动")
        filename = os.path.join(TEST_DIR, f"thread_{thread_id}.dat")
        
        # 从基础文件复制数据
        with open(base_file, 'rb') as src, open(filename, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        
        # 读取刚写入的文件
        with open(filename, 'rb') as f:
            data = f.read()
        
        print_with_pid(f"线程 {thread_id} 完成")
    
    # 启动多个线程
    threads = []
    for i in range(5):
        t = threading.Thread(target=thread_io, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print_with_pid("并行 I/O 操作完成")

def test_sequential_io():
    """测试顺序 I/O 模式"""
    print_with_pid("\n=== 测试顺序 I/O ===")
    large_file = write_large_file()
    read_large_file_sequential(large_file)

def test_random_io():
    """测试随机 I/O 模式"""
    print_with_pid("\n=== 测试随机 I/O ===")
    large_file = write_large_file()
    read_large_file_random(large_file)

def test_small_files_io():
    """测试小文件 I/O 模式"""
    print_with_pid("\n=== 测试小文件 I/O ===")
    small_files = write_small_files()
    read_small_files(small_files)

def test_mixed_io():
    """测试混合 I/O 模式"""
    print_with_pid("\n=== 测试混合 I/O ===")
    mixed_io_workload()

def test_parallel_io():
    """测试并行 I/O 模式"""
    print_with_pid("\n=== 测试并行 I/O ===")
    parallel_io_operations()

def main():
    global TEST_DIR, LARGE_FILE_SIZE_MB, SMALL_FILE_SIZE_MB, NUM_SMALL_FILES
    
    parser = argparse.ArgumentParser(description="I/O 模块测试程序")
    parser.add_argument("-p", "--pattern", choices=IO_PATTERNS + ["all"], default="all",
                        help="指定要测试的 I/O 模式")
    parser.add_argument("-d", "--directory", default=TEST_DIR,
                        help="指定测试目录")
    parser.add_argument("--large-file-size", type=int, default=LARGE_FILE_SIZE_MB,
                        help="大文件大小 (MB)")
    parser.add_argument("--small-file-size", type=int, default=SMALL_FILE_SIZE_MB,
                        help="小文件大小 (MB)")
    parser.add_argument("--num-small-files", type=int, default=NUM_SMALL_FILES,
                        help="小文件数量")
    
    args = parser.parse_args()
    
    # 更新全局配置
    
    TEST_DIR = args.directory
    LARGE_FILE_SIZE_MB = args.large_file_size
    SMALL_FILE_SIZE_MB = args.small_file_size
    NUM_SMALL_FILES = args.num_small_files
    
    try:
        print_with_pid(f"Python I/O 模块测试程序 - PID: {os.getpid()}")
        print_with_pid("等待 5 秒以便附加跟踪器...")
        time.sleep(5)
        
        create_test_directory()
        
        # 根据指定的模式或全部测试
        if args.pattern == "all" or args.pattern == "sequential":
            test_sequential_io()
            time.sleep(5)
        
        if args.pattern == "all" or args.pattern == "random":
            test_random_io()
            time.sleep(5)
        
        if args.pattern == "all" or args.pattern == "mixed":
            test_small_files_io()
            time.sleep(5)
            # test_mixed_io()
            # time.sleep(5)
            # test_parallel_io()
            # time.sleep(5)
        
        print_with_pid("\n所有 I/O 测试完成!")
    
    finally:
        # 清理
        clean_test_directory()

if __name__ == "__main__":
    main()