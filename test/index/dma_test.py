import torch
import time
import numpy as np

def test_dma_transfers(size_mb=100, iterations=10, device='cuda'):
    """
    使用PyTorch测试GPU DMA传输性能
    
    参数:
        size_mb: 测试数据大小(MB)
        iterations: 每次测试的迭代次数
        device: 使用的设备('cuda'或'cpu')
    """
    # 转换为元素个数(使用float32，每个元素4字节)
    num_elements = size_mb * 1024 * 1024 // 4
    print(f"\n测试PyTorch DMA传输 - 数据大小: {size_mb}MB")
    
    # 检查CUDA是否可用
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，回退到CPU测试")
        device = 'cpu'
    
    # 创建主机数据
    host_data = torch.randn(num_elements, dtype=torch.float32)
    
    # 测试主机到设备(H2D)传输
    h2d_times = []
    for _ in range(iterations):
        start = time.time()
        device_data = host_data.to(device)
        torch.cuda.synchronize()  # 确保传输完成
        h2d_time = time.time() - start
        h2d_times.append(h2d_time)
    
    avg_h2d = np.mean(h2d_times[1:])  # 忽略第一次可能存在的初始化开销
    print(f"主机到设备(H2D)传输 - 平均时间: {avg_h2d:.4f}s, 带宽: {size_mb/avg_h2d:.2f} MB/s")
    
    # 测试设备到主机(D2H)传输
    d2h_times = []
    for _ in range(iterations):
        start = time.time()
        host_receive = device_data.cpu()
        torch.cuda.synchronize()  # 确保传输完成
        d2h_time = time.time() - start
        d2h_times.append(d2h_time)
    
    avg_d2h = np.mean(d2h_times[1:])
    print(f"设备到主机(D2H)传输 - 平均时间: {avg_d2h:.4f}s, 带宽: {size_mb/avg_d2h:.2f} MB/s")
    
    # 测试设备到设备(D2D)传输(如果在CUDA设备上)
    if device == 'cuda':
        device_data2 = torch.empty_like(device_data)
        
        d2d_times = []
        for _ in range(iterations):
            start = time.time()
            device_data2.copy_(device_data)
            torch.cuda.synchronize()
            d2d_time = time.time() - start
            d2d_times.append(d2d_time)
        
        avg_d2d = np.mean(d2d_times[1:])
        print(f"设备到设备(D2D)传输 - 平均时间: {avg_d2d:.4f}s, 带宽: {size_mb/avg_d2d:.2f} MB/s")

def test_pinned_memory(size_mb=100, iterations=10):
    """
    测试使用固定内存(pinned memory)的DMA性能
    
    参数:
        size_mb: 测试数据大小(MB)
        iterations: 每次测试的迭代次数
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，无法测试固定内存")
        return
    
    num_elements = size_mb * 1024 * 1024 // 4
    print(f"\n测试固定内存DMA性能 - 数据大小: {size_mb}MB")
    
    # 创建固定内存的主机数据
    host_data = torch.empty(num_elements, dtype=torch.float32).pin_memory()
    host_data.uniform_()  # 填充随机数据
    
    # 测试固定内存H2D
    h2d_times = []
    for _ in range(iterations):
        start = time.time()
        device_data = host_data.to('cuda')
        torch.cuda.synchronize()
        h2d_time = time.time() - start
        h2d_times.append(h2d_time)
    
    avg_h2d = np.mean(h2d_times[1:])
    print(f"固定内存H2D传输 - 平均时间: {avg_h2d:.4f}s, 带宽: {size_mb/avg_h2d:.2f} MB/s")
    
    # 测试固定内存D2H
    d2h_times = []
    for _ in range(iterations):
        start = time.time()
        host_receive = device_data.pin_memory().cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start
        d2h_times.append(d2h_time)
    
    avg_d2h = np.mean(d2h_times[1:])
    print(f"固定内存D2H传输 - 平均时间: {avg_d2h:.4f}s, 带宽: {size_mb/avg_d2h:.2f} MB/s")

if __name__ == "__main__":
    import os
    print(f"Python PID: {os.getpid()}")
    import time
    time.sleep(10) 
    # 显示设备信息
    if torch.cuda.is_available():
        print(f"检测到的GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("未检测到可用的CUDA设备")
    
    # 测试常规DMA传输
    test_dma_transfers(size_mb=100, iterations=10, device='cuda')
    