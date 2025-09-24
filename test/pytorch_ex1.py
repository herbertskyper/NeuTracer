# for simple python test

import torch
import os
import time
import traceback  # 导入调用栈模块

print(f"Python 进程 PID: {os.getpid()}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 设备数量: {torch.cuda.device_count()}")

# 暂停，让您有时间附加跟踪器
# print("等待 1 秒以便附加跟踪器...")
time.sleep(10)

import datetime  # 添加这个导入

# 添加用于显存碎片化的函数
def fragment_gpu_memory(num_allocations=100, min_size=1024, max_size=1024*1024):
    """
    通过分配大量不同大小的小内存块，并交替释放其中一些，
    来故意造成GPU显存碎片化
    
    参数:
        num_allocations: 要分配的内存块数量
        min_size: 最小内存块大小（字节）
        max_size: 最大内存块大小（字节）
    """
    print("\n开始执行显存碎片化...")
    
    # 记录初始显存状态
    initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"初始显存使用: {initial_memory:.2f} MB")
    
    # 创建不同大小的张量列表
    tensors = []
    
    # 第一轮：分配各种大小的内存块
    for i in range(num_allocations):
        # 随机大小，以字节为单位，转换为元素数量
        size = torch.randint(min_size, max_size, (1,)).item()
        # 将字节大小转换为元素数量 (每个浮点数占4字节)
        num_elements = size // 4
        
        # 调整为1D张量的合适大小
        tensor_size = max(1, int(num_elements ** 0.5))
        
        # 创建张量并添加到列表
        tensor = torch.rand(tensor_size, tensor_size, device='cuda')
        tensors.append(tensor)
        
        # 每分配10个张量打印一次内存状态
        if (i + 1) % 10 == 0:
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"已分配 {i+1}/{num_allocations} 个内存块，当前显存使用: {current_memory:.2f} MB")
    
    # 第二轮：释放一部分内存块（释放偶数索引的张量）
    print("\n开始释放部分内存块，制造碎片...")
    for i in range(0, len(tensors), 2):
        tensors[i] = None  # 释放偶数索引的张量
    
    # 强制垃圾回收
    torch.cuda.empty_cache()
    
    # 第三轮：再次分配一些小块
    print("\n再次分配小内存块，填充碎片...")
    for i in range(num_allocations // 2):
        size = min_size + (i % (max_size // 10))
        num_elements = size // 4
        tensor_size = max(1, int(num_elements ** 0.5))
        tensor = torch.rand(tensor_size, tensor_size, device='cuda')
        tensors.append(tensor)
    
    # 显示最终内存状态
    final_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"\n显存碎片化完成！最终显存使用: {final_memory:.2f} MB")
    print(f"显存增加量: {final_memory - initial_memory:.2f} MB")
    
    # 返回张量列表，防止它们被立即释放
    return tensors

# 修改 print_stack 函数以包含时间戳
def print_stack():
    # 获取当前时间，精确到微秒
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    stack = traceback.format_stack()[:-1]  # 排除当前函数的调用
    print(f"\n[{current_time}] 当前 Python 调用栈:")
    for line in stack:
        print(f"  {line.strip()}")
    print("----------")

# 创建多级嵌套的 CUDA 操作函数

def perform_matrix_multiply(tensor1, tensor2):
    """最内层嵌套 - 执行矩阵乘法"""
    # print("执行矩阵乘法前的调用栈:")
    # print_stack()
    return torch.matmul(tensor1, tensor2)

def create_and_run_convolution(channels_in=3, channels_out=16, kernel_size=3):
    """卷积层封装函数"""
    # print("执行卷积层前的调用栈:")
    # print_stack()
    conv = torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size).cuda()
    input_tensor = torch.randn(10, channels_in, 32, 32, device='cuda')
    return conv(input_tensor)

def apply_pooling(tensor, pool_size=2):
    """池化操作嵌套函数"""
    # print("执行池化层前的调用栈:")
    # print_stack()
    pool = torch.nn.MaxPool2d(pool_size).cuda()
    return pool(tensor)

def run_deep_learning_operations(iteration):
    """中层嵌套 - 组合多种深度学习操作"""
    # 创建 GPU 上的张量
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    # 执行矩阵乘法
    z = perform_matrix_multiply(x, y)
    
    # 卷积操作
    conv_output = create_and_run_convolution(3, 16 + iteration % 5, 3)
    
    # 池化操作
    pool_output = apply_pooling(conv_output)
    
    # 返回结果
    return z, conv_output, pool_output

def process_iteration(i):
    """外层嵌套 - 处理单次迭代"""
    if i % 10 == 0:  # 每隔10次迭代打印一次基础调用栈
        print_stack()
    
    # 执行深度学习操作
    matrix_result, conv_result, pool_result = run_deep_learning_operations(i)
    
    # 复杂的后处理
    def post_process_results():
        """内嵌的后处理函数"""
        cpu_result = matrix_result.cpu()
        result_sum = cpu_result.sum().item()
        return result_sum, conv_result.size(), pool_result.size()
    
    # 执行后处理
    sum_value, conv_shape, pool_shape = post_process_results()
    
    print(f"迭代 {i+1}: 矩阵乘法结果和: {sum_value}")
    print(f"       卷积层输出尺寸: {conv_shape}")
    print(f"       池化层输出尺寸: {pool_shape}")
    
    return sum_value

# 主循环
# 修改后的主循环
def main_loop():
    """最外层函数 - 主循环"""
    # 在函数开始时初始化 fragmented_tensors
    fragmented_tensors = []
    
    # 在开始主循环前执行显存碎片化
    print("开始显存碎片化...")
    fragmented_tensors = fragment_gpu_memory(num_allocations=20000, min_size=1024, max_size=512*1024)
    print("显存已碎片化，开始主循环...\n")
    for i in range(2):
        try:
            print("\n尝试分配超大内存...")
            # 计算当前可用显存
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            print(f"当前剩余显存: {free_memory / (1024**3):.2f} GB")
            
            # 尝试分配比可用显存大 2 倍的内存
            giant_tensor_size = int((free_memory * 2) / 4)  # 每个浮点数占4字节
            giant_tensor_side = int(giant_tensor_size ** 0.5)*1000 + 1  # 将字节转换为二维张量尺寸
            
            print(f"尝试分配 {giant_tensor_side}x{giant_tensor_side} 的张量 (约 {(giant_tensor_side**2*4)/(1024**3):.2f} GB)")
            giant_tensor = torch.ones(giant_tensor_side, giant_tensor_side, device='cuda')
            print("警告：超大内存分配成功，没有报错！")
            
        except RuntimeError as e:
            print(f"预期中的 CUDA 错误: {e}")
    
    for i in range(20):
        try:
            # 每20次迭代再次执行碎片化，保持碎片状态
            if i > 0 and i % 20 == 0:
                print("\n再次执行显存碎片化...")
                more_tensors = fragment_gpu_memory(num_allocations=10000, min_size=2048, max_size=256*1024)
                # 将新的张量添加到列表中
                fragmented_tensors.extend(more_tensors[:len(more_tensors)//2])  # 只保留一半，让另一半被释放
            
            result = process_iteration(i)
            # 增加一些变化性
            if i % 3 == 0:
                # 额外的嵌套操作
                def extra_operation():
                    extra_tensor = torch.randn(500, 500, device='cuda')
                    extra_result = torch.nn.functional.relu(extra_tensor)
                    return extra_result.sum().item()
                
                extra_value = extra_operation()
                print(f"       额外操作结果: {extra_value}")
            
            # 打印当前显存状态
            if i % 10 == 0:
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"当前显存使用: {current_memory:.2f} MB")
            
            time.sleep(0.2)  # 短暂延迟
        except Exception as e:
            print(f"迭代 {i} 出错: {e}")
            traceback.print_exc()
            
    # 循环结束后，打印最终内存状态
    print("\n主循环结束")
    print(f"最终显存使用: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
    
    # 清理显存，可选
    fragmented_tensors = None
    torch.cuda.empty_cache()
    print(f"清理后显存使用: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")

# 启动主循环
if __name__ == "__main__":
    main_loop()