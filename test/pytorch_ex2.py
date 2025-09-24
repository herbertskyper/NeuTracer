# for cuda api test(cudalaunchkernel cudamalloc cudafree cudamemcpy)

import torch
import os
import time
import gc
import traceback
import datetime
import numpy as np
import multiprocessing as mp
import sys
from torch.utils.data import DataLoader, TensorDataset

print(f"Python 进程 PID: {os.getpid()}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 设备数量: {torch.cuda.device_count()}")
print(f"CUDA 设备名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 暂停，让您有时间附加跟踪器
print("等待 5 秒以便附加跟踪器...")
time.sleep(5)

def print_memory_stats(prefix=""):
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"[{current_time}] {prefix} GPU内存 - 已分配: {allocated:.2f}MB, 已保留: {reserved:.2f}MB")

def print_stack():
    """打印当前调用栈，带时间戳"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    stack = traceback.format_stack()[:-1]
    print(f"\n[{current_time}] 当前 Python 调用栈:")
    for line in stack:
        print(f"  {line.strip()}")
    print("----------")

def force_cuda_allocation(size_mb, tag=""):
    """强制分配指定大小的CUDA内存"""
    print_stack()
    print_memory_stats(f"分配前({tag})")
    tensor = torch.ones((size_mb * 256, 1024), device='cuda')  # 约1MB=1024*1024字节
    print_memory_stats(f"分配后({tag})")
    return tensor

def force_cuda_free(tensor, tag=""):
    """强制释放CUDA内存"""
    print_memory_stats(f"释放前({tag})")
    del tensor
    torch.cuda.empty_cache()  # 尝试释放未使用的缓存
    print_memory_stats(f"释放后({tag})")
    return None

def create_model_with_buffers(complexity=1):
    """创建包含多个缓冲区的模型，触发多次内存分配"""
    class ComplexModel(torch.nn.Module):
        def __init__(self, complexity):
            super().__init__()
            self.complexity = complexity
            self.input_size = 32  # 输入图像大小
            
            # 创建多个层，每个层触发单独的内存分配
            self.features = torch.nn.ModuleList()
            for i in range(complexity):
                # 每个特征块包含卷积、批归一化和池化，会触发多次内存分配
                self.features.append(torch.nn.Sequential(
                    torch.nn.Conv2d(3 if i == 0 else 16*2**min(i, 3), 16*2**min(i+1, 3), 3, padding=1),
                    torch.nn.BatchNorm2d(16*2**min(i+1, 3)),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2)
                ))
            
            # 计算最终特征图大小
            # 每个池化层将大小减半
            final_size = self.input_size // (2 ** complexity)
            # 最小为1x1，避免出现小于1的情况
            final_size = max(1, final_size)
            
            # 计算展平后的特征维度
            final_channels = 16 * (2 ** min(complexity, 3))
            flattened_features = final_channels * final_size * final_size
            
            print(f"模型特征维度: {final_channels}x{final_size}x{final_size} = {flattened_features}")
            
            # 最后的分类层
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(flattened_features, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, 10)
            )
            
            # 注册缓冲区，进一步增加内存分配
            for i in range(complexity):
                size = 1000 * (i + 1)
                self.register_buffer(f'dummy_buffer_{i}', torch.randn(size, device='cuda'))
        
        def forward(self, x):
            # 调试输出
            if self.training:
                print(f"输入形状: {x.shape}")
                
            for i, feature in enumerate(self.features):
                x = feature(x)
                if self.training:
                    print(f"第 {i+1} 特征层后形状: {x.shape}")
            
            # 展平特征图
            x = x.view(x.size(0), -1)
            if self.training:
                print(f"展平后形状: {x.shape}")
                print(f"分类器第一层权重形状: {self.classifier[0].weight.shape}")
            
            return self.classifier(x)
    
    print_memory_stats("模型创建前")
    model = ComplexModel(complexity).cuda()
    print_memory_stats("模型创建后")
    return model

def train_for_iterations(model, iterations=5, batch_size=32, input_size=(3, 32, 32)):
    """
    运行多次训练迭代，每次迭代触发新的内存分配和释放
    主要测试cudaMalloc和cudaFree的调用
    """
    # 创建随机数据
    input_shape = (batch_size,) + input_size
    data = torch.randn(500, *input_size, device='cuda')
    labels = torch.randint(0, 10, (500,), device='cuda')
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 设置优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    for iteration in range(iterations):
        print(f"\n===== 开始迭代 {iteration+1}/{iterations} =====")
        total_loss = 0.0
        batch_count = 0
        
        # 在每个迭代开始时，强制分配一些额外内存（模拟缓存）
        cache_tensors = []
        if iteration % 2 == 0:  # 偶数迭代时分配额外内存
            for i in range(2):
                cache_tensors.append(force_cuda_allocation(
                    20 + (iteration*5), f"迭代{iteration+1}临时缓存{i+1}"))
        
        for inputs, targets in dataloader:
            # 强制执行垃圾回收，触发可能的内存释放
            gc.collect()
            
            try:
                # 前向传播
                print_memory_stats(f"迭代{iteration+1}批次{batch_count+1}前向传播前")
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                print_memory_stats(f"迭代{iteration+1}批次{batch_count+1}前向传播后")
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            except Exception as e:
                print(f"训练中发生错误: {e}")
                traceback.print_exc()
                # 继续下一个批次
            
            batch_count += 1
            if batch_count >= 3:  # 每个迭代仅处理3个批次
                break
        
        # 手动释放一些缓存内存
        if len(cache_tensors) > 0:
            print("\n>> 释放临时缓存...")
            for i, tensor in enumerate(cache_tensors):
                force_cuda_free(tensor, f"迭代{iteration+1}临时缓存{i+1}释放")
            cache_tensors = []
        
        # 每隔一定迭代，重建某些模型组件以触发更多的内存操作
        if iteration % 2 == 1:  # 奇数迭代时重建部分模型
            print("\n>> 重建模型组件...")
            # 备份旧参数
            print_memory_stats("重建模型前")
            
            try:
                # 保存第一层的输入特征大小
                in_features = model.classifier[0].in_features
                
                # 替换分类器，触发新内存分配和旧内存释放
                old_classifier = model.classifier
                model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(in_features, 128 + (iteration * 10) % 64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(128 + (iteration * 10) % 64, 10)
                ).cuda()
                
                # 强制释放旧分类器
                del old_classifier
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"重建模型时发生错误: {e}")
                
            print_memory_stats("重建模型后")
        
        if batch_count > 0:
            print(f"迭代 {iteration+1} 平均损失: {total_loss/batch_count:.4f}")
        
        # 周期性地尝试清空缓存和收集垃圾
        if iteration % 2 == 0:
            print("\n>> 强制清理内存...")
            print_memory_stats("清理前")
            gc.collect()
            torch.cuda.empty_cache()
            print_memory_stats("清理后")
    
    return model


def test_large_model_transfers(iterations=10, model_complexity=50):
    """
    测试将大型神经网络模型在GPU和CPU之间反复传输，
    并确保每次迭代都清空缓存
    测试cudaMemcpy的调用
    """
    print("\n===== 开始大型模型 GPU-CPU 传输测试 =====")
    print(f"当前进程 PID: {os.getpid()}")
    
    print_memory_stats("大型模型创建前")
    # 创建一个更复杂的大型模型
    model = create_model_with_buffers(complexity=model_complexity)
    print(f"创建了复杂度为 {model_complexity} 的大型模型")
    print_memory_stats("大型模型创建后")
    
    # 添加更多参数，增加模型大小
    # 添加大量的额外缓冲区，增加内存使用量
    for i in range(10):
        buffer_size = 500000 * (i + 1)  # 约2MB每个缓冲区
        model.register_buffer(f'extra_large_buffer_{i}', torch.randn(buffer_size, device='cuda'))
        print(f"添加了额外缓冲区 {i+1}，大小: {buffer_size * 4 / (1024*1024):.2f}MB")
    
    print_memory_stats("添加额外缓冲区后")
    
    # 记录模型大小
    param_size = 0
    buffer_size = 0
    
    # 计算参数大小
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # 计算缓冲区大小
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    print(f"模型总大小: {total_size / (1024*1024):.2f}MB")
    print(f"  - 参数大小: {param_size / (1024*1024):.2f}MB")
    print(f"  - 缓冲区大小: {buffer_size / (1024*1024):.2f}MB")
    
    # 执行多次传输
    for i in range(iterations):
        print(f"\n----- 大型模型传输迭代 {i+1}/{iterations} -----")
        
        # 在每次迭代开始前清空缓存
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_stats("迭代开始前缓存清理后")
        
        # GPU -> CPU 传输
        print("\n>> 大型模型 GPU到CPU传输测试")
        print_memory_stats("GPU->CPU 传输前")
        print_stack()
        start_time = time.time()
        cpu_model = model.cpu()
        end_time = time.time()
        transfer_time = end_time - start_time
        print(f"GPU->CPU 传输完成: {transfer_time:.4f} 秒")
        transfer_speed = total_size / transfer_time / (1024*1024)
        print(f"传输速度: {transfer_speed:.2f} MB/s")
        device = next(cpu_model.parameters()).device
        print(f"模型当前位置: {device}")
        print_memory_stats("GPU->CPU 传输后")
        
        # 清空GPU缓存
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_stats("GPU->CPU 传输后缓存清理")
        
        # 稍微修改一下CPU上的模型数据
        for name, param in cpu_model.named_parameters():
            if 'weight' in name and i % 2 == 0:
                with torch.no_grad():
                    noise = torch.randn_like(param.data) * 0.01
                    param.data = param.data + noise
        
        # CPU -> GPU 传输
        print("\n>> 大型模型 CPU到GPU传输测试")
        print_memory_stats("CPU->GPU 传输前")
        print_stack()
        start_time = time.time()
        gpu_model = cpu_model.cuda()
        torch.cuda.synchronize()  # 确保传输完成
        end_time = time.time()
        transfer_time = end_time - start_time
        print(f"CPU->GPU 传输完成: {transfer_time:.4f} 秒")
        transfer_speed = total_size / transfer_time / (1024*1024)
        print(f"传输速度: {transfer_speed:.2f} MB/s")
        device = next(gpu_model.parameters()).device
        print(f"模型当前位置: {device}")
        print_memory_stats("CPU->GPU 传输后")
        
        # 确保模型引用正确
        model = gpu_model
        
        # 更新模型的一些参数以避免优化
        if i % 2 == 0:
            # 更新随机权重以确保传输不被优化掉
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.data = param.data + torch.randn_like(param.data) * 0.01
        
        # 清空CPU上模型的引用和其他临时变量
        del cpu_model
        gc.collect()
        
        # 每隔几次迭代执行一次推理，确保模型功能正常
        if i % 2 == 0:
            try:
                print("\n>> 执行大型模型测试推理")
                dummy_input = torch.randn(2, 3, 32, 32, device='cuda')
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"推理输出形状: {output.shape}")
                # 清理推理后的缓存
                del dummy_input, output
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"推理失败: {e}")
                traceback.print_exc()
        
        # 每次迭代结束时清空缓存
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_stats("迭代结束后缓存清理")
        
        # 在多次迭代之间添加短暂延迟
        print("等待 1 秒...")
        time.sleep(1)  # 1秒延迟，便于观察
    
    print("\n===== 大型模型传输测试完成 =====")
    print_memory_stats("最终状态")
    
    # 手动清理模型和其他资源
    print("清理所有资源...")
    del model, gpu_model
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_stats("清理后")
    
    return True

def test_custom_cuda_operations(size_mb=50):
    """
    测试自定义 CUDA 操作，触发更多 cudaMemcpy 调用
    """
    print("\n===== 开始自定义 CUDA 操作测试 =====")
    print_memory_stats("测试开始")
    
    # 创建用于操作的张量
    a = torch.rand((size_mb * 64, 1024), device='cuda')
    b = torch.rand((size_mb * 64, 1024), device='cuda')
    
    # 执行多个 CUDA 操作，这些会创建新的临时张量并触发 cudaMemcpy
    print("\n>> 执行 CUDA 张量操作")
    start_time = time.time()
    
    # 1. 逐元素操作
    c = a * b
    
    # 2. 矩阵乘法（会触发大量 CUDA 核心运算和内存操作）
    try:
        a_2d = a[:1024, :1024]
        b_2d = b[:1024, :1024]
        d = torch.matmul(a_2d, b_2d)
    except Exception as e:
        print(f"矩阵乘法失败: {e}")
        d = a[:100, :100] @ b[:100, :100]
    
    # 3. 操作不同设备上的数据
    cpu_tensor = torch.rand((1000, 1000))
    e = c[:100, :100] + cpu_tensor[:100, :100].cuda()
    
    # 4. 多次在 CPU 和 GPU 之间复制相同数据
    for i in range(5):
        temp = d.cpu()
        d_copy = temp.cuda()
    
    # 5. 使用 torch.to() 在设备之间传输
    f = e.to('cpu').to('cuda')
    
    end_time = time.time()
    print(f"CUDA 操作完成: {end_time - start_time:.4f} 秒")
    print_memory_stats("CUDA 操作后")
    
    # 清理
    del a, b, c, d, e, f, cpu_tensor
    torch.cuda.empty_cache()
    print_memory_stats("内存清理后")

def run_inference_tests(model, count=3):
    """运行几次推理，触发更多内存操作"""
    print("\n===== 开始推理测试 =====")
    
    for i in range(count):
        try:
            # 创建不同大小的输入，触发不同大小的内存分配
            size = 32 * (i + 1)
            batch_size = 4 * (i + 1)
            
            print(f"\n>> 推理测试 {i+1}: 大小={size}, 批次大小={batch_size}")
            input_tensor = torch.randn(batch_size, 3, size, size, device='cuda')
            
            print_memory_stats("推理前")
            # 执行推理
            with torch.no_grad():
                try:
                    # 禁用训练模式以避免调试输出
                    model.train(False)
                    output = model(input_tensor)
                    print(f"推理输出形状: {output.shape}")
                except Exception as e:
                    print(f"推理失败: {e}")
                finally:
                    # 恢复训练模式
                    model.train(True)
            print_memory_stats("推理后")
        except Exception as e:
            print(f"推理测试 {i+1} 失败: {e}")
        finally:
            # 手动释放输入张量
            try:
                del input_tensor
                torch.cuda.empty_cache()
            except:
                pass
            
def test_explicit_cuda_api_calls():
    """
    使用 ctypes 直接调用 CUDA Runtime API
    主要测试cuda API的直接调用
    """
    import ctypes
    from ctypes import c_void_p, c_size_t, c_int, byref, CDLL
    
    print("\n===== 开始显式 CUDA API 调用测试 =====")
    
    # 加载 CUDA Runtime 库
    try:
        cuda = CDLL('libcudart.so')
        print("成功加载 CUDA Runtime 库")
    except Exception as e:
        print(f"无法加载 CUDA Runtime 库: {e}")
        return
    
    # 定义一些常量
    cudaSuccess = 0
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    
    # 创建测试数据
    size = 10 * 1024 * 1024  # 10MB
    h_data = (ctypes.c_float * (size // 4))()
    for i in range(len(h_data)):
        h_data[i] = 1.0
    
    print(f"已创建主机内存数据，大小: {size / (1024*1024):.2f} MB")
    
    # 分配设备内存 - cudaMalloc
    d_data = c_void_p()
    print("\n>> 调用 cudaMalloc API")
    print_stack()
    ret = cuda.cudaMalloc(byref(d_data), size)
    if ret != cudaSuccess:
        print(f"cudaMalloc 失败，错误码: {ret}")
        return
    print(f"cudaMalloc 成功，分配了 {size / (1024*1024):.2f} MB")
    print_memory_stats("cudaMalloc 后")
    
    # 从主机复制到设备 - cudaMemcpy (H2D)
    print("\n>> 调用 cudaMemcpy API (H2D)")
    print_stack()
    start_time = time.time()
    ret = cuda.cudaMemcpy(d_data, ctypes.cast(h_data, c_void_p), size, cudaMemcpyHostToDevice)
    if ret != cudaSuccess:
        print(f"cudaMemcpy H2D 失败，错误码: {ret}")
        cuda.cudaFree(d_data)
        return
    end_time = time.time()
    print(f"cudaMemcpy H2D 成功: {end_time - start_time:.4f} 秒")
    print_memory_stats("cudaMemcpy H2D 后")
    
    # 从设备复制到主机 - cudaMemcpy (D2H)
    h_result = (ctypes.c_float * (size // 4))()
    print("\n>> 调用 cudaMemcpy API (D2H)")
    print_stack()
    start_time = time.time()
    ret = cuda.cudaMemcpy(ctypes.cast(h_result, c_void_p), d_data, size, cudaMemcpyDeviceToHost)
    if ret != cudaSuccess:
        print(f"cudaMemcpy D2H 失败，错误码: {ret}")
        cuda.cudaFree(d_data)
        return
    end_time = time.time()
    print(f"cudaMemcpy D2H 成功: {end_time - start_time:.4f} 秒")
    print_memory_stats("cudaMemcpy D2H 后")
    
    # 验证数据
    valid = True
    for i in range(min(10, len(h_result))):
        if h_result[i] != 1.0:
            valid = False
            print(f"数据验证失败: h_result[{i}] = {h_result[i]}")
            break
    if valid:
        print("数据验证成功")
    
    # 分配第二块设备内存用于设备间复制
    d_data2 = c_void_p()
    print("\n>> 调用第二次 cudaMalloc API")
    ret = cuda.cudaMalloc(byref(d_data2), size)
    if ret == cudaSuccess:
        # 设备到设备复制 - cudaMemcpy (D2D)
        print("\n>> 调用 cudaMemcpy API (D2D)")
        start_time = time.time()
        ret = cuda.cudaMemcpy(d_data2, d_data, size, 3)  # 3 = cudaMemcpyDeviceToDevice
        end_time = time.time()
        if ret == cudaSuccess:
            print(f"cudaMemcpy D2D 成功: {end_time - start_time:.4f} 秒")
        else:
            print(f"cudaMemcpy D2D 失败，错误码: {ret}")
        
        # 释放第二块内存
        print("\n>> 调用第二次 cudaFree API")
        cuda.cudaFree(d_data2)
    
    # 释放设备内存 - cudaFree
    print("\n>> 调用 cudaFree API")
    print_stack()
    ret = cuda.cudaFree(d_data)
    if ret != cudaSuccess:
        print(f"cudaFree 失败，错误码: {ret}")
    else:
        print("cudaFree 成功")
    print_memory_stats("cudaFree 后")
    
    print("\n===== 显式 CUDA API 调用测试完成 =====")

def worker_process(worker_id, size_mb=100, iterations=10, delay=1.0):
    """工作进程函数，执行 CUDA 操作"""
    print(f"工作进程 {worker_id} 启动 (PID: {os.getpid()})")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    try:
        # 等待一段时间以便于附加分析工具
        print(f"工作进程 {worker_id} 等待 {delay} 秒...")
        time.sleep(delay)
        
        tensors = []
        
        # 分配内存
        print(f"工作进程 {worker_id} 开始分配内存...")
        for i in range(iterations):
            print(f"工作进程 {worker_id} 迭代 {i+1}/{iterations}")
            print_memory_stats(f"工作进程 {worker_id} 分配前")
            
            # 分配 GPU 内存并执行一些操作
            tensor = torch.ones((size_mb * 256, 1024), device='cuda')
            tensors.append(tensor)
            
            # 执行一些 CUDA 操作
            result = tensor * 2.0
            tensor.copy_(result)
            
            print_memory_stats(f"工作进程 {worker_id} 分配后")
            
            # 定期执行内存复制操作
            if i % 2 == 0:
                print(f"工作进程 {worker_id} 执行 CPU<->GPU 内存复制")
                cpu_tensor = tensor.cpu()
                new_gpu_tensor = cpu_tensor.cuda()
                tensors.append(new_gpu_tensor)
            
            # 根据设置的延迟暂停
            time.sleep(delay)
        
        # 释放一部分内存
        num_to_free = len(tensors) // 2
        print(f"工作进程 {worker_id} 释放 {num_to_free} 个张量...")
        for i in range(num_to_free):
            del tensors[0]
            torch.cuda.empty_cache()
        
        # 简单的矩阵运算测试
        print(f"工作进程 {worker_id} 执行矩阵运算...")
        matrix_size = 1000
        a = torch.randn((matrix_size, matrix_size), device='cuda')
        b = torch.randn((matrix_size, matrix_size), device='cuda')
        c = torch.matmul(a, b)
        print(f"矩阵乘法结果形状: {c.shape}")
        
        # 释放所有内存
        print(f"工作进程 {worker_id} 释放所有内存...")
        del tensors, a, b, c
        torch.cuda.empty_cache()
        print_memory_stats(f"工作进程 {worker_id} 结束")
        
    except Exception as e:
        print(f"工作进程 {worker_id} 发生错误: {str(e)}")
        
    finally:
        print(f"工作进程 {worker_id} 结束运行")
        # 显式退出，确保进程终止
        sys.exit(0)

def run_multiprocess_test(num_processes=3, size_mb=100, iterations=10, delay=1.0):
    """
    启动多个工作进程
    测试多进程下的 CUDA 操作
    """
    print(f"主进程 PID: {os.getpid()}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    print(f"CUDA 设备名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    print(f"启动 {num_processes} 个工作进程...")
    
    # 创建并启动工作进程
    processes = []
    for i in range(num_processes):
        # 每个进程的内存大小和迭代次数略有不同，使测试更多样化
        p_size = size_mb + (i * 10)
        p_iterations = iterations + (i % 3)
        # 延迟启动，以便于工具连接
        p_delay = delay + (i * 0.5)
        
        p = mp.Process(
            target=worker_process, 
            args=(i+1, p_size, p_iterations, p_delay)
        )
        processes.append(p)
        p.start()
        print(f"已启动工作进程 {i+1}, PID: {p.pid}")
        
        # 主进程稍作等待，避免所有进程同时启动
        time.sleep(1.0)
    
    # 等待所有进程完成
    print("等待所有工作进程完成...")
    for i, p in enumerate(processes):
        p.join()
        print(f"工作进程 {i+1} 已结束")
    
    print("所有工作进程已完成")


def test_error_cuda_operations():
    """
    测试 CUDA 操作中的错误处理
    为了触发cudaMalloc时的错误，尝试分配一个过大的内存块
    """
    print("\n===== 测试 CUDA 操作中的错误处理 =====")
    print_memory_stats("测试 CUDA 操作中的错误处理前")
    try:
        # 尝试分配一个过大的内存块
        time.sleep(2)
        d_data = torch.empty(1000000000000, device='cuda')
        time.sleep(2)  # 确保分配完成
    except RuntimeError as e:
        print(f"捕获到 RuntimeError: {str(e)}")
    finally:
        print_memory_stats("测试 CUDA 操作中的错误处理后")
        
        
def test_memory_leak_and_oom(duration_minutes=5, oom_interval_sec=30):
    """
    持续分配内存而不释放，模拟内存泄漏，并定期触发OOM错误
    
    Args:
        duration_minutes: 测试持续时间（分钟）
        oom_interval_sec: 触发OOM测试的间隔（秒）
    """
    print(f"\n===== 开始内存泄漏和OOM测试 =====")
    print(f"测试持续时间: {duration_minutes} 分钟")
    print(f"OOM测试间隔: {oom_interval_sec} 秒")
    print(f"当前进程 PID: {os.getpid()}")
    
    # 存储分配的张量，模拟内存泄漏
    leaked_tensors = []
    allocation_counter = 0
    start_time = time.time()
    last_oom_time = start_time
    
    # 获取GPU总内存信息
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        total_memory = gpu_properties.total_memory
        print(f"GPU总内存: {total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA不可用，退出测试")
        return
    
    print_memory_stats("测试开始前")
    
    try:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 检查是否达到测试时间限制
            if elapsed_time > duration_minutes * 60:
                print(f"\n测试时间已达到 {duration_minutes} 分钟，结束测试")
                break
            
            allocation_counter += 1
            
            # 动态决定分配策略
            if allocation_counter % 10 == 0:
                # 每10次分配一个大内存块 (50-200MB)
                size_mb = np.random.randint(50, 201)
                block_type = "大内存块"
                print(f"\n[{allocation_counter:04d}] 分配{block_type}: {size_mb}MB")
                
                try:
                    print_stack()
                    print_memory_stats(f"分配{block_type}前")
                    tensor = torch.ones((size_mb * 256, 1024), device='cuda', dtype=torch.float32)
                    leaked_tensors.append(tensor)
                    print_memory_stats(f"分配{block_type}后")
                    print(f"✓ 成功分配{block_type} {size_mb}MB，累计泄漏张量: {len(leaked_tensors)}")
                    
                except RuntimeError as e:
                    print(f"✗ 分配{block_type}失败: {str(e)}")
                    if "out of memory" in str(e).lower():
                        print("检测到显存不足错误")
                    
            elif allocation_counter % 3 == 0:
                # 每3次分配一个中等内存块 (10-30MB)
                size_mb = np.random.randint(10, 31)
                block_type = "中等内存块"
                print(f"\n[{allocation_counter:04d}] 分配{block_type}: {size_mb}MB")
                
                try:
                    print_memory_stats(f"分配{block_type}前")
                    tensor = torch.randn((size_mb * 256, 1024), device='cuda', dtype=torch.float32)
                    leaked_tensors.append(tensor)
                    print_memory_stats(f"分配{block_type}后")
                    print(f"✓ 成功分配{block_type} {size_mb}MB，累计泄漏张量: {len(leaked_tensors)}")
                    
                except RuntimeError as e:
                    print(f"✗ 分配{block_type}失败: {str(e)}")
                    if "out of memory" in str(e).lower():
                        print("检测到显存不足错误")
                        
            else:
                # 大部分时间分配小内存碎片 (1-5MB)
                size_mb = np.random.randint(1, 6)
                block_type = "小内存碎片"
                print(f"[{allocation_counter:04d}] 分配{block_type}: {size_mb}MB", end="")
                
                try:
                    tensor = torch.zeros((size_mb * 256, 1024), device='cuda', dtype=torch.float32)
                    leaked_tensors.append(tensor)
                    print(f" ✓ 成功，累计: {len(leaked_tensors)}")
                    
                    # 每50个小碎片打印一次内存状态
                    if allocation_counter % 50 == 0:
                        print_memory_stats(f"小碎片分配第{allocation_counter}次后")
                        
                except RuntimeError as e:
                    print(f" ✗ 失败: {str(e)}")
                    if "out of memory" in str(e).lower():
                        print("检测到显存不足错误")
            
            # 定期尝试分配超大内存导致OOM
            if current_time - last_oom_time >= oom_interval_sec:
                print(f"\n{'='*60}")
                print(f"[OOM测试] 第 {int((current_time - start_time) // oom_interval_sec) + 1} 次OOM测试")
                print(f"{'='*60}")
                
                try:
                    # 获取当前可用内存
                    current_allocated = torch.cuda.memory_allocated()
                    current_reserved = torch.cuda.memory_reserved()
                    free_memory = total_memory - current_reserved
                    
                    print(f"当前已分配: {current_allocated / (1024**3):.2f} GB")
                    print(f"当前已保留: {current_reserved / (1024**3):.2f} GB") 
                    print(f"估计可用: {free_memory / (1024**3):.2f} GB")
                    
                    # 尝试分配比可用内存大很多的内存
                    oom_size_bytes = int(free_memory * 3)  # 3倍于可用内存
                    oom_size_mb = oom_size_bytes // (1024 * 1024)
                    
                    print(f"\n>> 尝试分配超大内存块: {oom_size_mb}MB ({oom_size_bytes / (1024**3):.2f} GB)")
                    print(">> 预期此操作将失败并产生OOM错误...")
                    
                    print_stack()  # 打印调用栈
                    print_memory_stats("OOM测试前")
                    
                    # 这应该会失败
                    oom_tensor = torch.ones(oom_size_bytes // 4, device='cuda', dtype=torch.float32)
                    print("⚠️  警告: 超大内存分配竟然成功了！这可能表明显存检测有误")
                    leaked_tensors.append(oom_tensor)
                    
                except RuntimeError as e:
                    print(f"✓ 预期的OOM错误发生: {str(e)}")
                    if "out of memory" in str(e).lower():
                        print("✓ 确认这是显存不足错误")
                        
                        # 尝试获取详细的CUDA错误信息
                        try:
                            print(f"CUDA最后错误: {torch.cuda.get_device_name(0)}")
                            print(f"CUDA内存使用情况:")
                            print(f"  分配的内存: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                            print(f"  缓存的内存: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
                        except:
                            pass
                    else:
                        print(f"✗ 意外的错误类型: {str(e)}")
                        
                except Exception as e:
                    print(f"✗ 意外的异常: {str(e)}")
                    traceback.print_exc()
                
                finally:
                    print_memory_stats("OOM测试后")
                    last_oom_time = current_time
                    print(f"{'='*60}")
            
            # 记录统计信息
            if allocation_counter % 100 == 0:
                elapsed_minutes = elapsed_time / 60
                remaining_minutes = duration_minutes - elapsed_minutes
                total_leaked_memory = sum(t.numel() * t.element_size() for t in leaked_tensors)
                
                print(f"\n{'*'*60}")
                print(f"[进度报告] 分配次数: {allocation_counter}")
                print(f"[进度报告] 已用时间: {elapsed_minutes:.1f} 分钟")
                print(f"[进度报告] 剩余时间: {remaining_minutes:.1f} 分钟")
                print(f"[进度报告] 泄漏张量数: {len(leaked_tensors)}")
                print(f"[进度报告] 估计泄漏内存: {total_leaked_memory / (1024**3):.2f} GB")
                print_memory_stats("进度报告")
                print(f"{'*'*60}")
            
            # 短暂延迟，避免过于频繁的分配
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\n用户中断测试 (Ctrl+C)")
        
    except Exception as e:
        print(f"\n测试过程中发生意外错误: {str(e)}")
        traceback.print_exc()
        
    finally:
        # 测试结束统计
        end_time = time.time()
        total_duration = end_time - start_time
        total_leaked_memory = sum(t.numel() * t.element_size() for t in leaked_tensors) if leaked_tensors else 0
        
        print(f"\n{'='*80}")
        print(f"内存泄漏和OOM测试完成")
        print(f"{'='*80}")
        print(f"总测试时间: {total_duration / 60:.2f} 分钟")
        print(f"总分配次数: {allocation_counter}")
        print(f"最终泄漏张量数: {len(leaked_tensors)}")
        print(f"最终估计泄漏内存: {total_leaked_memory / (1024**3):.2f} GB")
        print_memory_stats("测试结束时")
        
        # 询问是否清理内存
        print(f"\n注意: 当前有 {len(leaked_tensors)} 个张量未释放")
        print(f"这些张量将在程序结束时自动释放")
        print(f"如需手动清理，可以调用 torch.cuda.empty_cache()")

def test_fragmentation_pattern():
    """
    专门测试显存碎片化模式
    通过多轮分配和释放不规则大小的内存块来制造更严重的碎片化
    """
    print(f"\n===== 开始显存碎片化模式测试 =====")
    
    all_tensors = []
    fragmentation_rounds = 50  # 增加碎片化轮数
    
    try:
        # 获取GPU总内存信息
        if torch.cuda.is_available():
            gpu_properties = torch.cuda.get_device_properties(0)
            total_memory = gpu_properties.total_memory
            print(f"GPU总内存: {total_memory / (1024**3):.2f} GB")
        
        for round_num in range(fragmentation_rounds):
            print(f"\n{'='*70}")
            print(f"第 {round_num + 1}/{fragmentation_rounds} 轮碎片化测试")
            print(f"{'='*70}")
            
            # 第一阶段：分配各种大小的内存块
            print(f"\n>> 第一阶段: 分配不规则大小的内存块 (轮次 {round_num + 1})")
            
            # 每轮使用不同的大小模式
            if round_num == 0:
                # 第一轮：较大的不规则块
                sizes_mb = [5, 12, 28, 65, 134, 257, 512, 789, 1024]
            elif round_num == 1:
                # 第二轮：中等大小的块
                sizes_mb = [3, 7, 15, 31, 63, 127, 255, 378, 501]
            elif round_num == 2:
                # 第三轮：小块混合
                sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            elif round_num == 3:
                # 第四轮：随机大小
                sizes_mb = [np.random.randint(1, 200) for _ in range(12)]
            else:
                # 第五轮：极其不规则的大小
                sizes_mb = [1, 3, 9, 27, 81, 243, 45, 135, 405, 89, 267, 801]
            
            print(f"本轮将分配的内存块大小: {sizes_mb} MB")
            
            round_tensors = []  # 当前轮次的张量
            
            for i, size_mb in enumerate(sizes_mb):
                try:
                    print(f"  [轮次{round_num+1}] 分配内存块 {i+1}/{len(sizes_mb)}: {size_mb}MB")
                    print_stack()
                    print_memory_stats(f"轮次{round_num+1}分配块{i+1}前")
                    
                    tensor = torch.ones((size_mb * 256, 1024), device='cuda', dtype=torch.float32)
                    tensor_info = (f"round{round_num+1}_block_{i+1}_{size_mb}MB", tensor)
                    all_tensors.append(tensor_info)
                    round_tensors.append(tensor_info)
                    
                    print_memory_stats(f"轮次{round_num+1}分配块{i+1}后")
                    print(f"  ✓ 成功分配，当前总张量数: {len(all_tensors)}")
                    
                    # 添加延迟，便于观察
                    time.sleep(0.2)
                    
                except RuntimeError as e:
                    print(f"  ✗ 分配失败: {e}")
                    if "out of memory" in str(e).lower():
                        print(f"  显存不足，停止当前轮次的分配")
                        break
                    
            print(f"\n轮次 {round_num + 1} 分配阶段完成，成功分配 {len(round_tensors)} 个内存块")
            print_memory_stats(f"轮次{round_num+1}分配完成")
            
            # 第二阶段：有策略地释放内存块，制造碎片
            print(f"\n>> 第二阶段: 有策略地释放内存块制造碎片 (轮次 {round_num + 1})")
            
            if len(round_tensors) > 0:
                # 使用不同的释放策略
                if round_num % 3 == 0:
                    # 策略1：释放奇数位置的块
                    indices_to_free = [i for i in range(len(round_tensors)) if i % 2 == 1]
                    strategy_name = "奇数位置释放"
                elif round_num % 3 == 1:
                    # 策略2：释放中间部分的块
                    start = len(round_tensors) // 4
                    end = 3 * len(round_tensors) // 4
                    indices_to_free = list(range(start, end))
                    strategy_name = "中间部分释放"
                else:
                    # 策略3：随机释放60%的块
                    num_to_free = int(len(round_tensors) * 0.6)
                    indices_to_free = np.random.choice(len(round_tensors), num_to_free, replace=False).tolist()
                    strategy_name = "随机60%释放"
                
                print(f"  使用策略: {strategy_name}")
                print(f"  将释放 {len(indices_to_free)}/{len(round_tensors)} 个内存块")
                
                # 执行释放
                freed_count = 0
                for idx in sorted(indices_to_free, reverse=True):
                    if idx < len(round_tensors):
                        name, tensor = round_tensors[idx]
                        print(f"    [轮次{round_num+1}] 释放 {name}")
                        
                        # 从总列表中移除
                        all_tensors = [(n, t) for n, t in all_tensors if n != name]
                        
                        del tensor
                        round_tensors.pop(idx)
                        freed_count += 1
                        
                        # 每释放几个块就清空一次缓存
                        if freed_count % 3 == 0:
                            torch.cuda.empty_cache()
                            print_memory_stats(f"轮次{round_num+1}释放{freed_count}个块后")
                        
                        time.sleep(0.1)  # 短暂延迟
                
                print(f"  轮次 {round_num + 1} 释放阶段完成，释放了 {freed_count} 个内存块")
                print(f"  剩余张量数: {len(all_tensors)}")
                print_memory_stats(f"轮次{round_num+1}释放完成")
            
            # 第三阶段：在碎片化内存中尝试分配新的连续大块
            print(f"\n>> 第三阶段: 在碎片化内存中测试大块分配 (轮次 {round_num + 1})")
            
            test_sizes = [200, 350, 500, 750, 1000]  # 测试不同大小的连续分配
            successful_allocations = 0
            
            for test_size in test_sizes:
                try:
                    print(f"    尝试分配连续大块: {test_size}MB")
                    print_memory_stats(f"大块分配{test_size}MB前")
                    
                    test_tensor = torch.ones((test_size * 256, 1024), device='cuda', dtype=torch.float32)
                    all_tensors.append((f"round{round_num+1}_test_block_{test_size}MB", test_tensor))
                    successful_allocations += 1
                    
                    print(f"    ✓ 成功分配 {test_size}MB 连续块")
                    print_memory_stats(f"大块分配{test_size}MB后")
                    
                except RuntimeError as e:
                    print(f"    ✗ 分配 {test_size}MB 失败: {e}")
                    if "out of memory" in str(e).lower():
                        print(f"    可能由于碎片化导致无法分配 {test_size}MB 连续内存")
                    break
            
            print(f"  轮次 {round_num + 1} 大块分配测试完成，成功分配 {successful_allocations}/{len(test_sizes)} 个大块")
            
            # 轮次间的内存状态报告
            if len(all_tensors) > 0:
                total_allocated = sum(t.numel() * t.element_size() for _, t in all_tensors)
                print(f"\n轮次 {round_num + 1} 完成:")
                print(f"  当前张量总数: {len(all_tensors)}")
                print(f"  估计占用内存: {total_allocated / (1024**3):.2f} GB")
                print_memory_stats(f"轮次{round_num+1}结束")
            
            # 轮次间延迟
            if round_num < fragmentation_rounds - 1:
                print(f"\n等待 2 秒后开始下一轮...")
                time.sleep(2)
        
        # 最终碎片化效果测试
        print(f"\n{'='*80}")
        print(f"最终碎片化效果测试")
        print(f"{'='*80}")
        
        print(f"\n>> 最终测试: 尝试分配各种大小的内存块")
        final_test_sizes = [50, 100, 200, 400, 800, 1200, 1600, 2000]
        final_success_count = 0
        
        for size in final_test_sizes:
            try:
                print(f"  最终测试分配: {size}MB")
                final_tensor = torch.ones((size * 256, 1024), device='cuda', dtype=torch.float32)
                all_tensors.append((f"final_test_{size}MB", final_tensor))
                final_success_count += 1
                print(f"  ✓ 成功")
                print_memory_stats(f"最终测试{size}MB后")
            except RuntimeError as e:
                print(f"  ✗ 失败: {e}")
                if "out of memory" in str(e).lower():
                    print(f"  碎片化导致无法分配 {size}MB 连续内存")
                break
        
        print(f"\n最终分配测试结果: {final_success_count}/{len(final_test_sizes)} 成功")
        
        # 碎片化分析报告
        print(f"\n{'='*80}")
        print(f"碎片化分析报告")
        print(f"{'='*80}")
        
        if len(all_tensors) > 0:
            total_tensors = len(all_tensors)
            total_memory = sum(t.numel() * t.element_size() for _, t in all_tensors)
            avg_size = total_memory / total_tensors if total_tensors > 0 else 0
            
            # 按大小分类统计
            small_blocks = sum(1 for _, t in all_tensors if t.numel() * t.element_size() < 50 * 1024 * 1024)
            medium_blocks = sum(1 for _, t in all_tensors if 50 * 1024 * 1024 <= t.numel() * t.element_size() < 200 * 1024 * 1024)
            large_blocks = sum(1 for _, t in all_tensors if t.numel() * t.element_size() >= 200 * 1024 * 1024)
            
            print(f"总张量数量: {total_tensors}")
            print(f"总内存使用: {total_memory / (1024**3):.2f} GB")
            print(f"平均块大小: {avg_size / (1024**2):.2f} MB")
            print(f"内存块分布:")
            print(f"  小块 (<50MB): {small_blocks} 个 ({small_blocks/total_tensors*100:.1f}%)")
            print(f"  中块 (50-200MB): {medium_blocks} 个 ({medium_blocks/total_tensors*100:.1f}%)")
            print(f"  大块 (>200MB): {large_blocks} 个 ({large_blocks/total_tensors*100:.1f}%)")
            
            print(f"\n碎片化程度评估:")
            if small_blocks / total_tensors > 0.6:
                print("  🔴 严重碎片化 - 小块占比超过60%")
            elif small_blocks / total_tensors > 0.4:
                print("  🟡 中等碎片化 - 小块占比在40-60%")
            else:
                print("  🟢 轻度碎片化 - 小块占比低于40%")
                
    except Exception as e:
        print(f"\n碎片化测试过程中发生错误: {e}")
        traceback.print_exc()
        
    finally:
        # 清理所有剩余内存
        print(f"\n{'='*80}")
        print(f"清理阶段: 释放所有剩余内存")
        print(f"{'='*80}")
        
        cleanup_count = 0
        for name, tensor in all_tensors:
            print(f"清理 {name}")
            del tensor
            cleanup_count += 1
            
            # 每清理10个张量打印一次状态
            if cleanup_count % 10 == 0:
                torch.cuda.empty_cache()
                print_memory_stats(f"清理{cleanup_count}个张量后")
        
        all_tensors.clear()
        
        # 最终清理
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_stats("最终清理完成后")
        
        print(f"碎片化测试完成，共清理了 {cleanup_count} 个张量")
        
def test_fragmentation_pattern1():
    """
    专门测试显存碎片化模式
    通过多轮分配和释放不规则大小的内存块来制造更严重的碎片化
    增强版：创造更多大块空隙以便可视化
    """
    print(f"\n===== 开始显存碎片化模式测试 (增强版) =====")
    
    all_tensors = []
    fragmentation_rounds = 200  # 减少轮数但增加每轮的复杂度
    
    try:
        # 获取GPU总内存信息
        if torch.cuda.is_available():
            gpu_properties = torch.cuda.get_device_properties(0)
            total_memory = gpu_properties.total_memory
            print(f"GPU总内存: {total_memory / (1024**3):.2f} GB")
        
        for round_num in range(fragmentation_rounds):
            print(f"\n{'='*70}")
            print(f"第 {round_num + 1}/{fragmentation_rounds} 轮碎片化测试")
            print(f"{'='*70}")
            
            # 第一阶段：分配各种大小的内存块，特别增加大块
            print(f"\n>> 第一阶段: 分配不规则大小的内存块 (轮次 {round_num + 1})")
            
            # 每轮使用不同的大小模式，特别加强大块分配
            if round_num == 0:
                # 第一轮：大量大块 + 少量小块
                sizes_mb = [512, 1024, 768, 256, 512, 384, 896, 1280, 640, 5, 12, 28]
            elif round_num == 1:
                # 第二轮：超大块 + 中块
                sizes_mb = [1536, 2048, 1024, 512, 768, 1280, 15, 31, 63, 127]
            elif round_num == 2:
                # 第三轮：大中小混合，但以大块为主
                sizes_mb = [800, 1200, 400, 600, 1000, 200, 8, 16, 32, 64]
            elif round_num == 3:
                # 第四轮：极大块
                sizes_mb = [2048, 1536, 2560, 1792, 1024, 3, 7, 15]
            elif round_num == 4:
                # 第五轮：创建"条纹"模式 - 大块间隔小块
                sizes_mb = [1024, 5, 1024, 10, 1024, 15, 1024, 20, 1024]
            else:
                # 其他轮次：随机但偏向大块
                large_blocks = [np.random.randint(500, 2000) for _ in range(6)]  # 大块
                medium_blocks = [np.random.randint(100, 500) for _ in range(4)]   # 中块
                small_blocks = [np.random.randint(1, 50) for _ in range(3)]       # 小块
                sizes_mb = large_blocks + medium_blocks + small_blocks
                np.random.shuffle(sizes_mb)  # 随机打乱顺序
            
            print(f"本轮将分配的内存块大小: {sizes_mb} MB")
            
            round_tensors = []  # 当前轮次的张量
            
            for i, size_mb in enumerate(sizes_mb):
                try:
                    print(f"  [轮次{round_num+1}] 分配内存块 {i+1}/{len(sizes_mb)}: {size_mb}MB")
                    print_stack()
                    print_memory_stats(f"轮次{round_num+1}分配块{i+1}前")
                    
                    tensor = torch.ones((size_mb * 256, 1024), device='cuda', dtype=torch.float32)
                    tensor_info = (f"round{round_num+1}_block_{i+1}_{size_mb}MB", tensor)
                    all_tensors.append(tensor_info)
                    round_tensors.append(tensor_info)
                    
                    print_memory_stats(f"轮次{round_num+1}分配块{i+1}后")
                    print(f"  ✓ 成功分配，当前总张量数: {len(all_tensors)}")
                    
                    # 添加延迟，便于观察
                    time.sleep(0.2)
                    
                except RuntimeError as e:
                    print(f"  ✗ 分配失败: {e}")
                    if "out of memory" in str(e).lower():
                        print(f"  显存不足，停止当前轮次的分配")
                        break
                    
            print(f"\n轮次 {round_num + 1} 分配阶段完成，成功分配 {len(round_tensors)} 个内存块")
            print_memory_stats(f"轮次{round_num+1}分配完成")
            
            # 第二阶段：战略性释放大块以创造大空隙
            print(f"\n>> 第二阶段: 战略性释放大块制造大空隙 (轮次 {round_num + 1})")
            
            if len(round_tensors) > 0:
                # 不同的大块释放策略
                if round_num % 4 == 0:
                    # 策略1：优先释放最大的块
                    tensor_sizes = [(i, int(name.split('_')[-1].replace('MB', ''))) 
                                   for i, (name, _) in enumerate(round_tensors)]
                    tensor_sizes.sort(key=lambda x: x[1], reverse=True)  # 按大小降序
                    indices_to_free = [x[0] for x in tensor_sizes[:len(tensor_sizes)//2]]  # 释放最大的一半
                    strategy_name = "释放最大的块"
                    
                elif round_num % 4 == 1:
                    # 策略2：释放间隔的大块（创造均匀空隙）
                    large_block_indices = []
                    for i, (name, tensor) in enumerate(round_tensors):
                        size_mb = int(name.split('_')[-1].replace('MB', ''))
                        if size_mb >= 200:  # 认为是大块
                            large_block_indices.append(i)
                    # 释放每隔一个大块
                    indices_to_free = [large_block_indices[i] for i in range(0, len(large_block_indices), 2)]
                    strategy_name = "间隔释放大块"
                    
                elif round_num % 4 == 2:
                    # 策略3：释放中间大小的块，保留最大和最小的
                    tensor_sizes = [(i, int(name.split('_')[-1].replace('MB', ''))) 
                                   for i, (name, _) in enumerate(round_tensors)]
                    tensor_sizes.sort(key=lambda x: x[1])  # 按大小升序
                    # 释放中间部分的块
                    start_idx = len(tensor_sizes) // 4
                    end_idx = 3 * len(tensor_sizes) // 4
                    indices_to_free = [tensor_sizes[i][0] for i in range(start_idx, end_idx)]
                    strategy_name = "释放中等大小块"
                    
                else:
                    # 策略4：创建"棋盘"模式 - 按位置间隔释放
                    indices_to_free = [i for i in range(len(round_tensors)) if i % 3 == 1]
                    strategy_name = "棋盘模式释放"
                
                print(f"  使用策略: {strategy_name}")
                print(f"  将释放 {len(indices_to_free)}/{len(round_tensors)} 个内存块")
                
                # 计算将要释放的内存大小
                total_to_free_mb = 0
                for idx in indices_to_free:
                    if idx < len(round_tensors):
                        name = round_tensors[idx][0]
                        size_mb = int(name.split('_')[-1].replace('MB', ''))
                        total_to_free_mb += size_mb
                
                print(f"  将释放总计约 {total_to_free_mb} MB 内存")
                
                # 执行释放
                freed_count = 0
                freed_mb = 0
                for idx in sorted(indices_to_free, reverse=True):
                    if idx < len(round_tensors):
                        name, tensor = round_tensors[idx]
                        size_mb = int(name.split('_')[-1].replace('MB', ''))
                        print(f"    [轮次{round_num+1}] 释放 {name} ({size_mb}MB)")
                        
                        # 从总列表中移除
                        all_tensors = [(n, t) for n, t in all_tensors if n != name]
                        
                        del tensor
                        round_tensors.pop(idx)
                        freed_count += 1
                        freed_mb += size_mb
                        
                        # 每释放几个大块就清空一次缓存
                        if freed_count % 2 == 0:
                            torch.cuda.empty_cache()
                            print_memory_stats(f"轮次{round_num+1}释放{freed_count}个块后")
                        
                        time.sleep(0.15)  # 稍长延迟以便观察
                
                print(f"  轮次 {round_num + 1} 释放阶段完成，释放了 {freed_count} 个内存块，总计 {freed_mb} MB")
                print(f"  剩余张量数: {len(all_tensors)}")
                print_memory_stats(f"轮次{round_num+1}释放完成")
            
            # 第三阶段：在大空隙中测试各种大小的分配
            print(f"\n>> 第三阶段: 在大空隙中测试分配 (轮次 {round_num + 1})")
            
            # 测试更多样化的大小，包括可能填补空隙的大小
            test_sizes = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500]
            successful_allocations = 0
            
            for test_size in test_sizes:
                try:
                    print(f"    尝试在空隙中分配: {test_size}MB")
                    print_memory_stats(f"空隙分配{test_size}MB前")
                    
                    test_tensor = torch.ones((test_size * 256, 1024), device='cuda', dtype=torch.float32)
                    all_tensors.append((f"round{round_num+1}_gap_fill_{test_size}MB", test_tensor))
                    successful_allocations += 1
                    
                    print(f"    ✓ 成功在空隙中分配 {test_size}MB")
                    print_memory_stats(f"空隙分配{test_size}MB后")
                    
                    # 成功分配大块后稍作延迟
                    if test_size >= 1000:
                        time.sleep(0.3)
                    
                except RuntimeError as e:
                    print(f"    ✗ 空隙分配 {test_size}MB 失败: {e}")
                    if "out of memory" in str(e).lower():
                        print(f"    显存不足或空隙不够大，无法分配 {test_size}MB 连续内存")
                    break
            
            print(f"  轮次 {round_num + 1} 空隙分配测试完成，成功分配 {successful_allocations}/{len(test_sizes)} 个块")
            
            # 第四阶段：再次释放一些刚分配的块，创造新的空隙模式
            if successful_allocations > 0 and round_num % 2 == 0:
                print(f"\n>> 第四阶段: 再次释放创造新空隙模式 (轮次 {round_num + 1})")
                
                # 找到刚刚分配的gap_fill块
                gap_fill_blocks = [(i, (name, tensor)) for i, (name, tensor) in enumerate(all_tensors) 
                                  if f"round{round_num+1}_gap_fill_" in name]
                
                if len(gap_fill_blocks) > 1:
                    # 释放其中一些，创造新的空隙模式
                    num_to_free = len(gap_fill_blocks) // 2
                    blocks_to_free = gap_fill_blocks[:num_to_free]
                    
                    print(f"    将再次释放 {num_to_free} 个gap_fill块，创造新空隙")
                    
                    for _, (name, tensor) in blocks_to_free:
                        size_mb = int(name.split('_')[-1].replace('MB', ''))
                        print(f"    再次释放 {name} ({size_mb}MB)")
                        
                        all_tensors = [(n, t) for n, t in all_tensors if n != name]
                        del tensor
                    
                    torch.cuda.empty_cache()
                    print_memory_stats(f"轮次{round_num+1}再次释放后")
            
            # 轮次间的详细内存状态报告
            if len(all_tensors) > 0:
                total_allocated = sum(t.numel() * t.element_size() for _, t in all_tensors)
                
                # 按大小分类统计当前内存块
                large_blocks = sum(1 for _, t in all_tensors if t.numel() * t.element_size() >= 500 * 1024 * 1024)
                medium_blocks = sum(1 for _, t in all_tensors if 50 * 1024 * 1024 <= t.numel() * t.element_size() < 500 * 1024 * 1024)
                small_blocks = sum(1 for _, t in all_tensors if t.numel() * t.element_size() < 50 * 1024 * 1024)
                
                print(f"\n轮次 {round_num + 1} 完成:")
                print(f"  当前张量总数: {len(all_tensors)}")
                print(f"  估计占用内存: {total_allocated / (1024**3):.2f} GB")
                print(f"  大块(≥500MB): {large_blocks} 个")
                print(f"  中块(50-500MB): {medium_blocks} 个") 
                print(f"  小块(<50MB): {small_blocks} 个")
                print_memory_stats(f"轮次{round_num+1}结束")
                
                # 估算空隙
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    potential_gaps = reserved - allocated
                    print(f"  潜在空隙大小: {potential_gaps / (1024**3):.2f} GB")
            
            # 轮次间延迟
            if round_num < fragmentation_rounds - 1:
                print(f"\n等待 3 秒后开始下一轮...")
                time.sleep(3)
        
        # 最终状态和空隙测试
        print(f"\n{'='*80}")
        print(f"最终大空隙利用测试")
        print(f"{'='*80}")
        
        print(f"\n>> 最终测试: 尝试利用各种大小的空隙")
        final_test_sizes = [25, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000]
        final_success_count = 0
        
        for size in final_test_sizes:
            try:
                print(f"  最终空隙测试分配: {size}MB")
                final_tensor = torch.ones((size * 256, 1024), device='cuda', dtype=torch.float32)
                all_tensors.append((f"final_gap_test_{size}MB", final_tensor))
                final_success_count += 1
                print(f"  ✓ 成功利用空隙分配 {size}MB")
                print_memory_stats(f"最终空隙测试{size}MB后")
            except RuntimeError as e:
                print(f"  ✗ 失败: {e}")
                if "out of memory" in str(e).lower():
                    print(f"  空隙不足，无法分配 {size}MB 连续内存")
                break
        
        print(f"\n最终空隙利用测试结果: {final_success_count}/{len(final_test_sizes)} 成功")
        
        # 详细的碎片化和空隙分析报告
        print(f"\n{'='*80}")
        print(f"碎片化和空隙分析报告")
        print(f"{'='*80}")
        
        if len(all_tensors) > 0:
            total_tensors = len(all_tensors)
            total_memory = sum(t.numel() * t.element_size() for _, t in all_tensors)
            avg_size = total_memory / total_tensors if total_tensors > 0 else 0
            
            # 详细的大小分类统计
            huge_blocks = sum(1 for _, t in all_tensors if t.numel() * t.element_size() >= 1000 * 1024 * 1024)  # ≥1GB
            large_blocks = sum(1 for _, t in all_tensors if 500 * 1024 * 1024 <= t.numel() * t.element_size() < 1000 * 1024 * 1024)  # 500MB-1GB
            medium_blocks = sum(1 for _, t in all_tensors if 100 * 1024 * 1024 <= t.numel() * t.element_size() < 500 * 1024 * 1024)  # 100-500MB
            small_blocks = sum(1 for _, t in all_tensors if 10 * 1024 * 1024 <= t.numel() * t.element_size() < 100 * 1024 * 1024)   # 10-100MB
            tiny_blocks = sum(1 for _, t in all_tensors if t.numel() * t.element_size() < 10 * 1024 * 1024)  # <10MB
            
            print(f"总张量数量: {total_tensors}")
            print(f"总内存使用: {total_memory / (1024**3):.2f} GB")
            print(f"平均块大小: {avg_size / (1024**2):.2f} MB")
            print(f"详细内存块分布:")
            print(f"  超大块 (≥1GB): {huge_blocks} 个 ({huge_blocks/total_tensors*100:.1f}%)")
            print(f"  大块 (500MB-1GB): {large_blocks} 个 ({large_blocks/total_tensors*100:.1f}%)")
            print(f"  中块 (100-500MB): {medium_blocks} 个 ({medium_blocks/total_tensors*100:.1f}%)")
            print(f"  小块 (10-100MB): {small_blocks} 个 ({small_blocks/total_tensors*100:.1f}%)")
            print(f"  微块 (<10MB): {tiny_blocks} 个 ({tiny_blocks/total_tensors*100:.1f}%)")
            
            # GPU内存整体状态
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                free_cached = reserved - allocated
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
                free_total = total_gpu_memory - reserved
                
                print(f"\nGPU内存状态:")
                print(f"  已分配内存: {allocated / (1024**3):.2f} GB ({allocated/total_gpu_memory*100:.1f}%)")
                print(f"  已保留内存: {reserved / (1024**3):.2f} GB ({reserved/total_gpu_memory*100:.1f}%)")
                print(f"  缓存空隙: {free_cached / (1024**3):.2f} GB ({free_cached/total_gpu_memory*100:.1f}%)")
                print(f"  完全空闲: {free_total / (1024**3):.2f} GB ({free_total/total_gpu_memory*100:.1f}%)")
                
                print(f"\n空隙分析:")
                if free_cached > 1024**3:  # 大于1GB的缓存空隙
                    print("  🟢 存在大量缓存空隙，有利于可视化大空白块")
                elif free_cached > 512*1024**2:  # 大于512MB
                    print("  🟡 存在中等缓存空隙")
                else:
                    print("  🔴 缓存空隙较少")
            
            print(f"\n碎片化程度评估:")
            large_and_huge = huge_blocks + large_blocks
            if large_and_huge / total_tensors > 0.3:
                print("  🟢 大块比例高，存在良好的大空隙潜力")
            elif tiny_blocks / total_tensors > 0.6:
                print("  🔴 严重碎片化 - 微小块占比超过60%")
            elif small_blocks / total_tensors > 0.4:
                print("  🟡 中等碎片化 - 小块占比在40-60%")
            else:
                print("  🟢 轻度碎片化 - 大中块为主")
                
    except Exception as e:
        print(f"\n碎片化测试过程中发生错误: {e}")
        traceback.print_exc()
        
    finally:
        # 分阶段清理，便于观察空隙变化
        print(f"\n{'='*80}")
        print(f"分阶段清理阶段: 观察空隙释放过程")
        print(f"{'='*80}")
        
        if all_tensors:
            # 按大小分组清理
            tensor_by_size = {}
            for name, tensor in all_tensors:
                size_bytes = tensor.numel() * tensor.element_size()
                if size_bytes >= 1000 * 1024 * 1024:
                    category = "huge"
                elif size_bytes >= 500 * 1024 * 1024:
                    category = "large"
                elif size_bytes >= 100 * 1024 * 1024:
                    category = "medium"
                else:
                    category = "small"
                
                if category not in tensor_by_size:
                    tensor_by_size[category] = []
                tensor_by_size[category].append((name, tensor))
            
            # 分类别清理
            cleanup_order = ["small", "medium", "large", "huge"]
            total_cleanup_count = 0
            
            for category in cleanup_order:
                if category in tensor_by_size:
                    print(f"\n>> 清理 {category} 类别张量...")
                    category_count = 0
                    
                    for name, tensor in tensor_by_size[category]:
                        size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                        print(f"清理 {name} ({size_mb:.1f}MB)")
                        del tensor
                        category_count += 1
                        total_cleanup_count += 1
                        
                        # 每清理几个张量打印一次状态
                        if category_count % 5 == 0:
                            torch.cuda.empty_cache()
                            print_memory_stats(f"清理{category}类第{category_count}个后")
                    
                    torch.cuda.empty_cache()
                    print_memory_stats(f"清理{category}类完成")
                    time.sleep(1)  # 延迟以便观察
        
        all_tensors.clear()
        
        # 最终清理
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_stats("最终清理完成后")
        
        print(f"增强碎片化测试完成，共清理了 {total_cleanup_count} 个张量")
        print("现在应该有更多大空隙便于可视化")

def test_extreme_fragmentation():
    """
    极端碎片化测试 - 更长时间的反复分配和释放
    """
    print(f"\n===== 开始极端碎片化测试 =====")
    
    all_tensors = []
    test_duration_minutes = 3  # 测试持续3分钟
    start_time = time.time()
    
    allocation_counter = 0
    cycle_counter = 0
    
    try:
        print(f"测试将持续 {test_duration_minutes} 分钟")
        print(f"当前进程 PID: {os.getpid()}")
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 检查是否达到时间限制
            if elapsed_time > test_duration_minutes * 60:
                print(f"\n测试时间已达到 {test_duration_minutes} 分钟，结束测试")
                break
            
            cycle_counter += 1
            print(f"\n--- 开始第 {cycle_counter} 个分配-释放周期 ---")
            
            # 动态调整分配策略
            if cycle_counter % 4 == 1:
                # 周期1：分配很多小块
                sizes = [np.random.randint(1, 10) for _ in range(20)]
                strategy = "大量小块"
            elif cycle_counter % 4 == 2:
                # 周期2：分配中等块
                sizes = [np.random.randint(20, 100) for _ in range(10)]
                strategy = "中等块"
            elif cycle_counter % 4 == 3:
                # 周期3：分配少量大块
                sizes = [np.random.randint(100, 300) for _ in range(5)]
                strategy = "少量大块"
            else:
                # 周期4：混合大小
                sizes = [np.random.randint(1, 200) for _ in range(15)]
                strategy = "混合大小"
            
            print(f"策略: {strategy}, 将分配 {len(sizes)} 个内存块")
            
            # 分配阶段
            cycle_tensors = []
            successful_allocations = 0
            
            for i, size_mb in enumerate(sizes):
                allocation_counter += 1
                try:
                    tensor = torch.randn((size_mb * 256, 1024), device='cuda', dtype=torch.float32)
                    tensor_info = (f"cycle{cycle_counter}_alloc{allocation_counter}_{size_mb}MB", tensor)
                    all_tensors.append(tensor_info)
                    cycle_tensors.append(tensor_info)
                    successful_allocations += 1
                    
                    if allocation_counter % 50 == 0:
                        print(f"  已完成 {allocation_counter} 次分配")
                        print_memory_stats(f"第{allocation_counter}次分配后")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  显存不足，周期 {cycle_counter} 分配阶段结束")
                        break
                    else:
                        print(f"  分配错误: {e}")
            
            print(f"周期 {cycle_counter} 分配完成: {successful_allocations}/{len(sizes)} 成功")
            
            # 释放阶段 - 使用不同的释放模式
            if len(cycle_tensors) > 0:
                release_mode = cycle_counter % 5
                
                if release_mode == 0:
                    # 释放所有
                    indices_to_free = list(range(len(cycle_tensors)))
                    mode_name = "全部释放"
                elif release_mode == 1:
                    # 释放一半（随机）
                    num_to_free = len(cycle_tensors) // 2
                    indices_to_free = np.random.choice(len(cycle_tensors), num_to_free, replace=False).tolist()
                    mode_name = "随机一半释放"
                elif release_mode == 2:
                    # 释放奇数位置
                    indices_to_free = [i for i in range(len(cycle_tensors)) if i % 2 == 1]
                    mode_name = "奇数位置释放"
                elif release_mode == 3:
                    # 释放前三分之一
                    indices_to_free = list(range(len(cycle_tensors) // 3))
                    mode_name = "前三分之一释放"
                else:
                    # 不释放，累积碎片
                    indices_to_free = []
                    mode_name = "不释放（累积）"
                
                print(f"释放模式: {mode_name}, 将释放 {len(indices_to_free)} 个内存块")
                
                for idx in sorted(indices_to_free, reverse=True):
                    if idx < len(cycle_tensors):
                        name, tensor = cycle_tensors[idx]
                        all_tensors = [(n, t) for n, t in all_tensors if n != name]
                        del tensor
                        cycle_tensors.pop(idx)
                
                # 定期清理缓存
                if cycle_counter % 3 == 0:
                    torch.cuda.empty_cache()
            
            # 周期性状态报告
            if cycle_counter % 10 == 0:
                elapsed_minutes = elapsed_time / 60
                remaining_minutes = test_duration_minutes - elapsed_minutes
                total_leaked_memory = sum(t.numel() * t.element_size() for _, t in all_tensors) if all_tensors else 0
                
                print(f"\n{'*'*60}")
                print(f"[极端碎片化进度] 周期: {cycle_counter}")
                print(f"[极端碎片化进度] 已用时间: {elapsed_minutes:.1f} 分钟")
                print(f"[极端碎片化进度] 剩余时间: {remaining_minutes:.1f} 分钟")
                print(f"[极端碎片化进度] 累积张量: {len(all_tensors)}")
                print(f"[极端碎片化进度] 累积内存: {total_leaked_memory / (1024**3):.2f} GB")
                print_memory_stats("极端碎片化进度")
                print(f"{'*'*60}")
            
            # 短暂延迟
            time.sleep(0.05)
    
    except Exception as e:
        print(f"极端碎片化测试发生错误: {e}")
        traceback.print_exc()
    
    finally:
        print(f"\n极端碎片化测试清理...")
        for name, tensor in all_tensors:
            del tensor
        all_tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_stats("极端碎片化测试清理完成")
        
def main():
    """运行内存分配测试"""
    for cycle in range(200):
        try:
            mp.set_start_method('spawn', force=True)
            print("\n===== 创建小模型 =====")
            small_model = create_model_with_buffers(complexity=2)
            train_for_iterations(small_model, iterations=3)
            # test_large_model_transfers()
            test_custom_cuda_operations()
            # run_multiprocess_test(num_processes=3, size_mb=50, iterations=5, delay=0.5)
            # test_error_cuda_operations()
            # test_explicit_cuda_api_calls()
            
            # 释放小模型
            print("\n===== 释放小模型 =====")
            print_memory_stats("释放小模型前")
            del small_model
            gc.collect()
            torch.cuda.empty_cache()
            print_memory_stats("释放小模型后")
            
            # 分配大量临时内存然后释放
            print("\n===== 大量临时内存分配测试 =====")
            tensors = []
            for i in range(5):
                size = 50 + i * 20  # 递增的内存大小
                print(f"\n>> 分配临时内存块 {i+1}: {size}MB")
                tensors.append(force_cuda_allocation(size, f"大块{i+1}"))
                time.sleep(0.5)  # 短暂延迟
            
            # 随机顺序释放内存
            indices = list(range(len(tensors)))
            np.random.shuffle(indices)
            for idx in indices:
                print(f"\n>> 释放临时内存块 {idx+1}")
                force_cuda_free(tensors[idx], f"大块{idx+1}")
                tensors[idx] = None
                time.sleep(0.5)  # 短暂延迟
            
            print("\n===== 创建大模型 =====")
            large_model = create_model_with_buffers(complexity=4)
            
            # 运行推理测试
            run_inference_tests(large_model)
            
            # 运行训练
            train_for_iterations(large_model, iterations=4)
            
            print("\n===== 测试完成 =====")
            print_memory_stats("最终状态")
            
        except Exception as e:
            print(f"测试过程发生错误: {e}")
            traceback.print_exc()

def run_leak_test_only():
    """仅运行内存泄漏和OOM测试"""
    print(f"Python 进程 PID: {os.getpid()}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print("等待 5 秒以便附加跟踪器...")
    time.sleep(5)
    
    # 运行较短时间的测试，便于观察
    test_memory_leak_and_oom(duration_minutes=10, oom_interval_sec=20)
    
    print("内存泄漏测试完成，程序即将退出")

def run_fragmentation_pattern_only():   
    """仅运行显存碎片化模式测试"""
    print(f"Python 进程 PID: {os.getpid()}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print("等待 5 秒以便附加跟踪器...")
    time.sleep(5)
    test_extreme_fragmentation()
    # test_fragmentation_pattern1()

    print("显存碎片化模式测试完成，程序即将退出")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "leak":
        run_leak_test_only()
    elif len(sys.argv) > 1 and sys.argv[1] == "frag":
        run_fragmentation_pattern_only()
    else:
        main()