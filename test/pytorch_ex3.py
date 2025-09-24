# 内存泄漏测例1
import torch
import gc
import os
import time

# def create_and_hold_tensor1(device,num_iterations=1000):
#     tensors = []
#     for i in range(10):
#         for _ in range(100):
#             # 分配一个大的张量到GPU
#             tensor = torch.empty(1000, 1000, device=device)
#             tensors.append(tensor)
        
#         # 只释放一小部分张量
#         for _ in range(10):
#             tensor_to_free = tensors.pop()
#             del tensor_to_free
#         print(f"Iteration {i+1} completed, {len(tensors)} tensors held in memory.")
#         # gc.collect()
#         torch.cuda.empty_cache()
    

def create_and_hold_tensor(device,num_iterations=1000):
    # 创建一个大的张量并存储在列表中
    tensors = []
    for i in range(num_iterations):
        for _ in range(100):
            # 分配一个大的张量到GPU
            tensor = torch.randn(100, 100, device=device)
            tensors.append(tensor)
        
        # 只释放一小部分张量
        for _ in range(10):
            tensor_to_free = tensors.pop()
            del tensor_to_free
        print(f"Iteration {i+1} completed, {len(tensors)} tensors held in memory.")
        gc.collect()
        torch.cuda.empty_cache()

# 在GPU上运行
pid = os.getpid()
print(f"Process ID: {pid}")
time.sleep(5)
print(f"Current process ID: {pid}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
create_and_hold_tensor(device,500)
# create_and_hold_tensor1(device,1000)
