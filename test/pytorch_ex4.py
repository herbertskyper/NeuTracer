# 内存泄漏例子2
import torch
# from loguru import logger
import os
import psutil


def log_device_usage(count, use_cuda):
    mem_Mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    cuda_mem_Mb = torch.cuda.memory_allocated(0) / 1024 ** 2 if use_cuda else 0
    print(
        f"iter {count}, mem: {int(mem_Mb)}Mb, gpu mem:{int(cuda_mem_Mb)}Mb"
    )


def leak():
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    val = torch.rand(100,100,device="cuda") 
    count = 0
    log_iter = 20000
    log_device_usage(count, use_cuda)
    while True:
        value = torch.rand(100,100,device="cuda") 
        val += value.requires_grad_()
        count += 1
        if count % log_iter == 0:
            log_device_usage(count, use_cuda)


if __name__ == "__main__":
    leak()