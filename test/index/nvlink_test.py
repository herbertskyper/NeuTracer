#!/usr/bin/env python3
import torch
import time
import os

# 对应 C 枚举
NVLINK_FUNC_STRCPY = 0
NVLINK_FUNC_MEMCPY = 1
NVLINK_FUNC_MEMSET = 2

def sync():
    torch.cuda.synchronize()

def timed(fn):
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0

def test_memcpy():
    """MEMCPY: GPU-0 → GPU-1 整块拷贝"""
    size = 1 * 1024**3 // 4  
    src = torch.randn(size, device='cuda:0')
    def run():
        dst = src.to('cuda:1', non_blocking=False)
    return timed(run)

def test_strcpy():
    """STRCPY: 逐字节 copy_ 模拟"""
    size = 512 * 1024**2 // 4  # 512 MiB 防止太慢
    src = torch.randn(size, device='cuda:0')
    dst = torch.empty_like(src, device='cuda:1')
    def run():
        dst.copy_(src)  # 逐元素走 NVLink
    return timed(run)

def test_memset():
    """MEMSET: 跨卡 fill_"""
    size = 1 * 1024**3 // 4  # 1 GiB
    dst = torch.empty(size, device='cuda:1')
    def run():
        dst.fill_(42)
    return timed(run)

def run_op(op_id, name, fn):
    elapsed = fn()
    size = {
        NVLINK_FUNC_MEMCPY: 2 * 1024**3,
        NVLINK_FUNC_STRCPY: 512 * 1024**2,
        NVLINK_FUNC_MEMSET: 1 * 1024**3,
    }[op_id]
    bw = size / elapsed / 1e9
    print(f"[{name}] {size/1e9:.2f} GB in {elapsed*1000:.2f} ms → {bw:.1f} GB/s")
    return bw

def main():
    if torch.cuda.device_count() < 2:
        print("❌ Need ≥2 GPUs")
        return

    torch.cuda.set_device(0)

    results = [
        (NVLINK_FUNC_MEMCPY, "MEMCPY", test_memcpy),
        (NVLINK_FUNC_STRCPY, "STRCPY", test_strcpy),
        (NVLINK_FUNC_MEMSET, "MEMSET", test_memset),
    ]

    for op, name, fn in results:
        run_op(op, name, fn)


if __name__ == "__main__":
    import os
    print(f"Python PID: {os.getpid()}")
    import time
    time.sleep(10) 
    main()