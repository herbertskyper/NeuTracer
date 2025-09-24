# 碎片化测例1
import torch
import time
import random

def create_memory_holes(small_blocks, hole_count=3, min_hole_size=5, max_hole_size=15):
    """
    在内存块列表中创建几个较大的空洞
    
    参数:
        small_blocks: 内存块列表
        hole_count: 要创建的空洞数量
        min_hole_size: 每个空洞最小包含的块数
        max_hole_size: 每个空洞最大包含的块数
    """
    total_blocks = len(small_blocks)
    if total_blocks < hole_count * min_hole_size:
        raise ValueError("没有足够的块来创建所需数量的空洞")
    
    # 计算每个空洞的大小
    hole_sizes = []
    remaining_blocks = total_blocks
    for i in range(hole_count):

        size = random.randint(min_hole_size, 
                                min(max_hole_size, remaining_blocks - (hole_count - i - 1) * min_hole_size))
        hole_sizes.append(size)
        remaining_blocks -= size
    # 随机选择空洞的起始位置
    hole_starts = sorted(random.sample(range(total_blocks - sum(hole_sizes) + 1), hole_count))
    
    # 创建空洞
    freed_count = 0
    for start, size in zip(hole_starts, hole_sizes):
        end = start + size
        # 调整后续空洞的起始位置
        for i in range(len(hole_starts)):
            if hole_starts[i] > start:
                hole_starts[i] += size
        
        # 释放当前空洞的块
        for idx in range(start, end):
            small_blocks[idx] = None
            freed_count += 1
    
    print(f"创建了 {hole_count} 个空洞，释放了 {freed_count} 个块")
    return small_blocks

device = "cuda:0"
time.sleep(5)
# 1. 先分配大量小块来制造碎片
print("Allocating small blocks to create fragmentation...")
small_blocks = []
for _ in range(79000):  # 分配6万个小块
    # 每个块0.5MB左右
    tensor = torch.randn(128*1024, device=device)  # 128*1024*4 bytes = ~512KB
    small_blocks.append(tensor)
    time.sleep(0.001)  # 确保每次分配之间有间隔

print(f"Memory after small allocations:")
print(torch.cuda.memory_summary())

# 2. 随机选择一些块并确保真正释放
print("\nRandomly freeing blocks to create gaps...")
# 随机选择要释放的块的索引
indices_to_free = random.sample(range(len(small_blocks)//2), len(small_blocks)//8)
print(f"Selected {len(indices_to_free)} blocks to free")

# 先将这些块设为None，确保没有其他引用
for idx in indices_to_free:
    small_blocks[idx] = None

# # 强制垃圾回收和CUDA缓存清理
import gc
gc.collect()
torch.cuda.empty_cache()

# 从列表中移除None项
small_blocks = [x for x in small_blocks if x is not None]
time.sleep(10)

small_blocks = create_memory_holes(small_blocks, hole_count=3, min_hole_size=2500, max_hole_size=5000)
gc.collect()
torch.cuda.empty_cache()
# 从列表中移除None项
small_blocks = [x for x in small_blocks if x is not None]
# 强制垃圾回收和CUDA缓存清理

time.sleep(10)

print(f"Memory after freeing blocks:")
print(torch.cuda.memory_summary())

# 3. 尝试分配一个大块
print("\nTrying to allocate a large block...")
try:
    # 尝试分配一个2GB的大块
    large_block = torch.randn(10*1024*1024*1024//4, device=device)  # 5GB
    print("Successfully allocated large block!")
except RuntimeError as e:
    print(f"Failed to allocate large block: {e}")

print(f"\nFinal memory state:")
print(torch.cuda.memory_summary())

# 等待60秒观察
print("\nWaiting for 10 seconds...")
time.sleep(10)

# 清理
del small_blocks
if 'large_block' in locals():
    del large_block
torch.cuda.empty_cache()
