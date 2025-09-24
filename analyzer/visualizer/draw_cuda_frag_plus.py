import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import re
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

def parse_log_file(filename):
    """解析日志文件，返回每个时间点的合并后内存块列表"""
    all_frames = []
    is_oom = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    frame = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') :
            continue
        if 'is_oom:' in line:
            m = re.search(r'is_oom\s*:\s*(\w+)', line)
            if m:
                is_oom.append(m.group(1).lower() == 'true')
                # print(f"OOM状态: {is_oom[-1]}")
            continue
        if '---' in line:
            all_frames.append(frame)
            frame = []
            continue
        # 匹配 0x地址, 大小 bytes
        m = re.search(r'0x([0-9a-fA-F]+),\s*(\d+)', line)
        if m:
            addr = int(m.group(1), 16)
            size = int(m.group(2))
            frame.append((addr, size))
    if frame:
        all_frames.append(frame)
    return all_frames, is_oom

def merge_blocks(blocks):
    """合并连续的内存块"""
    if not blocks:
        return []
    # 按地址排序
    blocks = sorted(blocks)
    merged = []
    cur_start, cur_size = blocks[0]
    for addr, size in blocks[1:]:
        if addr == cur_start + cur_size:
            cur_size += size
        else:
            merged.append((cur_start, cur_size))
            cur_start, cur_size = addr, size
    merged.append((cur_start, cur_size))
    return merged

def plot_timeline(frames,is_oom):
    """画出随时间变化的内存分布（横轴为时间，纵轴为地址空间）"""
    # 统计全局地址范围
    all_blocks = [block for frame in frames for block in frame]
    if not all_blocks:
        print("没有有效的内存分布数据")
        return
    min_addr = min(addr for addr, _ in all_blocks)
    max_addr = max(addr + size for addr, size in all_blocks)
    addr_range = max_addr - min_addr
    blocks_size = []
    for t, blocks in enumerate(frames):
        merged = merge_blocks(blocks)
        for addr, size in merged:
            blocks_size.append(size)
    min_size = min(size for size in blocks_size)
    max_size = max(size for size in blocks_size)

    cmap = plt.cm.Blues  # 定义颜色映射
    norm = plt.Normalize(vmin=min_size, vmax=max_size)  # 用于颜色条

    fig, ax = plt.subplots(figsize=(16, 8))
    for t, blocks in enumerate(tqdm(frames, desc="处理进度")):
        merged = merge_blocks(blocks)
        for addr, size in merged:
            # print(f"时间帧 {t}: 地址 {addr:#x}, 大小 {size} bytes")
            rel_bottom = (addr - min_addr) / addr_range
            rel_top = (addr + size - min_addr) / addr_range
            norm_size = (size - min_size) / (max_size - min_size + 1e-8)
            color = cmap(0.4 + 0.6 * norm_size)  # 0.4~1.0区间，避免太浅
            if is_oom[t]:
                color = plt.cm.Reds(0.4 + 0.6 * norm_size)
            else:
                color = plt.cm.Blues(0.4 + 0.6 * norm_size)
            ax.fill_betweenx([rel_bottom, rel_top], t, t+1, color=color, alpha=0.85)
            # print(f"时间帧 {t}: 地址 {rel_bottom} ~ {rel_top}")

            if is_oom[t]:
                ax.text(t + 0.5, 1.03, "OOM", color='red', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Memory')
    ax.set_xlabel('Time')
    # ax.set_title('CUDA内存分布随时间变化（合并连续块）')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(frames))
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Block Size（bytes）')
    
    plt.tight_layout()
    plt.savefig("cuda_mem_timeline.png")
if __name__ == "__main__":
    # filename = "../log/NeuTracer_log_gpu_memfragment_20250816_221651_.log"  # 替换为你的日志文件路径
    filename = "../analyzer/visualizer/temp/NeuTracer_log_gpu_memfragment_20250817_000635_.log"
    frames, is_oom = parse_log_file(filename)
    plot_timeline(frames,is_oom)