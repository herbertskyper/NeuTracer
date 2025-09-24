import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

def parse_cuda_memory_file(filename):
    """解析CUDA内存分配文件"""
    addresses = []
    sizes = []
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # 跳过注释行和空行
            if line.startswith('#') or not line:
                continue
            
            # 解析地址和大小
            parts = line.split(', ')
            if len(parts) >= 2:
                addr_str = parts[0].strip()
                size_str = parts[1].strip()
                
                # 转换地址（去掉0x前缀）
                address = int(addr_str, 16)
                size = int(size_str)
                
                addresses.append(address)
                sizes.append(size)
    
    return addresses, sizes

def create_memory_visualization(addresses, sizes, filename):
    """创建内存分布可视化图"""
    
    # 转换为MB单位便于显示
    sizes_mb = [size / (1024 * 1024) for size in sizes]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # 图1: 内存块热力图分布
    min_addr = min(addresses)
    max_addr = max(addresses) + max(sizes)
    addr_range = max_addr - min_addr
    
    # 根据内存块大小创建颜色映射
    min_size_mb = min(sizes_mb)
    max_size_mb = max(sizes_mb)
    
    # 使用对数缩放以便更好地显示大小差异
    log_sizes = np.log10(np.array(sizes_mb) + 1)  # +1避免log(0)
    min_log_size = min(log_sizes)
    max_log_size = max(log_sizes)
    
    # 归一化到0-1范围用于颜色映射
    normalized_sizes = (log_sizes - min_log_size) / (max_log_size - min_log_size)
    
    # 使用热力图颜色映射（从蓝色到红色，蓝色表示小块，红色表示大块）
    colormap = plt.cm.plasma  # 也可以使用 'hot', 'viridis', 'inferno' 等
    colors = colormap(normalized_sizes)
    
    y_pos = 0
    for i, (addr, size, size_mb) in enumerate(zip(addresses, sizes, sizes_mb)):
        # 计算相对位置
        rel_start = (addr - min_addr) / addr_range
        rel_width = size / addr_range
        
        # 绘制内存块，颜色根据大小决定
        rect = Rectangle((rel_start, y_pos), rel_width, 0.8, 
                        facecolor=colors[i], alpha=0.8, edgecolor='white', linewidth=0.3)
        ax1.add_patch(rect)
        
        # # 添加大小标签（只对大块内存显示）
        # if size_mb > 500:  # 只显示大于500MB的块
        #     # 根据背景颜色选择文字颜色
        #     text_color = 'white' if normalized_sizes[i] > 0.5 else 'black'
        #     ax1.text(rel_start + rel_width/2, y_pos + 0.4, f'{size_mb:.0f}MB', 
        #             ha='center', va='center', fontsize=8, color=text_color, weight='bold')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.2, 0.8)
    ax1.set_yticks([])  # 隐藏y轴刻度值
    ax1.set_xlabel('内存地址空间 (相对位置)')
    ax1.set_title('CUDA内存分布热力图 - 颜色表示块大小')
    ax1.grid(True, alpha=0.2)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_size_mb, vmax=max_size_mb))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.1, shrink=0.6)
    cbar.set_label('内存块大小 (MB)', fontsize=10)
    
    # 设置颜色条的刻度为实际大小值
    tick_positions = np.linspace(min_size_mb, max_size_mb, 6)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f'{pos:.0f}' for pos in tick_positions])
    
    # 图2: 内存大小分布直方图（增强版）
    # 使用相同的颜色映射
    n, bins, patches = ax2.hist(sizes_mb, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 为直方图的每个柱子根据大小范围着色
    for i, (patch, left_edge, right_edge) in enumerate(zip(patches, bins[:-1], bins[1:])):
        # 计算该柱子代表的大小范围的中点
        mid_size = (left_edge + right_edge) / 2
        log_mid_size = np.log10(mid_size + 1)
        normalized_mid = (log_mid_size - min_log_size) / (max_log_size - min_log_size)
        patch.set_facecolor(colormap(normalized_mid))
    
    ax2.set_xlabel('内存块大小 (MB)')
    ax2.set_ylabel('数量')
    ax2.set_title('内存块大小分布直方图')
    ax2.grid(True, alpha=0.3)
    
    # 添加大小分类的统计信息
    large_blocks = [s for s in sizes_mb if s > 500]
    medium_blocks = [s for s in sizes_mb if 50 <= s <= 500]
    small_blocks = [s for s in sizes_mb if s < 50]
    
    # 在图上添加统计文本
    stats_text = f'大块(>500MB): {len(large_blocks)}个\n中块(50-500MB): {len(medium_blocks)}个\n小块(<50MB): {len(small_blocks)}个'
    ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # 添加总体统计信息
    total_memory_gb = sum(sizes) / (1024**3)
    avg_size_mb = np.mean(sizes_mb)
    
    # fig.suptitle(f'CUDA内存热力图分析 - 总内存: {total_memory_gb:.2f}GB, 平均块大小: {avg_size_mb:.1f}MB, 总块数: {len(addresses)}', 
    #              fontsize=12, y=0.98)
    
    # 保存图片
    output_filename = filename.replace('.txt', '_heatmap_visualization.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"热力图可视化已保存为: {output_filename}")
    
    plt.show()
    
    return fig

def print_memory_stats(addresses, sizes):
    """打印内存统计信息"""
    sizes_mb = [size / (1024 * 1024) for size in sizes]
    
    print("=== CUDA内存分配统计 ===")
    print(f"总分配块数: {len(addresses)}")
    print(f"总内存大小: {sum(sizes) / (1024**3):.2f} GB")
    print(f"平均块大小: {np.mean(sizes_mb):.1f} MB")
    print(f"最大块大小: {max(sizes_mb):.1f} MB")
    print(f"最小块大小: {min(sizes_mb):.1f} MB")
    print(f"地址范围: 0x{min(addresses):x} - 0x{max(addresses) + max(sizes):x}")
    
    # 大小分布统计
    large_blocks = [s for s in sizes_mb if s > 500]  # 大于500MB
    medium_blocks = [s for s in sizes_mb if 50 <= s <= 500]  # 50-500MB
    small_blocks = [s for s in sizes_mb if s < 50]  # 小于50MB
    
    print(f"\n大块内存(>500MB): {len(large_blocks)} 个")
    print(f"中等内存(50-500MB): {len(medium_blocks)} 个")
    print(f"小块内存(<50MB): {len(small_blocks)} 个")

# 主程序
if __name__ == "__main__":
    filename = "analyzer/visualizer/cuda_frag/cuda_memory_allocations_20250628_210108.txt"
    
    try:
        # 解析文件
        addresses, sizes = parse_cuda_memory_file(filename)
        
        if not addresses:
            print("文件中没有找到有效的内存分配数据")
            exit(1)
        
        # 打印统计信息
        print_memory_stats(addresses, sizes)
        
        # 创建可视化图
        create_memory_visualization(addresses, sizes, filename)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")