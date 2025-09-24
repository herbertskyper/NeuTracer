import matplotlib.pyplot as plt
import numpy as np
import re
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 黑体
plt.rcParams['axes.unicode_minus'] = False

focus_samples = ['service2.csv', 'service7.csv', 'service8.csv', 'service11.csv']
total_metrics = 37  # 指标总数
max_metrics = 8     # 只画前8个指标

with open('data1_re.txt', encoding='utf-8') as f:
    lines = f.readlines()

sample_cycles = {}
current_sample = None
current_cycle = None

for line in lines:
    line = line.strip()
    m = re.match(r'Processing .*/(service\d+\.csv)', line)
    if m:
        current_sample = m.group(1)
        sample_cycles[current_sample] = []
        continue
    m = re.match(r'cycle (\d+)', line)
    if m:
        current_cycle = int(m.group(1))
        sample_cycles[current_sample].append([0]*total_metrics)
        continue
    m = re.match(r'(\d+): \[(.*)\]', line)
    if m and current_sample is not None and current_cycle is not None:
        period = int(m.group(1))
        metrics = m.group(2).split(',')
        for metric in metrics:
            metric = metric.strip()
            if metric:
                idx = int(metric)
                sample_cycles[current_sample][current_cycle][idx] = period

# 计算最大周期（用于色阶归一化）
max_period = 0
for sample in focus_samples:
    if sample in sample_cycles:
        arr = np.array(sample_cycles[sample])[:, :max_metrics]
        max_period = max(max_period, arr.max())

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes = axes.flatten()

for i, sample in enumerate(focus_samples):
    if sample not in sample_cycles:
        continue
    data = np.array(sample_cycles[sample])  # shape: (cycle数, 37)
    data = data[:, :max_metrics]            # 只画前8个指标
    ax = axes[i]
    im = ax.imshow(
        data.T, aspect='auto', cmap='Blues_r', interpolation='nearest',
        vmin=0, vmax=max_period
    )
    ax.set_xlabel('窗口编号 (Cycle Index)')
    ax.set_ylabel('指标编号 (Metric Index)')
    ax.set_title(sample)
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_yticks(np.arange(max_metrics))
    ax.set_yticklabels(np.arange(max_metrics))
    # 只在每行最后一个子图右侧加色条
    if i % 4 == 3:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='周期（Period）')

plt.suptitle('典型样本前8个指标各窗口周期热力图（周期越小颜色越深，无周期为浅色，窗口大小为7200）', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('focus_samples_2x2_heatmap.png', dpi=300)
plt.show()