import numpy as np
import matplotlib.pyplot as plt

# 1. 读取原始数据
data = np.loadtxt('col2.csv', delimiter=',')

# 2. 读取重建数据
rec = np.loadtxt('recpy.txt', delimiter=',')

# 3. 归一化原始数据（每列分别归一化到[0,1]）
data_min = data.min(axis=0)
data_max = data.max(axis=0)
data_norm = (data - data_min) / (data_max - data_min + 1e-8)
data_norm = data_norm[:600]
rec = rec[:600]

# 5. 画图对比
plt.figure(figsize=(12, 5))
for i in range(data.shape[1]):
    plt.subplot(data.shape[1], 1, i+1)
    plt.plot(data_norm[:, i], label='Original (norm)')
    plt.plot(rec[:, i], label='Reconstructed (norm)')
    plt.legend()
    plt.title(f'Column {i+1}')
plt.tight_layout()
plt.savefig('comparison_plot.png')
plt.show()