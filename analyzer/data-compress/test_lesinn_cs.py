import matplotlib.pyplot as plt
import numpy as np
from time import time
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 黑体
plt.rcParams['axes.unicode_minus'] = False
# 创建测试数据
def create_test_data(n=600, d=2):
    np.random.seed(42)
    data = np.zeros((n, d))
    for i in range(d):
        # 创建一些周期性的数据
        t = np.linspace(0, 6*np.pi, n)
        data[:, i] = np.sin(t + i*np.pi/4) + 0.1*np.random.randn(n)
        # 加入一些异常点
        anomaly_positions = np.random.choice(n, size=int(0.05*n), replace=False)
        data[anomaly_positions, i] += 0.5 * np.random.randn(len(anomaly_positions))
    return data

def create_test_data_from_csv(filepath, columns=None, n=None):
    df = pd.read_csv(filepath)
    if columns is not None:
        df = df[columns]
    if n is not None:
        df = df.iloc[:n]
    return df.values

# 测试压缩-重建效果
def test_compression():
    # data = create_test_data(n=600, d=2)
    data = create_test_data_from_csv(
        './tmp/a1000.csv',
        columns=['feature_1'],
        n=600
    )
    
    # reconstructed, compression_ratio, indices, scores = compress_and_reconstruct(
    #     data, high_percent=0.2, low_percent=0.2
    # )
    from compress_lesinn import compress_and_reconstruct_cvx
    t1 = time()
    reconstructed, compression_ratio, indices, noise, anomoly_index, anomoly_value = compress_and_reconstruct_cvx(
        data, anomaly_percent=0.2, normal_percent=0.05,phi=10
    )
    t2 = time()
    
    # 计算重建误差
    mse = np.mean((data - reconstructed)**2)
    relative_mse = mse / np.mean(data ** 2)
    print(f"Compression ratio: {compression_ratio:.2f}")
    print(f"Relative Mean squared error: {relative_mse:.6f}")
    print(f"Elapsed time: {t2-t1:.3f} seconds")

    # Visualization
    import os
    os.makedirs('image', exist_ok=True)
    plt.figure(figsize=(14, 10))
    # 1. 原数据+重建
    plt.subplot(3, 1, 1)
    plt.plot(data[:, 0], label='原数据', alpha=0.7, linestyle='--')
    plt.plot(reconstructed[:, 0], label='重建数据')
    plt.legend()
    plt.title(f'原数据+重建 (压缩率={compression_ratio:.2f}, 相对均方根误差={relative_mse:.3f})')

    # 2. 原数据+环境噪声
    plt.subplot(3, 1, 2)
    plt.plot(data[:, 0], label='原数据', alpha=0.7, linestyle='--')
    plt.plot(noise[:, 0], label='环境噪声',color = 'green')
    plt.legend()
    plt.title('原数据+环境噪声')

    # 3. 原数据+采样点
    plt.subplot(3, 1, 3)
    plt.plot(data[:, 0], label='原数据', alpha=0.7, linestyle='--')
    plt.scatter(anomoly_index, anomoly_value, c='red', s=30, label='高熵值采样')
    plt.legend()
    plt.title('原数据+采样点')

    plt.tight_layout()
    plt.savefig('image/compression_reconstruction_3in1.png', dpi=200)
    # plt.show()
    

if __name__ == "__main__":
    test_compression()
