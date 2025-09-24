import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False # 正常显示负号

def recover_signal(n, d, key_idx, key_vals, anom_idx, anom_vals):
    """
    n: 总长度
    d: 维度
    key_idx: 关键点索引 (1D array)
    key_vals: 关键点值 (2D array, shape=(len(key_idx), d))
    anom_idx: 异常点索引 (1D array)
    anom_vals: 异常点值 (2D array, shape=(len(anom_idx), d))
    """
    rec = np.zeros((n, d))
    for j in range(d):
        rec[:, j] = np.interp(
            np.arange(n),
            key_idx,
            key_vals[:, j]
        )
    rec[anom_idx] = rec[anom_idx] + (anom_vals - rec[anom_idx])
    return rec

if __name__ == "__main__":
    # 加载压缩数据
    npz = np.load("result/col1_compressed_points.npz")
    rec_ori = np.loadtxt("result/rec.txt", delimiter=",")
    key_idx = npz["key_idx"]
    key_vals = npz["key_vals"]
    anom_idx = npz["anom_idx"]
    anom_vals = npz["anom_vals"]

    n = int(np.max([key_idx.max(), anom_idx.max()]) + 1)
    d = key_vals.shape[1] if key_vals.ndim > 1 else 1
    if d == 1:
        key_vals = key_vals.reshape(-1, 1)
        anom_vals = anom_vals.reshape(-1, 1)

    # 恢复信号
    rec = recover_signal(n, d, key_idx, key_vals, anom_idx, anom_vals)

    # 保存恢复结果
    # np.savetxt("recovered_signal.csv", rec, delimiter=",")
    print("恢复信号已保存到 recovered_signal.csv")

    # 画图对比（反归一化）
    df = pd.read_csv("./dataset/col1.csv")
    orig = df.values[:n, :d]  # 保证长度和维度一致

    # 计算归一化用到的最小值和最大值
    data_min = orig.min(axis=0)
    data_max = orig.max(axis=0)

    # 反归一化
    rec_denorm = rec * (data_max - data_min) + data_min
    key_vals = key_vals * (data_max - data_min) + data_min
    anom_vals = anom_vals * (data_max - data_min) + data_min
    orig_denorm = rec_ori * (data_max - data_min) + data_min

    plt.figure(figsize=(12, 9))

    # 1. 原信号+重建信号
    plt.subplot(3, 1, 1)
    plt.plot(orig[:], label=f'原信号', alpha=0.5, linestyle=':')
    plt.plot(rec_denorm[:], label=f'重建信号 ')
    plt.legend()
    plt.title("原信号+重建信号")

    # 3. 原信号+白噪声原信号（如有）
    plt.subplot(3, 1, 3)
    plt.plot(orig[:], label=f'原信号', alpha=0.5, linestyle=':')
    plt.plot(orig_denorm[:], label=f'白噪声', linestyle='--',color = 'green')
    plt.legend()
    plt.title("原信号+白噪声")

    # 2. 原信号+高熵采样点
    plt.subplot(3, 1, 2)
    plt.plot(orig[:], label=f'原信号', alpha=0.5, linestyle=':')
    plt.scatter(anom_idx, anom_vals[:], color='red', label=f'高熵采样点', s=10)
    plt.legend()
    plt.title("原信号+高熵采样点")

    plt.tight_layout()
    plt.dpi = 300
    plt.savefig("result/recovered_3in1.png")
    plt.show()