import numpy as np
from noise_lesinn import compress_data_cvx_denoising
from time import time

def reconstruct_with_noise_and_anomaly(n, d, normal_indices, anomaly_indices, anomaly_values, data):
    # 1. 用插值重建白噪声
    noise_estimate = np.zeros((n, d))
    for j in range(d):
        noise_estimate[:, j] = np.interp(
            np.arange(n),
            np.sort(normal_indices),
            data[np.sort(normal_indices), j]
        )
    # 2. 初始化重建信号为白噪声
    reconstructed = noise_estimate.copy()
    # 3. 在异常点位置加上异常值
    reconstructed[anomaly_indices] = anomaly_values
    return reconstructed

def compress_and_reconstruct_cvx(data, anomaly_percent=0.2, normal_percent=0.2, t=50, phi=20, random_state=None):
    n, d = data.shape
    t1 = time()
    normal_indices, anomaly_indices, anomaly_values, noise = compress_data_cvx_denoising(
        data, anomaly_percent, normal_percent, t, phi, random_state
    )
    t2 = time()
    print(f"Compression time: {t2 - t1:.3f} seconds")
    reconstructed = reconstruct_with_noise_and_anomaly(
        n, d, normal_indices, anomaly_indices, anomaly_values, data
    )
    t3 = time()
    print(f"Reconstruction time: {t3 - t2:.3f} seconds")
    compression_ratio = (len(normal_indices) + len(anomaly_indices)) / n
    return reconstructed, compression_ratio, normal_indices, noise, anomaly_indices, anomaly_values