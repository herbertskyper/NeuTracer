import numpy as np
from algorithm.lesinn import online_lesinn
from algorithm.localized_sample import localized_sample
from scipy.interpolate import interp1d, CubicSpline

def interpolate_window(x, xp, fp, method='linear'):
    """
    x: 目标插值点（如 np.arange(win_r - win_l)）
    xp: 已知采样点索引
    fp: 已知采样点的值
    method: 'linear', 'nearest', 'cubic'
    """
    if method == 'linear':
        return np.interp(x, xp, fp)
    elif method == 'nearest':
        f = interp1d(xp, fp, kind='nearest', fill_value="extrapolate")
        return f(x)
    elif method == 'cubic':
        if len(xp) < 4:
            # Cubic 需要至少4个点，否则退化为线性
            return np.interp(x, xp, fp)
        cs = CubicSpline(xp, fp, extrapolate=True)
        return cs(x)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
def estimate_noise_cvx(
    data, normal_percent=0.05, t=50, phi=20, random_state=None,
    window_size=60, latest_windows=20, interp_method='linear'
):
    n, d = data.shape
    noise_estimate = np.zeros_like(data)
    all_sampled_indices = []

    win_l = 0
    while win_l < n:
        win_r = min(n, win_l + window_size)
        hb = max(0, win_l - latest_windows)
        latest = data[hb:win_l]
        window = data[win_l:win_r]
        scores = online_lesinn(window, latest, t=t, phi=phi, random_state=random_state)
        normal_count = int((win_r - win_l) * normal_percent)
        sample_score = 1.0 / (scores + 1e-8)
        sample_score = (sample_score - sample_score.min()) / (sample_score.max() - sample_score.min() + 1e-8)
        _, sampled_indices = localized_sample(
            window, normal_count, sample_score, random_state=random_state
        )
        sampled_indices = np.sort(sampled_indices)
        sampled_values = window[sampled_indices]        
        # 插值重建
        window_noise = np.zeros((win_r - win_l, d))
        for j in range(d):
            window_noise[:, j] = interpolate_window(
                np.arange(win_r - win_l),
                sampled_indices,
                sampled_values[:, j],
                method=interp_method
            )
        noise_estimate[win_l:win_r] = window_noise
        all_sampled_indices.extend(list(win_l + sampled_indices))
        win_l += window_size

    all_sampled_indices = np.sort(np.unique(all_sampled_indices))
    return noise_estimate, all_sampled_indices

def compress_data_cvx_denoising(data, anomaly_percent=0.2, normal_percent=0.05, t=50, phi=20, random_state=None):
    n, d = data.shape
    # 1. 估计白噪声
    noise_estimate, normal_indices = estimate_noise_cvx(data, normal_percent, t, phi, random_state)
    # 2. 数据去噪
    data_denoised = data - noise_estimate
    # 3. 检测异常点（在去噪数据上）
    window_size = 60
    stride = 30
    scores = np.zeros(len(data_denoised))
    counts = np.zeros(len(data_denoised))
    for start in range(0, len(data_denoised), stride):
        end = min(len(data_denoised), start + window_size)
        window_scores = online_lesinn(data_denoised[start:end], historical_data=None, t=t, phi=phi, random_state=random_state)
        scores[start:end] += window_scores
        counts[start:end] += 1
    scores /= np.maximum(counts, 1)
    # scores = online_lesinn(data_denoised,historical_data=None, t=t, phi=phi, random_state=random_state)
    sorted_indices = np.argsort(scores)
    anomaly_count = int(n * anomaly_percent)
    anomaly_indices = sorted_indices[-anomaly_count:]
    # 4. 只存储异常点的索引和值
    return normal_indices, anomaly_indices, data[anomaly_indices], noise_estimate