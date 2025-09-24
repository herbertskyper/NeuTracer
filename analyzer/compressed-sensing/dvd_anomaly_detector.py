import numpy as np
from scipy.fftpack import fft, fftfreq
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from algorithm.fourier_transform import fourier_transform
from numpy.linalg import LinAlgError
from detect import anomaly_score_example
from queue import Queue
from time import time

def smoothing_extreme_values(values, ab_portion):
    """
    通常情况下，时间序列中的异常点比例小于5%[1]。
    因此，简单地移除偏离均值最大的前5%数据，并用线性插值进行填充。

    参数:
        values (np.ndarray): 已经经过线性插值和标准化（零均值、单位方差）的一维时间序列
        ab_portion (float): 需要处理的异常比例（如0.05表示5%）

    返回:
        np.ndarray: 平滑处理后的时间序列
    """
    values = np.asarray(values, np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` 必须是一维数组')

    # 计算每个点偏离零均值的绝对值
    values_deviation = np.abs(values)

    # 异常比例
    abnormal_portion = ab_portion

    # 用线性插值替换异常点
    abnormal_max = np.max(values_deviation)
    abnormal_index = np.argwhere(values_deviation >= abnormal_max * (1 - abnormal_portion))
    abnormal = abnormal_index.reshape(len(abnormal_index))
    normal_index = np.argwhere(values_deviation < abnormal_max * (1 - abnormal_portion))
    normal = normal_index.reshape(len(normal_index))
    normal_values = values[normal]
    if len(normal) > 0:
        abnormal_values = np.interp(abnormal, normal, normal_values)
        values[abnormal] = abnormal_values

    return values

def DMD_control(seasonals, resids, data, r):
    X1 = seasonals[:, : -288]
    Br = resids[:, : -288]
    X2 = data[:, 288 :] - Br

    u, s, v = np.linalg.svd(X1, full_matrices = False)
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    eigval, eigvec = np.linalg.eig(A_tilde)
    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ eigvec
    A = Psi @ np.diag(eigval) @ np.linalg.pinv(Psi)

    return Psi, A

def compute_DMD(data, r):
    T = data.shape[1]
    # Build data matrices
    X1 = data[:, : -288]
    X2 = data[:, 288 :]
    # Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(X1, full_matrices = False)
    if np.any(s[:r] == 0):
        s[s == 0] = 1e-8  
    # Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    if np.any(np.isnan(A_tilde)) or np.any(np.isinf(A_tilde)):
        print("[ERROR] A_tilde contains NaN or inf, returning zeros")
        return np.zeros_like(X2), np.zeros_like(X2)
    # Perform eigenvalue decomposition on A_tilde
    eigval, eigvec = np.linalg.eig(A_tilde)
    # Compute the coefficient matrix
    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ eigvec

    A = Psi @ np.diag(eigval) @ np.linalg.pinv(Psi)

    return Psi, A


def window_reconstruct(
        data: np.array,
        groups: list,
        random_state=None,
):
    """
    :param data: 原始数据(窗口数据)
    :param groups: 分组
    :param random_state: 随机种子
    :return: 重建数据
    """
    # 数据量, 维度
    n, d = data.shape
    rec_time = 0

    rec = np.zeros(shape=(n, d))
    # 按组重构，用values(m, len(groups[i]))重构x_re(n, len(groups[i]))
    for i in range(len(groups)):
        t = time()

        baselines = []
        seasonals = []
        resids = []
        data_T = data[:, groups[i]].T

        # Fourier transform
        u_dat = data[:, groups[i][0]]
        fft_series = fft(u_dat)
        power = np.abs(fft_series)
        sample_freq = fftfreq(fft_series.size)
        
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]

        top_k_seasons = 1
        top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
        fft_periods = (1 / freqs[top_k_idxs]).astype(int)
        if fft_periods[0] < 12 or fft_periods[0] > (n // 10):
            for idx in range(data_T.shape[0]):
                temp = smoothing_extreme_values(data_T[idx], 0.2)
                baselines.append(temp)

            baselines = np.asarray(baselines, np.float32)
            r = 3 if data_T.shape[0] > 3 else data_T.shape[0]
            Psi, A = compute_DMD(baselines, r)
            x_re = (A.real @ baselines).T
        else:
            for idx in range(data_T.shape[0]):

                result = seasonal_decompose(data_T[idx], model='additive', period=fft_periods[0])
                seasonal = result.seasonal
                resid = np.concatenate((result.resid[fft_periods[0]:fft_periods[0]*2], result.resid[fft_periods[0]:-fft_periods[0]], result.resid[-fft_periods[0]*2:-fft_periods[0]]), axis=0)
                seasonals.append(seasonal)
                resids.append(smoothing_extreme_values(resid, 0.2))
                baselines.append(data_T[idx])

            seasonals = np.asarray(seasonals, np.float32)
            resids = np.asarray(resids, np.float32)
            baselines = np.asarray(baselines, np.float32)
            r = 3 if data_T.shape[0] > 3 else data_T.shape[0]

            try:
                Psi, A = DMD_control(seasonals, resids, baselines, r)
                x_re = (A.real @ baselines).T
            except LinAlgError:
                x_re = seasonals.T
            finally:
                pass

        rec_time += time() - t

        for j in range(len(groups[i])):
            rec[:, groups[i][j]] = x_re[:, j]
    return rec


class DVDAnomalyDetector:
    """
    基于动态模分解的异常检测器
    """

    def __init__(
            self,
            random_state=None,
    ):
        """
        :param random_state: 随机数种子
        """
        self._random_state = random_state
        # 距离计算方法

    def reconstruct(
            self, data: np.array,
            window: int,
            windows_per_cycle: int,
            stride: int = 1,
    ):
        """
        离线预测输入数据的以时间窗为单位的异常概率预测, 多线程
        :param data: 输入数据
        :param window: 时间窗口长度(点)
        :param windows_per_cycle: 周期长度: 以时间窗口为单位
        :param stride: 时间窗口步长
        """
        if windows_per_cycle < 1:
            raise Exception('a cycle contains 1 window at least')
        # 周期长度
        cycle = windows_per_cycle * window
        # 周期特征: 按周期分组
        groups = self._get_cycle_feature(data, cycle, window)
        
        cycles = []
        for i in range(len(groups)):
            clusters = []
            for k, v in groups[i].items():
                clusters.append(v)
            cycles.append(clusters)
        reconstructed = self._get_reconstructed_data(
            data, window, windows_per_cycle, cycles, stride)
        return reconstructed

    def predict(
            self,
            data: np.array,
            reconstructed: np.array,
            window: int,
            stride: int = 1,
    ):
        """
        离线处理: 利用参数进行评估, 得到每个点的异常得分
        :param data: 原始数据
        :param reconstructed: 重建好的数据
        :param window: 数据窗口长度
        :param stride: 窗口步长
        :return: 每个点的异常得分
        """
        if reconstructed.shape != data.shape:
            raise Exception('shape mismatches')
        n, d = data.shape
        # 异常得分
        anomaly_score = np.zeros((n,))
        # 表示当时某个位置上被已重建窗口的数量
        anomaly_score_weight = np.zeros((n,))
        # 窗口左端点索引
        wb = 0
        while True:
            we = min(n, wb + window)
            # 窗口右端点索引 窗口数据[wb, we)
            score = anomaly_score_example(data[wb:we], reconstructed[wb:we])
            for i in range(we - wb):
                w = i + wb
                weight = anomaly_score_weight[w]
                anomaly_score[w] = \
                    (anomaly_score[w] * weight + score) / (weight + 1)
            anomaly_score_weight[wb:we] += 1
            if we >= n:
                break
            wb += stride
        return anomaly_score

    def _get_reconstructed_data(
            self,
            data: np.array,
            window: int,
            windows_per_cycle: int,
            groups: list,
            stride: int,
    ):
        """
        离线预测输入数据的以时间窗为单位的异常概率预测
        :param data: 输入数据
        :param window: 时间窗口长度(点)
        :param windows_per_cycle: 周期长度: 以时间窗口为单位
        :param groups: 每个周期的分组
        :param stride: 时间窗口步长
        :return:
        """
        n, d = data.shape
        # 重建的数据
        reconstructed = np.zeros((n, d))
        # 表示当时某个位置上被已重建窗口的数量(时间维度)
        reconstructing_weight = np.zeros((n,))
        needed_weight = np.zeros((n,))
        # 周期长度
        cycle = window * windows_per_cycle

        # 窗口左端点索引
        win_l = 0
        while True:
            win_r = min(n, win_l + window)
            # 窗口右端点索引 窗口数据[win_l, win_r)
            # win_l // cycle: 在第几个周期里
            group_idx = win_l // cycle
            rec_window = window_reconstruct(
                data[win_l:win_r], groups[group_idx], self._random_state
            )
            for index in range(rec_window.shape[0]):
                w = index + win_l
                weight = reconstructing_weight[w]
                reconstructed[w, :] = (reconstructed[w, :] * weight + rec_window[index]) / (weight + 1)
            reconstructing_weight[win_l:win_r] += 1
            # 每个位置目前被命中的次数
            needed_weight[win_l:win_r] += 1
            if win_r >= n:
                break
            win_l += stride

        mismatch_weights = []
        for i in range(n):
            if reconstructing_weight[i] != needed_weight[i]:
                mismatch_weights.append('%d' % i)
        if len(mismatch_weights):
            from sys import stderr
            stderr.write('BUG empty weight: index: %s\n' %
                         ','.join(mismatch_weights))
        return reconstructed

    def _get_cycle_feature(
            self,
            data: np.array,
            cycle: int,
            window: int
    ):
        """
        将数据按周期进行划分后计算得到每个周期的分组
        :param data: 数据
        :param cycle: 周期长度
        :return: 分组结果
        """
        # 数据量, 维度
        n, d = data.shape
        # 每周期分组结果
        cycle_groups = []
        # 工作数量
        group_index = 0
        # 周期开始的index
        cb = 0
        # 执行完得到每个KPI为一组的默认分组其他分组为[]的cycle_groups和含有(group_index, data[cb:ce])的task_queue
        while cb < n:
            # 周期结束的index
            ce = min(n, cb + cycle)  # 一周期数据为data[cb, ce)
            # 初始化追加列表引用
            if group_index == -1:
                # 没有历史数据
                # 分组默认每个kpi一组
                init_group = []
                if not self._without_grouping:
                    for i in range(d):
                        init_group.append([i])
                cycle_groups.append(init_group)
            else:
                cycle_groups.append([])
            group_index += 1
            cb += cycle

        cb = 0
        for group_index in range(len(cycle_groups)):
            # 周期结束的index
            ce = min(n, cb + cycle)
            # 计算每个周期的分组
            group = fourier_transform(data[cb:ce], window)
            cycle_groups[group_index] = group
            cb += cycle
        
        return cycle_groups
