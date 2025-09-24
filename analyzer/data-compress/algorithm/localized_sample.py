import numpy as np
import random

def localized_sample(x, m, score, random_state=None):
    """
    分区均匀采样+低score优先
    """
    n = x.shape[0]
    if random_state is not None:
        np.random.seed(random_state)
    # 分区
    bins = np.linspace(0, n, m+1, dtype=int)
    selected = []
    for i in range(m):
        l, r = bins[i], bins[i+1]
        if l >= r:
            continue
        idx = np.argmax(score[l:r]) + l
        selected.append(idx)
    selected = np.array(selected)
    sample_mat = np.zeros((m, n))
    for i, idx in enumerate(selected):
        sample_mat[i, idx] = 1
    sample_mat = np.asmatrix(sample_mat)
    return sample_mat, selected

# def localized_sample(
#         x, m, score, scale=2, rho=None, sigma=1 / 12, random_state=None
# ):
#     """
#     根据采样得分score_func函数对x进行随机局部化采样
#     :param x:采样点的原数据矩阵, shape=(n,k), n是数据个数, k是kpi种数
#     :param m:采样数量
#     :param score: 每个采样点的采样得分, 得分越高, 越容易采样到该点的信息
#     :param rho: 采样单元中心点被采样概率, 若为None, 取1/(sqrt(2pi)*sigma)
#     :param scale: 将原采样点数扩充至原来的scale倍
#     :param sigma: 高斯随机采样标准差
#     :param random_state: 随机数种子
#     :return:采样矩阵 shape=(m,n), 每个采样单元的所在时间序列位置列表
#     """
#     n = x.shape[0]
#     t = np.zeros(n + 1)
#     t[0] = 0
#     for i in range(n):
#         t[i + 1] = t[i] + score[i]
#     if t[n] == 0:
#         t = np.linspace(0, 1, n + 1)  # 均匀分布，避免全为0
#     else:
#         t = t / t[n]
#     #     raise ValueError("All scores are zero or too small, cannot normalize for sampling.")
#     if random_state:
#         random.seed(random_state)
#         np.random.seed(random_state)
#     sample_mat = np.zeros((m, n))
#     su_center = []
#     su_timestamp = np.random.choice(range(n), m, replace=False)
#     for i in range(m):
#         c = su_timestamp[i]
#         su_center.append(t[c])
#         # 将采样单元所属区间的采样权重置为1, 防止该单元什么都不采样
#         sample_mat[i][c] = 1
#     step = each_step = 1 / (scale * n)
#     if rho is None:
#         rho = 1 / (np.sqrt(2 * np.pi) * sigma)
#     y = 1
#     while step > t[y]:
#         y += 1
#     while step <= 1:
#         for j in range(m):
#             c = su_center[j]
#             if np.abs(c - step) > 3 * sigma:
#                 continue
#             p = rho * np.exp(np.square((c - step) / sigma) / -2)
#             if random.random() < p:
#                 sample_mat[j][y - 1] += 1
#         step += each_step
#         while step > t[y] and y < n:
#             y += 1
#     for row in range(m):
#         # 权重归一化
#         sample_mat[row] /= np.sum(sample_mat[row])
#     sample_mat = np.asmatrix(sample_mat)
#     return sample_mat, su_timestamp
