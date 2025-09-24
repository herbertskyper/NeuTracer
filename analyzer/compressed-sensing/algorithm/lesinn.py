import numpy as np
import cupy as cp
import random
from time import perf_counter

USE_NUMPY = True
CUDA_ENABLE = False

# lesinn_cuda_kernel_raw = ""
# with open("/home/light7an/neu-trace/analyzer/compressed-sensing/algorithm/cuda/lesinn.cu") as f:
#     lesinn_cuda_kernel_raw = f.read()

# lesinn_cuda_kernel = cp.RawKernel(lesinn_cuda_kernel_raw, 'lesinn')


def similarity(x: np.ndarray, y: np.ndarray):
    """
    计算两个向量的相似度
    :param x:
    :param y:
    :return:
    """
    a = 1 / (1 + np.sqrt(np.sum(np.square(x - y))))
    return a

def batch_similarity(x: np.ndarray, y: np.ndarray):
    """
    批量计算 x 与 y 中每个向量的相似度
    x: shape=(d,) 或 (m, d)
    y: shape=(n, d)
    返回: shape=(n,) 或 (m, n)
    """
    # x: (d,) -> (1, d)
    x = np.atleast_2d(x)
    # 广播计算欧氏距离
    diff = x[:, None, :] - y[None, :, :]  # (m, n, d)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (m, n)
    sim = 1 / (1 + dist)
    return sim


def online_lesinn(
        incoming_data: np.array,
        historical_data: np.array,
        t: int = 50,
        phi: int = 20,
        random_state: int = None
):
    """
    在线离群值算法 lesinn
    """
    global CUDA_ENABLE
    global USE_NUMPY

    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)
    
    m = incoming_data.shape[0]
    
    # 将历史所有数据和需要计算离群值的数据拼接到一起
    if historical_data.shape[0] > 0:
        all_data = np.concatenate([historical_data, incoming_data], axis=0)
    else:
        all_data = incoming_data
    
    n, d = all_data.shape
    
    if CUDA_ENABLE:
        try:
            # 转换为 CuPy 数组
            incoming_data_gpu = cp.array(incoming_data, order='C', dtype=cp.float64)
            all_data_gpu = cp.array(all_data, order='C', dtype=cp.float64)
            data_score_gpu = cp.zeros((m,), dtype=cp.float64) 
            
            # 调用 CUDA 内核
            lesinn_cuda_kernel(
                (1,), (m,), 
                (incoming_data_gpu, all_data_gpu, m, n, d, t, phi, data_score_gpu)
            )
            cp.cuda.runtime.deviceSynchronize()
            
            # 转换回 NumPy 数组
            return cp.asnumpy(data_score_gpu)
            
        except Exception as e:
            print(f"CUDA kernel failed: {e}")
            print("Falling back to CPU version...")
            CUDA_ENABLE = False
    
    # CPU 版本
    if USE_NUMPY:
        # data_score = np.zeros((m,))
        # for i in range(m):
        #     score = 0
        #     for j in range(t):
        #         # sample = random.sample(range(0, n), k=phi)
        #         sample = np.random.choice(n, size=phi, replace=False)
        #         sims = batch_similarity(incoming_data[i], all_data[sample])
        #         nn_sim = np.max(sims)
        #         score += nn_sim
        #     if score:
        #         data_score[i] = t / score
        
        # 1. 批量计算所有 similarity
        similarity_matrix = np.zeros((m, n))
        for i in range(m):
            similarity_matrix[i, :] = batch_similarity(incoming_data[i], all_data)

        # 2. 采样索引 shape=(m, t, phi)
        samples = np.array([
            [np.random.choice(n, size=phi, replace=False) for _ in range(t)]
            for _ in range(m)
        ])  # shape=(m, t, phi)

        # 3. 计算每个点的score
        data_score = np.zeros((m,))
        for i in range(m):
            score = 0
            for j in range(t):
                nn_sim = np.max(similarity_matrix[i, samples[i, j]])
                score += nn_sim
            if score:
                data_score[i] = t / score
        return data_score
    else:
        for i in range(m):
            score = 0
            for j in range(t):
                sample = random.sample(range(0, n), k=phi)
                nn_sim = 0
                for each in sample:
                    nn_sim = max(
                        nn_sim, similarity(incoming_data[i, :], all_data[each, :])
                    )
                score += nn_sim
            if score:
                data_score[i] = t / score

    return data_score


def lesinn(data, t=50, phi=20, random_state=None):
    """
    :param data: 数据矩阵, 行主序 shape=(n, d) 数据量n 数据维度d
    :param t: 每个数据点取t个data中子集作为离群值参考
    :param phi: 每个数据t个子集的大小
    :param random_state: 指定随机种子, 如果为None则是不指定
    :return: 每个data元素的离群值数组
    """
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)
    n = data.shape[0]
    data_score = np.zeros((n, ))
    sample = set()
    for i in range(n):
        score = 0
        for j in range(t):
            sample.clear()
            while len(sample) < phi and len(sample) < n:
                sample.add(random.randint(0, n - 1))
            nn_sim = 0
            for each in sample:
                nn_sim = max(nn_sim, similarity(data[i, :], data[each, :]))
            score += nn_sim
        if score:
            data_score[i] = (t / score)
    return data_score


def toy():
    """
    用二维高斯分布点测试lesinn算法:
    :return:
    """
    from matplotlib import pyplot as plt
    import time
    n = 1000
    data = []
    for p in range(n):
        data.append(np.random.normal(0, 1, 2))
    s = lesinn(np.array(data))
    data_set = np.mat(data)
    t1 = time.time()
    print(time.time() - t1)
    ds = []
    for p in range(n):
        ds.append(np.array([data_set[p][0], data_set[p][1], s[p]]))
    ds.sort(key=lambda x: x[2], reverse=True)
    ano = []
    for p in range(int(n / 20)):
        ano.append(ds[p])
    ano = np.array(ano)
    plt.figure()
    plt.scatter(data_set[:, 0], data_set[:, 1])
    plt.scatter(ano[:, 0], ano[:, 1], c='r', marker='x')
    plt.show()


if __name__ == '__main__':
    toy()
