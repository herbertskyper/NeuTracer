# 白噪声采样算法

在采样之前，NeuTracer 先通过
LESINN (Local Exceptionality Similarity-based INdex of Nonconformity) 算法尽可能选择低熵的点。该算法核心思想是计算每个数据点与随机选择的子集中最相似点的相似度，重复多次采样累计相似度分数，相似度得分越低（最终输出的异常分数越高）的点越可能是异常点。算法通过局部相似性度量来评估数据点的异常程度，避免了全局统计假设，适用于复杂的多维数据分布。在项目中LESINN有 Python 和 CUDA 实现版本。

相似度计算采用距离倒数公式，通过欧几里得距离的倒数变换将距离转换为相似度分数。

```python
def similarity(x: np.ndarray, y: np.ndarray):
    a = 1 / (1 + np.sqrt(np.sum(np.square(x - y))))
    return a

def online_lesinn(
        incoming_data: np.array,
        historical_data: np.array,
        t: int = 50,
        phi: int = 20,
        random_state: int = None
):
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)
    
    m = incoming_data.shape[0]
    data_score = np.zeros((m,))
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
```
CUDA 版本利用 GPU 并行加速，每个线程处理一个数据点，使用 cuRAND 生成随机样本，并行计算每个数据点的异常分数。核心实现是每个线程为一个数据点随机选取 t 组样本（每组大小为 phi），计算与每组样本中最相似点的相似度并累加，最终异常分数等于 t 除以累计相似度。
```c
extern "C" __global__ void lesinn(const double *incoming_data, const double *all_data, int incoming_window, int all_data_window, int dimension, int t, int phi, double *similarity)
{
    // 线程ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState_t state;
    /* 初始化随机数状态 */
    curand_init(0, tid, 0, &state);
    const double *current_data = incoming_data + dimension * tid;
    double score = 0;
    for (size_t sample_id = 0; sample_id < t; sample_id++)
    {
        double max_sim = 0;
        for (size_t s = 0; s < phi; s++)
        {
            int sample = curand(&state) % all_data_window;
            double tmp = 0;
            // 计算相似度
            for (size_t data_id = 0; data_id < dimension; data_id++)
            {
                tmp += (current_data[data_id] - all_data[sample * dimension + data_id]) * (current_data[data_id] - all_data[sample * dimension + data_id]);
            }
            tmp = sqrt(tmp);
            tmp = 1 / (1 + tmp);
            max_sim = max(max_sim, tmp);
        }
        score += max_sim;
    }
    
    similarity[tid] = t / score;
}
```
之后 NeuTracer 在正常的点附近进行局部化随机采样。局部化随机采样特别适用于时间序列数据。该算法能够根据数据点的重要性得分进行有偏采样。算法核心思想是基于数据点的重要性得分构建概率分布，随机选择采样单元的中心点，围绕中心点进行高斯分布的局部采样，最后进行采样权重归一化处理。

局部化随机采样算法接收原始数据矩阵 x（形状为 n×k，其中 n 是数据点数量，k 是特征维度）、最终需要的采样单元数量 m、每个数据点的重要性得分 score、采样密度控制参数 scale、采样单元中心点被采样概率 rho、高斯采样的标准差 sigma 以及随机数种子 random_state。实现逻辑为，根据得分计算累积概率分布，随机选择 m 个不重复的时间戳作为采样单元中心，根据高斯概率密度函数在每个中心点周围进行局部采样，确保每个采样单元的权重和为1。
//#加一个伪代码

```python
# 核心采样逻辑
def localized_sample(
        x, m, score, scale=2, rho=None, sigma=1 / 12, random_state=None
):
    n = x.shape[0]
    t = np.zeros(n + 1)
    t[0] = 0
    for i in range(n):
        t[i + 1] = t[i] + score[i]
    t = t / t[n]
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)
    sample_mat = np.zeros((m, n))
    su_center = []
    su_timestamp = np.random.choice(range(n), m, replace=False)
    for i in range(m):
        c = su_timestamp[i]
        su_center.append(t[c])
        # 将采样单元所属区间的采样权重置为1, 防止该单元什么都不采样
        sample_mat[i][c] = 1
    step = each_step = 1 / (scale * n)
    if rho is None:
        rho = 1 / (np.sqrt(2 * np.pi) * sigma)
    y = 1
    while step > t[y]:
        y += 1
        while step <= 1:
        for j in range(m):
            c = su_center[j]
            if np.abs(c - step) > 3 * sigma:
                continue
            p = rho * np.exp(np.square((c - step) / sigma) / -2)
            if random.random() < p:
                sample_mat[j][y - 1] += 1
        step += each_step
        while step > t[y] and y < n:
            y += 1
    for row in range(m):
        # 权重归一化
        sample_mat[row] /= np.sum(sample_mat[row])
    sample_mat = np.asmatrix(sample_mat)
    return sample_mat, su_timestamp
```