# 基于形状的时间序列聚类算法(cluster.py)

## 算法概述

基于形状的时间序列聚类算法，主要包括 k-shape 聚类和层次聚类两种方法。这些算法特别适用于时间序列数据，能够有效捕捉时序数据的形状特征进行分组。

## 核心技术

### 形状相似度计算 (SBD - Shape-Based Distance)

形状相似度测量是该聚类算法的基础，通过以下步骤实现：

- **时间序列标准化**：使用 `zscore()` 函数将时间序列数据标准化
- **互相关计算**：使用 `_ncc_c()` 函数计算两个时间序列的归一化互相关系数
- **最优对齐**：确定时间序列间最佳对齐位置，通过 `_sbd()` 函数计算形状距离

核心代码示例：
```python
def _sbd(x, y):
    """计算两个时间序列的形状距离和最优对齐"""
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))
    return dist, yshift
```
### k-shape 聚类算法
k-shape 是一种专为时间序列设计的划分聚类方法，类似于 k-means，但使用形状距离而非欧氏距离：

- 随机初始化：随机为每个时间序列分配聚类标签
- 迭代优化：
    - 提取聚类中心：使用 `_extract_shape()` 函数计算每个聚类的代表形状
    - 重新分配：基于形状距离将每个时间序列分配到最近的聚类
```py
def _kshape(x, k):
    m = x.shape[0]  # 一共有m组数据
    idx = randint(0, k, size=m)  # 生成k个最小为0的数据
    centroids = np.zeros((k, x.shape[1]))  # k行，timestamps列
    distances = np.empty((m, k))  # m行 k列

    for _ in range(100):
        old_idx = idx
        for j in range(k):
            centroids[j] = _extract_shape(idx, x, j, centroids[j])

        distances = (1 - _ncc_c_3dim(x, centroids).max(axis=2)).T

        idx = distances.argmin(1)
        if np.array_equal(old_idx, idx):
            break

    return idx, centroids
```

### 3. 层次聚类 (直接聚类法)
层次聚类通过自底向上的方式构建聚类树：

- 初始化：每个时间序列作为独立的叶子节点
- 合并：迭代找到最相似的两个节点合并
- 阈值划分：基于距离阈值确定最终聚类结果
```py
def direct_cluster(simi_matrix):
    ''' 直接聚类法 '''
    # 只用list来构建树
    N = len(simi_matrix)  # 得到矩阵对应的点的数目
    nodes = [make_leaf(label) for label in range(N)]  # 构建一堆叶子节点
    np.fill_diagonal(simi_matrix, float('Inf'))  # 先用最大值填充对角线
    root = 0  # 用于记录根节点的下标
    while N > 1:
        # 然后视图寻找相似度矩阵中最小的那个数的下标
        idx = np.where(simi_matrix == simi_matrix.min())
        x, y = get_idx(idx)  # 得到下标值==> x 和 y 进行了合并
        distance = simi_matrix[x][y]  # 得到两者的距离
        cluster = make_cluster(distance, nodes[x], nodes[y])
        nodes[y] = cluster  # 更新
        root = y  # 更新下标值
        # 删除x行, y列的元素
        simi_matrix[x] = float('Inf')
        simi_matrix[:, x] = float('Inf')
        N = N - 1
    return nodes[root]
```
## 应用场景
- 异常检测：识别异常时间序列模式
- 负载模式分析：分析系统负载的不同模式
- 用户行为：识别用户行为的典型模式
- 时序数据分类：将大量时序数据分组
## 算法优势
- 形状敏感：关注时间序列的形状特征而非绝对值
- 对齐能力：能处理时间偏移和不同长度的时间序列
- 可扩展性：支持大规模时序数据分析
- 层次结构：层次聚类可提供数据的多层次划分视图


# 压缩感知重建算法（cvxpy.py）

## 简介
该算法通过离散余弦变换 (DCT) 实现信号的稀疏表示，并使用二阶凸规划 (SOCP) 优化方法重建完整信号，适用于时序数据的压缩重建。

## 算法原理
压缩感知基于以下核心思想：

- **稀疏性**：信号在某个变换域（如 DCT）中具有稀疏表示
- **不相干采样**：采样矩阵与稀疏基之间存在低相干性
- **非线性重建**：通过 L1 范数最小化重建原始信号

## 算法流程
1. **构建稀疏变换矩阵（离散余弦变换）**
2. **根据采样点位置构建观测矩阵**
3. **求解 L1 范数最小化问题**
4. **将重建的稀疏信号逆变换回原始域**

## 核心函数
1. **二维离散余弦变换**
```py
def dct2(x):
    # 对二维数据执行正交形式的离散余弦变换
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    # 对二维数据执行正交形式的逆离散余弦变换
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
```
2. **信号重建函数**
```py
def reconstruct(n, d, index, value):
    """
    压缩感知采样重建算法
    :param n: 需重建数据的数据量
    :param d: 需重建数据的维度
    :param index: 采样点的时间维度坐标 属于[0, n-1]
    :param value: 采样点的KPI值，shape=(m, d), m为采样数据量
    :return:x_re: 重建的KPI数据，shape=(n, d)
    """
    #省略了部分初始化代码
    # 根据采样点位置构建观测矩阵
    ri = index
    b = x.T.flat[ri]
    transform_mat = np.kron(
        spfft.idct(np.identity(d), norm='ortho', axis=0),
        spfft.idct(np.identity(n), norm='ortho', axis=0)
    )
    transform_mat = transform_mat[ri, :]  
    # 求解 L1 范数最小化问题
    vx = cvx.Variable(d * n)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [transform_mat @ vx == b]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='OSQP')
    x_transformed = np.array(vx.value).squeeze()
    # 重建信号
    x_t = x_transformed.reshape(d, n).T  
    x_re = idct2(x_t)
```
## 优化模型
该算法求解的优化问题形式为：

$$
\min_{\mathbf{v}} |\mathbf{v}|_1 \quad \text{s.t.} \quad \Phi\Psi\mathbf{v} = \mathbf{b}
$$

其中：
- $\mathbf{v}$ 是待求解的稀疏系数向量
- $|\mathbf{v}|_1$ 是 $\mathbf{v}$ 的 L1 范数
- $\Phi$ 是采样矩阵
- $\Psi$ 是 DCT 变换矩阵
- $\mathbf{b}$ 是采样数据向量

## 特点
- 高效构建二维 DCT 变换矩阵，避免直接构建大矩阵带来的内存问题
- 使用自定义的 GPU 加速 OSQP 求解器(custom_solver.py)
- 采样点索引处理：支持多维数据的扁平化索引转换，实现从随机采样位置到线性索引的映射

## 应用场景
- 时序数据压缩存储：在保留关键特征的情况下减少数据存储需求
- 缺失数据修复：根据部分观测恢复完整数据
- 高效传输：传输关键数据点，接收端重建完整信号
- 系统状态监控：从稀疏采样点重建完整系统状态
## 其他
- 采样率：原始数据量的 20%-30% 通常可以获得良好的重建效果
- 适用于具有一定稀疏性或平滑性的数据

# SPOT: 流式极值检测算法(spot.py)

## 算法概述
SPOT (Streaming Peaks-Over-Threshold) 是一种基于统计极值理论的实时异常检测算法。该算法特别适用于连续数据流中的异常点检测，能够动态调整阈值，适应数据分布的变化。

## 核心原理
SPOT 基于极值理论中的广义帕累托分布 (GPD)，主要包括以下核心思想：

- **阈值超越法**：跟踪超过初始阈值的峰值。
- **动态阈值更新**：根据观测数据实时调整异常检测阈值。
- **自适应能力**：能够适应数据分布的逐渐变化。

## 算法
实现了四种 SPOT 算法变体：

- **SPOT**：基础版本，用于单侧（上界）异常检测。
- **biSPOT**：双向版本，同时检测上下两侧异常。
- **dSPOT**：带漂移补偿的单侧版本，通过移动平均处理非平稳数据。
- **bidSPOT**：带漂移补偿的双向版本，结合了 biSPOT 和 dSPOT 的优点。

## 实现
每个 SPOT 算法类都遵循相似的使用流程：

1. **初始化**：创建检测器对象并设置风险参数。
2. **数据导入**：导入初始校准数据和流数据。
3. **算法校准**：基于初始数据计算初始阈值。
4. **运行检测**：在数据流上运行算法。
5. **可视化结果**：绘制检测结果。

## 主要函数
SPOT 算法的关键函数包括：

- `_grimshaw()`：使用 Grimshaw 方法估计 GPD 参数。
- `_quantile()`：根据估计的参数计算极端分位数。
- `_rootsFinder()`：使用数值方法寻找方程的根。
- `_log_likelihood()`：计算 GPD 分布的对数似然。
- `backMean()`：计算移动平均（用于漂移版本）。

## 应用场景
SPOT 算法适用于：

- **系统监控**：检测流量异常或性能问题。
- **安全保护**：识别异常行为或入侵检测。


## 优势特点
- **无需假设数据分布**：基于极值理论，适应各种数据分布。
- **在线学习能力**：能够实时调整模型参数。
- **带漂移补偿**：可处理非平稳时间序列。
- **可配置的风险水平**：通过参数 $ q $ 控制检测灵敏度。
- **双向检测支持**：可同时检测异常高值和异常低值。

## 其他
- **$ q $（风险参数）**：通常设置为 $ 10^{-4} $ 到 $ 10^{-2} $，值越小阈值越严格。
- **depth（窗口大小）**：对于 dSPOT/bidSPOT，建议根据数据周期特性设置。
- **校准数据量**：需要使用足够多的数据点（至少几百个）进行初始化。

