# 基于形状的时间序列聚类算法

该算法的核心是形状相似度计算（SBD - Shape-Based Distance），通过互相关计算和最优对齐来测量时间序列间的形状距离。最早由 Paparrizos 和 Gravano 在 2015 年论文《k-Shape: Efficient and Accurate Clustering of Time Series》中提出。SBD 通过最大化两条时间序列的互相关（Cross-Correlation）值来衡量它们的形状相似性，从而容忍相位偏移（时延）和振幅差异。经过测试，聚类效果比基于频谱的方式好。因为基于频谱的聚类方式容易收到噪声的影响，导致频谱特征不明显。

在我们的代码中，通过 `_ncc_c()` 函数计算两个时间序列的归一化互相关系数，确定时间序列间最佳对齐位置并计算形状距离。
```python
def _sbd(x, y):
    """计算两个时间序列的形状距离和最优对齐"""
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))
    return dist, yshift
    ```
 
在计算完每个维度的形状距离后，使用层次聚类通过自底向上的方式构建聚类树，每个时间序列作为独立的叶子节点初始化，迭代找到最相似的两个节点合并，基于距离阈值确定最终聚类结果。

```python
def cluster(X: np.ndarray, threshold: float = 0.01):
    ny, nx = X.shape

    if nx <= 1:
        return [[0]]  # 只有一个维度，所有点归为一类

    distance = np.zeros((nx, nx))
    for (i, j), v in np.ndenumerate(distance):
        distance[i, j] = _sbd(X[:, i], X[:, j])[0]
    tree = direct_cluster(distance)
    return (get_classify(threshold, tree))

def direct_cluster(simi_matrix):
    N = len(simi_matrix)
    nodes = [make_leaf(label) for label in range(N)]
    np.fill_diagonal(simi_matrix, float('Inf'))
    
    while N > 1:
        idx = np.where(simi_matrix == simi_matrix.min())
        x, y = get_idx(idx)
        distance = simi_matrix[x][y]
        cluster = make_cluster(distance, nodes[x], nodes[y])
        nodes[y] = cluster
        simi_matrix[x] = float('Inf')
        simi_matrix[:, x] = float('Inf')
        N = N - 1
    return nodes[root]
```
