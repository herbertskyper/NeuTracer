# 压缩感知重建算法

压缩感知重建算法通过离散余弦变换（DCT）实现信号的稀疏表示，并使用凸规划方法重建完整信号，适用于时序数据的压缩重建。该算法基于三个核心思想：信号在某个变换域（如DCT）中具有稀疏表示，采样矩阵与稀疏基之间存在低相干性，通过L1范数最小化重建原始信号。算法流程包括构建稀疏变换矩阵（离散余弦变换），根据采样点位置构建观测矩阵，求解L1范数最小化问题，将重建的稀疏信号逆变换回原始域。

```python
def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def reconstruct(n, d, index, value):
    ri = index
    b = x.T.flat[ri]
    transform_mat = np.kron(
        spfft.idct(np.identity(d), norm='ortho', axis=0),
        spfft.idct(np.identity(n), norm='ortho', axis=0)
    )
    transform_mat = transform_mat[ri, :]
    
    # L1范数最小化求解
    vx = cvx.Variable(d * n)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [transform_mat @ vx == b]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='OSQP')
    
    x_transformed = np.array(vx.value).squeeze()
    x_t = x_transformed.reshape(d, n).T
    x_re = idct2(x_t)
```
DCT变换通过正交形式的离散余弦变换对二维数据进行稀疏表示，逆变换则将稀疏系数恢复为原始信号。重建函数接收需重建数据的数据量、维度、采样点坐标和KPI值，输出重建的完整KPI数据。算法通过构建观测矩阵，利用Kronecker积高效构建二维DCT变换矩阵，避免直接构建大矩阵带来的内存问题。该算法求解的优化问题形式为：min ||v||₁ s.t. Av = b，其中v是待求解的稀疏系数向量，||v||₁是v的L1范数，A是采样矩阵Φ与DCT变换矩阵Ψ的乘积，b是采样数据向量。算法使用OSQP求解器，原始数据量的40%-50%采样率通常可以获得良好的重建效果。