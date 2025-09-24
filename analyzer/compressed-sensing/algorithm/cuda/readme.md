# LESINN 异常检测算法介绍
LESINN (Local Exceptionality Similarity-based INdex of Nonconformity) 是一个用于异常检测的算法，在项目中有 Python 和 CUDA 实现版本。

## 算法原理
LESINN 基于相似度计算来识别数据中的异常点。其核心思想是：

- 计算每个数据点与随机选择的子集中最相似点的相似度
- 重复多次采样，累计相似度分数
- 相似度得分越低（最终输出的异常分数越高）的点越可能是异常点
## 实现对比
Python 实现 (lesinn.py)
Python 版本提供了两个主要函数：

lesinn：处理完整数据集
online_lesinn：在线处理新增数据与历史数据
相似度计算采用以下公式：
```py
a = 1 / (1 + np.sqrt(np.sum(np.square(x - y))))
```
CUDA 实现 (lesinn.cu)
CUDA 版本利用 GPU 并行加速，每个线程处理一个数据点：

- 使用 cuRAND 生成随机样本
- 并行计算每个数据点的异常分数

核心实现细节：

- 每个线程计算一个数据点的异常分数
- 为每个数据点随机选取 t 组样本，每组大小为 phi
- 计算与每组样本中最相似点的相似度，并累加
- 最终异常分数 = t / 累计相似度
```cpp
for (size_t data_id = 0; data_id < dimension; data_id++)
            {
                tmp += (current_data[data_id] - all_data[sample * dimension + data_id]) * (current_data[data_id] - all_data[sample * dimension + data_id]);
            }
            tmp = sqrt(tmp);
            tmp = 1 / (1 + tmp);
            max_sim = max(max_sim, tmp);
```
## 应用场景

- 多维数据异常检测
- 实时数据流异常点识别
- 可扩展到大规模数据集