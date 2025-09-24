# NeuTracer 异常检测模块

本模块是 NeuTracer 项目的压缩感知异常检测组件，利用稀疏采样和信号重建技术高效检测时间序列数据中的异常。

## 功能概述

NeuTracer 异常检测模块使用压缩感知理论，通过稀疏采样和精确重建实现高效的时间序列异常检测。主要特点：

- **智能采样**：基于 LESINN 算法自适应选择信息量最大的数据点
- **时间序列分组**：自动聚类相关时间序列，提高重建精度
- **精确重建**：利用 CVXPY 优化框架重建完整信号
- **异常识别**：基于重建误差检测异常点
- **可视化分析**：提供多种直观的可视化结果

## 快速开始

### 前提条件

- Python 3.7+
- 相关依赖包：numpy, pandas, matplotlib, scikit-learn, tqdm, pyyaml, cvxpy

### 安装

```bash
pip install -r requirements.txt
```

### 使用方法

1. 配置检测参数：
```yaml
# 编辑 detector-config.yml
detector_arguments:
  sample_rate: 0.66  # 采样率
  cluster_threshold: 0.01  # 聚类阈值
  latest_windows: 100  # 历史窗口数
  # 其他参数...
```

2. 运行检测：
```python
from detect import detect

# 输入文件、输出文件、起始位置
detect("input.csv", "output.csv", 0)
```

3. 查看结果：
```
检测到xx个异常点
异常分数范围：0.0056 - 3.2145
异常分数平均值：0.4235
异常分数标准差：0.5612
阈值：2.1071
```

## 核心组件

### 1. 窗口重建处理 (WindowReconstructProcess)

负责对滑动窗口内的数据进行采样和重建：

```python
process = WindowReconstructProcess(
    data=data,
    cycle=cycle,
    latest_windows=latest_windows,
    sample_rate=sample_rate,
    scale=scale, 
    rho=rho, 
    sigma=sigma,
    random_state=random_state,
    retry_limit=retry_limit
)
```

### 2. 采样算法 (localized_sample)

使用局部化采样策略，根据点重要性分数选择最具代表性的样本点：

```python
sample_matrix, timestamp = localized_sample(
    x=data_mat, m=m,
    score=score,
    scale=self.scale, rho=self.rho, sigma=self.sigma,
    random_state=random_state
)
```

### 3. 聚类 (cluster)

自动将相关的时间序列分组，提高重建精度：

```python
cycle_groups.append(cluster(data[cb:ce], cluster_threshold))
```

### 4. 异常分数计算 (anomaly_score_example)

基于原始数据与重建数据之间的差异计算异常分数：

```python
score = anomaly_score_example(data[wb:we], reconstructed[wb:we])
```

### 5. 异常预测 (sliding_anomaly_predict)

使用滑动窗口和动态阈值进行异常检测：

```python
predict = sliding_anomaly_predict(anomaly_score, 50, 5, 3)
```

## 参数配置

`detector-config.yml` 文件包含以下关键参数：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| sample_rate | 采样率（采样点占总点数比例） | 0.5-0.7 |
| cluster_threshold | 时间序列聚类的相似度阈值 | 0.01-0.05 |
| latest_windows | 计算采样重要性时参考的历史窗口数 | 50-200 |
| scale | 采样扩展比例 | 3-10 |
| rho | 中心采样概率 | 0.1-0.3 |
| sigma | 高斯采样标准差 | 0.3-0.7 |
| retry_limit | 重建失败时的重试次数上限 | 50-100 |
| reconstruct.window | 重建窗口大小 | 30-100 |
| reconstruct.stride | 滑动步长 | 1-10 |

## 输出结果

1. **异常检测标记**：包含 0（正常）和 1（异常）的 CSV 文件
2. **可视化图表**：
   - 原始数据图 (`_raw_kpi.png`)
   - 重建对比图 (`_reconstruct_kpi.png`)
   - 异常点标记图 (`_diff_plot.png`)
   - 带时间轴的完整分析图 (`_time_series.png`)
   - 聚类结果图 (`_cluster_kpi.png`，多维数据专用）

## 技术原理

本模块基于压缩感知理论，遵循以下步骤：

1. **数据预处理**：对多维时间序列数据进行归一化
2. **相关性聚类**：将相关的时间序列分为同一组
3. **自适应采样**：基于 LESINN 算法选择信息量最大的点
4. **信号重建**：使用凸优化方法重建完整信号
5. **异常检测**：计算重建误差，识别超出阈值的异常点

## 特点

- **自适应采样率**：重建失败时自动调整采样率
- **进度处理**：使用 tqdm 提供进度反馈

## 限制与注意事项

- 重建质量依赖于数据的稀疏性和合适的采样率
- 处理高维数据时可能需要更高的采样率
- 重建过程可能在某些情况下失败，此时会自动增加采样率重试

