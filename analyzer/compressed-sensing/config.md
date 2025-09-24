# 异常检测配置参数指南

## 采样评分方法 (sample_score_method)
控制如何为数据点分配采样重要性得分。

### LESINN 参数
```yml
lesinn:
  phi: 20   # 每次采样的样本数量
  t: 40     # 重复采样次数，值越大结果越稳定但计算量增加
```

## 异常评分 (anomaly_scoring)
定义如何基于重建误差计算异常分数。
```yml
anomaly_score_example:
  percentage: 90   # 选择百分比阈值，高于此值视为异常
  topn: 2          # 每个检测窗口中最多认定的异常点数量
```
## 全局设置 (global)
控制全局行为的参数。
```yml
global:
  random_state: 42   # 随机数种子，确保结果可重现
```
## 数据配置 (data)
数据源和处理参数。
```yml
data:  
  # 数据范围控制
  row_begin: 0        # 起始行
  row_end: 30000      # 结束行
  col_begin: 0        # 起始列
  col_end: 100        # 结束列
  
  rec_windows_per_cycle: 24   # 每个周期的重建窗口数
  
  # 重建窗口设置
  reconstruct:
    window: 50    # 重建窗口大小
    stride: 5     # 重建窗口滑动步长
  
  # 检测窗口设置
  detect:
    window: 12    # 检测窗口大小
    stride: 2     # 检测窗口滑动步长
```
## 检测器参数 (detector_arguments)
控制检测算法核心行为的参数。
```yml
detector_arguments:
  cluster_threshold: 0.01             # 聚类相似度阈值
  sample_rate: 0.66                   # 采样率（总样本点比例）
  latest_windows: 100                 # 使用的最新窗口数量
  
  # 局部化采样参数
  scale: 5                            # 采样密度控制参数 
  rho: 0.1                            # 中心点采样概率
  sigma: 0.5                          # 高斯分布标准差
  
  retry_limit: 100                    # 重试次数限制
```
