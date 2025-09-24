# Utils_Detect: 时间序列分析与异常检测工具集

## 概述
`utils_detect` 为时间序列数据的预处理、转换、可视化和评估提供了丰富的函数。
## 核心组件

### 数据处理 (data_process.py)
提供基础的数据预处理函数：
- **normalization**: 数据归一化，将数据映射到 [0,1] 范围
- **standardization**: 数据标准化，转换为均值为0、标准差为1的分布
- **median**: 计算列表中位数
- **smooth**: 使用滑动平均进行数据平滑处理

### 降维与距离计算 (reduce_dimension.py)
提供多种数据降维和距离计算方法：
- **降维函数 reduce_dimension**: 支持多种距离度量方式
  - **Mean**: 基于平均绝对差异
  - **Euclidean**: 欧几里得距离
  - **Manhattan**: 曼哈顿距离
  - **Chebyshev**: 切比雪夫距离
  - **Cosine**: 余弦相似度
- **数据标准化 norm**: 多种标准化方法
  - **linear**: 线性归一化
  - **z-score**: Z分数标准化
  - **atan/sigmoid/tanh**: 非线性变换

### 可视化工具 (plot.py & plot_subimages.py)
全面的可视化函数集合：
- **基本绘图**
  - **plot_PRC**: 绘制精确率-召回率曲线
  - **plot_diff**: 比较原始数据与重建数据
- **KPI数据可视化**
  - **plot_raw_kpi**: 绘制原始KPI数据
  - **plot_cluster_kpi**: 绘制聚类后的KPI数据
  - **plot_sample_kpi**: 展示带采样点的KPI数据
  - **plot_reconstruct_kpi**: 对比原始与重建的KPI数据
- **综合异常检测可视化**
  - **plot_time_series_with_anomalies**: 多层次展示时间序列、重建数据和异常分数

### 评估指标 (metrics.py)
提供全面的异常检测评估工具：
- **基础指标计算**
  - **precision_recall_score**: 计算精确率和召回率
  - **f_score**: 计算F1分数
- **阈值确定**
  - **search_best_score**: 搜索最佳F1分数的阈值
  - **dynamic_threshold**: 基于统计特性的动态阈值设置
  - **sliding_anomaly_predict**: 滑动窗口动态阈值预测
- **SPOT/dSPOT集成**
  - **spot_eval**: 使用SPOT方法计算异常阈值
  - **dspot_eval**: 使用带漂移补偿的dSPOT方法计算阈值
