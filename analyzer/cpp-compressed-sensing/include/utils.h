#pragma once
#include <vector>
#include <Eigen/Dense>

// 计算百分位数
double percentile(std::vector<double>& data, double percent);

// 计算异常分数（与 anomaly_score_example Python 版本等价）
double anomaly_score_example(const Eigen::MatrixXd& source, const Eigen::MatrixXd& reconstructed, int percentage = 90, int topn = 2);

// 动态阈值
Eigen::VectorXi dynamic_threshold(const Eigen::VectorXd& score, double ratio = 3.0);

// 滑动窗口动态阈值异常预测
Eigen::VectorXi sliding_anomaly_predict(const Eigen::VectorXd& score, int window_size = 1440, int stride = 10, double ratio = 3.0);