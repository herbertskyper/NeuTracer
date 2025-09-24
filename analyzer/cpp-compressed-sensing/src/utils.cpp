#include <algorithm>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include <cmath>

// 计算百分位数
double percentile(std::vector<double>& data, double percent) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    double k = (percent / 100.0) * (data.size() - 1);
    size_t f = std::floor(k);
    size_t c = std::ceil(k);
    if (f == c) return data[f];
    return data[f] * (c - k) + data[c] * (k - f);
}

// C++版本的 anomaly_score_example
double anomaly_score_example(const Eigen::MatrixXd& source, const Eigen::MatrixXd& reconstructed, int percentage = 90, int topn = 2) {
    int n = source.rows();
    int d = source.cols();
    std::vector<double> d_dis(d, 0.0);

    for (int i = 0; i < d; ++i) {
        std::vector<double> dis(n);
        double mean = 0.0;
        for (int j = 0; j < n; ++j) {
            dis[j] = std::abs(source(j, i) - reconstructed(j, i));
            mean += dis[j];
        }
        mean /= n;
        for (int j = 0; j < n; ++j) dis[j] -= mean;
        d_dis[i] = percentile(dis, percentage);
    }

    // 计算分数
    if (d <= topn) {
        double sum_inv = 0.0;
        for (int i = 0; i < d; ++i) sum_inv += 1.0 / d_dis[i];
        return d / sum_inv;
    } else {
        // 取倒数后最大的topn个
        std::vector<std::pair<double, int>> inv_dis(d);
        for (int i = 0; i < d; ++i) inv_dis[i] = {1.0 / d_dis[i], i};
        std::sort(inv_dis.begin(), inv_dis.end(), [](auto& a, auto& b) { return a.first > b.first; });
        double sum_topn = 0.0;
        for (int i = 0; i < topn; ++i) sum_topn += inv_dis[i].first;
        return topn / sum_topn;
    }
}

Eigen::VectorXi dynamic_threshold(const Eigen::VectorXd& score, double ratio) {
    double mean = score.mean();
    double std = std::sqrt((score.array() - mean).square().mean());
    double threshold = mean + ratio * std;
    if (threshold >= 1.0) threshold = 0.999;
    else if (threshold <= 0.0) threshold = 0.001;
    Eigen::VectorXi proba(score.size());
    for (int i = 0; i < score.size(); ++i)
        proba[i] = (score[i] > threshold) ? 1 : 0;
    return proba;
}

// 滑动窗口动态阈值异常预测
Eigen::VectorXi sliding_anomaly_predict(const Eigen::VectorXd& score, int window_size, int stride, double ratio) {
    int n = score.size();
    Eigen::VectorXi predict = Eigen::VectorXi::Zero(n);
    int start = 0;
    while (start < n) {
        int end = std::min(n, start + window_size);
        Eigen::VectorXd window = score.segment(start, end - start);
        Eigen::VectorXi window_pred = dynamic_threshold(window, ratio);
        for (int i = start; i < end; ++i)
            predict[i] = window_pred[i - start];
        start += stride;
    }
    return predict;
}