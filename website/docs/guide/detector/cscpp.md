# 压缩感知C++版本

此外，项目还基于 Osqp 和 Eigen 的 C++ 库实现了压缩感知算法的高性能 C++ 版本，进一步提升了大规模数据处理的效率。由于时间有限，加上压缩感知的数学问题形式比较复杂，我们为了保证算法正确性，简化了一些并行逻辑，因此加速比只有2.3倍。未来预计对 压缩感知C++版本 进行进一步优化。
```cpp
#include "detector.h"

int main(int argc, char const *argv[])
{
    Eigen::MatrixXd input_matrix(total_rows, total_cols);
    for (int i = 0; i < total_rows; ++i) {
        auto row = csv.GetRow<double>(i);
        for (int j = 0; j < total_cols; ++j)
            input_matrix(i, j) = row[j];
    }

    int cycle = reconstruct_window * windows_per_cycle;
    int n = input_matrix.rows();
    int d = input_matrix.cols();

    Eigen::MatrixXd norm_matrix = input_matrix;
    for (int i = 0; i < d; ++i) {
        double minv = norm_matrix.col(i).minCoeff();
        double maxv = norm_matrix.col(i).maxCoeff();
        double range = maxv - minv;
        if (range == 0)
            norm_matrix.col(i).setConstant(0.5);
        else
            norm_matrix.col(i) = (norm_matrix.col(i).array() - minv) / range;
    }

    std::vector<std::vector<std::vector<int>>> cycle_groups;
    while (cb < n) {
        int ce = std::min(n, cb + cycle);
        if (group_index == 0) {
            std::vector<std::vector<int>> init_group;
            for (int i = 0; i < d; ++i)
                init_group.push_back({i});
            cycle_groups.push_back(init_group);
        } else {
            Eigen::MatrixXd cycle_data = norm_matrix.block(cb, 0, ce - cb, d);
            auto groups = cluster(cycle_data, cluster_threshold);
            cycle_groups.push_back(groups);
        }
        group_index++;
        cb += cycle;
    }

    Eigen::MatrixXd reconstructed = Eigen::MatrixXd::Zero(n, d);
    Eigen::VectorXd reconstructing_weight = Eigen::VectorXd::Zero(n);

    while (win_l < n) {
        int win_r = std::min(n, win_l + reconstruct_window);
        if (win_r - win_l < reconstruct_window / 2) break;

        Eigen::MatrixXd window_data = norm_matrix.block(win_l, 0, win_r - win_l, d);
        int cycle_id = std::min((int)cycle_groups.size() - 1, win_l / cycle);
        const auto& groups = cycle_groups[cycle_id];

        Eigen::MatrixXd outlier_score;
        if (win_l == 0) {
            outlier_score = Eigen::MatrixXd::Ones(win_r - win_l, 1);
        } else {
            int hb = std::max(0, win_l - latest_windows);
            Eigen::MatrixXd latest_data = norm_matrix.block(hb, 0, win_l - hb, d);
            auto tmp_score = lesinn(window_data, latest_data, 40, 20);
            outlier_score = 1.0 / tmp_score.array();
            double min_score = outlier_score.minCoeff();
            double max_score = outlier_score.maxCoeff();
            if (max_score > min_score) {
                outlier_score = (outlier_score.array() - min_score) / (max_score - min_score);
            } else {
                outlier_score = Eigen::MatrixXd::Constant(outlier_score.rows(), outlier_score.cols(), 0.5);
            }
        }

        bool success = false;
        double current_sample_rate = sample_rate;
        Eigen::MatrixXd rec_window = Eigen::MatrixXd::Zero(win_r - win_l, d);
        while (!success && retry_count < max_retry) {
            Eigen::MatrixXd sampled_value;
            std::vector<int> su_rows;
            std::tie(sampled_value, su_rows) =
                sample(window_data, std::round((win_r - win_l) * current_sample_rate), outlier_score);
            success = true;

            for (const auto& group : groups) {
                Eigen::MatrixXd a = sampled_value(Eigen::all, group);
                auto cs_sub_matrix_opt = reconstruct(win_r - win_l, su_rows, a);

                if (cs_sub_matrix_opt.size() == 0) {
                    current_sample_rate += (1.0 - current_sample_rate) / 4.0;
                    retry_count++;
                    success = false;
                    break;
                }

                auto& cs_sub_matrix = cs_sub_matrix_opt;
                for (size_t i = 0; i < group.size(); ++i)
                    rec_window.col(group[i]) = cs_sub_matrix.col(i);
            }

            if (!success && retry_count >= max_retry) {
                rec_window = window_data;
                break;
            }
        }

        for (int i = 0; i < win_r - win_l; ++i) {
            int global_idx = win_l + i;
            double weight = reconstructing_weight[global_idx];
            reconstructed.row(global_idx) =
                (reconstructed.row(global_idx) * weight + rec_window.row(i)) / (weight + 1.0);
            reconstructing_weight[global_idx] += 1.0;
        }

        win_l += reconstruct_stride;
        window_count++;
    }

    Eigen::VectorXd anomaly_score = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd anomaly_score_weight = Eigen::VectorXd::Zero(n);

    int wb = 0;
    while (wb < n) {
        int we = std::min(n, wb + detect_window);
        if (we - wb < detect_window / 2) break;

        Eigen::MatrixXd win_data = norm_matrix.block(wb, 0, we - wb, d);
        Eigen::MatrixXd win_rec = reconstructed.block(wb, 0, we - wb, d);

        double score = anomaly_score_example(win_data, win_rec, 90, 2);

        for (int i = 0; i < we - wb; ++i) {
            int global_idx = wb + i;
            double weight = anomaly_score_weight[global_idx];
            anomaly_score[global_idx] =
                (anomaly_score[global_idx] * weight + score) / (weight + 1.0);
            anomaly_score_weight[global_idx] += 1.0;
        }

        wb += detect_stride;
    }

    Eigen::VectorXi predict = sliding_anomaly_predict(anomaly_score, detect_window, detect_stride, 3.0);
    
    return predict;
}
```

小结：通过以上这些优化措施，压缩感知模块不仅在 Python 环境下具备高效性，也能在 C++ 环境下实现更低延迟和更高吞吐量，满足生产级系统的实时监控需求。