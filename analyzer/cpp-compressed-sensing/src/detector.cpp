#include "detector.h"
#include "rapidcsv.h"
#include <vector>
#include <iostream>
#include <fstream>
// #define DEBUG

int main(int argc, char const *argv[])
{
    // 1. 读取数据
    std::string data_file = "./col1.csv";
    if (argc > 1)
        data_file = argv[1];

    rapidcsv::Document csv(data_file);
    int total_rows = csv.GetRowCount();
    int total_cols = csv.GetColumnCount();
    
    std::cout << "Data loaded: " << total_rows << " rows, " << total_cols << " cols" << std::endl;

    Eigen::MatrixXd input_matrix(total_rows, total_cols);
    for (int i = 0; i < total_rows; ++i) {
        auto row = csv.GetRow<double>(i);
        for (int j = 0; j < total_cols; ++j)
            input_matrix(i, j) = row[j];
    }

    // 2. 参数配置（硬编码，可改为从配置文件读取）
    int reconstruct_window = 60;
    int reconstruct_stride = 10;
    int detect_window = 12;
    int detect_stride = 2;
    double cluster_threshold = 0.01;
    double sample_rate = 0.4;
    int windows_per_cycle = 24;
    int latest_windows = 96;

    // 计算周期长度
    int cycle = reconstruct_window * windows_per_cycle;
    int n = input_matrix.rows();
    int d = input_matrix.cols();

    // 3. 数据归一化
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
#ifdef DEBUG
std::ofstream norm_file("norm_matrix_cpp.txt");
for (int i = 0; i < norm_matrix.rows(); ++i) {
    for (int j = 0; j < norm_matrix.cols(); ++j) {
        norm_file << norm_matrix(i, j);
        if (j + 1 < norm_matrix.cols()) norm_file << ",";
    }
    norm_file << "\n";
}
norm_file.close();
#endif

    // 4. 周期聚类分组
    std::vector<std::vector<std::vector<int>>> cycle_groups;
    int group_index = 0;
    int cb = 0;
    while (cb < n) {
        int ce = std::min(n, cb + cycle);
        if (group_index == 0) {
            // 第一个周期，每个KPI一组
            std::vector<std::vector<int>> init_group;
            for (int i = 0; i < d; ++i)
                init_group.push_back({i});
            cycle_groups.push_back(init_group);
            std::cout << "First cycle: each KPI as a group, total " << d << " groups" << std::endl;
        } else {
            // 其它周期，聚类
            Eigen::MatrixXd cycle_data = norm_matrix.block(cb, 0, ce - cb, d);
            auto groups = cluster(cycle_data, cluster_threshold);
            cycle_groups.push_back(groups);
            std::cout << "Cycle " << group_index << ": clustered into " << groups.size() << " groups" << std::endl;
        }
        group_index++;
        cb += cycle;
    }

    // 5. 滑动窗口重构
    Eigen::MatrixXd reconstructed = Eigen::MatrixXd::Zero(n, d);
    Eigen::VectorXd reconstructing_weight = Eigen::VectorXd::Zero(n);
    
    int win_l = 0;
    int window_count = 0;
    while (win_l < n) {
        int win_r = std::min(n, win_l + reconstruct_window);
        if (win_r - win_l < reconstruct_window / 2) break; // 避免过小的窗口
        
        Eigen::MatrixXd window_data = norm_matrix.block(win_l, 0, win_r - win_l, d);
        
        // 获取当前窗口所属周期的分组
        int cycle_id = std::min((int)cycle_groups.size() - 1, win_l / cycle);
        const auto& groups = cycle_groups[cycle_id];

        // 计算采样分数（简化版，可用LESINN替换）
        Eigen::MatrixXd outlier_score;
        if (win_l == 0) {
            // 第一个窗口，使用全1分数
            outlier_score = Eigen::MatrixXd::Ones(win_r - win_l, 1);
        } else {
            // 使用历史窗口计算LESINN分数
            int hb = std::max(0, win_l - latest_windows);
            Eigen::MatrixXd latest_data = norm_matrix.block(hb, 0, win_l - hb, d);
            auto tmp_score = lesinn(window_data, latest_data, 40, 20);
            outlier_score = 1.0 / tmp_score.array();
// #ifdef DEBUG
// if(win_l < 50){
//     std::ofstream score_file("sample_score_cpp_" + std::to_string(win_l) + ".txt");
//     for (int i = 0; i < outlier_score.rows(); ++i)
//         score_file << tmp_score(i, 0) << "\n";
//     score_file.close();
// }
// #endif
            
            // 归一化到[0,1]
            double min_score = outlier_score.minCoeff();
            double max_score = outlier_score.maxCoeff();
            if (max_score > min_score) {
                outlier_score = (outlier_score.array() - min_score) / (max_score - min_score);
            } else {
                outlier_score = Eigen::MatrixXd::Constant(outlier_score.rows(), outlier_score.cols(), 0.5);
            }
        }

        // if (outlier_score.cols() > 1) {
        //     outlier_score = outlier_score.rowwise().mean();
        // }
#ifdef DEBUG
if (window_count < 6) {
    std::ofstream norm_score_file("normal_score_cpp_" + std::to_string(win_l) + ".txt");
    for (int i = 0; i < outlier_score.rows(); ++i)
        norm_score_file << outlier_score(i, 0) << "\n";
    norm_score_file.close();
}
#endif    


#ifdef DEBUG
if (window_count < 10) {
    std::ofstream win_file("window_data_cpp_" + std::to_string(win_l) + ".txt");
    for (int i = 0; i < window_data.rows(); ++i) {
        for (int j = 0; j < window_data.cols(); ++j) {
            win_file << window_data(i, j);
            if (j + 1 < window_data.cols()) win_file << ",";
        }
        win_file << "\n";
    }
    win_file.close();
}
#endif
        // 采样
        // Eigen::MatrixXd sampled_value;
        // std::vector<int> su_rows;
        // std::tie(sampled_value, su_rows) =
        //     sample(window_data, std::round((win_r - win_l) * sample_rate), outlier_score);

    int retry_count = 0;
    const int max_retry = 5;
    bool success = false;
    double current_sample_rate = sample_rate;
    Eigen::MatrixXd rec_window = Eigen::MatrixXd::Zero(win_r - win_l, d);
    while (!success && retry_count < max_retry) {
        // 整个窗口采样一次
        Eigen::MatrixXd sampled_value;
        std::vector<int> su_rows;
        std::tie(sampled_value, su_rows) =
            sample(window_data, std::round((win_r - win_l) * current_sample_rate), outlier_score);

        // Eigen::MatrixXd rec_window = Eigen::MatrixXd::Zero(win_r - win_l, d);
        success = true;

        for (const auto& group : groups) {
            Eigen::MatrixXd a = sampled_value(Eigen::all, group);
            auto cs_sub_matrix_opt = reconstruct(win_r - win_l,  su_rows, a);

            if (cs_sub_matrix_opt.size() == 0) {
                // 只要有一个 group 失败，整体重采
                current_sample_rate += (1.0 - current_sample_rate) / 4.0;
                std::cerr << "Reconstruct failed for window " << window_count
                        << ", group size " << group.size()
                        << ", increasing sample_rate to " << current_sample_rate
                        << ", retry " << retry_count + 1 << std::endl;
                retry_count++;
                success = false;
                break; // 跳出 group 循环，整体重采
            }

            // 重构成功
            auto& cs_sub_matrix = cs_sub_matrix_opt;
            for (size_t i = 0; i < group.size(); ++i)
                rec_window.col(group[i]) = cs_sub_matrix.col(i);
        }

        if (!success && retry_count >= max_retry) {
            std::cerr << "Reconstruct failed after " << max_retry
                    << " retries for window " << window_count << std::endl;
            // 用原始数据填充
            rec_window = window_data;
            break;
        }
    }        
        
        // 加权累积重建结果
        for (int i = 0; i < win_r - win_l; ++i) {
            int global_idx = win_l + i;
            double weight = reconstructing_weight[global_idx];
            reconstructed.row(global_idx) = 
                (reconstructed.row(global_idx) * weight + rec_window.row(i)) / (weight + 1.0);
            reconstructing_weight[global_idx] += 1.0;
        }
        
        win_l += reconstruct_stride;
        window_count++;
        
        if (window_count % 10 == 0) {
            std::cout << "Processed " << window_count << " windows, current position: " << win_l << "/" << n << std::endl;
        }
    }

    std::cout << "Reconstruction finished. Total windows processed: " << window_count << std::endl;

    // 6. 异常分数计算
    Eigen::VectorXd anomaly_score = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd anomaly_score_weight = Eigen::VectorXd::Zero(n);

    int wb = 0;
    while (wb < n) {
        int we = std::min(n, wb + detect_window);
        if (we - wb < detect_window / 2) break; // 避免过小的检测窗口
        
        Eigen::MatrixXd win_data = norm_matrix.block(wb, 0, we - wb, d);
        Eigen::MatrixXd win_rec = reconstructed.block(wb, 0, we - wb, d);

        // 计算窗口级异常分数
        double score = anomaly_score_example(win_data, win_rec, 90, 2);
        
        // 累积到全局异常分数
        for (int i = 0; i < we - wb; ++i) {
            int global_idx = wb + i;
            double weight = anomaly_score_weight[global_idx];
            anomaly_score[global_idx] = 
                (anomaly_score[global_idx] * weight + score) / (weight + 1.0);
            anomaly_score_weight[global_idx] += 1.0;
        }
        
        wb += detect_stride;
    }

    // 7. 动态阈值滑动窗口异常预测
    Eigen::VectorXi predict = sliding_anomaly_predict(anomaly_score, detect_window, detect_stride, 3.0);
    
    std::cout << "Anomaly detection completed:" << std::endl;

    // 8. 保存结果
    // 保存重建结果
    std::ofstream rec_file("rec.txt");
    if (rec_file.is_open()) {
        for (int i = 0; i < reconstructed.rows(); ++i) {
            for (int j = 0; j < reconstructed.cols(); ++j) {
                rec_file << reconstructed(i, j);
                if (j + 1 < reconstructed.cols()) rec_file << ",";
            }
            rec_file << "\n";
        }
        rec_file.close();
        std::cout << "Reconstructed data saved to rec.txt" << std::endl;
    }

    // 保存异常分数
    std::ofstream score_file("anomaly_score.txt");
    if (score_file.is_open()) {
        for (int i = 0; i < anomaly_score.size(); ++i)
            score_file << anomaly_score[i] << "\n";
        score_file.close();
        std::cout << "Anomaly scores saved to anomaly_score.txt" << std::endl;
    }

    // 保存异常点索引
    std::ofstream predict_file("predict.txt");
    if (predict_file.is_open()) {
        for (int i = 0; i < predict.size(); ++i)
            predict_file << predict[i] << "\n";
        predict_file.close();
        std::cout << "Prediction saved to predict.txt" << std::endl;
    }

    return 0;
}

