#include "detector.h"

#include "rapidcsv.h"
#include <vector>

// detect
void Detector::detect()
{
    std::vector<std::vector<double>> input;
    std::vector<Eigen::MatrixXd> rec_windows; // 保存每个窗口的重建结果
    std::vector<Eigen::MatrixXd> raw_windows; // 保存每个窗口的原始数据
    std::vector<double> anomaly_scores;       // 保存每个窗口的异常分数
    for (;;)
    {
        input.push_back(this->data.pop());
        if (input.size() >= this->window_size)
        {
            auto input_matrix = toEigenMatrix<double, std::vector<std::vector<double>>>(input);
            auto rows = input_matrix.rows();
            auto cols = input_matrix.cols();
            Eigen::MatrixXd reconstructed_matrix(rows, cols);

            // Normalization
            for (size_t i = 0; i < input_matrix.cols(); i++)
            {
                double range = input_matrix.col(i).maxCoeff() - input_matrix.col(i).minCoeff();
                if (range == 0)  // All the same
                    input_matrix.col(i) = Eigen::MatrixXd::Constant(input_matrix.rows(), 1, 0.5);
                else  // Do normalizatoin
                    input_matrix.col(i) = (input_matrix.col(i).array() - input_matrix.col(i).minCoeff()) / range;
            }

            auto groups = cluster(input_matrix, this->threshold);
            if (this->last_window_data.size() == 0)
                this->last_window_data = input_matrix;

            // LESINN outlier score
            auto tmp_score = lesinn(input_matrix, this->last_window_data, 40, 20);  // TODO: hard-coded t, phi

            // Inverse and normalization: the larger the score, the more likely it is an outlier
            Eigen::MatrixXd outlier_score = 1.0 / tmp_score.array();

            // Min-max normalization to [0, 1]
            double min_score = outlier_score.minCoeff();
            double max_score = outlier_score.maxCoeff();
            if (max_score > min_score) {
                outlier_score = (outlier_score.array() - min_score) / (max_score - min_score);
            } else {
                outlier_score = Eigen::MatrixXd::Constant(outlier_score.rows(), outlier_score.cols(), 0.5);
            }
            int retry_count = 0;
            const int max_retry = 5;
            bool success = false;
            double current_sample_rate = this->sample_rate;
            Eigen::MatrixXd rec_window = Eigen::MatrixXd::Zero(rows, cols);

            while (!success && retry_count < max_retry) {
                Eigen::MatrixXd sampled_value;
                std::vector<int> su_rows;
                std::tie(sampled_value, su_rows) =
                    sample(input_matrix, std::round(rows * current_sample_rate), outlier_score);

                success = true;
                for (const auto& group : groups) {
                    Eigen::MatrixXd a = sampled_value(Eigen::all, group);
                    auto cs_sub_matrix_opt = reconstruct(rows, su_rows, a);

                    if (cs_sub_matrix_opt.size() == 0) {
                        // 只要有一个 group 失败，整体重采
                        current_sample_rate += (1.0 - current_sample_rate) / 4.0;
                        std::cerr << "Reconstruct failed for group, increasing sample_rate to " << current_sample_rate
                                << ", retry " << retry_count + 1 << std::endl;
                        retry_count++;
                        success = false;
                        break;
                    }

                    // 重构成功
                    auto& cs_sub_matrix = cs_sub_matrix_opt;
                    for (size_t i = 0; i < group.size(); ++i)
                        rec_window.col(group[i]) = cs_sub_matrix.col(i);
                }

                if (!success && retry_count >= max_retry) {
                    std::cerr << "Reconstruct failed after " << max_retry << " retries, using original data." << std::endl;
                    rec_window = input_matrix;
                    break;
                }
            }

             // 保存窗口重建和原始数据
            rec_windows.push_back(reconstructed_matrix);
            raw_windows.push_back(input_matrix);

            // 计算异常分数
            double score = anomaly_score_example(input_matrix, reconstructed_matrix, 90, 2);
            anomaly_scores.push_back(score);

            // End of the algorithm
            input.clear();
            this->last_window_data = input_matrix;
        }
    }
}

void Detector::submit(std::vector<std::vector<double>> &input)
{
    for (auto item : input)
    {
        this->data.push(std::move(item));
    }
}

// Constructor
Detector::Detector(int window_size, double threshold)
{
    this->window_size = window_size;
    this->threshold = threshold;
    this->sample_rate = 0.4; // Initial sample rate

    this->detect_thread = std::thread([this]() { this->detect(); });
}

Detector::~Detector()
{
    this->detect_thread.join();
}
// Implement a data loader
// loads data from a CSV file and feeds it to the detector

ClusterManager::ClusterManager(const Eigen::MatrixXd& data,
                               int rec_windows_per_cycle,
                               int reconstruct_window,
                               double cluster_threshold)
{
    int n = data.rows();
    int d = data.cols();
    cycle_ = rec_windows_per_cycle * reconstruct_window;
    int group_index = 0;
    for (int cb = 0; cb < n; cb += cycle_) {
        int ce = std::min(n, cb + cycle_);
        if (group_index == 0) {
            // 第一个周期：每个KPI一组
            Group init_group;
            for (int i = 0; i < d; ++i)
                init_group.push_back({i});
            cycle_groups_.push_back(init_group);
        } else {
            // 其它周期：聚类
            Eigen::MatrixXd cycle_data = data.block(cb, 0, ce - cb, d);
            cycle_groups_.push_back(cluster(cycle_data, cluster_threshold));
        }
        group_index++;
    }
}

const Group& ClusterManager::get_group_for_window(int win_l) const {
    int cycle_id = win_l / cycle_;
    if (cycle_id >= cycle_groups_.size())
        return cycle_groups_.back();
    return cycle_groups_[cycle_id];
}

int main(int argc, char const *argv[])
{
    const int window_size = 60;
    auto detector = new Detector(window_size, 0.01);
    // auto cluster_manager = new ClusterManager(detector->last_window_data, 10, 5, 0.01);

    // Get the data
    std::string data_file = "./col2.csv";
    if (argc > 1)
        data_file = argv[1];

    // Read in
    rapidcsv::Document csv(data_file);

    int total_rows = csv.GetRowCount();
    std::vector<std::vector<double>> submit_data;
    for (size_t i = 0; i < total_rows; i++)
    {
        submit_data.push_back(csv.GetRow<double>(i));
        if (i % window_size == 0)
        {
            detector->submit(submit_data);
            submit_data.clear();
        }
    }
    std::cout << "Number of rows submitted: " << total_rows << " rows" << std::endl;
    delete detector;  // Wait for detector
    return 0;
}
