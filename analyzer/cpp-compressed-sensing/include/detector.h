#pragma once
#include "sync_queue.h"
#include "utils.h"
#include <Eigen/Dense>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

std::vector<std::vector<int>> cluster(Eigen::MatrixXd &x, float threshold);

Eigen::MatrixXd lesinn(Eigen::MatrixXd &input_data, Eigen::MatrixXd &last_window_data, int t, int phi);

std::tuple<Eigen::MatrixXd, std::vector<int>>
sample(Eigen::MatrixXd &input_data, int sample_amount, Eigen::MatrixXd &outlier_score);

Eigen::MatrixXd reconstruct(int original_rows, std::vector<int> &sample_rows, Eigen::MatrixXd &sampled_values);

// CVXPY风格重建（使用OSQP模拟CLARABEL）
Eigen::MatrixXd reconstruct_cvxpy(int n, int d, std::vector<int>& index, Eigen::MatrixXd& value);

// 直接线性规划重建（模拟scipy.linprog with HiGHS）
Eigen::MatrixXd reconstruct_my_cvxpy(int n, int d, std::vector<int>& index, Eigen::MatrixXd& value);

class Detector
{
   private:
    sync_queue<std::vector<double>> data;
    int window_size;
    double sample_rate = 0.5;
    double threshold;
    void detect();
    std::thread detect_thread;
    Eigen::MatrixXd last_window_data;

   public:
    void submit(std::vector<std::vector<double>> &x);
    Detector(int window_size, double threshold);

    ~Detector();
};

// A helper to convert STL to Eigen matrix
template<typename Scalar, typename Container>
inline static Eigen::Matrix<Scalar, -1, -1> toEigenMatrix(const Container &vectors)
{
    Eigen::Matrix<Scalar, -1, -1> M(vectors.size(), vectors.front().size());
    for (size_t i = 0; i < vectors.size(); i++)
        for (size_t j = 0; j < vectors.front().size(); j++)
            M(i, j) = vectors[i][j];
    return M;
}

// 每个周期的分组：vector<vector<int>>
using Group = std::vector<std::vector<int>>;

class ClusterManager {
public:
    ClusterManager(const Eigen::MatrixXd& data,
                   int rec_windows_per_cycle,
                   int reconstruct_window,
                   double cluster_threshold);

    // 获取某个窗口所属周期的分组
    const Group& get_group_for_window(int win_l) const;

    // 周期长度
    int cycle_len() const { return cycle_; }

private:
    std::vector<Group> cycle_groups_;
    int cycle_;
};
