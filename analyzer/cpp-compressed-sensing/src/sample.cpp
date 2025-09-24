#include "detector.h"

#include <random>

std::tuple<Eigen::MatrixXd, std::vector<int>>
sample(Eigen::MatrixXd &input_data, int sample_amount, Eigen::MatrixXd &outlier_score)
{
    const int scale = 5;
    const double rho = 0.1;
    // const double rho = 1.0 / (std::sqrt(2 * M_PI) * sigma);
    const double sigma = 0.5;

    Eigen::Index cols = input_data.cols();
    Eigen::Index rows = input_data.rows();

    Eigen::MatrixXd sample_matrix = Eigen::MatrixXd::Zero(sample_amount, rows);

    // Sanity check
    if (outlier_score.cols() == rows && outlier_score.rows() == 1) {
    // 如果是 shape=(1, rows)，转置为列向量
    outlier_score.transposeInPlace();
    } else if (outlier_score.rows() != rows || outlier_score.cols() != 1) {
        // 如果不是 shape=(rows, 1)，只取第一行并转置
        Eigen::MatrixXd tmp = outlier_score.row(0).transpose();
        outlier_score = tmp;
    }
    assert(outlier_score.rows() == rows && outlier_score.cols() == 1);

    // Initialize the sample indices matrix
    std::vector<int> sample_indices(rows);
    for (size_t i = 0; i < rows; i++)
        sample_indices[i] = i;
    // Shuffle the sample indices, so that the first sample_amount item is the sample indice
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<double> rand01(0, 1);
    std::shuffle(sample_indices.begin(), sample_indices.end(), g);
    std::vector<int> su_row(sample_indices.begin(), sample_indices.begin() + sample_amount);

    // Build a linead increasing stacked score vector
    std::vector<double> stacked_score(rows + 1);
    stacked_score[0] = 0;
    for (size_t i = 1; i < rows + 1; i++)
        stacked_score[i] = stacked_score[i - 1] + outlier_score(i - 1);
    // for (size_t i = 0; i < rows; i++)  // Normalization
    //     stacked_score[i] /= stacked_score[rows - 1];
    for (size_t i = 0; i < rows + 1; i++)
        stacked_score[i] /= stacked_score[rows];

    std::vector<double> su_center;
    for (size_t i = 0; i < sample_amount; i++)
    {
        su_center.push_back(stacked_score[su_row[i]]);
        sample_matrix(i, su_row[i]) = 1;
    }

    double step, each_step;
    step = each_step = (double)1 / (scale * rows);
    int y = 1;
    while (step > stacked_score[y])
        y++;
    while (step <= 1)
    {
        for (size_t j = 0; j < sample_amount; j++)
        {
            double c = su_center[j];
            if (abs(c - step) > 3 * sigma)
                continue;
            double p = rho * exp(pow((c - step) / sigma, 2) / -2);
            if (rand01(g) < p)
                sample_matrix(j, y - 1) += 1;
        }
        step += each_step;
        while (step > stacked_score[y] && y < rows)
            y++;
    }
    for (size_t j = 0; j < sample_amount; j++)
    {
        sample_matrix.row(j) /= sample_matrix.row(j).sum();
    }
    Eigen::MatrixXd sampled = sample_matrix * input_data;

    Eigen::MatrixXd sorted_sampled = Eigen::MatrixXd::Zero(sample_amount, cols);
    std::vector<std::tuple<Eigen::MatrixXd, int>> sorting_vec;
    for (size_t i = 0; i < sample_amount; i++)
    {
        sorting_vec.push_back(std::make_tuple(sampled.row(i), su_row[i]));
    }
    std::sort(
        sorting_vec.begin(),
        sorting_vec.end(),
        [](const std::tuple<Eigen::MatrixXd, int> &lhs, const std::tuple<Eigen::MatrixXd, int> &rhs)
        { return std::get<1>(lhs) < std::get<1>(rhs); });
    for (size_t i = 0; i < sample_amount; i++)
    {
        sorted_sampled.row(i) = std::get<0>(sorting_vec[i]);
        su_row[i] = std::get<1>(sorting_vec[i]);
    }

    return { sorted_sampled, su_row };
}