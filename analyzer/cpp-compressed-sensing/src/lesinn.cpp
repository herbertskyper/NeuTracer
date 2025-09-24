#include "detector.h"

#include <EigenRand/EigenRand>
// modified
// rows * 2 -> combine
double vec_similarity(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
{
    Eigen::VectorXd tmp = x - y;
    tmp = tmp.array() * tmp.array();
    double sum = tmp.sum();
    sum = sqrt(sum);
    return 1 / (1 + sum);
}

Eigen::MatrixXd lesinn(Eigen::MatrixXd &input_data, Eigen::MatrixXd &last_window_data, int t, int phi)
{
    Eigen::Index rows = input_data.rows();
    Eigen::Index cols = input_data.cols();

    // Combine the input data and the last window data
    Eigen::MatrixXd combined_data(rows + last_window_data.rows(), cols);
    combined_data << last_window_data, input_data;
    int n = combined_data.rows();

    Eigen::MatrixXd score = Eigen::MatrixXd::Zero(rows, 1);

    std::random_device rd;
    std::mt19937 g(42);

    for (Eigen::Index i = 0; i < rows; i++)
    {
        double current_score = 0;
        for (int j = 0; j < t; j++)
        {
            // 采样 phi 个不重复的行索引
            std::vector<int> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), g);
            std::vector<int> sample_indices(indices.begin(), indices.begin() + phi);

            double max_sim = 0;
            for (int k = 0; k < phi; k++)
            {
                int sample_idx = sample_indices[k];
                double sim = vec_similarity(input_data.row(i), combined_data.row(sample_idx));
                if (sim > max_sim)
                    max_sim = sim;
            }
            current_score += max_sim;
        }
        if (current_score != 0)
            score(i, 0) = t / current_score;
    }
    return score;
}

// Eigen::MatrixXd lesinn(Eigen::MatrixXd &input_data, Eigen::MatrixXd &last_window_data, int t, int phi)
// {
//     Eigen::Rand::P8_mt19937_64 urng{ 42 };  // hard-coded random seed for now
//     Eigen::Index rows = input_data.rows();
//     Eigen::Index cols = input_data.cols();

//     Eigen::MatrixXi sample_matrix = Eigen::Rand::uniformInt<Eigen::MatrixXi>(phi, t * rows, urng, 0, cols);

//     Eigen::MatrixXd similarity_matrix = Eigen::MatrixXd::Zero(rows, rows + last_window_data.rows());

//     Eigen::MatrixXd combined_data(rows + last_window_data.rows(), cols);  // Combine the input data and the last window data
//     combined_data << last_window_data, input_data;

//     // Calculate the similarity_matrix
//     for (Eigen::Index i = 0; i < rows; i++)
//     {
//         for (Eigen::Index j = 0; j < rows + last_window_data.rows(); j++)
//         {
//             similarity_matrix(i, j) = vec_similarity(input_data.row(i), combined_data.row(j));
//         }
//     }

//     Eigen::MatrixXd score = Eigen::MatrixXd::Zero(rows, 1);

//     // Calculate the outlier score
//     for (size_t i = 0; i < rows; i++)
//     {
//         double current_score = 0;
//         for (size_t j = 0; j < t; j++)
//         {
//             double max_sim = 0;
//             for (size_t k = 0; k < phi; k++)
//             {
//                 int sample_idx = sample_matrix(k, i * j);
//                 max_sim = std::max(max_sim, similarity_matrix(i, sample_idx));
//             }
//             current_score += max_sim;
//         }
//         if (current_score != 0)
//             score(i,0) = t / current_score;
//     }
//     return score;
// }