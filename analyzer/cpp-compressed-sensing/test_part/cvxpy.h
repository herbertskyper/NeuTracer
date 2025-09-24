#pragma once
#include <Eigen/Dense>
#include <vector>
Eigen::MatrixXd reconstruct_cvxpy(int n, int d, std::vector<int>& index, Eigen::MatrixXd& value);
Eigen::MatrixXd reconstruct(int original_rows, std::vector<int> &sample_rows, Eigen::MatrixXd &sampled_values);
Eigen::MatrixXd idct2_cvxpy(const Eigen::MatrixXd& x);