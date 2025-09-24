#include <Eigen/SparseCore>
#include <OsqpEigen/OsqpEigen.h>
#include <fftw3.h>
#include <cmath>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>
#include <iostream>
#include <fstream>

static Eigen::VectorXd idct1_cvxpy(const Eigen::VectorXd& x) {
    int N = x.size();
    if (N == 0) return x;

    // 预处理：调整输入以匹配 SciPy 的权重
    Eigen::VectorXd in_modified = x;
    in_modified(0) = x(0) / std::sqrt(2.0);  // 首元素缩放
    for (int i = 1; i < N; ++i) {
        in_modified(i) = x(i) / 2.0;         // 其余元素缩放
    }

    // 执行 FFTW
    Eigen::VectorXd out(N);
    fftw_plan p = fftw_plan_r2r_1d(
        N, in_modified.data(), out.data(), FFTW_REDFT01, FFTW_ESTIMATE
    );
    fftw_execute(p);
    fftw_destroy_plan(p);

    // 正交归一化缩放
    out *= std::sqrt(2.0 / N);
    return out;
}



static Eigen::MatrixXd idct(Eigen::MatrixXd const &x)
{
    double *in, *out;
    fftw_plan p;

    int length = x.size();
    Eigen::Index rows = x.rows();
    Eigen::Index cols = x.cols();
    Eigen::MatrixXd ret(rows, cols);

    in = (double *)fftw_malloc(sizeof(double) * rows);
    out = (double *)fftw_malloc(sizeof(double) * rows);
    p = fftw_plan_r2r_1d(rows, in, out, FFTW_REDFT01, FFTW_MEASURE | FFTW_DESTROY_INPUT);

    for (size_t i = 0; i < cols; i++)
    {
        std::memcpy(in, x.col(i).data(), rows * sizeof(double));
        fftw_execute(p);
        std::memcpy(ret.col(i).data(), out, rows * sizeof(double));
        ret.col(i) /= sqrt(ret.col(i).cwiseProduct(ret.col(i)).sum());
    }

    fftw_destroy_plan(p);

    fftw_free(in);
    fftw_free(out);
    return ret;
}

// 正交归一化的2D IDCT
Eigen::MatrixXd idct2_cvxpy(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd tmp = x;
    for (int i = 0; i < tmp.cols(); ++i)
        tmp.col(i) = idct1_cvxpy(tmp.col(i));
    for (int i = 0; i < tmp.rows(); ++i)
        tmp.row(i) = idct1_cvxpy(tmp.row(i).transpose()).transpose();
    return tmp;
}

Eigen::MatrixXd reconstruct(int original_rows, std::vector<int> &sample_rows, Eigen::MatrixXd &sampled_values)
{
    Eigen::Index rows = sampled_values.rows();
    Eigen::Index dims = sampled_values.cols();

    auto tmp1 = idct(Eigen::MatrixXd::Identity(dims, dims));
    auto tmp2 = idct(Eigen::MatrixXd::Identity(original_rows, original_rows));
    Eigen::MatrixXd tmp3 = Eigen::KroneckerProduct(tmp1, tmp2);
// 在构造 transform_mat 后
// std::ofstream fout("transform_mat_man_cpp.csv");
// for (int i = 0; i < tmp3.rows(); ++i) {
//     for (int j = 0; j < tmp3.cols(); ++j) {
//         fout << tmp3(i, j);
//         if (j + 1 < tmp3.cols()) fout << ",";
//     }
//     fout << "\n";
// }
// fout.close();

    // Flatten the sample_rows
    std::vector<int> flatten_sampled_rows;
    for (int i = 0; i < dims; i++)
    {
        for (auto row : sample_rows)
            flatten_sampled_rows.push_back(row + i * original_rows);
    }
    Eigen::MatrixXd transform_matrix = tmp3(flatten_sampled_rows, Eigen::all);

    // Solve Ax=b
    Eigen::VectorXd x(original_rows * dims);
    int m = rows * dims;
    int n = original_rows * dims;
    double Inf = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd constraint_matrix(m + n + n, n + m + n);
    Eigen::MatrixXd P(m + n + n, n + m + n);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(m + n + n, 1);
    Eigen::MatrixXd upper_bound(m + n + n, 1);
    Eigen::MatrixXd lower_bound(m + n + n, 1);
    Eigen::MatrixXd In = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(m, m);
    Eigen::MatrixXd On = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd Onm = Eigen::MatrixXd::Zero(n, m);
    constraint_matrix << transform_matrix, -Im, Onm.transpose(), In, Onm, -In, In, Onm, In;
    P << On, Onm, On, Onm.transpose(), Im, Onm.transpose(), On, Onm, On;
    lower_bound << sampled_values.reshaped(), Eigen::MatrixXd::Constant(n, 1, -Inf), Eigen::MatrixXd::Constant(n, 1, 0);
    upper_bound << sampled_values.reshaped(), Eigen::MatrixXd::Constant(n, 1, 0), Eigen::MatrixXd::Constant(n, 1, Inf);

    Eigen::SparseMatrix<double> sparse_constraint = constraint_matrix.sparseView();
    Eigen::SparseMatrix<double> sparse_P = P.sparseView();

    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(false);
    solver.data()->setNumberOfVariables(m + n + n);
    solver.data()->setNumberOfConstraints((m + n + n));
    solver.data()->setLinearConstraintsMatrix(sparse_constraint);
    solver.data()->setHessianMatrix(sparse_P);
    solver.data()->setBounds(lower_bound, upper_bound);
    solver.data()->setGradient(Q);

    if (!solver.initSolver())
    {
        std::cerr << "Failed to initialize solver" << std::endl;
        exit(-1);
    }
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cerr << "Failed to solve problem" << std::endl;
        return Eigen::MatrixXd();
    }
    x = solver.getSolution();

    // Eigen::MatrixXd ret = x.head(original_rows * dims).reshaped(original_rows, dims);
    Eigen::MatrixXd ret = Eigen::Map<const Eigen::MatrixXd>(x.head(original_rows * dims).data(), dims, original_rows).transpose();
    Eigen::MatrixXd recovered = idct2_cvxpy(ret);

    Eigen::MatrixXd recon_vals(sample_rows.size(), dims);
    for (int i = 0; i < dims; ++i) {
        for (int j = 0; j < sample_rows.size(); ++j) {
            recon_vals(j, i) = recovered(sample_rows[j], i);
        }
    }
    // 计算误差矩阵
    Eigen::MatrixXd err_mat = (recon_vals - sampled_values).cwiseAbs();
    // 找最大误差
    double max_error = err_mat.maxCoeff();
    bool has_large_error = (err_mat.array() > 1e-3).any();
    if (has_large_error) {
        std::cerr << "Warning: reconstruction error at sample indices is large. Max error = " << max_error << std::endl;
    }
    return recovered;
}