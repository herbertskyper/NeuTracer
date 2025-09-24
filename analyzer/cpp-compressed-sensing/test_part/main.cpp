#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <iostream>
#include "cvxpy.h" // 或你的reconstruct_cvxpy头文件
#include <string>

std::vector<int> load_index(const std::string& path) {
    std::vector<int> idx;
    std::ifstream in(path);
    int v;
    while (in >> v) idx.push_back(v);
    return idx;
}

Eigen::MatrixXd load_matrix(const std::string& path, int rows, int cols) {
    Eigen::MatrixXd mat(rows, cols);
    std::ifstream in(path);
    std::string line;
    int r = 0;
    while (std::getline(in, line) && r < rows) {
        std::stringstream ss(line);
        std::string val;
        int c = 0;
        while (std::getline(ss, val, ',') && c < cols) {
            mat(r, c++) = std::stod(val);
        }
        ++r;
    }
    return mat;
}

int main() {
    int n = 10, d = 1;
    auto index = load_index("test_index.txt");
    Eigen::MatrixXd value = load_matrix("test_value.txt", index.size(), d);

    // Eigen::MatrixXd x_re = reconstruct_cvxpy(n, d, index, value);
    Eigen::MatrixXd x_re = reconstruct(n, index, value);

    std::ofstream out("test_x_re_man_cpp.txt");
    for (int i = 0; i < x_re.rows(); ++i) {
        for (int j = 0; j < x_re.cols(); ++j) {
            out << x_re(i, j);
            if (j + 1 < x_re.cols()) out << ",";
        }
        out << "\n";
    }
    out.close();
    return 0;
}
// int main() {
//     // 读取输入
//     Eigen::MatrixXd x(8, 3);
//     std::ifstream fin("test_idct2_input.csv");
//     for (int i = 0; i < 8; ++i)
//         for (int j = 0; j < 3; ++j) {
//             char comma;
//             fin >> x(i, j);
//             if (j < 2) fin >> comma;
//         }

//     Eigen::MatrixXd y = idct2_cvxpy(x);

//     // 保存输出
//     std::ofstream fout("test_idct2_cpp.csv");
//     for (int i = 0; i < y.rows(); ++i) {
//         for (int j = 0; j < y.cols(); ++j) {
//             fout << y(i, j);
//             if (j + 1 < y.cols()) fout << ",";
//         }
//         fout << "\n";
//     }
//     fout.close();
//     return 0;
// }