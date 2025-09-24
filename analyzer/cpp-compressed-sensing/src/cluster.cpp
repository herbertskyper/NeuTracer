#include "detector.h"

#include <Eigen/Dense>
#include <cmath>
#include <fftw3.h>
#include <iostream>
#include <vector>

// fft
// length is the output length
Eigen::MatrixXcd fft(Eigen::MatrixXd const &x, int length)
{
    fftw_complex *out;
    double *in;
    fftw_plan p;

    // input should be no more than (length -1) *2
    in = (double *)fftw_malloc(sizeof(double) * (length - 1) * 2);
    std::memset(in, 0, sizeof(double) * (length - 1) * 2);
    // output is length
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * length);
    p = fftw_plan_dft_r2c_1d(length, in, out, FFTW_ESTIMATE);

    // Copy the data into fftw plan
    std::memcpy(in, x.data(), x.size() * sizeof(double));

    fftw_execute(p);

    fftw_destroy_plan(p);

    Eigen::RowVectorXcd ret(length);
    std::memcpy(ret.data(), out, length * sizeof(fftw_complex));
    fftw_free(in);
    fftw_free(out);
    return ret;
}

// ifft
// length is the output length
Eigen::MatrixXd ifft(Eigen::MatrixXcd const &x, int length)
{
    double *out;
    fftw_complex *in;
    fftw_plan p;

    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * length);
    out = (double *)fftw_malloc(sizeof(double) * length);
    p = fftw_plan_dft_c2r_1d(length, in, out, FFTW_ESTIMATE);

    // Copy the data into fftw plan
    std::memcpy(in, x.data(), x.cols() * sizeof(fftw_complex));

    fftw_execute(p);

    fftw_destroy_plan(p);

    Eigen::RowVectorXd ret(length);
    memcpy(ret.data(), out, length * sizeof(double));
    fftw_free(in);
    fftw_free(out);
    ret = ret / length;
    return ret;
}

// ncc_c
// Param: two 1D vectors
Eigen::MatrixXd ncc_c(Eigen::MatrixXd const &x, Eigen::MatrixXd const &y)
{
    double den = x.norm() * y.norm();
    den = den == 0 ? std::numeric_limits<double>::max() : den;  // Prevent a div zero
    int fft_size = 1L << int(std::ceil(log2(x.cols() * 2 - 1)));
    Eigen::MatrixXcd xx = fft(x, fft_size);
    Eigen::MatrixXcd yy = fft(y, fft_size).conjugate();
    Eigen::MatrixXcd tmp = xx.cwiseProduct(yy);
    Eigen::RowVectorXd cc = ifft(tmp, fft_size);
    Eigen::MatrixXd ret(1, x.cols() * 2 - 1);
    ret << cc.tail(x.cols() - 1), cc.head(x.cols());
    ret = ret / den;
    return ret;
}

double shape_based_distance(Eigen::MatrixXd const &x, Eigen::MatrixXd const &y)
{
    Eigen::MatrixXd ncc = ncc_c(x, y);
    double dist = 1 - ncc.maxCoeff();
    // assert(dist >= -0.1);
    return dist;
}

struct node
{
    bool is_leaf;
    int label;
    double distance;
    node *left;
    node *right;
};

std::vector<std::vector<int>> classify_nodes(node const *node, double threshold)
{
    std::vector<std::vector<int>> ret;
    if (node->is_leaf)
    {
        ret.push_back({ node->label });
        return ret;
    }
    else if (node->distance < threshold)
    {
        auto left = classify_nodes(node->left, threshold);
        auto right = classify_nodes(node->right, threshold);
        // Due to the tree construction, the sub-tree must have smaller distance
        assert(left.size() == 1);
        assert(right.size() == 1);
        left[0].insert(left[0].end(), right[0].begin(), right[0].end());
        ret.push_back(left[0]);
        return ret;
    }
    else
    {
        auto left = classify_nodes(node->left, threshold);
        auto right = classify_nodes(node->right, threshold);
        ret.insert(ret.end(), left.begin(), left.end());
        ret.insert(ret.end(), right.begin(), right.end());
        return ret;
    }
}

// cluster do a clustering by shape_based_distance
// exptected input dimension: [m, n] where m is the window size, n is the data dimension
std::vector<std::vector<int>> cluster(Eigen::MatrixXd &x, float threshold)
{
    int data_dim = x.cols();
    Eigen::MatrixXd distance(data_dim, data_dim);
    for (size_t i = 0; i < data_dim; i++)
    {
        for (size_t j = 0; j < data_dim; j++)
        {
            if (i == j)
                distance(i, j) = std::numeric_limits<double>::max();
            else
                distance(i, j) = shape_based_distance(x.col(i).transpose(), x.col(j).transpose());
        }
    }

    // Peform a clustering
    // Initialize the tree
    std::vector<node *> tree(data_dim);
    node *root;
    for (size_t i = 0; i < data_dim; i++)
    {
        node *leaf = new node();
        leaf->is_leaf = true;
        leaf->label = i;
        leaf->distance = 0;
        leaf->left = nullptr;
        leaf->right = nullptr;
        tree[i] = leaf;
    }
    root = tree[0];
    for (size_t i = 0; i < data_dim - 1; i++)
    {
        Eigen::Index minRow, minCol;
        double min = distance.minCoeff(&minRow, &minCol);
        node *new_node = new node();
        node *old_node = tree[minCol];
        new_node->is_leaf = false;
        new_node->label = 0;
        new_node->distance = min;
        new_node->left = tree[minRow];
        new_node->right = old_node;
        tree[minCol] = new_node;
        root = new_node;
        distance.row(minRow).setConstant(std::numeric_limits<double>::max());
        distance.col(minRow).setConstant(std::numeric_limits<double>::max());
    }

    // Do classification on the tree
    return classify_nodes(root, threshold);
}

#ifdef CLUSTER_DEBUG

int main(int argc, char const *argv[])
{
    // Test ncc_c
    {
        Eigen::RowVectorXd x(4);
        Eigen::RowVectorXd y(4);
        x << 1, 2, 3, 4;
        y << 1, 2, 3, 4;
        std::cout << ncc_c(x, y) << "\n";
    }
    {
        Eigen::RowVectorXd x(3);
        Eigen::RowVectorXd y(3);
        x << 1, 1, 1;
        y << 1, 1, 1;
        std::cout << ncc_c(x, y) << "\n";
    }
    {
        Eigen::RowVectorXd x(3);
        Eigen::RowVectorXd y(3);
        x << 1, 2, 3;
        y << -1, -1, -1;
        std::cout << ncc_c(x, y) << "\n";
    }

    // Test shape_based_distance
    {
        Eigen::RowVectorXd x(3);
        Eigen::RowVectorXd y(3);
        x << 1, 1, 1;
        y << 1, 1, 1;
        std::cout << shape_based_distance(x, y) << '\n';
    }
    {
        Eigen::RowVectorXd x(3);
        Eigen::RowVectorXd y(3);
        x << 0, 1, 2;
        y << 1, 2, 3;
        std::cout << shape_based_distance(x, y) << '\n';
    }
    {
        Eigen::RowVectorXd x(3);
        Eigen::RowVectorXd y(3);
        x << 1, 2, 3;
        y << 0, 1, 2;
        std::cout << shape_based_distance(x, y) << '\n';
    }
}

#endif