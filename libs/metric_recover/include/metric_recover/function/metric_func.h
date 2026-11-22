#pragma once

#include <lsqcpp/lsqcpp.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace zxh::metric_recover {

//* The variable is constrained in SPD matrix space.
class metric_func_spd_t {
public:
    constexpr static bool ComputesJacobian = true;

    metric_func_spd_t() = default;
    metric_func_spd_t(const Eigen::MatrixXd edges,
                      const Eigen::VectorXd edge_lens)
        : edges(edges), edge_lens(edge_lens) {}

    void operator()(const Eigen::Vector3d &xval,
                    Eigen::VectorXd &fval,
                    Eigen::Matrix<double, -1, 3> &jacobian) const;

public:
    Eigen::MatrixXd edges;
    Eigen::VectorXd edge_lens;
};

//* The variable is constrained in matrix space.
class metric_func_init_t {
public:
    constexpr static bool ComputesJacobian = true;

    metric_func_init_t() = default;
    metric_func_init_t(const Eigen::MatrixXd edges,
                       const Eigen::VectorXd edge_lens)
        : edges(edges), edge_lens(edge_lens) {}

    void operator()(const Eigen::Vector3d &xval,
                    Eigen::VectorXd &fval,
                    Eigen::Matrix<double, -1, 3> &jacobian) const;

public:
    Eigen::MatrixXd edges;
    Eigen::VectorXd edge_lens;
};

template <typename T>
Eigen::Matrix<T, 2, 2> metric_eig_space_to_mat(
    const Eigen::Vector<T, 3> &metric_eig) {
    Eigen::Matrix<T, 2, 2> eig_vec;
    Eigen::Vector<T, 2> eig_val;
    eig_val(0) = pow(metric_eig(0), 2);
    eig_val(1) = pow(metric_eig(1), 2);
    eig_vec(0, 0) = cos(metric_eig(2));
    eig_vec(0, 1) = -sin(metric_eig(2));
    eig_vec(1, 0) = sin(metric_eig(2));
    eig_vec(1, 1) = cos(metric_eig(2));
    Eigen::Matrix<T, 2, 2> metric_mat = eig_vec * eig_val.asDiagonal() * eig_vec.transpose();

    return metric_mat;
}

template <typename T>
Eigen::Vector<T, 3> metric_mat_to_eig_space(
    const Eigen::Matrix<T, 2, 2> &metric) {

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 2, 2>> eigensolver(metric);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "Failed to compute eigenvalues." << std::endl;
        return Eigen::Vector<T, 3>::Zero();
    }
    const auto eig_val = eigensolver.eigenvalues();
    const auto eig_vec = eigensolver.eigenvectors();

    Eigen::Vector<T, 3> metric_eig;
    metric_eig(0) = sqrt(fabs(eig_val(0)));
    metric_eig(1) = sqrt(fabs(eig_val(1)));
    metric_eig(2) = atan2(eig_vec(1, 0), eig_vec(0, 0));

    return metric_eig;
}

template <typename T>
T edge_len_diff(
    const Eigen::Vector<T, 2> &edge,
    const Eigen::Vector<T, 3> &metric_coeff,
    const T edge_len) {
    Eigen::Matrix<T, 2, 2> metric_mat;
    metric_mat << metric_coeff(0), metric_coeff(1),
        metric_coeff(1), metric_coeff(2);

    T metric_len = edge.dot(metric_mat * edge);
    return edge_len - metric_len;
}

template <typename T>
T edge_len_diff_spd(
    const Eigen::Vector<T, 2> &edge,
    const Eigen::Vector<T, 3> &metric_eig,
    const T edge_len) {
    Eigen::Matrix<T, 2, 2> metric_mat = metric_eig_space_to_mat(metric_eig);
    T metric_len = edge.dot(metric_mat * edge);
    return edge_len - metric_len;
}

} // namespace zxh::metric_recover
