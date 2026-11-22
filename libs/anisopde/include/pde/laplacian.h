#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace zxh::pde {

int squared_edge_length(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &M,
    Eigen::MatrixXd &E);

int compute_cotangent_entries(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &M,
    Eigen::MatrixXd &C);

//* M: face-wise metric tensor for generalized Poisson equation
int compute_laplacian_matrix(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &M,
    Eigen::SparseMatrix<double> &L);

//* M: face-wise metric tensor for generalized Poisson equation
int compute_mass_matrix(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &M,
    Eigen::SparseMatrix<double> &Mass);
} // namespace zxh::pde