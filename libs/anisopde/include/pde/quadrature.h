#pragma once

#include <Eigen/Dense>

namespace zxh::pde {

static constexpr int num_quadrature_points = 7;

static constexpr double r[num_quadrature_points] = {
    1.0 / 3.0, 0.0597158717, 0.4701420641, 0.4701420641,
    0.7974269853, 0.1012865073, 0.1012865073};

static constexpr double s[num_quadrature_points] = {
    1.0 / 3.0, 0.4701420641, 0.0597158717, 0.4701420641,
    0.1012865073, 0.7974269853, 0.1012865073};

static constexpr double w[num_quadrature_points] = {
    9.0 / 40.0, 0.13239415, 0.13239415, 0.13239415,
    0.12593918, 0.12593918, 0.12593918};

Eigen::MatrixXd get_quadrature_points_location(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F);

Eigen::MatrixXd get_quadrature_points_value(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &X);

} // namespace zxh::pde