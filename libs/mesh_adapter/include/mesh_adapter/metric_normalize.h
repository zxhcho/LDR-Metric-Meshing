#pragma once

#include <Eigen/Dense>

namespace zxh::mesh_adapter {

int metric_normalize(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &metric,
    const int dof = 10000,
    const double hmin = 0.001,
    const double hmax = 0.1);

int metric_normalize_isotropic(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &metrics,
    const int dof,
    const double hmin,
    const double hmax);

} // namespace zxh::mesh_adapter