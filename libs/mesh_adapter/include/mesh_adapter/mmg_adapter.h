#pragma once

#include <Eigen/Dense>

namespace zxh::mesh_adapter {

int adapt_mmg_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &metric,
    Eigen::MatrixXd &V_out,
    Eigen::MatrixXi &F_out,
    const double hmin = 0.001,
    const double hmax = 1000.0,
    const double hausd = 0.01,
    const double hgrad = 0.0);

int adapt_mmg_2d_iso(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &V_out,
    Eigen::MatrixXi &F_out,
    const double hausd = 0.01,
    const double hgrad = 0.0);

} // namespace zxh::mesh_adapter