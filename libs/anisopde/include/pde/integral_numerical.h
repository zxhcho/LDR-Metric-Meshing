#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <igl/doublearea.h>
#include <igl/grad.h>

#include <pde/quadrature.h>

namespace zxh::pde {

//* compute the jacobian of scalar field integral face-wise L1 and L2 norm
int triangle_integral_jac(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &X,
    Eigen::VectorXd &jacs_l1,
    Eigen::VectorXd &jacs_l2);

//* compute the jacobian of gradient field integral face-wise L1 and L2 norm
int triangle_integral_jac(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &G,
    Eigen::VectorXd &jacs_l1,
    Eigen::VectorXd &jacs_l2);

//* project results on mesh b onto mesh a (hence non-symmetric)
//* then compute the jacobian of scalar field difference face-wise L1 and L2 norm
int triangle_integral_diff_jac(
    const Eigen::MatrixXd &V_a,
    const Eigen::MatrixXi &F_a,
    const Eigen::VectorXd &X_a,
    const Eigen::MatrixXd &V_b,
    const Eigen::MatrixXi &F_b,
    const Eigen::VectorXd &X_b,
    Eigen::VectorXd &jacs_l1,
    Eigen::VectorXd &jacs_l2);

//* project results on mesh b onto mesh a (hence non-symmetric)
//* then compute the jacobian of gradient field difference face-wise L1 and L2 norm
int triangle_integral_diff_jac(
    const Eigen::MatrixXd &V_a,
    const Eigen::MatrixXi &F_a,
    const Eigen::MatrixXd &G_a,
    const Eigen::MatrixXd &V_b,
    const Eigen::MatrixXi &F_b,
    const Eigen::MatrixXd &G_b,
    Eigen::VectorXd &jacs_l1,
    Eigen::VectorXd &jacs_l2);

} // namespace zxh::pde