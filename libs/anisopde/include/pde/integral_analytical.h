#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <igl/doublearea.h>
#include <igl/grad.h>

#include <pde/quadrature.h>

namespace zxh::pde {

template <typename FuncValT>
Eigen::VectorXd triangle_integral_jac(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    FuncValT Func) {
    const int nf = F.rows();
    Eigen::VectorXd jac = Eigen::VectorXd::Zero(nf);

    Eigen::MatrixXd quadrature_points = get_quadrature_points_location(V, F);
    for (int fi = 0; fi < nf; ++fi) {
        for (int qi = 0; qi < num_quadrature_points; ++qi) {
            const Eigen::Vector3d p = quadrature_points.row(fi * num_quadrature_points + qi);
            jac(fi) += w[qi] * Func(p.x(), p.y());
        }
    }

    return jac;
}

template <typename FuncValT>
Eigen::VectorXd triangle_integral_diff_jac(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &X,
    FuncValT Func) {

    const int nf = F.rows();
    Eigen::VectorXd jac = Eigen::VectorXd::Zero(nf);

    Eigen::MatrixXd quadrature_points = get_quadrature_points_location(V, F);
    for (int fi = 0; fi < nf; ++fi) {
        for (int qi = 0; qi < num_quadrature_points; ++qi) {
            const Eigen::Vector3d p = quadrature_points.row(fi * num_quadrature_points + qi);
            const double val_analytical = Func(p.x(), p.y());
            const double val_sparse =
                (1 - r[qi] - s[qi]) * X(F(fi, 0)) +
                r[qi] * X(F(fi, 1)) +
                s[qi] * X(F(fi, 2));
            jac(fi) += w[qi] * fabs(val_analytical - val_sparse);
        }
    }

    return jac;
}

template <typename FuncValT>
Eigen::VectorXd triangle_integral(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &A,
    FuncValT Func) {

    return triangle_integral_jac(V, F, Func).cwiseProduct(A);
}

template <typename FuncGradT>
Eigen::VectorXd triangle_grad_diff_jac(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &X,
    FuncGradT Func) {

    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);
    Eigen::MatrixXd grad = G * X;

    Eigen::VectorXd jac = Eigen::VectorXd::Zero(F.rows());
    for (int i = 0; i < F.rows(); ++i) {
        const Eigen::Vector2d v1 = V.row(F(i, 0)).head<2>();
        const Eigen::Vector2d v2 = V.row(F(i, 1)).head<2>();
        const Eigen::Vector2d v3 = V.row(F(i, 2)).head<2>();

        for (int nq = 0; nq < 7; ++nq) {
            const Eigen::Vector2d p = (1 - r[nq] - s[nq]) * v1 + r[nq] * v2 + s[nq] * v3;
            const Eigen::Vector2d grad_discrete = grad.row(i).head<2>();
            const Eigen::Vector2d grad_analytical = Func(p.x(), p.y());
            const double diff = (grad_discrete - grad_analytical).norm();
            jac(i) += w[nq] * diff;
        }
    }

    return jac;
}
}