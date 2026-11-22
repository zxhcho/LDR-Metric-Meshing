#include <pde/quadrature.h>

namespace zxh::pde {

Eigen::MatrixXd get_quadrature_points_location(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F) {
    const int nf = F.rows();

    Eigen::MatrixXd quadrature_points;
    quadrature_points.resize(nf * num_quadrature_points, 3);
    for (int fi = 0; fi < nf; ++fi) {
        const Eigen::Vector3d v1 = V.row(F(fi, 0));
        const Eigen::Vector3d v2 = V.row(F(fi, 1));
        const Eigen::Vector3d v3 = V.row(F(fi, 2));

        for (int qi = 0; qi < num_quadrature_points; ++qi) {
            quadrature_points.row(fi * num_quadrature_points + qi) =
                (1 - r[qi] - s[qi]) * v1 + r[qi] * v2 + s[qi] * v3;
        }
    }

    return quadrature_points;
}

Eigen::MatrixXd get_quadrature_points_value(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &X) {
    const int nf = F.rows();

    Eigen::MatrixXd quadrature_vals;
    quadrature_vals.resize(nf, num_quadrature_points);
    for (int fi = 0; fi < nf; ++fi) {
        const double v0 = X(F(fi, 0));
        const double v1 = X(F(fi, 1));
        const double v2 = X(F(fi, 2));

        for (int qi = 0; qi < num_quadrature_points; ++qi) {
            quadrature_vals(fi, qi) =
                (1 - r[qi] - s[qi]) * v0 + r[qi] * v1 + s[qi] * v2;
        }
    }

    return quadrature_vals;
}

} // namespace zxh::pde