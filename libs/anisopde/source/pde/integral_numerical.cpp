#include <pde/integral_numerical.h>
#include <pde/project.h>

#include <igl/signed_distance.h>
#include <igl/barycentric_coordinates.h>
#include <igl/grad.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace zxh::pde {

int triangle_integral_jac(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &X,
    Eigen::VectorXd &jacs_l1,
    Eigen::VectorXd &jacs_l2) {

    const int nf = F.rows();
    jacs_l1 = Eigen::VectorXd::Zero(nf);
    jacs_l2 = Eigen::VectorXd::Zero(nf);

    Eigen::MatrixXd quadrature_vals = get_quadrature_points_value(V, F, X);
    for (int fi = 0; fi < nf; ++fi) {
        for (int qi = 0; qi < num_quadrature_points; ++qi) {
            const double val = quadrature_vals(fi, qi);
            jacs_l1(fi) += w[qi] * fabs(val);
            jacs_l2(fi) += w[qi] * val * val;
        }
    }

    return 0;
}

int triangle_integral_jac(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &G,
    Eigen::VectorXd &jacs_l1,
    Eigen::VectorXd &jacs_l2) {
    const int nf = F.rows();

    jacs_l1 = G.rowwise().lpNorm<1>();
    jacs_l2 = G.rowwise().squaredNorm();

    return 0;
}

int triangle_integral_diff_jac(
    const Eigen::MatrixXd &V_a,
    const Eigen::MatrixXi &F_a,
    const Eigen::VectorXd &X_a,
    const Eigen::MatrixXd &V_b,
    const Eigen::MatrixXi &F_b,
    const Eigen::VectorXd &X_b,
    Eigen::VectorXd &jacs_l1,
    Eigen::VectorXd &jacs_l2) {

    const int nf_a = F_a.rows();

    jacs_l1 = Eigen::VectorXd::Zero(nf_a);
    jacs_l2 = Eigen::VectorXd::Zero(nf_a);

    //* Compute the location of the quadrature points in ambient space
    Eigen::MatrixXd quadrature_points = get_quadrature_points_location(V_a, F_a);
    Eigen::MatrixXd quadrature_vals_a = get_quadrature_points_value(V_a, F_a, X_a);

    //* Project the quadrature points to the sparse mesh
    Eigen::VectorXi I;
    Eigen::MatrixXd C, BC;
    project_onto_mesh(
        quadrature_points, V_b, F_b, I, C, BC);

    Eigen::VectorXd X_proj;
    project_scalar_field_onto_mesh(
        X_b, F_b, I, BC, X_proj);

    for (int fi = 0; fi < nf_a; ++fi) {
        for (int qi = 0; qi < num_quadrature_points; ++qi) {
            const int qidx = fi * num_quadrature_points + qi;

            const double val_d = quadrature_vals_a(fi, qi);
            const double val_s = X_proj(qidx);
            const double val_error = val_s - val_d;
            jacs_l1(fi) += w[qi] * fabs(val_error);
            jacs_l2(fi) += w[qi] * val_error * val_error;
        }
    }

    return 0;
}
int triangle_integral_diff_jac(
    const Eigen::MatrixXd &V_a,
    const Eigen::MatrixXi &F_a,
    const Eigen::MatrixXd &G_a,
    const Eigen::MatrixXd &V_b,
    const Eigen::MatrixXi &F_b,
    const Eigen::MatrixXd &G_b,
    Eigen::VectorXd &jacs_l1,
    Eigen::VectorXd &jacs_l2) {
    const int nf_a = F_a.rows();
    jacs_l1 = Eigen::VectorXd::Zero(nf_a);
    jacs_l2 = Eigen::VectorXd::Zero(nf_a);

    Eigen::MatrixXd quadrature_points = get_quadrature_points_location(V_a, F_a);
    Eigen::VectorXi I;
    Eigen::MatrixXd C, BC;
    project_onto_mesh(
        quadrature_points, V_b, F_b, I, C, BC);

    for (int fi_a = 0; fi_a < nf_a; ++fi_a) {
        const Eigen::Vector3d g_a = G_a.row(fi_a); //* gradient is constant in triangle
        for (int qi = 0; qi < num_quadrature_points; ++qi) {
            const int fi_b = I(fi_a * num_quadrature_points + qi);
            const Eigen::Vector3d g_b = G_b.row(fi_b);
            const Eigen::Vector3d val_error = g_b - g_a;
            jacs_l1(fi_a) += w[qi] * val_error.lpNorm<1>();
            jacs_l2(fi_a) += w[qi] * val_error.squaredNorm();
        }
    }
    return 0;
}

} // namespace zxh::pde