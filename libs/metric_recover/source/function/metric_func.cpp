#include <metric_recover/function/metric_func.h>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

namespace zxh::metric_recover {

void metric_func_spd_t::operator()(
    const Eigen::Vector3d &xval,
    Eigen::VectorXd &fval,
    Eigen::Matrix<double, -1, 3> &jacobian) const {

    using namespace autodiff;

    const int n = edges.rows();
    assert(n >= 3); // at least 3 edges, otherwise under-determined

    fval.resize(n);
    jacobian.resize(n, 3);
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d edge = edges.row(i);
        const double edge_len = edge_lens(i);
        fval(i) = edge_len_diff_spd<double>(edge, xval, edge_len);

        real u;
        Eigen::Vector<real, 2> edge_real = edge;
        Eigen::Vector<real, 3> xval_real = xval.cast<real>();
        real edge_len_real = edge_len;
        Eigen::VectorXd grad = gradient(
            edge_len_diff_spd<real>, wrt(xval_real),
            at(edge_real, xval_real, edge_len_real), u);
        jacobian.row(i) = grad.transpose();
    }
}

void metric_func_init_t::operator()(
    const Eigen::Vector3d &xval,
    Eigen::VectorXd &fval,
    Eigen::Matrix<double, -1, 3> &jacobian) const {

    using namespace autodiff;

    const int n = edges.rows();
    assert(n >= 3); // at least 3 edges, otherwise under-determined

    fval.resize(n);
    jacobian.resize(n, 3);
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d edge = edges.row(i);
        const double edge_len = edge_lens(i);
        fval(i) = edge_len_diff<double>(edge, xval, edge_len);

        real u;
        Eigen::Vector<real, 2> edge_real = edge;
        Eigen::Vector<real, 3> xval_real = xval.cast<real>();
        real edge_len_real = edge_len;
        Eigen::VectorXd grad = gradient(
            edge_len_diff<real>, wrt(xval_real),
            at(edge_real, xval_real, edge_len_real), u);
        jacobian.row(i) = grad.transpose();
    }
}

} // namespace zxh::metric_recover