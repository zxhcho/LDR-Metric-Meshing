#pragma once

#include <spdlog/spdlog.h>

#include <Eigen/Core>

namespace zxh::metric_utils {

enum MetricType {
    METRIC_P1 = 0, //* for P1 element, interpolation error is dependent on the second derivative(hessian)
    METRIC_P0 = 1, //* for P0 element, interpolation error is dependent on the first derivative(gradient)
};

//* set the eigenvalues lower bound to epsilon
Eigen::Matrix2d make_spd(
    const Eigen::Matrix2d &m,
    const double epsilon = 1e-1);

//* iterate over a batch of matrices
//* find out the average positive eigenvalue
//* set the eigenvalues lower bound to factor*avg
int make_spd_batch(
    Eigen::MatrixXd &m,
    const double factor = 1e-4);

// transform the metric from face tangent space(shape 2 * 2) to ambient space(shape 3 * 3)
Eigen::MatrixXd metric_from_tangent_to_ambient(
    const Eigen::MatrixXd &B1,
    const Eigen::MatrixXd &B2,
    const Eigen::MatrixXd &metric);

// average the metric from faces to vertex
Eigen::MatrixXd metric_face_to_vertex(
    const std::vector<std::vector<int>> &VF,
    const Eigen::MatrixXd &metric);

Eigen::MatrixXd principal_curvature_as_metric(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const int k);

//* A function that returns the hessian matrix at a point as metric field
template <typename HessianFunc>
Eigen::MatrixXd analytical_metric(
    const Eigen::MatrixXd &V,
    HessianFunc func) {

    const int nv = V.rows();
    Eigen::MatrixXd metric = Eigen::MatrixXd::Zero(nv, 3 * 3);
    for (int i = 0; i < nv; ++i) {
        Eigen::Matrix3d H = func(V(i, 0), V(i, 1), V(i, 2));
        metric.row(i) = Eigen::Map<Eigen::VectorXd>(H.data(), 3 * 3);
    }

    return metric;
}

//* Calculate the hessian of the scalar field as metric field
//* The hessian is recovered from the field by least square fitting of k-ring neighbors
Eigen::MatrixXd metric_from_hessian_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &field,
    const int k = 3,
    const MetricType m_type = METRIC_P1);

//* Calculate the hessian of the complex scalar field(representation of N-Symmetry Fields) as metric field
//* The hessian is recovered from the field by least square fitting of k-ring neighbors
Eigen::MatrixXd metric_from_hessian_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXcd &field,
    const int k = 3,
    const MetricType m_type = METRIC_P1);

Eigen::MatrixXd metric_from_hessian_3d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &field,
    const int k = 3);

//* project the metric to spd by flipping the negative eigenvalues
int metric_spd(Eigen::MatrixXd &metric);

//* return the scale factor as the maximal eigenvalue of all metric
double metric_scale_factor_max(const Eigen::MatrixXd &metric);
double metric_scale_factor_avg(const Eigen::MatrixXd &metric);

Eigen::MatrixXd metric_2d_to_3d(const Eigen::MatrixXd &metric);

Eigen::MatrixXd metric_3d_to_2d(const Eigen::MatrixXd &metric);

//* calculate the intersection of two metrics
Eigen::Matrix2d metric_intersection_2d(
    const Eigen::Matrix2d &a,
    const Eigen::Matrix2d &b);

Eigen::Matrix3d metric_intersection_3d(
    const Eigen::Matrix3d &a,
    const Eigen::Matrix3d &b);

// project the 3d metric to the tangent plane of the normal
Eigen::Matrix3d metric_projection(
    const Eigen::Matrix3d &m,
    const Eigen::Vector3d &n);

Eigen::MatrixXd metric_projection_batch(
    const Eigen::MatrixXd &m,
    const Eigen::MatrixXd &n);

// must follow the metric_projection
Eigen::Matrix2d metric_projection_to_2d(
    const Eigen::Matrix3d &m,
    const Eigen::Vector3d &b1,
    const Eigen::Vector3d &b2);

Eigen::MatrixXd metric_projection_to_2d_batch(
    const Eigen::MatrixXd &M,
    const Eigen::MatrixXd &B);

//* batch wise intersection of two metrics
Eigen::MatrixXd metric_intersection_batch(
    const Eigen::MatrixXd &a,
    const Eigen::MatrixXd &b);

//* Actually, what we calculated is the squared length under the metric instead of the edge length
//* But for efficiency, we just return the squared length here

Eigen::MatrixXd metric_induced_halfedge_length_3d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &E,
    const Eigen::MatrixXd &metric);

Eigen::VectorXd metric_induced_edge_length_3d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &E,
    const Eigen::MatrixXd &metric);

Eigen::MatrixXd metric_induced_halfedge_length_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXd &B,
    const Eigen::MatrixXi &E,
    const Eigen::MatrixXd &metric);

Eigen::VectorXd metric_induced_edge_length_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXd &B,
    const Eigen::MatrixXi &E,
    const Eigen::MatrixXd &metric);
} // namespace zxh::metric_utils