#pragma once

#include <Eigen/Dense>

#include <metric_recover/mesh/mesh.h>

namespace zxh::metric_recover {

Eigen::Vector3d solve_initial_guess(
    const Eigen::MatrixXd &edges,
    const Eigen::VectorXd &lengths);

int vertex_metric_solver_halfedge(
    const mesh_t &mesh,
    const Eigen::MatrixXd &edge_lens,
    Eigen::MatrixXd &metric);

int vertex_metric_solver_edge(
    const mesh_t &mesh,
    const Eigen::VectorXd &edge_lens,
    Eigen::MatrixXd &metric);

int vertex_metric_solver_dual_edge(
    const mesh_t &mesh,
    const Eigen::VectorXd &edge_lens,
    Eigen::MatrixXd &metric);

//* project the edge to tangent space of vertex as planar metric
int vertex_metric_solver_edge_tangent(
    const mesh_t &mesh,
    const Eigen::VectorXd &edge_lens,
    Eigen::MatrixXd &metric);

int vertex_metric_solver_dual_edge_tangent(
    const mesh_t &mesh,
    const Eigen::VectorXd &edge_lens,
    Eigen::MatrixXd &metric);
} // namespace zxh::metric_recover