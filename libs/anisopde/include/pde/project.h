#pragma once

#include <Eigen/Dense>

namespace zxh::pde {

//* query: the points to be projected
//* V: the vertices of the mesh
//* F: the faces of the mesh
//* I: the indices of the faces that the points are projected onto
//* P: the projected points
//* BC: the barycentric coordinates of the projected points
int project_onto_mesh(
    const Eigen::MatrixXd &query,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::VectorXi &I,
    Eigen::MatrixXd &P,
    Eigen::MatrixXd &BC);

int project_field_onto_mesh(
    const Eigen::VectorXd &f,
    const Eigen::MatrixXd &g,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXi &I,
    const Eigen::MatrixXd &BC,
    Eigen::VectorXd &f_proj,
    Eigen::MatrixXd &g_proj);

int project_scalar_field_onto_mesh(
    const Eigen::VectorXd &f,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXi &I,
    const Eigen::MatrixXd &BC,
    Eigen::VectorXd &f_proj);

int project_vector_field_onto_mesh(
    const Eigen::MatrixXd &g,
    const Eigen::VectorXi &I,
    Eigen::MatrixXd &g_proj);

} // namespace zxh::pde