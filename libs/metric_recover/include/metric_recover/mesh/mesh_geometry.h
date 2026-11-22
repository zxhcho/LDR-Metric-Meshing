#pragma once

#include <Eigen/Dense>

#include <metric_recover/mesh/mesh_topology.h>

namespace zxh::metric_recover {

class mesh_geometry_t {
public:
    mesh_geometry_t(
        Eigen::MatrixXd FN,
        Eigen::MatrixXd VN)
        : FN(FN), VN(VN) {}

public:
    // face normals
    Eigen::MatrixXd FN, VN;
};

mesh_geometry_t compute_mesh_geometry(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F);

} // namespace zxh::metric_recover