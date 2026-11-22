#pragma once

#include <Eigen/Dense>

#include <vector>

namespace zxh::metric_recover {

class mesh_topology_t {
public:
    mesh_topology_t(Eigen::MatrixXi EV,
                    Eigen::MatrixXi EF,
                    std::vector<std::vector<int>> VE)
        : EV(EV), EF(EF), VE(VE) {
    }

public:
    // edge-vertex, face-edge, edge-face
    Eigen::MatrixXi EV, EF;

    std::vector<std::vector<int>> VE;
};

mesh_topology_t compute_mesh_topology(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F);

} // namespace zxh::metric_recover