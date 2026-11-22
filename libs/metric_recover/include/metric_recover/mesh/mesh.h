#pragma once

#include <Eigen/Dense>

#include <metric_recover/mesh/mesh_topology.h>
#include <metric_recover/mesh/mesh_geometry.h>

namespace zxh::metric_recover {

// static mesh for optimization
class mesh_t {
public:
    mesh_t(const Eigen::MatrixXd &V,
           const Eigen::MatrixXi &F,
           mesh_topology_t topology,
           mesh_geometry_t geometry)
        : V(V), F(F),
          topology(topology),
          geometry(geometry) {
    }

    int nv() const { return V.rows(); }
    int nf() const { return F.rows(); }
    int ne() const { return topology.EV.rows(); }

public:
    const Eigen::MatrixXd &V; // vertices
    const Eigen::MatrixXi &F; // faces

    // mesh topology
    const mesh_topology_t topology;

    const mesh_geometry_t geometry;
};

mesh_t compute_mesh(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F);

} // namespace zxh::metric_recover