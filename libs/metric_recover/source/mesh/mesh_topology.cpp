#include <metric_recover/mesh/mesh_topology.h>

#include <igl/edge_topology.h>
#include <igl/boundary_loop.h>

namespace zxh::metric_recover {

mesh_topology_t compute_mesh_topology(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F) {
    Eigen::MatrixXi EV, FE, EF;
    igl::edge_topology(V, F, EV, FE, EF);

    const int nv = V.rows();
    std::vector<std::vector<int>> VE(nv);
    for (int i = 0; i < EV.rows(); ++i) {
        VE[EV(i, 0)].push_back(i);
        VE[EV(i, 1)].push_back(i);
    }

    return mesh_topology_t(EV, EF, VE);
}

} // namespace zxh::metric_recover