#include <metric_recover/mesh/mesh.h>

namespace zxh::metric_recover {

mesh_t compute_mesh(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F) {
    const mesh_topology_t topology = compute_mesh_topology(V, F);
    const mesh_geometry_t geometry = compute_mesh_geometry(V, F);
    return mesh_t(V, F, topology, geometry);
}

} // namespace zxh::metric_recover