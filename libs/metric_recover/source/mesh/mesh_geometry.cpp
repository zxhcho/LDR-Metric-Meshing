#include <metric_recover/mesh/mesh_geometry.h>

#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

namespace zxh::metric_recover {

mesh_geometry_t compute_mesh_geometry(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F) {
    Eigen::MatrixXd FN, VN;

    igl::per_face_normals(V, F, FN);
    igl::per_vertex_normals(V, F, FN, VN);

    return mesh_geometry_t(FN, VN);
}

} // namespace zxh::metric_recover