#include <pde/project.h>

#include <igl/signed_distance.h>
#include <igl/barycentric_coordinates.h>

namespace zxh::pde {

int project_onto_mesh(
    const Eigen::MatrixXd &query,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::VectorXi &I,
    Eigen::MatrixXd &P,
    Eigen::MatrixXd &BC) {
    const int nq = query.rows();
    I.resize(nq);
    P.resize(nq, 3);
    BC.resize(nq, 3);

    Eigen::VectorXd S;
    Eigen::MatrixXd N;
    igl::signed_distance(
        query, V, F, igl::SIGNED_DISTANCE_TYPE_DEFAULT,
        S, I, P, N);

    for (int qi = 0; qi < nq; ++qi) {
        const int fi = I(qi);
        const Eigen::RowVector3d p = P.row(qi);
        const Eigen::RowVector3d v1 = V.row(F(fi, 0));
        const Eigen::RowVector3d v2 = V.row(F(fi, 1));
        const Eigen::RowVector3d v3 = V.row(F(fi, 2));
        Eigen::RowVector3d bc;
        igl::barycentric_coordinates(
            p, v1, v2, v3, bc);
        BC.row(qi) = bc;
    }

    return 0;
}

int project_field_onto_mesh(
    const Eigen::VectorXd &f,
    const Eigen::MatrixXd &g,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXi &I,
    const Eigen::MatrixXd &BC,
    Eigen::VectorXd &f_proj,
    Eigen::MatrixXd &g_proj) {

    project_scalar_field_onto_mesh(f, F, I, BC, f_proj);
    project_vector_field_onto_mesh(g, I, g_proj);

    return 0;
}

int project_scalar_field_onto_mesh(
    const Eigen::VectorXd &f,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXi &I,
    const Eigen::MatrixXd &BC,
    Eigen::VectorXd &f_proj) {
    const int nq = I.size();
    f_proj.resize(nq);
    for (int qi = 0; qi < nq; ++qi) {
        const int fi = I(qi);
        const Eigen::Vector3d bc = BC.row(qi);
        f_proj(qi) = f(F(fi, 0)) * bc(0) + f(F(fi, 1)) * bc(1) + f(F(fi, 2)) * bc(2);
    }

    return 0;
}

int project_vector_field_onto_mesh(
    const Eigen::MatrixXd &g,
    const Eigen::VectorXi &I,
    Eigen::MatrixXd &g_proj) {
    const int nq = I.size();
    g_proj.resize(nq, 3);
    for (int qi = 0; qi < nq; ++qi) {
        const int fi = I(qi);
        g_proj.row(qi) = g.row(fi);
    }

    return 0;
}

} // namespace zxh::pde