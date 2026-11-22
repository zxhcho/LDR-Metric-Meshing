#include <mesh_adapter/mmg_adapter.h>

#include <mmg/libmmg.h>

#include <spdlog/spdlog.h>

namespace zxh::mesh_adapter {

int adapt_mmg_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &metric,
    Eigen::MatrixXd &V_out,
    Eigen::MatrixXi &F_out,
    const double hmin,
    const double hmax,
    const double hausd,
    const double hgrad) {

    const int nv = V.rows();
    const int nf = F.rows();
    assert(metric.rows() == nv);

    MMG5_pMesh mesh = nullptr;
    MMG5_pSol sol = nullptr;

    MMGS_Init_mesh(MMG5_ARG_start,
                   MMG5_ARG_ppMesh, &mesh,
                   MMG5_ARG_ppMet, &sol,
                   MMG5_ARG_end);

    int rtn;
    const int flag_success = 1;
    rtn = MMG2D_Set_meshSize(mesh, nv, nf, 0, 0);
    if (rtn != flag_success) {
        spdlog::error("MMG Error: Set_meshSize: {}", rtn);
        return __LINE__;
    }

    for (int vi = 0; vi < nv; ++vi) {
        mesh->point[vi + 1].c[0] = V(vi, 0);
        mesh->point[vi + 1].c[1] = V(vi, 1);
        mesh->point[vi + 1].ref = 1;
    }

    for (int fi = 0; fi < nf; ++fi) {
        mesh->tria[fi + 1].v[0] = F(fi, 0) + 1;
        mesh->tria[fi + 1].v[1] = F(fi, 1) + 1;
        mesh->tria[fi + 1].v[2] = F(fi, 2) + 1;
        mesh->tria[fi + 1].ref = 1;
    }

    rtn = MMG2D_Set_solSize(mesh, sol, MMG5_Vertex, nv, 3);
    if (rtn != flag_success) {
        spdlog::error("MMG Error: Set_solSize");
        return __LINE__;
    }
    for (int vi = 0; vi < nv; vi++) {
        sol->m[3 * (vi + 1) + 0] = metric(vi, 0); // M4
        sol->m[3 * (vi + 1) + 1] = metric(vi, 1); // M42
        sol->m[3 * (vi + 1) + 2] = metric(vi, 4); // M42
    }

    MMG2D_Set_dparameter(mesh, sol, MMG2D_DPARAM_hmin, hmin);
    MMG2D_Set_dparameter(mesh, sol, MMG2D_DPARAM_hmax, hmax);
    MMG2D_Set_dparameter(mesh, sol, MMG2D_DPARAM_hgrad, hgrad);
    MMG2D_Set_dparameter(mesh, sol, MMG2D_DPARAM_hausd, hausd);

    rtn = MMG2D_mmg2dlib(mesh, sol);
    if (rtn != MMG5_SUCCESS) {
        spdlog::error("Error during mesh adaptation. Code: {}", rtn);
        MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &sol, MMG5_ARG_end);
        return __LINE__;
    }

    const int nv_out = mesh->np;
    const int nf_out = mesh->nt;
    V_out.resize(nv_out, 3);
    F_out.resize(nf_out, 3);
    for (int vi = 0; vi < nv_out; ++vi) {
        V_out(vi, 0) = mesh->point[vi + 1].c[0];
        V_out(vi, 1) = mesh->point[vi + 1].c[1];
        V_out(vi, 2) = 0.0;
    }
    for (int i = 0; i < nf_out; ++i) {
        F_out(i, 0) = mesh->tria[i + 1].v[0] - 1;
        F_out(i, 1) = mesh->tria[i + 1].v[1] - 1;
        F_out(i, 2) = mesh->tria[i + 1].v[2] - 1;
    }

    MMG2D_Free_all(MMG5_ARG_start,
                   MMG5_ARG_ppMesh, &mesh,
                   MMG5_ARG_ppMet, &sol,
                   MMG5_ARG_end);

    return 0;
}

int adapt_mmg_2d_iso(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &V_out,
    Eigen::MatrixXi &F_out,
    const double hausd,
    const double hgrad) {
    // TODO : implement isotropic remeshing
    return 0;
}

} // namespace zxh::mesh_adapter